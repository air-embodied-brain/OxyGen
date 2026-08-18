from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from language_adaptation.libero import data
from openpi.models import pi05_config
from openpi.shared import nnx_utils
from openpi.training import weight_loaders

PARAM_FILTER = nnx.All(
    nnx.Param,
    nnx_utils.PathRegex(".*lora.*"),
    nnx.Not(nnx_utils.PathRegex(".*bias")),
)


def load_model(checkpoint: Path, *, rank: int, alpha: float, seed: int):
    """Load frozen pi0.5 weights and initialize a suffix-LoRA model."""
    config = pi05_config.Pi05Config(
        pi05=True,
        action_horizon=10,
        max_token_len=data.PROMPT_TOKEN_LEN,
        discrete_state_input=data.DISCRETE_STATE_INPUT,
        language_adapter="suffix_lora",
        language_suffix_adapter_rank=rank,
        language_lora_alpha=alpha,
    )
    model = config.create(jax.random.key(seed))
    graphdef, state = nnx.split(model)
    state = nnx_utils.state_map(
        state,
        nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora_b.*")),
        lambda parameter: parameter.replace(jnp.zeros_like(parameter.value)),
    )
    state = nnx_utils.state_map(
        state,
        nnx.All(nnx.Param, nnx.Not(PARAM_FILTER)),
        lambda parameter: (
            parameter if parameter.value is None else parameter.replace(parameter.value.astype(jnp.bfloat16))
        ),
    )
    loaded = weight_loaders.CheckpointWeightLoader(str(checkpoint / "params")).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    return config, nnx.merge(graphdef, jax.device_put(state))


def arrays(state: nnx.State) -> dict[str, np.ndarray]:
    return {
        "/".join(map(str, path)): np.asarray(variable.value)
        for path, variable in state.filter(PARAM_FILTER).flat_state().items()
    }


def apply(state: nnx.State, path: Path) -> None:
    values = np.load(path)
    found = set()
    for parameter_path, variable in state.filter(PARAM_FILTER).flat_state().items():
        name = "/".join(map(str, parameter_path))
        if name not in values:
            raise KeyError(f"Missing adapter parameter {name} in {path}")
        value = values[name]
        if value.shape != variable.value.shape:
            raise ValueError(f"Shape mismatch for {name}: {value.shape} vs {variable.value.shape}")
        variable.value = jnp.asarray(value, dtype=variable.value.dtype)
        found.add(name)
    if found != set(values.files):
        raise ValueError(f"Unexpected adapter arrays: {sorted(set(values.files) - found)}")


def save(path: Path, state: nnx.State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays(state))
