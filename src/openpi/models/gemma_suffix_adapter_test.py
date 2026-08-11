import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma


def _make_llm(rank: int = 4):
    language_config = dataclasses.replace(gemma.get_config("dummy"), depth=2, suffix_adapter_rank=rank)
    action_config = dataclasses.replace(gemma.get_config("dummy"), depth=2)
    llm = nnx_bridge.ToNNX(gemma.Module(configs=[language_config, action_config], embed_dtype="float32"))
    llm.lazy_init(rngs=nnx.Rngs(0), method="init", use_adarms=[False, False])
    return llm


def _forward(llm, *, active: bool):
    embedded = [jnp.ones((1, 3, 64), dtype=jnp.float32), None]
    positions = jnp.arange(3, dtype=jnp.int32)[None]
    mask = jnp.tril(jnp.ones((1, 3, 3), dtype=jnp.bool_))
    output, cache = llm(
        embedded,
        positions=positions,
        mask=mask,
        suffix_adapter_active=active,
    )
    return output[0], cache


def test_suffix_adapter_is_zero_initialized_and_gated():
    llm = _make_llm()
    disabled_before, cache_before = _forward(llm, active=False)
    enabled_initial, _ = _forward(llm, active=True)
    np.testing.assert_array_equal(disabled_before, enabled_initial)

    state = nnx.state(llm)
    for path, variable in state.flat_state().items():
        if path[-3:] == ("suffix_lora", "up", "kernel"):
            variable.value = jnp.full_like(variable.value, 0.01)
    nnx.update(llm, state)

    disabled_after, cache_after = _forward(llm, active=False)
    enabled_after, _ = _forward(llm, active=True)
    np.testing.assert_array_equal(disabled_before, disabled_after)
    np.testing.assert_array_equal(cache_before[0], cache_after[0])
    np.testing.assert_array_equal(cache_before[1], cache_after[1])
    assert not np.array_equal(disabled_after, enabled_after)


def test_suffix_adapter_parameter_count():
    llm = _make_llm(rank=4)
    adapter_params = {
        path: variable.value for path, variable in nnx.state(llm).flat_state().items() if "suffix_lora" in path
    }
    assert {path[-2] for path in adapter_params} == {"down", "up"}
    assert sum(value.size for value in adapter_params.values()) == 2 * 2 * 64 * 4
