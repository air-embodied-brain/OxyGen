from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
from pathlib import Path
import time
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from experiments.libero_language_adapter import data
from openpi.models import model as model_lib
from openpi.models import pi05_config
from openpi.models import tokenizer as tokenizer_lib
from openpi.shared import nnx_utils
from openpi.training import weight_loaders

ADAPTER_FILTER = nnx.All(
    nnx.Param,
    nnx_utils.PathRegex(".*lora.*"),
    nnx.Not(nnx_utils.PathRegex(".*bias")),
)


@dataclasses.dataclass(frozen=True)
class Args:
    split: Path
    checkpoint: Path
    norm_stats: Path
    output_dir: Path
    adapter_type: Literal["final_mlp", "suffix_lora", "full_lora"] = "suffix_lora"
    loss_mode: Literal["full", "incremental"] = "full"
    init_adapter: Path | None = None
    rank: int = 16
    alpha: float = 16.0
    learning_rate: float = 1e-4
    batch_size: int = 2
    steps: int = 2_000
    warmup_steps: int = 100
    suffix_len: int = 20
    seed: int = 7
    eval_every: int = 100
    validation_batches: int = 20
    log_every: int = 10


def load_split(path: Path):
    payload = json.loads(path.read_text())
    train = [data.SampleRef(**item) for item in payload["train"]]
    validation = [data.SampleRef(**item) for item in payload["validation"]]
    return train, validation, payload["manifest"]


def load_model(args: Args):
    config = pi05_config.Pi05Config(
        pi05=True,
        action_horizon=10,
        max_token_len=data.PROMPT_TOKEN_LEN,
        discrete_state_input=data.DISCRETE_STATE_INPUT,
        language_adapter=args.adapter_type,
        language_suffix_adapter_rank=args.rank,
        language_lora_alpha=args.alpha,
    )
    if args.rank == 0:
        params = model_lib.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
        return config, config.load(params)

    model = config.create(jax.random.key(args.seed))
    graphdef, state = nnx.split(model)
    state = nnx_utils.state_map(
        state,
        nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora_b.*")),
        lambda parameter: parameter.replace(jnp.zeros_like(parameter.value)),
    )
    frozen_filter = nnx.All(nnx.Param, nnx.Not(ADAPTER_FILTER))
    state = nnx_utils.state_map(
        state,
        frozen_filter,
        lambda parameter: (
            parameter if parameter.value is None else parameter.replace(parameter.value.astype(jnp.bfloat16))
        ),
    )
    loaded = weight_loaders.CheckpointWeightLoader(str(args.checkpoint / "params")).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    # CheckpointWeightLoader returns host arrays. Keep the frozen model resident on
    # the accelerator; otherwise evaluation would copy all parameters on every JIT call.
    state = jax.device_put(state)
    model = nnx.merge(graphdef, state)
    return config, model


def adapter_arrays(state: nnx.State) -> dict[str, np.ndarray]:
    return {
        "/".join(map(str, path)): np.asarray(variable.value)
        for path, variable in state.filter(ADAPTER_FILTER).flat_state().items()
    }


def apply_adapter(state: nnx.State, adapter_path: Path) -> None:
    arrays = np.load(adapter_path)
    found = set()
    for path, variable in state.filter(ADAPTER_FILTER).flat_state().items():
        name = "/".join(map(str, path))
        if name not in arrays:
            raise KeyError(f"Missing adapter parameter {name} in {adapter_path}")
        value = arrays[name]
        if value.shape != variable.value.shape:
            raise ValueError(f"Shape mismatch for {name}: {value.shape} vs {variable.value.shape}")
        variable.value = jnp.asarray(value, dtype=variable.value.dtype)
        found.add(name)
    if found != set(arrays.files):
        raise ValueError(f"Unexpected adapter arrays: {sorted(set(arrays.files) - found)}")


def _save_adapter(path: Path, state: nnx.State) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **adapter_arrays(state))


def _make_train_step(model_def, tx, *, loss_mode: str, seed_len: int):
    @jax.jit
    def train_step(state, opt_state, batch, rng):
        model = nnx.merge(model_def, state)
        observation, suffix_inputs, suffix_targets, suffix_mask, suffix_loss_mask = batch

        def loss_fn(train_model):
            if loss_mode == "incremental":
                losses = train_model.compute_language_suffix_incremental_loss(
                    rng,
                    observation,
                    suffix_inputs,
                    suffix_targets,
                    suffix_mask,
                    suffix_loss_mask,
                    seed_len=seed_len,
                    train=True,
                )
            else:
                losses = train_model.compute_language_suffix_loss(
                    rng,
                    observation,
                    suffix_inputs,
                    suffix_targets,
                    suffix_mask,
                    suffix_loss_mask,
                    train=True,
                )
            return jnp.mean(losses)

        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, ADAPTER_FILTER))(model)
        params = nnx.state(model, ADAPTER_FILTER)
        updates, opt_state = tx.update(grads, opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        return nnx.state(model), opt_state, loss, optax.global_norm(grads)

    return train_step


def make_eval_step(model_def, *, loss_mode: str = "full", seed_len: int = 0):
    @jax.jit
    def eval_step(state, batch):
        model = nnx.merge(model_def, state)
        observation, suffix_inputs, suffix_targets, suffix_mask, suffix_loss_mask = batch
        if loss_mode == "incremental":
            root_cache, prefix_mask = model.prefill_language_root(
                None,
                observation,
                train=False,
                max_suffix_tokens=suffix_inputs.shape[1],
            )
            logits = model.forward_language_suffix_incremental(
                root_cache,
                prefix_mask,
                suffix_inputs,
                suffix_mask,
                seed_len=seed_len,
                adapter_active=True,
            )
        else:
            root_cache, prefix_mask = model.prefill_language_root(None, observation, train=False)
            logits = model.forward_language_suffix(
                root_cache,
                prefix_mask,
                suffix_inputs,
                suffix_mask,
                adapter_active=True,
            )
        token_loss = -jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            suffix_targets[..., None],
            axis=-1,
        )[..., 0]
        losses = jnp.sum(token_loss * suffix_loss_mask, axis=-1) / jnp.maximum(jnp.sum(suffix_loss_mask, axis=-1), 1)
        predictions = jnp.argmax(logits, axis=-1)
        correct = (predictions == suffix_targets) & suffix_loss_mask
        return jnp.mean(losses), jnp.sum(correct), jnp.sum(suffix_loss_mask)

    return eval_step


def _evaluate(state, eval_step, dataset, *, batch_size: int, batches: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    losses = []
    correct = 0
    tokens = 0
    for _ in range(batches):
        indices = rng.integers(0, len(dataset), size=batch_size)
        batch = data.collate([dataset[int(index)] for index in indices])
        loss, batch_correct, batch_tokens = eval_step(state, batch)
        loss, batch_correct, batch_tokens = jax.device_get((loss, batch_correct, batch_tokens))
        losses.append(float(loss))
        correct += int(batch_correct)
        tokens += int(batch_tokens)
    return {
        "validation_loss": float(np.mean(losses)),
        "validation_perplexity": float(np.exp(min(np.mean(losses), 20.0))),
        "validation_token_accuracy": correct / max(tokens, 1),
    }


def _write_jsonl(path: Path, payload: Mapping) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def main(args: Args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_refs, validation_refs, split_manifest = load_split(args.split)
    tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=data.PROMPT_TOKEN_LEN)
    dataset_kwargs = {
        "norm_stats_path": args.norm_stats,
        "prompt_tokenizer": tokenizer,
        "action_dim": 32,
        "suffix_len": args.suffix_len,
    }
    train_dataset = data.LiberoLanguageDataset(train_refs, **dataset_kwargs)
    validation_dataset = data.LiberoLanguageDataset(validation_refs, **dataset_kwargs)
    train_batches = data.batches(train_dataset, batch_size=args.batch_size, seed=args.seed)

    started = time.time()
    config, model = load_model(args)
    model_loaded = time.time()
    model_def, state = nnx.split(model)
    if args.init_adapter is not None:
        apply_adapter(state, args.init_adapter)
    adapter_params = state.filter(ADAPTER_FILTER)
    adapter_count = sum(variable.value.size for variable in adapter_params.flat_state().values())
    # Standard LoRA covers Q/K/V/O and the gated/up/down FFN projections in every layer.
    expected_count = 2 * 2_048 * args.rank if args.adapter_type == "final_mlp" else 18 * args.rank * 96_768
    if adapter_count != expected_count:
        raise ValueError(f"Expected {expected_count:,} parameters for {args.adapter_type}, found {adapter_count:,}")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.learning_rate / max(args.warmup_steps, 1),
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=max(args.steps, args.warmup_steps + 1),
        end_value=args.learning_rate * 0.1,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0.0),
    )
    opt_state = tx.init(adapter_params)
    seed_len = len(tokenizer.tokenize_language_seed("Subtask: "))
    train_step = _make_train_step(model_def, tx, loss_mode=args.loss_mode, seed_len=seed_len)
    eval_step = make_eval_step(model_def, loss_mode=args.loss_mode, seed_len=seed_len)

    metadata = {
        "args": dataclasses.asdict(args),
        "model": dataclasses.asdict(config),
        "split_manifest": split_manifest,
        "adapter_parameters": adapter_count,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_git_commit": _checkpoint_commit(args.checkpoint),
        "input_protocol": {
            "discrete_state_input": data.DISCRETE_STATE_INPUT,
            "raw_hdf5_image_transform": "rotate_180",
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n")
    metrics_path = args.output_dir / "metrics.jsonl"

    compile_finished = None
    for step in range(1, args.steps + 1):
        batch = next(train_batches)
        step_started = time.perf_counter()
        state, opt_state, loss, grad_norm = train_step(
            state,
            opt_state,
            batch,
            jax.random.fold_in(jax.random.key(args.seed), step),
        )
        loss, grad_norm = jax.device_get((loss, grad_norm))
        elapsed = time.perf_counter() - step_started
        if compile_finished is None:
            compile_finished = time.time()
        if step == 1 or step % args.log_every == 0:
            record = {
                "step": step,
                "train_loss": float(loss),
                "grad_norm": float(grad_norm),
                "step_seconds": elapsed,
                "wall_seconds": time.time() - started,
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            _write_jsonl(metrics_path, record)
        if step % args.eval_every == 0 or step == args.steps:
            evaluation = _evaluate(
                state,
                eval_step,
                validation_dataset,
                batch_size=args.batch_size,
                batches=args.validation_batches,
                seed=args.seed + step,
            )
            record = {"step": step, **evaluation, "wall_seconds": time.time() - started}
            print(json.dumps(record, sort_keys=True), flush=True)
            _write_jsonl(metrics_path, record)
            _save_adapter(args.output_dir / f"adapter_step_{step}.npz", state)

    summary = {
        **metadata,
        "load_seconds": model_loaded - started,
        "first_step_compile_seconds": compile_finished - model_loaded,
        "total_seconds": time.time() - started,
        "final_step": args.steps,
        "final_adapter": str(args.output_dir / f"adapter_step_{args.steps}.npz"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    train_dataset.close()
    validation_dataset.close()


def _checkpoint_commit(checkpoint: Path) -> str | None:
    head = checkpoint / ".git" / "HEAD"
    if not head.exists():
        return None
    value = head.read_text().strip()
    if value.startswith("ref: "):
        ref = checkpoint / ".git" / value.removeprefix("ref: ")
        if ref.exists():
            return ref.read_text().strip()
        packed = checkpoint / ".git" / "packed-refs"
        if packed.exists():
            ref_name = value.removeprefix("ref: ")
            for line in packed.read_text().splitlines():
                if line and not line.startswith("#") and line.endswith(f" {ref_name}"):
                    return line.split()[0]
    return value


if __name__ == "__main__":
    main(tyro.cli(Args))
