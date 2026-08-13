from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re
import time
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from experiments.libero_language_adapter import data
from experiments.libero_language_adapter import train
from openpi.models import tokenizer as tokenizer_lib


@dataclasses.dataclass(frozen=True)
class Args:
    split: Path
    checkpoint: Path
    norm_stats: Path
    output_dir: Path
    adapter: Path | None = None
    adapter_type: Literal["final_mlp", "suffix_lora", "full_lora"] = "suffix_lora"
    rank: int = 16
    alpha: float = 16.0
    suffix_len: int = 20
    suffix_seed: str = "Subtask: "
    samples_per_task: int = 2
    max_generated_samples: int = 0
    seed: int = 7
    timing_warmups: int = 5
    timing_repeats: int = 50
    teacher_forced_samples: int = 0
    teacher_forced_loss_mode: Literal["full", "incremental"] = "full"
    action_invariance_samples: int = 20


def apply_adapter(state: nnx.State, adapter_path: Path) -> None:
    arrays = np.load(adapter_path)
    found = set()
    for path, variable in state.filter(train.ADAPTER_FILTER).flat_state().items():
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


def _set_adapter(state: nnx.State, arrays: dict[str, np.ndarray]) -> None:
    for path, variable in state.filter(train.ADAPTER_FILTER).flat_state().items():
        variable.value = jnp.asarray(arrays["/".join(map(str, path))], dtype=variable.value.dtype)


def _select_samples(samples: list[data.SampleRef], *, per_task: int, seed: int) -> list[data.SampleRef]:
    grouped: dict[tuple[str, str], list[data.SampleRef]] = {}
    for sample in samples:
        grouped.setdefault((sample.suite, sample.task), []).append(sample)
    rng = np.random.default_rng(seed)
    selected = []
    for task_samples in grouped.values():
        unique_targets: dict[str, list[data.SampleRef]] = {}
        for sample in task_samples:
            unique_targets.setdefault(sample.target, []).append(sample)
        candidates = [items[int(rng.integers(0, len(items)))] for items in unique_targets.values()]
        rng.shuffle(candidates)
        if len(candidates) < per_task:
            remaining = [item for item in task_samples if item not in candidates]
            rng.shuffle(remaining)
            candidates.extend(remaining[: per_task - len(candidates)])
        selected.extend(candidates[:per_task])
    return selected


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def word_f1(prediction: str, target: str) -> float:
    prediction_words = normalize_text(prediction).split()
    target_words = normalize_text(target).split()
    if not prediction_words or not target_words:
        return float(prediction_words == target_words)
    prediction_counts = {word: prediction_words.count(word) for word in set(prediction_words)}
    target_counts = {word: target_words.count(word) for word in set(target_words)}
    overlap = sum(min(count, target_counts.get(word, 0)) for word, count in prediction_counts.items())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_words)
    recall = overlap / len(target_words)
    return 2 * precision * recall / (precision + recall)


def _make_functions(model_def, *, suffix_len: int, seed_len: int):
    @jax.jit
    def prefill(state, observation):
        model = nnx.merge(model_def, state)
        return model.prefill_language_root(None, observation, train=False)

    @jax.jit
    def prefill_incremental(state, observation):
        model = nnx.merge(model_def, state)
        return model.prefill_language_root(
            None,
            observation,
            train=False,
            max_suffix_tokens=suffix_len,
        )

    @jax.jit
    def suffix_active(state, root_cache, prefix_mask, inputs, mask):
        model = nnx.merge(model_def, state)
        return model.forward_language_suffix(
            root_cache,
            prefix_mask,
            inputs,
            mask,
            adapter_active=True,
        )

    @jax.jit
    def suffix_disabled(state, root_cache, prefix_mask, inputs, mask):
        model = nnx.merge(model_def, state)
        return model.forward_language_suffix(
            root_cache,
            prefix_mask,
            inputs,
            mask,
            adapter_active=False,
        )

    @jax.jit
    def suffix_step_active(state, kv_cache, prefix_mask, token, suffix_index):
        model = nnx.merge(model_def, state)
        return model.forward_language_suffix_token(
            kv_cache,
            prefix_mask,
            token,
            suffix_index,
            adapter_active=True,
        )

    @jax.jit
    def suffix_step_disabled(state, kv_cache, prefix_mask, token, suffix_index):
        model = nnx.merge(model_def, state)
        return model.forward_language_suffix_token(
            kv_cache,
            prefix_mask,
            token,
            suffix_index,
            adapter_active=False,
        )

    @jax.jit
    def suffix_block_active(state, kv_cache, prefix_mask, tokens, mask):
        model = nnx.merge(model_def, state)
        return model.append_language_suffix_block(
            kv_cache,
            prefix_mask,
            tokens,
            mask,
            adapter_active=True,
        )

    @jax.jit
    def suffix_block_disabled(state, kv_cache, prefix_mask, tokens, mask):
        model = nnx.merge(model_def, state)
        return model.append_language_suffix_block(
            kv_cache,
            prefix_mask,
            tokens,
            mask,
            adapter_active=False,
        )

    def make_request(*, adapter_active: bool):
        @jax.jit
        def request(state, kv_cache, prefix_mask, tokens):
            model = nnx.merge(model_def, state)
            seed_tokens = jnp.pad(
                tokens[:, :seed_len],
                ((0, 0), (0, suffix_len - seed_len)),
            )
            seed_mask = jnp.broadcast_to(
                jnp.arange(suffix_len)[None, :] < seed_len,
                seed_tokens.shape,
            )
            _, kv_cache = model.append_language_suffix_block(
                kv_cache,
                prefix_mask,
                seed_tokens,
                seed_mask,
                adapter_active=adapter_active,
            )

            def step(cache, token_and_index):
                token, suffix_index = token_and_index
                logits, cache = model.forward_language_suffix_token(
                    cache,
                    prefix_mask,
                    token[:, None],
                    suffix_index,
                    adapter_active=adapter_active,
                )
                return cache, logits

            return jax.lax.scan(
                step,
                kv_cache,
                (
                    tokens[:, seed_len:].T,
                    jnp.arange(seed_len, tokens.shape[1], dtype=jnp.int32),
                ),
            )

        return request

    @jax.jit
    def actions(state, observation, noise):
        model = nnx.merge(model_def, state)
        return model.sample_actions(jax.random.key(0), observation, num_steps=10, noise=noise)

    return (
        prefill,
        prefill_incremental,
        suffix_active,
        suffix_disabled,
        suffix_step_active,
        suffix_step_disabled,
        suffix_block_active,
        suffix_block_disabled,
        make_request(adapter_active=True),
        make_request(adapter_active=False),
        actions,
    )


def _generate(
    state,
    observation,
    tokenizer,
    prefill_incremental,
    suffix_step_active,
    suffix_block_active,
    *,
    suffix_len: int,
    suffix_seed: str,
) -> str:
    kv_cache, prefix_mask = prefill_incremental(state, observation)
    seed_tokens = tokenizer.tokenize_language_seed(suffix_seed)
    seed_inputs = np.zeros((1, suffix_len), dtype=np.int32)
    seed_inputs[0, : len(seed_tokens)] = seed_tokens
    seed_mask = np.zeros(seed_inputs.shape, dtype=np.bool_)
    seed_mask[:, : len(seed_tokens)] = True
    seed_logits, kv_cache = suffix_block_active(state, kv_cache, prefix_mask, seed_inputs, seed_mask)
    logits = seed_logits[:, len(seed_tokens) - 1 : len(seed_tokens)]
    suffix_index = len(seed_tokens)

    generated = []
    max_new_tokens = suffix_len - len(seed_tokens) + 1
    for generation_index in range(max_new_tokens):
        assert logits is not None
        token = int(jax.device_get(jnp.argmax(logits[0, 0])))
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        if generation_index + 1 == max_new_tokens:
            break
        logits, kv_cache = suffix_step_active(
            state,
            kv_cache,
            prefix_mask,
            np.asarray([[token]], dtype=np.int32),
            np.asarray(suffix_index, dtype=np.int32),
        )
        suffix_index += 1
    return tokenizer.detokenize(np.asarray(generated, dtype=np.int32)).strip()


def _timed(callable_, *, warmups: int, repeats: int) -> dict[str, float]:
    for _ in range(warmups):
        jax.block_until_ready(callable_())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(callable_())
        samples.append((time.perf_counter() - started) * 1_000)
    return {
        "median_ms": float(np.median(samples)),
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


def _adapter_only_timing(
    adapter: dict[str, np.ndarray],
    *,
    tokens: int,
    warmups: int,
    repeats: int,
) -> dict[str, float]:
    """Measure the incremental cost added by all adapter layers."""
    down = next(value for name, value in adapter.items() if name.endswith("suffix_lora_down/kernel"))
    up = next(value for name, value in adapter.items() if name.endswith("suffix_lora_up/kernel"))
    down = jax.device_put(down.astype(np.float32))
    up = jax.device_put(up.astype(np.float32))
    hidden = jax.device_put(np.ones((1, tokens, down.shape[0]), dtype=np.float32))

    @jax.jit
    def run(inputs):
        return inputs + jax.nn.gelu(inputs @ down) @ up

    return _timed(lambda: run(hidden), warmups=warmups, repeats=repeats)


def _teacher_forced_evaluation(
    state,
    model_def,
    dataset,
    *,
    samples: int,
    loss_mode: str,
    seed_len: int,
    batch_size: int = 2,
) -> dict[str, float | int | str]:
    if samples <= 0:
        raise ValueError("teacher-forced sample count must be positive")
    sample_count = min(samples, len(dataset))
    eval_step = train.make_eval_step(model_def, loss_mode=loss_mode, seed_len=seed_len)
    loss_sum = 0.0
    correct = 0
    token_count = 0
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        batch = data.collate([dataset[index] for index in range(start, stop)])
        loss, batch_correct, batch_tokens = jax.device_get(eval_step(state, batch))
        loss_sum += float(loss) * (stop - start)
        correct += int(batch_correct)
        token_count += int(batch_tokens)
        if stop % 500 == 0 or stop == sample_count:
            print(f"teacher-forced validation: {stop}/{sample_count}", flush=True)
    return {
        "loss_mode": loss_mode,
        "samples": sample_count,
        "mean_sample_loss": loss_sum / sample_count,
        "perplexity": float(np.exp(min(loss_sum / sample_count, 20.0))),
        "token_accuracy": correct / max(token_count, 1),
        "tokens": token_count,
    }


def main(args: Args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, validation_refs, _ = train.load_split(args.split)
    tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=data.PROMPT_TOKEN_LEN)
    language_seed_tokens = tokenizer.tokenize_language_seed(args.suffix_seed)
    dataset = data.LiberoLanguageDataset(
        validation_refs,
        norm_stats_path=args.norm_stats,
        prompt_tokenizer=tokenizer,
        action_dim=32,
        suffix_len=args.suffix_len,
        suffix_seed=args.suffix_seed,
    )
    model_args = train.Args(
        split=args.split,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        output_dir=args.output_dir,
        adapter_type=args.adapter_type,
        rank=args.rank,
        alpha=args.alpha,
        suffix_seed=args.suffix_seed,
    )
    _, model = train.load_model(model_args)
    model_def, state = nnx.split(model)
    if args.adapter is not None:
        apply_adapter(state, args.adapter)
    trained_adapter = train.adapter_arrays(state)
    (
        prefill,
        prefill_incremental,
        suffix_active,
        suffix_disabled,
        suffix_step_active,
        suffix_step_disabled,
        suffix_block_active,
        suffix_block_disabled,
        suffix_request_active,
        suffix_request_disabled,
        actions,
    ) = _make_functions(
        model_def,
        suffix_len=args.suffix_len,
        seed_len=len(language_seed_tokens),
    )

    teacher_forced = None
    if args.teacher_forced_samples > 0:
        teacher_forced = _teacher_forced_evaluation(
            state,
            model_def,
            dataset,
            samples=args.teacher_forced_samples,
            loss_mode=args.teacher_forced_loss_mode,
            seed_len=len(language_seed_tokens),
        )
        (args.output_dir / "teacher_forced.json").write_text(
            json.dumps(teacher_forced, indent=2, sort_keys=True) + "\n"
        )

    selected = _select_samples(validation_refs, per_task=args.samples_per_task, seed=args.seed)
    if args.max_generated_samples:
        selected = selected[: args.max_generated_samples]
    index_by_ref = {
        (sample.suite, sample.task, sample.demo, sample.frame): index for index, sample in enumerate(validation_refs)
    }
    predictions = []
    for sample in selected:
        index = index_by_ref[(sample.suite, sample.task, sample.demo, sample.frame)]
        observation = data.collate([dataset[index]])[0]
        prediction = _generate(
            state,
            observation,
            tokenizer,
            prefill_incremental,
            suffix_step_active,
            suffix_block_active,
            suffix_len=args.suffix_len,
            suffix_seed=args.suffix_seed,
        )
        record = {
            **dataclasses.asdict(sample),
            "prediction": prediction,
            "normalized_exact": normalize_text(prediction) == normalize_text(sample.target),
            "word_f1": word_f1(prediction, sample.target),
        }
        predictions.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    with (args.output_dir / "predictions.jsonl").open("w") as stream:
        for record in predictions:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    first_batch = data.collate([dataset[0]])
    first_observation, first_suffix_inputs, _, first_suffix_mask, first_suffix_loss_mask = first_batch
    trained_root_cache, _ = prefill(state, first_observation)
    action_sample_count = min(args.action_invariance_samples, len(selected))
    action_observations = [
        data.collate([dataset[index_by_ref[(sample.suite, sample.task, sample.demo, sample.frame)]]])[0]
        for sample in selected[:action_sample_count]
    ]
    noises = [
        jax.random.normal(jax.random.fold_in(jax.random.key(123), index), (1, 10, 32))
        for index in range(action_sample_count)
    ]
    trained_actions = [
        np.asarray(actions(state, observation, noise))
        for observation, noise in zip(action_observations, noises, strict=True)
    ]
    zero_adapter = {name: np.zeros_like(value) for name, value in trained_adapter.items()}
    _set_adapter(state, zero_adapter)
    zero_root_cache, _ = prefill(state, first_observation)
    zero_actions = [
        np.asarray(actions(state, observation, noise))
        for observation, noise in zip(action_observations, noises, strict=True)
    ]
    action_max_abs_diff = max(
        float(np.max(np.abs(trained - zero))) for trained, zero in zip(trained_actions, zero_actions, strict=True)
    )
    action_exact = all(
        np.array_equal(trained, zero) for trained, zero in zip(trained_actions, zero_actions, strict=True)
    )
    root_cache_max_abs_diff = max(
        float(np.max(np.abs(np.asarray(trained) - np.asarray(zero))))
        for trained, zero in zip(jax.tree.leaves(trained_root_cache), jax.tree.leaves(zero_root_cache), strict=True)
    )
    root_cache_exact = all(
        np.array_equal(np.asarray(trained), np.asarray(zero))
        for trained, zero in zip(jax.tree.leaves(trained_root_cache), jax.tree.leaves(zero_root_cache), strict=True)
    )
    _set_adapter(state, trained_adapter)

    full_root_cache, full_prefix_mask = prefill(state, first_observation)
    full_logits = suffix_active(
        state,
        full_root_cache,
        full_prefix_mask,
        first_suffix_inputs,
        first_suffix_mask,
    )
    full_disabled_logits = suffix_disabled(
        state,
        full_root_cache,
        full_prefix_mask,
        first_suffix_inputs,
        first_suffix_mask,
    )
    incremental_cache, incremental_prefix_mask = prefill_incremental(state, first_observation)
    valid_suffix_tokens = int(np.asarray(first_suffix_mask).sum())
    seed_len = len(language_seed_tokens)
    seed_inputs = np.zeros_like(np.asarray(first_suffix_inputs))
    seed_inputs[:, :seed_len] = np.asarray(first_suffix_inputs[:, :seed_len])
    seed_mask = np.zeros_like(np.asarray(first_suffix_mask))
    seed_mask[:, :seed_len] = np.asarray(first_suffix_mask[:, :seed_len])
    seed_logits, incremental_cache = suffix_block_active(
        state,
        incremental_cache,
        incremental_prefix_mask,
        seed_inputs,
        seed_mask,
    )
    incremental_logits = [seed_logits[:, :seed_len]]
    incremental_disabled_cache, _ = prefill_incremental(state, first_observation)
    disabled_seed_logits, incremental_disabled_cache = suffix_block_disabled(
        state,
        incremental_disabled_cache,
        incremental_prefix_mask,
        seed_inputs,
        seed_mask,
    )
    incremental_disabled_logits = [disabled_seed_logits[:, :seed_len]]
    for suffix_index in range(seed_len, valid_suffix_tokens):
        token_logits, incremental_cache = suffix_step_active(
            state,
            incremental_cache,
            incremental_prefix_mask,
            first_suffix_inputs[:, suffix_index : suffix_index + 1],
            np.asarray(suffix_index, dtype=np.int32),
        )
        incremental_logits.append(token_logits)
        disabled_token_logits, incremental_disabled_cache = suffix_step_disabled(
            state,
            incremental_disabled_cache,
            incremental_prefix_mask,
            first_suffix_inputs[:, suffix_index : suffix_index + 1],
            np.asarray(suffix_index, dtype=np.int32),
        )
        incremental_disabled_logits.append(disabled_token_logits)
    incremental_logits = jnp.concatenate(incremental_logits, axis=1)
    incremental_disabled_logits = jnp.concatenate(incremental_disabled_logits, axis=1)
    full_valid_logits, incremental_logits, full_disabled_logits, incremental_disabled_logits = jax.device_get(
        (
            full_logits[:, :valid_suffix_tokens],
            incremental_logits,
            full_disabled_logits[:, :valid_suffix_tokens],
            incremental_disabled_logits,
        )
    )
    evaluated_positions = np.asarray(first_suffix_loss_mask[0, :valid_suffix_tokens], dtype=np.bool_)
    full_valid_logits = np.asarray(full_valid_logits)[:, evaluated_positions]
    incremental_logits = np.asarray(incremental_logits)[:, evaluated_positions]
    full_disabled_logits = np.asarray(full_disabled_logits)[:, evaluated_positions]
    incremental_disabled_logits = np.asarray(incremental_disabled_logits)[:, evaluated_positions]
    incremental_logit_abs_diff = np.abs(full_valid_logits - incremental_logits)
    incremental_disabled_logit_abs_diff = np.abs(full_disabled_logits - incremental_disabled_logits)
    incremental_argmax_exact = np.array_equal(
        np.argmax(full_valid_logits, axis=-1),
        np.argmax(incremental_logits, axis=-1),
    )
    incremental_disabled_argmax_exact = np.array_equal(
        np.argmax(full_disabled_logits, axis=-1),
        np.argmax(incremental_disabled_logits, axis=-1),
    )

    root_cache, prefix_mask = prefill_incremental(state, first_observation)
    seed_tokens = tokenizer.tokenize_language_seed(args.suffix_seed)
    one_token = np.asarray([[seed_tokens[0]]], dtype=np.int32)
    suffix_index_zero = np.asarray(0, dtype=np.int32)
    prefix_timing = _timed(
        lambda: prefill_incremental(state, first_observation),
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    suffix_enabled_timing = _timed(
        lambda: suffix_step_active(state, root_cache, prefix_mask, one_token, suffix_index_zero),
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    suffix_disabled_timing = _timed(
        lambda: suffix_step_disabled(state, root_cache, prefix_mask, one_token, suffix_index_zero),
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    target_token_lengths = []
    suffix_input_lengths = []
    for sample in selected:
        _, _, suffix_mask, loss_mask = tokenizer.tokenize_language_suffix(
            args.suffix_seed, sample.target, max_len=args.suffix_len
        )
        target_token_lengths.append(int(loss_mask.sum()))
        suffix_input_lengths.append(int(suffix_mask.sum()))
    median_target_tokens = float(np.median(target_token_lengths))
    median_suffix_input_tokens = int(np.median(suffix_input_lengths))
    request_tokens = np.asarray(first_suffix_inputs[:, :median_suffix_input_tokens])
    suffix_request_enabled_timing = _timed(
        lambda: suffix_request_active(state, root_cache, prefix_mask, request_tokens),
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    suffix_request_disabled_timing = _timed(
        lambda: suffix_request_disabled(state, root_cache, prefix_mask, request_tokens),
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    adapter_only_one_token = None
    request_overhead_ms = suffix_request_enabled_timing["median_ms"] - suffix_request_disabled_timing["median_ms"]
    if args.adapter_type == "final_mlp":
        adapter_only_one_token = _adapter_only_timing(
            trained_adapter,
            tokens=1,
            warmups=args.timing_warmups,
            repeats=args.timing_repeats,
        )

    summary = {
        "adapter": None if args.adapter is None else str(args.adapter.resolve()),
        "adapter_type": args.adapter_type,
        "generated_samples": len(predictions),
        "normalized_exact": float(np.mean([item["normalized_exact"] for item in predictions])),
        "mean_word_f1": float(np.mean([item["word_f1"] for item in predictions])),
        "action_exact_with_adapter_disabled_path": action_exact,
        "action_max_abs_diff": action_max_abs_diff,
        "action_invariance_samples": action_sample_count,
        "root_cache_exact_with_adapter_zeroed": root_cache_exact,
        "root_cache_max_abs_diff": root_cache_max_abs_diff,
        "incremental_teacher_forced_positions": int(evaluated_positions.sum()),
        "incremental_teacher_forced_argmax_exact": incremental_argmax_exact,
        "incremental_teacher_forced_logit_max_abs_diff": float(np.max(incremental_logit_abs_diff)),
        "incremental_teacher_forced_logit_mean_abs_diff": float(np.mean(incremental_logit_abs_diff)),
        "incremental_teacher_forced_argmax_match_by_position": (
            np.argmax(full_valid_logits, axis=-1) == np.argmax(incremental_logits, axis=-1)
        )[0].tolist(),
        "incremental_teacher_forced_logit_max_abs_diff_by_position": np.max(incremental_logit_abs_diff, axis=-1)[
            0
        ].tolist(),
        "incremental_disabled_argmax_exact": incremental_disabled_argmax_exact,
        "incremental_disabled_logit_max_abs_diff": float(np.max(incremental_disabled_logit_abs_diff)),
        "incremental_disabled_logit_mean_abs_diff": float(np.mean(incremental_disabled_logit_abs_diff)),
        "prefix_prefill": prefix_timing,
        "one_token_suffix_adapter_enabled": suffix_enabled_timing,
        "one_token_suffix_adapter_disabled": suffix_disabled_timing,
        "median_suffix_input_tokens": median_suffix_input_tokens,
        "median_suffix_adapter_enabled": suffix_request_enabled_timing,
        "median_suffix_adapter_disabled": suffix_request_disabled_timing,
        "adapter_only_one_token": adapter_only_one_token,
        "median_target_tokens": median_target_tokens,
        "measured_adapter_overhead_per_median_request_ms": request_overhead_ms,
        "prefix_saving_to_adapter_overhead_ratio": (
            None if request_overhead_ms <= 0 else prefix_timing["median_ms"] / request_overhead_ms
        ),
    }
    if teacher_forced is not None:
        summary["teacher_forced_validation"] = teacher_forced
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    dataset.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
