from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import re
import time

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
    adapter: Path
    output_dir: Path
    rank: int
    suffix_len: int = 20
    samples_per_task: int = 2
    seed: int = 7
    timing_warmups: int = 5
    timing_repeats: int = 50
    teacher_forced_samples: int = 0
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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def _word_f1(prediction: str, target: str) -> float:
    prediction_words = _normalize_text(prediction).split()
    target_words = _normalize_text(target).split()
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


def _make_functions(model_def):
    @jax.jit
    def prefill(state, observation):
        model = nnx.merge(model_def, state)
        return model.prefill_language_root(None, observation, train=False)

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
    def actions(state, observation, noise):
        model = nnx.merge(model_def, state)
        return model.sample_actions(jax.random.key(0), observation, num_steps=10, noise=noise)

    return prefill, suffix_active, suffix_disabled, actions


def _generate(
    state,
    observation,
    tokenizer,
    prefill,
    suffix_active,
    *,
    suffix_len: int,
) -> str:
    root_cache, prefix_mask = prefill(state, observation)
    seed_tokens = tokenizer.tokenize_language_seed("Subtask: ")
    inputs = np.zeros((1, suffix_len), dtype=np.int32)
    mask = np.zeros((1, suffix_len), dtype=np.bool_)
    inputs[0, : len(seed_tokens)] = seed_tokens
    mask[0, : len(seed_tokens)] = True
    generated = []
    for position in range(len(seed_tokens) - 1, suffix_len):
        logits = suffix_active(state, root_cache, prefix_mask, inputs, mask)
        token = int(jax.device_get(jnp.argmax(logits[0, position])))
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        if position + 1 == suffix_len:
            break
        inputs[0, position + 1] = token
        mask[0, position + 1] = True
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
    down = next(value for name, value in adapter.items() if name.endswith("/down/kernel"))
    up = next(value for name, value in adapter.items() if name.endswith("/up/kernel"))
    down = jax.device_put(down.astype(np.float32))
    up = jax.device_put(up.astype(np.float32))
    hidden = jax.device_put(np.ones((1, tokens, down.shape[1]), dtype=np.float32))

    @jax.jit
    def run(inputs):
        def layer(value, weights):
            layer_down, layer_up = weights
            update = jax.nn.gelu(value @ layer_down) @ layer_up
            return value + update, None

        output, _ = jax.lax.scan(layer, inputs, (down, up))
        return output

    return _timed(lambda: run(hidden), warmups=warmups, repeats=repeats)


def _teacher_forced_evaluation(
    state,
    model_def,
    dataset,
    *,
    samples: int,
    batch_size: int = 2,
) -> dict[str, float | int]:
    if samples <= 0:
        raise ValueError("teacher-forced sample count must be positive")
    sample_count = min(samples, len(dataset))
    eval_step = train.make_eval_step(model_def)
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
        "samples": sample_count,
        "mean_sample_loss": loss_sum / sample_count,
        "perplexity": float(np.exp(min(loss_sum / sample_count, 20.0))),
        "token_accuracy": correct / max(token_count, 1),
        "tokens": token_count,
    }


def main(args: Args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, validation_refs, _ = train.load_split(args.split)
    tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=200)
    dataset = data.LiberoLanguageDataset(
        validation_refs,
        norm_stats_path=args.norm_stats,
        prompt_tokenizer=tokenizer,
        action_dim=32,
        suffix_len=args.suffix_len,
    )
    model_args = train.Args(
        split=args.split,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        output_dir=args.output_dir,
        rank=args.rank,
    )
    _, model = train.load_model(model_args)
    model_def, state = nnx.split(model)
    apply_adapter(state, args.adapter)
    trained_adapter = train.adapter_arrays(state)
    prefill, suffix_active, suffix_disabled, actions = _make_functions(model_def)

    teacher_forced = None
    if args.teacher_forced_samples > 0:
        teacher_forced = _teacher_forced_evaluation(
            state,
            model_def,
            dataset,
            samples=args.teacher_forced_samples,
        )

    selected = _select_samples(validation_refs, per_task=args.samples_per_task, seed=args.seed)
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
            prefill,
            suffix_active,
            suffix_len=args.suffix_len,
        )
        record = {
            **dataclasses.asdict(sample),
            "prediction": prediction,
            "normalized_exact": _normalize_text(prediction) == _normalize_text(sample.target),
            "word_f1": _word_f1(prediction, sample.target),
        }
        predictions.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    with (args.output_dir / "predictions.jsonl").open("w") as stream:
        for record in predictions:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

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
    _set_adapter(state, trained_adapter)

    first_observation = data.collate([dataset[0]])[0]
    root_cache, prefix_mask = prefill(state, first_observation)
    one_token = np.asarray([[tokenizer.tokenize_language_seed("Subtask: ")[-1]]], dtype=np.int32)
    one_mask = np.ones_like(one_token, dtype=np.bool_)
    prefix_timing = _timed(
        lambda: prefill(state, first_observation),
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    suffix_enabled_timing = _timed(
        lambda: suffix_active(state, root_cache, prefix_mask, one_token, one_mask),
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    suffix_disabled_timing = _timed(
        lambda: suffix_disabled(state, root_cache, prefix_mask, one_token, one_mask),
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    target_token_lengths = []
    for sample in selected:
        _, _, _, loss_mask = tokenizer.tokenize_language_suffix("Subtask: ", sample.target, max_len=args.suffix_len)
        target_token_lengths.append(int(loss_mask.sum()))
    median_target_tokens = float(np.median(target_token_lengths))
    adapter_only_one_token = _adapter_only_timing(
        trained_adapter,
        tokens=1,
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    # Language decoding is autoregressive, so charge the one-token adapter cost
    # once per generated token instead of timing all target tokens in parallel.
    request_overhead_ms = adapter_only_one_token["median_ms"] * median_target_tokens

    summary = {
        "adapter": str(args.adapter.resolve()),
        "generated_samples": len(predictions),
        "normalized_exact": float(np.mean([item["normalized_exact"] for item in predictions])),
        "mean_word_f1": float(np.mean([item["word_f1"] for item in predictions])),
        "action_exact_with_adapter_disabled_path": action_exact,
        "action_max_abs_diff": action_max_abs_diff,
        "action_invariance_samples": action_sample_count,
        "prefix_prefill": prefix_timing,
        "one_token_suffix_adapter_enabled": suffix_enabled_timing,
        "one_token_suffix_adapter_disabled": suffix_disabled_timing,
        "adapter_only_one_token": adapter_only_one_token,
        "median_target_tokens": median_target_tokens,
        "estimated_adapter_overhead_per_median_request_ms": request_overhead_ms,
        "prefix_saving_to_adapter_overhead_ratio": prefix_timing["median_ms"] / max(request_overhead_ms, 1e-6),
    }
    if teacher_forced is not None:
        summary["teacher_forced_validation"] = teacher_forced
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    dataset.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
