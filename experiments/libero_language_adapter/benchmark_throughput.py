"""Controlled end-to-end throughput benchmark for the suffix language adapter."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import statistics
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from experiments.libero_language_adapter import data
from experiments.libero_language_adapter import evaluate
from experiments.libero_language_adapter import train
from openpi.models import model as model_lib
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import policy as policy_lib


@dataclasses.dataclass(frozen=True)
class Args:
    split: Path
    checkpoint: Path
    norm_stats: Path
    adapter: Path
    output: Path
    rank: int = 16
    alpha: float = 16.0
    seed: int = 7
    sample_index: int = 0
    max_decoding_steps: int = 20
    steps_per_frame: int = 5
    num_action_steps: int = 10
    warmup_frames: int = 8
    measured_frames: int = 30
    repeats: int = 3
    suffix_seed: str = "Subtask: "


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _sync_tree(tree) -> None:
    jax.block_until_ready(tree)


def _run_isolated_frame(
    policy: policy_lib.Policy,
    raw_observation: dict,
    noise: np.ndarray,
    *,
    adapter_active: bool,
    max_decoding_steps: int,
    num_action_steps: int,
) -> None:
    inputs, _ = policy._prepare_batched_inputs([raw_observation], allow_variable_length=False)
    observation = model_lib.Observation.from_dict(inputs)
    policy._rng, text_rng, action_rng = jax.random.split(policy._rng, 3)

    # Isolated execution computes the observation/task prefix independently for
    # action and language. The language branch otherwise matches the CB path.
    action_root = policy._prefill(observation, align_right=False, max_decoding_steps=0)
    actions = policy._sample_actions_with_kv(
        action_rng,
        observation,
        action_root,
        num_steps=num_action_steps,
        noise=jnp.asarray(noise)[None],
    )
    language_root = policy._prefill(
        observation,
        align_right=False,
        max_decoding_steps=max_decoding_steps,
    )
    state = policy._init_language_adapter_incremental_state(
        language_root,
        text_rng,
        jnp.asarray(policy._language_seed_tokens),
        adapter_active=adapter_active,
    )
    tokens, _, _ = policy._generate_n_tokens(
        state,
        tokens_to_generate=max_decoding_steps,
        PALIGEMMA_EOS_TOKEN=-1,
        temperature=0.0,
        adapter_active=adapter_active,
    )
    _sync_tree((actions, tokens))


def _run_isolated(
    policy,
    raw_observation,
    noise,
    args: Args,
    *,
    adapter_active: bool,
) -> list[float]:
    samples = []
    for frame in range(args.warmup_frames + args.measured_frames):
        start = time.perf_counter()
        _run_isolated_frame(
            policy,
            raw_observation,
            noise,
            adapter_active=adapter_active,
            max_decoding_steps=args.max_decoding_steps,
            num_action_steps=args.num_action_steps,
        )
        elapsed = (time.perf_counter() - start) * 1_000
        if frame >= args.warmup_frames:
            samples.append(elapsed)
    return samples


def _run_continuous(
    policy,
    raw_observation,
    noise,
    args: Args,
    *,
    adapter_active: bool,
) -> tuple[list[float], list[int]]:
    manager = policy.init_continuous_batching()
    samples = []
    batch_sizes = []
    total_frames = args.warmup_frames + args.measured_frames
    for frame in range(total_frames):
        active = manager.get_all_active_requests()
        observations = [raw_observation] * (1 + len(active))
        request_ids = [None, *active]
        start = time.perf_counter()
        outputs = policy.infer_text_actions_continuous_batch(
            observations,
            manager,
            request_ids=request_ids,
            steps_per_frame=args.steps_per_frame,
            num_action_steps=args.num_action_steps,
            max_decoding_steps=args.max_decoding_steps,
            temperature=0.0,
            PALIGEMMA_EOS_TOKEN=-1,
            noise=noise,
            generate_actions_for_resumed=False,
            language_adapter_active=adapter_active,
        )
        _sync_tree(outputs[0]["actions"])
        elapsed = (time.perf_counter() - start) * 1_000
        if frame >= args.warmup_frames:
            samples.append(elapsed)
            batch_sizes.append(len(outputs))
    return samples, batch_sizes


def _run_paired_continuous(policy, raw_observation, noise, args: Args) -> dict:
    """Alternate adapter paths frame-by-frame to cancel slow clock drift."""
    managers = {active: policy.init_continuous_batching() for active in (False, True)}
    samples = {active: [] for active in (False, True)}
    total_frames = args.warmup_frames + args.measured_frames
    for frame in range(total_frames):
        order = (False, True) if frame % 2 == 0 else (True, False)
        for active in order:
            manager = managers[active]
            request_ids = manager.get_all_active_requests()
            start = time.perf_counter()
            outputs = policy.infer_text_actions_continuous_batch(
                [raw_observation] * (1 + len(request_ids)),
                manager,
                request_ids=[None, *request_ids],
                steps_per_frame=args.steps_per_frame,
                num_action_steps=args.num_action_steps,
                max_decoding_steps=args.max_decoding_steps,
                temperature=0.0,
                PALIGEMMA_EOS_TOKEN=-1,
                noise=noise,
                generate_actions_for_resumed=False,
                language_adapter_active=active,
            )
            _sync_tree(outputs[0]["actions"])
            if frame >= args.warmup_frames:
                samples[active].append((time.perf_counter() - start) * 1_000)
    paired_deltas = [on - off for on, off in zip(samples[True], samples[False])]
    return {
        "oxygen_adapter_off": _summary(samples[False]),
        "oxygen_adapter_on": _summary(samples[True]),
        "paired_on_minus_off": _summary(paired_deltas),
    }


def main(args: Args) -> None:
    _, validation_refs, _ = train.load_split(args.split)
    tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=data.PROMPT_TOKEN_LEN)
    dataset = data.LiberoLanguageDataset(
        validation_refs,
        norm_stats_path=args.norm_stats,
        prompt_tokenizer=tokenizer,
        action_dim=32,
        suffix_len=args.max_decoding_steps,
        suffix_seed=args.suffix_seed,
    )
    observation = dataset[args.sample_index][0]
    raw_observation = {
        key: jax.tree.map(np.asarray, value) for key, value in observation.to_dict().items() if value is not None
    }
    model_args = train.Args(
        split=args.split,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        output_dir=args.output.parent,
        adapter_type="suffix_lora",
        rank=args.rank,
        alpha=args.alpha,
        suffix_seed=args.suffix_seed,
        seed=args.seed,
    )
    _, model = train.load_model(model_args)
    graphdef, state = nnx.split(model)
    evaluate.apply_adapter(state, args.adapter)
    model = nnx.merge(graphdef, state)
    policy = policy_lib.Policy(model, rng=jax.random.key(args.seed), language_seed=args.suffix_seed)
    noise = np.random.default_rng(args.seed).standard_normal(
        (model.action_horizon, model.action_dim), dtype=np.float32
    )

    # Compile both static adapter paths before collecting any repeat.
    smoke = {}
    for active in (False, True):
        manager = policy.init_continuous_batching()
        output = policy.infer_text_actions_continuous_batch(
            [raw_observation],
            manager,
            steps_per_frame=args.steps_per_frame,
            num_action_steps=args.num_action_steps,
            max_decoding_steps=args.max_decoding_steps,
            temperature=0.0,
            PALIGEMMA_EOS_TOKEN=-1,
            noise=noise,
            language_adapter_active=active,
        )[0]
        smoke[str(active)] = {
            "actions": np.asarray(output["actions"]),
            "request_active": not output["is_finished"],
            "tokens": np.asarray(output["tokens_this_frame"]),
        }
        _run_isolated_frame(
            policy,
            raw_observation,
            noise,
            adapter_active=active,
            max_decoding_steps=args.max_decoding_steps,
            num_action_steps=args.num_action_steps,
        )

    repeat_results = []
    run_functions = {
        "isolated_adapter_off": lambda: (
            _run_isolated(policy, raw_observation, noise, args, adapter_active=False),
            None,
        ),
        "isolated_adapter_on": lambda: (
            _run_isolated(policy, raw_observation, noise, args, adapter_active=True),
            None,
        ),
        "oxygen_adapter_off": lambda: _run_continuous(
            policy, raw_observation, noise, args, adapter_active=False
        ),
        "oxygen_adapter_on": lambda: _run_continuous(
            policy, raw_observation, noise, args, adapter_active=True
        ),
    }
    names = list(run_functions)
    for repeat in range(args.repeats):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        repeat_result = {"repeat": repeat, "execution_order": order}
        for name in order:
            samples, batch_sizes = run_functions[name]()
            repeat_result[name] = _summary(samples)
            if batch_sizes is not None:
                repeat_result[f"{name}_batch_sizes"] = batch_sizes
        repeat_results.append(repeat_result)

    medians = {
        key: statistics.median(result[key]["median_ms"] for result in repeat_results)
        for key in ("isolated_adapter_off", "isolated_adapter_on", "oxygen_adapter_off", "oxygen_adapter_on")
    }
    paired = _run_paired_continuous(policy, raw_observation, noise, args)
    payload = {
        "configuration": {
            **dataclasses.asdict(args),
            "split": str(args.split.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "norm_stats": str(args.norm_stats.resolve()),
            "adapter": str(args.adapter.resolve()),
            "backend": "JAX",
            "eos_enabled": False,
            "arrivals_per_frame": 1,
        },
        "smoke": {
            "action_max_abs_diff_adapter_on_vs_off": float(
                np.max(np.abs(smoke["True"]["actions"] - smoke["False"]["actions"]))
            ),
            "same_request_lifecycle": smoke["True"]["request_active"] == smoke["False"]["request_active"],
            "adapter_changes_tokens": not np.array_equal(smoke["True"]["tokens"], smoke["False"]["tokens"]),
        },
        "repeats": repeat_results,
        "paired_continuous": paired,
        "median_of_repeat_medians_ms": medians,
        "derived": {
            "oxygen_speedup_without_lora": medians["isolated_adapter_off"] / medians["oxygen_adapter_off"],
            "oxygen_speedup_with_lora": medians["isolated_adapter_on"] / medians["oxygen_adapter_on"],
            "isolated_lora_frame_delta_ms": medians["isolated_adapter_on"] - medians["isolated_adapter_off"],
            "isolated_lora_frame_slowdown_fraction": (
                medians["isolated_adapter_on"] / medians["isolated_adapter_off"] - 1
            ),
            "lora_frame_delta_ms": medians["oxygen_adapter_on"] - medians["oxygen_adapter_off"],
            "lora_frame_slowdown_fraction": medians["oxygen_adapter_on"] / medians["oxygen_adapter_off"] - 1,
            "lora_throughput_reduction_fraction": 1 - medians["oxygen_adapter_off"] / medians["oxygen_adapter_on"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main(tyro.cli(Args))
