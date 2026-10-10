#!/usr/bin/env python3
"""Direct stage timing for pi0.5 baseline/shared-KV at small language lengths."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from experiments.baseline.runner import run_baseline
from experiments.common.setup import collect_metadata, create_policy, setup_jax_cache
from experiments.common.workload import create_synthetic_observation
from experiments.shared_kv.runner import run_shared_kv
from openpi.models import model as _model
from openpi.policies.policy import _to_jax_batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="pi05_o2_libero")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prompt", default="pick the red cup")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--decode-steps", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measured-runs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed(fn):
    started = time.perf_counter()
    output = fn()
    jax.block_until_ready(output)
    return output, (time.perf_counter() - started) * 1000.0


def prepare(policy, raw_observation):
    inputs = jax.tree.map(lambda value: value, raw_observation)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(_to_jax_batch, inputs)
    return _model.Observation.from_dict(inputs)


def profile_baseline(policy, observation, denoise_steps, decode_steps, rng, noise):
    action_prefix, action_prefill_ms = timed(
        lambda: policy._prefill(
            observation, align_right=False, max_decoding_steps=0
        )
    )
    actions, denoise_ms = timed(
        lambda: policy._sample_actions_with_kv(
            rng, observation, action_prefix,
            num_steps=denoise_steps, noise=noise,
        )
    )
    text_prefix, text_prefill_ms = timed(
        lambda: policy._prefill(
            observation, align_right=False, max_decoding_steps=decode_steps
        )
    )
    text_result, decode_ms = timed(
        lambda: policy._sample_text_with_kv(
            rng, text_prefix,
            max_decoding_steps=decode_steps,
            PALIGEMMA_EOS_TOKEN=-1,
            temperature=0.0,
        )
    )
    return {
        "action_prefill_ms": action_prefill_ms,
        "denoise_ms": denoise_ms,
        "text_prefill_ms": text_prefill_ms,
        "decode_ms": decode_ms,
        "stage_sum_ms": action_prefill_ms + denoise_ms + text_prefill_ms + decode_ms,
        "action_checksum": float(jnp.sum(actions)),
        "token_checksum": int(jnp.sum(text_result[0])),
    }


def profile_shared(policy, observation, denoise_steps, decode_steps, rng, noise):
    prefix, prefill_ms = timed(
        lambda: policy._prefill(
            observation, align_right=False, max_decoding_steps=decode_steps
        )
    )
    text_result, decode_ms = timed(
        lambda: policy._sample_text_with_kv(
            rng, prefix,
            max_decoding_steps=decode_steps,
            PALIGEMMA_EOS_TOKEN=-1,
            temperature=0.0,
        )
    )
    actions, denoise_ms = timed(
        lambda: policy._sample_actions_with_kv(
            rng, observation, prefix,
            num_steps=denoise_steps, noise=noise,
        )
    )
    return {
        "prefill_ms": prefill_ms,
        "denoise_ms": denoise_ms,
        "decode_ms": decode_ms,
        "stage_sum_ms": prefill_ms + denoise_ms + decode_ms,
        "action_checksum": float(jnp.sum(actions)),
        "token_checksum": int(jnp.sum(text_result[0])),
    }


def summarize(rows):
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, float) and not key.endswith("checksum")
    ]
    return {
        key: {
            "median_ms": statistics.median(row[key] for row in rows),
            "mean_ms": statistics.fmean(row[key] for row in rows),
            "min_ms": min(row[key] for row in rows),
            "max_ms": max(row[key] for row in rows),
        }
        for key in numeric_keys
    }


def main():
    args = parse_args()
    from random_init_support import pi_checkpoint
    args.checkpoint_dir = pi_checkpoint(args.checkpoint_dir, args.random_init)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    setup_jax_cache()
    policy = create_policy(args.policy, checkpoint_dir=args.checkpoint_dir)
    raw_observation = create_synthetic_observation(
        args.prompt, seed=42, policy_config=args.policy
    )
    # Some input transforms mutate nested observation containers. Keep the
    # raw workload pristine for the unmodified end-to-end runner measurements.
    observation = prepare(policy, copy.deepcopy(raw_observation))
    rng = jax.random.key(42)
    noise = jax.random.normal(
        rng,
        (observation.state.shape[0], policy._model.action_horizon, policy._model.action_dim),
    )
    jax.block_until_ready(noise)

    results = []
    for decode_steps in args.decode_steps:
        baseline_stages = []
        shared_stages = []
        baseline_e2e = []
        shared_e2e = []
        total_runs = args.warmup_runs + args.measured_runs
        for repeat in range(total_runs):
            # Alternate order to reduce monotonic thermal/order bias.
            order = ("baseline", "shared") if repeat % 2 == 0 else ("shared", "baseline")
            current = {}
            for mode in order:
                if mode == "baseline":
                    current[mode] = profile_baseline(
                        policy, observation, args.denoise_steps, decode_steps, rng, noise
                    )
                else:
                    current[mode] = profile_shared(
                        policy, observation, args.denoise_steps, decode_steps, rng, noise
                    )
            walls = {}
            # Reverse the component order for the end-to-end calls so neither
            # mode is systematically measured first or second.
            for mode in reversed(order):
                runner = run_baseline if mode == "baseline" else run_shared_kv
                walls[mode] = runner(
                    policy, raw_observation,
                    num_denoise_steps=args.denoise_steps,
                    max_decoding_steps=decode_steps,
                )
            baseline_wall = walls["baseline"]
            shared_wall = walls["shared"]
            if repeat >= args.warmup_runs:
                current["baseline"]["repeat"] = repeat - args.warmup_runs
                current["shared"]["repeat"] = repeat - args.warmup_runs
                baseline_stages.append(current["baseline"])
                shared_stages.append(current["shared"])
                baseline_e2e.append(baseline_wall)
                shared_e2e.append(shared_wall)
        results.append({
            "S": args.denoise_steps,
            "N": decode_steps,
            "baseline": {
                "stages": baseline_stages,
                "stage_summary": summarize(baseline_stages),
                "e2e_frames": baseline_e2e,
                "e2e_median_ms": statistics.median(row["frame_ms"] for row in baseline_e2e),
                "e2e_mean_ms": statistics.fmean(row["frame_ms"] for row in baseline_e2e),
            },
            "shared_kv": {
                "stages": shared_stages,
                "stage_summary": summarize(shared_stages),
                "e2e_frames": shared_e2e,
                "e2e_median_ms": statistics.median(row["frame_ms"] for row in shared_e2e),
                "e2e_mean_ms": statistics.fmean(row["frame_ms"] for row in shared_e2e),
            },
        })
    payload = {
        "metadata": {
            **collect_metadata(),
            "policy": args.policy,
            "checkpoint_dir": str(args.checkpoint_dir),
            "prompt": args.prompt,
            "seed": 42,
            "denoise_steps": args.denoise_steps,
            "decode_steps": args.decode_steps,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "stage_timing": "JIT components, synchronized after every stage",
            "e2e_timing": "unmodified submission baseline/shared_kv runners",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
