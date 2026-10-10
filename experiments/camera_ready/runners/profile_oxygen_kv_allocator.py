#!/usr/bin/env python3
"""Profile OxyGen KV aliasing and allocator behavior under request churn."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import jax
import numpy as np

from experiments.common.setup import collect_metadata, create_policy, setup_jax_cache
from experiments.common.workload import create_synthetic_observation
from experiments.continuous_batching.runner import run_continuous_batching
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
    parser.add_argument("--decode-steps", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=24)
    parser.add_argument("--measured-frames", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def memory_stats(device):
    stats = device.memory_stats() or {}
    keys = (
        "num_allocs",
        "bytes_in_use",
        "peak_bytes_in_use",
        "largest_alloc_size",
        "bytes_limit",
        "bytes_reserved",
        "peak_bytes_reserved",
        "largest_free_block_bytes",
        "pool_bytes",
        "peak_pool_bytes",
    )
    return {key: int(stats.get(key, 0)) for key in keys}


def prepare(policy, raw_observation):
    inputs = policy._input_transform(jax.tree.map(lambda value: value, raw_observation))
    inputs = jax.tree.map(_to_jax_batch, inputs)
    return _model.Observation.from_dict(inputs)


def buffer_pointer(array):
    try:
        return int(array.unsafe_buffer_pointer())
    except (AttributeError, ValueError, RuntimeError):
        return None


def alias_probe(policy, device, args):
    raw = create_synthetic_observation(args.prompt, seed=42, policy_config=args.policy)
    observation = prepare(policy, raw)
    rng = jax.random.key(42)

    # Compile all paths before taking allocator snapshots.
    warm_root = policy._prefill(
        observation, align_right=False, max_decoding_steps=args.decode_steps
    )
    warm_state = policy._init_incremental_state(warm_root, rng)
    warm_actions = policy._sample_actions_with_kv(
        rng, observation, warm_root, num_steps=args.denoise_steps
    )
    jax.block_until_ready((warm_state, warm_actions))
    del warm_root, warm_state, warm_actions
    gc.collect()

    before_root = memory_stats(device)
    root = policy._prefill(
        observation, align_right=False, max_decoding_steps=args.decode_steps
    )
    jax.block_until_ready(root)
    after_root = memory_stats(device)

    start = time.perf_counter()
    state = policy._init_incremental_state(root, rng)
    jax.block_until_ready(state)
    state_init_ms = (time.perf_counter() - start) * 1000.0
    after_state = memory_stats(device)

    root_leaves = jax.tree.leaves(root[1])
    state_leaves = jax.tree.leaves(state.kv_cache)
    aliases = []
    for index, (root_leaf, state_leaf) in enumerate(zip(root_leaves, state_leaves, strict=True)):
        root_ptr = buffer_pointer(root_leaf)
        state_ptr = buffer_pointer(state_leaf)
        aliases.append({
            "leaf": index,
            "shape": list(root_leaf.shape),
            "dtype": str(root_leaf.dtype),
            "bytes": int(root_leaf.size * root_leaf.dtype.itemsize),
            "root_pointer": root_ptr,
            "state_pointer": state_ptr,
            "same_pointer": root_ptr is not None and root_ptr == state_ptr,
        })

    # Python-side expert handles are references to the same immutable root tuple.
    start = time.perf_counter()
    read_only_handles = [root] * 10_000
    handle_create_us = (time.perf_counter() - start) * 1e6 / len(read_only_handles)
    after_handles = memory_stats(device)

    start = time.perf_counter()
    actions = policy._sample_actions_with_kv(
        rng, observation, root, num_steps=args.denoise_steps
    )
    jax.block_until_ready(actions)
    action_read_ms = (time.perf_counter() - start) * 1000.0
    after_action = memory_stats(device)

    # Retain K independently appendable states from one root. This measures the
    # current JAX implementation, which materializes a fixed-capacity cache at
    # the init JIT boundary rather than implementing copy-on-write paging.
    del actions, read_only_handles, state
    gc.collect()
    scaling_warm = policy._init_incremental_state(root, jax.random.fold_in(rng, 0))
    jax.block_until_ready(scaling_warm)
    del scaling_warm
    gc.collect()
    append_scaling = []
    for expert_count in (1, 2, 4):
        before = memory_stats(device)
        states = []
        start = time.perf_counter()
        for expert_index in range(expert_count):
            expert_state = policy._init_incremental_state(
                root, jax.random.fold_in(rng, expert_index)
            )
            jax.block_until_ready(expert_state)
            states.append(expert_state)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        after = memory_stats(device)
        append_scaling.append({
            "append_expert_count": expert_count,
            "total_init_ms": elapsed_ms,
            "init_ms_per_expert": elapsed_ms / expert_count,
            "bytes_in_use_delta": after["bytes_in_use"] - before["bytes_in_use"],
            "num_allocs_delta": after["num_allocs"] - before["num_allocs"],
        })
        del states, expert_state
        gc.collect()

    result = {
        "before_root": before_root,
        "after_root": after_root,
        "after_state": after_state,
        "after_handles": after_handles,
        "after_action": after_action,
        "root_kv_bytes": sum(item["bytes"] for item in aliases),
        "kv_leaf_count": len(aliases),
        "kv_alias_count": sum(item["same_pointer"] for item in aliases),
        "all_kv_leaves_alias": all(item["same_pointer"] for item in aliases),
        "state_init_ms": state_init_ms,
        "read_only_handle_create_us": handle_create_us,
        "action_read_ms": action_read_ms,
        "append_expert_scaling": append_scaling,
        "kv_aliases": aliases,
    }

    del root, observation
    gc.collect()
    return result


class ProfiledPolicy:
    def __init__(self, policy, device):
        self._policy = policy
        self._device = device
        self.samples = []

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def infer_text_actions_continuous_batch(self, obs_list, cache_manager, **kwargs):
        before = memory_stats(self._device)
        start = time.perf_counter()
        outputs = self._policy.infer_text_actions_continuous_batch(
            obs_list, cache_manager, **kwargs
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        after = memory_stats(self._device)
        timing = outputs[0]["policy_timing"] if outputs else {}
        self.samples.append({
            "frame": len(self.samples),
            "batch_size": len(obs_list),
            "active_states_after": len(cache_manager.active_states),
            "elapsed_ms": elapsed_ms,
            "policy_timing": timing,
            "before": before,
            "after": after,
        })
        return outputs


def summarize_churn(samples):
    measured = [row for row in samples if not row["is_warmup"]]
    if not measured:
        raise ValueError("No measured frames")

    first = measured[0]["after"]
    last = measured[-1]["after"]
    in_use = [row["after"]["bytes_in_use"] for row in measured]
    pool = [row["after"]["pool_bytes"] for row in measured]
    reserved = [row["after"]["bytes_reserved"] for row in measured]
    allocs = [row["after"]["num_allocs"] for row in measured]
    latencies = [row["elapsed_ms"] for row in measured]
    batches = [row["batch_size"] for row in measured]

    def slope(values):
        count = len(values)
        x_mean = (count - 1) / 2.0
        y_mean = statistics.fmean(values)
        denominator = sum((index - x_mean) ** 2 for index in range(count))
        if denominator == 0:
            return 0.0
        return sum(
            (index - x_mean) * (value - y_mean)
            for index, value in enumerate(values)
        ) / denominator

    window = min(10, len(measured))

    return {
        "measured_frames": len(measured),
        "batch_size_min": min(batches),
        "batch_size_max": max(batches),
        "batch_size_mean": statistics.fmean(batches),
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_median_ms": statistics.median(latencies),
        "latency_max_ms": max(latencies),
        "latency_first_window_mean_ms": statistics.fmean(latencies[:window]),
        "latency_last_window_mean_ms": statistics.fmean(latencies[-window:]),
        "latency_slope_ms_per_frame": slope(latencies),
        "num_allocs_first": allocs[0],
        "num_allocs_last": allocs[-1],
        "num_allocs_growth": allocs[-1] - allocs[0],
        "allocs_per_frame_upper_bound": (allocs[-1] - allocs[0]) / max(1, len(measured) - 1),
        "bytes_in_use_min": min(in_use),
        "bytes_in_use_max": max(in_use),
        "bytes_in_use_drift": last["bytes_in_use"] - first["bytes_in_use"],
        "bytes_in_use_first_window_mean": statistics.fmean(in_use[:window]),
        "bytes_in_use_last_window_mean": statistics.fmean(in_use[-window:]),
        "bytes_in_use_slope_per_frame": slope(in_use),
        "pool_bytes_min": min(pool),
        "pool_bytes_max": max(pool),
        "pool_bytes_drift": last["pool_bytes"] - first["pool_bytes"],
        "bytes_reserved_min": min(reserved),
        "bytes_reserved_max": max(reserved),
        "bytes_reserved_drift": last["bytes_reserved"] - first["bytes_reserved"],
        "largest_free_block_min": min(row["after"]["largest_free_block_bytes"] for row in measured),
        "largest_free_block_max": max(row["after"]["largest_free_block_bytes"] for row in measured),
    }


def churn_probe(policy, device, args, label, arrival_pattern):
    profiled = ProfiledPolicy(policy, device)
    total_frames = args.warmup_frames + args.measured_frames
    result = run_continuous_batching(
        profiled,
        policy_config=args.policy,
        prompt=args.prompt,
        num_denoise_steps=args.denoise_steps,
        max_decoding_steps=args.decode_steps,
        steps_per_frame=args.steps_per_frame,
        total_frames=total_frames,
        warmup_frames=args.warmup_frames,
        arrival_pattern=arrival_pattern,
    )
    active_frames = [frame for frame in result["frames"] if frame["n_total"] > 0]
    if len(active_frames) != len(profiled.samples):
        raise RuntimeError(
            f"Allocator sample mismatch: {len(profiled.samples)} samples for "
            f"{len(active_frames)} active frames"
        )
    for sample, frame in zip(profiled.samples, active_frames, strict=True):
        sample["workload_frame"] = frame["frame_idx"]
        sample["is_warmup"] = frame["is_warmup"]
    return {
        "label": label,
        "arrival_pattern": arrival_pattern,
        "summary": summarize_churn(profiled.samples),
        "allocator_samples": profiled.samples,
        "runner_frames": result["frames"],
        "completed_request_count": len(result["completed_requests"]),
    }


def main():
    args = parse_args()
    from random_init_support import pi_checkpoint
    args.checkpoint_dir = pi_checkpoint(args.checkpoint_dir, args.random_init)
    setup_jax_cache()
    policy = create_policy(args.policy, checkpoint_dir=args.checkpoint_dir)
    device = jax.devices()[0]

    payload = {
        "metadata": {
            **collect_metadata(),
            "policy": args.policy,
            "checkpoint_dir": str(args.checkpoint_dir),
            "prompt": args.prompt,
            "denoise_steps": args.denoise_steps,
            "decode_steps": args.decode_steps,
            "steps_per_frame": args.steps_per_frame,
            "warmup_frames": args.warmup_frames,
            "measured_frames": args.measured_frames,
            "allocator": "JAX default GPU allocator",
            "notes": "Allocation counts include all scheduler inputs, outputs, stacking, and model temporaries; they are an upper bound on KV-view overhead.",
        },
        "alias_probe": alias_probe(policy, device, args),
        "churn_probes": [],
    }

    payload["churn_probes"].append(churn_probe(
        policy,
        device,
        args,
        "steady_one_arrival_per_frame",
        f"uniform_arrivals(rate=1, t_max={args.decode_steps})",
    ))
    payload["churn_probes"].append(churn_probe(
        policy,
        device,
        args,
        "bursty_four_arrivals",
        f"bursty_arrivals(burst_size=4, burst_every=3, t_max={args.decode_steps})",
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
