#!/usr/bin/env python3
"""Aligned full-control-frame profile for OxyGen continuous batching."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any

import jax
import numpy as np
import torch

from experiments.baseline.runner import run_baseline
from experiments.common.setup import create_policy, setup_jax_cache
from experiments.common.workload import create_synthetic_observation
from experiments.continuous_batching.grid_search import warmup_batch_sizes
from openpi.models.kv_cache_manager import ContinuousBatchManager
import openpi.policies.policy as policy_module


PROMPT = "pick up the red bowl and place it on the plate"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="pi05_o2_libero")
    parser.add_argument("--backend", choices=("jax", "pytorch"), default="jax")
    parser.add_argument("--pytorch-device", default="cuda:0")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--decode-steps", type=int, default=30)
    parser.add_argument("--k-values", type=int, nargs="+", default=(5, 1))
    parser.add_argument("--steady-warmup", type=int, default=10)
    parser.add_argument("--measured-frames", type=int, default=20)
    parser.add_argument("--baseline-warmups", type=int, default=2)
    parser.add_argument("--baseline-repeats", type=int, default=10)
    parser.add_argument("--official-prewarm", action="store_true")
    parser.add_argument("--prompt", default=PROMPT)
    return parser.parse_args()


def block_tree(value):
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def sync(backend: str, value=None):
    if backend == "pytorch":
        torch.cuda.synchronize()
    elif value is not None:
        block_tree(value)


def mean(values):
    return statistics.fmean(values) if values else None


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


class TimedManager(ContinuousBatchManager):
    def __init__(self):
        super().__init__()
        self.reset_timing()

    def reset_timing(self):
        self.times_ns = {name: 0 for name in ("get", "store", "remove", "list")}
        self.calls = {name: 0 for name in self.times_ns}

    def _timed(self, name, fn, *args):
        start = time.perf_counter_ns()
        result = fn(*args)
        self.times_ns[name] += time.perf_counter_ns() - start
        self.calls[name] += 1
        return result

    def get_state(self, request_id):
        return self._timed("get", super().get_state, request_id)

    def store_state(self, request_id, state):
        return self._timed("store", super().store_state, request_id, state)

    def remove_state(self, request_id):
        return self._timed("remove", super().remove_state, request_id)

    def get_all_active_requests(self):
        return self._timed("list", super().get_all_active_requests)

    def snapshot(self):
        result = {f"crud_{name}_ms": value / 1e6 for name, value in self.times_ns.items()}
        result.update({f"crud_{name}_calls": value for name, value in self.calls.items()})
        return result


class FullPathProfiler:
    """Adds GPU barriers around every model/cache phase in the existing policy path."""

    def __init__(self, backend: str):
        self.backend = backend
        self.samples: dict[str, list[float]] = {}
        self.originals = []

    def reset(self):
        self.samples = {}

    def _record(self, label, original, args, kwargs):
        sync(self.backend, (args, kwargs))
        start = time.perf_counter_ns()
        result = original(*args, **kwargs)
        sync(self.backend, result)
        self.samples.setdefault(label, []).append((time.perf_counter_ns() - start) / 1e6)
        return result

    def wrap(self, owner, name, label):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            return self._record(label, original, args, kwargs)

        self.originals.append((owner, name, original))
        setattr(owner, name, wrapped)

    def wrap_split(self, owner, name):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            batch_size = args[1] if len(args) > 1 else kwargs["batch_size"]
            label = "admission_split_ms" if batch_size == 1 else "kv_split_ms"
            return self._record(label, original, args, kwargs)

        self.originals.append((owner, name, original))
        setattr(owner, name, wrapped)

    def install(self, policy):
        self.wrap(policy, "_prefill", "prefix_ms")
        init_name = "_init_static_incremental_state" if self.backend == "pytorch" else "_init_incremental_state"
        self.wrap(policy, init_name, "state_init_ms")
        self.wrap(policy, "_sample_actions_with_kv", "denoise_ms")
        self.wrap(policy, "_generate_n_tokens", "decode_ms")
        if self.backend == "pytorch":
            self.wrap(policy_module, "_stack_pytorch_incremental_states", "kv_stack_ms")
            self.wrap_split(policy_module, "_split_pytorch_incremental_state")
        else:
            self.wrap(policy_module, "_stack_incremental_states", "kv_stack_ms")
            self.wrap_split(policy_module, "_split_incremental_state")

    def uninstall(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)

    def totals(self):
        return {name: sum(values) for name, values in self.samples.items()}


def make_obs(policy_name, prompt, seed):
    return create_synthetic_observation(prompt, seed=seed, policy_config=policy_name)


def run_continuous(policy, args, k: int, *, profiled: bool):
    manager = TimedManager()
    profiler = FullPathProfiler(args.backend) if profiled else None
    if profiler:
        profiler.install(policy)
    dummy_obs = make_obs(args.policy, args.prompt, 0)
    rows = []
    lifetime = (args.decode_steps + k - 1) // k
    warmup_frames = lifetime + args.steady_warmup
    total_frames = warmup_frames + args.measured_frames

    try:
        for frame in range(total_frames):
            manager.reset_timing()
            if profiler:
                profiler.reset()
            active_ids = manager.get_all_active_requests()
            obs_list = [make_obs(args.policy, args.prompt, 1000 + frame)] + [dummy_obs] * len(active_ids)
            request_ids = [None] + active_ids
            sync(args.backend)
            start = time.perf_counter_ns()
            outputs = policy.infer_text_actions_continuous_batch(
                obs_list,
                manager,
                request_ids=request_ids,
                steps_per_frame=k,
                num_action_steps=args.denoise_steps,
                max_decoding_steps=args.decode_steps,
                PALIGEMMA_EOS_TOKEN=-1,
                generate_actions_for_resumed=False,
            )
            sync(args.backend, outputs)
            frame_ms = (time.perf_counter_ns() - start) / 1e6
            timing = outputs[0]["policy_timing"]
            phases = profiler.totals() if profiler else {}
            crud = manager.snapshot()
            row = {
                "frame": frame,
                "is_measured": frame >= warmup_frames,
                "k": k,
                "batch_size": timing["batch_size"],
                "new_requests": timing["new_requests"],
                "resumed_requests": timing["resumed_requests"],
                "active_after": len(manager.active_states),
                "frame_ms": frame_ms,
                **phases,
                **crud,
            }
            row["crud_total_ms"] = sum(v for key, v in crud.items() if key.endswith("_ms"))
            if profiled:
                row["manager_total_ms"] = (
                    row.get("kv_stack_ms", 0.0) + row.get("kv_split_ms", 0.0) + row["crud_total_ms"]
                )
                row["admission_total_ms"] = row.get("state_init_ms", 0.0) + row.get("admission_split_ms", 0.0)
                accounted = sum(
                    row.get(name, 0.0)
                    for name in (
                        "prefix_ms", "denoise_ms", "state_init_ms", "admission_split_ms",
                        "kv_stack_ms", "decode_ms", "kv_split_ms", "crud_total_ms",
                    )
                )
                row["unattributed_ms"] = frame_ms - accounted
            rows.append(row)
            if frame in (0, warmup_frames - 1, total_frames - 1):
                print(json.dumps(row), flush=True)
    finally:
        if profiler:
            profiler.uninstall()
    return rows


def summarize_rows(rows, profiled):
    measured = [row for row in rows if row["is_measured"]]
    keys = {"frame_ms", "batch_size", "active_after"}
    if profiled:
        keys.update(
            {
                "prefix_ms", "denoise_ms", "state_init_ms", "admission_split_ms",
                "kv_stack_ms", "decode_ms", "kv_split_ms", "crud_total_ms",
                "manager_total_ms", "admission_total_ms", "unattributed_ms",
            }
        )
    summary = {f"{key}_mean": mean([row.get(key, 0.0) for row in measured]) for key in sorted(keys)}
    frame_values = [row["frame_ms"] for row in measured]
    summary["frame_ms_p50"] = percentile(frame_values, 50)
    summary["frame_ms_p95"] = percentile(frame_values, 95)
    summary["measured_frames"] = len(measured)
    if profiled:
        summary["manager_fraction_of_full_frame"] = summary["manager_total_ms_mean"] / summary["frame_ms_mean"]
        summary["stack_fraction_of_full_frame"] = summary["kv_stack_ms_mean"] / summary["frame_ms_mean"]
        summary["split_fraction_of_full_frame"] = summary["kv_split_ms_mean"] / summary["frame_ms_mean"]
    return summary


def run_baseline_series(policy, args):
    obs = make_obs(args.policy, args.prompt, 4242)
    rows = []
    total = args.baseline_warmups + args.baseline_repeats
    for repeat in range(total):
        sync(args.backend)
        result = run_baseline(
            policy,
            obs,
            num_denoise_steps=args.denoise_steps,
            max_decoding_steps=args.decode_steps,
        )
        sync(args.backend, result)
        rows.append(
            {
                "repeat": repeat,
                "is_measured": repeat >= args.baseline_warmups,
                "frame_ms": result["frame_ms"],
                "policy_timing": result["policy_timing"],
            }
        )
    measured = [row["frame_ms"] for row in rows if row["is_measured"]]
    return rows, {
        "frame_ms_mean": mean(measured),
        "frame_ms_p50": percentile(measured, 50),
        "frame_ms_p95": percentile(measured, 95),
        "measured_repeats": len(measured),
    }


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    args = parse_args()
    if args.random_init:
        torch.manual_seed(0)
    from random_init_support import pi_checkpoint
    args.checkpoint_dir = pi_checkpoint(args.checkpoint_dir, args.random_init)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    setup_jax_cache()
    kwargs = {"pytorch_device": args.pytorch_device} if args.backend == "pytorch" else {}
    policy = create_policy(args.policy, checkpoint_dir=args.checkpoint_dir, **kwargs)
    actual_backend = "pytorch" if policy._is_pytorch_model else "jax"
    if actual_backend != args.backend:
        raise RuntimeError(f"Requested {args.backend}, loaded {actual_backend}")

    metadata = {
        "hostname": platform.node(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).splitlines()[0],
        "jax_version": jax.__version__,
        "torch_version": torch.__version__,
        "backend": args.backend,
        "checkpoint_dir": str(args.checkpoint_dir),
        "policy": args.policy,
        "prompt": args.prompt,
        "denoise_steps": args.denoise_steps,
        "decode_steps": args.decode_steps,
        "eos_token": -1,
        "git_commit": git_commit(),
        "jax_disable_jit": bool(jax.config.jax_disable_jit),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "jax_allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default pooled allocator"),
        "dtype_note": "checkpoint/model defaults; no dtype override",
        "profile_semantics": "GPU synchronization before and after each wrapped full-path phase",
        "natural_semantics": "GPU synchronization only at the existing output/completion boundary",
        "args": {**vars(args), "checkpoint_dir": str(args.checkpoint_dir), "output": str(args.output)},
    }

    baseline_rows, baseline_summary = run_baseline_series(policy, args)
    points = {}
    for k in args.k_values:
        if args.official_prewarm:
            max_batch = (args.decode_steps + k - 1) // k + 1
            print(f"Official shape prewarm N={args.decode_steps}, k={k}, batch=1..{max_batch}", flush=True)
            warmup_batch_sizes(
                policy,
                policy_config=args.policy,
                prompt=args.prompt,
                max_batch=max_batch,
                num_denoise_steps=args.denoise_steps,
                max_decoding_steps=args.decode_steps,
                steps_per_frame=k,
            )
        print(f"Running natural OxyGen N={args.decode_steps}, k={k}", flush=True)
        natural = run_continuous(policy, args, k, profiled=False)
        print(f"Running synchronized full-path profile N={args.decode_steps}, k={k}", flush=True)
        profiled = run_continuous(policy, args, k, profiled=True)
        natural_summary = summarize_rows(natural, profiled=False)
        profile_summary = summarize_rows(profiled, profiled=True)
        natural_summary["speedup_vs_baseline"] = baseline_summary["frame_ms_mean"] / natural_summary["frame_ms_mean"]
        points[str(k)] = {
            "natural_rows": natural,
            "profiled_rows": profiled,
            "natural_summary": natural_summary,
            "profiled_summary": profile_summary,
        }

    payload = {
        "metadata": metadata,
        "baseline": {"rows": baseline_rows, "summary": baseline_summary},
        "points": points,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"metadata": metadata, "baseline": baseline_summary,
                      "points": {k: {"natural": v["natural_summary"], "profiled": v["profiled_summary"]}
                                 for k, v in points.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
