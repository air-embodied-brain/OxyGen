#!/usr/bin/env python3
"""Long-run full-model stress test for OxyGen's continuous KV manager."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import jax
import numpy as np
import torch

from experiments.common.setup import collect_metadata, create_policy, setup_jax_cache
from experiments.common.workload import create_synthetic_observation
from openpi.models.kv_cache_manager import ContinuousBatchManager
import openpi.policies.policy as policy_module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="pi05_o2_libero")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--pytorch-device")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scenario", choices=("steady", "fluctuating", "backlog", "fixed_large"), required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--decode-steps", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--arrivals-per-frame", type=int, default=1)
    parser.add_argument(
        "--arrival-pattern",
        default="",
        help="Optional comma-separated arrival counts, repeated cyclically for steady/backlog scenarios",
    )
    parser.add_argument(
        "--arrival-scale",
        type=int,
        default=1,
        help="Integer multiplier for the fluctuating scenario's deterministic burst counts",
    )
    parser.add_argument("--max-active", type=int, default=0, help="0 means no admission limit")
    parser.add_argument("--sample-memory-every", type=int, default=10)
    parser.add_argument("--phase-sync", action="store_true")
    parser.add_argument("--prompt", default="pick up the red bowl and place it on the plate")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def jax_memory(device):
    stats = device.memory_stats() or {}
    return {
        key: int(stats.get(key, 0))
        for key in (
            "num_allocs", "bytes_in_use", "peak_bytes_in_use", "bytes_limit",
            "bytes_reserved", "peak_bytes_reserved", "largest_free_block_bytes",
            "pool_bytes", "peak_pool_bytes",
        )
    }


def torch_memory(device):
    stats = torch.cuda.memory_stats(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "active_bytes": int(stats.get("active_bytes.all.current", 0)),
        "inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
        "allocation_count": int(stats.get("allocation.all.current", 0)),
        "segment_count": int(stats.get("segment.all.current", 0)),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "num_ooms": int(stats.get("num_ooms", 0)),
    }


def block_tree(value):
    for leaf in jax.tree.leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def logical_state_bytes(state, backend):
    """Logical tensor bytes retained by one request; does not infer allocator slabs."""
    if backend == "jax":
        return sum(int(leaf.size * leaf.dtype.itemsize) for leaf in jax.tree.leaves(state) if hasattr(leaf, "size"))
    seen_objects, seen_tensors = set(), set()

    def visit(value):
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return 0
        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if isinstance(value, torch.Tensor):
            pointer = (value.device.type, value.data_ptr())
            if pointer in seen_tensors:
                return 0
            seen_tensors.add(pointer)
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        if dataclasses.is_dataclass(value):
            return sum(visit(getattr(value, field.name)) for field in dataclasses.fields(value))
        if hasattr(value, "__dict__"):
            return sum(visit(item) for item in vars(value).values())
        return 0

    return visit(state)


class PhaseProfiler:
    def __init__(self, backend: str, device: Any, sync: bool):
        self.backend = backend
        self.device = device
        self.sync = sync
        self.samples: dict[str, list[float]] = {}
        self._originals = []

    def reset(self):
        self.samples = {}

    def _synchronize(self, value=None):
        if not self.sync:
            return
        if self.backend == "pytorch":
            torch.cuda.synchronize(self.device)
        elif value is not None:
            block_tree(value)

    def wrap_callable(self, owner, name, label):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            # For JAX, blocking the actual input leaves prevents asynchronous work
            # dispatched by the preceding phase from leaking into this timer.
            self._synchronize((args, kwargs))
            start = time.perf_counter_ns()
            result = original(*args, **kwargs)
            self._synchronize(result)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            self.samples.setdefault(label, []).append(elapsed)
            return result

        self._originals.append((owner, name, original))
        setattr(owner, name, wrapped)

    def install(self, policy):
        if self.backend == "pytorch":
            self.wrap_callable(policy_module, "_stack_pytorch_incremental_states", "kv_stack_ms")
            self.wrap_callable(policy_module, "_split_pytorch_incremental_state", "kv_split_ms")
            self.wrap_callable(policy, "_init_static_incremental_state", "request_state_init_ms")
        else:
            self.wrap_callable(policy_module, "_stack_incremental_states", "kv_stack_ms")
            self.wrap_callable(policy_module, "_split_incremental_state", "kv_split_ms")
            self.wrap_callable(policy, "_init_incremental_state", "request_state_init_ms")

    def uninstall(self):
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)

    def totals(self):
        return {name: sum(values) for name, values in self.samples.items()}


class TimedManager(ContinuousBatchManager):
    def __init__(self):
        super().__init__()
        self.reset_timing()

    def reset_timing(self):
        self.times_ns = {"get": 0, "store": 0, "remove": 0, "list": 0}
        self.calls = {"get": 0, "store": 0, "remove": 0, "list": 0}

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

    def snapshot_ms(self):
        return {f"manager_{name}_ms": value / 1e6 for name, value in self.times_ns.items()} | {
            f"manager_{name}_calls": value for name, value in self.calls.items()
        }


def arrivals_for_frame(args, frame):
    if args.scenario == "fixed_large":
        return [(args.decode_steps, i) for i in range(args.arrivals_per_frame)] if frame == 0 else []
    if args.scenario in ("steady", "backlog"):
        if args.arrival_pattern:
            pattern = tuple(int(value) for value in args.arrival_pattern.split(","))
            count = pattern[frame % len(pattern)]
        else:
            count = args.arrivals_per_frame
        return [(args.decode_steps, frame * 1000 + i) for i in range(count)]
    # An intentionally non-periodic-looking deterministic trace: idle gaps, small and large bursts,
    # and heterogeneous lifetimes. Physical cache capacity remains args.decode_steps.
    burst_cycle = (0, 1, 4, 0, 2, 0, 0, 3, 1, 0, 5, 0, 1, 0, 2, 0, 0)
    lifetimes = (5, 10, 30, min(100, args.decode_steps), 10, 5, min(100, args.decode_steps))
    count = burst_cycle[frame % len(burst_cycle)] * args.arrival_scale
    return [
        (min(args.decode_steps, lifetimes[(frame * 5 + i * 3) % len(lifetimes)]), frame * 1000 + i)
        for i in range(count)
    ]


def summarize(rows, completions, backend, oom):
    measured = [r for r in rows if not r["is_warmup"]]
    lat = [r["frame_ms"] for r in measured if r["batch_size"]]
    manager = [r["manager_total_ms"] for r in measured if r["batch_size"]]
    batches = [r["batch_size"] for r in measured]
    active = [r["active_after"] for r in measured]
    request_ms = [r["wall_latency_ms"] for r in completions if not r["warmup"]]
    request_frames = [r["frame_latency"] for r in completions if not r["warmup"]]
    window = max(1, min(100, len(lat) // 5))
    result = {
        "backend": backend,
        "oom": oom,
        "measured_frames": len(measured),
        "completed_requests": len(request_ms),
        "frame_ms_mean": statistics.fmean(lat) if lat else None,
        "frame_ms_p50": percentile(lat, 50),
        "frame_ms_p95": percentile(lat, 95),
        "frame_ms_p99": percentile(lat, 99),
        "frame_ms_max": max(lat) if lat else None,
        "frame_ms_first_window": statistics.fmean(lat[:window]) if lat else None,
        "frame_ms_last_window": statistics.fmean(lat[-window:]) if lat else None,
        "manager_ms_mean": statistics.fmean(manager) if manager else None,
        "manager_ms_p99": percentile(manager, 99),
        "manager_fraction_mean": statistics.fmean(
            r["manager_total_ms"] / r["frame_ms"] for r in measured if r["frame_ms"] > 0
        ) if lat else None,
        "batch_mean": statistics.fmean(batches) if batches else None,
        "batch_max": max(batches) if batches else None,
        "active_mean": statistics.fmean(active) if active else None,
        "active_max": max(active) if active else None,
        "request_wall_ms_p50": percentile(request_ms, 50),
        "request_wall_ms_p95": percentile(request_ms, 95),
        "request_wall_ms_p99": percentile(request_ms, 99),
        "request_frame_latency_mean": statistics.fmean(request_frames) if request_frames else None,
    }
    memory_rows = [r for r in measured if r.get("memory")]
    if memory_rows:
        if backend == "pytorch":
            for key in ("allocated_bytes", "reserved_bytes", "inactive_split_bytes", "num_alloc_retries", "num_ooms"):
                values = [r["memory"][key] for r in memory_rows]
                result[f"memory_{key}_first"] = values[0]
                result[f"memory_{key}_last"] = values[-1]
                result[f"memory_{key}_min"] = min(values)
                result[f"memory_{key}_max"] = max(values)
        else:
            for key in ("bytes_in_use", "pool_bytes", "bytes_reserved", "num_allocs"):
                values = [r["memory"][key] for r in memory_rows]
                result[f"memory_{key}_first"] = values[0]
                result[f"memory_{key}_last"] = values[-1]
                result[f"memory_{key}_min"] = min(values)
                result[f"memory_{key}_max"] = max(values)
    return result


def main():
    args = parse_args()
    if args.random_init:
        torch.manual_seed(0)
    from random_init_support import pi_checkpoint
    args.checkpoint_dir = pi_checkpoint(args.checkpoint_dir, args.random_init)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "frames.jsonl"
    completion_path = args.output_dir / "requests.jsonl"
    summary_path = args.output_dir / "summary.json"
    setup_jax_cache()
    kwargs = {}
    if args.pytorch_device:
        kwargs["pytorch_device"] = args.pytorch_device
    policy = create_policy(args.policy, checkpoint_dir=args.checkpoint_dir, **kwargs)
    backend = "pytorch" if policy._is_pytorch_model else "jax"
    device = torch.device(args.pytorch_device) if backend == "pytorch" else jax.devices()[0]
    memory_fn = (lambda: torch_memory(device)) if backend == "pytorch" else (lambda: jax_memory(device))
    if backend == "pytorch":
        torch.cuda.reset_peak_memory_stats(device)

    manager = TimedManager()
    profiler = PhaseProfiler(backend, device, args.phase_sync)
    profiler.install(policy)
    dummy_obs = create_synthetic_observation(args.prompt, seed=0, policy_config=args.policy)
    request_meta = {}
    rows, completions = [], []
    retained_state_bytes = None
    start_run = time.monotonic()
    oom = None

    with raw_path.open("w", buffering=1) as raw_file, completion_path.open("w", buffering=1) as req_file:
        try:
            for frame in range(args.frames):
                active_ids = manager.get_all_active_requests()
                arrivals = arrivals_for_frame(args, frame)
                if args.max_active:
                    arrivals = arrivals[:max(0, args.max_active - len(active_ids))]
                obs_list, request_ids, pending_lifetimes = [], [], []
                for lifetime, seed in arrivals:
                    obs_list.append(create_synthetic_observation(args.prompt, seed=seed, policy_config=args.policy))
                    request_ids.append(None)
                    pending_lifetimes.append(lifetime)
                obs_list.extend([dummy_obs] * len(active_ids))
                request_ids.extend(active_ids)

                manager.reset_timing()
                profiler.reset()
                frame_start = time.monotonic()
                results = []
                if obs_list:
                    results = policy.infer_text_actions_continuous_batch(
                        obs_list, manager, request_ids=request_ids,
                        steps_per_frame=args.steps_per_frame,
                        num_action_steps=args.denoise_steps,
                        max_decoding_steps=args.decode_steps,
                        PALIGEMMA_EOS_TOKEN=-1,
                        generate_actions_for_resumed=False,
                    )
                frame_ms = (time.monotonic() - frame_start) * 1000

                for index, result in enumerate(results[:len(arrivals)]):
                    request_meta[result["request_id"]] = {
                        "arrival_frame": frame,
                        "arrival_time": frame_start,
                        "lifetime": pending_lifetimes[index],
                        "warmup": frame < args.warmup_frames,
                    }
                if retained_state_bytes is None and manager.active_states:
                    retained_state_bytes = logical_state_bytes(next(iter(manager.active_states.values())), backend)
                forced_remove = 0
                for result in results:
                    rid = result["request_id"]
                    meta = request_meta[rid]
                    tokens = len(result.get("tokens_full", []))
                    done = result["is_finished"] or tokens >= meta["lifetime"]
                    if done:
                        if not result["is_finished"]:
                            manager.remove_state(rid)
                            forced_remove += 1
                        item = {
                            "request_id": rid,
                            "arrival_frame": meta["arrival_frame"],
                            "finish_frame": frame,
                            "frame_latency": frame - meta["arrival_frame"] + 1,
                            "wall_latency_ms": (time.monotonic() - meta["arrival_time"]) * 1000,
                            "tokens": tokens,
                            "warmup": meta["warmup"],
                        }
                        completions.append(item)
                        req_file.write(json.dumps(item) + "\n")
                        request_meta.pop(rid, None)

                phase = profiler.totals()
                manager_times = manager.snapshot_ms()
                manager_total = sum(v for k, v in manager_times.items() if k.endswith("_ms"))
                manager_total += phase.get("kv_stack_ms", 0) + phase.get("kv_split_ms", 0)
                memory = memory_fn() if frame % args.sample_memory_every == 0 or frame + 1 == args.frames else None
                timing = results[0].get("policy_timing", {}) if results else {}
                row = {
                    "frame": frame,
                    "elapsed_run_s": time.monotonic() - start_run,
                    "is_warmup": frame < args.warmup_frames,
                    "n_new": len(arrivals),
                    "batch_size": len(obs_list),
                    "active_after": len(manager.active_states),
                    "forced_remove": forced_remove,
                    "frame_ms": frame_ms,
                    "policy_timing": timing,
                    **manager_times,
                    **phase,
                    "manager_total_ms": manager_total,
                    "logical_retained_bytes_per_request": retained_state_bytes,
                    "memory": memory,
                }
                rows.append(row)
                raw_file.write(json.dumps(row) + "\n")
                if frame % 100 == 0:
                    print(json.dumps({k: row[k] for k in ("frame", "batch_size", "active_after", "frame_ms", "manager_total_ms", "memory")}))
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            if "out of memory" not in message.lower():
                raise
            oom = {"frame": len(rows), "message": message, "memory": memory_fn()}
            raw_file.write(json.dumps({"event": "oom", **oom}) + "\n")
        finally:
            profiler.uninstall()

    metadata = {
        **collect_metadata(),
        "backend": backend,
        "args": vars(args) | {"checkpoint_dir": str(args.checkpoint_dir), "output_dir": str(args.output_dir)},
        "phase_timing_semantics": (
            "GPU-complete component times; synchronization is intrusive" if args.phase_sync
            else "CPU dispatch/host manipulation times; GPU work is charged at the policy synchronization point"
        ),
        "allocator_env": {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
    }
    summary = summarize(rows, completions, backend, oom)
    summary["logical_retained_bytes_per_request"] = retained_state_bytes
    payload = {"metadata": metadata, "summary": summary}
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
