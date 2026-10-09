#!/usr/bin/env python3
"""Benchmark real heterogeneous StarVLA experts with language continuous batching."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from oxygen_models.common.cache import (  # noqa: E402
    init_language_from_prefix,
    staticize_language_state,
)
from oxygen_models.starvla.batching import batched_language_steps  # noqa: E402
from oxygen_models.starvla.experts import (  # noqa: E402
    RuntimeAdapter,
    build_inputs,
    example,
    prefill,
)
from oxygen_models.starvla.runtime import persistent_continuous_once, run_head
from starVLA.model.framework.base_framework import baseframework
from oxygen_models.starvla.loading import from_config_only  # noqa: E402


def int_list(value):
    return [int(item) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--pi-checkpoint", required=True)
    parser.add_argument("--groot-checkpoint", required=True)
    parser.add_argument("--oft-checkpoint", required=True)
    parser.add_argument("--random-init", action=argparse.BooleanOptionalAction, default=True, help="Use model configs and metadata without trained weights.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expert-counts", type=int_list, default=[2, 3, 4])
    parser.add_argument("--language-tokens", type=int_list, default=[30])
    parser.add_argument("--denoise-steps", type=int_list, default=[4])
    parser.add_argument("--steps-per-frame", type=int_list, default=[5])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--single-measured-frames", type=int, default=10)
    parser.add_argument("--single-adaptive-frames", type=int, default=30)
    parser.add_argument("--single-cv-threshold", type=float, default=0.02)
    parser.add_argument("--single-trend-threshold", type=float, default=0.02)
    parser.add_argument("--measured-frames", type=int, default=50)
    parser.add_argument("--continuous-warmup-frames", type=int, default=5)
    parser.add_argument("--persistent-continuous", action="store_true")
    parser.add_argument(
        "--continuous-runtime",
        choices=["legacy", "persistent", "both"],
        default="persistent",
        help="Continuous-batching runtime to benchmark. --persistent-continuous is kept as a compatibility alias for both.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = fn()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - start) * 1000.0


def load_models(args):
    overrides = [
        f"framework.qwenvl.base_vlm={args.base_vlm}",
        "framework.qwenvl.attn_implementation=sdpa",
    ]
    loader = from_config_only if args.random_init else baseframework.from_pretrained
    pi = loader(args.pi_checkpoint, config_overrides=overrides)
    pi = pi.to(torch.bfloat16).to(args.device).eval()
    groot = None
    if max(args.expert_counts) >= 3:
        groot_model = loader(args.groot_checkpoint, config_overrides=overrides)
        groot = groot_model.action_model.to(torch.bfloat16).to(args.device).eval()
        del groot_model
        gc.collect()
    oft_model = None
    # OFT supplies the common action-query token layout.
    oft_model = loader(args.oft_checkpoint, config_overrides=overrides)
    oft_model.action_model = oft_model.action_model.to(torch.bfloat16).to(args.device).eval()
    del oft_model.qwen_vl_interface
    gc.collect()
    return pi, groot, oft_model


def expert_names(count):
    return ["language", "pi_v3", "groot", "oft"][:count]




def execute_once(pi, groot, oft, inputs, names, language_tokens, denoise_steps, shared):
    components = {}
    outputs = {}
    total_start = time.perf_counter()
    if shared:
        common, components["prefill_ms"] = timed(lambda: prefill(pi, inputs))
        prefixes = {name: common for name in names}
    else:
        prefixes = {}
        prefill_times = []
        for name in names:
            prefixes[name], elapsed = timed(lambda: prefill(pi, inputs))
            prefill_times.append(elapsed)
        components["prefill_ms"] = sum(prefill_times)
        components["prefill_each_ms"] = prefill_times
    for name in names:
        outputs[name], components[f"{name}_ms"] = timed(
            lambda name=name: run_head(
                name, pi, groot, oft, prefixes[name], inputs,
                language_tokens, denoise_steps,
            )
        )
    torch.cuda.synchronize()
    components["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    return outputs, components


def frame_stability(frame_ms):
    mean_ms = statistics.fmean(frame_ms)
    cv = statistics.stdev(frame_ms) / mean_ms if len(frame_ms) > 1 else 0.0
    count = len(frame_ms)
    x_mean = (count - 1) / 2
    denominator = sum((index - x_mean) ** 2 for index in range(count))
    slope = (
        sum((index - x_mean) * (value - mean_ms) for index, value in enumerate(frame_ms))
        / denominator
        if denominator
        else 0.0
    )
    total_trend_fraction = abs(slope) * max(count - 1, 0) / mean_ms
    return {
        "mean_ms": mean_ms,
        "cv": cv,
        "slope_ms_per_frame": slope,
        "total_trend_fraction": total_trend_fraction,
    }


def measure_single_repeats(
    pi, groot, oft, inputs, names, language_tokens, denoise_steps,
    repeats, warmup, measured_frames,
):
    records = []
    last_outputs = {}
    for repeat in range(repeats):
        order = [False, True] if repeat % 2 == 0 else [True, False]
        for shared in order:
            mode = "shared" if shared else "separate"
            for _ in range(warmup):
                execute_once(
                    pi, groot, oft, inputs, names, language_tokens,
                    denoise_steps, shared,
                )
            frames = []
            for frame in range(measured_frames):
                outputs, components = execute_once(
                    pi, groot, oft, inputs, names, language_tokens,
                    denoise_steps, shared,
                )
                last_outputs[mode] = outputs
                frames.append({
                    "frame": frame,
                    **components,
                })
            stability = frame_stability([frame["total_ms"] for frame in frames])
            records.append({
                "expert_count": len(names),
                "experts": names,
                "language_tokens": language_tokens,
                "denoise_steps": denoise_steps,
                "mode": mode,
                "repeat": repeat,
                "measured_frames": measured_frames,
                "frames": frames,
                **stability,
            })
    return records, last_outputs


def output_checks(outputs, names, processor):
    checks = {}
    for mode, mode_outputs in outputs.items():
        checks[mode] = {}
        for name in names:
            value = mode_outputs[name]
            if name == "language":
                token_ids = list(value.generated_ids)
                checks[mode][name] = {
                    "token_count": len(token_ids),
                    "token_ids": token_ids,
                    "text": processor.decode(token_ids, skip_special_tokens=True),
                }
            else:
                checks[mode][name] = {
                    "shape": list(value.shape),
                    "finite": bool(torch.isfinite(value).all()),
                    "checksum": float(value.float().sum()),
                }
    comparisons = {}
    if "separate" in outputs and "shared" in outputs:
        for name in names:
            if name == "language":
                comparisons[name] = {
                    "token_ids_equal": (
                        outputs["separate"][name].generated_ids
                        == outputs["shared"][name].generated_ids
                    )
                }
            else:
                comparisons[name] = {
                    "max_abs": float(
                        (outputs["separate"][name].float() - outputs["shared"][name].float())
                        .abs().max()
                    )
                }
    return {"by_mode": checks, "separate_vs_shared": comparisons}


def continuous_once(
    pi, groot, oft, inputs, names, language_tokens, denoise_steps,
    steps_per_frame, measured_frames,
):
    adapter = RuntimeAdapter(pi.qwen_vl_interface)
    processor = pi.qwen_vl_interface.processor
    active = {}
    completed = []
    frame_times = []
    frame_batches = []
    frame_tokens = []
    warmup_frames = math.ceil(language_tokens / steps_per_frame)
    for frame in range(warmup_frames + measured_frames):
        torch.cuda.synchronize()
        start = time.perf_counter()
        prefix = prefill(pi, inputs)
        for name in names:
            if name != "language":
                run_head(name, pi, groot, oft, prefix, inputs, language_tokens, denoise_steps)
        state = init_language_from_prefix(processor, prefix, stop_on_eos=False)
        staticize_language_state(state, language_tokens)
        active[f"request_{frame}"] = state
        request_ids = list(active)
        states = [active[request_id] for request_id in request_ids]
        before = sum(len(item.generated_ids) for item in states)
        batched_language_steps(adapter, states, steps_per_frame)
        generated = sum(len(item.generated_ids) for item in states) - before
        for request_id in request_ids:
            if active[request_id].finished:
                completed.append(active.pop(request_id))
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        if frame >= warmup_frames:
            frame_times.append(elapsed)
            frame_batches.append(len(states))
            frame_tokens.append(generated)
    texts = [processor.batch_decode([state.generated_ids], skip_special_tokens=True)[0] for state in completed]
    return {
        "mean_frame_ms": statistics.fmean(frame_times),
        "p50_frame_ms": statistics.median(frame_times),
        "avg_batch_size": statistics.fmean(frame_batches),
        "language_tokens_per_s": sum(frame_tokens) / (sum(frame_times) / 1000.0),
        "completed_requests": len(completed),
        "all_completed_lengths_correct": all(len(state.generated_ids) == language_tokens for state in completed),
        "all_completed_text_nonempty": bool(texts) and all(bool(text.strip()) for text in texts),
        "completed_text_samples": texts[-min(5, len(texts)):],
    }










def release_cuda_temporaries():
    """Release per-run caches outside the measured region and report live memory."""
    gc.collect()
    allocated_before_empty = torch.cuda.memory_allocated()
    reserved_before_empty = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    return {
        "allocated_before_empty_bytes": allocated_before_empty,
        "reserved_before_empty_bytes": reserved_before_empty,
        "allocated_after_empty_bytes": torch.cuda.memory_allocated(),
        "reserved_after_empty_bytes": torch.cuda.memory_reserved(),
    }


def summarize(records, keys):
    return {key: statistics.median(row[key] for row in records) for key in keys}


def summarize_single_mode(records, mode, names):
    selected = [row for row in records if row["mode"] == mode]
    component_keys = ["prefill_ms"] + [f"{name}_ms" for name in names] + ["total_ms"]
    repeat_component_means = {
        key: [statistics.fmean(frame[key] for frame in row["frames"]) for row in selected]
        for key in component_keys
    }
    return {
        "latency_ms": statistics.median(repeat_component_means["total_ms"]),
        "component_ms": {
            key: statistics.median(values)
            for key, values in repeat_component_means.items()
        },
        "repeat_means_ms": repeat_component_means["total_ms"],
        "repeat_cv": [row["cv"] for row in selected],
        "repeat_total_trend_fraction": [row["total_trend_fraction"] for row in selected],
    }


def git_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_metadata(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main():
    args = parse_args()
    if args.random_init:
        torch.manual_seed(0)
    continuous_runtime = "both" if args.persistent_continuous else args.continuous_runtime
    if continuous_runtime != "persistent":
        raise ValueError("This example requires --continuous-runtime persistent")
    pi, groot, oft = load_models(args)
    image, instruction = example()
    # Use one identical OFT-compatible common prefix for every expert count.
    # This isolates expert-count scaling; it is not the ordinary portability or
    # action-quality workload.
    inputs = build_inputs(pi.qwen_vl_interface, instruction, image, oft)
    records = []
    summary = []
    pilot_records = []
    initial_validation = {}

    for count in args.expert_counts:
        names = expert_names(count)
        separate_outputs, _ = execute_once(
            pi, groot, oft, inputs, names, args.language_tokens[0],
            args.denoise_steps[0], False,
        )
        shared_outputs, _ = execute_once(
            pi, groot, oft, inputs, names, args.language_tokens[0],
            args.denoise_steps[0], True,
        )
        initial_validation[str(count)] = output_checks(
            {"separate": separate_outputs, "shared": shared_outputs},
            names, pi.qwen_vl_interface.processor,
        )

    preflight = persistent_continuous_once(
        pi, groot, oft, inputs, expert_names(max(args.expert_counts)),
        args.language_tokens[0], args.denoise_steps[0], 1, 1, 2,
    )

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "starvla_commit": git_commit(Path(__file__).resolve().parents[1]),
            "base_vlm": args.base_vlm,
            "pi_checkpoint": checkpoint_metadata(args.pi_checkpoint),
            "groot_checkpoint": checkpoint_metadata(args.groot_checkpoint),
            "oft_checkpoint": checkpoint_metadata(args.oft_checkpoint),
            "scheduler_semantics": "one shared prefill per frame; all selected action experts run; one new language request enters the persistent continuous batch",
            "expert_count_includes_language": True,
            "expert_semantics": {
                "language": "autoregressive task-conditioned language generation",
                "pi_v3": "layer-wise flow-matching action policy",
                "groot": "last-layer flow-matching action policy",
                "oft": "one-shot action policy",
            },
            "workload_scope": "systems compatibility and performance only; no language or action quality claim",
            "prefix_mode": "fixed_oft_compatible_for_all_expert_counts",
            "common_prefix_contents": "same image, task, and OFT action-query tokens for E=2/3/4",
            "prefix_tokens": int(inputs["input_ids"].shape[-1]),
            "attention": "pytorch_sdpa",
            "dtype": "bfloat16",
            "compile": False,
            "cuda_graph": False,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(torch.device(args.device)),
            "sdpa_backend_flags": {
                "flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math_enabled": torch.backends.cuda.math_sdp_enabled(),
            },
            "denoise_steps": args.denoise_steps,
            "single_measurement": {
                "warmup_frames_per_mode_repeat": args.warmup,
                "initial_measured_frames": args.single_measured_frames,
                "adaptive_measured_frames": args.single_adaptive_frames,
                "cv_threshold": args.single_cv_threshold,
                "trend_definition": "absolute fitted first-to-last drift divided by mean",
                "trend_threshold": args.single_trend_threshold,
                "aggregation": "median of three per-repeat frame means",
            },
            "continuous_measurement": {
                "scheduler_ramp_up": "prepopulated staggered request ages plus one decode step",
                "unmeasured_warmup_frames_per_repeat": args.continuous_warmup_frames,
                "measured_frames_per_repeat": args.measured_frames,
                "aggregation": "median of three per-repeat frame means",
            },
            "repeats": args.repeats,
            "continuous_runtime": continuous_runtime,
        },
        "initial_validation": initial_validation,
        "persistent_scheduler_preflight": preflight,
        "pilot_records": pilot_records,
        "records": records,
        "summary": summary,
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    save()
    for count in args.expert_counts:
        names = expert_names(count)
        for language_tokens in args.language_tokens:
            for denoise_steps in args.denoise_steps:
                point_records, last_outputs = measure_single_repeats(
                    pi, groot, oft, inputs, names, language_tokens,
                    denoise_steps, args.repeats, args.warmup,
                    args.single_measured_frames,
                )
                unstable = any(
                    row["cv"] > args.single_cv_threshold
                    or row["total_trend_fraction"] > args.single_trend_threshold
                    for row in point_records
                )
                if unstable:
                    pilot_records.extend(point_records)
                    point_records, last_outputs = measure_single_repeats(
                        pi, groot, oft, inputs, names, language_tokens,
                        denoise_steps, args.repeats, args.warmup,
                        args.single_adaptive_frames,
                    )
                records.extend(point_records)
                separate = summarize_single_mode(point_records, "separate", names)
                shared = summarize_single_mode(point_records, "shared", names)
                item = {
                    "expert_count": count,
                    "experts": names,
                    "language_tokens": language_tokens,
                    "denoise_steps": denoise_steps,
                    "single_adaptive_rerun": unstable,
                    "single_measured_frames_per_repeat": (
                        args.single_adaptive_frames if unstable
                        else args.single_measured_frames
                    ),
                    "separate": separate,
                    "shared": shared,
                    "shared_kv_speedup": separate["latency_ms"] / shared["latency_ms"],
                    "single_correctness": output_checks(
                        last_outputs, names, pi.qwen_vl_interface.processor
                    ),
                    "persistent_continuous": [],
                }
                for steps_per_frame in args.steps_per_frame:
                    if language_tokens % steps_per_frame:
                        continue
                    persistent_rows = []
                    persistent_memory = []
                    for repeat in range(args.repeats):
                        row = persistent_continuous_once(
                            pi, groot, oft, inputs, names, language_tokens,
                            denoise_steps, steps_per_frame,
                            args.continuous_warmup_frames, args.measured_frames,
                        )
                        row["repeat"] = repeat
                        persistent_rows.append(row)
                        persistent_memory.append(release_cuda_temporaries())
                    oxy_mean_ms = statistics.median(
                        row["mean_frame_ms"] for row in persistent_rows
                    )
                    item["persistent_continuous"].append({
                        "steps_per_frame": steps_per_frame,
                        "mean_frame_ms": oxy_mean_ms,
                        "speedup_vs_separate": separate["latency_ms"] / oxy_mean_ms,
                        "speedup_vs_shared": shared["latency_ms"] / oxy_mean_ms,
                        "avg_batch_size": statistics.median(
                            row["avg_batch_size"] for row in persistent_rows
                        ),
                        "language_tokens_per_s": statistics.median(
                            row["language_tokens_per_s"] for row in persistent_rows
                        ),
                        "completed_requests": [
                            row["completed_requests"] for row in persistent_rows
                        ],
                        "all_completed_lengths_correct": all(
                            row["all_completed_lengths_correct"]
                            for row in persistent_rows
                        ),
                        "all_completed_text_nonempty": all(
                            row["all_completed_text_nonempty"]
                            for row in persistent_rows
                        ),
                        "all_actions_finite": all(
                            row["all_actions_finite"] for row in persistent_rows
                        ),
                        "completed_text_samples": persistent_rows[-1]["completed_text_samples"],
                        "memory_after_repeats": persistent_memory,
                        "repeat_rows": persistent_rows,
                    })
                summary.append(item)
                print(json.dumps(item, ensure_ascii=False), flush=True)
                save()


if __name__ == "__main__":
    main()
