#!/usr/bin/env python3
"""Qwen3.5 + random PI_v3 end-to-end benchmark with hybrid-cache batching."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_modeling

from oxygen_models.qwen35 import helpers as q25
from oxygen_models.qwen35.runtime import Qwen35NativeAdapter, Qwen35StaticHybridCache, build_age_snapshots, mixed_age_cache_audit, persistent_decode_steps  # noqa: E402








def int_list(value):
    return [int(item) for item in value.split(",")]


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def callable_source(value):
    if value is None:
        return None
    return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__name__', type(value).__name__)}"


def compiler_counters():
    dynamo = {
        group: {name: int(value) for name, value in counters.items()}
        for group, counters in torch._dynamo.utils.counters.items()
        if counters
    }
    try:
        from torch._inductor import metrics as inductor_metrics

        generated_kernels = int(inductor_metrics.generated_kernel_count)
    except Exception:
        generated_kernels = None
    return {
        "dynamo": dynamo,
        "inductor_generated_kernel_count": generated_kernels,
    }


def linear_attention_placement(model):
    layers = model.model.language_model.layers
    layer_types = list(model.config.text_config.layer_types)
    linear_indices = [index for index, kind in enumerate(layer_types) if kind == "linear_attention"]
    if not linear_indices:
        return {"linear_layer_count": 0, "sample_layer": None, "projections": {}}
    sample_index = linear_indices[0]
    module = layers[sample_index].linear_attn
    projections = {}
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        weight = getattr(module, name).weight
        placements = getattr(weight, "placements", None)
        projections[name] = {
            "parameter_type": type(weight).__name__,
            "is_dtensor": type(weight).__name__ == "DTensor",
            "placements": [str(value) for value in placements] if placements is not None else None,
            "shape": list(weight.shape),
        }
    return {
        "linear_layer_count": len(linear_indices),
        "sample_layer": sample_index,
        "projections": projections,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--random-init", action=argparse.BooleanOptionalAction, default=True, help="Instantiate from config without trained parameters.")
    parser.add_argument("--decode-steps", type=int_list, default=[30])
    parser.add_argument("--steps-per-frame", type=int_list, default=[5])
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-measured-frames", type=int, default=10)
    parser.add_argument("--continuous-measured-frames", type=int, default=50)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--require-fast-path", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_inputs(processor, device):
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((22, 66, 86, 132), fill=(190, 35, 35))
    draw.rectangle((134, 62, 207, 145), fill=(35, 70, 180))
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "Robot task: move the red block next to the blue block. "
                    "Respond with a detailed three-step plan in plain English."
                ),
            },
        ],
    }]]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)














@torch.inference_mode()
def persistent_continuous(
    model,
    action_head,
    adapter,
    processor,
    inputs,
    decode_steps,
    steps_per_frame,
    warmup_frames,
    measured_frames,
    device,
):
    batch = decode_steps // steps_per_frame
    bootstrap_prefix = q25.prefill(model, inputs)
    ages = [index * steps_per_frame for index in range(batch)]
    snapshots, next_by_age, rope_by_age, generated_by_age = build_age_snapshots(
        adapter, processor, bootstrap_prefix, max(ages), decode_steps
    )
    cache = Qwen35StaticHybridCache.stack([snapshots[age] for age in ages])
    next_tokens = torch.cat([next_by_age[age] for age in ages], dim=0)
    rope_deltas = torch.cat([rope_by_age[age] for age in ages], dim=0)
    generated = [list(generated_by_age[age]) for age in ages]

    cache, next_tokens, ages, token_rows = persistent_decode_steps(
        adapter, cache, next_tokens, rope_deltas, ages, steps_per_frame
    )
    for index, values in enumerate(token_rows[:, :, 0].detach().cpu().tolist()):
        generated[index].extend(values)
    free_slots = [index for index, age in enumerate(ages) if age >= decode_steps]
    if len(free_slots) != 1:
        raise RuntimeError(f"expected one initial free slot, got {free_slots}")
    free_slot = free_slots[0]

    frame_ms = []
    stage_times_ms = {
        "prefill_ms": [],
        "action_denoise_ms": [],
        "language_manager_ms": [],
        "language_decode_ms": [],
    }
    completed = []
    action = None
    for frame_index in range(warmup_frames + measured_frames):
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        events[0].record()
        frame_prefix = q25.prefill(model, inputs)
        events[1].record()
        action = action_head.predict(frame_prefix)
        events[2].record()
        new_state = q25.init_language_from_prefix(processor, frame_prefix, stop_on_eos=False)
        cache.replace_row_from_prefix(free_slot, new_state.cache)
        next_tokens[free_slot].copy_(new_state.next_token[0])
        rope_deltas[free_slot].copy_(new_state.rope_deltas[0])
        ages[free_slot] = 0
        generated[free_slot] = []
        events[3].record()
        cache, next_tokens, ages, token_rows = persistent_decode_steps(
            adapter, cache, next_tokens, rope_deltas, ages, steps_per_frame
        )
        events[4].record()
        torch.cuda.synchronize(device)
        local_ms = torch.tensor(
            [(time.perf_counter() - started) * 1000], device=device, dtype=torch.float64
        )
        dist.all_reduce(local_ms, op=dist.ReduceOp.MAX)
        stage_values = q25._rank_max(
            [events[index].elapsed_time(events[index + 1]) for index in range(4)], device
        )
        for index, values in enumerate(token_rows[:, :, 0].detach().cpu().tolist()):
            generated[index].extend(values)
        free_slots = [index for index, age in enumerate(ages) if age >= decode_steps]
        if len(free_slots) != 1:
            raise RuntimeError(f"expected one completed request, got {free_slots}")
        free_slot = free_slots[0]
        if frame_index >= warmup_frames:
            frame_ms.append(float(local_ms.item()))
            for name, value in zip(stage_times_ms, stage_values):
                stage_times_ms[name].append(value)
            completed.append(list(generated[free_slot]))
    return {
        "latency_ms": statistics.fmean(frame_ms),
        "mean_frame_ms": statistics.fmean(frame_ms),
        "p50_frame_ms": statistics.median(frame_ms),
        "frame_cv": statistics.stdev(frame_ms) / statistics.fmean(frame_ms)
        if len(frame_ms) > 1
        else 0.0,
        "half_mean_relative_difference": abs(
            statistics.fmean(frame_ms[: len(frame_ms) // 2])
            - statistics.fmean(frame_ms[-(len(frame_ms) // 2) :])
        )
        / statistics.fmean(frame_ms)
        if len(frame_ms) > 1
        else 0.0,
        "frame_times_ms": frame_ms,
        "stage_times_ms": stage_times_ms,
        "mean_stage_ms": {
            name: statistics.fmean(values) for name, values in stage_times_ms.items()
        },
        "avg_batch_size": batch,
        "completed_requests": len(completed),
        "all_completed_lengths_correct": all(len(row) == decode_steps for row in completed),
        "completed_text_sample": processor.decode(completed[-1], skip_special_tokens=True),
        "all_completed_text_nonempty": all(
            bool(processor.decode(row, skip_special_tokens=True).strip()) for row in completed
        ),
        "completed_token_sample": completed[-1],
        "action_checksum": float(action.float().sum()),
        "action_shape": list(action.shape),
        "action_finite": bool(torch.isfinite(action).all()),
    }


def release_memory():
    gc.collect()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    return {"allocated_bytes": allocated, "reserved_bytes": reserved}


def main():
    args = parse_args()
    if args.random_init:
        torch.manual_seed(0)
    if args.require_fast_path and not qwen35_modeling.is_fast_path_available:
        raise RuntimeError("Qwen3.5 fast path is required but unavailable")
    compiler_state_before = compiler_counters()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    world_size = dist.get_world_size()
    if args.random_init and world_size > 1:
        raise RuntimeError("Random-init TP2 is not implemented; refusing replicated models labelled as TP2")
    load_kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if world_size > 1:
        load_kwargs["tp_plan"] = "auto"
    if args.random_init:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
        config._attn_implementation = "sdpa"
        model = Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16).eval()
    else:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, **load_kwargs).eval()
    if world_size == 1 or args.random_init:
        model = model.to(device)
    action_head = q25.ReplicatedPIv3Action(model.config, args.denoise_steps)
    action_head = action_head.to(dtype=torch.bfloat16, device=device).eval()
    action_head_parameter_count = sum(parameter.numel() for parameter in action_head.parameters())
    linear_attention_projection_placement = linear_attention_placement(model)
    allocated_after_load_bytes = torch.cuda.memory_allocated()
    reserved_after_load_bytes = torch.cuda.memory_reserved()
    load_memory = torch.tensor(
        [allocated_after_load_bytes, reserved_after_load_bytes], device=device, dtype=torch.int64
    )
    load_memory_gathered = [torch.empty_like(load_memory) for _ in range(world_size)]
    dist.all_gather(load_memory_gathered, load_memory)
    load_memory_per_rank = [row.cpu().tolist() for row in load_memory_gathered]
    processor = AutoProcessor.from_pretrained(args.model)
    inputs = build_inputs(processor, device)
    adapter = Qwen35NativeAdapter(model)

    for _ in range(args.warmup):
        q25.run_shared(model, action_head, adapter, processor, inputs, min(args.decode_steps))
    cache_audit = mixed_age_cache_audit(model, adapter, processor, inputs)
    from oxygen_models.qwen35.validation import validate_cache_implementation
    cache_audit["validation_policy"] = "matched_batch_cache_integrity_v2"
    try:
        validate_cache_implementation(cache_audit)
    except AssertionError as error:
        raise RuntimeError(f"Qwen3.5 implementation audit failed: {error}; {cache_audit}") from error

    decode_steps_grid = [10] if args.smoke else args.decode_steps
    steps_per_frame_grid = [5] if args.smoke else args.steps_per_frame
    repeats = 1 if args.smoke else args.repeats
    baseline_frames = 2 if args.smoke else args.baseline_measured_frames
    continuous_frames = 3 if args.smoke else args.continuous_measured_frames
    warmup = 1 if args.smoke else args.warmup
    profile_repeats = 0 if args.smoke else args.profile_repeats
    collective_profiles = [
        q25.collective_profile(lambda: q25.prefill(model, inputs), device)
        for _ in range(profile_repeats)
    ]

    records = []
    for repeat in range(repeats):
        for _ in range(warmup):
            q25.prefill(model, inputs)
        torch.cuda.reset_peak_memory_stats()
        _, timing = q25.measure_stable_frames(
            lambda: q25.prefill(model, inputs), device, baseline_frames,
            maximum_frames=baseline_frames if args.smoke else 30,
        )
        records.append({
            "mode": "prefill_only",
            "decode_steps": None,
            "steps_per_frame": None,
            "repeat": repeat,
            **timing,
            **q25.peak_memory(device),
        })
    for decode_steps in decode_steps_grid:
        for mode, work in (("separate", q25.run_separate), ("shared_kv", q25.run_shared)):
            for repeat in range(repeats):
                for _ in range(warmup):
                    work(model, action_head, adapter, processor, inputs, decode_steps)
                torch.cuda.reset_peak_memory_stats()
                (action, language), timing = q25.measure_stable_frames(
                    lambda work=work: work(
                        model, action_head, adapter, processor, inputs, decode_steps
                    ),
                    device,
                    baseline_frames,
                    maximum_frames=baseline_frames if args.smoke else 30,
                )
                records.append({
                    "mode": mode,
                    "decode_steps": decode_steps,
                    "steps_per_frame": None,
                    "repeat": repeat,
                    **timing,
                    "breakdown": q25.timed_mode_breakdown(
                        mode,
                        model,
                        action_head,
                        adapter,
                        processor,
                        inputs,
                        decode_steps,
                        device,
                    ),
                    "action_checksum": float(action.float().sum()),
                    "action_shape": list(action.shape),
                    "action_finite": bool(torch.isfinite(action).all()),
                    "tokens": list(language.generated_ids),
                    "text": processor.decode(language.generated_ids, skip_special_tokens=True),
                    **q25.peak_memory(device),
                })
        for steps_per_frame in steps_per_frame_grid:
            if decode_steps % steps_per_frame:
                continue
            for repeat in range(repeats):
                torch.cuda.reset_peak_memory_stats()
                row = persistent_continuous(
                    model,
                    action_head,
                    adapter,
                    processor,
                    inputs,
                    decode_steps,
                    steps_per_frame,
                    warmup,
                    continuous_frames,
                    device,
                )
                row.update({
                    "mode": "continuous_batching",
                    "decode_steps": decode_steps,
                    "steps_per_frame": steps_per_frame,
                    "repeat": repeat,
                    "memory_after_repeat": release_memory(),
                    **q25.peak_memory(device),
                })
                records.append(row)

    payload = {
        "metadata": {
            "model": args.model,
            "world_size": world_size,
            "backend": "native_hf_tp" if world_size > 1 else "single_gpu",
            "scope": "end_to_end_random_pi_v3_action_and_language_hybrid_cache",
            "workload": "common_prefix_plus_random_pi_v3_action_plus_language",
            "smoke": args.smoke,
            "denoise_steps": args.denoise_steps,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "compile": False,
            "cuda_graph": False,
            "fast_path_required": args.require_fast_path,
            "fast_path_available": bool(qwen35_modeling.is_fast_path_available),
            "fast_path_packages": {
                "flash-linear-attention": package_version("flash-linear-attention"),
                "causal-conv1d": package_version("causal-conv1d"),
            },
            "fast_path_functions": {
                "causal_conv1d_fn": callable_source(qwen35_modeling.causal_conv1d_fn),
                "causal_conv1d_update": callable_source(qwen35_modeling.causal_conv1d_update),
                "chunk_gated_delta_rule": callable_source(qwen35_modeling.chunk_gated_delta_rule),
                "fused_recurrent_gated_delta_rule": callable_source(
                    qwen35_modeling.fused_recurrent_gated_delta_rule
                ),
            },
            "compiler_state_before": compiler_state_before,
            "compiler_state_after": compiler_counters(),
            "prefix_tokens": int(inputs["input_ids"].shape[-1]),
            "input_signature": {
                "input_ids": inputs["input_ids"].detach().cpu().tolist(),
                "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
                "image_grid_thw": inputs.get("image_grid_thw", torch.empty(0)).detach().cpu().tolist(),
            },
            "vlm_num_hidden_layers": int(model.config.text_config.num_hidden_layers),
            "vlm_layer_types": list(model.config.text_config.layer_types),
            "action_num_dit_layers": action_head.num_action_dit_layers,
            "action_head_parameter_count": action_head_parameter_count,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "action_dit_hidden_dim": 1024,
            "warmup": warmup,
            "repeats": repeats,
            "baseline_measured_frames": baseline_frames,
            "baseline_extension_rule": "30 frames when 10-frame CV or first-half/last-half relative difference exceeds 2%",
            "continuous_measured_frames": continuous_frames,
            "aggregation": "median of three per-repeat frame means",
            "tp_plan": model._tp_plan if world_size > 1 else None,
            "linear_attention_projection_placement": linear_attention_projection_placement,
            "allocated_after_load_bytes": allocated_after_load_bytes,
            "reserved_after_load_bytes": reserved_after_load_bytes,
            "allocated_after_load_bytes_per_rank": [row[0] for row in load_memory_per_rank],
            "reserved_after_load_bytes_per_rank": [row[1] for row in load_memory_per_rank],
            "hybrid_cache_manager": "full_attention_kv_plus_linear_attention_conv_and_recurrent_states",
            "mixed_age_cache_audit": cache_audit,
        },
        "prefill_collective_profile_rank0": collective_profiles[0] if collective_profiles else [],
        "prefill_collective_profiles_rank0": collective_profiles,
        "records": records,
    }
    if dist.get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(payload["metadata"], ensure_ascii=False), flush=True)
    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
