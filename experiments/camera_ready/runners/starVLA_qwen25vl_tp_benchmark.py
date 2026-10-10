#!/usr/bin/env python3
"""TP=1/2 end-to-end benchmark for PI_v3 action and language experts."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from torch.profiler import ProfilerActivity, profile
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

XIAOMI_SCRIPTS = str(Path(__file__).resolve().parent)
STARVLA_SCRIPTS = str(Path(__file__).resolve().parent)
for script_dir in (XIAOMI_SCRIPTS, STARVLA_SCRIPTS):
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

from oxygen_heterogeneous_sweep import (  # noqa: E402
    PerRequestStaticLanguageCache,
    persistent_decode_steps,
)
from oxygen_runtime import (  # noqa: E402
    PrefixState,
    _snapshot_cache,
    init_language_from_prefix,
    language_step,
    staticize_language_state,
)
from starVLA.model.framework.VLM4A.QwenPI_v3 import QwenPI_v3DefaultConfig  # noqa: E402
from starVLA.model.framework.share_tools import (  # noqa: E402
    merge_framework_config,
    populate_layerwise_dit_cfg,
)
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (  # noqa: E402
    get_action_model,
)


class NativeAdapter:
    def __init__(self, model):
        self.vlm = NativeWrapper(model)


class NativeWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, **kwargs):
        return self.model(**kwargs)


class ReplicatedPIv3Action(nn.Module):
    """Random PI_v3 head replicated on every TP rank."""

    def __init__(self, vlm_config, denoise_steps):
        super().__init__()
        text_config = getattr(vlm_config, "text_config", vlm_config)
        hidden_size = int(text_config.hidden_size)
        num_layers = int(text_config.num_hidden_layers)
        config = merge_framework_config(
            QwenPI_v3DefaultConfig,
            OmegaConf.create({
                "framework": {
                    "name": "QwenPI_v3",
                    "action_model": {
                        "action_dim": 7,
                        "state_dim": 7,
                        "action_horizon": 16,
                        "num_inference_timesteps": denoise_steps,
                        "diffusion_model_cfg": {"action_dit_hidden_dim": 1024},
                    },
                },
                "trainer": {"pretrained_checkpoint": None},
            }),
        )
        config.framework.qwenvl.vl_hidden_dim = hidden_size
        config.framework.qwenvl.num_vl_layers = num_layers
        populate_layerwise_dit_cfg(config, dit_hidden_dim=1024, num_dit_layers=num_layers)
        self.action_model = get_action_model(config=config)
        self.action_model.num_inference_timesteps = denoise_steps
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)
        self.project_layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1024))
            for _ in range(self.num_action_dit_layers)
        ])

    @torch.inference_mode()
    def predict(self, prefix, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        hidden = list(prefix.outputs.hidden_states[-self.num_action_dit_layers:])
        if any(type(value).__name__ == "DTensor" for value in hidden):
            raise TypeError("PI_v3 expects replicated VLM hidden states, got DTensor")
        hidden = [project(value) for project, value in zip(self.project_layers, hidden)]
        mask = prefix.outputs.attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.float32):
            return self.action_model.predict_action(
                hidden, state=None, encoder_attention_mask=mask
            )


def int_list(value):
    return [int(item) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--decode-steps", type=int_list, default=[10, 20, 30])
    parser.add_argument("--steps-per-frame", type=int_list, default=[1, 5, 10])
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-measured-frames", type=int, default=10)
    parser.add_argument("--continuous-measured-frames", type=int, default=50)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rank_max(values, device):
    """Return rank-max timings without changing the measured work."""
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return [float(value) for value in tensor.cpu().tolist()]


def timed_mode_breakdown(mode, model, action_head, adapter, processor, inputs,
                         decode_steps, device):
    """One synchronized diagnostic frame; end-to-end records remain uninstrumented."""
    dist.barrier(device_ids=[device.index])
    torch.cuda.synchronize(device)
    local = []
    started = time.perf_counter()
    if mode == "separate":
        prefix = prefill(model, inputs)
        torch.cuda.synchronize(device)
        local.append((time.perf_counter() - started) * 1000)
        action_started = time.perf_counter()
        action = action_head.predict(prefix)
        torch.cuda.synchronize(device)
        local.append((time.perf_counter() - action_started) * 1000)
        language_started = time.perf_counter()
        language_prefix = prefill(model, inputs)
        torch.cuda.synchronize(device)
        prefill2 = (time.perf_counter() - language_started) * 1000
        decode_started = time.perf_counter()
        language = run_language(adapter, processor, language_prefix, decode_steps)
        torch.cuda.synchronize(device)
        local.extend([prefill2, (time.perf_counter() - decode_started) * 1000])
        names = ["prefill_action_ms", "action_denoise_ms", "prefill_language_ms", "language_decode_ms"]
    elif mode == "shared_kv":
        prefix = prefill(model, inputs)
        torch.cuda.synchronize(device)
        local.append((time.perf_counter() - started) * 1000)
        action_started = time.perf_counter()
        action = action_head.predict(prefix)
        torch.cuda.synchronize(device)
        local.append((time.perf_counter() - action_started) * 1000)
        decode_started = time.perf_counter()
        language = run_language(adapter, processor, prefix, decode_steps)
        torch.cuda.synchronize(device)
        local.append((time.perf_counter() - decode_started) * 1000)
        names = ["prefill_ms", "action_denoise_ms", "language_decode_ms"]
    else:
        raise ValueError(mode)
    values = _rank_max(local, device)
    return {name: value for name, value in zip(names, values)}


def build_inputs(processor, device):
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((22, 66, 86, 132), fill=(190, 35, 35))
    draw.rectangle((134, 62, 207, 145), fill=(35, 70, 180))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Move the red block next to the blue block."},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(device)


@torch.inference_mode()
def prefill(model, inputs):
    outputs = model(
        **inputs,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
        logits_to_keep=1,
    )
    outputs.attention_mask = inputs["attention_mask"]
    return PrefixState(
        outputs=outputs,
        action_kv=_snapshot_cache(outputs.past_key_values),
        prefix_length=int(inputs["attention_mask"].shape[-1]),
    )


@torch.inference_mode()
def run_language(adapter, processor, prefix, decode_steps):
    state = init_language_from_prefix(processor, prefix, stop_on_eos=False)
    for _ in range(decode_steps):
        language_step(adapter, state)
    return state


@torch.inference_mode()
def run_separate(model, action_head, adapter, processor, inputs, decode_steps):
    action_prefix = prefill(model, inputs)
    action = action_head.predict(action_prefix)
    language_prefix = prefill(model, inputs)
    language = run_language(adapter, processor, language_prefix, decode_steps)
    return action, language


@torch.inference_mode()
def run_shared(model, action_head, adapter, processor, inputs, decode_steps):
    common_prefix = prefill(model, inputs)
    action = action_head.predict(common_prefix)
    language = run_language(adapter, processor, common_prefix, decode_steps)
    return action, language


def timed_max(work, device):
    dist.barrier(device_ids=[device.index])
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = work()
    torch.cuda.synchronize(device)
    local_ms = torch.tensor(
        [(time.perf_counter() - started) * 1000], device=device, dtype=torch.float64
    )
    dist.all_reduce(local_ms, op=dist.ReduceOp.MAX)
    return output, float(local_ms.item())


def measure_stable_frames(work, device, minimum_frames, maximum_frames=30):
    outputs = None
    frame_times_ms = []
    for _ in range(minimum_frames):
        outputs, latency_ms = timed_max(work, device)
        frame_times_ms.append(latency_ms)
    mean_ms = statistics.fmean(frame_times_ms)
    cv = statistics.stdev(frame_times_ms) / mean_ms if len(frame_times_ms) > 1 else 0.0
    half = len(frame_times_ms) // 2
    trend = abs(
        statistics.fmean(frame_times_ms[:half])
        - statistics.fmean(frame_times_ms[-half:])
    ) / mean_ms
    extended = (cv > 0.02 or trend > 0.02) and minimum_frames < maximum_frames
    if extended:
        for _ in range(maximum_frames - minimum_frames):
            outputs, latency_ms = timed_max(work, device)
            frame_times_ms.append(latency_ms)
        mean_ms = statistics.fmean(frame_times_ms)
        cv = statistics.stdev(frame_times_ms) / mean_ms
        half = len(frame_times_ms) // 2
        trend = abs(
            statistics.fmean(frame_times_ms[:half])
            - statistics.fmean(frame_times_ms[-half:])
        ) / mean_ms
    return outputs, {
        "latency_ms": mean_ms,
        "mean_frame_ms": mean_ms,
        "p50_frame_ms": statistics.median(frame_times_ms),
        "frame_times_ms": frame_times_ms,
        "frame_cv": cv,
        "half_mean_relative_difference": trend,
        "extended_to_30_frames": extended,
    }


@torch.inference_mode()
def persistent_continuous(
    model, action_head, adapter, processor, inputs, decode_steps, steps_per_frame,
    warmup_frames, measured_frames, device
):
    batch = decode_steps // steps_per_frame
    bootstrap_prefix = prefill(model, inputs)
    ages = [index * steps_per_frame for index in range(batch)]
    template = init_language_from_prefix(processor, bootstrap_prefix, stop_on_eos=False)
    staticize_language_state(template, decode_steps)
    next_token_by_age = [template.next_token.clone()]
    for _ in range(max(ages)):
        language_step(adapter, template)
        next_token_by_age.append(template.next_token.clone())
    cache = PerRequestStaticLanguageCache.from_tensors(
        [key.expand(batch, -1, -1, -1).clone() for key in template.cache.key_cache],
        [value.expand(batch, -1, -1, -1).clone() for value in template.cache.value_cache],
        template.cache.prefix_length,
        template.cache.max_decode_steps,
    )
    next_tokens = torch.cat([next_token_by_age[age] for age in ages])
    rope_deltas = template.rope_deltas.expand(batch, -1).clone()
    generated = [list(template.generated_ids[:age]) for age in ages]
    cache, next_tokens, ages, token_rows = persistent_decode_steps(
        adapter, cache, next_tokens, rope_deltas, ages, steps_per_frame
    )
    for index, values in enumerate(token_rows[:, :, 0].detach().cpu().tolist()):
        generated[index].extend(values)
    free_slot = [index for index, age in enumerate(ages) if age >= decode_steps]
    if len(free_slot) != 1:
        raise RuntimeError(f"expected one initial free slot, got {free_slot}")
    free_slot = free_slot[0]

    frame_ms = []
    stage_times_ms = {
        "prefill_ms": [],
        "action_denoise_ms": [],
        "language_manager_ms": [],
        "language_decode_ms": [],
    }
    completed = []
    action = None
    prefix_length = cache.prefix_length
    for frame_index in range(warmup_frames + measured_frames):
        dist.barrier(device_ids=[device.index])
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        events[0].record()
        frame_prefix = prefill(model, inputs)
        events[1].record()
        action = action_head.predict(frame_prefix)
        events[2].record()
        new_state = init_language_from_prefix(processor, frame_prefix, stop_on_eos=False)
        staticize_language_state(new_state, decode_steps)
        for layer in range(len(cache.key_cache)):
            cache.key_cache[layer][free_slot, :, :prefix_length].copy_(
                new_state.cache.key_cache[layer][0, :, :prefix_length]
            )
            cache.value_cache[layer][free_slot, :, :prefix_length].copy_(
                new_state.cache.value_cache[layer][0, :, :prefix_length]
            )
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
        stage_values = _rank_max(
            [events[index].elapsed_time(events[index + 1]) for index in range(4)],
            device,
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
        "frame_cv": statistics.stdev(frame_ms) / statistics.fmean(frame_ms),
        "half_mean_relative_difference": abs(
            statistics.fmean(frame_ms[: len(frame_ms) // 2])
            - statistics.fmean(frame_ms[-(len(frame_ms) // 2):])
        ) / statistics.fmean(frame_ms),
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
            bool(processor.decode(row, skip_special_tokens=True).strip())
            for row in completed
        ),
        "action_checksum": float(action.float().sum()),
        "action_shape": list(action.shape),
        "action_finite": bool(torch.isfinite(action).all()),
    }


def collective_profile(work, device):
    dist.barrier(device_ids=[device.index])
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        work()
        torch.cuda.synchronize(device)
    dist.barrier(device_ids=[device.index])
    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        name = event.key.lower()
        if any(term in name for term in ("nccl", "all_reduce", "all_gather", "reduce_scatter")):
            payload_bytes = None
            if event.input_shapes and event.input_shapes[0]:
                payload_elements = math.prod(event.input_shapes[0])
                payload_bytes = int(event.count * payload_elements * 2)
            rows.append({
                "name": event.key,
                "count": event.count,
                "cpu_time_total_us": float(event.cpu_time_total),
                "device_time_total_us": float(getattr(event, "device_time_total", 0.0)),
                "input_shapes": event.input_shapes,
                "aggregate_input_payload_bytes_assuming_bf16": payload_bytes,
            })
    return rows


def release_memory():
    gc.collect()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    return {"allocated_bytes": allocated, "reserved_bytes": reserved}


def peak_memory(device):
    local = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        device=device,
        dtype=torch.int64,
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    per_rank = [row.cpu().tolist() for row in gathered]
    return {
        "peak_allocated_bytes": max(row[0] for row in per_rank),
        "peak_reserved_bytes": max(row[1] for row in per_rank),
        "peak_allocated_bytes_per_rank": [row[0] for row in per_rank],
        "peak_reserved_bytes_per_rank": [row[1] for row in per_rank],
    }


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    world_size = dist.get_world_size()
    load_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if world_size > 1:
        load_kwargs["tp_plan"] = "auto"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, **load_kwargs
    ).eval()
    if world_size == 1:
        model = model.to(device)
    action_head = ReplicatedPIv3Action(model.config, args.denoise_steps)
    action_head = action_head.to(dtype=torch.bfloat16, device=device).eval()
    allocated_after_load_bytes = torch.cuda.memory_allocated()
    reserved_after_load_bytes = torch.cuda.memory_reserved()
    load_memory = torch.tensor(
        [allocated_after_load_bytes, reserved_after_load_bytes],
        device=device,
        dtype=torch.int64,
    )
    load_memory_gathered = [torch.empty_like(load_memory) for _ in range(world_size)]
    dist.all_gather(load_memory_gathered, load_memory)
    load_memory_per_rank = [row.cpu().tolist() for row in load_memory_gathered]
    processor = AutoProcessor.from_pretrained(args.model)
    inputs = build_inputs(processor, device)
    adapter = NativeAdapter(model)

    for _ in range(args.warmup):
        run_shared(
            model, action_head, adapter, processor, inputs, min(args.decode_steps)
        )
    collective_profiles = [
        collective_profile(lambda: prefill(model, inputs), device)
        for _ in range(args.profile_repeats)
    ]

    records = []
    for repeat in range(args.repeats):
        for _ in range(args.warmup):
            prefill(model, inputs)
        torch.cuda.reset_peak_memory_stats()
        _, timing = measure_stable_frames(
            lambda: prefill(model, inputs), device, args.baseline_measured_frames
        )
        records.append({
            "mode": "prefill_only",
            "decode_steps": None,
            "steps_per_frame": None,
            "repeat": repeat,
            **timing,
            **peak_memory(device),
        })
    for decode_steps in args.decode_steps:
        for mode, work in (
            ("separate", run_separate),
            ("shared_kv", run_shared),
        ):
            for repeat in range(args.repeats):
                for _ in range(args.warmup):
                    work(model, action_head, adapter, processor, inputs, decode_steps)
                torch.cuda.reset_peak_memory_stats()
                (action, language), timing = measure_stable_frames(
                    lambda work=work: work(
                        model, action_head, adapter, processor, inputs, decode_steps
                    ),
                    device,
                    args.baseline_measured_frames,
                )
                records.append({
                    "mode": mode,
                    "decode_steps": decode_steps,
                    "steps_per_frame": None,
                    "repeat": repeat,
                    **timing,
                    "breakdown": timed_mode_breakdown(
                        mode, model, action_head, adapter, processor, inputs,
                        decode_steps, device,
                    ),
                    "action_checksum": float(action.float().sum()),
                    "action_shape": list(action.shape),
                    "action_finite": bool(torch.isfinite(action).all()),
                    "tokens": list(language.generated_ids),
                    "text": processor.decode(language.generated_ids, skip_special_tokens=True),
                    **peak_memory(device),
                })
        for steps_per_frame in args.steps_per_frame:
            if decode_steps % steps_per_frame:
                continue
            for repeat in range(args.repeats):
                torch.cuda.reset_peak_memory_stats()
                row = persistent_continuous(
                    model, action_head, adapter, processor, inputs, decode_steps,
                    steps_per_frame, args.warmup, args.continuous_measured_frames, device,
                )
                row.update({
                    "mode": "continuous_batching",
                    "decode_steps": decode_steps,
                    "steps_per_frame": steps_per_frame,
                    "repeat": repeat,
                    "memory_after_repeat": release_memory(),
                    **peak_memory(device),
                })
                records.append(row)

    payload = {
        "metadata": {
            "model": args.model,
            "world_size": world_size,
            "backend": "native_hf_tp" if world_size > 1 else "single_gpu",
            "scope": "end_to_end_random_pi_v3_action_and_language_systems_path",
            "workload": "common_prefix_plus_random_pi_v3_action_plus_language",
            "denoise_steps": args.denoise_steps,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
            "compile": False,
            "cuda_graph": False,
            "prefix_tokens": int(inputs["input_ids"].shape[-1]),
            "input_signature": {
                "input_ids": inputs["input_ids"].detach().cpu().tolist(),
                "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
                "image_grid_thw": inputs.get("image_grid_thw", torch.empty(0)).detach().cpu().tolist(),
            },
            "vlm_num_hidden_layers": int(getattr(model.config.text_config, "num_hidden_layers")),
            "action_num_dit_layers": action_head.num_action_dit_layers,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
            "action_dit_hidden_dim": 1024,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "baseline_measured_frames": args.baseline_measured_frames,
            "baseline_extension_rule": "30 frames when 10-frame CV or first-half/last-half relative difference exceeds 2%",
            "continuous_measured_frames": args.continuous_measured_frames,
            "aggregation": "median of three per-repeat frame means",
            "tp_plan": model._tp_plan if world_size > 1 else None,
            "allocated_after_load_bytes": allocated_after_load_bytes,
            "reserved_after_load_bytes": reserved_after_load_bytes,
            "allocated_after_load_bytes_per_rank": [row[0] for row in load_memory_per_rank],
            "reserved_after_load_bytes_per_rank": [row[1] for row in load_memory_per_rank],
        },
        "prefill_collective_profile_rank0": collective_profiles[0],
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
