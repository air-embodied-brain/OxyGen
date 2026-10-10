#!/usr/bin/env python3
"""Isolated Qwen3.5 text-decoder scaling diagnostic for 4B/9B and TP1/TP2."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.profiler import ProfilerActivity, profile
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_modeling
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
from transformers.utils import logging as transformers_logging


PREFILL_BATCH_SIZES = (1, 3, 6)
DECODE_BATCH_SIZES = (1, 3, 6, 30)
PREFILL_PROFILE_BATCH_SIZES = (1, 6)
DECODE_PROFILE_BATCH_SIZES = (1, 30)
WARMUP = 3
REPEATS = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compiler_counters():
    dynamo = {
        group: {name: int(value) for name, value in values.items()}
        for group, values in torch._dynamo.utils.counters.items()
        if values
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


def callable_source(value):
    return f"{value.__module__}.{value.__name__}"


def checkpoint_parameter_count(model_path: Path, random_init=False):
    if random_init:
        return None, None
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        files = sorted(set(index["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    total = 0
    tensor_count = 0
    for filename in files:
        with safe_open(model_path / filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                total += math.prod(handle.get_slice(key).get_shape())
                tensor_count += 1
    return total, tensor_count


def local_numel(parameter):
    if type(parameter).__name__ == "DTensor":
        return parameter.to_local().numel()
    return parameter.numel()


def rank_max_time(work, device):
    dist.barrier(device_ids=[device.index])
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = work()
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(
        [(time.perf_counter() - started) * 1000], device=device, dtype=torch.float64
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return output, float(elapsed.item())


def summarize_times(values):
    return {
        "rank_max_times_ms": values,
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "cv": statistics.stdev(values) / statistics.fmean(values),
    }


def gemm_flops(key, shapes, count):
    try:
        if key == "aten::mm" and len(shapes) >= 2:
            left, right = shapes[0], shapes[1]
            return count * 2 * left[-2] * left[-1] * right[-1]
        if key == "aten::addmm" and len(shapes) >= 3:
            left, right = shapes[1], shapes[2]
            return count * 2 * left[-2] * left[-1] * right[-1]
        if key == "aten::bmm" and len(shapes) >= 2:
            left, right = shapes[0], shapes[1]
            return count * 2 * left[-3] * left[-2] * left[-1] * right[-1]
    except (IndexError, TypeError):
        return 0
    return 0


def profile_work(work, device):
    dist.barrier(device_ids=[device.index])
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        output = work()
        torch.cuda.synchronize(device)
    dist.barrier(device_ids=[device.index])

    rows = []
    gemm_total_flops = 0
    gemm_device_time_us = 0.0
    for event in profiler.key_averages(group_by_input_shape=True):
        self_device_time = float(getattr(event, "self_device_time_total", 0.0))
        device_time = float(getattr(event, "device_time_total", 0.0))
        shapes = event.input_shapes
        flops = gemm_flops(event.key, shapes, event.count)
        if flops:
            gemm_total_flops += flops
            gemm_device_time_us += self_device_time or device_time
        rows.append({
            "operator": event.key,
            "count": int(event.count),
            "self_device_time_us": self_device_time,
            "device_time_total_us": device_time,
            "input_shapes": shapes,
            "estimated_gemm_flops": flops,
        })
    rows.sort(key=lambda row: row["self_device_time_us"], reverse=True)
    summary = {
        "top_operators_by_self_device_time": rows[:25],
        "estimated_aten_gemm_flops": gemm_total_flops,
        "estimated_aten_gemm_device_time_us": gemm_device_time_us,
        "estimated_aten_gemm_tflops": (
            gemm_total_flops / gemm_device_time_us / 1e6
            if gemm_device_time_us
            else None
        ),
    }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, summary)
    return output, gathered


class NativeTextWrapper:
    def __init__(self, model):
        self.model = model

    def prefill(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            logits_to_keep=1,
        )

    def decode(self, input_ids, attention_mask, cache):
        logical_position = attention_mask.long().sum(dim=-1, keepdim=True) - 1
        rope_deltas = self.model.model.rope_deltas
        if rope_deltas is None:
            rope_deltas = torch.zeros_like(logical_position)
        position_ids = (logical_position + rope_deltas).unsqueeze(0).expand(3, -1, -1)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            logits_to_keep=1,
        )


def build_100_tokens(processor, device):
    text = (
        "The red block is on the left and the blue block is on the right. "
        "Plan how to move the red block next to the blue block. "
    )
    seed = processor.tokenizer.encode(text, add_special_tokens=False)
    token_ids = (seed * math.ceil(100 / len(seed)))[:100]
    return torch.tensor([token_ids], device=device, dtype=torch.long), token_ids, text


def next_token_and_checks(output):
    logits = output.logits[:, -1]
    assert torch.isfinite(logits).all()
    tokens = logits.argmax(dim=-1)
    assert bool((tokens == tokens[0]).all()), "identical batch rows diverged"
    return tokens[:, None]


def replicate_native_cache(cache, config, batch_size):
    replicated = Qwen3_5DynamicCache(config)

    def repeat(value):
        if value is None:
            return None
        assert value.shape[0] == 1
        repeats = [batch_size] + [1] * (value.ndim - 1)
        return value.repeat(*repeats)

    replicated.key_cache = [repeat(value) for value in cache.key_cache]
    replicated.value_cache = [repeat(value) for value in cache.value_cache]
    replicated.conv_states = [repeat(value) for value in cache.conv_states]
    replicated.recurrent_states = [repeat(value) for value in cache.recurrent_states]
    return replicated


@torch.inference_mode()
def main():
    args = parse_args()
    if args.random_init:
        torch.manual_seed(0)
    transformers_logging.disable_progress_bar()
    if not qwen35_modeling.is_fast_path_available:
        raise RuntimeError("Qwen3.5 fast path is unavailable")
    state_before = compiler_counters()
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
    wrapper = NativeTextWrapper(model)
    processor = AutoProcessor.from_pretrained(args.model)
    base_ids, token_ids, source_text = build_100_tokens(processor, device)

    loaded_global_numel = sum(parameter.numel() for parameter in model.parameters())
    loaded_local_numel = sum(local_numel(parameter) for parameter in model.parameters())
    text_global_numel = sum(
        parameter.numel() for parameter in model.model.language_model.parameters()
    )
    text_local_numel = sum(
        local_numel(parameter) for parameter in model.model.language_model.parameters()
    )
    checkpoint_numel, checkpoint_tensor_count = checkpoint_parameter_count(args.model, random_init=args.random_init)
    memory = torch.tensor(
        [torch.cuda.memory_allocated(), torch.cuda.memory_reserved()],
        device=device,
        dtype=torch.int64,
    )
    memory_by_rank = [torch.empty_like(memory) for _ in range(world_size)]
    dist.all_gather(memory_by_rank, memory)

    records = []
    profiles = []
    for batch_size in PREFILL_BATCH_SIZES:
        input_ids = base_ids.expand(batch_size, -1).contiguous()
        attention_mask = torch.ones_like(input_ids)

        for _ in range(WARMUP):
            wrapper.prefill(input_ids, attention_mask)
        prefill_times = []
        prefill_output = None
        torch.cuda.reset_peak_memory_stats()
        for _ in range(REPEATS):
            prefill_output, elapsed = rank_max_time(
                lambda: wrapper.prefill(input_ids, attention_mask), device
            )
            prefill_times.append(elapsed)
        prefill_tokens = next_token_and_checks(prefill_output)
        records.append({
            "stage": "prefill",
            "batch_size": batch_size,
            **summarize_times(prefill_times),
            "output_tokens": prefill_tokens[:, 0].detach().cpu().tolist(),
            "finite": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        if batch_size in PREFILL_PROFILE_BATCH_SIZES:
            profiled_output, gathered = profile_work(
                lambda: wrapper.prefill(input_ids, attention_mask), device
            )
            next_token_and_checks(profiled_output)
            if dist.get_rank() == 0:
                profiles.append({
                    "stage": "prefill",
                    "batch_size": batch_size,
                    "per_rank": gathered,
                })
            del profiled_output

        del prefill_output
        torch.cuda.empty_cache()

    singleton_mask = torch.ones_like(base_ids)
    for batch_size in DECODE_BATCH_SIZES:
        prefix = wrapper.prefill(base_ids, singleton_mask)
        cache = replicate_native_cache(
            prefix.past_key_values, model.config.text_config, batch_size
        )
        next_tokens = next_token_and_checks(prefix).expand(batch_size, -1).clone()
        decode_mask = singleton_mask.expand(batch_size, -1).clone()

        def decode_step():
            nonlocal cache, next_tokens, decode_mask
            decode_mask = torch.cat(
                [decode_mask, torch.ones((batch_size, 1), device=device, dtype=decode_mask.dtype)],
                dim=-1,
            )
            output = wrapper.decode(next_tokens, decode_mask, cache)
            cache = output.past_key_values
            next_tokens = next_token_and_checks(output)
            return output

        for _ in range(WARMUP):
            decode_step()
        decode_times = []
        decode_output = None
        torch.cuda.reset_peak_memory_stats()
        for _ in range(REPEATS):
            decode_output, elapsed = rank_max_time(decode_step, device)
            decode_times.append(elapsed)
        records.append({
            "stage": "incremental_decode",
            "batch_size": batch_size,
            **summarize_times(decode_times),
            "output_tokens": next_tokens[:, 0].detach().cpu().tolist(),
            "finite": bool(torch.isfinite(decode_output.logits).all()),
            "start_sequence_length": 100,
            "last_measured_sequence_length": int(decode_mask.shape[-1]),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        if batch_size in DECODE_PROFILE_BATCH_SIZES:
            profiled_output, gathered = profile_work(decode_step, device)
            next_token_and_checks(profiled_output)
            if dist.get_rank() == 0:
                profiles.append({
                    "stage": "incremental_decode",
                    "batch_size": batch_size,
                    "per_rank": gathered,
                })

        del prefix, decode_output, cache
        torch.cuda.empty_cache()

    config = model.config.text_config
    payload = {
        "metadata": {
            "model": str(args.model),
            "world_size": world_size,
            "backend": "native_hf_tp" if world_size > 1 else "single_gpu",
            "scope": "diagnostic_only_not_formal_table",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "compile": False,
            "cuda_graph": False,
            "warmup": WARMUP,
            "repeats": REPEATS,
            "prefill_batch_sizes": list(PREFILL_BATCH_SIZES),
            "incremental_decode_batch_sizes": list(DECODE_BATCH_SIZES),
            "decode_cache_construction": (
                "replicate_singleton_native_qwen3_5_dynamic_hybrid_cache"
            ),
            "input_token_count": len(token_ids),
            "input_token_ids": token_ids,
            "input_source_text": source_text,
            "random_init": args.random_init,
            "checkpoint_parameter_count": checkpoint_numel,
            "checkpoint_tensor_count": checkpoint_tensor_count,
            "loaded_global_parameter_count": loaded_global_numel,
            "loaded_local_parameter_count_per_rank": loaded_local_numel,
            "text_global_parameter_count": text_global_numel,
            "text_local_parameter_count_per_rank": text_local_numel,
            "parameter_counting": (
                "global counts sum each deduplicated model parameter once using its "
                "global shape; local counts sum each rank-local tensor shape without "
                "multiplying by world size; checkpoint count sums each safetensors "
                "entry once"
            ),
            "allocated_after_load_bytes_per_rank": [
                int(values[0]) for values in (row.cpu().tolist() for row in memory_by_rank)
            ],
            "reserved_after_load_bytes_per_rank": [
                int(values[1]) for values in (row.cpu().tolist() for row in memory_by_rank)
            ],
            "config": {
                "hidden_size": int(config.hidden_size),
                "intermediate_size": int(config.intermediate_size),
                "num_hidden_layers": int(config.num_hidden_layers),
                "num_attention_heads": int(config.num_attention_heads),
                "num_key_value_heads": int(config.num_key_value_heads),
                "layer_type_counts": dict(Counter(config.layer_types)),
            },
            "fast_path_packages": {
                "flash-linear-attention": importlib.metadata.version("flash-linear-attention"),
                "causal-conv1d": importlib.metadata.version("causal-conv1d"),
            },
            "fast_path_functions": {
                "causal_conv1d_fn": callable_source(qwen35_modeling.causal_conv1d_fn),
                "causal_conv1d_update": callable_source(qwen35_modeling.causal_conv1d_update),
                "chunk_gated_delta_rule": callable_source(qwen35_modeling.chunk_gated_delta_rule),
                "fused_recurrent_gated_delta_rule": callable_source(
                    qwen35_modeling.fused_recurrent_gated_delta_rule
                ),
            },
            "tp_plan": model._tp_plan if world_size > 1 else None,
            "compiler_state_before": state_before,
            "compiler_state_after": compiler_counters(),
            "timing": "wall_clock_rank_max",
        },
        "records": records,
        "profiles": profiles if dist.get_rank() == 0 else None,
    }
    if dist.get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload["metadata"], indent=2), flush=True)
    dist.barrier(device_ids=[device.index])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
