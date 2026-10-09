#!/usr/bin/env python3
"""TP=1/2 end-to-end benchmark for PI_v3 action and language experts."""

from __future__ import annotations

import math
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile

from oxygen_models.common.cache import (  # noqa: E402
    PrefixState,
    _snapshot_cache,
    init_language_from_prefix,
    language_step,
)
from starVLA.model.framework.VLM4A.QwenPI_v3 import QwenPI_v3DefaultConfig  # noqa: E402
from starVLA.model.framework.share_tools import (  # noqa: E402
    merge_framework_config,
    populate_layerwise_dit_cfg,
)
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (  # noqa: E402
    get_action_model,
)


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
        _action = action_head.predict(prefix)
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
        _action = action_head.predict(prefix)
        torch.cuda.synchronize(device)
        local.append((time.perf_counter() - action_started) * 1000)
        decode_started = time.perf_counter()
        language = run_language(adapter, processor, prefix, decode_steps)  # noqa: F841 -- retain output through timed stage
        torch.cuda.synchronize(device)
        local.append((time.perf_counter() - decode_started) * 1000)
        names = ["prefill_ms", "action_denoise_ms", "language_decode_ms"]
    else:
        raise ValueError(mode)
    values = _rank_max(local, device)
    return {name: value for name, value in zip(names, values)}



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
