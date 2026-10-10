#!/usr/bin/env python3
"""OxyGen-style heterogeneous AR-language + PI_v3 flow-action sweep."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

XIAOMI_SCRIPTS = str(Path(__file__).resolve().parent)
if XIAOMI_SCRIPTS not in sys.path:
    sys.path.insert(0, XIAOMI_SCRIPTS)

from oxygen_runtime import (  # noqa: E402
    PrefixState,
    StaticLanguageCache,
    _snapshot_cache,
    init_language_from_prefix,
    language_step,
    staticize_language_state,
)
from starVLA.model.framework.base_framework import baseframework  # noqa: E402
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3  # noqa: E402


class RuntimeAdapter:
    def __init__(self, interface):
        self.vlm = interface
        self.device = interface.model.device


def int_list(value):
    return [int(item) for item in value.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--pi-checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoise-steps", type=int_list, default=[10])
    parser.add_argument("--max-decoding-steps", type=int_list, default=[5, 10, 15, 20, 30])
    parser.add_argument("--steps-per-frame", type=int_list, default=[1, 5, 10])
    parser.add_argument("--measured-repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-frames", type=int, default=50)
    parser.add_argument("--modes", nargs="+", default=["baseline", "shared_kv", "continuous_batching"])
    parser.add_argument(
        "--continuous-runtime",
        choices=["legacy", "persistent"],
        default="persistent",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_example():
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((22, 66, 86, 132), fill=(190, 35, 35))
    draw.rectangle((134, 62, 207, 145), fill=(35, 70, 180))
    return {
        "image": [image],
        "lang": "Move the red block next to the blue block.",
        "state": np.zeros((1, 7), dtype=np.float32),
    }


def random_pi_config(base_vlm):
    return OmegaConf.create({
        "framework": {
            "name": "QwenPI_v3",
            "qwenvl": {"base_vlm": base_vlm, "attn_implementation": "sdpa"},
            "action_model": {
                "action_model_type": "LayerwiseFM",
                "action_dim": 7,
                "state_dim": 7,
                "action_horizon": 16,
                "num_inference_timesteps": 10,
                "diffusion_model_cfg": {"action_dit_hidden_dim": 1024},
            },
        },
        "datasets": {"vla_data": {"obs_image_size": [224, 224]}},
        "trainer": {"pretrained_checkpoint": None},
    })


def load_model(args):
    if args.pi_checkpoint:
        model = baseframework.from_pretrained(
            args.pi_checkpoint,
            config_overrides=[
                f"framework.qwenvl.base_vlm={args.base_vlm}",
                "framework.qwenvl.attn_implementation=sdpa",
            ],
        )
        semantics = "trained_qwen3vl_pi_v3"
    else:
        model = Qwen_PI_v3(random_pi_config(args.base_vlm))
        semantics = "base_vlm_with_random_pi_v3_head"
    model = model.to(torch.bfloat16).to(args.device).eval()
    return model, semantics


def build_common_inputs(model, example):
    # Bypass the checkpoint's action-only CoT/state suffix. Both consumers see
    # exactly the same observation + task chat prompt.
    interface = model.qwen_vl_interface
    return interface.build_qwenvl_inputs(
        images=[example["image"]], instructions=[example["lang"]]
    )


@torch.inference_mode()
def prefill(model, inputs):
    outputs = model.qwen_vl_interface(
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
def run_action_from_prefix(model, prefix, denoise_steps, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.action_model.num_inference_timesteps = denoise_steps
    hidden = list(prefix.outputs.hidden_states[-model.num_action_dit_layers:])
    hidden = model._project_vl_hidden_for_action(hidden)
    mask = prefix.outputs.attention_mask.to(dtype=torch.bool)
    with torch.autocast("cuda", dtype=torch.float32):
        return model.action_model.predict_action(
            hidden, state=None, encoder_attention_mask=mask
        )


def run_language(adapter, processor, prefix, max_decode):
    state = init_language_from_prefix(processor, prefix, stop_on_eos=False)
    for _ in range(max_decode):
        language_step(adapter, state)
    return state


@torch.inference_mode()
def batched_language_steps(adapter, states, steps):
    """Batch different request ages using right-aligned physical KV slots.

    Qwen2.5-VL requires a one-dimensional cache_position shared by the batch.
    Keys retain their logical RoPE positions when moved between physical cache
    slots, so histories can be right-aligned before each batched dispatch.
    """
    if not states:
        return
    if any(not isinstance(state.cache, StaticLanguageCache) for state in states):
        raise TypeError("All requests must use StaticLanguageCache")
    reference = states[0].cache
    prefix = reference.prefix_length
    capacity = reference.max_decode_steps
    ages = [len(state.generated_ids) for state in states]
    max_age = max(ages)
    common_position = prefix + max_age
    key_cache = []
    value_cache = []
    for layer in range(len(reference.key_cache)):
        key = torch.cat([
            state.cache.key_cache[layer].new_zeros(state.cache.key_cache[layer].shape)
            for state in states
        ], dim=0)
        value = torch.cat([
            state.cache.value_cache[layer].new_zeros(state.cache.value_cache[layer].shape)
            for state in states
        ], dim=0)
        for index, (state, age) in enumerate(zip(states, ages)):
            key[index, :, :prefix] = state.cache.key_cache[layer][0, :, :prefix]
            value[index, :, :prefix] = state.cache.value_cache[layer][0, :, :prefix]
            if age:
                start = common_position - age
                key[index, :, start:common_position] = state.cache.key_cache[layer][0, :, prefix:prefix + age]
                value[index, :, start:common_position] = state.cache.value_cache[layer][0, :, prefix:prefix + age]
        key_cache.append(key)
        value_cache.append(value)
    cache = StaticLanguageCache.from_tensors(key_cache, value_cache, prefix, capacity)
    next_tokens = torch.cat([state.next_token for state in states], dim=0)
    rope_deltas = torch.cat([state.rope_deltas for state in states], dim=0)

    for _ in range(steps):
        ages = [len(state.generated_ids) for state in states]
        active = [age < capacity for age in ages]
        if not any(active):
            break
        input_tokens = next_tokens.clone()
        for index, state in enumerate(states):
            if active[index]:
                state.generated_ids.append(int(input_tokens[index, 0]))
        attention_mask = torch.zeros(
            (len(states), prefix + capacity), device=next_tokens.device, dtype=torch.bool
        )
        attention_mask[:, :prefix] = True
        for index, age in enumerate(ages):
            if age:
                attention_mask[index, common_position - age:common_position] = True
            attention_mask[index, common_position] = True
        logical_positions = torch.tensor(
            [prefix + age for age in ages], device=next_tokens.device, dtype=torch.long
        )[:, None]
        position_ids = (logical_positions + rope_deltas).unsqueeze(0).expand(3, -1, -1)
        outputs = adapter.vlm(
            input_ids=input_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=torch.tensor([common_position], device=next_tokens.device),
            use_cache=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        next_tokens = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        common_position += 1

    final_ages = [len(state.generated_ids) for state in states]
    for index, (state, age) in enumerate(zip(states, final_ages)):
        canonical_keys = []
        canonical_values = []
        for layer in range(len(cache.key_cache)):
            key = cache.key_cache[layer][index:index + 1].new_zeros(
                cache.key_cache[layer][index:index + 1].shape
            )
            value = cache.value_cache[layer][index:index + 1].new_zeros(
                cache.value_cache[layer][index:index + 1].shape
            )
            key[:, :, :prefix] = cache.key_cache[layer][index:index + 1, :, :prefix]
            value[:, :, :prefix] = cache.value_cache[layer][index:index + 1, :, :prefix]
            if age:
                key[:, :, prefix:prefix + age] = cache.key_cache[layer][
                    index:index + 1, :, common_position - age:common_position
                ]
                value[:, :, prefix:prefix + age] = cache.value_cache[layer][
                    index:index + 1, :, common_position - age:common_position
                ]
            canonical_keys.append(key)
            canonical_values.append(value)
        state.cache = StaticLanguageCache.from_tensors(
            canonical_keys, canonical_values, prefix, capacity
        )
        state.next_token = next_tokens[index:index + 1].clone()
        state.finished = age >= capacity


def sync_measure(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = fn()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000, torch.cuda.max_memory_allocated()


def measure_single(model, adapter, inputs, mode, denoise, max_decode):
    def work():
        if mode == "baseline":
            action_prefix = prefill(model, inputs)
            action = run_action_from_prefix(model, action_prefix, denoise)
            language_prefix = prefill(model, inputs)
        else:
            language_prefix = prefill(model, inputs)
            action = run_action_from_prefix(model, language_prefix, denoise)
        language = run_language(adapter, model.qwen_vl_interface.processor, language_prefix, max_decode)
        return action, language
    (action, language), latency_ms, peak = sync_measure(work)
    return {
        "mode": mode,
        "denoise_steps": denoise,
        "max_decoding_steps": max_decode,
        "steps_per_frame": None,
        "mean_frame_ms": latency_ms,
        "action_checksum": float(action.float().sum()),
        "action_shape": list(action.shape),
        "action_finite": bool(torch.isfinite(action).all()),
        "language_token_ids": language.generated_ids,
        "language_text": model.qwen_vl_interface.processor.batch_decode(
            [language.generated_ids], skip_special_tokens=True
        )[0],
        "peak_allocated_bytes": peak,
        "language_throughput_tokens_s": max_decode / (latency_ms / 1000),
        "action_frequency_hz": action.shape[-2] / (latency_ms / 1000),
    }


def continuous_simulation(
    model, adapter, inputs, denoise, max_decode, steps_per_frame,
    warmup_frames, measured_frames, collect,
):
    processor = model.qwen_vl_interface.processor
    active = {}
    frame_ms = []
    frame_tokens = []
    frame_batches = []
    completed = []
    action = None
    for frame in range(warmup_frames + measured_frames):
        torch.cuda.synchronize()
        started = time.perf_counter()
        prefix = prefill(model, inputs)
        action = run_action_from_prefix(model, prefix, denoise)
        language = init_language_from_prefix(processor, prefix, stop_on_eos=False)
        staticize_language_state(language, max_decode)
        active[f"req_{frame}"] = language
        request_ids = list(active)
        states = [active[item] for item in request_ids]
        before = sum(len(item.generated_ids) for item in states)
        batched_language_steps(adapter, states, steps_per_frame)
        generated = sum(len(item.generated_ids) for item in states) - before
        for request_id in request_ids:
            if active[request_id].finished:
                completed.append(active.pop(request_id).generated_ids)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        if collect and frame >= warmup_frames:
            frame_ms.append(elapsed_ms)
            frame_tokens.append(generated)
            frame_batches.append(len(states))
    return frame_ms, frame_tokens, frame_batches, completed, action


def measure_continuous(model, adapter, inputs, denoise, max_decode, steps_per_frame, measured_frames):
    warmup_frames = math.ceil(max_decode / steps_per_frame)
    continuous_simulation(
        model, adapter, inputs, denoise, max_decode, steps_per_frame,
        warmup_frames, measured_frames, False,
    )
    torch.cuda.reset_peak_memory_stats()
    frame_ms, tokens, batches, completed, action = continuous_simulation(
        model, adapter, inputs, denoise, max_decode, steps_per_frame,
        warmup_frames, measured_frames, True,
    )
    seconds = sum(frame_ms) / 1000
    return {
        "mode": "continuous_batching",
        "denoise_steps": denoise,
        "max_decoding_steps": max_decode,
        "steps_per_frame": steps_per_frame,
        "mean_frame_ms": statistics.fmean(frame_ms),
        "p50_frame_ms": statistics.median(frame_ms),
        "avg_batch_size": statistics.fmean(batches),
        "action_checksum": float(action.float().sum()),
        "action_shape": list(action.shape),
        "action_finite": bool(torch.isfinite(action).all()),
        "completed_requests": len(completed),
        "completed_text_sample": (
            model.qwen_vl_interface.processor.batch_decode(
                [completed[-1]], skip_special_tokens=True
            )[0] if completed else ""
        ),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "language_throughput_tokens_s": sum(tokens) / seconds,
        "action_frequency_hz": action.shape[-2] * len(frame_ms) / seconds,
    }


class PerRequestStaticLanguageCache(StaticLanguageCache):
    write_positions = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        positions = self.write_positions.reshape(-1)
        rows = torch.arange(key_states.shape[0], device=key_states.device)
        self.key_cache[layer_idx][rows, :, positions, :] = key_states[:, :, 0, :]
        self.value_cache[layer_idx][rows, :, positions, :] = value_states[:, :, 0, :]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


@torch.inference_mode()
def persistent_decode_steps(adapter, cache, next_tokens, rope_deltas, ages, steps):
    prefix = cache.prefix_length
    capacity = cache.max_decode_steps
    token_rows = []
    for _ in range(steps):
        token_rows.append(next_tokens.clone())
        logical = torch.tensor(
            [prefix + age for age in ages], device=next_tokens.device
        )[:, None]
        mask = torch.zeros(
            (len(ages), prefix + capacity), device=next_tokens.device, dtype=torch.bool
        )
        mask[:, :prefix] = True
        for index, age in enumerate(ages):
            mask[index, prefix:prefix + age + 1] = True
        positions = (logical + rope_deltas).unsqueeze(0).expand(3, -1, -1)
        cache.write_positions = logical
        outputs = adapter.vlm(
            input_ids=next_tokens,
            attention_mask=mask[:, None, None, :],
            position_ids=positions,
            past_key_values=cache,
            cache_position=torch.tensor([0], device=next_tokens.device),
            use_cache=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        next_tokens = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        ages = [age + 1 for age in ages]
    return cache, next_tokens, ages, torch.stack(token_rows, dim=1)


@torch.inference_mode()
def measure_persistent_continuous(
    model, adapter, inputs, denoise, max_decode, steps_per_frame, measured_frames
):
    processor = model.qwen_vl_interface.processor
    batch = max_decode // steps_per_frame
    bootstrap_prefix = prefill(model, inputs)
    ages = [index * steps_per_frame for index in range(batch)]
    template = init_language_from_prefix(processor, bootstrap_prefix, stop_on_eos=False)
    staticize_language_state(template, max_decode)
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
    free_slots = [index for index, age in enumerate(ages) if age >= max_decode]
    if len(free_slots) != 1:
        raise RuntimeError(f"expected one free steady-state slot, got {free_slots}")
    free_slot = free_slots[0]

    frame_times = []
    completed_tokens = []
    completed_texts = []
    action = None
    prefix_length = cache.prefix_length
    torch.cuda.reset_peak_memory_stats()
    for _ in range(measured_frames):
        torch.cuda.synchronize()
        started = time.perf_counter()
        frame_prefix = prefill(model, inputs)
        action = run_action_from_prefix(model, frame_prefix, denoise)
        new_state = init_language_from_prefix(processor, frame_prefix, stop_on_eos=False)
        staticize_language_state(new_state, max_decode)
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
        cache, next_tokens, ages, token_rows = persistent_decode_steps(
            adapter, cache, next_tokens, rope_deltas, ages, steps_per_frame
        )
        torch.cuda.synchronize()
        frame_times.append((time.perf_counter() - started) * 1000)
        for index, values in enumerate(token_rows[:, :, 0].detach().cpu().tolist()):
            generated[index].extend(values)
        free_slots = [index for index, age in enumerate(ages) if age >= max_decode]
        if len(free_slots) != 1:
            raise RuntimeError(f"expected one completed request, got {free_slots}")
        free_slot = free_slots[0]
        completed_tokens.append(list(generated[free_slot]))
        completed_texts.append(
            processor.decode(generated[free_slot], skip_special_tokens=True)
        )

    mean_ms = statistics.fmean(frame_times)
    return {
        "mode": "continuous_batching",
        "continuous_runtime": "persistent",
        "denoise_steps": denoise,
        "max_decoding_steps": max_decode,
        "steps_per_frame": steps_per_frame,
        "mean_frame_ms": mean_ms,
        "p50_frame_ms": statistics.median(frame_times),
        "frame_times_ms": frame_times,
        "avg_batch_size": batch,
        "action_checksum": float(action.float().sum()),
        "action_shape": list(action.shape),
        "action_finite": bool(torch.isfinite(action).all()),
        "completed_requests": len(completed_tokens),
        "all_completed_lengths_correct": all(
            len(tokens) == max_decode for tokens in completed_tokens
        ),
        "all_completed_text_nonempty": all(text.strip() for text in completed_texts),
        "completed_text_sample": completed_texts[-1] if completed_texts else "",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "language_throughput_tokens_s": max_decode / (mean_ms / 1000),
        "action_frequency_hz": action.shape[-2] / (mean_ms / 1000),
    }


def release_cuda_temporaries():
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


def main():
    args = parse_args()
    model, semantics = load_model(args)
    adapter = RuntimeAdapter(model.qwen_vl_interface)
    inputs = build_common_inputs(model, make_example())
    rows = []
    for denoise, max_decode in itertools.product(args.denoise_steps, args.max_decoding_steps):
        gc.collect()
        torch.cuda.empty_cache()
        single_modes = [item for item in args.modes if item != "continuous_batching"]
        for mode in single_modes:
            for _ in range(args.warmup_runs):
                measure_single(model, adapter, inputs, mode, denoise, max_decode)
        for repeat in range(args.measured_repeats):
            order = single_modes[repeat % len(single_modes):] + single_modes[:repeat % len(single_modes)]
            for mode in order:
                row = measure_single(model, adapter, inputs, mode, denoise, max_decode)
                row["repeat"] = repeat
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        if "continuous_batching" in args.modes:
            for steps in args.steps_per_frame:
                if max_decode % steps:
                    continue
                for repeat in range(args.measured_repeats):
                    measure = (
                        measure_persistent_continuous
                        if args.continuous_runtime == "persistent"
                        else measure_continuous
                    )
                    row = measure(
                        model, adapter, inputs, denoise, max_decode,
                        steps, args.measured_frames,
                    )
                    row["repeat"] = repeat
                    row["memory_after_repeat"] = release_cuda_temporaries()
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = {
        "metadata": {
            "base_vlm": args.base_vlm,
            "pi_checkpoint": args.pi_checkpoint,
            "checkpoint_semantics": semantics,
            "prompt_mode": "common_only_observation_plus_task",
            "prefix_tokens": int(inputs["input_ids"].shape[-1]),
            "attention": "sdpa",
            "dtype": "bfloat16",
            "compile": False,
            "cuda_graph": False,
            "measured_frames": args.measured_frames,
            "measured_repeats": args.measured_repeats,
            "continuous_runtime": args.continuous_runtime,
            "inference_mode": True,
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
