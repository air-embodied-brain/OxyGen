#!/usr/bin/env python3
"""Qwen3.5 + random PI_v3 end-to-end benchmark with hybrid-cache batching."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_modeling
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import starVLA_qwen25vl_tp_benchmark as q25  # noqa: E402


class Qwen35NativeAdapter:
    def __init__(self, model):
        self.vlm = Qwen35NativeWrapper(model)


class Qwen35NativeWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, **kwargs):
        if (
            kwargs.get("position_ids") is None
            and kwargs.get("past_key_values") is not None
            and kwargs.get("attention_mask") is not None
            and kwargs.get("input_ids") is not None
            and kwargs["input_ids"].shape[-1] < kwargs["attention_mask"].shape[-1]
        ):
            logical_position = kwargs["attention_mask"].long().sum(dim=-1, keepdim=True) - 1
            rope_deltas = self.model.model.rope_deltas
            kwargs["position_ids"] = (logical_position + rope_deltas).unsqueeze(0).expand(3, -1, -1)
        return self.model(**kwargs)


class Qwen35StaticHybridCache(Qwen3_5DynamicCache):
    """Fixed-slot cache for full-attention KV and linear-attention states."""

    is_compileable = True

    def __init__(self, dynamic_cache, max_decode_steps: int):
        self.layer_types = list(dynamic_cache.layer_types)
        self.transformer_layers = list(dynamic_cache.transformer_layers)
        self.last_linear_layer = int(dynamic_cache.last_linear_layer)
        self.prefix_length = int(dynamic_cache.get_seq_length())
        self.max_decode_steps = int(max_decode_steps)
        self.cache_size = self.prefix_length + self.max_decode_steps
        self.write_positions = None
        self.key_cache = []
        self.value_cache = []
        for key, value in zip(dynamic_cache.key_cache, dynamic_cache.value_cache):
            if key is None:
                self.key_cache.append(None)
                self.value_cache.append(None)
                continue
            key_static = key.new_zeros(
                (key.shape[0], key.shape[1], self.cache_size, key.shape[3])
            )
            value_static = value.new_zeros(
                (value.shape[0], value.shape[1], self.cache_size, value.shape[3])
            )
            key_static[:, :, : self.prefix_length].copy_(key)
            value_static[:, :, : self.prefix_length].copy_(value)
            self.key_cache.append(key_static)
            self.value_cache.append(value_static)
        self.conv_states = [None if state is None else state.clone() for state in dynamic_cache.conv_states]
        self.recurrent_states = [
            None if state is None else state.clone() for state in dynamic_cache.recurrent_states
        ]

    @classmethod
    def from_components(
        cls,
        reference,
        key_cache,
        value_cache,
        conv_states,
        recurrent_states,
    ):
        cache = cls.__new__(cls)
        cache.layer_types = list(reference.layer_types)
        cache.transformer_layers = list(reference.transformer_layers)
        cache.last_linear_layer = int(reference.last_linear_layer)
        cache.prefix_length = int(reference.prefix_length)
        cache.max_decode_steps = int(reference.max_decode_steps)
        cache.cache_size = int(reference.cache_size)
        cache.write_positions = None
        cache.key_cache = key_cache
        cache.value_cache = value_cache
        cache.conv_states = conv_states
        cache.recurrent_states = recurrent_states
        return cache

    def clone(self):
        return type(self).from_components(
            self,
            [None if value is None else value.clone() for value in self.key_cache],
            [None if value is None else value.clone() for value in self.value_cache],
            [None if value is None else value.clone() for value in self.conv_states],
            [None if value is None else value.clone() for value in self.recurrent_states],
        )

    @classmethod
    def stack(cls, caches):
        if not caches:
            raise ValueError("Cannot stack an empty hybrid-cache list")
        reference = caches[0]
        for cache in caches:
            if (
                cache.prefix_length != reference.prefix_length
                or cache.max_decode_steps != reference.max_decode_steps
                or cache.layer_types != reference.layer_types
            ):
                raise ValueError("All Qwen3.5 cache rows must use the same layout")

        def stack_optional(values):
            if values[0] is None:
                if any(value is not None for value in values):
                    raise ValueError("Inconsistent optional hybrid-cache state")
                return None
            return torch.cat(values, dim=0)

        return cls.from_components(
            reference,
            [stack_optional([cache.key_cache[i] for cache in caches]) for i in range(len(reference))],
            [stack_optional([cache.value_cache[i] for cache in caches]) for i in range(len(reference))],
            [stack_optional([cache.conv_states[i] for cache in caches]) for i in range(len(reference))],
            [
                stack_optional([cache.recurrent_states[i] for cache in caches])
                for i in range(len(reference))
            ],
        )

    def __len__(self):
        return len(self.layer_types)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        positions = self.write_positions
        if positions is None:
            positions = (cache_kwargs or {}).get("cache_position")
        if positions is None:
            raise ValueError("Qwen35StaticHybridCache requires write positions")
        if positions.ndim == 1:
            positions = positions[None].expand(key_states.shape[0], -1)
        rows = torch.arange(key_states.shape[0], device=key_states.device)
        for token_index in range(positions.shape[1]):
            self.key_cache[layer_idx][rows, :, positions[:, token_index], :] = key_states[
                :, :, token_index, :
            ]
            self.value_cache[layer_idx][rows, :, positions[:, token_index], :] = value_states[
                :, :, token_index, :
            ]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int | None = 0):
        return self.cache_size

    def get_mask_sizes(self, cache_position, layer_idx: int):
        return self.cache_size, 0

    def get_max_cache_shape(self, layer_idx: int = 0):
        return self.cache_size

    def replace_row_from_prefix(self, row: int, prefix_cache):
        fresh = type(self)(prefix_cache, self.max_decode_steps)
        for layer in range(len(self)):
            if self.key_cache[layer] is not None:
                self.key_cache[layer][row].zero_()
                self.value_cache[layer][row].zero_()
                self.key_cache[layer][row, :, : self.prefix_length].copy_(
                    fresh.key_cache[layer][0, :, : self.prefix_length]
                )
                self.value_cache[layer][row, :, : self.prefix_length].copy_(
                    fresh.value_cache[layer][0, :, : self.prefix_length]
                )
            if self.conv_states[layer] is not None:
                self.conv_states[layer][row].copy_(fresh.conv_states[layer][0])
                self.recurrent_states[layer][row].copy_(fresh.recurrent_states[layer][0])


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
    parser.add_argument("--random-init", action="store_true", help="Instantiate from config without trained parameters.")
    parser.add_argument("--decode-steps", type=int_list, default=[10, 20, 30])
    parser.add_argument("--steps-per-frame", type=int_list, default=[1, 5, 10])
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


def staticize(state, max_decode_steps):
    if state.generated_ids:
        raise ValueError("Static conversion must happen before incremental decode")
    state.cache = Qwen35StaticHybridCache(state.cache, max_decode_steps)


def build_age_snapshots(adapter, processor, prefix, max_age, capacity):
    state = q25.init_language_from_prefix(processor, prefix, stop_on_eos=False)
    staticize(state, capacity)
    caches = [state.cache.clone()]
    next_tokens = [state.next_token.clone()]
    rope_deltas = [state.rope_deltas.clone()]
    generated = [[]]
    for _ in range(max_age):
        q25.language_step(adapter, state)
        caches.append(state.cache.clone())
        next_tokens.append(state.next_token.clone())
        rope_deltas.append(state.rope_deltas.clone())
        generated.append(list(state.generated_ids))
    return caches, next_tokens, rope_deltas, generated


@torch.inference_mode()
def persistent_decode_steps(adapter, cache, next_tokens, rope_deltas, ages, steps):
    prefix = cache.prefix_length
    capacity = cache.max_decode_steps
    token_rows = []
    for _ in range(steps):
        token_rows.append(next_tokens.clone())
        logical = torch.tensor([prefix + age for age in ages], device=next_tokens.device)[:, None]
        mask = torch.zeros(
            (len(ages), prefix + capacity), device=next_tokens.device, dtype=torch.bool
        )
        mask[:, :prefix] = True
        for index, age in enumerate(ages):
            mask[index, prefix : prefix + age + 1] = True
        position_ids = (logical + rope_deltas).unsqueeze(0).expand(3, -1, -1)
        cache.write_positions = logical
        outputs = adapter.vlm(
            input_ids=next_tokens,
            attention_mask=mask[:, None, None, :],
            position_ids=position_ids,
            past_key_values=cache,
            # Must be positive so Qwen3.5 uses cached conv/recurrent states.
            # Full-attention physical writes use cache.write_positions above.
            cache_position=torch.tensor([prefix], device=next_tokens.device),
            use_cache=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        next_tokens = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        ages = [age + 1 for age in ages]
    return cache, next_tokens, ages, torch.stack(token_rows, dim=1)


def cache_row_errors(batch_cache, row, reference):
    errors = {"key": 0.0, "value": 0.0, "conv": 0.0, "recurrent": 0.0}
    for layer in range(len(reference)):
        for name, batched, expected in (
            ("key", batch_cache.key_cache[layer], reference.key_cache[layer]),
            ("value", batch_cache.value_cache[layer], reference.value_cache[layer]),
            ("conv", batch_cache.conv_states[layer], reference.conv_states[layer]),
            ("recurrent", batch_cache.recurrent_states[layer], reference.recurrent_states[layer]),
        ):
            if expected is None:
                continue
            actual = batched[row : row + 1]
            if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
                errors[name] = float("inf")
            else:
                errors[name] = max(errors[name], float((actual.float() - expected.float()).abs().max()))
    return errors


def cache_row_diagnostics(batch_cache, row, reference):
    diagnostics = {}
    for name, batched_values, expected_values in (
        ("key", batch_cache.key_cache, reference.key_cache),
        ("value", batch_cache.value_cache, reference.value_cache),
        ("conv", batch_cache.conv_states, reference.conv_states),
        ("recurrent", batch_cache.recurrent_states, reference.recurrent_states),
    ):
        layers = []
        for layer, (batched, expected) in enumerate(zip(batched_values, expected_values)):
            if expected is None:
                continue
            actual = batched[row : row + 1].float()
            target = expected.float()
            difference = actual - target
            target_norm = float(torch.linalg.vector_norm(target))
            layers.append({
                "layer": layer,
                "max_abs_error": float(difference.abs().max()),
                "target_max_abs": float(target.abs().max()),
                "relative_l2_error": float(torch.linalg.vector_norm(difference))
                / max(target_norm, 1e-12),
            })
        diagnostics[name] = layers
    return diagnostics


@torch.inference_mode()
def mixed_age_cache_audit(model, adapter, processor, inputs):
    prefix = q25.prefill(model, inputs)
    ages = [0, 1, 3, 5, 8]
    rollout_steps = 30
    capacity = max(ages) + rollout_steps + 2
    caches, next_tokens, rope_deltas, generated = build_age_snapshots(
        adapter, processor, prefix, max(ages) + rollout_steps, capacity
    )
    cache = Qwen35StaticHybridCache.stack([caches[age] for age in ages])
    prestep_state_errors = [
        cache_row_errors(cache, row, caches[age]) for row, age in enumerate(ages)
    ]
    batch_next = torch.cat([next_tokens[age] for age in ages], dim=0)
    batch_rope = torch.cat([rope_deltas[age] for age in ages], dim=0)
    expected_input = [int(next_tokens[age][0, 0]) for age in ages]
    cache, after_next, after_ages, token_rows = persistent_decode_steps(
        adapter, cache, batch_next, batch_rope, list(ages), 1
    )
    observed_input = token_rows[:, 0, 0].detach().cpu().tolist()
    expected_next = [int(next_tokens[age + 1][0, 0]) for age in ages]
    observed_next = after_next[:, 0].detach().cpu().tolist()
    state_errors = [cache_row_errors(cache, row, caches[age + 1]) for row, age in enumerate(ages)]
    state_diagnostics = [
        cache_row_diagnostics(cache, row, caches[age + 1]) for row, age in enumerate(ages)
    ]
    same_age_controls = []
    mixed_vs_same_age_errors = []
    for row, age in enumerate(ages):
        control_cache = Qwen35StaticHybridCache.stack(
            [caches[age].clone() for _ in ages]
        )
        control_next = next_tokens[age].expand(len(ages), -1).clone()
        control_rope = rope_deltas[age].expand(len(ages), -1).clone()
        control_cache, control_after_next, _, control_token_rows = persistent_decode_steps(
            adapter,
            control_cache,
            control_next,
            control_rope,
            [age for _ in ages],
            1,
        )
        native_errors = cache_row_errors(control_cache, 0, caches[age + 1])
        native_diagnostics = cache_row_diagnostics(control_cache, 0, caches[age + 1])
        mixed_vs_control = cache_row_errors(cache, row, Qwen35StaticHybridCache.from_components(
            control_cache,
            [None if value is None else value[0:1] for value in control_cache.key_cache],
            [None if value is None else value[0:1] for value in control_cache.value_cache],
            [None if value is None else value[0:1] for value in control_cache.conv_states],
            [None if value is None else value[0:1] for value in control_cache.recurrent_states],
        ))
        mixed_vs_same_age_errors.append(mixed_vs_control)
        same_age_controls.append({
            "age": age,
            "input_token": int(control_token_rows[0, 0, 0]),
            "next_token": int(control_after_next[0, 0]),
            "native_batch_vs_singleton_max_abs_error": native_errors,
            "native_batch_vs_singleton_diagnostics": native_diagnostics,
            "mixed_age_vs_same_age_batch_max_abs_error": mixed_vs_control,
        })

    rollout_cache = Qwen35StaticHybridCache.stack([caches[age] for age in ages])
    rollout_next = torch.cat([next_tokens[age] for age in ages], dim=0)
    rollout_rope = torch.cat([rope_deltas[age] for age in ages], dim=0)
    rollout_cache, rollout_after_next, rollout_after_ages, rollout_token_rows = persistent_decode_steps(
        adapter,
        rollout_cache,
        rollout_next,
        rollout_rope,
        list(ages),
        rollout_steps,
    )
    observed_rollout_tokens = rollout_token_rows[:, :, 0].detach().cpu().tolist()
    expected_rollout_tokens = [
        [int(next_tokens[age + offset][0, 0]) for offset in range(rollout_steps)]
        for age in ages
    ]
    observed_rollout_next = rollout_after_next[:, 0].detach().cpu().tolist()
    expected_rollout_next = [int(next_tokens[age + rollout_steps][0, 0]) for age in ages]
    same_age_rollout_controls = []
    mixed_vs_same_age_rollout_errors = []
    mixed_vs_same_age_rollout_tokens_exact = []
    for row, age in enumerate(ages):
        control_cache = Qwen35StaticHybridCache.stack(
            [caches[age].clone() for _ in ages]
        )
        control_next = next_tokens[age].expand(len(ages), -1).clone()
        control_rope = rope_deltas[age].expand(len(ages), -1).clone()
        control_cache, control_after_next, _, control_token_rows = persistent_decode_steps(
            adapter,
            control_cache,
            control_next,
            control_rope,
            [age for _ in ages],
            rollout_steps,
        )
        control_row = Qwen35StaticHybridCache.from_components(
            control_cache,
            [None if value is None else value[0:1] for value in control_cache.key_cache],
            [None if value is None else value[0:1] for value in control_cache.value_cache],
            [None if value is None else value[0:1] for value in control_cache.conv_states],
            [None if value is None else value[0:1] for value in control_cache.recurrent_states],
        )
        state_error = cache_row_errors(rollout_cache, row, control_row)
        control_tokens = control_token_rows[0, :, 0].detach().cpu().tolist()
        tokens_exact = observed_rollout_tokens[row] == control_tokens
        mixed_vs_same_age_rollout_errors.append(state_error)
        mixed_vs_same_age_rollout_tokens_exact.append(tokens_exact)
        same_age_rollout_controls.append({
            "age": age,
            "tokens": control_tokens,
            "next_token": int(control_after_next[0, 0]),
            "mixed_age_tokens_exact": tokens_exact,
            "mixed_age_final_state_max_abs_error": state_error,
        })
    mixed_vs_same_age_max_errors = {
        name: max(row[name] for row in mixed_vs_same_age_errors)
        for name in mixed_vs_same_age_errors[0]
    }
    recycle_row = 2
    recycle_cache = Qwen35StaticHybridCache.stack([caches[age] for age in ages])
    before_recycle = recycle_cache.clone()
    fresh_state = q25.init_language_from_prefix(processor, prefix, stop_on_eos=False)
    fresh_cache = Qwen35StaticHybridCache(fresh_state.cache, capacity)
    recycle_cache.replace_row_from_prefix(recycle_row, fresh_state.cache)
    recycled_row_errors = cache_row_errors(recycle_cache, recycle_row, fresh_cache)
    untouched_row_errors = []
    for row in range(len(ages)):
        if row == recycle_row:
            continue
        before_row = Qwen35StaticHybridCache.from_components(
            before_recycle,
            [None if value is None else value[row : row + 1] for value in before_recycle.key_cache],
            [None if value is None else value[row : row + 1] for value in before_recycle.value_cache],
            [None if value is None else value[row : row + 1] for value in before_recycle.conv_states],
            [None if value is None else value[row : row + 1] for value in before_recycle.recurrent_states],
        )
        untouched_row_errors.append(cache_row_errors(recycle_cache, row, before_row))
    untouched_max_errors = {
        name: max(row[name] for row in untouched_row_errors) for name in untouched_row_errors[0]
    }
    max_errors = {
        name: max(row[name] for row in state_errors) for name in state_errors[0]
    }
    return {
        "ages": ages,
        "expected_input_tokens": expected_input,
        "observed_input_tokens": observed_input,
        "expected_next_tokens": expected_next,
        "observed_next_tokens": observed_next,
        "input_tokens_exact": observed_input == expected_input,
        "next_tokens_exact": observed_next == expected_next,
        "state_max_abs_error": max_errors,
        "prestep_state_max_abs_error": {
            name: max(row[name] for row in prestep_state_errors) for name in prestep_state_errors[0]
        },
        "state_diagnostics_by_row": state_diagnostics,
        "full_attention_state_exact": max_errors["key"] == 0.0 and max_errors["value"] == 0.0,
        "linear_attention_state_close": max_errors["conv"] <= 1e-5
        and max_errors["recurrent"] <= 1e-5,
        "same_age_replicated_controls": same_age_controls,
        "mixed_age_vs_same_age_batch_max_abs_error": mixed_vs_same_age_max_errors,
        "mixed_age_vs_same_age_batch_state_exact": all(
            value == 0.0 for value in mixed_vs_same_age_max_errors.values()
        ),
        "slot_recycle_isolation": {
            "recycled_row": recycle_row,
            "recycled_row_vs_fresh_prefix_max_abs_error": recycled_row_errors,
            "untouched_rows_max_abs_error": untouched_max_errors,
            "recycled_row_exact": all(value == 0.0 for value in recycled_row_errors.values()),
            "untouched_rows_exact": all(value == 0.0 for value in untouched_max_errors.values()),
        },
        "rollout_steps": rollout_steps,
        "rollout_expected_tokens": expected_rollout_tokens,
        "rollout_observed_tokens": observed_rollout_tokens,
        "rollout_tokens_exact": observed_rollout_tokens == expected_rollout_tokens,
        "rollout_expected_next_tokens": expected_rollout_next,
        "rollout_observed_next_tokens": observed_rollout_next,
        "rollout_next_tokens_exact": observed_rollout_next == expected_rollout_next,
        "rollout_expected_text": [
            processor.decode(row, skip_special_tokens=True) for row in expected_rollout_tokens
        ],
        "rollout_observed_text": [
            processor.decode(row, skip_special_tokens=True) for row in observed_rollout_tokens
        ],
        "rollout_text_exact": observed_rollout_tokens == expected_rollout_tokens,
        "same_age_replicated_rollout_controls": same_age_rollout_controls,
        "mixed_age_vs_same_age_rollout_tokens_exact": all(
            mixed_vs_same_age_rollout_tokens_exact
        ),
        "mixed_age_vs_same_age_rollout_state_max_abs_error": {
            name: max(row[name] for row in mixed_vs_same_age_rollout_errors)
            for name in mixed_vs_same_age_rollout_errors[0]
        },
        "mixed_age_vs_same_age_rollout_state_exact": all(
            value == 0.0
            for value in {
                name: max(row[name] for row in mixed_vs_same_age_rollout_errors)
                for name in mixed_vs_same_age_rollout_errors[0]
            }.values()
        ),
        "rollout_after_ages": rollout_after_ages,
        "after_ages": after_ages,
        "reference_text": processor.decode(generated[-1], skip_special_tokens=True),
    }


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
    from analyze_qwen35_scaling_tp import validate_cache_implementation
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
