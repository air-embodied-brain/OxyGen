"""Qwen3.5 batching for full-attention KV and recurrent linear-attention state."""
from __future__ import annotations
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
from oxygen_models.qwen35 import helpers as q25

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
