"""Language batching with independent positions for each request."""
from __future__ import annotations
import torch
from oxygen_models.common.cache import StaticLanguageCache

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
