"""Persistent language slots and action dispatch for heterogeneous experts."""
from __future__ import annotations
import statistics
import torch
import time
from oxygen_models.common.cache import StaticLanguageCache, init_language_from_prefix, staticize_language_state, language_step
from oxygen_models.starvla.experts import RuntimeAdapter, prefill, run_language, pi_action, groot_action, oft_action

def run_head(name, pi, groot, oft, prefix, inputs, language_tokens, denoise_steps):
    if name == "language":
        return run_language(RuntimeAdapter(pi.qwen_vl_interface), pi.qwen_vl_interface.processor, prefix, language_tokens)
    if name == "pi_v3":
        return pi_action(pi, prefix, denoise_steps)
    if name == "groot":
        return groot_action(groot, prefix, denoise_steps)
    if name == "oft":
        return oft_action(oft, prefix, inputs["input_ids"])
    raise ValueError(name)



class PerRequestStaticLanguageCache(StaticLanguageCache):
    write_positions = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        positions = self.write_positions.reshape(-1)
        rows = torch.arange(key_states.shape[0], device=key_states.device)
        self.key_cache[layer_idx][rows, :, positions, :] = key_states[:, :, 0, :]
        self.value_cache[layer_idx][rows, :, positions, :] = value_states[:, :, 0, :]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]



def persistent_cache_from_states(states):
    reference = states[0].cache
    return PerRequestStaticLanguageCache.from_tensors(
        [torch.cat([state.cache.key_cache[layer] for state in states]) for layer in range(len(reference))],
        [torch.cat([state.cache.value_cache[layer] for state in states]) for layer in range(len(reference))],
        reference.prefix_length,
        reference.max_decode_steps,
    )



@torch.inference_mode()
def persistent_decode_steps(adapter, cache, next_tokens, rope_deltas, ages, steps):
    prefix = cache.prefix_length
    capacity = cache.max_decode_steps
    token_rows = []
    for _ in range(steps):
        token_rows.append(next_tokens.clone())
        logical = torch.tensor([prefix + age for age in ages], device=next_tokens.device)[:, None]
        mask = torch.zeros((len(ages), prefix + capacity), device=next_tokens.device, dtype=torch.bool)
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
def persistent_continuous_once(
    pi, groot, oft, inputs, names, language_tokens, denoise_steps,
    steps_per_frame, warmup_frames, measured_frames,
):
    """Steady-state fixed-slot scheduler with canonical, persistent KV caches."""
    adapter = RuntimeAdapter(pi.qwen_vl_interface)
    processor = pi.qwen_vl_interface.processor
    batch = language_tokens // steps_per_frame
    bootstrap_prefix = prefill(pi, inputs)
    ages = [index * steps_per_frame for index in range(batch)]
    template = init_language_from_prefix(processor, bootstrap_prefix, stop_on_eos=False)
    staticize_language_state(template, language_tokens)
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

    # Advance once to establish a free slot before measured steady state.
    cache, next_tokens, ages, token_rows = persistent_decode_steps(
        adapter, cache, next_tokens, rope_deltas, ages, steps_per_frame
    )
    warm_tokens = token_rows[:, :, 0].detach().cpu().tolist()
    for index, values in enumerate(warm_tokens):
        generated[index].extend(values)
    free_slots = [index for index, age in enumerate(ages) if age >= language_tokens]
    if len(free_slots) != 1:
        raise RuntimeError(f"expected one free steady-state slot, got {free_slots}")
    free_slot = free_slots[0]

    frame_times = []
    completed_texts = []
    completed_token_ids = []
    completed_lengths = []
    last_action_outputs = {}
    prefix = cache.prefix_length
    for frame_index in range(warmup_frames + measured_frames):
        torch.cuda.synchronize()
        start = time.perf_counter()
        frame_prefix = prefill(pi, inputs)
        for name in names:
            if name != "language":
                last_action_outputs[name] = run_head(
                    name, pi, groot, oft, frame_prefix, inputs,
                    language_tokens, denoise_steps,
                )
        new_state = init_language_from_prefix(processor, frame_prefix, stop_on_eos=False)
        staticize_language_state(new_state, language_tokens)
        for layer in range(len(cache.key_cache)):
            cache.key_cache[layer][free_slot, :, :prefix].copy_(new_state.cache.key_cache[layer][0, :, :prefix])
            cache.value_cache[layer][free_slot, :, :prefix].copy_(new_state.cache.value_cache[layer][0, :, :prefix])
        next_tokens[free_slot].copy_(new_state.next_token[0])
        rope_deltas[free_slot].copy_(new_state.rope_deltas[0])
        ages[free_slot] = 0
        generated[free_slot] = []
        cache, next_tokens, ages, token_rows = persistent_decode_steps(
            adapter, cache, next_tokens, rope_deltas, ages, steps_per_frame
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        frame_token_ids = token_rows[:, :, 0].detach().cpu().tolist()
        for index, values in enumerate(frame_token_ids):
            generated[index].extend(values)
        free_slots = [index for index, age in enumerate(ages) if age >= language_tokens]
        if len(free_slots) != 1:
            raise RuntimeError(f"expected one completed request, got {free_slots}")
        free_slot = free_slots[0]
        if frame_index >= warmup_frames:
            frame_times.append(elapsed_ms)
            completed_lengths.append(len(generated[free_slot]))
            completed_token_ids.append(list(generated[free_slot]))
            completed_texts.append(processor.decode(generated[free_slot], skip_special_tokens=True))
    return {
        "mean_frame_ms": statistics.fmean(frame_times),
        "p50_frame_ms": statistics.median(frame_times),
        "frame_times_ms": frame_times,
        "avg_batch_size": batch,
        "language_tokens_per_s": language_tokens / (statistics.fmean(frame_times) / 1000.0),
        "scheduler_ramp_up": "prepopulated staggered request ages plus one decode step",
        "unmeasured_warmup_frames": warmup_frames,
        "completed_requests": measured_frames,
        "all_completed_lengths_correct": all(length == language_tokens for length in completed_lengths),
        "all_completed_text_nonempty": all(bool(text.strip()) for text in completed_texts),
        "completed_token_ids": completed_token_ids,
        "completed_texts": completed_texts,
        "completed_text_samples": completed_texts[-min(5, len(completed_texts)):],
        "action_checksums": {
            name: float(output.float().sum().item())
            for name, output in last_action_outputs.items()
        },
        "all_actions_finite": all(
            bool(torch.isfinite(output).all())
            for output in last_action_outputs.values()
        ),
    }
