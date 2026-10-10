"""Incremental language/action runtime for the official Xiaomi model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache


@dataclass
class PrefixState:
    outputs: Any
    action_kv: list[tuple[torch.Tensor, torch.Tensor]]
    prefix_length: int


@dataclass
class ActionState:
    x: torch.Tensor
    action_mask: torch.Tensor
    state_embed: torch.Tensor
    position_embeds: tuple[torch.Tensor, torch.Tensor]
    attention_mask: torch.Tensor
    prefix_kv: list[tuple[torch.Tensor, torch.Tensor]]
    num_steps: int
    step: int = 0

    @property
    def finished(self) -> bool:
        return self.step >= self.num_steps


@dataclass
class LanguageState:
    cache: Any
    attention_mask: torch.Tensor
    next_token: torch.Tensor
    eos_token_ids: set[int]
    rope_deltas: torch.Tensor | None
    stop_on_eos: bool = True
    forced_ids: list[int] | None = None
    forced_index: int = 0
    generated_ids: list[int] = field(default_factory=list)
    finished: bool = False


class StaticLanguageCache(Cache):
    """Fixed-capacity cache with independent write positions per request."""

    is_compileable = True

    def __init__(self, dynamic_cache, max_decode_steps: int):
        super().__init__(layers=[])
        snapshot = _snapshot_cache(dynamic_cache)
        self.prefix_length = snapshot[0][0].shape[2]
        self.max_decode_steps = max_decode_steps
        self.cache_size = self.prefix_length + max_decode_steps
        self.key_cache = []
        self.value_cache = []
        for key, value in snapshot:
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

    @classmethod
    def from_tensors(
        cls, key_cache, value_cache, prefix_length: int, max_decode_steps: int
    ):
        cache = cls.__new__(cls)
        Cache.__init__(cache, layers=[])
        cache.prefix_length = prefix_length
        cache.max_decode_steps = max_decode_steps
        cache.cache_size = prefix_length + max_decode_steps
        cache.key_cache = key_cache
        cache.value_cache = value_cache
        return cache

    def __len__(self):
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_position = (cache_kwargs or {}).get("cache_position")
        if cache_position is None:
            raise ValueError("StaticLanguageCache requires cache_position")
        if cache_position.ndim == 1:
            cache_position = cache_position[None].expand(key_states.shape[0], -1)
        batch_indices = torch.arange(key_states.shape[0], device=key_states.device)
        for seq_idx in range(cache_position.shape[1]):
            positions = cache_position[:, seq_idx]
            self.key_cache[layer_idx][batch_indices, :, positions, :] = key_states[
                :, :, seq_idx, :
            ]
            self.value_cache[layer_idx][batch_indices, :, positions, :] = value_states[
                :, :, seq_idx, :
            ]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0):
        return self.cache_size

    def get_mask_sizes(self, cache_position, layer_idx: int):
        return self.cache_size, 0

    def get_max_cache_shape(self, layer_idx: int = 0):
        return self.cache_size

    def batch_split(self):
        return [
            type(self).from_tensors(
                [key[i : i + 1].clone() for key in self.key_cache],
                [value[i : i + 1].clone() for value in self.value_cache],
                self.prefix_length,
                self.max_decode_steps,
            )
            for i in range(self.key_cache[0].shape[0])
        ]

    @classmethod
    def from_batch_splits(cls, splits):
        if not splits:
            raise ValueError("Cannot stack an empty cache list")
        reference = splits[0]
        if any(
            cache.prefix_length != reference.prefix_length
            or cache.max_decode_steps != reference.max_decode_steps
            for cache in splits
        ):
            raise ValueError("All language caches must use the same fixed layout")
        return cls.from_tensors(
            [
                torch.cat([cache.key_cache[i] for cache in splits], dim=0)
                for i in range(len(reference))
            ],
            [
                torch.cat([cache.value_cache[i] for cache in splits], dim=0)
                for i in range(len(reference))
            ],
            reference.prefix_length,
            reference.max_decode_steps,
        )


def _snapshot_cache(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Keep immutable tensor references for the action branch."""
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return [cache[layer_index] for layer_index in range(len(cache))]


def _fork_cache(model, prefix: PrefixState):
    """Create a cache whose layers initially reference the shared prefix tensors."""
    return DynamicCache(ddp_cache_data=prefix.action_kv)


def _decode_vlm(model, **kwargs):
    compiled = getattr(model, "_oxygen_compiled_text_decode", None)
    if compiled is not None:
        return compiled(**kwargs)
    return model.vlm(**kwargs)


@torch.inference_mode()
def prefill_vlm(model, vlm_inputs: dict[str, Any]) -> PrefixState:
    outputs = model.vlm(**vlm_inputs, use_cache=True, logits_to_keep=1)
    return PrefixState(
        outputs=outputs,
        action_kv=_snapshot_cache(outputs.past_key_values),
        prefix_length=int(vlm_inputs["attention_mask"].shape[-1]),
    )


def _seeded_noise_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    cpu_rng_state = torch.get_rng_state()
    gpu_rng_state = torch.cuda.get_rng_state(reference.device) if reference.is_cuda else None
    torch.manual_seed(seed)
    noise = torch.randn_like(reference)
    torch.set_rng_state(cpu_rng_state)
    if gpu_rng_state is not None:
        torch.cuda.set_rng_state(gpu_rng_state, reference.device)
    return noise


@torch.inference_mode()
def init_action(
    model,
    prefix: PrefixState,
    state: torch.Tensor,
    action_mask: torch.Tensor,
    num_steps: int,
    seed: int,
) -> ActionState:
    action_bs, action_length, _ = action_mask.shape
    state_length = state.shape[1]
    query_length = action_length + state_length + 1
    position_ids = (
        torch.arange(query_length, device=action_mask.device)
        .view(1, 1, -1)
        .repeat(3, action_bs, 1)
        + prefix.outputs.position_ids.max(dim=-1)[0][..., None]
        + 1
    )
    position_embeds = model.rotary_emb(action_mask, position_ids)
    dit_mask = torch.tril(
        torch.ones((action_bs, query_length, query_length), device=action_mask.device)
    )
    cache_mask = prefix.outputs.attention_mask[:, None, :].expand(-1, query_length, -1)
    attention_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
    return ActionState(
        x=_seeded_noise_like(action_mask, seed),
        action_mask=action_mask,
        state_embed=model.state_projector(state),
        position_embeds=position_embeds,
        attention_mask=attention_mask,
        prefix_kv=prefix.action_kv,
        num_steps=num_steps,
    )


@torch.inference_mode()
def action_step(model, state: ActionState) -> torch.Tensor:
    if state.finished:
        return state.x
    timestep = torch.full(
        (state.x.shape[0], 1, 1),
        state.step / state.num_steps,
        device=state.x.device,
        dtype=state.x.dtype,
    )
    velocity = model.dit_forward(
        noisy_action=state.x,
        t=timestep,
        action_mask=state.action_mask,
        state_embed=state.state_embed,
        position_embeds=state.position_embeds,
        past_key_values=state.prefix_kv,
        attn_mask=state.attention_mask,
    )
    state.x = state.x + velocity / state.num_steps
    state.step += 1
    return state.x


@torch.inference_mode()
def run_action(model, state: ActionState) -> torch.Tensor:
    while not state.finished:
        action_step(model, state)
    return state.x


def _eos_token_ids(processor) -> set[int]:
    eos = processor.tokenizer.eos_token_id
    if isinstance(eos, int):
        return {eos}
    return {int(token_id) for token_id in eos}


@torch.inference_mode()
def init_language_from_prefix(
    processor, prefix: PrefixState, *, stop_on_eos: bool = True
) -> LanguageState:
    """Start greedy decoding after a complete, independently-prefilled prompt."""
    return LanguageState(
        cache=prefix.outputs.past_key_values,
        attention_mask=prefix.outputs.attention_mask,
        next_token=prefix.outputs.logits[:, -1].argmax(dim=-1, keepdim=True),
        eos_token_ids=_eos_token_ids(processor),
        rope_deltas=prefix.outputs.rope_deltas,
        stop_on_eos=stop_on_eos,
    )


@torch.inference_mode()
def init_language(
    model,
    processor,
    prefix: PrefixState,
    suffix: str,
    *,
    stop_on_eos: bool = True,
) -> LanguageState:
    suffix_ids = processor.tokenizer(
        suffix,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)
    suffix_mask = torch.ones_like(suffix_ids)
    attention_mask = torch.cat(
        [prefix.outputs.attention_mask, suffix_mask], dim=-1
    )
    cache_position = torch.arange(
        prefix.prefix_length,
        prefix.prefix_length + suffix_ids.shape[1],
        device=model.device,
    )
    cache = _fork_cache(model, prefix)
    model.vlm.model.rope_deltas = prefix.outputs.rope_deltas
    outputs = model.vlm(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        logits_to_keep=1,
    )
    return LanguageState(
        cache=outputs.past_key_values,
        attention_mask=attention_mask,
        next_token=outputs.logits[:, -1].argmax(dim=-1, keepdim=True),
        eos_token_ids=_eos_token_ids(processor),
        rope_deltas=outputs.rope_deltas,
        stop_on_eos=stop_on_eos,
    )


@torch.inference_mode()
def extend_prefix(model, processor, prefix: PrefixState, suffix: str) -> PrefixState:
    """Append an expert-specific suffix without mutating the shared prefix cache."""
    suffix_ids = processor.tokenizer(
        suffix,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)
    suffix_mask = torch.ones_like(suffix_ids)
    attention_mask = torch.cat([prefix.outputs.attention_mask, suffix_mask], dim=-1)
    cache_position = torch.arange(
        prefix.prefix_length,
        prefix.prefix_length + suffix_ids.shape[1],
        device=model.device,
    )
    cache = _fork_cache(model, prefix)
    model.vlm.model.rope_deltas = prefix.outputs.rope_deltas
    outputs = model.vlm(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        logits_to_keep=1,
    )
    return PrefixState(
        outputs=outputs,
        action_kv=_snapshot_cache(outputs.past_key_values),
        prefix_length=int(attention_mask.shape[-1]),
    )


@torch.inference_mode()
def language_step(model, state: LanguageState) -> int | None:
    if state.finished:
        return None
    if state.forced_ids is not None and state.forced_index < len(state.forced_ids):
        token_id = int(state.forced_ids[state.forced_index])
        input_token = torch.tensor(
            [[token_id]], device=state.next_token.device, dtype=state.next_token.dtype
        )
        state.forced_index += 1
    else:
        token_id = int(state.next_token[0, 0].item())
        input_token = state.next_token
    state.generated_ids.append(token_id)
    if token_id in state.eos_token_ids and state.stop_on_eos:
        state.finished = True
        return token_id

    state.attention_mask = torch.cat(
        [state.attention_mask, torch.ones_like(input_token)], dim=-1
    )
    cache_position = torch.tensor(
        [state.attention_mask.shape[-1] - 1], device=input_token.device
    )
    model.vlm.model.rope_deltas = state.rope_deltas
    outputs = _decode_vlm(
        model,
        input_ids=input_token,
        attention_mask=state.attention_mask,
        past_key_values=state.cache,
        cache_position=cache_position,
        use_cache=True,
        logits_to_keep=1,
    )
    state.cache = outputs.past_key_values
    state.rope_deltas = outputs.rope_deltas
    state.next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    return token_id


@torch.inference_mode()
def run_language(model, state: LanguageState, max_new_tokens: int) -> list[int]:
    while not state.finished and len(state.generated_ids) < max_new_tokens:
        language_step(model, state)
    return state.generated_ids


def staticize_language_state(state: LanguageState, max_decode_steps: int) -> None:
    """Convert a newly initialized request to the fixed layout used for batching."""
    if state.generated_ids:
        raise ValueError("Static conversion must happen before incremental decode")
    state.cache = StaticLanguageCache(state.cache, max_decode_steps)


@torch.inference_mode()
def batched_language_steps(
    model,
    states: list[LanguageState],
    steps: int,
) -> list[list[int]]:
    """Advance active requests in one model forward per token step."""
    if not states:
        return []
    if any(not isinstance(state.cache, StaticLanguageCache) for state in states):
        raise TypeError("All requests must use StaticLanguageCache")

    cache = StaticLanguageCache.from_batch_splits([state.cache for state in states])
    prefix_length = cache.prefix_length
    max_decode_steps = cache.max_decode_steps
    next_tokens = torch.cat([state.next_token for state in states], dim=0)
    rope_deltas = torch.cat([state.rope_deltas for state in states], dim=0)
    tokens_per_state = [[] for _ in states]

    for _ in range(steps):
        current_steps = torch.tensor(
            [len(state.generated_ids) for state in states],
            device=next_tokens.device,
            dtype=torch.long,
        )
        active = current_steps < max_decode_steps
        if not bool(active.any()):
            break

        input_tokens = next_tokens.clone()
        for index, state in enumerate(states):
            if not active[index]:
                continue
            if state.forced_ids is not None and state.forced_index < len(state.forced_ids):
                input_tokens[index, 0] = state.forced_ids[state.forced_index]
                state.forced_index += 1
            token_id = int(input_tokens[index, 0].item())
            state.generated_ids.append(token_id)
            tokens_per_state[index].append(token_id)

        cache_position = (prefix_length + current_steps)[:, None]
        decoded_positions = torch.arange(max_decode_steps, device=next_tokens.device)
        decoded_mask = decoded_positions[None, :] <= current_steps[:, None]
        prefix_mask = torch.ones(
            (len(states), prefix_length),
            device=next_tokens.device,
            dtype=torch.bool,
        )
        attention_mask = torch.cat([prefix_mask, decoded_mask], dim=-1)
        position_ids = (cache_position + rope_deltas).unsqueeze(0).expand(3, -1, -1)

        model.vlm.model.rope_deltas = rope_deltas
        outputs = _decode_vlm(
            model,
            input_ids=input_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = outputs.past_key_values
        rope_deltas = outputs.rope_deltas
        next_tokens = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)

    cache_splits = cache.batch_split()
    for index, state in enumerate(states):
        state.cache = cache_splits[index]
        state.next_token = next_tokens[index : index + 1].clone()
        state.rope_deltas = rope_deltas[index : index + 1].clone()
        state.finished = len(state.generated_ids) >= max_decode_steps
    return tokens_per_state
