import logging
import pickle
import time

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override
from typing import Callable

from openpi.models import model as _model
from openpi.models import pi05_config
import openpi.models.gemma_05 as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from typing import Literal, TypeAlias, Self
from typing_extensions import deprecated

logger = logging.getLogger("openpi")
PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]

# Deprecated functions moved from pi05.py for better organization

class Pi05Deprecated:
    # Placeholder class for deprecated methods
    pass

# Note: These functions are deprecated and moved here for reference.
# They may not be maintained or compatible with current implementations.

@deprecated("Use sample_text_actions_shared_kv instead")
def sample_text_actions_dependent(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
) -> _model.Actions:
    observation = _model.preprocess_observation(None, observation, train=False)
    # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    assert batch_size == 1, "Batch size must be 1 for sample_actions, subtask can be of different length"
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    # Get all the prefix tokens, mask, and ar mask
    output_tokens, kv_cache, prefix_mask, prefix_ar_mask = self.sample_text(rng, observation, max_decoding_steps=20, PALIGEMMA_EOS_TOKEN=1, temperature=0.0)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        query_attn_mask = full_attn_mask[:, -suffix_tokens.shape[1]:, :] # [B, suffix_len, prefix_len + suffix_len]

        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_mask.shape[1] + suffix_tokens.shape[1],
        )
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=query_attn_mask,
            positions=positions,
            kv_cache=kv_cache, # kv_cache is not updated during multiple denoising steps
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return x_t + dt * v_t, time + dt

    def cond(carry):
        x_t, time = carry
        # robust to floating-point error
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    
    return (x_0, output_tokens)

@deprecated("Use sample_text_actions_shared_kv instead")
def sample_text_actions_par(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    max_decoding_steps: int = 20,
) -> tuple[_model.Actions, at.Int[at.Array, "b s"]]:
    rng_text, rng_action = jax.random.split(rng)
    
    # 1. Prefill
    prefill_result = self.prefill(observation, align_right=True, max_decoding_steps=max_decoding_steps)
    
    # 2. Parallel Generation (Text & Actions)
    # Dispatch Text
    output_tokens, kv_cache, mask, ar_mask = self.sample_text_with_kv(
        rng_text, prefill_result, max_decoding_steps=max_decoding_steps
    )
    
    # Dispatch Actions
    if noise is None:
        noise = jax.random.normal(rng_action, (observation.state.shape[0], self.action_horizon, self.action_dim))
        
    actions = self.sample_actions_with_kv(
        rng_action, observation, prefill_result, num_steps=num_steps, noise=noise
    )
    
    return actions, output_tokens

@deprecated("Use prefile_sample_text_actions_shared_kv instead")
def profile_sample_text_actions_par(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    max_decoding_steps: int = 20,
) -> tuple[_model.Actions, at.Int[at.Array, "b s"], dict[str, float]]:
    rng_text, rng_action = jax.random.split(rng)
    t0 = time.time()
    
    # 1. Prefill
    prefill_result = self.prefill(observation, align_right=True, max_decoding_steps=max_decoding_steps)
    sync(prefill_result)
    t1 = time.time()
    
    # 2. Parallel Generation
    # Dispatch Text
    output_tokens, kv_cache, mask, ar_mask = self.sample_text_with_kv(
        rng_text, prefill_result, max_decoding_steps=max_decoding_steps
    )
    
    # Dispatch Actions
    if noise is None:
        noise = jax.random.normal(rng_action, (observation.state.shape[0], self.action_horizon, self.action_dim))
        
    actions = self.sample_actions_with_kv(
        rng_action, observation, prefill_result, num_steps=num_steps, noise=noise
    )

    # Sync both
    sync([output_tokens, actions])
    t2 = time.time()
    
    timings = {
        "prefill": t1 - t0,
        "generation": t2 - t1,
        "total": t2 - t0
    }
    return actions, output_tokens, timings

@deprecated("Use sample_text_actions_shared_kv instead")
def sample_text_and_actions_interleaved_with_kv(
    self,
    rng_text: at.KeyArrayLike,
    rng_action: at.KeyArrayLike,
    observation: _model.Observation,
    prefill_result: tuple,
    *,
    num_steps: int = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    max_decoding_steps: int = 20,
    PALIGEMMA_EOS_TOKEN: int = 1,
    temperature: float = 0.0,
) -> tuple[_model.Actions, at.Int[at.Array, "b s"], at.Bool[at.Array, "b s"], at.Bool[at.Array, "s"]]:
    
    # --- Shared Setup ---
    (prefix_out, kv_cache_shared, prefix_mask, prefix_full_attn_output_mask, prefix_ar_mask, prefix_token_embeddings) = prefill_result
    batch_size = prefix_token_embeddings.shape[0]
    
    # Determine total iterations
    total_steps = max(max_decoding_steps, num_steps)

    # --- Text Generation Setup ---
    # kv_cache_text = kv_cache_shared # NNX cache is usually immutable pytree, sharing is fine if we update correctly? No, we need separate variable to track evolution.
    kv_cache_text = kv_cache_shared
    
    prefill_size = prefix_token_embeddings.shape[1]
    prefill_len = jnp.sum(prefix_mask, axis=-1)
    prefix_start = prefill_size - prefill_len

    last_token_embedding_text = prefix_out[:, -1:]
    last_logits_text = self.PaliGemma.llm(last_token_embedding_text, method="deembed")
    last_logits_text = jax.nn.log_softmax(last_logits_text, axis=-1)
    output_tokens = jnp.zeros((batch_size, max_decoding_steps), dtype=jnp.int32)
    
    # Pre-split text RNGs
    rng_text_steps = jax.random.split(rng_text, total_steps)
    
    # --- Action Generation Setup ---
    kv_len = prefix_full_attn_output_mask.shape[-1]
    prefix_len = prefix_mask.shape[-1]
    
    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    if noise is None:
        noise = jax.random.normal(rng_action, (batch_size, self.action_horizon, self.action_dim))
    
    x_t = noise
    time = 1.0
    
    # --- Interleaved Loop ---
    # State: (output_tokens, kv_cache_text, last_logits_text, text_finished, x_t, time)
    # Note: kv_cache_shared (for actions) acts as a constant read-only input for actions because actions don't update it.
    
    def loop_body(carry, loop_step):
        output_tokens, kv_cache_text, last_logits_text, text_finished, x_t, time = carry
        rng_step = rng_text_steps[loop_step]
        
        # --- Text Branch (Run if step < max_decoding_steps) ---
        def text_step_fn(args):
            _logits, _tokens, _cache, _step = args
            
            # Sample
            _rng_step = rng_step # use closed over variable
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(_rng_step, _logits / temperature, axis=-1),
                lambda _: jnp.argmax(_logits, axis=-1),
                operand=None,
            )
            
            # Update tokens
            _tokens = put_along_last_axis(_tokens, jnp.broadcast_to(_step, (token.shape[0], 1)), token)
            
            # Check EOS
            # If already finished, all_eos remains True. Check new tokens.
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)
            
            # forward
            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + _step
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] < (jnp.broadcast_to(prefill_size + _step + 1, (prefix_start.shape[0], 1, 1))),
            )
            
            (prefix_out_new, _), kv_cache_new = self.PaliGemma.llm(
                [token_embedding, None], mask=mask, positions=positions, adarms_cond=[None, None], kv_cache=_cache
            )
            
            last_token_emb = prefix_out_new[:, -1:]
            last_logits_new = self.PaliGemma.llm(last_token_emb, method="deembed")
            last_logits_new = jax.nn.log_softmax(last_logits_new, axis=-1)
            
            return last_logits_new, _tokens, kv_cache_new, all_eos

        # Run text step conditionally
        run_text = (loop_step < max_decoding_steps) & (~text_finished)
        last_logits_text, output_tokens, kv_cache_text, new_text_finished = jax.lax.cond(
            run_text,
            text_step_fn,
            lambda args: (args[0], args[1], args[2], text_finished), # Return same
            (last_logits_text, output_tokens, kv_cache_text, loop_step)
        )
        text_finished = new_text_finished

        # --- Action Branch (Run if step < num_steps) ---
        def action_step_fn(args):
            _x_t, _time = args
            
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, _x_t, jnp.broadcast_to(_time, batch_size)
            )
            
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            
            # Prefix mask construction (handling padding as in sample_actions_with_kv)
            padding_len = kv_len - prefix_len
            prefix_attn_mask_valid = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            
            if padding_len > 0:
                padding_mask = jnp.zeros((batch_size, suffix_tokens.shape[1], padding_len), dtype=jnp.bool_)
                prefix_attn_mask = jnp.concatenate([prefix_attn_mask_valid, padding_mask], axis=-1)
            else:
                prefix_attn_mask = prefix_attn_mask_valid
            
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            
            # Note: reusing kv_cache_shared (prefill result), NOT kv_cache_text
            (prefix_out_a, suffix_out_a), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache_shared,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out_a[:, -self.action_horizon :])
            
            return _x_t + dt * v_t, _time + dt
        
        run_action = loop_step < num_steps
        x_t, time = jax.lax.cond(
            run_action,
            action_step_fn,
            lambda args: args,
            (x_t, time)
        )
        
        return (output_tokens, kv_cache_text, last_logits_text, text_finished, x_t, time), None

    
    carry_init = (output_tokens, kv_cache_text, last_logits_text, False, x_t, time)
    
    final_carry, _ = jax.lax.scan(loop_body, carry_init, jnp.arange(total_steps))
    
    (output_tokens, kv_cache_text, last_logits_text, text_finished, x_t, time) = final_carry
    
    mask = jnp.concatenate([prefix_mask, (output_tokens!=0).astype(jnp.bool_)], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, jnp.ones(max_decoding_steps, dtype=jnp.bool_)], axis=0)

    return x_t, output_tokens, mask, ar_mask

@deprecated("Use sample_text_actions_shared_kv instead")
def sample_text_and_actions_interleaved(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    max_decoding_steps: int = 20,
) -> tuple[_model.Actions, at.Int[at.Array, "b s"]]:
    rng_text, rng_action = jax.random.split(rng)
    
    # 1. Prefill
    prefill_result = self.prefill(observation, align_right=True, max_decoding_steps=max_decoding_steps)
    
    # 2. Interleaved Generation
    if noise is None:
        noise = jax.random.normal(rng_action, (observation.state.shape[0], self.action_horizon, self.action_dim))
        
    actions, output_tokens, mask, ar_mask = self.sample_text_and_actions_interleaved_with_kv(
        rng_text, rng_action, observation, prefill_result, 
        num_steps=num_steps, noise=noise, max_decoding_steps=max_decoding_steps
    )
    
    return actions, output_tokens

@deprecated("Use profile_sample_text_actions_shared_kv instead")
def profile_sample_text_and_actions_interleaved(
    self,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    num_steps: int | at.Int[at.Array, ""] = 10,
    noise: at.Float[at.Array, "b ah ad"] | None = None,
    max_decoding_steps: int = 20,
) -> tuple[_model.Actions, at.Int[at.Array, "b s"], dict[str, float]]:
    rng_text, rng_action = jax.random.split(rng)
    t0 = time.time()
    
    # 1. Prefill
    prefill_result = self.prefill(observation, align_right=True, max_decoding_steps=max_decoding_steps)
    sync(prefill_result)
    t1 = time.time()
    
    # 2. Interleaved Generation
    if noise is None:
        noise = jax.random.normal(rng_action, (observation.state.shape[0], self.action_horizon, self.action_dim))
        
    actions, output_tokens, mask, ar_mask = self.sample_text_and_actions_interleaved_with_kv(
        rng_text, rng_action, observation, prefill_result, 
        num_steps=num_steps, noise=noise, max_decoding_steps=max_decoding_steps
    )
    
    sync([actions, output_tokens])
    t2 = time.time()
    
    timings = {
        "prefill": t1 - t0,
        "generation": t2 - t1,
        "total": t2 - t0
    }
    return actions, output_tokens, timings