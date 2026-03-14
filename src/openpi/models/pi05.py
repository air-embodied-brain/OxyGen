import logging
import pickle
import time
from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class IncrementalTextState:
    """State for incremental text generation across frames.

    This tracks all necessary information to resume text generation
    from a previous frame. Registered as a JAX pytree for JIT compatibility.
    """
    # Generation state (data fields - traced in JIT)
    rng: jax.Array                      # [B] or [B, 2] RNG for sampling
    last_logits: jax.Array              # [B, 1, vocab_size] - logits from last token
    output_tokens: jax.Array            # [B, max_tokens] - all tokens generated so far
    kv_cache: tuple                     # KV cache tuple
    current_step: jax.Array             # [B] - number of tokens generated so far (per-element)
    is_finished: jax.Array              # [B] - per-sample EOS flags
    prefill_len: jax.Array              # [B] - actual prefix length per sample

    # Prefill metadata (meta fields - static in JIT, define pytree structure)
    prefill_size: int                   # Size of prefix tokens
    max_decoding_steps: int             # Max tokens to generate
    cache_size: int                     # Full KV cache size (prefill_size + max_decoding_steps)


# Register IncrementalTextState as JAX pytree for JIT compatibility.
# Data fields are traced (arrays that change per call).
# Meta fields are static (ints that define the tree structure).
jax.tree_util.register_dataclass(
    IncrementalTextState,
    data_fields=['rng', 'last_logits', 'output_tokens', 'kv_cache',
                 'current_step', 'is_finished', 'prefill_len'],
    meta_fields=['prefill_size', 'max_decoding_steps', 'cache_size'],
)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def create_padding_mask(lengths: at.Int[at.Array, "b"], max_len: int) -> at.Bool[at.Array, "b max_len"]:
    """Create padding mask from sequence lengths.

    Args:
        lengths: Actual length of each sequence [B]
        max_len: Maximum length (padded length)

    Returns:
        Mask [B, max_len] where True = valid, False = padding
    """
    positions = jnp.arange(max_len)[None, :]  # [1, max_len]
    lengths_expanded = lengths[:, None]        # [B, 1]
    return positions < lengths_expanded        # [B, max_len]


def _gather_last_valid_token(out: jnp.ndarray, prefill_len: jnp.ndarray) -> jnp.ndarray:
    """Gather the last valid token embedding from left-aligned prefill output.

    With left-aligned tokens, valid tokens are at [0, prefill_len) and padding
    fills [prefill_len, prefill_size). Using out[:, -1:] would pick a padding
    position when prefill_len < prefill_size — padding positions produce garbage
    because their attention mask is all-zero (they can't attend to anything).

    IMPORTANT: Any new method that runs a prefill and extracts the seed embedding
    for autoregressive decoding MUST use this function instead of out[:, -1:].
    The remaining out[:, -1:] calls in decode loops are safe because they operate
    on single-token LLM outputs where the sequence dimension is always 1.

    Args:
        out: Prefill output [B, seq_len, D].
        prefill_len: Number of valid tokens per batch element [B].

    Returns:
        Last valid token embedding [B, 1, D].
    """
    batch_idx = jnp.arange(out.shape[0])
    return out[batch_idx, prefill_len - 1][:, None, :]


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


# Force sync in profiling functions
def sync(obj):
    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None, obj)



@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi05(_model.BaseModel):
    def __init__(self, config: pi05_config.Pi05Config, rngs: nnx.Rngs):
        """Initialize the Pi05 model.

        Args:
            config: Configuration for the Pi05 model.
            rngs: Random number generators for initialization.
        """
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed the prefix tokens from observation (images and language).

        Args:
            obs: The observation containing images and tokenized prompt.

        Returns:
            Tuple of (tokens, input_mask, ar_mask).
        """
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ### TODO: pi0 -> full attention between image and language inputs
            ### TODO: pi05 -> AR attention for subtask generation, but what about action expert?
            ar_mask += [True] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Embed the suffix tokens for action generation.

        Args:
            obs: The observation.
            noisy_actions: Noisy actions for flow matching.
            timestep: Timestep for conditioning.

        Returns:
            Tuple of (tokens, input_mask, ar_mask, adarms_cond).
        """
        input_mask = []
        ar_mask = []
        tokens = []

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        # time MLP (for adaRMS)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, real_action_dim: int=32, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Compute the loss for training the model.

        Args:
            rng: Random number generator.
            observation: The observation.
            actions: Ground truth actions.
            real_action_dim: Dimension of real actions.
            train: Whether in training mode.

        Returns:
            The computed loss.
        """
        # TODO: Support only use part of loss (e.g. only)
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        ### 1. Subtask-Generation Loss (Cross-Entropy Loss)
        # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
        # TODO: Do we need state to perform subtask generation?
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )

        # Use prefix tokens to perform subtask generation (Prefix: images*3, high-level prompt, low-level prompt, state?)
        # We input the last token because the last token is used for flow loss
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None], 
            mask=prefix_attn_mask, 
            positions=prefix_positions, 
            adarms_cond=[None, None]
        )
        prefix_out = prefix_out[:, :-1]

        # decode from embedding to logits
        logits = self.PaliGemma.llm(
            prefix_out[:, -targets.shape[1] :], method='deembed'
        )
        logp = jax.nn.log_softmax(logits, axis=-1)

        # Compute CE loss on token targets
        assert observation.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        subtask_generation_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

        ### 2. Flow Matching Loss (MSE Loss)
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        attn_mask = attn_mask[:, -suffix_tokens.shape[1]:, :] # Q is [B, action_dim, ...], KV is full length
        positions = jnp.cumsum(input_mask, axis=1) - 1
        positions = positions[:, -suffix_tokens.shape[1]:]
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], kv_cache=kv_cache, mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # Calculate flow loss with true actions (Real Action Dim <= Action Dim (Padding))
        flow_loss = jnp.mean(jnp.square(v_t[:, :, :real_action_dim] - u_t[:, :, :real_action_dim]), axis=-1)

        return subtask_generation_loss + jnp.mean(flow_loss, axis=-1)

    def prefill(
        self,
        observation: _model.Observation,
        align_right: bool = False,
        max_decoding_steps: int = 0, # We don't decode here, but reserve context for decoding
        sequence_lengths: at.Int[at.Array, "b"] | None = None,
    ):
        """Prefill the KV cache with prefix tokens (left-aligned).

        Tokens are left-aligned: valid tokens occupy positions [0, prefill_len),
        padding fills [prefill_len, prefill_size). During decoding, new KV entries
        are written sequentially starting at position prefill_size, so the
        attention mask must cover [0, prefill_len) | [prefill_size, prefill_size+step).

        IMPORTANT constraints for future development:
        1. KV cache layout is non-contiguous: valid entries are at [0, prefill_len)
           and [prefill_size, prefill_size+step), with garbage padding in between.
           Any new decoding mask or KV cache indexing must account for this gap.
        2. After prefill, use _gather_last_valid_token() to extract the seed
           embedding — never use out[:, -1:] which hits a padding position.

        Args:
            observation: The observation.
            align_right: Deprecated, kept for API compat. Must be False.
            max_decoding_steps: Number of KV cache slots to reserve for decoding.
            sequence_lengths: Optional sequence lengths for variable-length batching [B].

        Returns:
            Tuple of prefill results.
        """
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # Override prefix_mask if sequence_lengths provided (for variable-length batching)
        if sequence_lengths is not None:
            prefix_mask = create_padding_mask(sequence_lengths, prefix_token_embeddings.shape[1])

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        if max_decoding_steps > 0:
            prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))

        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None], mask=prefix_attn_mask, positions=prefix_positions, adarms_cond=[None, None]
        )
        return prefix_out, kv_cache, prefix_mask, prefix_attn_mask, prefix_ar_mask, prefix_token_embeddings

    def sample_text_with_kv(
        self,
        rng: at.KeyArrayLike,
        prefill_result: tuple,
        max_decoding_steps: int = 20,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> str:
        """Sample text using prefilled KV cache.

        Args:
            rng: Random number generator.
            prefill_result: Result from prefill.
            max_decoding_steps: Maximum steps for decoding.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Sampling temperature.

        Returns:
            Sampled text.
        """
        
        (prefix_out, kv_cache, prefix_mask, _, prefix_ar_mask, prefix_token_embeddings) = prefill_result
        
        batch_size = prefix_token_embeddings.shape[0]
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)

        last_token_embedding = _gather_last_valid_token(prefix_out, prefill_len)
        last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
        last_logits = jax.nn.log_softmax(last_logits, axis=-1)
        output_tokens = jnp.zeros((batch_size, max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry

            # Sample token from last logit
            # Split RNG for this step
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            ### TODO: erase extra decoded token due to mismatch
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding =  self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + step
            cache_pos = jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
            mask = (cache_pos < prefill_len[:, None, None]) | ((cache_pos >= prefill_size) & (cache_pos < prefill_size + step + 1))

            (prefix_out, _), kv_cache = self.PaliGemma.llm(
                [token_embedding, None], mask=mask, positions=positions, adarms_cond=[None, None], kv_cache=cache
            )
            last_token_embedding = prefix_out[:, -1:]
            last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
            last_logits = jax.nn.log_softmax(last_logits, axis=-1)

            return rng, last_logits, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, output_tokens, kv_cache, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logits, output_tokens, kv_cache, False, 0)
        )

        mask = jnp.concatenate([prefix_mask, (output_tokens!=0).astype(jnp.bool_)], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, jnp.ones(max_decoding_steps, dtype=jnp.bool_)], axis=0)
        return output_tokens, kv_cache, mask, ar_mask

    def sample_text_incremental(
        self,
        rng: at.KeyArrayLike,
        state_or_prefill: IncrementalTextState | tuple,
        tokens_to_generate: int = 5,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
        is_initial: bool = True,
    ) -> tuple[jax.Array, IncrementalTextState, bool]:
        """Generate a fixed number of tokens incrementally for continuous batching.

        This method enables incremental text generation across frames. On the first call,
        pass prefill_result and is_initial=True. On subsequent calls, pass the returned
        state and is_initial=False.

        Args:
            rng: Random number generator.
            state_or_prefill: Either IncrementalTextState (for resuming) or prefill_result tuple (for starting).
            tokens_to_generate: Number of tokens to generate in this call.
            PALIGEMMA_EOS_TOKEN: EOS token ID for early stopping.
            temperature: Sampling temperature.
            is_initial: True if starting from prefill, False if resuming from state.

        Returns:
            Tuple of (tokens_generated_this_call, updated_state, all_finished).
        """
        if is_initial:
            # Initialize state from prefill result
            (prefix_out, kv_cache, prefix_mask, prefix_attn_mask, prefix_ar_mask, prefix_token_embeddings) = state_or_prefill

            batch_size = prefix_token_embeddings.shape[0]
            prefill_size = prefix_token_embeddings.shape[1]
            prefill_len = jnp.sum(prefix_mask, axis=-1)

            # Infer max_decoding_steps and cache_size from prefix_attn_mask
            # The mask was padded with max_decoding_steps in prefill
            cache_size = prefix_attn_mask.shape[-1]
            max_decoding_steps = cache_size - prefill_size

            last_token_embedding = _gather_last_valid_token(prefix_out, prefill_len)
            last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
            last_logits = jax.nn.log_softmax(last_logits, axis=-1)

            output_tokens = jnp.zeros((batch_size, max_decoding_steps))
            is_finished = jnp.zeros(batch_size, dtype=jnp.bool_)

            state = IncrementalTextState(
                rng=rng,
                last_logits=last_logits,
                output_tokens=output_tokens,
                kv_cache=kv_cache,
                current_step=jnp.int32(0),
                is_finished=is_finished,
                prefill_size=prefill_size,
                prefill_len=prefill_len,
                max_decoding_steps=max_decoding_steps,
                cache_size=cache_size,
            )
        else:
            # Resume from existing state
            state = state_or_prefill

        # Generate tokens_to_generate tokens (or until all sequences finish)
        tokens_this_call = []

        # Update RNG
        current_rng = state.rng if not is_initial else rng

        for _ in range(tokens_to_generate):
            # Check if all sequences are finished or reached max
            if jnp.all(state.is_finished) or state.current_step >= state.max_decoding_steps:
                break

            # Sample token from last logits
            current_rng, rng_step = jax.random.split(current_rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, state.last_logits / temperature, axis=-1),
                lambda _: jnp.argmax(state.last_logits, axis=-1),
                operand=None,
            )
            # token shape: [B, vocab_size] -> [B] after argmax/categorical

            # Store token
            new_output_tokens = put_along_last_axis(
                state.output_tokens,
                jnp.broadcast_to(state.current_step, (token.shape[0], 1)),
                token
            )
            tokens_this_call.append(token)

            # Check for EOS
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            new_is_finished = jnp.logical_or(state.is_finished, has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = state.prefill_len[:, None] + state.current_step

            # Create attention mask with full cache size
            # Valid prefill tokens [0, prefill_len) and decoded tokens [prefill_size, prefill_size + step)
            cache_pos = jnp.arange(state.cache_size)[None, None, :]
            mask = (cache_pos < state.prefill_len[:, None, None]) | ((cache_pos >= state.prefill_size) & (cache_pos < state.prefill_size + state.current_step + 1))

            (prefix_out, _), new_kv_cache = self.PaliGemma.llm(
                [token_embedding, None], mask=mask, positions=positions, adarms_cond=[None, None], kv_cache=state.kv_cache
            )
            last_token_embedding = prefix_out[:, -1:]
            new_last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
            new_last_logits = jax.nn.log_softmax(new_last_logits, axis=-1)

            # Create new state (immutable update)
            state = replace(
                state,
                rng=current_rng,
                kv_cache=new_kv_cache,
                last_logits=new_last_logits,
                output_tokens=new_output_tokens,
                is_finished=new_is_finished,
                current_step=state.current_step + 1,
            )

        # Stack tokens generated in this call: [num_tokens, B, 1] -> [B, num_tokens]
        if tokens_this_call:
            tokens_generated = jnp.stack(tokens_this_call, axis=1)  # [B, num_tokens, 1]
            tokens_generated = jnp.squeeze(tokens_generated, axis=-1)  # [B, num_tokens]
        else:
            # No tokens generated (all finished or max reached)
            tokens_generated = jnp.zeros((state.output_tokens.shape[0], 0), dtype=jnp.int32)

        all_finished = jnp.all(state.is_finished) or state.current_step >= state.max_decoding_steps

        return tokens_generated, state, all_finished

    def init_incremental_state(
        self,
        prefill_result: tuple,
        rng: at.KeyArrayLike,
    ) -> IncrementalTextState:
        """Initialize IncrementalTextState from a prefill result.

        This is the first step for incremental text generation. Call this once
        per new request, then call generate_n_tokens to produce text.

        Designed to be wrapped with module_jit for efficient compilation.

        Args:
            prefill_result: Result from self.prefill() with max_decoding_steps > 0.
            rng: Random number generator for this request.

        Returns:
            IncrementalTextState ready for generate_n_tokens.
        """
        (prefix_out, kv_cache, prefix_mask, prefix_attn_mask, prefix_ar_mask, prefix_token_embeddings) = prefill_result

        batch_size = prefix_token_embeddings.shape[0]
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)

        cache_size = prefix_attn_mask.shape[-1]
        max_decoding_steps = cache_size - prefill_size

        last_token_embedding = _gather_last_valid_token(prefix_out, prefill_len)
        last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
        last_logits = jax.nn.log_softmax(last_logits, axis=-1)

        output_tokens = jnp.zeros((batch_size, max_decoding_steps))
        is_finished = jnp.zeros(batch_size, dtype=jnp.bool_)
        # Per-element step counter for batching requests at different stages
        current_step = jnp.zeros(batch_size, dtype=jnp.int32)

        # Per-element RNG for batched sampling
        # New JAX PRNG keys are 0-dimensional scalars, old-style keys are [2]
        # We need [batch_size, ...] for batched sampling
        if rng.ndim < 2:
            # Single RNG key (0D scalar or 1D array), split for each batch element
            rng = jax.random.split(rng, batch_size)

        return IncrementalTextState(
            rng=rng,
            last_logits=last_logits,
            output_tokens=output_tokens,
            kv_cache=kv_cache,
            current_step=current_step,
            is_finished=is_finished,
            prefill_size=prefill_size,
            prefill_len=prefill_len,
            max_decoding_steps=max_decoding_steps,
            cache_size=cache_size,
        )

    def generate_n_tokens(
        self,
        state: IncrementalTextState,
        *,
        tokens_to_generate: int = 5,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> tuple[jax.Array, IncrementalTextState, jax.Array]:
        """Generate N tokens from an existing IncrementalTextState.

        JIT-compatible: no Python-level GPU sync points. The Python for-loop
        is unrolled at JIT trace time. Finished sequences are masked rather
        than triggering early exit.

        Supports batched generation with per-element current_step, allowing
        different requests to be at different stages of generation.

        Designed to be wrapped with module_jit for efficient compilation.

        Args:
            state: IncrementalTextState from init_incremental_state or a previous generate_n_tokens call.
            tokens_to_generate: Number of tokens to produce (static, unrolled at trace time).
            PALIGEMMA_EOS_TOKEN: EOS token ID (static).
            temperature: Sampling temperature (static).

        Returns:
            Tuple of:
                - tokens_generated: [B, tokens_to_generate] tokens produced this call (0 for finished/masked positions).
                - updated_state: IncrementalTextState with updated KV cache, tokens, step counter.
                - all_finished: Scalar bool — True if all sequences have finished or reached max steps.
        """
        tokens_this_call = []
        batch_size = state.output_tokens.shape[0]

        # RNG handling: state.rng should be [B] array of scalar keys or [B, 2] old-style
        # After init_incremental_state, it's (B,) with new PRNG or (B, 2) with old
        current_rng = state.rng
        if current_rng.ndim < 2:
            # Scalar or 1D - need to ensure we have batch of keys
            if current_rng.ndim == 0:
                # Single scalar key, split for batch
                current_rng = jax.random.split(current_rng, batch_size)
            # else: already (B,) array of scalar keys, use as-is

        for _ in range(tokens_to_generate):
            # Determine which sequences should still be updated (per-element)
            # current_step is [B], so comparison is element-wise
            should_update = ~state.is_finished & (state.current_step < state.max_decoding_steps)

            # Sample token from last logits - need per-element RNG
            # Split each element's RNG: [B, 2] -> [B, 2, 2] -> current [B, 2], step [B, 2]
            current_rng = jax.vmap(jax.random.split)(current_rng)
            rng_step = current_rng[:, 1]  # [B, 2]
            current_rng = current_rng[:, 0]  # [B, 2]

            # Sample token
            if temperature > 0.0:
                token = jax.vmap(lambda r, l: jax.random.categorical(r, l / temperature, axis=-1))(
                    rng_step, state.last_logits
                )
            else:
                token = jnp.argmax(state.last_logits, axis=-1)

            # Store token in output buffer at per-element index (masked for finished sequences)
            # current_step is [B], need [B, 1] for put_along_last_axis
            updated_output_tokens = put_along_last_axis(
                state.output_tokens,
                state.current_step[:, None],
                token,
            )
            new_output_tokens = jnp.where(should_update[:, None], updated_output_tokens, state.output_tokens)

            # Collect token for return value (masked)
            masked_token = jnp.where(should_update[:, None], token, jnp.zeros_like(token))
            tokens_this_call.append(masked_token)

            # Check for EOS (per-element)
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            new_is_finished = state.is_finished | has_eos | (state.current_step >= state.max_decoding_steps)

            # Run one LLM forward pass to update KV cache and get next logits
            token_embedding = self.PaliGemma.llm(token, method="embed")

            # Position is per-element: prefill_len[i] + current_step[i]
            positions = (state.prefill_len + state.current_step)[:, None]

            # Mask must account for per-element current_step
            # Valid prefill tokens [0, prefill_len) and decoded tokens [prefill_size, prefill_size + step)
            cache_indices = jnp.arange(state.cache_size)  # [cache_size]
            mask = (cache_indices[None, None, :] < state.prefill_len[:, None, None]) | ((cache_indices[None, None, :] >= state.prefill_size) & (cache_indices[None, None, :] < state.prefill_size + state.current_step[:, None, None] + 1))
            mask = jnp.broadcast_to(mask, (batch_size, 1, state.cache_size))

            (prefix_out, _), new_kv_cache = self.PaliGemma.llm(
                [token_embedding, None], mask=mask, positions=positions,
                adarms_cond=[None, None], kv_cache=state.kv_cache,
            )
            last_token_embedding = prefix_out[:, -1:]
            new_last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
            new_last_logits = jax.nn.log_softmax(new_last_logits, axis=-1)

            state = replace(
                state,
                rng=current_rng,
                kv_cache=new_kv_cache,
                last_logits=new_last_logits,
                output_tokens=new_output_tokens,
                is_finished=new_is_finished,
                current_step=state.current_step + 1,  # Element-wise increment
            )

        # Stack tokens: list of [B, 1] -> [B, tokens_to_generate]
        tokens_generated = jnp.stack(tokens_this_call, axis=1)
        if tokens_generated.ndim > 2:
            tokens_generated = jnp.squeeze(tokens_generated, axis=-1)

        # All finished when all elements are done (per-element check)
        all_finished = jnp.all(state.is_finished) | jnp.all(state.current_step >= state.max_decoding_steps)
        return tokens_generated, state, all_finished

    def sample_text(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        max_decoding_steps: int = 20,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> str:
        """Sample text from observation.

        Args:
            rng: Random number generator.
            observation: The observation.
            max_decoding_steps: Maximum decoding steps.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Sampling temperature.

        Returns:
            Sampled text.
        """
        batch_size = observation.tokenized_prompt.shape[0]
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)

        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        # import pdb; pdb.set_trace()
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None], mask=prefix_attn_mask, positions=prefix_positions, adarms_cond=[None, None]
        )
        last_token_embedding = _gather_last_valid_token(prefix_out, prefill_len)
        last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
        last_logits = jax.nn.log_softmax(last_logits, axis=-1)
        output_tokens = jnp.zeros((batch_size, max_decoding_steps))

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry

            # Sample token from last logit
            # Split RNG for this step
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            ### TODO: erase extra decoded token due to mismatch
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding =  self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + step
            cache_pos = jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
            mask = (cache_pos < prefill_len[:, None, None]) | ((cache_pos >= prefill_size) & (cache_pos < prefill_size + step + 1))

            (prefix_out, _), kv_cache = self.PaliGemma.llm(
                [token_embedding, None], mask=mask, positions=positions, adarms_cond=[None, None], kv_cache=cache
            )
            last_token_embedding = prefix_out[:, -1:]
            last_logits = self.PaliGemma.llm(last_token_embedding, method="deembed")
            last_logits = jax.nn.log_softmax(last_logits, axis=-1)

            return rng, last_logits, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, output_tokens, kv_cache, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logits, output_tokens, kv_cache, False, 0)
        )

        mask = jnp.concatenate([prefix_mask, (output_tokens!=0).astype(jnp.bool_)], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, jnp.ones(max_decoding_steps, dtype=jnp.bool_)], axis=0)
        #  output_tokens [B, max_decoding_steps]
        #  kv_cache [B, prefix_len+max_decoding_steps, ...]
        #  mask [B, prefix_len+max_decoding_steps]
        #  ar_mask [prefix_len+max_decoding_steps]
        return output_tokens, kv_cache, mask, ar_mask
    
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions using flow matching.

        Args:
            rng: Random number generator.
            observation: The observation.
            num_steps: Number of denoising steps.
            noise: Initial noise.

        Returns:
            Sampled actions.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

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
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
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
        return x_0

    # The result is not exactly identical to sample_actions due to float precision
    # They use the same left-aligned memory layout for context
    def sample_actions_with_kv(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        prefill_result: tuple,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """Sample actions using prefilled KV cache.

        Args:
            rng: Random number generator.
            observation: The observation.
            prefill_result: Result from prefill.
            num_steps: Number of denoising steps.
            noise: Initial noise.

        Returns:
            Sampled actions.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        (_, kv_cache, prefix_mask, prefix_full_attn_output_mask, _, _) = prefill_result
        
        # kv_cache length might be longer than prefix_mask if padding was used (e.g. for task generation)
        # prefix_full_attn_output_mask has shape [B, L, KvLen]. We can get KvLen from it.
        kv_len = prefix_full_attn_output_mask.shape[-1]
        prefix_len = prefix_mask.shape[-1]

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens.
            # We must handle potential padding in KV cache (KvLen >= PrefixLen)
            padding_len = kv_len - prefix_len
            
            # 1. Create mask for the actual prefix (valid tokens)
            prefix_attn_mask_valid = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            
            # 2. Pad to match KV cache size if necessary
            if padding_len > 0:
                padding_mask = jnp.zeros((batch_size, suffix_tokens.shape[1], padding_len), dtype=jnp.bool_)
                prefix_attn_mask = jnp.concatenate([prefix_attn_mask_valid, padding_mask], axis=-1)
            else:
                prefix_attn_mask = prefix_attn_mask_valid
                
            # `combined_mask` is shape (b, suffix_len, KvLen + suffix_len)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                kv_len + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            # With left-aligned prefill, sum(prefix_mask) gives the first free position.
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
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
        return x_0

    # From BRUNOFANG's sample_actions, where action is dependent on generated subtasks
    # We don't use this in our setting, but keep it for reference
    def sample_text_actions_dependent(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> _model.Actions:
        """Sample actions dependent on generated subtasks (legacy).

        Args:
            rng: Random number generator.
            observation: The observation.
            num_steps: Number of denoising steps.
            noise: Initial noise.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Sampling temperature.

        Returns:
            Sampled actions.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        assert batch_size == 1, "Batch size must be 1 for sample_actions, subtask can be of different length"
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Get all the prefix tokens, mask, and ar mask
        output_tokens, kv_cache, prefix_mask, prefix_ar_mask = self.sample_text(rng, observation, max_decoding_steps=20, PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN, temperature=temperature)

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

    def sample_text_actions_shared_kv(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        max_decoding_steps: int = 20,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> tuple[_model.Actions, at.Int[at.Array, "b s"]]:
        """Sample text and actions with shared KV cache.

        Args:
            rng: Random number generator.
            observation: The observation.
            num_steps: Number of denoising steps.
            noise: Initial noise.
            max_decoding_steps: Maximum text decoding steps.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Sampling temperature.

        Returns:
            Tuple of (actions, output_tokens).
        """
        rng_text, rng_action = jax.random.split(rng)
        
        # 1. Prefill: use left-aligned KV for both text and actions
        prefill_result = self.prefill(observation, align_right=False, max_decoding_steps=max_decoding_steps)
        
        # 2. Text Generation based on prefill results
        output_tokens, kv_cache, mask, ar_mask = self.sample_text_with_kv(
            rng_text, prefill_result, max_decoding_steps=max_decoding_steps, PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN, temperature=temperature
        )
        
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng_action, (batch_size, self.action_horizon, self.action_dim))
        
        # 3. Action Generation based on prefill results
        actions = self.sample_actions_with_kv(
            rng_action, observation, prefill_result, num_steps=num_steps, noise=noise
        )

        return actions, output_tokens

    def sample_text_actions_incremental(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        text_states: list[IncrementalTextState | None],
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        tokens_per_frame: int = 5,
        max_decoding_steps: int = 20,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> tuple[_model.Actions, list[jax.Array], list[IncrementalTextState], list[bool]]:
        """Sample actions and generate text incrementally for continuous batching.

        This method enables continuous batching: it generates partial text for each
        observation while still generating full actions per frame.

        Args:
            rng: Random number generator.
            observation: The observation (batched).
            text_states: List of IncrementalTextState for each batch element (None for new requests).
            num_steps: Number of action denoising steps.
            noise: Initial noise for actions.
            tokens_per_frame: Number of text tokens to generate this frame.
            max_decoding_steps: Maximum total text tokens.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Text sampling temperature.

        Returns:
            Tuple of:
                - actions: [B, action_horizon, action_dim]
                - tokens_generated_this_frame: List of [tokens_generated] for each batch element
                - updated_states: List of IncrementalTextState for each batch element
                - finished_flags: List of bool indicating if text generation finished
        """
        rng_text, rng_action = jax.random.split(rng)
        batch_size = observation.state.shape[0]

        # 1. Prefill for new requests and prepare for text generation
        prefill_results = []
        is_new_request = []

        for i, state in enumerate(text_states):
            if state is None:
                # New request: need prefill
                # Extract single observation from batch
                obs_single = jax.tree.map(lambda x: x[i:i+1] if hasattr(x, 'shape') else x, observation)
                prefill_result = self.prefill(obs_single, align_right=False, max_decoding_steps=max_decoding_steps)
                prefill_results.append(prefill_result)
                is_new_request.append(True)
            else:
                # Existing request: use saved state
                prefill_results.append(state)
                is_new_request.append(False)

        # 2. Generate text incrementally for each request
        tokens_generated_list = []
        updated_states_list = []
        finished_list = []

        for i, (prefill_or_state, is_new) in enumerate(zip(prefill_results, is_new_request)):
            rng_text_i = jax.random.fold_in(rng_text, i)
            tokens_gen, updated_state, finished = self.sample_text_incremental(
                rng_text_i,
                prefill_or_state,
                tokens_to_generate=tokens_per_frame,
                PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
                temperature=temperature,
                is_initial=is_new,
            )
            tokens_generated_list.append(tokens_gen)
            updated_states_list.append(updated_state)
            finished_list.append(finished)

        # 3. Generate actions using prefill (reuse prefill for efficiency)
        # For new requests, use the prefill we just computed
        # For existing requests, we need to do prefill again (or cache it)
        # For simplicity, let's do prefill for all (can optimize later)

        if noise is None:
            noise = jax.random.normal(rng_action, (batch_size, self.action_horizon, self.action_dim))

        # Do prefill for actions (can be same as text prefill for new requests)
        # For now, do fresh prefill for all
        observation = _model.preprocess_observation(None, observation, train=False)
        prefill_result_actions = self.prefill(observation, align_right=False, max_decoding_steps=0)
        actions = self.sample_actions_with_kv(
            rng_action, observation, prefill_result_actions, num_steps=num_steps, noise=noise
        )

        return actions, tokens_generated_list, updated_states_list, finished_list

    def profile_sample_text(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        max_decoding_steps: int = 20,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> tuple[at.Int[at.Array, "b s"], dict[str, float]]:
        """Profile text sampling.

        Args:
            rng: Random number generator.
            observation: The observation.
            max_decoding_steps: Maximum decoding steps.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Sampling temperature.

        Returns:
            Tuple of (output_tokens, timings).
        """
        t0 = time.time()
        
        prefill_result = self.prefill(observation, align_right=False, max_decoding_steps=max_decoding_steps)
        sync(prefill_result)
        t1 = time.time()
        
        output_tokens, kv_cache, mask, ar_mask = self.sample_text_with_kv(
            rng, prefill_result, max_decoding_steps=max_decoding_steps, PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN, temperature=temperature
        )
        sync(output_tokens)
        t2 = time.time()
        
        return output_tokens, {"prefill": t1-t0, "text": t2-t1, "total": t2-t0}

    def profile_sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, dict[str, float]]:
        """Profile action sampling.

        Args:
            rng: Random number generator.
            observation: The observation.
            num_steps: Number of denoising steps.
            noise: Initial noise.

        Returns:
            Tuple of (actions, timings).
        """
        t0 = time.time()
        
        # 1. Prefill
        # Equivalent to sample_actions embedded logic: align_right=False
        prefill_result = self.prefill(observation, align_right=False, max_decoding_steps=0)
        sync(prefill_result)
        t1 = time.time()
        
        # 2. Action Generation
        actions = self.sample_actions_with_kv(
            rng, observation, prefill_result, num_steps=num_steps, noise=noise
        )
        sync(actions)
        t2 = time.time()
        
        return actions, {"prefill": t1 - t0, "actions": t2 - t1, "total": t2 - t0}

    def prefile_sample_text_actions_shared_kv(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        max_decoding_steps: int = 20,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> tuple[_model.Actions, at.Int[at.Array, "b s"], dict[str, float]]:
        """Profile text and action sampling with shared KV.

        Args:
            rng: Random number generator.
            observation: The observation.
            num_steps: Number of denoising steps.
            noise: Initial noise.
            max_decoding_steps: Maximum text decoding steps.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Sampling temperature.

        Returns:
            Tuple of (actions, output_tokens, timings).
        """
        rng_text, rng_action = jax.random.split(rng)
        t0 = time.time()
        
        # 1. Prefill
        prefill_result = self.prefill(observation, align_right=False, max_decoding_steps=max_decoding_steps)
        sync(prefill_result)
        t1 = time.time()
        
        # 2. Text Generation
        output_tokens, kv_cache, mask, ar_mask = self.sample_text_with_kv(
            rng_text, prefill_result, max_decoding_steps=max_decoding_steps, PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN, temperature=temperature
        )
        sync(output_tokens)
        t2 = time.time()
        
        # 3. Action Generation
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng_action, (batch_size, self.action_horizon, self.action_dim))
        
        actions = self.sample_actions_with_kv(
            rng_action, observation, prefill_result, num_steps=num_steps, noise=noise
        )
        sync(actions)
        t3 = time.time()
        
        timings = {
            "prefill": t1 - t0,
            "text": t2 - t1,
            "actions": t3 - t2,
            "total": t3 - t0
        }
        return actions, output_tokens, timings

    # Verify if sample with kv produces correct results
    def verify_unified_prefill(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> dict:
        """Verify that prefill produces correct results.

        Compares three paths against the baseline ``sample_actions``:
        - text: sample_text vs sample_text_with_kv (token equality)
        - action_kv: sample_actions_with_kv (left-aligned, no pad)
        - action_shared_kv: full sample_text_actions_shared_kv path

        Returns a dict with per-comparison diff arrays and bool matches.
        """
        batch_size = observation.state.shape[0]

        # --- 1. Text generation ---
        rng, rng_text = jax.random.split(rng)
        output_tokens_orig, _, mask_orig, _ = self.sample_text(rng_text, observation)
        prefill_with_decoding = self.prefill(observation, align_right=False, max_decoding_steps=20)
        output_tokens_kv, _, mask_kv, _ = self.sample_text_with_kv(rng_text, prefill_with_decoding)
        text_match = bool(
            jnp.array_equal(output_tokens_orig, output_tokens_kv)
            and jnp.array_equal(mask_orig, mask_kv)
        )

        # --- 2. Action comparisons (shared noise) ---
        rng, rng_action = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(rng_action, (batch_size, self.action_horizon, self.action_dim))

        # A. Baseline: sample_actions (left-aligned internally)
        actions_baseline = self.sample_actions(rng_action, observation, noise=noise)

        # B. Left-aligned KV reuse (no padding) — should be identical
        prefill_no_pad = self.prefill(observation, align_right=False, max_decoding_steps=0)
        actions_kv = self.sample_actions_with_kv(rng_action, observation, prefill_no_pad, noise=noise)
        diff_kv = jnp.abs(actions_baseline - actions_kv)

        # C. Left-aligned KV + padding (from text decoding path)
        actions_with_pad = self.sample_actions_with_kv(rng_action, observation, prefill_with_decoding, noise=noise)
        diff_with_pad = jnp.abs(actions_baseline - actions_with_pad)

        # D. Full shared-KV path (text + actions)
        actions_shared, _ = self.sample_text_actions_shared_kv(rng_action, observation, noise=noise)
        diff_shared = jnp.abs(actions_baseline - actions_shared)

        return {
            "text_match": text_match,
            "diff_kv": diff_kv,
            "diff_with_pad": diff_with_pad,
            "diff_shared_kv": diff_shared,
            "actions_baseline": actions_baseline,
            "actions_kv": actions_kv,
            "actions_with_pad": actions_with_pad,
            "actions_shared_kv": actions_shared,
        }
