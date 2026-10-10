import logging
import math
import os
from dataclasses import dataclass
from importlib import util
from pathlib import Path
import shutil
import site

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.cache_utils import Cache

import openpi.models.gemma as _gemma
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def ensure_transformers_replace_installed():
    """Install the local Transformers overrides into the active environment if needed."""
    spec = util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise ImportError("transformers must be installed before using the PyTorch model.")

    transformers_dir = Path(spec.origin).parent
    replace_dir = Path(__file__).parent / "transformers_replace"

    if not replace_dir.exists():
        raise RuntimeError(f"Missing transformers_replace overlay at {replace_dir}")

    try:
        site_packages = [Path(p).resolve() for p in site.getsitepackages()]
        site_packages.append(Path(site.getusersitepackages()).resolve())
        if not any(transformers_dir.resolve().is_relative_to(site_pkg) for site_pkg in site_packages):
            raise PermissionError(f"Refusing to write outside site-packages: {transformers_dir}")

        for src in replace_dir.rglob("*.py"):
            rel = src.relative_to(replace_dir)
            dst = transformers_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except OSError as exc:
        raise RuntimeError(
            "The PyTorch model requires the local transformers_replace overlay. "
            f"Could not install it into {transformers_dir}: {exc}"
        ) from exc


ensure_transformers_replace_installed()

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel  # noqa: E402


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _warn_ignored_compile_flags() -> None:
    ignored = [
        name
        for name in (
            "OPENPI_TORCH_COMPILE_DENOISE",
            "OPENPI_TORCH_COMPILE_PREFILL",
            "OPENPI_TORCH_COMPILE_TEXT_DECODE",
            "OPENPI_TORCH_COMPILE_DENOISE_CUDAGRAPHS",
            "OPENPI_TORCH_COMPILE_MODE",
        )
        if name in os.environ
    ]
    if ignored:
        logging.warning(
            "Ignoring %s because OPENPI_TORCH_COMPILE is not enabled.",
            ", ".join(ignored),
        )


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


@dataclass
class PytorchIncrementalTextState:
    """State needed to resume PyTorch text generation across frames."""

    past_key_values: object
    last_logits: torch.Tensor
    output_tokens: torch.Tensor
    current_step: torch.Tensor
    is_finished: torch.Tensor
    prefix_mask: torch.Tensor
    prefill_len: torch.Tensor
    max_decoding_steps: int
    prefill_size: int
    cache_size: int


class PI05StaticTextCache(Cache):
    """Fixed-size KV cache with per-batch decode positions for PI05 text generation."""

    is_compileable = True

    def __init__(
        self,
        dynamic_cache,
        *,
        max_decoding_steps: int,
        prefix_mask: torch.Tensor,
    ):
        super().__init__()
        self.max_decoding_steps = max_decoding_steps
        self.prefix_size = prefix_mask.shape[1]
        self.cache_size = self.prefix_size + max_decoding_steps
        self.prefix_mask = prefix_mask
        self.key_cache = []
        self.value_cache = []
        for key, value in zip(dynamic_cache.key_cache, dynamic_cache.value_cache, strict=True):
            key_static = key.new_zeros((key.shape[0], key.shape[1], self.cache_size, key.shape[3]))
            value_static = value.new_zeros((value.shape[0], value.shape[1], self.cache_size, value.shape[3]))
            key_static[:, :, : self.prefix_size, :].copy_(key)
            value_static[:, :, : self.prefix_size, :].copy_(value)
            self.key_cache.append(key_static)
            self.value_cache.append(value_static)

    @classmethod
    def from_tensors(
        cls,
        key_cache: list[torch.Tensor],
        value_cache: list[torch.Tensor],
        *,
        max_decoding_steps: int,
        prefix_mask: torch.Tensor,
    ):
        cache = cls.__new__(cls)
        Cache.__init__(cache)
        cache.max_decoding_steps = max_decoding_steps
        cache.prefix_size = prefix_mask.shape[1]
        cache.cache_size = cache.prefix_size + max_decoding_steps
        cache.prefix_mask = prefix_mask
        cache.key_cache = key_cache
        cache.value_cache = value_cache
        return cache

    def __len__(self):
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if cache_kwargs is None:
            cache_kwargs = {}
        cache_position = cache_kwargs.get("cache_position")
        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        key_states = key_states.to(key_cache.dtype)
        value_states = value_states.to(value_cache.dtype)

        if cache_position is None:
            key_cache[:, :, : key_states.shape[2], :].copy_(key_states)
            value_cache[:, :, : value_states.shape[2], :].copy_(value_states)
        elif cache_position.ndim == 1:
            key_cache.index_copy_(2, cache_position, key_states)
            value_cache.index_copy_(2, cache_position, value_states)
        else:
            batch_indices = torch.arange(key_cache.shape[0], device=key_cache.device)[:, None]
            for seq_idx in range(cache_position.shape[1]):
                pos = cache_position[:, seq_idx]
                key_cache[batch_indices[:, 0], :, pos, :] = key_states[:, :, seq_idx, :]
                value_cache[batch_indices[:, 0], :, pos, :] = value_states[:, :, seq_idx, :]
        return key_cache, value_cache

    def get_seq_length(self, layer_idx: int = 0):
        return self.cache_size

    def get_mask_sizes(self, cache_position, layer_idx: int):
        return self.cache_size, 0

    def get_max_cache_shape(self):
        return self.cache_size

    def reorder_cache(self, beam_idx: torch.LongTensor):
        self.key_cache = [key.index_select(0, beam_idx.to(key.device)) for key in self.key_cache]
        self.value_cache = [value.index_select(0, beam_idx.to(value.device)) for value in self.value_cache]

    def batch_split(self, full_batch_size: int, split_size: int):
        out = []
        for i in range(0, full_batch_size, split_size):
            out.append(
                type(self).from_tensors(
                    [tensor[i : i + split_size].clone() for tensor in self.key_cache],
                    [tensor[i : i + split_size].clone() for tensor in self.value_cache],
                    max_decoding_steps=self.max_decoding_steps,
                    prefix_mask=self.prefix_mask[i : i + split_size].clone(),
                )
            )
        return out

    @classmethod
    def from_batch_splits(cls, splits: list["PI05StaticTextCache"]):
        if not splits:
            raise ValueError("Cannot stack empty PI05StaticTextCache splits.")
        return cls.from_tensors(
            [torch.cat([split.key_cache[i] for split in splits], dim=0) for i in range(len(splits[0]))],
            [torch.cat([split.value_cache[i] for split in splits], dim=0) for i in range(len(splits[0]))],
            max_decoding_steps=splits[0].max_decoding_steps,
            prefix_mask=torch.cat([split.prefix_mask for split in splits], dim=0),
        )


class PI05Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not config.pi05:
            raise ValueError("PI05Pytorch requires a pi05 model config.")
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

        compile_all = _env_flag("OPENPI_TORCH_COMPILE")
        if not compile_all:
            _warn_ignored_compile_flags()

        if compile_all and _env_flag("OPENPI_TORCH_COMPILE_DENOISE", default=True):
            mode = os.environ.get("OPENPI_TORCH_COMPILE_MODE", "reduce-overhead")
            denoise_cudagraphs = _env_flag("OPENPI_TORCH_COMPILE_DENOISE_CUDAGRAPHS")
            if denoise_cudagraphs:
                logging.info("Compiling PI05 denoise_step with mode=%s and CUDA graphs enabled.", mode)
                self.denoise_step = torch.compile(self.denoise_step, mode=mode)
            else:
                if mode != "reduce-overhead":
                    logging.warning(
                        "OPENPI_TORCH_COMPILE_MODE=%s is ignored for denoise because "
                        "denoise compile disables CUDA graphs via options.",
                        mode,
                    )
                self.denoise_step = torch.compile(
                    self.denoise_step,
                    options={"triton.cudagraphs": False},
                )
                logging.info("Compiling PI05 denoise_step with CUDA graphs disabled.")
        if compile_all and _env_flag("OPENPI_TORCH_COMPILE_PREFILL", default=True):
            logging.info("Compiling PI05 VLM/language prefill with CUDA graphs disabled.")
            self._prefill_language_model = torch.compile(
                self._prefill_language_model,
                options={"triton.cudagraphs": False},
            )
        if compile_all and _env_flag("OPENPI_TORCH_COMPILE_TEXT_DECODE", default=True):
            logging.info("Compiling PI05 one-token language decode with CUDA graphs disabled.")
            self._decode_one_token = torch.compile(
                self._decode_one_token,
                options={"triton.cudagraphs": False},
            )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _deembed_prefix(self, hidden_states):
        return self.paligemma_with_expert.paligemma.lm_head(hidden_states)

    def _prefill_language_model(self, prefix_embs, attention_mask, position_ids):
        return self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            cache_position=None,
            adarms_cond=[None, None],
        )

    def _decode_one_token(self, token_embedding, attention_mask, position_ids, cache_position, past_key_values):
        return self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[token_embedding, None],
            use_cache=True,
            cache_position=cache_position,
            adarms_cond=[None, None],
        )

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [1] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def prefill(self, observation, max_decoding_steps: int = 0):
        """Prefill the language KV cache for text generation."""
        images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_out, _), past_key_values = self._prefill_language_model(
            prefix_embs, prefix_att_2d_masks_4d, prefix_position_ids
        )
        return prefix_out, past_key_values, prefix_pad_masks, max_decoding_steps

    def init_incremental_state(self, prefill_result) -> PytorchIncrementalTextState:
        """Initialize resumable text-generation state from a PyTorch prefill result."""
        prefix_out, past_key_values, prefix_mask, max_decoding_steps = prefill_result
        batch_size = prefix_out.shape[0]
        prefill_len = torch.sum(prefix_mask.to(torch.long), dim=-1)
        batch_idx = torch.arange(batch_size, device=prefix_out.device)
        last_token_embedding = prefix_out[batch_idx, prefill_len - 1][:, None, :]
        last_logits = F.log_softmax(self._deembed_prefix(last_token_embedding), dim=-1)
        output_tokens = torch.zeros(
            (batch_size, max_decoding_steps),
            dtype=torch.long,
            device=prefix_out.device,
        )
        return PytorchIncrementalTextState(
            past_key_values=past_key_values,
            last_logits=last_logits,
            output_tokens=output_tokens,
            current_step=torch.zeros(batch_size, dtype=torch.long, device=prefix_out.device),
            is_finished=torch.zeros(batch_size, dtype=torch.bool, device=prefix_out.device),
            prefix_mask=prefix_mask,
            prefill_len=prefill_len,
            max_decoding_steps=max_decoding_steps,
            prefill_size=prefix_mask.shape[1],
            cache_size=prefix_mask.shape[1] + max_decoding_steps,
        )

    def init_static_incremental_state(self, prefill_result) -> PytorchIncrementalTextState:
        """Initialize text state with fixed-size KV cache for batched/compiled decoding."""
        state = self.init_incremental_state(prefill_result)
        state.past_key_values = PI05StaticTextCache(
            state.past_key_values,
            max_decoding_steps=state.max_decoding_steps,
            prefix_mask=state.prefix_mask,
        )
        return state

    def generate_n_tokens(
        self,
        state: PytorchIncrementalTextState,
        *,
        tokens_to_generate: int = 5,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, PytorchIncrementalTextState, torch.Tensor]:
        """Generate a fixed number of language tokens from a PyTorch incremental state."""
        tokens_this_call = []
        batch_size = state.output_tokens.shape[0]
        device = state.output_tokens.device

        for _ in range(tokens_to_generate):
            should_update = (~state.is_finished) & (state.current_step < state.max_decoding_steps)

            if temperature > 0.0:
                probs = F.softmax(state.last_logits[:, -1, :] / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1).to(torch.long)
            else:
                token = torch.argmax(state.last_logits[:, -1, :], dim=-1, keepdim=True).to(torch.long)

            token_flat = token[:, 0]
            write_step = torch.clamp(state.current_step, max=state.max_decoding_steps - 1)
            updated_output_tokens = state.output_tokens.scatter(1, write_step[:, None], token)
            state.output_tokens = torch.where(should_update[:, None], updated_output_tokens, state.output_tokens)
            tokens_this_call.append(torch.where(should_update, token_flat, torch.zeros_like(token_flat)))

            has_eos = token_flat == PALIGEMMA_EOS_TOKEN
            next_finished = state.is_finished | has_eos | (state.current_step + 1 >= state.max_decoding_steps)

            token_embedding = self.paligemma_with_expert.embed_language_tokens(token)
            token_embedding = token_embedding * math.sqrt(token_embedding.shape[-1])
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                token_embedding = token_embedding.to(dtype=torch.bfloat16)

            decoded_len = state.max_decoding_steps
            decoded_positions = torch.arange(decoded_len, device=device)[None, :]
            decoded_mask = decoded_positions <= state.current_step[:, None]
            attention_mask = torch.cat([state.prefix_mask, decoded_mask], dim=1)[:, None, :]
            attention_mask = self._prepare_attention_masks_4d(attention_mask)
            position_ids = (state.prefill_len + state.current_step)[:, None]
            cache_position = (state.prefix_mask.shape[1] + state.current_step)[:, None]
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            (prefix_out, _), past_key_values = self._decode_one_token(
                token_embedding, attention_mask, position_ids, cache_position, state.past_key_values
            )
            state.past_key_values = past_key_values
            state.last_logits = F.log_softmax(self._deembed_prefix(prefix_out[:, -1:]), dim=-1)
            state.current_step = state.current_step + should_update.to(torch.long)
            state.is_finished = next_finished

        if tokens_this_call:
            tokens_generated = torch.stack(tokens_this_call, dim=1)
        else:
            tokens_generated = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        all_finished = torch.all(state.is_finished | (state.current_step >= state.max_decoding_steps))
        return tokens_generated, state, all_finished

    @torch.no_grad()
    def sample_text(
        self,
        device,
        observation,
        max_decoding_steps: int = 20,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, object, torch.Tensor, torch.Tensor]:
        """Sample text tokens from an observation."""
        prefill_result = self.prefill(observation, max_decoding_steps=max_decoding_steps)
        state = self.init_incremental_state(prefill_result)
        tokens, state, _ = self.generate_n_tokens(
            state,
            tokens_to_generate=max_decoding_steps,
            PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
            temperature=temperature,
        )
        mask = torch.cat([state.prefix_mask, state.output_tokens != 0], dim=1)
        ar_mask = torch.cat(
            [
                torch.zeros(state.prefix_mask.shape[1], dtype=torch.bool, device=device),
                torch.ones(max_decoding_steps, dtype=torch.bool, device=device),
            ],
            dim=0,
        )
        return tokens, state.past_key_values, mask, ar_mask

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            if noisy_actions.dtype != self.action_in_proj.weight.dtype:
                noisy_actions = noisy_actions.to(self.action_in_proj.weight.dtype)
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        # time MLP (for adaRMS)
        def time_mlp_func(time_emb):
            if time_emb.dtype != self.time_mlp_in.weight.dtype:
                time_emb = time_emb.to(self.time_mlp_in.weight.dtype)
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)  # swish == silu
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t.to(u_t.dtype), reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    @torch.no_grad()
    def sample_actions_with_kv(self, device, observation, prefill_result, noise=None, num_steps=10) -> Tensor:
        """Sample actions using an existing language prefix KV cache."""
        _, past_key_values, prefix_pad_masks, _ = prefill_result
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        _, _, _, _, state = self._preprocess_observation(observation, train=False)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        if suffix_out.dtype != self.action_out_proj.weight.dtype:
            suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out).to(dtype=torch.float32)
