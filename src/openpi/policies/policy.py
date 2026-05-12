from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, Optional, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.models.pi05 import IncrementalTextState
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _stack_incremental_states(states: list[IncrementalTextState]) -> IncrementalTextState:
    """Stack multiple IncrementalTextStates into a single batched state.

    All states must have the same metadata (prefill_size, max_decoding_steps, cache_size).
    This is required for efficient batched token generation.

    Args:
        states: List of IncrementalTextState objects, each with batch_size=1

    Returns:
        Single IncrementalTextState with batch_size=len(states)
    """
    if not states:
        raise ValueError("Cannot stack empty list of states")

    # Verify all states have compatible metadata
    ref = states[0]
    for s in states[1:]:
        if s.prefill_size != ref.prefill_size:
            raise ValueError(f"Incompatible prefill_size: {s.prefill_size} vs {ref.prefill_size}")
        if s.max_decoding_steps != ref.max_decoding_steps:
            raise ValueError(f"Incompatible max_decoding_steps: {s.max_decoding_steps} vs {ref.max_decoding_steps}")
        if s.cache_size != ref.cache_size:
            raise ValueError(f"Incompatible cache_size: {s.cache_size} vs {ref.cache_size}")

    # Stack array fields along batch dimension
    # KV cache has structure: (idx[layers, batch], k[layers, batch, ...], v[layers, batch, ...])
    # Batch dimension is axis=1 for KV cache arrays
    kv_cache_stacked = tuple(
        jnp.concatenate([s.kv_cache[i] for s in states], axis=1)
        for i in range(len(ref.kv_cache))
    )

    return IncrementalTextState(
        rng=jnp.concatenate([s.rng for s in states], axis=0),
        last_logits=jnp.concatenate([s.last_logits for s in states], axis=0),
        output_tokens=jnp.concatenate([s.output_tokens for s in states], axis=0),
        kv_cache=kv_cache_stacked,
        current_step=jnp.concatenate([s.current_step for s in states], axis=0),
        is_finished=jnp.concatenate([s.is_finished for s in states], axis=0),
        prefill_len=jnp.concatenate([s.prefill_len for s in states], axis=0),
        # Metadata (static)
        prefill_size=ref.prefill_size,
        max_decoding_steps=ref.max_decoding_steps,
        cache_size=ref.cache_size,
    )


def _split_incremental_state(state: IncrementalTextState, batch_size: int) -> list[IncrementalTextState]:
    """Split a batched IncrementalTextState into individual states.

    Args:
        state: Batched IncrementalTextState
        batch_size: Number of states to split into

    Returns:
        List of IncrementalTextState objects, each with batch_size=1
    """
    states = []
    for i in range(batch_size):
        # KV cache has batch at axis=1: (layers, batch, ...)
        kv_cache_i = tuple(kv[:, i:i+1] for kv in state.kv_cache)
        states.append(IncrementalTextState(
            rng=state.rng[i:i+1],
            last_logits=state.last_logits[i:i+1],
            output_tokens=state.output_tokens[i:i+1],
            kv_cache=kv_cache_i,
            current_step=state.current_step[i:i+1],
            is_finished=state.is_finished[i:i+1],
            prefill_len=state.prefill_len[i:i+1],
            prefill_size=state.prefill_size,
            max_decoding_steps=state.max_decoding_steps,
            cache_size=state.cache_size,
        ))
    return states


def _split_pytorch_incremental_state(state, batch_size: int) -> list:
    """Split a batched PyTorch incremental text state into per-request states."""
    cache_splits = state.past_key_values.batch_split(batch_size, 1)
    cls = type(state)
    return [
        cls(
            past_key_values=_clone_pytorch_cache(cache_splits[i]),
            last_logits=state.last_logits[i:i + 1],
            output_tokens=state.output_tokens[i:i + 1].clone(),
            current_step=state.current_step[i:i + 1].clone(),
            is_finished=state.is_finished[i:i + 1].clone(),
            prefix_mask=state.prefix_mask[i:i + 1].clone(),
            prefill_len=state.prefill_len[i:i + 1].clone(),
            max_decoding_steps=state.max_decoding_steps,
            prefill_size=state.prefill_size,
            cache_size=state.cache_size,
        )
        for i in range(batch_size)
    ]


def _stack_pytorch_incremental_states(states: list):
    """Stack per-request PyTorch incremental states for batched text decoding."""
    if not states:
        raise ValueError("Cannot stack empty list of PyTorch incremental states.")
    ref = states[0]
    for state in states[1:]:
        if state.prefill_size != ref.prefill_size:
            raise ValueError(f"Incompatible prefill_size: {state.prefill_size} vs {ref.prefill_size}")
        if state.max_decoding_steps != ref.max_decoding_steps:
            raise ValueError(
                f"Incompatible max_decoding_steps: {state.max_decoding_steps} vs {ref.max_decoding_steps}"
            )
        if state.cache_size != ref.cache_size:
            raise ValueError(f"Incompatible cache_size: {state.cache_size} vs {ref.cache_size}")

    cache_cls = type(ref.past_key_values)
    if hasattr(cache_cls, "from_batch_splits"):
        past_key_values = cache_cls.from_batch_splits([state.past_key_values for state in states])
    else:
        from transformers.cache_utils import DynamicCache

        past_key_values = DynamicCache.from_batch_splits([state.past_key_values for state in states])

    cls = type(ref)
    return cls(
        past_key_values=past_key_values,
        last_logits=torch.cat([state.last_logits for state in states], dim=0),
        output_tokens=torch.cat([state.output_tokens for state in states], dim=0),
        current_step=torch.cat([state.current_step for state in states], dim=0),
        is_finished=torch.cat([state.is_finished for state in states], dim=0),
        prefix_mask=torch.cat([state.prefix_mask for state in states], dim=0),
        prefill_len=torch.cat([state.prefill_len for state in states], dim=0),
        max_decoding_steps=ref.max_decoding_steps,
        prefill_size=ref.prefill_size,
        cache_size=ref.cache_size,
    )


def _to_jax_batch(x, add_batch_dim: bool = True):
    """Convert observation to JAX array, optionally adding batch dimension."""
    if isinstance(x, (str, bytes)):
        return x
    arr = jnp.asarray(x)
    if add_batch_dim:
        return arr[np.newaxis, ...]
    return arr


def _torch_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _clone_pytorch_cache(cache):
    """Clone a Hugging Face DynamicCache without sharing tensor storage."""
    if hasattr(type(cache), "from_tensors") and hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return type(cache).from_tensors(
            [tensor.clone() for tensor in cache.key_cache],
            [tensor.clone() for tensor in cache.value_cache],
            max_decoding_steps=cache.max_decoding_steps,
            prefix_mask=cache.prefix_mask.clone(),
        )
    if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
        return cache
    cloned = type(cache)()
    if hasattr(cache, "_seen_tokens"):
        cloned._seen_tokens = cache._seen_tokens
    cloned.key_cache = [tensor.clone() for tensor in cache.key_cache]
    cloned.value_cache = [tensor.clone() for tensor in cache.value_cache]
    return cloned


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            if hasattr(model, "sample_text"):
                self._sample_text = model.sample_text
            if hasattr(model, "prefill"):
                self._prefill = model.prefill
            if hasattr(model, "init_incremental_state"):
                self._init_incremental_state = model.init_incremental_state
            if hasattr(model, "init_static_incremental_state"):
                self._init_static_incremental_state = model.init_static_incremental_state
            elif hasattr(model, "init_incremental_state"):
                self._init_static_incremental_state = model.init_incremental_state
            if hasattr(model, "generate_n_tokens"):
                self._generate_n_tokens = model.generate_n_tokens
            if hasattr(model, "sample_actions_with_kv"):
                self._sample_actions_with_kv = model.sample_actions_with_kv
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            if hasattr(model, "prefill"):
                self._prefill = nnx_utils.module_jit(
                    model.prefill,
                    static_argnames=("align_right", "max_decoding_steps")
                )
            if hasattr(model, "sample_text_with_kv"):
                self._sample_text_with_kv = nnx_utils.module_jit(
                    model.sample_text_with_kv,
                    static_argnames=("max_decoding_steps", "PALIGEMMA_EOS_TOKEN", "temperature")
                )
            if hasattr(model, "sample_text"):
                self._sample_text = nnx_utils.module_jit(
                    model.sample_text,
                    static_argnames=("max_decoding_steps", "PALIGEMMA_EOS_TOKEN", "temperature")
                )
            if hasattr(model, "sample_text_actions_shared_kv"):
                self._sample_text_actions_shared_kv = nnx_utils.module_jit(
                    model.sample_text_actions_shared_kv,
                    static_argnames=("num_steps", "max_decoding_steps", "PALIGEMMA_EOS_TOKEN", "temperature")
                )
            if hasattr(model, "init_incremental_state"):
                self._init_incremental_state = nnx_utils.module_jit(
                    model.init_incremental_state,
                )
            if hasattr(model, "generate_n_tokens"):
                self._generate_n_tokens = nnx_utils.module_jit(
                    model.generate_n_tokens,
                    static_argnames=("tokens_to_generate", "PALIGEMMA_EOS_TOKEN", "temperature"),
                )
            if hasattr(model, "sample_actions_with_kv"):
                self._sample_actions_with_kv = nnx_utils.module_jit(
                    model.sample_actions_with_kv,
                    static_argnames=("num_steps",),
                )
            self._rng = rng or jax.random.key(0)

        # Cache the tokenizer to avoid repeated initialization
        self._tokenizer = _tokenizer.PaligemmaTokenizer()

    def _to_torch_tree(self, inputs: dict, *, add_batch_dim: bool = False) -> dict:
        """Convert transformed inputs to torch tensors on the configured device."""

        def _to_torch(x):
            if isinstance(x, (str, bytes)):
                return x
            if isinstance(x, list):
                return x
            if isinstance(x, torch.Tensor):
                tensor = x.to(self._pytorch_device)
            else:
                tensor = torch.as_tensor(np.array(x, copy=True), device=self._pytorch_device)
            if tensor.dtype in (torch.int8, torch.int16, torch.int32):
                tensor = tensor.to(torch.long)
            if add_batch_dim:
                tensor = tensor[None, ...]
            return tensor

        return jax.tree.map(_to_torch, inputs)

    def _sync_torch(self):
        if str(self._pytorch_device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(self._pytorch_device))

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Infer actions from observation dictionary.

        Args:
            obs: The observation dictionary.
            noise: Optional noise for sampling.

        Returns:
            Dictionary with actions and metadata.
        """
        return self.infer_actions(obs, noise=noise)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @override
    def infer_rtc(self, obs: dict, prefix_actions: jax.Array, inference_delay: int, prefix_attention_horizon: int, max_guidance_weight: float) -> dict:  # type: ignore[misc]
        """Infer actions with real-time control parameters.

        Args:
            obs: The observation dictionary.
            prefix_actions: Prefix actions for guidance.
            inference_delay: Inference delay.
            prefix_attention_horizon: Attention horizon.
            max_guidance_weight: Maximum guidance weight.

        Returns:
            Dictionary with actions and metadata.
        """
        t0 = time.monotonic()
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(_to_jax_batch, inputs)
        t1 = time.monotonic()

        self._rng, sample_rng = jax.random.split(self._rng) 
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions_rtc(sample_rng, _model.Observation.from_dict(inputs), prefix_actions, inference_delay, prefix_attention_horizon, max_guidance_weight),
        }
        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        # Ensure computation is finished for accurate timing
        if hasattr(outputs["actions"], "block_until_ready"):
            outputs["actions"].block_until_ready()
        t2 = time.monotonic()

        outputs = self._output_transform(outputs)
        t3 = time.monotonic()
        
        outputs["policy_timing"] = {
            "pre_proc_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_proc_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }
        outputs["policy_shapes"] = {
            "observation": jax.tree.map(lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs),
            "actions": tuple(outputs["actions"].shape),
        }
        return outputs

    def infer_actions(self, obs: dict, *, num_steps: int | None = None, noise: np.ndarray | None = None) -> dict:
        """Infer actions from observation dictionary.

        Args:
            obs: The observation dictionary.
            num_steps: Number of denoising steps.
            noise: Optional noise for sampling.

        Returns:
            Dictionary with actions and metadata.
        """
        t0 = time.monotonic()
        
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(_to_jax_batch, inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            def _to_torch_batch(x):
                if isinstance(x, (str, bytes)):
                    return x
                return torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...]
            inputs = jax.tree.map(_to_torch_batch, inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if num_steps is not None:
             sample_kwargs["num_steps"] = num_steps
             
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        t1 = time.monotonic()
        
        action_output = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        if isinstance(action_output, tuple):
            actions = action_output[0]
            output_tokens = action_output[1]
            tokenizer = self._tokenizer
            output_tokens = jnp.array(output_tokens, dtype=int)
        else:
            actions = action_output
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            # Ensure computation is finished for accurate timing
            if hasattr(outputs["actions"], "block_until_ready"):
                outputs["actions"].block_until_ready()

        t2 = time.monotonic()

        outputs = self._output_transform(outputs)
        t3 = time.monotonic()
        
        outputs["policy_timing"] = {
            "pre_proc_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_proc_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }
        outputs["policy_shapes"] = {
            "observation": jax.tree.map(lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs),
            "actions": tuple(outputs["actions"].shape),
        }
        return outputs

    def infer_text(self, obs: dict, max_decoding_steps: int = 25, temperature: float = 0.1, PALIGEMMA_EOS_TOKEN: int = -1) -> dict:  # type: ignore[misc]
        """Infer text from observation.

        Args:
            obs: The observation dictionary.
            max_decoding_steps: Maximum decoding steps.
            temperature: Sampling temperature.
            PALIGEMMA_EOS_TOKEN: End of sequence token. Default -1 (no early stopping).

        Returns:
            Dictionary with tokens and text.
        """
        if not hasattr(self, "_sample_text"):
            raise NotImplementedError("Model does not have sample_text method")

        t0 = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if self._is_pytorch_model:
            inputs = self._to_torch_tree(inputs, add_batch_dim=True)
            sample_rng_or_device = self._pytorch_device
        else:
            inputs = jax.tree.map(_to_jax_batch, inputs)
            self._rng, sample_rng_or_device = jax.random.split(self._rng)

        observation = _model.Observation.from_dict(inputs)
        t1 = time.monotonic()
        
        predicted_tokens, kv_cache, mask, ar_mask = self._sample_text(
            sample_rng_or_device,
            observation, 
            max_decoding_steps=max_decoding_steps, 
            PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN, 
            temperature=temperature
        )

        outputs = {
            "tokens": predicted_tokens,
        }
        # Unbatch and convert to np.ndarray.
        if self._is_pytorch_model:
            self._sync_torch()
            outputs = jax.tree.map(lambda x: _torch_to_numpy(x[0, ...]), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            if hasattr(outputs["tokens"], "block_until_ready"):
                outputs["tokens"].block_until_ready()
        t2 = time.monotonic()

        # Detokenize
        tokenizer = self._tokenizer
        text = tokenizer.detokenize(outputs["tokens"].astype(np.int32))
        outputs["text"] = text
        t3 = time.monotonic()

        outputs["policy_timing"] = {
            "pre_proc_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_proc_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }
        outputs["policy_shapes"] = {
            "observation": jax.tree.map(lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs),
            "tokens": tuple(outputs["tokens"].shape),
        }
        return outputs

    def infer_text_actions_shared_kv(self, obs: dict, num_steps: int = 10, max_decoding_steps: int = 20, noise: np.ndarray | None = None, PALIGEMMA_EOS_TOKEN: int = -1, temperature: float = 0.0) -> dict:
        """Infer text and actions with shared KV cache.

        Args:
            obs: The observation dictionary.
            num_steps: Number of denoising steps.
            max_decoding_steps: Maximum text decoding steps.
            noise: Optional noise.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Sampling temperature.

        Returns:
            Dictionary with actions, tokens, text, and timings.
        """
        t0 = time.monotonic()
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if self._is_pytorch_model:
            inputs = self._to_torch_tree(inputs, add_batch_dim=True)
            observation = _model.Observation.from_dict(inputs)
            if noise is not None:
                noise = torch.as_tensor(noise, device=self._pytorch_device)
                if noise.ndim == 2:
                    noise = noise[None, ...]

            t1 = time.monotonic()
            prefill_result = self._prefill(observation, max_decoding_steps=max_decoding_steps)
            actions = self._sample_actions_with_kv(
                self._pytorch_device,
                observation,
                prefill_result,
                num_steps=num_steps,
                noise=noise,
            )
            state = self._init_incremental_state(prefill_result)
            predicted_tokens, state, _ = self._generate_n_tokens(
                state,
                tokens_to_generate=max_decoding_steps,
                PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
                temperature=temperature,
            )
            self._sync_torch()
            outputs = {
                "state": _torch_to_numpy(inputs["state"][0]),
                "actions": _torch_to_numpy(actions[0]),
                "tokens": _torch_to_numpy(predicted_tokens[0]),
            }
            action_outputs = {k: v for k, v in outputs.items() if k in ["state", "actions"]}
            action_outputs = self._output_transform(action_outputs)
            outputs.update(action_outputs)
            t2 = time.monotonic()

            outputs["text"] = self._tokenizer.detokenize(outputs["tokens"].astype(np.int32))
            t3 = time.monotonic()
            outputs["policy_timing"] = {
                "pre_proc_ms": (t1 - t0) * 1000,
                "infer_ms": (t2 - t1) * 1000,
                "post_proc_ms": (t3 - t2) * 1000,
                "total_ms": (t3 - t0) * 1000,
                "backend": "pytorch",
            }
            outputs["policy_shapes"] = {
                "observation": jax.tree.map(lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs),
                "actions": tuple(outputs["actions"].shape),
                "tokens": tuple(outputs["tokens"].shape),
            }
            return outputs

        inputs = jax.tree.map(_to_jax_batch, inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        
        if noise is not None:
             noise = jnp.asarray(noise)
             if noise.ndim == 2: noise = noise[None, ...]
             
        observation = _model.Observation.from_dict(inputs)
        t1 = time.monotonic()
        
        actions, predicted_tokens = self._sample_text_actions_shared_kv(
            sample_rng, observation, num_steps=num_steps, max_decoding_steps=max_decoding_steps, noise=noise, PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN, temperature=temperature
        )
        
        outputs = {
            "state": inputs["state"],
            "actions": actions,
            "tokens": predicted_tokens
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        
        action_outputs = {k: v for k, v in outputs.items() if k in ["state", "actions"]}
        action_outputs = self._output_transform(action_outputs)
        outputs.update(action_outputs)
        # Ensure computation is finished for accurate timing
        if hasattr(outputs["actions"], "block_until_ready"):
            outputs["actions"].block_until_ready()
        if hasattr(outputs["tokens"], "block_until_ready"):
            outputs["tokens"].block_until_ready()
        t2 = time.monotonic()

        tokenizer = self._tokenizer
        text = tokenizer.detokenize(outputs["tokens"].astype(np.int32))
        outputs["text"] = text
        t3 = time.monotonic()
        
        outputs["policy_timing"] = {
            "pre_proc_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_proc_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }
        outputs["policy_shapes"] = {
            "observation": jax.tree.map(lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs),
            "actions": tuple(outputs["actions"].shape),
            "tokens": tuple(outputs["tokens"].shape),
        }
        return outputs

    def infer_profile_actions(self, obs: dict, num_steps: int = 10, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """Profile action inference.

        Args:
            obs: The observation dictionary.
            num_steps: Number of denoising steps.
            noise: Optional noise.

        Returns:
            Dictionary with actions and timings.
        """
        if not hasattr(self._model, "profile_sample_actions"):
            raise NotImplementedError("Model does not have profile_sample_actions method")

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(_to_jax_batch, inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        
        if noise is not None:
             noise = jnp.asarray(noise)
             if noise.ndim == 2:
                 noise = noise[None, ...]

        observation = _model.Observation.from_dict(inputs)
        
        # Call model profile method
        # Note: This assumes the model has been JIT-ed appropriately if high performance is expected
        actions, timings = self._model.profile_sample_actions(
            sample_rng, observation, num_steps=num_steps, noise=noise
        )

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        
        # Add timings
        outputs["policy_timing"] = {k: v * 1000 for k, v in timings.items()} # Convert s to ms
        outputs["policy_shapes"] = {
            "observation": jax.tree.map(lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs),
            "actions": tuple(outputs["actions"].shape),
        }
        return outputs

    def infer_profile_text(self, obs: dict, max_decoding_steps: int = 20, temperature: float = 0.0, PALIGEMMA_EOS_TOKEN: int = 1) -> dict:  # type: ignore[misc]
        """Profile text inference.

        Args:
            obs: The observation dictionary.
            max_decoding_steps: Maximum decoding steps.
            temperature: Sampling temperature.
            PALIGEMMA_EOS_TOKEN: End of sequence token.

        Returns:
            Dictionary with tokens, text, and timings.
        """
        if not hasattr(self._model, "profile_sample_text"):
            raise NotImplementedError("Model does not have profile_sample_text method")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(_to_jax_batch, inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(inputs)
        
        predicted_tokens, timings = self._model.profile_sample_text(
            sample_rng, observation, max_decoding_steps=max_decoding_steps, PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN, temperature=temperature
        )
        
        outputs = {
            "tokens": predicted_tokens,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        
        # Detokenize
        tokenizer = self._tokenizer
        text = tokenizer.detokenize(outputs["tokens"].astype(np.int32))
        outputs["text"] = text
        
        outputs["policy_timing"] = {k: v * 1000 for k, v in timings.items()}
        outputs["policy_shapes"] = {
            "observation": jax.tree.map(lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs),
            "tokens": tuple(outputs["tokens"].shape),
        }
        return outputs

    def infer_profile_text_actions_shared_kv(self, obs: dict, num_steps: int = 10, max_decoding_steps: int = 20, noise: np.ndarray | None = None, PALIGEMMA_EOS_TOKEN: int = -1, temperature: float = 0.0) -> dict:
        """Profile text and action inference with shared KV.

        Args:
            obs: The observation dictionary.
            num_steps: Number of denoising steps.
            max_decoding_steps: Maximum text decoding steps.
            noise: Optional noise.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Sampling temperature.

        Returns:
            Dictionary with actions, tokens, text, and timings.
        """
        if not hasattr(self._model, "prefile_sample_text_actions_shared_kv"):
             raise NotImplementedError("Model does not have prefile_sample_text_actions_shared_kv method")
        
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(_to_jax_batch, inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        
        if noise is not None:
             noise = jnp.asarray(noise)
             if noise.ndim == 2: noise = noise[None, ...]
             
        observation = _model.Observation.from_dict(inputs)
        
        actions, predicted_tokens, timings = self._model.prefile_sample_text_actions_shared_kv(
            sample_rng, observation, num_steps=num_steps, max_decoding_steps=max_decoding_steps, noise=noise, PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN, temperature=temperature
        )
        
        outputs = {
            "state": inputs["state"],
            "actions": actions,
            "tokens": predicted_tokens
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        
        # Handle Output Transform for actions
        # Split actions and other outputs because output_transform might only expect actions/state
        action_outputs = {k: v for k, v in outputs.items() if k in ["state", "actions"]}
        action_outputs = self._output_transform(action_outputs)
        outputs.update(action_outputs)

        # Detokenize
        tokenizer = self._tokenizer
        text = tokenizer.detokenize(outputs["tokens"].astype(np.int32))
        outputs["text"] = text
        
        outputs["policy_timing"] = {k: v * 1000 for k, v in timings.items()}
        outputs["policy_shapes"] = {
            "observation": jax.tree.map(lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs),
            "actions": tuple(outputs["actions"].shape),
            "tokens": tuple(outputs["tokens"].shape),
        }
        return outputs

    def _prepare_batched_inputs(
        self,
        obs_list: list[dict],
        allow_variable_length: bool = False,
        use_numpy_stack: bool = False,
    ) -> tuple[dict, dict]:
        """Prepare a batch of observations for inference.

        Args:
            obs_list: List of observation dicts
            allow_variable_length: If True, pad sequences to max length

        Returns:
            batched_obs: Batched observation dict
            metadata: Metadata including original lengths and batch_size
        """
        # Apply input transforms individually
        transformed_obs = []
        for obs in obs_list:
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self._input_transform(inputs)
            transformed_obs.append(inputs)

        sample_obs = transformed_obs[0]
        batched_inputs = {}
        metadata = {"batch_size": len(obs_list), "original_lengths": {}}

        for key in sample_obs:
            if isinstance(sample_obs[key], dict):
                # Nested dict (e.g., images)
                batched_inputs[key] = {}
                for nested_key in sample_obs[key]:
                    arrays = [obs[key][nested_key] for obs in transformed_obs]
                    batched_arr, arr_meta = self._batch_arrays(
                        arrays, f"{key}.{nested_key}", allow_variable_length, use_numpy_stack
                    )
                    batched_inputs[key][nested_key] = batched_arr
                    if "original_lengths" in arr_meta:
                        metadata["original_lengths"][f"{key}.{nested_key}"] = arr_meta["original_lengths"]
            elif isinstance(sample_obs[key], (str, bytes)):
                batched_inputs[key] = [obs[key] for obs in transformed_obs]
            else:
                arrays = [obs[key] for obs in transformed_obs]
                batched_arr, arr_meta = self._batch_arrays(arrays, key, allow_variable_length, use_numpy_stack)
                batched_inputs[key] = batched_arr
                if "original_lengths" in arr_meta:
                    metadata["original_lengths"][key] = arr_meta["original_lengths"]

        return batched_inputs, metadata

    def _batch_arrays(
        self,
        arrays: list,
        key: str,
        allow_variable_length: bool,
        use_numpy_stack: bool = False,
    ) -> tuple[jnp.ndarray | np.ndarray, dict]:
        """Batch arrays with optional padding.

        Args:
            arrays: List of arrays to batch
            key: Key name for error messages
            allow_variable_length: If True, pad to max length

        Returns:
            Batched array and metadata dict
        """
        if not arrays:
            return np.array([]), {}

        arrays = [np.asarray(arr) for arr in arrays]
        shapes = [arr.shape for arr in arrays]

        # Check if shapes match
        if all(s == shapes[0] for s in shapes):
            # Uniform shape - simple stack
            if use_numpy_stack:
                return np.stack(arrays, axis=0), {}
            return jnp.stack(arrays, axis=0), {}

        if not allow_variable_length:
            # Enforce matching shapes
            raise ValueError(
                f"Batched inference requires matching shapes for '{key}'. "
                f"Got shapes: {shapes}. Set allow_variable_length=True or pad inputs externally."
            )

        # Variable-length padding
        max_shape = tuple(max(s[i] for s in shapes) for i in range(len(shapes[0])))
        padded = []
        original_lengths = []

        for arr in arrays:
            pad_width = [(0, max_shape[i] - arr.shape[i]) for i in range(arr.ndim)]
            padded_arr = np.pad(arr, pad_width, constant_values=0)
            padded.append(padded_arr)
            original_lengths.append(arr.shape[0] if arr.ndim > 0 else 1)

        batched = np.stack(padded, axis=0) if use_numpy_stack else jnp.stack(padded, axis=0)
        metadata = {"original_lengths": original_lengths}

        return batched, metadata

    def _check_shapes_match(self, arrays: list, key: str):
        """Verify all arrays have the same shape."""
        if not arrays:
            return
        shapes = [np.asarray(arr).shape for arr in arrays]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(
                f"Batched inference requires matching shapes for '{key}'. "
                f"Got shapes: {shapes}. Pad inputs externally or use single inference."
            )

    def _unbatch_outputs(self, outputs: dict, metadata: dict) -> list[dict]:
        """Split batched outputs into individual outputs.

        Args:
            outputs: Batched outputs
            metadata: Metadata dict with batch_size and original_lengths

        Returns:
            List of individual output dicts
        """
        batch_size = metadata["batch_size"]
        original_lengths = metadata.get("original_lengths", {})
        result = [{} for _ in range(batch_size)]

        for key, value in outputs.items():
            if isinstance(value, (jax.Array, np.ndarray)) and value.shape[0] == batch_size:
                # Check if we need to trim to original lengths
                if key in original_lengths:
                    lengths = original_lengths[key]
                    for i in range(batch_size):
                        result[i][key] = np.asarray(value[i, :lengths[i]])
                else:
                    for i in range(batch_size):
                        result[i][key] = np.asarray(value[i])
            elif isinstance(value, dict):
                unbatched = self._unbatch_outputs(value, metadata)
                for i in range(batch_size):
                    result[i][key] = unbatched[i]
            elif isinstance(value, list) and len(value) == batch_size:
                for i in range(batch_size):
                    result[i][key] = value[i]
            else:
                for i in range(batch_size):
                    result[i][key] = value

        return result

    def infer_actions_batch(
        self,
        obs_list: list[dict],
        *,
        num_steps: int | None = None,
        noise: np.ndarray | None = None,
        allow_variable_length: bool = False,
    ) -> list[dict]:
        """Infer actions for a batch of observations.

        Args:
            obs_list: List of observations
            num_steps: Number of denoising steps
            noise: Optional noise for sampling
            allow_variable_length: If True, pad sequences to max length

        Returns:
            List of action dicts
        """
        if not obs_list:
            return []

        if len(obs_list) == 1:
            return [self.infer_actions(obs_list[0], num_steps=num_steps, noise=noise)]

        batch_size = len(obs_list)
        t0 = time.monotonic()

        inputs, metadata = self._prepare_batched_inputs(
            obs_list, allow_variable_length, use_numpy_stack=self._is_pytorch_model
        )
        if self._is_pytorch_model:
            inputs_for_model = self._to_torch_tree(inputs)
        else:
            inputs_for_model = inputs
        
        t1 = time.monotonic()
        
        sample_kwargs = dict(self._sample_kwargs)
        if num_steps is not None:
            sample_kwargs["num_steps"] = num_steps
        
        if noise is not None:
            if self._is_pytorch_model:
                noise = torch.as_tensor(noise, device=self._pytorch_device)
                if noise.ndim == 2:
                    noise = noise[None, ...].expand(batch_size, *noise.shape)
            else:
                noise = jnp.asarray(noise)
                if noise.ndim == 2:
                    noise = jnp.broadcast_to(noise[None, ...], (batch_size,) + noise.shape)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs_for_model)
        if self._is_pytorch_model:
            actions = self._sample_actions(self._pytorch_device, observation, **sample_kwargs)
            self._sync_torch()
        else:
            self._rng, sample_rng = jax.random.split(self._rng)
            actions = self._sample_actions(sample_rng, observation, **sample_kwargs)
            if hasattr(actions, "block_until_ready"):
                actions.block_until_ready()
        
        t2 = time.monotonic()
        
        outputs = {"state": inputs_for_model["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: _torch_to_numpy(x) if self._is_pytorch_model else (np.asarray(x) if hasattr(x, 'shape') else x), outputs)

        output_list = self._unbatch_outputs(outputs, metadata)
        
        for i in range(batch_size):
            output_list[i] = self._output_transform(output_list[i])
        
        t3 = time.monotonic()
        
        timing = {
            "pre_proc_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_proc_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
            "batch_size": batch_size,
            "per_sample_ms": (t3 - t0) * 1000 / batch_size,
        }
        
        for output in output_list:
            output["policy_timing"] = timing
        
        return output_list

    def infer_text_batch(
        self,
        obs_list: list[dict],
        *,
        max_decoding_steps: int = 25,
        temperature: float = 0.1,
        PALIGEMMA_EOS_TOKEN: int = -1,
        allow_variable_length: bool = False,
    ) -> list[dict]:
        """Infer text for a batch of observations.

        Args:
            obs_list: List of observation dictionaries.
            max_decoding_steps: Maximum decoding steps.
            temperature: Sampling temperature.
            PALIGEMMA_EOS_TOKEN: End of sequence token. Default -1 (no early stopping).
            allow_variable_length: If True, pad sequences to max length.

        Returns:
            List of dictionaries with tokens and text.
        """
        if not hasattr(self, "_sample_text"):
            raise NotImplementedError("Model does not have sample_text method")

        if not obs_list:
            return []

        if len(obs_list) == 1:
            return [self.infer_text(
                obs_list[0],
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
                PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
            )]

        batch_size = len(obs_list)
        t0 = time.monotonic()

        inputs, metadata = self._prepare_batched_inputs(
            obs_list, allow_variable_length, use_numpy_stack=self._is_pytorch_model
        )
        if self._is_pytorch_model:
            inputs_for_model = self._to_torch_tree(inputs)
        else:
            inputs_for_model = inputs

        observation = _model.Observation.from_dict(inputs_for_model)

        t1 = time.monotonic()

        if self._is_pytorch_model:
            sample_rng_or_device = self._pytorch_device
        else:
            self._rng, sample_rng_or_device = jax.random.split(self._rng)

        predicted_tokens, kv_cache, mask, ar_mask = self._sample_text(
            sample_rng_or_device,
            observation,
            max_decoding_steps=max_decoding_steps,
            PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
            temperature=temperature,
        )

        if self._is_pytorch_model:
            self._sync_torch()
        elif hasattr(predicted_tokens, "block_until_ready"):
            predicted_tokens.block_until_ready()

        t2 = time.monotonic()

        # Detokenize each sample
        tokenizer = self._tokenizer
        tokens_np = _torch_to_numpy(predicted_tokens) if self._is_pytorch_model else np.asarray(predicted_tokens)
        
        output_list = []
        for i in range(batch_size):
            tokens_i = tokens_np[i]
            text_i = tokenizer.detokenize(tokens_i.astype(np.int32))
            output_list.append({
                "tokens": tokens_i,
                "text": text_i,
            })

        t3 = time.monotonic()

        timing = {
            "pre_proc_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_proc_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
            "batch_size": batch_size,
            "per_sample_ms": (t3 - t0) * 1000 / batch_size,
        }

        shapes = {
            "observation": jax.tree.map(
                lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs
            ),
            "tokens": tuple(tokens_np.shape),
        }

        for output in output_list:
            output["policy_timing"] = timing
            output["policy_shapes"] = shapes

        return output_list

    def infer_text_actions_shared_kv_batch(
        self,
        obs_list: list[dict],
        *,
        num_steps: int = 10,
        max_decoding_steps: int = 20,
        noise: np.ndarray | None = None,
        PALIGEMMA_EOS_TOKEN: int = -1,
        temperature: float = 0.0,
        allow_variable_length: bool = False,
    ) -> list[dict]:
        """Infer text and actions with shared KV cache for a batch of observations.

        Args:
            obs_list: List of observation dictionaries.
            num_steps: Number of denoising steps.
            max_decoding_steps: Maximum text decoding steps.
            noise: Optional noise array.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            temperature: Sampling temperature.
            allow_variable_length: If True, pad sequences to max length.

        Returns:
            List of dictionaries with actions, tokens, text, and timings.
        """
        if not obs_list:
            return []

        if len(obs_list) == 1:
            return [self.infer_text_actions_shared_kv(
                obs_list[0],
                num_steps=num_steps,
                max_decoding_steps=max_decoding_steps,
                noise=noise,
                PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
                temperature=temperature,
            )]

        batch_size = len(obs_list)
        t0 = time.monotonic()

        inputs, metadata = self._prepare_batched_inputs(
            obs_list, allow_variable_length, use_numpy_stack=self._is_pytorch_model
        )

        self._rng, sample_rng = jax.random.split(self._rng)

        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = jnp.broadcast_to(noise[None, ...], (batch_size,) + noise.shape)

        observation = _model.Observation.from_dict(inputs)

        t1 = time.monotonic()

        actions, predicted_tokens = self._sample_text_actions_shared_kv(
            sample_rng,
            observation,
            num_steps=num_steps,
            max_decoding_steps=max_decoding_steps,
            noise=noise,
            PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
            temperature=temperature,
        )

        if hasattr(actions, "block_until_ready"):
            actions.block_until_ready()
        if hasattr(predicted_tokens, "block_until_ready"):
            predicted_tokens.block_until_ready()

        t2 = time.monotonic()

        actions_np = np.asarray(actions)
        tokens_np = np.asarray(predicted_tokens)
        state_np = np.asarray(inputs["state"])

        tokenizer = self._tokenizer

        output_list = []
        for i in range(batch_size):
            outputs_i = {
                "state": state_np[i],
                "actions": actions_np[i],
                "tokens": tokens_np[i],
            }

            # Apply output transform to action outputs
            action_outputs_i = {"state": outputs_i["state"], "actions": outputs_i["actions"]}
            action_outputs_i = self._output_transform(action_outputs_i)
            outputs_i.update(action_outputs_i)

            # Detokenize
            text_i = tokenizer.detokenize(outputs_i["tokens"].astype(np.int32))
            outputs_i["text"] = text_i

            output_list.append(outputs_i)

        t3 = time.monotonic()

        timing = {
            "pre_proc_ms": (t1 - t0) * 1000,
            "infer_ms": (t2 - t1) * 1000,
            "post_proc_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
            "batch_size": batch_size,
            "per_sample_ms": (t3 - t0) * 1000 / batch_size,
        }

        shapes = {
            "observation": jax.tree.map(
                lambda x: tuple(x.shape) if hasattr(x, "shape") else (), inputs
            ),
            "actions": tuple(actions_np.shape),
            "tokens": tuple(tokens_np.shape),
        }

        for output in output_list:
            output["policy_timing"] = timing
            output["policy_shapes"] = shapes

        return output_list

    def init_continuous_batching(self):
        """Initialize continuous batch manager.

        Returns:
            ContinuousBatchManager instance for tracking ongoing generations
        """
        from openpi.models.kv_cache_manager import ContinuousBatchManager
        return ContinuousBatchManager()

    def infer_text_continuous(
        self,
        obs: dict,
        cache_manager,
        request_id: Optional[str] = None,
        max_decoding_steps: int = 20,
        steps_per_frame: int = 5,
        temperature: float = 0.1,
        PALIGEMMA_EOS_TOKEN: int = -1,
    ) -> dict:
        """Generate text with continuous batching support.

        This method allows text generation to continue across multiple frames
        without restarting from scratch. It saves KV caches between calls.

        Args:
            obs: Observation dict
            cache_manager: ContinuousBatchManager instance
            request_id: Optional request ID for resuming generation
            max_decoding_steps: Max total decoding steps
            steps_per_frame: How many tokens to generate this frame
            temperature: Sampling temperature
            PALIGEMMA_EOS_TOKEN: EOS token ID

        Returns:
            Result dict with tokens, text, request_id, is_finished
        """
        if not hasattr(self, "_sample_text"):
            raise NotImplementedError("Model does not have sample_text method")

        # Check if resuming or starting new
        if request_id and request_id in cache_manager.active_caches:
            # Resume from cache - not fully implemented yet
            # This would require modifying sample_text to accept initial cache
            raise NotImplementedError(
                "Resuming from cache not yet implemented. "
                "Need to extend sample_text to accept initial KV cache."
            )
        else:
            # Start new generation
            obs_transformed = self._input_transform(obs)
            # Add batch dimension
            obs_transformed = jax.tree.map(_to_jax_batch, obs_transformed)

            # Prefill
            observation = _model.Observation.from_dict(obs_transformed)

            # For now, just do standard text generation
            # In future, would generate steps_per_frame tokens and save cache
            self._rng, sample_rng = jax.random.split(self._rng)

            # Generate all tokens for now (not incremental)
            predicted_tokens, kv_cache, mask, ar_mask = self._sample_text(
                sample_rng,
                observation,
                max_decoding_steps=max_decoding_steps,
                PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
                temperature=temperature,
            )

            # Convert to numpy
            tokens_np = np.asarray(predicted_tokens)

            # Detokenize
            text = self._tokenizer.detokenize(tokens_np.astype(np.int32))

            # Create request_id if needed
            if request_id is None:
                request_id = f"req_{cache_manager.next_request_id}"
                cache_manager.next_request_id += 1

            return {
                "tokens": tokens_np,
                "text": text,
                "request_id": request_id,
                "is_finished": True,  # For now, always finish in one frame
            }

    def infer_text_actions_continuous_batch(
        self,
        obs_list: list[dict],
        cache_manager,
        request_ids: Optional[list[str]] = None,
        steps_per_frame: int = 5,
        num_action_steps: int = 10,
        max_decoding_steps: int = 20,
        temperature: float = 0.1,
        PALIGEMMA_EOS_TOKEN: int = -1,
        noise: np.ndarray | None = None,
        generate_actions_for_resumed: bool = False,
    ) -> list[dict]:
        """Batch inference with continuous text generation.

        Designed for the realistic robotics scenario where each frame:
        - 1 new request arrives: needs prefill + actions + text start
        - N resumed requests continue: text generation only (actions already
          generated when the request first arrived)

        GPU efficiency:
        - New requests: ONE prefill call, reused for both text init AND action sampling
        - All requests (new + resumed): ONE batched generate_n_tokens call
        - No redundant prefills for resumed requests

        Args:
            obs_list: List of observations. For new requests, the observation is used
                for prefill and action generation. For resumed requests, the observation
                is only used if generate_actions_for_resumed=True.
            cache_manager: ContinuousBatchManager instance.
            request_ids: List of request IDs. None entries indicate new requests.
                If the whole list is None, all requests are new.
            steps_per_frame: Tokens to generate per frame for each request.
            num_action_steps: Number of action denoising steps (for new requests).
            max_decoding_steps: Max total text tokens per request (for new requests).
            temperature: Sampling temperature.
            PALIGEMMA_EOS_TOKEN: EOS token ID.
            noise: Optional noise for action sampling.
            generate_actions_for_resumed: If True, generate actions for resumed requests
                too (requires prefill for all observations). Default False.

        Returns:
            List of result dicts. Each dict contains:
                - actions: Action array (only for new requests, None for resumed unless
                  generate_actions_for_resumed=True)
                - tokens_this_frame: Tokens generated this frame
                - tokens_full: All tokens generated so far
                - text: Detokenized text so far
                - request_id: Request identifier
                - is_finished: Whether text generation is complete
        """
        if not obs_list:
            return []

        if self._is_pytorch_model:
            with torch.inference_mode():
                return self._infer_text_actions_continuous_batch_pytorch(
                    obs_list,
                    cache_manager,
                    request_ids=request_ids,
                    steps_per_frame=steps_per_frame,
                    num_action_steps=num_action_steps,
                    max_decoding_steps=max_decoding_steps,
                    temperature=temperature,
                    PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
                    noise=noise,
                    generate_actions_for_resumed=generate_actions_for_resumed,
                )

        batch_size = len(obs_list)
        t0 = time.monotonic()

        # Determine which requests are new vs resumed
        if request_ids is None:
            request_ids = [None] * batch_size

        new_indices = []
        resumed_indices = []
        resumed_states = []

        for i, rid in enumerate(request_ids):
            if rid is None or rid not in cache_manager.active_states:
                new_indices.append(i)
            else:
                resumed_indices.append(i)
                resumed_states.append(cache_manager.get_state(rid))

        t1 = time.monotonic()

        # Split RNG
        self._rng, rng_text, rng_action = jax.random.split(self._rng, 3)

        # === NEW REQUESTS: Prefill → text init + actions (shared KV) ===
        new_states = []
        new_actions = {}  # index -> actions array

        if new_indices:
            # Prepare inputs for new requests only
            new_obs_list = [obs_list[i] for i in new_indices]
            new_inputs, _ = self._prepare_batched_inputs(new_obs_list, allow_variable_length=False)
            new_observation = _model.Observation.from_dict(new_inputs)

            # ONE prefill for new requests (reused for text AND actions)
            prefill_result_new = self._prefill(
                new_observation, align_right=False, max_decoding_steps=max_decoding_steps,
            )

            # Text: init incremental state from prefill
            rng_new = jax.random.split(rng_text, len(new_indices))
            batched_new_state = self._init_incremental_state(prefill_result_new, rng_new[0])
            new_states = _split_incremental_state(batched_new_state, len(new_indices))

            # Actions: reuse same prefill result (shared KV cache)
            if noise is not None:
                noise_new = jnp.asarray(noise)
                if noise_new.ndim == 2:
                    noise_new = jnp.broadcast_to(
                        noise_new[None, ...], (len(new_indices),) + noise_new.shape
                    )
            else:
                noise_new = None

            actions_new = self._sample_actions_with_kv(
                rng_action, new_observation, prefill_result_new,
                num_steps=num_action_steps, noise=noise_new,
            )

            # Map actions back to original indices
            actions_new_np = np.asarray(actions_new)
            for j, idx in enumerate(new_indices):
                new_actions[idx] = actions_new_np[j]

        t_prefill = time.monotonic()

        # === RESUMED REQUESTS: Actions if requested ===
        resumed_actions = {}
        if resumed_indices and generate_actions_for_resumed:
            resumed_obs_list = [obs_list[i] for i in resumed_indices]
            resumed_inputs, _ = self._prepare_batched_inputs(resumed_obs_list, allow_variable_length=False)
            resumed_observation = _model.Observation.from_dict(resumed_inputs)

            prefill_result_resumed = self._prefill(
                resumed_observation, align_right=False, max_decoding_steps=0,
            )
            actions_resumed = self._sample_actions_with_kv(
                jax.random.fold_in(rng_action, 1), resumed_observation,
                prefill_result_resumed, num_steps=num_action_steps,
            )
            actions_resumed_np = np.asarray(actions_resumed)
            for j, idx in enumerate(resumed_indices):
                resumed_actions[idx] = actions_resumed_np[j]

        # === BATCHED TEXT GENERATION (all requests together) ===
        # Reorder states to match original indices
        all_states = [None] * batch_size
        for idx, state in zip(new_indices, new_states):
            all_states[idx] = state
        for idx, state in zip(resumed_indices, resumed_states):
            all_states[idx] = state

        # Stack ALL states for ONE batched generate_n_tokens call
        batched_state = _stack_incremental_states(all_states)

        tokens_generated, updated_batched_state, _ = self._generate_n_tokens(
            batched_state,
            tokens_to_generate=steps_per_frame,
            PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
            temperature=temperature,
        )

        # Block until all GPU work is done
        if hasattr(tokens_generated, "block_until_ready"):
            tokens_generated.block_until_ready()

        t2 = time.monotonic()

        # === PROCESS RESULTS ===
        updated_states = _split_incremental_state(updated_batched_state, batch_size)
        tokens_np = np.asarray(tokens_generated)

        tokenizer = self._tokenizer

        output_list = []
        for i in range(batch_size):
            tokens_frame_i = tokens_np[i]

            state_i = updated_states[i]
            current_step = int(state_i.current_step[0])
            full_tokens_i = np.asarray(state_i.output_tokens[0, :current_step])
            is_finished_i = bool(state_i.is_finished[0]) or current_step >= state_i.max_decoding_steps

            # Detokenize
            text_i = tokenizer.detokenize(full_tokens_i.astype(np.int32)) if len(full_tokens_i) > 0 else ""

            # Get actions (only for new requests by default)
            actions_i = new_actions.get(i, resumed_actions.get(i, None))

            outputs_i = {
                "tokens_this_frame": tokens_frame_i,
                "tokens_full": full_tokens_i,
                "text": text_i,
                "is_finished": is_finished_i,
            }

            # Only include actions and apply output transform when actions were generated
            if actions_i is not None:
                outputs_i["actions"] = actions_i
                action_outputs_i = {"state": np.asarray(new_inputs["state"][new_indices.index(i)]) if i in new_indices else None, "actions": actions_i}
                if action_outputs_i["state"] is not None:
                    action_outputs_i = self._output_transform(action_outputs_i)
                    outputs_i.update(action_outputs_i)
            else:
                outputs_i["actions"] = None

            # Assign request ID
            if request_ids[i] is None:
                request_ids[i] = f"req_{cache_manager.next_request_id}"
                cache_manager.next_request_id += 1

            outputs_i["request_id"] = request_ids[i]

            # Store state in cache manager for resumption
            if not is_finished_i:
                cache_manager.store_state(request_ids[i], updated_states[i])
            else:
                cache_manager.remove_state(request_ids[i])

            output_list.append(outputs_i)

        t3 = time.monotonic()

        timing = {
            "pre_proc_ms": (t1 - t0) * 1000,
            "prefill_actions_ms": (t_prefill - t1) * 1000,
            "text_gen_ms": (t2 - t_prefill) * 1000,
            "post_proc_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
            "batch_size": batch_size,
            "new_requests": len(new_indices),
            "resumed_requests": len(resumed_indices),
        }

        for output in output_list:
            output["policy_timing"] = timing

        return output_list

    def _infer_text_actions_continuous_batch_pytorch(
        self,
        obs_list: list[dict],
        cache_manager,
        request_ids: Optional[list[str]] = None,
        steps_per_frame: int = 5,
        num_action_steps: int = 10,
        max_decoding_steps: int = 20,
        temperature: float = 0.1,
        PALIGEMMA_EOS_TOKEN: int = -1,
        noise: np.ndarray | None = None,
        generate_actions_for_resumed: bool = False,
    ) -> list[dict]:
        """PyTorch backend for continuous batching.

        New requests are batched for prefill and action denoising. Text states
        use a fixed-size cache so all active requests can be advanced in one
        batched generation call, matching the JAX scheduling path.
        """
        batch_size = len(obs_list)
        t0 = time.monotonic()
        if request_ids is None:
            request_ids = [None] * batch_size

        new_indices: list[int] = []
        resumed_indices: list[int] = []
        resumed_states = {}
        for i, rid in enumerate(request_ids):
            if rid is None or rid not in cache_manager.active_states:
                new_indices.append(i)
            else:
                resumed_indices.append(i)
                resumed_states[i] = cache_manager.get_state(rid)

        t1 = time.monotonic()

        new_states = {}
        new_actions = {}
        new_state_inputs = {}
        if new_indices:
            new_obs_list = [obs_list[i] for i in new_indices]
            new_inputs, _ = self._prepare_batched_inputs(
                new_obs_list, allow_variable_length=False, use_numpy_stack=True
            )
            new_inputs_torch = self._to_torch_tree(new_inputs)
            new_observation = _model.Observation.from_dict(new_inputs_torch)

            prefill_result = self._prefill(new_observation, max_decoding_steps=max_decoding_steps)
            if noise is not None:
                noise_new = torch.as_tensor(noise, device=self._pytorch_device)
                if noise_new.ndim == 2:
                    noise_new = noise_new[None, ...].expand(len(new_indices), *noise_new.shape)
            else:
                noise_new = None
            actions_batched = self._sample_actions_with_kv(
                self._pytorch_device,
                new_observation,
                prefill_result,
                num_steps=num_action_steps,
                noise=noise_new,
            )

            batched_state = self._init_static_incremental_state(prefill_result)
            split_states = _split_pytorch_incremental_state(batched_state, len(new_indices))
            for j, idx in enumerate(new_indices):
                new_states[idx] = split_states[j]
                new_state_inputs[idx] = _torch_to_numpy(new_inputs_torch["state"][j])
                new_actions[idx] = _torch_to_numpy(actions_batched[j])

        t_prefill = time.monotonic()

        resumed_actions = {}
        if resumed_indices and generate_actions_for_resumed:
            resumed_obs_list = [obs_list[i] for i in resumed_indices]
            resumed_outputs = self.infer_actions_batch(resumed_obs_list, num_steps=num_action_steps)
            for j, idx in enumerate(resumed_indices):
                resumed_actions[idx] = resumed_outputs[j]["actions"]

        all_states = {}
        all_states.update(new_states)
        all_states.update(resumed_states)

        ordered_states = [all_states[i] for i in range(batch_size)]
        batched_text_state = _stack_pytorch_incremental_states(ordered_states)
        tokens_batched, updated_batched_state, _ = self._generate_n_tokens(
            batched_text_state,
            tokens_to_generate=steps_per_frame,
            PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
            temperature=temperature,
        )
        split_updated_states = _split_pytorch_incremental_state(updated_batched_state, batch_size)
        updated_states = {i: split_updated_states[i] for i in range(batch_size)}
        tokens_frame = {i: _torch_to_numpy(tokens_batched[i]) for i in range(batch_size)}

        self._sync_torch()
        t2 = time.monotonic()

        output_list = []
        for i in range(batch_size):
            state_i = updated_states[i]
            current_step = int(state_i.current_step[0].detach().cpu().item())
            full_tokens_i = _torch_to_numpy(state_i.output_tokens[0, :current_step])
            is_finished_i = bool(state_i.is_finished[0].detach().cpu().item()) or current_step >= state_i.max_decoding_steps
            text_i = self._tokenizer.detokenize(full_tokens_i.astype(np.int32)) if len(full_tokens_i) > 0 else ""

            actions_i = new_actions.get(i, resumed_actions.get(i, None))
            outputs_i = {
                "tokens_this_frame": tokens_frame[i],
                "tokens_full": full_tokens_i,
                "text": text_i,
                "is_finished": is_finished_i,
                "actions": actions_i,
            }

            if actions_i is not None and i in new_state_inputs:
                outputs_i.update(self._output_transform({"state": new_state_inputs[i], "actions": actions_i}))

            if request_ids[i] is None:
                request_ids[i] = f"req_{cache_manager.next_request_id}"
                cache_manager.next_request_id += 1
            outputs_i["request_id"] = request_ids[i]

            if not is_finished_i:
                cache_manager.store_state(request_ids[i], state_i)
            else:
                cache_manager.remove_state(request_ids[i])
            output_list.append(outputs_i)

        t3 = time.monotonic()
        timing = {
            "pre_proc_ms": (t1 - t0) * 1000,
            "prefill_actions_ms": (t_prefill - t1) * 1000,
            "text_gen_ms": (t2 - t_prefill) * 1000,
            "post_proc_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
            "batch_size": batch_size,
            "new_requests": len(new_indices),
            "resumed_requests": len(resumed_indices),
            "backend": "pytorch",
        }
        for output in output_list:
            output["policy_timing"] = timing

        return output_list


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
