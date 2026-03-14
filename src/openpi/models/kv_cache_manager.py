"""KV Cache Manager for Continuous Batching.

This module provides utilities for managing IncrementalTextState across multiple
inference frames, enabling continuous text generation without restarting
from scratch.
"""

from typing import Optional, Any
import jax.numpy as jnp
import numpy as np


class ContinuousBatchManager:
    """Manages incremental text generation states across frames for continuous batching.

    This class handles:
    - Tracking ongoing text generation requests
    - Storing IncrementalTextState for resumption
    - Lifecycle management (start, update, finish)
    """

    def __init__(self):
        # Store IncrementalTextState objects (from pi05.IncrementalTextState)
        self.active_states: dict[str, Any] = {}  # request_id -> IncrementalTextState
        self.next_request_id: int = 0

    def store_state(
        self,
        request_id: str,
        state: Any,  # IncrementalTextState
    ):
        """Store incremental text state for a request.

        Args:
            request_id: Request ID
            state: IncrementalTextState object
        """
        self.active_states[request_id] = state

    def get_state(self, request_id: str) -> Any:
        """Get incremental text state for a request.

        Args:
            request_id: Request ID

        Returns:
            IncrementalTextState if exists, None otherwise
        """
        return self.active_states.get(request_id)

    def remove_state(self, request_id: str):
        """Remove state for a finished request.

        Args:
            request_id: Request ID to remove
        """
        if request_id in self.active_states:
            del self.active_states[request_id]

    def get_status(self, request_id: str) -> dict:
        """Get generation status for a request.

        Args:
            request_id: Request ID to query

        Returns:
            Status dict with generation progress
        """
        if request_id not in self.active_states:
            return {"status": "finished_or_not_found"}

        state = self.active_states[request_id]
        return {
            "status": "active",
            "tokens_generated": state.current_step,
            "current_len": state.current_step,
            "max_len": state.max_decoding_steps,
            "is_finished": bool(jnp.all(state.is_finished)),
        }

    def get_all_active_requests(self) -> list[str]:
        """Get list of all active request IDs."""
        return list(self.active_states.keys())

    def cancel_request(self, request_id: str):
        """Cancel an active request and clean up its state."""
        self.remove_state(request_id)

    def clear_all(self):
        """Clear all active states."""
        self.active_states.clear()


# Legacy KVCacheState for backward compatibility with existing tests
import dataclasses


@dataclasses.dataclass
class KVCacheState:
    """State for ongoing text generation (DEPRECATED - use IncrementalTextState).

    This class is kept for backward compatibility with existing tests.
    """

    # KV cache tuple (idx, k_cache, v_cache)
    kv_cache: tuple

    # Mask state
    prefix_mask: jnp.ndarray          # [L_current]
    prefix_attn_mask: jnp.ndarray     # [L_current, L_total]
    ar_mask: jnp.ndarray              # [L_current]

    # Generation metadata
    prefill_len: int                   # Length after prefill
    current_len: int                   # Current decoded length
    max_decoding_steps: int            # Total allowed steps

    # Tracking
    tokens_generated: list[int]        # Generated token IDs
    is_finished: bool                  # EOS encountered
    frame_id: int                      # Which frame this cache belongs to

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "kv_cache": self.kv_cache,
            "prefix_mask": np.array(self.prefix_mask),
            "prefix_attn_mask": np.array(self.prefix_attn_mask),
            "ar_mask": np.array(self.ar_mask),
            "prefill_len": self.prefill_len,
            "current_len": self.current_len,
            "max_decoding_steps": self.max_decoding_steps,
            "tokens_generated": self.tokens_generated,
            "is_finished": self.is_finished,
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KVCacheState":
        """Deserialize from storage."""
        return cls(
            kv_cache=data["kv_cache"],
            prefix_mask=jnp.array(data["prefix_mask"]),
            prefix_attn_mask=jnp.array(data["prefix_attn_mask"]),
            ar_mask=jnp.array(data["ar_mask"]),
            prefill_len=data["prefill_len"],
            current_len=data["current_len"],
            max_decoding_steps=data["max_decoding_steps"],
            tokens_generated=data["tokens_generated"],
            is_finished=data["is_finished"],
            frame_id=data["frame_id"],
        )


# Legacy methods for backward compatibility
class _LegacyContinuousBatchManager(ContinuousBatchManager):
    """Extended manager with legacy methods for backward compatibility."""

    def __init__(self):
        super().__init__()
        self.active_caches = {}  # Alias for tests

    def start_generation(
        self,
        prefill_result: tuple,
        request_id: Optional[str] = None
    ) -> str:
        """Legacy method - creates KVCacheState from prefill result."""
        if request_id is None:
            request_id = f"req_{self.next_request_id}"
            self.next_request_id += 1

        prefix_out, kv_cache, prefix_mask, prefix_attn_mask, ar_mask, _ = prefill_result

        cache_state = KVCacheState(
            kv_cache=kv_cache,
            prefix_mask=prefix_mask,
            prefix_attn_mask=prefix_attn_mask,
            ar_mask=ar_mask,
            prefill_len=int(jnp.sum(prefix_mask)),
            current_len=int(jnp.sum(prefix_mask)),
            max_decoding_steps=prefix_attn_mask.shape[-1] - int(jnp.sum(prefix_mask)),
            tokens_generated=[],
            is_finished=False,
            frame_id=0,
        )

        self.active_caches[request_id] = cache_state
        return request_id

    def update_cache(
        self,
        request_id: str,
        new_tokens: list[int],
        new_kv_cache: tuple,
        is_finished: bool = False
    ):
        """Legacy method - updates KVCacheState."""
        if request_id not in self.active_caches:
            raise KeyError(f"Request {request_id} not found")

        state = self.active_caches[request_id]
        state.kv_cache = new_kv_cache
        state.tokens_generated.extend(new_tokens)
        state.current_len += len(new_tokens)
        state.is_finished = is_finished
        state.frame_id += 1

        if is_finished:
            del self.active_caches[request_id]

    def prepare_batch(
        self,
        request_ids: list[str]
    ) -> tuple[dict, dict]:
        """Legacy method - batches KVCacheState objects."""
        if not request_ids:
            return {}, {}

        states = [self.active_caches[rid] for rid in request_ids]

        max_cache_size = max(
            s.kv_cache[1].shape[1]
            for s in states
        )

        padded_idx = []
        padded_k = []
        padded_v = []
        original_sizes = []

        for state in states:
            idx, k_cache, v_cache = state.kv_cache
            cache_size = k_cache.shape[1]

            if idx.ndim > 0 and idx.shape[0] == 1:
                idx = idx[0]
            if k_cache.ndim > 3 and k_cache.shape[0] == 1:
                k_cache = k_cache[0]
            if v_cache.ndim > 3 and v_cache.shape[0] == 1:
                v_cache = v_cache[0]

            if cache_size < max_cache_size:
                pad_width = ((0, max_cache_size - cache_size), (0, 0), (0, 0))
                k_cache = jnp.pad(k_cache, pad_width)
                v_cache = jnp.pad(v_cache, pad_width)

            padded_idx.append(idx)
            padded_k.append(k_cache)
            padded_v.append(v_cache)
            original_sizes.append(cache_size)

        batched_kv_cache = (
            jnp.stack(padded_idx, axis=0),
            jnp.stack(padded_k, axis=0),
            jnp.stack(padded_v, axis=0),
        )

        max_prefix_len = max(s.prefix_mask.shape[0] for s in states)
        batched_prefix_masks = []

        for state in states:
            mask = state.prefix_mask
            if mask.shape[0] < max_prefix_len:
                mask = jnp.pad(mask, (0, max_prefix_len - mask.shape[0]))
            batched_prefix_masks.append(mask)

        batched_caches = {
            "kv_cache": batched_kv_cache,
            "prefix_masks": jnp.stack(batched_prefix_masks, axis=0),
            "current_lens": jnp.array([s.current_len for s in states]),
        }

        metadata = {
            "request_ids": request_ids,
            "original_cache_sizes": original_sizes,
            "frame_ids": [s.frame_id for s in states],
        }

        return batched_caches, metadata

        if request_id is None:
            request_id = f"req_{self.next_request_id}"
            self.next_request_id += 1

        prefix_out, kv_cache, prefix_mask, prefix_attn_mask, ar_mask, _ = prefill_result

        cache_state = KVCacheState(
            kv_cache=kv_cache,
            prefix_mask=prefix_mask,
            prefix_attn_mask=prefix_attn_mask,
            ar_mask=ar_mask,
            prefill_len=int(jnp.sum(prefix_mask)),
            current_len=int(jnp.sum(prefix_mask)),
            max_decoding_steps=prefix_attn_mask.shape[-1] - int(jnp.sum(prefix_mask)),
            tokens_generated=[],
            is_finished=False,
            frame_id=0,
        )

        self.active_caches[request_id] = cache_state
        return request_id

    def update_cache(
        self,
        request_id: str,
        new_tokens: list[int],
        new_kv_cache: tuple,
        is_finished: bool = False
    ):
        """Update cache after generating tokens.

        Args:
            request_id: Request ID to update
            new_tokens: Newly generated token IDs
            new_kv_cache: Updated KV cache tuple
            is_finished: Whether generation is complete
        """
        if request_id not in self.active_caches:
            raise KeyError(f"Request {request_id} not found")

        state = self.active_caches[request_id]
        state.kv_cache = new_kv_cache
        state.tokens_generated.extend(new_tokens)
        state.current_len += len(new_tokens)
        state.is_finished = is_finished
        state.frame_id += 1

        # Cleanup if finished
        if is_finished:
            del self.active_caches[request_id]

    def prepare_batch(
        self,
        request_ids: list[str]
    ) -> tuple[dict, dict]:
        """Prepare batched inputs from multiple ongoing requests.

        This method takes KV caches from different requests (potentially
        at different stages of generation) and pads them to a common size
        for batched processing.

        Args:
            request_ids: List of request IDs to batch together

        Returns:
            batched_caches: Dict with batched KV caches and masks
            metadata: Batch metadata including original cache sizes
        """
        if not request_ids:
            return {}, {}

        states = [self.active_caches[rid] for rid in request_ids]

        # Find max cache size across all requests
        max_cache_size = max(
            s.kv_cache[1].shape[1]  # k_cache.shape[1]
            for s in states
        )

        # Pad all caches to max size
        padded_idx = []
        padded_k = []
        padded_v = []
        original_sizes = []

        for state in states:
            idx, k_cache, v_cache = state.kv_cache
            cache_size = k_cache.shape[1]

            # Squeeze batch dimension (assumes each cache is [1, ...])
            if idx.ndim > 0 and idx.shape[0] == 1:
                idx = idx[0]
            if k_cache.ndim > 3 and k_cache.shape[0] == 1:
                k_cache = k_cache[0]
            if v_cache.ndim > 3 and v_cache.shape[0] == 1:
                v_cache = v_cache[0]

            # Pad cache if needed
            if cache_size < max_cache_size:
                pad_width = ((0, max_cache_size - cache_size), (0, 0), (0, 0))
                k_cache = jnp.pad(k_cache, pad_width)
                v_cache = jnp.pad(v_cache, pad_width)

            padded_idx.append(idx)
            padded_k.append(k_cache)
            padded_v.append(v_cache)
            original_sizes.append(cache_size)

        # Stack along batch dimension
        batched_kv_cache = (
            jnp.stack(padded_idx, axis=0),      # [B]
            jnp.stack(padded_k, axis=0),        # [B, max_cache_size, ...]
            jnp.stack(padded_v, axis=0),        # [B, max_cache_size, ...]
        )

        # Batch masks
        max_prefix_len = max(s.prefix_mask.shape[0] for s in states)
        batched_prefix_masks = []

        for state in states:
            mask = state.prefix_mask
            if mask.shape[0] < max_prefix_len:
                mask = jnp.pad(mask, (0, max_prefix_len - mask.shape[0]))
            batched_prefix_masks.append(mask)

        batched_caches = {
            "kv_cache": batched_kv_cache,
            "prefix_masks": jnp.stack(batched_prefix_masks, axis=0),
            "current_lens": jnp.array([s.current_len for s in states]),
        }

        metadata = {
            "request_ids": request_ids,
            "original_cache_sizes": original_sizes,
            "frame_ids": [s.frame_id for s in states],
        }

        return batched_caches, metadata

    def get_status(self, request_id: str) -> dict:
        """Get generation status for a request.

        Args:
            request_id: Request ID to query

        Returns:
            Status dict with generation progress
        """
        if request_id not in self.active_caches:
            return {"status": "finished_or_not_found"}

        state = self.active_caches[request_id]
        return {
            "status": "active",
            "tokens_generated": len(state.tokens_generated),
            "current_len": state.current_len,
            "max_len": state.prefill_len + state.max_decoding_steps,
            "frame_id": state.frame_id,
            "tokens": state.tokens_generated,
        }

    def get_all_active_requests(self) -> list[str]:
        """Get list of all active request IDs."""
        return list(self.active_caches.keys())

    def cancel_request(self, request_id: str):
        """Cancel an active request and clean up its cache."""
        if request_id in self.active_caches:
            del self.active_caches[request_id]

    def clear_all(self):
        """Clear all active caches."""
        self.active_caches.clear()
