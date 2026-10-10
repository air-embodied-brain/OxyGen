"""Shared KV runner: single inference call sharing KV cache between text and action generation."""

import time

PALIGEMMA_EOS_TOKEN = -1  # No early stop for latency profiling


def run_shared_kv(
    policy,
    obs: dict,
    *,
    num_denoise_steps: int,
    max_decoding_steps: int,
) -> dict:
    """Run a single shared_kv frame: one infer_text_actions_shared_kv call.

    Args:
        policy: Initialized Policy object.
        obs: Observation dict (from create_synthetic_observation).
        num_denoise_steps: Denoising iterations for action inference.
        max_decoding_steps: Max tokens for text decoding.

    Returns:
        Dict with frame_ms, total_tokens_this_frame, request counts, and policy_timing.
    """
    t0 = time.monotonic()

    out = policy.infer_text_actions_shared_kv(
        obs,
        num_steps=num_denoise_steps,
        max_decoding_steps=max_decoding_steps,
        PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
    )

    t1 = time.monotonic()
    frame_ms = (t1 - t0) * 1000.0

    return {
        "frame_ms": frame_ms,
        "total_tokens_this_frame": max_decoding_steps,
        "n_new": 1,
        "n_resumed": 0,
        "n_total": 1,
        "policy_timing": out["policy_timing"],
    }
