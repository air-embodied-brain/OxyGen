"""Baseline runner: sequential isolated execution (infer_actions + infer_text)."""

import time

import jax

PALIGEMMA_EOS_TOKEN = -1  # No early stop for latency profiling


def run_baseline(
    policy,
    obs: dict,
    *,
    num_denoise_steps: int,
    max_decoding_steps: int,
) -> dict:
    """Run a single baseline frame: infer_actions then infer_text sequentially.

    Args:
        policy: Initialized Policy object.
        obs: Observation dict (from create_synthetic_observation).
        num_denoise_steps: Denoising iterations for action inference.
        max_decoding_steps: Max tokens for text decoding.

    Returns:
        Dict with frame_ms, total_tokens_this_frame, request counts, and policy_timing.
    """
    t0 = time.monotonic()

    # Actions
    actions_out = policy.infer_actions(obs, num_steps=num_denoise_steps)
    # Block until JAX computation completes
    jax.block_until_ready(actions_out["actions"])

    # Text
    text_out = policy.infer_text(
        obs,
        max_decoding_steps=max_decoding_steps,
        PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
    )
    jax.block_until_ready(text_out["tokens"])

    t1 = time.monotonic()
    frame_ms = (t1 - t0) * 1000.0

    # Merge policy_timing with prefixed keys
    policy_timing = {}
    for k, v in actions_out["policy_timing"].items():
        policy_timing[f"actions_{k}"] = v
    for k, v in text_out["policy_timing"].items():
        policy_timing[f"text_{k}"] = v

    return {
        "frame_ms": frame_ms,
        "total_tokens_this_frame": max_decoding_steps,
        "n_new": 1,
        "n_resumed": 0,
        "n_total": 1,
        "policy_timing": policy_timing,
    }
