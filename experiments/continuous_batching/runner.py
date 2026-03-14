"""Continuous batching runner: multi-frame stateful text generation."""

import logging
import time

import experiments.common.workload as workload_module
from experiments.common.workload import create_synthetic_observation

logger = logging.getLogger(__name__)

PALIGEMMA_EOS_TOKEN = -1  # No early stop for latency profiling


def _parse_arrival_pattern(pattern_str: str):
    """Parse an arrival pattern string into a callable.

    Uses eval() with the workload module's functions as namespace.

    Args:
        pattern_str: e.g. "uniform_arrivals(rate=1)"

    Returns:
        Callable(frame_idx) -> list[int]
    """
    namespace = {
        name: getattr(workload_module, name)
        for name in dir(workload_module)
        if callable(getattr(workload_module, name)) and not name.startswith("_")
    }
    return eval(pattern_str, {"__builtins__": {}}, namespace)


def run_continuous_batching(
    policy,
    *,
    policy_config: str,
    prompt: str,
    num_denoise_steps: int,
    max_decoding_steps: int,
    steps_per_frame: int,
    total_frames: int,
    warmup_frames: int,
    arrival_pattern: str,
) -> dict:
    """Run a multi-frame continuous batching simulation.

    Args:
        policy: Initialized Policy object.
        policy_config: Policy config name for observation creation.
        prompt: Text prompt for synthetic observations.
        num_denoise_steps: Denoising steps for action inference.
        max_decoding_steps: Max total text tokens per request.
        steps_per_frame: Tokens generated per frame per request.
        total_frames: Total number of frames to simulate.
        warmup_frames: Number of initial frames treated as warmup.
        arrival_pattern: Pattern string, e.g. "uniform_arrivals(rate=1)".

    Returns:
        Dict with 'frames' and 'completed_requests'.
    """
    cache_manager = policy.init_continuous_batching()
    arrival_fn = _parse_arrival_pattern(arrival_pattern)

    frames = []
    completed_requests = []
    # Track arrival frame per request_id
    arrival_frame_map: dict[str, int] = {}
    # Track per-request t_max for early eviction
    request_t_max: dict[str, int] = {}
    # Reusable dummy obs for resumed requests
    dummy_obs = create_synthetic_observation(
        prompt, seed=0, policy_config=policy_config,
    )

    for frame_idx in range(total_frames):
        # Determine new arrivals this frame
        new_t_max_list = arrival_fn(frame_idx)
        active_request_ids = cache_manager.get_all_active_requests()

        n_new_arrivals = len(new_t_max_list)
        n_resumed = len(active_request_ids)

        # Build obs_list and request_ids for the batch call
        obs_list = []
        request_ids = []
        pending_t_max: list[int] = []  # t_max for each new request

        # New requests
        for j in range(n_new_arrivals):
            obs = create_synthetic_observation(
                prompt,
                seed=frame_idx * 100 + j,
                policy_config=policy_config,
            )
            obs_list.append(obs)
            request_ids.append(None)  # None = new request
            pending_t_max.append(new_t_max_list[j])

        # Resumed requests
        for rid in active_request_ids:
            obs_list.append(dummy_obs)
            request_ids.append(rid)

        if not obs_list:
            # No requests at all this frame
            frames.append({
                "frame_idx": frame_idx,
                "frame_ms": 0.0,
                "total_tokens_this_frame": 0,
                "n_new": 0,
                "n_resumed": 0,
                "n_total": 0,
                "policy_timing": {},
                "is_warmup": frame_idx < warmup_frames,
            })
            continue

        logger.info(
            "Frame %d/%d: batch=%d (new=%d, resumed=%d) — calling policy...",
            frame_idx, total_frames, len(obs_list), n_new_arrivals, n_resumed,
        )
        t0 = time.monotonic()
        results = policy.infer_text_actions_continuous_batch(
            obs_list,
            cache_manager,
            request_ids=request_ids,
            steps_per_frame=steps_per_frame,
            num_action_steps=num_denoise_steps,
            max_decoding_steps=max_decoding_steps,
            PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
            generate_actions_for_resumed=False,
        )
        t1 = time.monotonic()
        frame_ms = (t1 - t0) * 1000.0

        # Track arrival frames for newly assigned request_ids
        # and register per-request t_max for new arrivals
        for idx, r in enumerate(results):
            rid = r["request_id"]
            if rid not in arrival_frame_map:
                arrival_frame_map[rid] = frame_idx
            if idx < n_new_arrivals:
                request_t_max[rid] = pending_t_max[idx]

        # Track completed requests (policy-finished or per-request t_max reached)
        for r in results:
            rid = r["request_id"]
            tokens_so_far = len(r.get("tokens_full", []))
            per_req_max = request_t_max.get(rid, max_decoding_steps)
            is_done = r["is_finished"] or tokens_so_far >= per_req_max

            if is_done:
                # Force eviction if the policy didn't already remove it
                if not r["is_finished"]:
                    cache_manager.remove_state(rid)
                completed_requests.append({
                    "request_id": rid,
                    "arrival_frame": arrival_frame_map.get(rid, frame_idx),
                    "finish_frame": frame_idx,
                    "total_wall_ms": frame_ms,
                })
                request_t_max.pop(rid, None)

        policy_timing = results[0]["policy_timing"]
        total_tokens = sum(len(r["tokens_this_frame"]) for r in results)

        frames.append({
            "frame_idx": frame_idx,
            "frame_ms": frame_ms,
            "total_tokens_this_frame": total_tokens,
            "n_new": policy_timing["new_requests"],
            "n_resumed": policy_timing["resumed_requests"],
            "n_total": policy_timing["batch_size"],
            "policy_timing": policy_timing,
            "is_warmup": frame_idx < warmup_frames,
        })

    return {
        "frames": frames,
        "completed_requests": completed_requests,
    }
