"""Grid search for continuous batching setting."""

import itertools
import json
import logging
import re
from pathlib import Path

import jax
import torch

from experiments.continuous_batching.runner import run_continuous_batching
from experiments.common.setup import collect_metadata
from experiments.common.workload import create_synthetic_observation

logger = logging.getLogger(__name__)


def warmup_batch_sizes(
    policy,
    *,
    policy_config: str,
    prompt: str,
    max_batch: int,
    num_denoise_steps: int,
    max_decoding_steps: int,
    steps_per_frame: int,
) -> None:
    """Warm up continuous batching for batch sizes 1..max_batch."""
    logger.info("Warmup: continuous batching batch sizes 1..%d", max_batch)
    for bs in range(1, max_batch + 1):
        logger.info("  Warmup bs=%d/%d — compiling...", bs, max_batch)
        cm = policy.init_continuous_batching()
        obs = [
            create_synthetic_observation(prompt, seed=i, policy_config=policy_config)
            for i in range(bs)
        ]
        res = policy.infer_text_actions_continuous_batch(
            obs,
            cm,
            steps_per_frame=steps_per_frame,
            max_decoding_steps=max_decoding_steps,
            num_action_steps=num_denoise_steps,
        )
        val = res[0]["actions"] if res[0].get("actions") is not None else res[0]["tokens_this_frame"]
        if isinstance(val, torch.Tensor):
            if val.is_cuda:
                torch.cuda.synchronize(val.device)
        else:
            jax.block_until_ready(val)
        logger.info("  bs=%d done", bs)
    logger.info("Warmup complete.")


def resolve_search_space(search_space: dict) -> list[dict]:
    """Expand a search space dict into a list of parameter combinations (cartesian product).

    Args:
        search_space: Mapping of parameter name -> list of values.

    Returns:
        List of dicts, each a single combination.
    """
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _is_valid_combo(params: dict) -> bool:
    """Check if max_decoding_steps is divisible by steps_per_frame."""
    if params["max_decoding_steps"] % params["steps_per_frame"] != 0:
        logger.warning(
            "Skipping invalid combo: max_decoding_steps=%d not divisible by steps_per_frame=%d",
            params["max_decoding_steps"],
            params["steps_per_frame"],
        )
        return False
    return True


def _arrival_slug(pattern_str: str) -> str:
    """Convert an arrival pattern string to a filename-safe slug.

    Examples:
        "uniform_arrivals(rate=1)" -> "uniform1"
        "poisson_arrivals(lam=2.0)" -> "poisson2.0"
        "bursty_arrivals(burst_size=4,burst_every=8)" -> "bursty4_8"
        "random_length_arrivals(t_max_values=[5,20], weights=[0.7,0.3], ...)"
            -> "random_length_v5_20_w0.7_0.3_1_42"
    """
    s = pattern_str.replace(" ", "")
    m = re.match(r"(\w+?)_arrivals\((.+)\)", s)
    if not m:
        # Fallback: sanitize the whole string
        return re.sub(r"[^a-zA-Z0-9]", "_", s).strip("_")
    prefix = m.group(1)
    args_str = m.group(2)

    parts = [prefix]

    # Extract list values: t_max_values=[5,20] -> "v5_20"
    list_match = re.search(r"t_max_values=\[([0-9,]+)\]", args_str)
    if list_match:
        vals = list_match.group(1).replace(",", "_")
        parts.append(f"v{vals}")

    # Extract weights: weights=[0.7,0.3] -> "w0.7_0.3"
    weights_match = re.search(r"weights=\[([0-9.,]+)\]", args_str)
    if weights_match:
        wvals = weights_match.group(1).replace(",", "_")
        parts.append(f"w{wvals}")

    # Extract scalar keyword args (rate=1, seed=42, lam=2.0, etc.)
    scalar_nums = re.findall(r"(?<!=\[)(?:^|,)(\w+)=([0-9.]+)", args_str)
    for _name, val in scalar_nums:
        parts.append(val)

    return "_".join(parts)


def _run_single_grid_point(
    policy,
    params: dict,
    fixed_params: dict,
) -> dict:
    """Run continuous batching for a single grid point.

    Runs the simulation twice: the first run warms up all dispatch paths
    (including resumed-request batching), the second run is the actual
    measurement. warmup_frames is auto-computed as the ramp-up period
    (max_decoding_steps // steps_per_frame).

    Returns:
        Result dict matching the JSON format in experiments/README.md.
    """
    policy_config = params["policy_config"]
    num_denoise_steps = params["num_denoise_steps"]
    max_decoding_steps = params["max_decoding_steps"]
    steps_per_frame = params["steps_per_frame"]

    prompt = fixed_params.get("prompt", "pick the red cup")
    total_frames = fixed_params["total_frames"]
    arrival_pattern = fixed_params["arrival_pattern"]

    # Auto-compute warmup_frames: ramp-up period before steady state
    warmup_frames = max_decoding_steps // steps_per_frame

    run_kwargs = dict(
        policy_config=policy_config,
        prompt=prompt,
        num_denoise_steps=num_denoise_steps,
        max_decoding_steps=max_decoding_steps,
        steps_per_frame=steps_per_frame,
        total_frames=total_frames,
        warmup_frames=warmup_frames,
        arrival_pattern=arrival_pattern,
    )

    # First run: warmup all dispatch paths (resumed-request batching, etc.)
    logger.info("Warmup run (full simulation)...")
    run_continuous_batching(policy, **run_kwargs)

    # Second run: actual measurement
    logger.info("Measurement run...")
    run_result = run_continuous_batching(policy, **run_kwargs)

    metadata = collect_metadata()
    metadata["total_frames"] = total_frames
    metadata["warmup_frames"] = warmup_frames

    return {
        "setting": "continuous_batching",
        "params": {
            "policy_config": policy_config,
            "num_denoise_steps": num_denoise_steps,
            "max_decoding_steps": max_decoding_steps,
            "steps_per_frame": steps_per_frame,
            "arrival_pattern": arrival_pattern,
        },
        "frames": run_result["frames"],
        "completed_requests": run_result["completed_requests"],
        "gpu_monitor": [],
        "metadata": metadata,
    }


def _save_result(result: dict, results_dir: Path) -> Path:
    """Save a result dict to a JSON file.

    File path: {results_dir}/continuous_batching/{policy_config}/denoise{N}_decode{M}_step{K}_{arrival}.json

    Returns:
        Path to the saved file.
    """
    params = result["params"]
    out_dir = (
        results_dir
        / "continuous_batching"
        / params["policy_config"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _arrival_slug(params["arrival_pattern"])
    filename = (
        f"denoise{params['num_denoise_steps']}"
        f"_decode{params['max_decoding_steps']}"
        f"_step{params['steps_per_frame']}"
        f"_{slug}.json"
    )
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", path)
    return path


def run_grid_search(
    policy,
    search_space: dict,
    fixed_params: dict,
    results_dir: Path,
) -> list[dict]:
    """Run cartesian product of search_space for the continuous batching setting.

    Args:
        policy: Already-initialized Policy object.
        search_space: Parameter name -> list of values.
            Expected keys: policy_config, num_denoise_steps,
            max_decoding_steps, steps_per_frame.
        fixed_params: Parameters held constant. Expected keys:
            prompt, total_frames, warmup_frames, arrival_pattern.
        results_dir: Where to save raw JSON results.

    Returns:
        List of result dicts (one per valid grid point).
    """
    combos = resolve_search_space(search_space)
    results = []

    max_decode = max(search_space.get("max_decoding_steps", [20]))
    prompt = fixed_params.get("prompt", "pick the red cup")
    first_policy = search_space["policy_config"][0]
    first_denoise = search_space["num_denoise_steps"][0]
    total_frames = fixed_params["total_frames"]
    arrival_pattern = fixed_params["arrival_pattern"]

    # Simulate the arrival pattern to find peak batch size per steps_per_frame.
    # This accounts for bursts/poisson that exceed the steady-state estimate.
    from experiments.continuous_batching.runner import _parse_arrival_pattern
    arrival_fn = _parse_arrival_pattern(arrival_pattern)

    for spf in search_space.get("steps_per_frame", [5]):
        frames_to_finish = max_decode // spf
        # Run multiple simulations to account for stochastic patterns (poisson)
        peak_bs = 0
        for _ in range(5):
            active_remaining: list[int] = []
            for frame_idx in range(total_frames):
                new_arrivals = arrival_fn(frame_idx)
                for _ in new_arrivals:
                    active_remaining.append(frames_to_finish)
                peak_bs = max(peak_bs, len(active_remaining))
                active_remaining = [r - 1 for r in active_remaining if r - 1 > 0]
        peak_bs = max(peak_bs + 1, 1)  # +1 safety margin
        logger.info(
            "Estimated peak batch size for spf=%d: %d (from %d-frame simulation)",
            spf, peak_bs, total_frames,
        )
        warmup_batch_sizes(
            policy,
            policy_config=first_policy,
            prompt=prompt,
            max_batch=peak_bs,
            num_denoise_steps=first_denoise,
            max_decoding_steps=max_decode,
            steps_per_frame=spf,
        )

    for i, params in enumerate(combos):
        if not _is_valid_combo(params):
            logger.info("Skipping invalid combo: %s", params)
            continue

        logger.info(
            "Grid point %d/%d: %s", i + 1, len(combos), params,
        )
        result = _run_single_grid_point(policy, params, fixed_params)
        _save_result(result, results_dir)
        results.append(result)

    return results
