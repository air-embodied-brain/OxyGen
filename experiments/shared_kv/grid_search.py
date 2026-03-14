"""Grid search for shared_kv setting."""

import itertools
import json
import logging
from pathlib import Path

from experiments.shared_kv.runner import run_shared_kv
from experiments.common.setup import collect_metadata
from experiments.common.workload import create_synthetic_observation

logger = logging.getLogger(__name__)


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
    """Check if a parameter combination is valid. Always True for shared_kv."""
    return True


def _run_single_grid_point(
    policy,
    params: dict,
    fixed_params: dict,
    num_measured_runs: int,
    warmup_runs: int,
) -> dict:
    """Run warmup + measured iterations for a single grid point.

    Returns:
        Result dict matching the JSON format in experiments/README.md.
    """
    num_denoise_steps = params["num_denoise_steps"]
    max_decoding_steps = params["max_decoding_steps"]
    policy_config = params["policy_config"]
    prompt = fixed_params.get("prompt", "pick the red cup")

    obs = create_synthetic_observation(
        prompt, seed=42, policy_config=policy_config,
    )

    frames = []
    total_runs = warmup_runs + num_measured_runs

    for i in range(total_runs):
        is_warmup = i < warmup_runs
        result = run_shared_kv(
            policy, obs,
            num_denoise_steps=num_denoise_steps,
            max_decoding_steps=max_decoding_steps,
        )
        frames.append({
            "frame_idx": i,
            "frame_ms": result["frame_ms"],
            "total_tokens_this_frame": result["total_tokens_this_frame"],
            "n_new": result["n_new"],
            "n_resumed": result["n_resumed"],
            "n_total": result["n_total"],
            "policy_timing": result["policy_timing"],
            "is_warmup": is_warmup,
        })

    metadata = collect_metadata()
    metadata["num_measured_runs"] = num_measured_runs
    metadata["warmup_runs"] = warmup_runs

    return {
        "setting": "shared_kv",
        "params": {
            "policy_config": policy_config,
            "num_denoise_steps": num_denoise_steps,
            "max_decoding_steps": max_decoding_steps,
        },
        "frames": frames,
        "gpu_monitor": [],
        "metadata": metadata,
    }


def _save_result(result: dict, results_dir: Path) -> Path:
    """Save a result dict to a JSON file.

    File path: {results_dir}/shared_kv/{policy_config}/denoise{N}_decode{M}.json

    Returns:
        Path to the saved file.
    """
    params = result["params"]
    out_dir = (
        results_dir
        / "shared_kv"
        / params["policy_config"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"denoise{params['num_denoise_steps']}_decode{params['max_decoding_steps']}.json"
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
    num_measured_runs: int = 3,
    warmup_runs: int = 1,
) -> list[dict]:
    """Run cartesian product of search_space for the shared_kv setting.

    Args:
        policy: Already-initialized Policy object.
        search_space: Parameter name -> list of values.
        fixed_params: Parameters held constant (e.g. prompt).
        results_dir: Where to save raw JSON results.
        num_measured_runs: Repeated measurements per grid point.
        warmup_runs: Discarded warmup runs per grid point.

    Returns:
        List of result dicts (one per grid point).
    """
    combos = resolve_search_space(search_space)
    results = []

    for i, params in enumerate(combos):
        if not _is_valid_combo(params):
            logger.info("Skipping invalid combo: %s", params)
            continue

        logger.info(
            "Grid point %d/%d: %s", i + 1, len(combos), params,
        )
        result = _run_single_grid_point(
            policy, params, fixed_params,
            num_measured_runs=num_measured_runs,
            warmup_runs=warmup_runs,
        )
        _save_result(result, results_dir)
        results.append(result)

    return results
