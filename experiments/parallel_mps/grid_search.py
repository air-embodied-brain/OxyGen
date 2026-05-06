"""Grid search for parallel_mps setting.

Manages NVIDIA MPS daemon lifecycle and spawns two worker processes per
policy config to run action and text inference in parallel on the same GPU.
"""

import itertools
import json
import logging
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

from experiments.common.setup import collect_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MPS daemon management
# ---------------------------------------------------------------------------

def start_mps(gpu_id: int | None = None) -> None:
    """Start the NVIDIA MPS daemon and server.

    Sets CUDA_MPS_PIPE_DIRECTORY and CUDA_MPS_LOG_DIRECTORY env vars
    (user-specific to avoid permission issues).

    Args:
        gpu_id: If provided, sets CUDA_VISIBLE_DEVICES. Otherwise assumes
                the caller already set it.
    """
    user = os.environ.get("USER", "unknown")
    pipe_dir = f"/tmp/nvidia-mps-{user}"
    log_dir = f"/tmp/nvidia-mps-log-{user}"

    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Kill old MPS
    logger.info("Cleaning up old MPS processes...")
    subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True,
                    capture_output=True, timeout=5)
    time.sleep(1)
    subprocess.run(f"pkill -u {user} -f nvidia-cuda-mps-control",
                    shell=True, capture_output=True)
    subprocess.run(f"pkill -u {user} -f nvidia-cuda-mps-server",
                    shell=True, capture_output=True)

    # Clean and recreate dirs
    for d in (pipe_dir, log_dir):
        subprocess.run(["rm", "-rf", d], capture_output=True)
    os.makedirs(pipe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Start daemon
    logger.info("Starting MPS daemon...")
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True, timeout=5)
    time.sleep(1)

    # Start server with correct UID
    uid = os.getuid()
    subprocess.run(
        f"echo 'start_server -uid {uid}' | nvidia-cuda-mps-control",
        shell=True, check=True, timeout=5,
    )
    logger.info("MPS daemon started.")


def stop_mps() -> None:
    """Stop the MPS daemon and clean up pipe/log directories."""
    user = os.environ.get("USER", "unknown")
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY",
                               f"/tmp/nvidia-mps-{user}")
    log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY",
                              f"/tmp/nvidia-mps-log-{user}")

    logger.info("Stopping MPS daemon...")
    subprocess.run("echo quit | nvidia-cuda-mps-control", shell=True,
                    capture_output=True, timeout=5)
    time.sleep(1)
    for d in (pipe_dir, log_dir):
        subprocess.run(["rm", "-rf", d], capture_output=True)
    logger.info("MPS daemon stopped.")


# ---------------------------------------------------------------------------
# Grid search helpers
# ---------------------------------------------------------------------------

def resolve_search_space(search_space: dict) -> list[dict]:
    """Expand a search space dict into a list of parameter combinations."""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _run_single_grid_point(
    pool,
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

    frames = []
    total_runs = warmup_runs + num_measured_runs

    for i in range(total_runs):
        is_warmup = i < warmup_runs
        result = pool.run_frame(
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
        "setting": "parallel_mps",
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
    """Save a result dict to JSON.

    File path: {results_dir}/parallel_mps/{policy_config}/denoise{N}_decode{M}.json
    """
    params = result["params"]
    out_dir = results_dir / "parallel_mps" / params["policy_config"]
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
    checkpoint_dir: Path | str | None = None,
    pytorch_device: str | None = None,
) -> list[dict]:
    """Run cartesian product of search_space for the parallel_mps setting.

    Args:
        policy: Ignored (workers create their own policies). Accepted for
                API compatibility with other settings.
        search_space: Parameter name -> list of values.
            Expected keys: policy_config, num_denoise_steps, max_decoding_steps.
        fixed_params: Parameters held constant (e.g. prompt).
        results_dir: Where to save raw JSON results.
        num_measured_runs: Repeated measurements per grid point.
        warmup_runs: Discarded warmup runs per grid point.
        checkpoint_dir: Optional checkpoint directory override. When it
            contains ``model.safetensors`` the workers use the PyTorch
            backend; otherwise they fall back to the default JAX checkpoint.
        pytorch_device: Optional PyTorch device override (e.g. ``"cuda:0"``).
            Only meaningful for PyTorch checkpoints.

    Returns:
        List of result dicts (one per grid point).
    """
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    prompt = fixed_params.get("prompt", "pick the red cup")

    # Group grid points by policy_config to reuse workers
    combos = resolve_search_space(search_space)
    by_policy: dict[str, list[dict]] = {}
    for params in combos:
        policy_cfg = params["policy_config"]
        by_policy.setdefault(policy_cfg, []).append(params)

    from experiments.parallel_mps.runner import MPSWorkerPool

    results = []
    total = len(combos)
    idx = 0

    for policy_cfg, policy_combos in by_policy.items():
        logger.info("Starting MPSWorkerPool for policy '%s'...", policy_cfg)
        with MPSWorkerPool(
            policy_cfg,
            prompt,
            checkpoint_dir=checkpoint_dir,
            pytorch_device=pytorch_device,
        ) as pool:
            for params in policy_combos:
                idx += 1
                logger.info("Grid point %d/%d: %s", idx, total, params)
                result = _run_single_grid_point(
                    pool, params, fixed_params,
                    num_measured_runs=num_measured_runs,
                    warmup_runs=warmup_runs,
                )
                _save_result(result, results_dir)
                results.append(result)

    return results
