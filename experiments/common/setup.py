"""Shared setup utilities for experiments."""

import datetime
import os
import subprocess
from pathlib import Path

import jax
import torch

from openpi.policies.policy_config import create_trained_policy
from openpi.shared import download
from openpi.training.config import get_config


def setup_jax_cache(cache_dir: str | Path | None = None) -> Path:
    """Enable persistent JAX compilation cache.

    Args:
        cache_dir: Directory for the cache. Defaults to ~/.cache/jax_compilation_cache.

    Returns:
        The resolved cache directory path.
    """
    if cache_dir is None:
        cache_dir = Path(os.path.expanduser("~/.cache/jax_compilation_cache"))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    return cache_dir


def resolve_policy_checkpoint(policy_config: str) -> Path:
    """Download and return the checkpoint path for a given policy config name.

    Args:
        policy_config: Policy config name (e.g. "pi05_o2_droid").

    Returns:
        Local path to the downloaded checkpoint.
    """
    if "droid" in policy_config:
        return download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
    elif "libero" in policy_config:
        return download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    else:
        return download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")


def create_policy(
    policy_config: str,
    *,
    checkpoint_dir: str | Path | None = None,
    pytorch_device: str | None = None,
):
    """Create a trained policy from a config name.

    Args:
        policy_config: Policy config name (e.g. "pi05_o2_droid").

    Returns:
        An initialized Policy object.
    """
    config = get_config(policy_config)
    checkpoint = Path(checkpoint_dir) if checkpoint_dir is not None else resolve_policy_checkpoint(policy_config)
    return create_trained_policy(config, checkpoint, pytorch_device=pytorch_device)


def collect_metadata() -> dict:
    """Collect environment metadata for result reproducibility.

    Returns:
        Dict with timestamp, GPU name, and JAX version.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu = "unknown"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpu = result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {
        "timestamp": timestamp,
        "gpu": gpu,
        "jax_version": jax.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
    }
