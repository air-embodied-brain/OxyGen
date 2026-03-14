"""Top-level orchestrator for running all experiment settings.

Usage:
    uv run python -m experiments.run_experiments --settings baseline shared_kv --gpu 0
    uv run python -m experiments.run_experiments --settings parallel_mps --policies pi05_o2_droid --gpu 0
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from experiments.common.setup import create_policy, setup_jax_cache

logger = logging.getLogger(__name__)

# Default search spaces per setting (see experiments/README.md)
DEFAULT_SEARCH_SPACE = {
    "policy_config": ["pi05_o2_droid"],
    "num_denoise_steps": [5, 10, 15, 20],
    "max_decoding_steps": [5, 10, 15, 20, 30],
}

# Continuous batching has extra parameters
CB_SEARCH_SPACE_EXTRA = {
    "steps_per_frame": [1, 2, 3, 5, 6, 10, 15, 30],
}

CB_FIXED_PARAMS = {
    "total_frames": 50,
    "arrival_pattern": "uniform_arrivals(rate=1)",
}

SETTINGS_WITH_SHARED_POLICY = {"baseline", "shared_kv", "continuous_batching"}
SETTINGS_WITHOUT_POLICY = {"parallel_mps"}
ALL_SETTINGS = ["baseline", "shared_kv", "continuous_batching", "parallel_mps"]


def _import_grid_search(setting: str):
    """Dynamically import the run_grid_search function for a setting."""
    if setting == "baseline":
        from experiments.baseline.grid_search import run_grid_search
    elif setting == "shared_kv":
        from experiments.shared_kv.grid_search import run_grid_search
    elif setting == "continuous_batching":
        from experiments.continuous_batching.grid_search import run_grid_search
    elif setting == "parallel_mps":
        from experiments.parallel_mps.grid_search import run_grid_search
    else:
        raise ValueError(f"Unknown setting: {setting}")
    return run_grid_search


def _build_search_space(
    setting: str,
    policy_configs: list[str],
) -> dict:
    """Build the search space dict for a given setting."""
    space = dict(DEFAULT_SEARCH_SPACE)
    space["policy_config"] = policy_configs

    if setting == "continuous_batching":
        space.update(CB_SEARCH_SPACE_EXTRA)

    return space


def _build_fixed_params(setting: str, prompt: str) -> dict:
    """Build the fixed_params dict for a given setting."""
    fixed = {"prompt": prompt}
    if setting == "continuous_batching":
        fixed.update(CB_FIXED_PARAMS)
    return fixed


def _run_setting_in_subprocess(
    setting: str,
    policy_configs: list[str],
    results_dir: Path,
    gpu_id: int,
    prompt: str,
) -> None:
    """Run a single setting in an isolated subprocess to reclaim GPU memory on exit."""
    cmd = [
        sys.executable, "-m", "experiments.run_experiments",
        "--_isolated",
        "--settings", setting,
        "--policies", *policy_configs,
        "--results-dir", str(results_dir),
        "--gpu", str(gpu_id),
        "--prompt", prompt,
    ]
    logger.info("Launching subprocess for setting '%s': %s", setting, " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess for setting '{setting}' exited with code {result.returncode}"
        )


def run_all(
    settings: list[str],
    policy_configs: list[str],
    results_dir: Path,
    gpu_id: int = 0,
    prompt: str = "pick the red cup",
) -> None:
    """Run grid search for each requested setting.

    Args:
        settings: List of setting names to run.
        policy_configs: Policy config names to sweep.
        results_dir: Root directory for saving results.
        gpu_id: GPU index to use.
        prompt: Prompt string for synthetic observations.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    setup_jax_cache()

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for setting in settings:
        logger.info("=" * 60)
        logger.info("Running setting: %s", setting)
        logger.info("=" * 60)

        run_grid_search = _import_grid_search(setting)
        fixed_params = _build_fixed_params(setting, prompt)

        if setting in SETTINGS_WITH_SHARED_POLICY:
            # Create a separate policy per config so transforms/checkpoint match
            for pc in policy_configs:
                logger.info("Creating policy for config '%s'...", pc)
                policy = create_policy(pc)

                search_space = _build_search_space(setting, [pc])
                kwargs = dict(
                    policy=policy,
                    search_space=search_space,
                    fixed_params=fixed_params,
                    results_dir=results_dir,
                )
                if setting in ("baseline", "shared_kv"):
                    kwargs["num_measured_runs"] = 3
                    kwargs["warmup_runs"] = 1

                results = run_grid_search(**kwargs)
                logger.info("Setting '%s' config '%s' complete: %d grid points.",
                            setting, pc, len(results))
        elif setting == "parallel_mps":
            # parallel_mps creates its own policies internally
            search_space = _build_search_space(setting, policy_configs)
            kwargs = dict(
                policy=None,
                search_space=search_space,
                fixed_params=fixed_params,
                results_dir=results_dir,
                num_measured_runs=3,
                warmup_runs=1,
            )
            from experiments.parallel_mps.grid_search import start_mps, stop_mps
            start_mps(gpu_id=gpu_id)
            try:
                results = run_grid_search(**kwargs)
            finally:
                stop_mps()
            logger.info("Setting '%s' complete: %d grid points.", setting,
                         len(results))
        else:
            raise ValueError(f"Unknown setting category: {setting}")

    logger.info("All settings complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment grid searches.",
    )
    parser.add_argument(
        "--settings", nargs="+", default=ALL_SETTINGS,
        choices=ALL_SETTINGS,
        help="Which settings to run (default: all).",
    )
    parser.add_argument(
        "--policies", nargs="+", default=["pi05_o2_droid"],
        help="Policy config names to sweep.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("experiments/results"),
        help="Root directory for saving results.",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU index to use.",
    )
    parser.add_argument(
        "--prompt", default="pick the red cup",
        help="Prompt for synthetic observations.",
    )
    parser.add_argument(
        "--_isolated", action="store_true", help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    # force=True: override any handlers installed by imported libraries (JAX/TF)
    # so our log messages actually appear.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    # Flush after every log message so output appears before long JAX compilations
    for handler in logging.root.handlers:
        if hasattr(handler, "flush"):
            original_emit = handler.emit
            def _make_flushing(emit, h):
                def flushing_emit(record):
                    emit(record)
                    h.flush()
                return flushing_emit
            handler.emit = _make_flushing(original_emit, handler)

    if args._isolated or len(args.settings) <= 1:
        run_all(
            settings=args.settings,
            policy_configs=args.policies,
            results_dir=args.results_dir,
            gpu_id=args.gpu,
            prompt=args.prompt,
        )
    else:
        for setting in args.settings:
            _run_setting_in_subprocess(
                setting=setting,
                policy_configs=args.policies,
                results_dir=args.results_dir,
                gpu_id=args.gpu,
                prompt=args.prompt,
            )


if __name__ == "__main__":
    main()
