"""Run continuous batching under different arrival patterns (workload sweep).

Usage:
    uv run python -m experiments.run_workload_sweep --gpu 0
    uv run python -m experiments.run_workload_sweep --gpu 0 --policy pi05_o2_droid
"""

import argparse
import logging
import os
from pathlib import Path

from experiments.common.setup import create_policy, setup_jax_cache
from experiments.continuous_batching.grid_search import (
    _arrival_slug,
    _run_single_grid_point,
    _save_result,
    warmup_batch_sizes,
)
from experiments.continuous_batching.runner import _parse_arrival_pattern

logger = logging.getLogger(__name__)

# Fixed CB parameters for the sweep
NUM_DENOISE_STEPS = 10
MAX_DECODING_STEPS = 30
STEPS_PER_FRAME = 5
TOTAL_FRAMES = 50
PROMPT = "pick the red cup"


def _build_arrival_patterns() -> list[str]:
    """Return the list of arrival pattern strings to sweep."""
    patterns = []

    # Uniform: rate >= 1 means 1 request every N frames;
    #          rate < 1 means multiple requests per frame
    for rate in [0.25, 0.5, 1, 2, 4]:
        patterns.append(f"uniform_arrivals(rate={rate}, t_max={MAX_DECODING_STEPS})")

    # Poisson
    for lam in [0.5, 1.0, 1.5, 2.0]:
        patterns.append(f"poisson_arrivals(lam={lam}, t_max={MAX_DECODING_STEPS})")

    # Random length distributions: vary ratio of long (t_max=20) requests
    for long_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        short_ratio = round(1.0 - long_ratio, 1)
        patterns.append(
            f"random_length_arrivals(t_max_values=[5, 20], "
            f"weights=[{short_ratio}, {long_ratio}], rate=1, seed=42)"
        )

    return patterns


def _simulate_peak_batch_size(
    arrival_pattern: str,
    total_frames: int,
    steps_per_frame: int,
) -> int:
    """Simulate arrival pattern to find the peak concurrent batch size."""
    arrival_fn = _parse_arrival_pattern(arrival_pattern)
    active: list[int] = []  # remaining steps per active request
    peak = 0
    for frame_idx in range(total_frames):
        new_t_maxes = arrival_fn(frame_idx)
        active.extend(new_t_maxes)
        # Each request gets steps_per_frame tokens this frame
        active = [t - steps_per_frame for t in active]
        active = [t for t in active if t > 0]
        # New arrivals + continuing = batch size this frame
        batch_size = len(new_t_maxes) + len(active)
        peak = max(peak, batch_size)
    return peak


def run_sweep(
    policy_config: str,
    results_dir: Path,
    gpu_id: int = 0,
    category: str | None = None,
) -> None:
    """Run workload sweep for a single policy."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    setup_jax_cache()

    logger.info("Creating policy '%s'...", policy_config)
    policy = create_policy(policy_config)

    patterns = _build_arrival_patterns()

    # Filter by category if specified
    if category:
        _PREFIX_MAP = {
            "uniform": "uniform_arrivals(",
            "poisson": "poisson_arrivals(",
            "random_length": "random_length_arrivals(",
        }
        prefix = _PREFIX_MAP[category]
        patterns = [p for p in patterns if p.startswith(prefix)]
        logger.info("Filtered to %d '%s' patterns.", len(patterns), category)

    # Collect all batch sizes we'll need across all patterns
    all_batch_sizes: set[int] = set()
    pattern_peaks: dict[str, int] = {}
    for pattern in patterns:
        peak = _simulate_peak_batch_size(pattern, TOTAL_FRAMES, STEPS_PER_FRAME)
        pattern_peaks[pattern] = peak
        all_batch_sizes.add(peak)
        logger.info("Pattern '%s' → peak batch size %d", _arrival_slug(pattern), peak)

    # Warmup JIT for batch sizes 1..max_peak
    max_peak = max(all_batch_sizes) + 1  # +1 safety margin
    logger.info("Warming up JIT for batch sizes 1..%d", max_peak)
    warmup_batch_sizes(
        policy,
        policy_config=policy_config,
        prompt=PROMPT,
        max_batch=max_peak,
        num_denoise_steps=NUM_DENOISE_STEPS,
        max_decoding_steps=MAX_DECODING_STEPS,
        steps_per_frame=STEPS_PER_FRAME,
    )

    # Run each pattern
    for pattern in patterns:
        slug = _arrival_slug(pattern)
        logger.info("Running pattern: %s (slug=%s)", pattern, slug)

        params = {
            "policy_config": policy_config,
            "num_denoise_steps": NUM_DENOISE_STEPS,
            "max_decoding_steps": MAX_DECODING_STEPS,
            "steps_per_frame": STEPS_PER_FRAME,
        }
        fixed_params = {
            "total_frames": TOTAL_FRAMES,
            "arrival_pattern": pattern,
            "prompt": PROMPT,
        }
        result = _run_single_grid_point(policy, params, fixed_params)
        _save_result(result, results_dir)
        logger.info("  → saved result for %s", slug)

    logger.info("Workload sweep complete for '%s'.", policy_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", default="pi05_o2_libero",
        help="Policy config name (default: pi05_o2_libero).",
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
        "--category", type=str, default=None,
        choices=["uniform", "poisson", "random_length"],
        help="Only run patterns of this category (default: all).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    for handler in logging.root.handlers:
        if hasattr(handler, "flush"):
            original_emit = handler.emit
            def _make_flushing(emit, h):
                def flushing_emit(record):
                    emit(record)
                    h.flush()
                return flushing_emit
            handler.emit = _make_flushing(original_emit, handler)

    run_sweep(
        policy_config=args.policy,
        results_dir=args.results_dir,
        gpu_id=args.gpu,
        category=args.category,
    )


if __name__ == "__main__":
    main()
