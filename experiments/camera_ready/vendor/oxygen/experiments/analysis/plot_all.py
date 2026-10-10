#!/usr/bin/env python3
"""Run all three plotting scripts with default parameters."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"SUCCESS: {description} completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Root directory for experiment results",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=5,
        help="Steps per frame for CB plots",
    )
    parser.add_argument(
        "--num-denoise-steps",
        type=int,
        default=10,
        help="Number of denoise steps",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="all",
        help="Filter to specific policy (default: all)",
    )
    parser.add_argument(
        "--use-hatch",
        action="store_true",
        help="Use hatch patterns for print-friendly plots",
    )
    args = parser.parse_args()

    # 1. Run plot_speedup_cb
    cmd_cb = [
        "uv", "run", "python", "-m", "experiments.analysis.plot_speedup_cb",
        "--results-root-dir", str(args.results_root_dir),
        "--plot-type", "all",
        "--steps-per-frame", str(args.steps_per_frame),
        "--num-denoise-steps", str(args.num_denoise_steps),
        "--policy", args.policy,
    ]
    if args.use_hatch:
        cmd_cb.append("--use-hatch")
    run_command(cmd_cb, "plot_speedup_cb (heatmaps and line plots)")

    # 2. Run plot_speedup_ablation
    cmd_ablation = [
        "uv", "run", "python", "-m", "experiments.analysis.plot_speedup_ablation",
        "--results-root-dir", str(args.results_root_dir),
        "--policy", "pi05_o2_libero",
        "--num-denoise-steps", str(args.num_denoise_steps),
        # "--max-decoding-steps", "10", "20", "30",
    ]
    if args.use_hatch:
        cmd_ablation.append("--use-hatch")
    run_command(cmd_ablation, "plot_speedup_ablation (ablation bar chart)")

    # 3. Run plot_workload_sweep
    cmd_sweep = [
        "uv", "run", "python", "-m", "experiments.analysis.plot_workload_sweep",
        "--results-root-dir", str(args.results_root_dir),
    ]
    if args.use_hatch:
        cmd_sweep.append("--use-hatch")
    run_command(cmd_sweep, "plot_workload_sweep (workload sweep plots)")

    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"Output directory: {args.results_root_dir}/plot/")
    print('='*60)


if __name__ == "__main__":
    main()
