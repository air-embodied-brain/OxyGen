"""Copy generated plot PDFs from results tree into the paper figures directory."""

import argparse
import shutil
from pathlib import Path

PLOTS_TO_COPY = [
    "e2e_speedup_heatmap_vs_decoding_steps_spf5.pdf",
    "e2e_speedup_heatmap_vs_steps_per_frame.pdf",
    "e2e_latency_throughput_vs_decoding_steps.pdf",
    "e2e_latency_throughput_vs_steps_per_frame.pdf",
    "ablation_speedup.pdf",
    "workload_sweep.pdf",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root-dir",
        type=Path,
        default=Path("experiments/results"),
    )
    parser.add_argument(
        "--paper-figures-dir",
        type=Path,
        default=Path("ECCV-2026---MoT-VLA-Inference/figures/experiments"),
    )
    args = parser.parse_args()

    args.paper_figures_dir.mkdir(parents=True, exist_ok=True)

    for filename in PLOTS_TO_COPY:
        matches = list(args.results_root_dir.rglob(filename))
        if not matches:
            print(f"WARNING: {filename} not found under {args.results_root_dir}")
            continue
        src = matches[0]
        if len(matches) > 1:
            print(f"NOTE: multiple matches for {filename}, using {src}")
        dst = args.paper_figures_dir / filename
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    main()
