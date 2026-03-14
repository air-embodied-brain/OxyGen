"""Plot workload sweep: 1×3 subplots for uniform, poisson, random_length.

Each subplot shows action frequency bars (primary y) and avg batch size line
(secondary y).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.analysis.plot_utils import setup_style, PlotColors, BarStyle, round_dual_axis_ranges


SUBPLOT_CONFIGS = [
    {
        "title": "Uniform",
        "category": "uniform",
        "xlabel": "Arrivals per Frame",
    },
    {
        "title": "Poisson",
        "category": "poisson",
        "xlabel": r"Mean Arrivals per Frame ($\lambda$)",
    },
    {
        "title": "Random Length (5 or 20)",
        "category": "random_length",
        "xlabel": "Long / Short Ratio",
    },
]


def _plot_subplot(ax: plt.Axes, sub: pd.DataFrame, xlabel: str, col_idx: int,
                  y_data_max: float | None = None, batch_data_max: float | None = None,
                  use_hatch: bool = False):
    """Draw bars + reference line + batch size on a single subplot."""
    x_pos = np.arange(len(sub))
    labels = sub["param_label"].values
    action_freq = sub["action_frequency_hz"].values
    batch = sub["avg_batch_size"].values
    has_baseline = ("baseline_action_freq_hz" in sub.columns
                    and sub["baseline_action_freq_hz"].notna().any())

    w = 0.8

    # CB action frequency bars
    ax.bar(x_pos, action_freq, w, color=PlotColors.OURS_PRIMARY,
           edgecolor=BarStyle.EDGE_COLOR, linewidth=BarStyle.EDGE_WIDTH,
           hatch=BarStyle.HATCH_OURS if use_hatch else None,
           label="Ours Action Freq.")

    # Baseline: constant horizontal dotted reference line (single-request, not scaled)
    if has_baseline:
        bl_freq = sub["baseline_action_freq_hz"].values[0]
        ax.axhline(y=bl_freq, color=PlotColors.BASELINE_PRIMARY,
                   linestyle=(0, (4, 2)), linewidth=1.8,
                   label="Baseline Action Freq.", zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel(xlabel)
    ax.set_ylim(bottom=0)

    ax.grid(axis='y', alpha=0.8, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    if col_idx == 0:
        ax.set_ylabel("Action Frequency (Hz)")

    # Secondary y-axis for batch size (shared range across subplots)
    ax2 = ax.twinx()
    ax2.plot(x_pos, batch, color=PlotColors.NEUTRAL_DARK, marker="^", markersize=6,
             linewidth=1.5, markeredgecolor='white', markeredgewidth=0.6,
             label="Ours Avg. Batch Size")

    # Co-optimize dual-axis ranges with shared tick count
    y_max_nice, batch_y_max_nice, n_ticks = round_dual_axis_ranges(
        y_data_max, batch_data_max
    )
    ax.set_ylim(bottom=0, top=y_max_nice)
    ax2.set_ylim(bottom=0, top=batch_y_max_nice)

    # Create evenly spaced ticks for both axes
    lat_ticks = np.linspace(0, y_max_nice, n_ticks)
    batch_ticks = np.linspace(0, batch_y_max_nice, n_ticks)
    ax.set_yticks(lat_ticks)
    ax2.set_yticks(batch_ticks)

    if col_idx == len(SUBPLOT_CONFIGS) - 1:
        ax2.set_ylabel("Avg. Batch Size")

    return ax, ax2


def plot_workload_sweep(df: pd.DataFrame, plot_dir: Path, use_hatch: bool = False):
    """Generate the 1×3 workload sweep figure."""
    ncols = len(SUBPLOT_CONFIGS)
    fig, axes = plt.subplots(1, ncols, figsize=(2.5 * ncols + 1, 2), squeeze=False)

    csv_parts = []
    for ci, cfg in enumerate(SUBPLOT_CONFIGS):
        ax = axes[0, ci]
        sub = df[df["category"] == cfg["category"]].copy()

        # Filter out excluded labels
        exclude = cfg.get("exclude_labels", set())
        if exclude:
            sub = sub[~sub["param_label"].isin(exclude)]

        sub = sub.sort_values("param_value").reset_index(drop=True)

        if sub.empty:
            ax.set_title(cfg["title"])
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        # Per-subplot y ranges
        y_data_max = sub["action_frequency_hz"].max()
        batch_data_max = sub["avg_batch_size"].max()

        csv_parts.append(sub)
        ax.set_title(cfg["title"], fontweight="bold")
        _plot_subplot(ax, sub, cfg["xlabel"], ci, y_data_max=y_data_max,
                      batch_data_max=batch_data_max, use_hatch=use_hatch)

    if csv_parts:
        csv_out = plot_dir / "workload_sweep.csv"
        pd.concat(csv_parts, ignore_index=True).to_csv(csv_out, index=False)
        print(f"Wrote {csv_out}")

    # Shared legend at bottom
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=PlotColors.OURS_PRIMARY, edgecolor=BarStyle.EDGE_COLOR,
              linewidth=BarStyle.EDGE_WIDTH, label="Ours Action Freq."),
        Line2D([0], [0], color=PlotColors.BASELINE_PRIMARY,
               linestyle=(0, (4, 2)), linewidth=1.8,
               label="Baseline Action Freq."),
        Line2D([0], [0], color=PlotColors.NEUTRAL_DARK, marker="^", markersize=6,
               linewidth=1.5, markeredgecolor='white', markeredgewidth=0.6,
               label="Ours Avg. Batch Size"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,
               fontsize=12, columnspacing=2.0, handletextpad=0.5)

    fig.tight_layout(h_pad=0.5)
    fig.subplots_adjust(bottom=0.18)
    out = plot_dir / "workload_sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root-dir", type=Path, default=Path("experiments/results"),
    )
    parser.add_argument(
        "--analysis-root-dir", type=Path, default=None,
    )
    parser.add_argument(
        "--plot-root-dir", type=Path, default=None,
    )
    parser.add_argument(
        "--use-hatch",
        action="store_true",
        help="Use hatch patterns for print-friendly plots",
    )
    args = parser.parse_args()

    analysis_root = args.analysis_root_dir or args.results_root_dir / "analysis"
    plot_root = args.plot_root_dir or args.results_root_dir / "plot"
    plot_dir = plot_root / "plot_workload_sweep"
    plot_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    csv_path = analysis_root / "compute_workload_sweep" / "workload_sweep.csv"
    df = pd.read_csv(csv_path)

    if df.empty:
        print("No data in workload_sweep.csv")
        return

    plot_workload_sweep(df, plot_dir, use_hatch=args.use_hatch)


if __name__ == "__main__":
    main()

