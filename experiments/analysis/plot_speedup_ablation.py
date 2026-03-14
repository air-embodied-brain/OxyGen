"""Generate ablation bar chart: Shared KV + CB (ours) vs Shared KV only vs baseline.

Reads speedup_ablation.csv and produces absolute action frequency bar chart
with one subplot per policy.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.analysis.plot_utils import setup_style, PlotColors, BarStyle, _round_to_nice_value

POLICY_ORDER = ["pi05_o2_aloha", "pi05_o2_droid", "pi05_o2_libero"]
POLICY_LABELS = {"pi05_o2_aloha": "ALOHA", "pi05_o2_droid": "DROID", "pi05_o2_libero": "LIBERO"}

XLABEL = "Language Decoding Steps (Total)"


def _get_policies(df: pd.DataFrame) -> list[str]:
    """Return policies present in df, in canonical order."""
    return [p for p in POLICY_ORDER if p in df["policy"].values]


def plot_absolute(df: pd.DataFrame, plot_dir: Path, use_hatch: bool = False):
    """Horizontal overlapping bar chart for single policy ablation study."""
    policies = _get_policies(df)
    # Use first policy (libero by default based on POLICY_ORDER)
    policy = policies[0] if policies else None
    if policy is None:
        print("No policy data found")
        return

    psub = df[df["policy"] == policy].sort_values("max_decoding_steps")
    psub = psub.reset_index(drop=True)

    csv_out = plot_dir / "ablation_speedup.csv"
    psub.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")
    y_vals = psub["max_decoding_steps"].values
    y_pos = np.arange(len(y_vals))

    # Create horizontal bar chart with wider figure but constrained plot area
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    # fig, ax = plt.subplots(1, 1, figsize=(8.5, 1.8))

    bar_height = 0.6

    # Overlapping bars: Ours (tallest, behind) to Baseline (shortest, in front)
    ax.barh(y_pos, psub["cb_action_freq_hz"], bar_height,
            color=PlotColors.OURS_PRIMARY, edgecolor=BarStyle.EDGE_COLOR,
            linewidth=BarStyle.EDGE_WIDTH, hatch=BarStyle.HATCH_OURS if use_hatch else None,
            label="Ours", zorder=1)
    ax.barh(y_pos, psub["shared_kv_action_freq_hz"], bar_height,
            color=PlotColors.VARIANT_GOLD, edgecolor=BarStyle.EDGE_COLOR,
            linewidth=BarStyle.EDGE_WIDTH, hatch=BarStyle.HATCH_VARIANT if use_hatch else None,
            label="Ours w/o Batching", zorder=2)
    ax.barh(y_pos, psub["parallel_mps_action_freq_hz"], bar_height,
            color=PlotColors.MPS_PRIMARY, edgecolor=BarStyle.EDGE_COLOR,
            linewidth=BarStyle.EDGE_WIDTH, hatch=BarStyle.HATCH_MPS if use_hatch else None,
            label="Parallel", zorder=3)
    ax.barh(y_pos, psub["baseline_action_freq_hz"], bar_height,
            color=PlotColors.BASELINE_PRIMARY, edgecolor=BarStyle.EDGE_COLOR,
            linewidth=BarStyle.EDGE_WIDTH, hatch=BarStyle.HATCH_BASELINE if use_hatch else None,
            label="Baseline", zorder=4)

    # Upper Bound: vertical line (now that axes are swapped), extended and thicker
    # Calculate proper y range first to avoid line affecting axis limits
    y_min = -0.5
    y_max_pos = len(y_vals) - 0.5

    # Create extended y range for smooth line
    y_extended = np.linspace(y_min, y_max_pos, 200)
    upper_bound_interp = np.interp(y_extended, y_pos, psub["shared_kv_5_action_freq_hz"])
    ax.plot(upper_bound_interp, y_extended,
            color=PlotColors.NEUTRAL_DARK, linestyle=(0, (3, 1, 1, 1)), linewidth=1.8,
            label="Batching Upper Bound", zorder=5, clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_vals)
    ax.set_ylabel("Lang. Decoding\nSteps (Total)")
    ax.set_title(POLICY_LABELS.get(policy, policy), fontweight="bold", loc="center")

    # X-axis label will be placed separately below

    # Calculate x_max
    x_data_max = max([
        psub["baseline_action_freq_hz"].max(),
        psub["parallel_mps_action_freq_hz"].max(),
        psub["shared_kv_action_freq_hz"].max(),
        psub["cb_action_freq_hz"].max(),
    ])
    x_max = x_data_max * 1.3 # larger range to plot upper bound
    x_max_nice = _round_to_nice_value(x_max, x_data_max, small_range_threshold=30)

    for n_ticks in [5, 4, 3]:
        if x_max_nice <= x_data_max * 1.5:
            break
        x_max_nice = _round_to_nice_value(x_max * 0.9, x_data_max, small_range_threshold=30)
    
    # Let me hardcode one ...
    x_max_nice = 80
    n_ticks = 5

    ax.set_xlim(left=0, right=x_max_nice)
    ax.set_ylim(bottom=y_min, top=y_max_pos)

    x_ticks = np.linspace(0, x_max_nice, n_ticks)
    ax.set_xticks(x_ticks)

    ax.grid(axis='x', alpha=0.8, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Legend on the right side, single column, positioned higher
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = [
        "Ours", "Ours w/o Batching", "Parallel", "Baseline",
        "Batching Upper Bound",
    ]
    ordered_handles = [label_to_handle[l] for l in ordered_labels]

    ax.legend(ordered_handles, ordered_labels,
              loc="center left", bbox_to_anchor=(1, 0.6),
              frameon=False, fontsize=10)

    # Add x-axis label below the legend on the right
    fig.text(0.81, 0.2, "Action Frequency (Hz)", ha='center', fontsize=12, fontweight='bold')

    # Adjust subplot position to leave blank space on left and right
    # This must be done after tight_layout
    fig.tight_layout()
    # fig.subplots_adjust(left=0.15, right=0.65)

    out = plot_dir / "ablation_speedup.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root-dir",
        type=Path,
        default=Path("experiments/results"),
    )
    parser.add_argument(
        "--analysis-root-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--plot-root-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="pi05_o2_libero",
        help="Policy to plot (default: pi05_o2_libero)",
    )
    parser.add_argument(
        "--num-denoise-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-decoding-steps",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 30],
        help="Filter to these max_decoding_steps values (default: [5, 10, 15, 20, 30])",
    )
    parser.add_argument(
        "--use-hatch",
        action="store_true",
        help="Use hatch patterns for print-friendly plots",
    )
    args = parser.parse_args()

    analysis_root_dir = args.analysis_root_dir or args.results_root_dir / "analysis"
    plot_root_dir = args.plot_root_dir or args.results_root_dir / "plot"
    plot_dir = plot_root_dir / "plot_speedup_ablation"
    plot_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    csv_path = analysis_root_dir / "compute_speedup_ablation" / "speedup_ablation.csv"
    df = pd.read_csv(csv_path)

    if args.policy != "all":
        df = df[df["policy"] == args.policy]
    else:
        print("Error: --policy must specify a single policy, not 'all'")
        return
    df = df[df["num_denoise_steps"] == args.num_denoise_steps]
    df = df[df["max_decoding_steps"].isin(args.max_decoding_steps)]

    if df.empty:
        print(f"No data for policy={args.policy}, denoise={args.num_denoise_steps}")
        return

    plot_absolute(df, plot_dir, use_hatch=args.use_hatch)


if __name__ == "__main__":
    main()
