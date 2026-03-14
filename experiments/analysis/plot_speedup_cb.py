"""Generate publication-quality figures comparing CB vs baseline.

Reads speedup_cb_unified.csv and produces:
  A) Heatmap grids (one per comparison_method × steps_per_frame)
  B) Line plot of absolute action frequency/throughput (CB vs same-config baseline)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.analysis.plot_utils import setup_style, PlotColors, BarStyle, round_dual_axis_ranges

POLICY_ORDER = ["pi05_o2_aloha", "pi05_o2_droid", "pi05_o2_libero"]
POLICY_LABELS = {"pi05_o2_aloha": "ALOHA", "pi05_o2_droid": "DROID", "pi05_o2_libero": "LIBERO"}


def load_data(analysis_root_dir: Path) -> pd.DataFrame:
    """Load speedup_cb_unified.csv."""
    path = analysis_root_dir / "compute_speedup_cb" / "speedup_cb_unified.csv"
    df = pd.read_csv(path)
    df["steps_per_frame"] = df["steps_per_frame"].astype(int)
    return df


def _get_policies(df: pd.DataFrame) -> list[str]:
    """Return policies present in df, in canonical order."""
    return [p for p in POLICY_ORDER if p in df["policy"].values]


def plot_heatmap(df: pd.DataFrame, plot_dir: Path, comparison: str = "same"):
    """Generate heatmap grid per steps_per_frame.

    One PDF per (comparison_method × steps_per_frame).
    1×3 grid: single row = speedup, cols = (policy).

    For "same": x-axis = total decoding steps (both ours and baseline have same total)
    For "trunc": x-axis = decoding steps per frame (baseline fixed at 30 total)
    """
    if comparison == "same":
        speedup_col = "same_action_frequency_speedup"
        x_col = "max_decoding_steps"
        xlabel = "Language Decoding Steps (Total)"
    else:
        speedup_col = "same_action_frequency_speedup" # Do not use trunc speedup (outdated)
        x_col = "steps_per_frame"
        xlabel = "Language Decoding Steps (per Frame)"
        # Filter to baseline with 30 total decoding steps and specific steps per frame
        df = df[(df["max_decoding_steps"] == 30) & (df["steps_per_frame"].isin([1,2,3,5,10]))].copy()

    policies = _get_policies(df)
    ncols = len(policies)

    if comparison == "same":
        # Iterate over steps_per_frame for "same" comparison
        spf_values = sorted(df["steps_per_frame"].unique())
    else:
        # For "trunc", we only have one configuration (max_decoding_steps=30)
        # Group by num_denoise_steps instead
        spf_values = [None]  # Single plot

    for spf in spf_values:
        if comparison == "same":
            sub = df[df["steps_per_frame"] == spf]
            suffix = f"_spf{spf}"
        else:
            sub = df
            suffix = ""

        fig, axes = plt.subplots(1, ncols, figsize=(2.5 * ncols + 1, 2.5), squeeze=False, constrained_layout=True)

        vmin = sub[speedup_col].min()
        vmax = sub[speedup_col].max()

        for ci, policy in enumerate(policies):
            ax = axes[0, ci]
            psub = sub[sub["policy"] == policy]
            pivot = psub.pivot_table(
                index="num_denoise_steps", columns=x_col,
                values=speedup_col, aggfunc="mean",
            )
            pivot = pivot.sort_index(ascending=True)
            pivot = pivot[sorted(pivot.columns)]

            im = ax.imshow(
                pivot.values, aspect="auto", cmap="YlOrRd",
                vmin=vmin, vmax=vmax, origin="lower",
            )
            # Annotate cells
            for y in range(pivot.shape[0]):
                for x in range(pivot.shape[1]):
                    val = pivot.values[y, x]
                    if np.isfinite(val):
                        # White text on dark cells for contrast
                        norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        text_color = "white" if norm_val > 0.6 else "black"
                        ax.text(x, y, f"{val:.2f}", ha="center", va="center", color=text_color)

            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)

            ax.set_title(POLICY_LABELS.get(policy, policy), fontweight="bold")
            # Only show x-axis title on the middle subplot
            if ci == ncols // 2:
                ax.set_xlabel(xlabel)
            if ci == 0:
                ax.set_ylabel("Action Denoising Steps")
            else:
                ax.set_yticklabels([])

        # Shared colorbar
        cbar = fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Speedup", rotation=270, labelpad=15)

        comp_label = "decoding_steps" if comparison == "same" else "steps_per_frame"
        csv_out = plot_dir / f"e2e_speedup_heatmap_vs_{comp_label}{suffix}.csv"
        sub.to_csv(csv_out, index=False)
        print(f"Wrote {csv_out}")
        out = plot_dir / f"e2e_speedup_heatmap_vs_{comp_label}{suffix}.pdf"
        fig.savefig(out)
        plt.close(fig)
        print(f"Wrote {out}")


def _plot_dual_axis(
    df: pd.DataFrame,
    policies: list[str],
    x_col: str,
    cb_freq_col: str,
    bl_freq_col: str,
    mps_freq_col: str,
    cb_thr_col: str,
    bl_thr_col: str,
    mps_thr_col: str,
    bl_label: str,
    xlabel: str,
    title: str,
    out_path: Path,
    use_hatch: bool = False,
):
    """Shared dual-axis plot: bars for action frequency (left y), curves for throughput (right y).

    1×N subplots (N = policies). Left y = action frequency bars, right y = throughput lines.
    Includes baseline, MPS, and CB (ours).
    """
    bar_width = 0.28  # increased bar width

    ncols = max(len(policies), 1)
    fig, axes = plt.subplots(1, ncols, figsize=(2.5 * ncols + 1, 2), squeeze=False)

    for ci, policy in enumerate(policies):
        ax = axes[0, ci]
        psub = df[df["policy"] == policy].sort_values(x_col)
        x_vals = psub[x_col].values
        x_pos = np.arange(len(x_vals))

        ax.bar(x_pos - bar_width, psub[cb_freq_col].values,
               bar_width, color=PlotColors.OURS_PRIMARY,
               edgecolor=BarStyle.EDGE_COLOR, linewidth=BarStyle.EDGE_WIDTH,
               hatch=BarStyle.HATCH_OURS if use_hatch else None,
               label="Ours Action Freq.")
        ax.bar(x_pos, psub[mps_freq_col].values,
               bar_width, color=PlotColors.MPS_PRIMARY,
               edgecolor=BarStyle.EDGE_COLOR, linewidth=BarStyle.EDGE_WIDTH,
               hatch=BarStyle.HATCH_MPS if use_hatch else None,
               label="Parallel Action Freq.")
        ax.bar(x_pos + bar_width, psub[bl_freq_col].values,
               bar_width, color=PlotColors.BASELINE_PRIMARY,
               edgecolor=BarStyle.EDGE_COLOR, linewidth=BarStyle.EDGE_WIDTH,
               hatch=BarStyle.HATCH_BASELINE if use_hatch else None,
               label=f"{bl_label} Action Freq.")
        if ci == 0:
            ax.set_ylabel("Action Frequency (Hz)")
        # Only show x-axis label on middle subplot
        if ci == ncols // 2:
            ax.set_xlabel(xlabel)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_vals)
        ax.set_title(POLICY_LABELS.get(policy, policy), fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.8, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        ax2 = ax.twinx()
        ax2.plot(x_pos, psub[cb_thr_col].values,
                 marker="o", color=PlotColors.OURS_DARK, linewidth=1.5, markersize=6,
                 markeredgecolor='white', markeredgewidth=0.6,
                 label="Ours Throughput")
        ax2.plot(x_pos, psub[mps_thr_col].values,
                 marker="^", color=PlotColors.MPS_DARK, linewidth=1.5, markersize=6,
                 markeredgecolor='white', markeredgewidth=0.6,
                 label="Parallel Throughput")
        ax2.plot(x_pos, psub[bl_thr_col].values,
                 marker="s", color=PlotColors.BASELINE_DARK, linewidth=1.5, markersize=6,
                 markeredgecolor='white', markeredgewidth=0.6,
                 label=f"{bl_label} Throughput")
        ax2.set_ylim(bottom=0)

        # Co-optimize dual-axis ranges with shared tick count
        freq_max = psub[[cb_freq_col, bl_freq_col, mps_freq_col]].max().max()
        thr_max = psub[[cb_thr_col, bl_thr_col, mps_thr_col]].max().max()

        freq_max_nice, thr_max_nice, n_ticks = round_dual_axis_ranges(
            freq_max, thr_max
        )

        ax.set_ylim(top=freq_max_nice)
        ax2.set_ylim(top=thr_max_nice)

        # Create evenly spaced ticks for both axes
        freq_ticks = np.linspace(0, freq_max_nice, n_ticks)
        thr_ticks = np.linspace(0, thr_max_nice, n_ticks)

        ax.set_yticks(freq_ticks)
        ax2.set_yticks(thr_ticks)

        # Hide left y-axis tick labels for non-leftmost subplots
        if ci != 0:
            ax.set_yticklabels([])

        # Only show right y-axis label on rightmost subplot
        if ci == ncols - 1:
            ax2.set_ylabel("Throughput (tok/s)")
        else:
            ax2.set_yticklabels([])

        # Add speedup annotation at the position with largest throughput speedup
        cb_thr = psub[cb_thr_col].values
        bl_thr = psub[bl_thr_col].values
        speedups = cb_thr / bl_thr
        max_speedup_idx = np.argmax(speedups)
        max_speedup = speedups[max_speedup_idx]

        # Arrow from baseline to ours on throughput axis
        x_annot = x_pos[max_speedup_idx]
        y_bl = bl_thr[max_speedup_idx]
        y_cb = cb_thr[max_speedup_idx]
        y_annot = 0.5 * y_bl + 0.5 * y_cb
        k = 0.1
        y_start = (1 - k) * y_bl + k * y_cb
        y_end = k * y_bl + (1 - k) * y_cb

        # Draw arrow on ax2 (throughput axis)
        arrow_color = '#000000'
        ax2.annotate(
            '',
            xy=(x_annot, y_end),
            xytext=(x_annot, y_start),
            arrowprops=dict(
                arrowstyle='->',
                color=arrow_color,
                lw=1.5,
                shrinkA=0,
                shrinkB=0
            )
        )
        # Text annotation
        ax2.text(
            x_annot,
            y_annot,
            f'{max_speedup:.1f}×',
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold',
            color=arrow_color,
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='cornsilk',
                edgecolor=arrow_color,
                linewidth=1,
                alpha=1
            )
        )

    # Combined legend with specified order
    handles, labels = [], []
    for a in fig.axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    # Reorder legend: Ours Action Freq., Ours Throughput, Parallel Action Freq., ... Throughput, Baseline Action Freq., ... Throughput
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = [
        "Ours Action Freq.", "Ours Throughput",
        "Parallel Action Freq.", "Parallel Throughput",
        f"{bl_label} Action Freq.", f"{bl_label} Throughput",
    ]
    ordered_handles = [label_to_handle[l] for l in ordered_labels if l in label_to_handle]
    ordered_labels = [l for l in ordered_labels if l in label_to_handle]

    fig.legend(ordered_handles, ordered_labels, loc="upper center",
               bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_line(df: pd.DataFrame, plot_dir: Path, use_hatch: bool = False):
    """Line plot: CB vs same-config baseline by max_decoding_steps."""
    spf = df["steps_per_frame"].max()
    sub = df[df["steps_per_frame"] == spf].copy()

    # Filter to [10, 20, 30] decoding steps
    sub = sub[sub["max_decoding_steps"].isin([5, 10, 15, 20, 30])]

    # Load MPS data for comparison
    analysis_root_dir = plot_dir.parent.parent / "analysis"
    metrics_df = pd.read_csv(analysis_root_dir / "aggregate_metrics" / "metrics.csv")
    mps_df = metrics_df[metrics_df["setting"] == "parallel_mps"].copy()
    mps_df = mps_df[["policy", "num_denoise_steps", "max_decoding_steps", "action_frequency_hz", "language_throughput_tps"]]
    mps_df = mps_df.rename(columns={
        "action_frequency_hz": "mps_action_frequency_hz",
        "language_throughput_tps": "mps_language_throughput_tps"
    })

    # Merge MPS data
    sub = sub.merge(mps_df, on=["policy", "num_denoise_steps", "max_decoding_steps"], how="left")

    csv_out = plot_dir / "e2e_latency_throughput_vs_decoding_steps.csv"
    sub.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")

    _plot_dual_axis(
        df=sub,
        policies=_get_policies(sub),
        x_col="max_decoding_steps",
        cb_freq_col="action_frequency_hz",
        bl_freq_col="same_baseline_action_frequency_hz",
        mps_freq_col="mps_action_frequency_hz",
        cb_thr_col="language_throughput_tps",
        bl_thr_col="same_baseline_language_throughput_tps",
        mps_thr_col="mps_language_throughput_tps",
        bl_label="Baseline",
        xlabel="Language Decoding Steps (Total)",
        title=f"Ours vs. Baseline (Decoding Steps per Frame ={spf})",
        out_path=plot_dir / "e2e_latency_throughput_vs_decoding_steps.pdf",
        use_hatch=use_hatch,
    )


def plot_line_spf(df: pd.DataFrame, plot_dir: Path, use_hatch: bool = False):
    """Line plot: CB vs same-config baseline by steps_per_frame.

    Fixes max_decoding_steps per spf (e.g., 30 for all spf values) to isolate variables.
    Uses same-config baseline (fixed decoding length) instead of truncated baseline.
    """
    SPF_MDS_MAP = {1:30, 2: 30, 3: 30, 5: 30, 10: 30}  # Filter to [1, 3, 5] steps per frame

    rows = []
    for spf in sorted(df["steps_per_frame"].unique()):
        if spf not in SPF_MDS_MAP:  # Skip spf values not in filter
            continue
        target_mds = SPF_MDS_MAP.get(spf)
        spf_df = df[df["steps_per_frame"] == spf]
        if target_mds is not None:
            matched = spf_df[spf_df["max_decoding_steps"] == target_mds]
            if not matched.empty:
                rows.append(matched)

    if not rows:
        print("No data for line_spf plot")
        return
    sub = pd.concat(rows)

    # Load MPS data for comparison
    analysis_root_dir = plot_dir.parent.parent / "analysis"
    metrics_df = pd.read_csv(analysis_root_dir / "aggregate_metrics" / "metrics.csv")
    mps_df = metrics_df[metrics_df["setting"] == "parallel_mps"].copy()
    mps_df = mps_df[["policy", "num_denoise_steps", "max_decoding_steps", "action_frequency_hz", "language_throughput_tps"]]
    mps_df = mps_df.rename(columns={
        "action_frequency_hz": "mps_action_frequency_hz",
        "language_throughput_tps": "mps_language_throughput_tps"
    })

    # Merge MPS data
    sub = sub.merge(mps_df, on=["policy", "num_denoise_steps", "max_decoding_steps"], how="left")

    csv_out = plot_dir / "e2e_latency_throughput_vs_steps_per_frame.csv"
    sub.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")

    mds_vals = sorted(SPF_MDS_MAP.values())
    mds_str = str(mds_vals[0]) if len(set(mds_vals)) == 1 else ",".join(str(v) for v in mds_vals)
    _plot_dual_axis(
        df=sub,
        policies=_get_policies(sub),
        x_col="steps_per_frame",
        cb_freq_col="action_frequency_hz",
        bl_freq_col="same_baseline_action_frequency_hz",
        mps_freq_col="mps_action_frequency_hz",
        cb_thr_col="language_throughput_tps",
        bl_thr_col="same_baseline_language_throughput_tps",
        mps_thr_col="mps_language_throughput_tps",
        bl_label="Baseline",
        xlabel="Language Decoding Steps (per Frame)",
        title=f"Ours vs. Baseline (Total Decoding Steps ={mds_str})",
        out_path=plot_dir / "e2e_latency_throughput_vs_steps_per_frame.pdf",
        use_hatch=use_hatch,
    )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Root directory for experiment results",
    )
    parser.add_argument(
        "--analysis-root-dir",
        type=Path,
        default=None,
        help="Root directory for analysis inputs (default: {results-root-dir}/analysis)",
    )
    parser.add_argument(
        "--plot-root-dir",
        type=Path,
        default=None,
        help="Root directory for plot outputs (default: {results-root-dir}/plot)",
    )
    parser.add_argument(
        "--plot-type",
        choices=["all", "heatmap", "line", "line_spf"],
        default="all",
        help="Which figure(s) to generate",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=5,
        help="Filter to this steps_per_frame value (0 to keep all)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="all",
        help="Filter to this policy (pass 'all' to keep all)",
    )
    parser.add_argument(
        "--num-denoise-steps",
        type=int,
        default=10,
        help="Filter to this num_denoise_steps value (0 to keep all)",
    )
    parser.add_argument(
        "--use-hatch",
        action="store_true",
        help="Use hatch patterns for print-friendly plots",
    )
    args = parser.parse_args()

    analysis_root_dir = args.analysis_root_dir or args.results_root_dir / "analysis"
    plot_root_dir = args.plot_root_dir or args.results_root_dir / "plot"
    plot_dir = plot_root_dir / "plot_speedup_cb"
    plot_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    df = load_data(analysis_root_dir)

    # Apply filters
    if args.policy != "all":
        df = df[df["policy"] == args.policy]

    # For "same" heatmap: filter by steps_per_frame
    if args.steps_per_frame:
        df_heatmap_same = df[df["steps_per_frame"] == args.steps_per_frame]
    else:
        df_heatmap_same = df

    # For "trunc" heatmap: don't filter by steps_per_frame (we want all values on x-axis)
    df_heatmap_trunc = df.copy()

    if args.num_denoise_steps:
        df_line = df[df["num_denoise_steps"] == args.num_denoise_steps]
    else:
        df_line = df
    if args.steps_per_frame:
        df_line = df_line[df_line["steps_per_frame"] == args.steps_per_frame]

    if args.plot_type in ("all", "heatmap"):
        plot_heatmap(df_heatmap_same, plot_dir, comparison="same")
        plot_heatmap(df_heatmap_trunc, plot_dir, comparison="trunc")
    if args.plot_type in ("all", "line"):
        plot_line(df_line, plot_dir, use_hatch=args.use_hatch)
    if args.plot_type in ("all", "line_spf"):
        # For line_spf, filter denoise but keep all spf values
        df_spf = df.copy()
        if args.num_denoise_steps:
            df_spf = df_spf[df_spf["num_denoise_steps"] == args.num_denoise_steps]
        plot_line_spf(df_spf, plot_dir, use_hatch=args.use_hatch)

if __name__ == "__main__":
    main()
