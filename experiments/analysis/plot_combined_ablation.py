"""Combined ablation bar chart: RTX 4090 + Jetson AGX Thor side-by-side.

Produces a 1x2 figure (one panel per device) with shared y-axis range and
title, but independent x-axis ranges (hardcoded: 80 for 4090, 40 for Jetson).
Legend is placed as a single horizontal row at the bottom.

Each panel keeps the horizontal overlapping-bar style of
plot_speedup_ablation.py.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.analysis.plot_utils import (
    BarStyle,
    PlotColors,
    setup_style,
)

POLICY_ORDER = ["pi05_o2_aloha", "pi05_o2_droid", "pi05_o2_libero"]
POLICY_LABELS = {
    "pi05_o2_aloha": "ALOHA",
    "pi05_o2_droid": "DROID",
    "pi05_o2_libero": "LIBERO",
}


def _draw_panel(ax, psub: pd.DataFrame, x_max: float, use_hatch: bool,
                show_yticklabels: bool, device_label: str,
                parallel_on_top: bool = False):
    """Draw one device's horizontal overlapping bar chart on ax.

    When ``parallel_on_top`` is True, the Parallel bar is drawn in front of
    the Baseline bar (used when Parallel is slower than Baseline, as on
    Jetson AGX Thor).
    """
    y_vals = psub["max_decoding_steps"].values
    y_pos = np.arange(len(y_vals))
    bar_height = 0.6

    # Overlapping bars from tallest (Ours, behind) to shortest (front).
    ax.barh(y_pos, psub["cb_action_freq_hz"], bar_height,
            color=PlotColors.OURS_PRIMARY, edgecolor=BarStyle.EDGE_COLOR,
            linewidth=BarStyle.EDGE_WIDTH,
            hatch=BarStyle.HATCH_OURS if use_hatch else None,
            label="Ours", zorder=1)
    ax.barh(y_pos, psub["shared_kv_action_freq_hz"], bar_height,
            color=PlotColors.VARIANT_GOLD, edgecolor=BarStyle.EDGE_COLOR,
            linewidth=BarStyle.EDGE_WIDTH,
            hatch=BarStyle.HATCH_VARIANT if use_hatch else None,
            label="Ours w/o Batching", zorder=2)

    parallel_z = 4 if parallel_on_top else 3
    baseline_z = 3 if parallel_on_top else 4
    ax.barh(y_pos, psub["parallel_mps_action_freq_hz"], bar_height,
            color=PlotColors.MPS_PRIMARY, edgecolor=BarStyle.EDGE_COLOR,
            linewidth=BarStyle.EDGE_WIDTH,
            hatch=BarStyle.HATCH_MPS if use_hatch else None,
            label="Parallel", zorder=parallel_z)
    ax.barh(y_pos, psub["baseline_action_freq_hz"], bar_height,
            color=PlotColors.BASELINE_PRIMARY, edgecolor=BarStyle.EDGE_COLOR,
            linewidth=BarStyle.EDGE_WIDTH,
            hatch=BarStyle.HATCH_BASELINE if use_hatch else None,
            label="Baseline", zorder=baseline_z)

    y_min = -0.5
    y_max_pos = len(y_vals) - 0.5

    y_extended = np.linspace(y_min, y_max_pos, 200)
    upper_bound_interp = np.interp(
        y_extended, y_pos, psub["shared_kv_5_action_freq_hz"],
    )
    ax.plot(upper_bound_interp, y_extended,
            color=PlotColors.NEUTRAL_DARK, linestyle=(0, (3, 1, 1, 1)),
            linewidth=1.8, label="Batching Upper Bound",
            zorder=5, clip_on=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_vals)
    if not show_yticklabels:
        # Hide tick labels for the right panel without clearing them on the
        # shared y-axis (which would also erase the left panel's labels).
        plt.setp(ax.get_yticklabels(), visible=False)

    ax.set_title(device_label, fontweight="bold", loc="center")

    n_ticks = 5
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=y_min, top=y_max_pos)
    ax.set_xticks(np.linspace(0, x_max, n_ticks))

    ax.grid(axis='x', alpha=0.8, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)


def _filter(df: pd.DataFrame, policy: str,
            num_denoise_steps: int, mds_values: list[int]) -> pd.DataFrame:
    sub = df[df["policy"] == policy]
    sub = sub[sub["num_denoise_steps"] == num_denoise_steps]
    sub = sub[sub["max_decoding_steps"].isin(mds_values)]
    return sub.sort_values("max_decoding_steps").reset_index(drop=True)


def plot_combined(
    df_4090: pd.DataFrame, df_jet: pd.DataFrame, policy: str,
    num_denoise_steps: int, mds_values: list[int],
    x_max_4090: float, x_max_jetson: float,
    plot_dir: Path, use_hatch: bool,
):
    sub_4090 = _filter(df_4090, policy, num_denoise_steps, mds_values)
    sub_jet = _filter(df_jet, policy, num_denoise_steps, mds_values)

    if sub_4090.empty or sub_jet.empty:
        print(
            f"Missing data: 4090 rows={len(sub_4090)}, jetson rows={len(sub_jet)}"
        )
        return

    sub_4090.to_csv(plot_dir / f"ablation_speedup_4090_{policy}.csv",
                     index=False)
    sub_jet.to_csv(plot_dir / f"ablation_speedup_jetson_{policy}.csv",
                    index=False)

    fig, (ax_4090, ax_jet) = plt.subplots(
        1, 2, figsize=(8.5, 2.4), sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    _draw_panel(ax_4090, sub_4090, x_max=x_max_4090, use_hatch=use_hatch,
                show_yticklabels=True, device_label="GeForce RTX 4090",
                parallel_on_top=False)
    _draw_panel(ax_jet, sub_jet, x_max=x_max_jetson, use_hatch=use_hatch,
                show_yticklabels=False, device_label="Jetson AGX Thor",
                parallel_on_top=True)

    ax_4090.set_ylabel("Lang. Decoding\nSteps (Total)")
    for ax in (ax_4090, ax_jet):
        ax.set_xlabel("Action Frequency (Hz)")

    handles, labels = ax_4090.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = [
        "Ours", "Ours w/o Batching", "Parallel", "Baseline",
        "Batching Upper Bound",
    ]
    ordered_handles = [label_to_handle[l] for l in ordered_labels]

    # Reserve a horizontal strip at the bottom for the shared legend row and
    # place the legend inside that strip. Using subplots_adjust (instead of
    # tight_layout) keeps the figure-level legend from confusing the layout
    # engine.
    fig.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.32,
                        wspace=0.08)
    fig.legend(
        ordered_handles, ordered_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=len(ordered_labels), frameon=False, fontsize=10,
        columnspacing=1.8, handlelength=1.8, handletextpad=0.6,
    )

    out = plot_dir / f"combined_ablation_{policy}.pdf"
    with plt.rc_context({"savefig.bbox": "standard",
                         "savefig.pad_inches": 0.02}):
        fig.savefig(out)
    fig.savefig(plot_dir / f"combined_ablation_{policy}.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root-4090", type=Path,
        default=Path("experiments/results_eccv"),
    )
    parser.add_argument(
        "--results-root-jetson", type=Path,
        default=Path("experiments/results_eccv_jetson_pytorch"),
    )
    parser.add_argument("--plot-root-dir", type=Path, default=None)
    parser.add_argument("--policy", type=str, default="pi05_o2_libero")
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument(
        "--max-decoding-steps", type=int, nargs="+",
        default=[5, 10, 15, 20, 30],
    )
    parser.add_argument("--x-max-4090", type=float, default=80.0)
    parser.add_argument("--x-max-jetson", type=float, default=40.0)
    parser.add_argument("--use-hatch", action="store_true")
    args = parser.parse_args()

    plot_root = args.plot_root_dir or args.results_root_4090 / "plot"
    plot_dir = plot_root / "plot_combined_ablation"
    plot_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    df_4090 = pd.read_csv(
        args.results_root_4090 / "analysis" / "compute_speedup_ablation"
        / "speedup_ablation.csv"
    )
    df_jet = pd.read_csv(
        args.results_root_jetson / "analysis" / "compute_speedup_ablation"
        / "speedup_ablation.csv"
    )

    plot_combined(
        df_4090=df_4090, df_jet=df_jet, policy=args.policy,
        num_denoise_steps=args.num_denoise_steps,
        mds_values=args.max_decoding_steps,
        x_max_4090=args.x_max_4090, x_max_jetson=args.x_max_jetson,
        plot_dir=plot_dir, use_hatch=args.use_hatch,
    )


if __name__ == "__main__":
    main()
