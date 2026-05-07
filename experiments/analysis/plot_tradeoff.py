"""Action frequency vs language throughput: tradeoff figure.

1 row x 2 columns, one subplot per device (RTX 4090, Jetson Orin Thor).
Each subplot has independent x and y ranges.

At fixed N = ref_N:
  - Ours: one curve per k in --k-values, sweeping mds (shades of blue;
    smaller k = darker). Each curve labelled in-plot with its k value.
  - Baseline (sequential): red solid curve sweeping mds.
  - Parallel MPS: orange solid curve sweeping mds.
  - Dashed arrow from the baseline reference point to ours' peak-throughput
    point, annotated with the throughput speedup alongside the arrow.
  - Optional de-emphasized dashed vertical deadline line.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from experiments.analysis.plot_utils import (
    BarStyle,
    PlotColors,
    setup_style,
)

# Smaller k = darker blue (faster path through the transformer).
OURS_K_COLORS = {
    1: "#1F4A8F",
    5: PlotColors.OURS_PRIMARY,
    10: "#8FB3E0",
}
OURS_LEGEND_COLOR = PlotColors.OURS_PRIMARY
MARKER_EDGE = "white"
MARKER_EDGE_WIDTH = 1.2
SPEEDUP_COLOR = "#1F6B2E"   # dark green arrow & text
SPEEDUP_BG = "#C9EACA"      # light green highlight behind the speedup label

# Hard caps on panel y-range so different devices stay readable.
Y_MAX_OVERRIDES = {
    "Jetson AGX Thor": 100.0,
}



def _ceil_to_step(v: float, step: float) -> float:
    return float(np.ceil(v / step) * step)



def _load_metrics(results_root: Path) -> pd.DataFrame:
    return pd.read_csv(
        results_root / "analysis" / "aggregate_metrics" / "metrics.csv"
    )


def _ours_family(df, policy, k_values, mds_values, ref_N):
    base = df[
        (df["setting"] == "continuous_batching")
        & (df["runner"] == "grid_search")
        & (df["arrival_pattern"] == "uniform_arrivals(rate=1)")
        & (df["policy"] == policy)
        & (df["num_denoise_steps"] == ref_N)
    ]
    out = {}
    for k in k_values:
        sub = base[
            (base["steps_per_frame"] == k)
            & (base["max_decoding_steps"].isin(mds_values))
        ].sort_values("max_decoding_steps")
        if sub.empty:
            continue
        out[k] = list(zip(
            sub["max_decoding_steps"].astype(int).tolist(),
            sub["action_frequency_hz"].tolist(),
            sub["language_throughput_tps"].tolist(),
        ))
    return out


def _baseline_curve(df, policy, setting, mds_values, ref_N):
    sub = df[
        (df["setting"] == setting)
        & (df["policy"] == policy)
        & (df["num_denoise_steps"] == ref_N)
        & (df["max_decoding_steps"].isin(mds_values))
    ].sort_values("max_decoding_steps")
    if sub.empty:
        return []
    return list(zip(
        sub["max_decoding_steps"].astype(int).tolist(),
        sub["action_frequency_hz"].tolist(),
        sub["language_throughput_tps"].tolist(),
    ))


def _draw_panel(
    ax, device_df, policy, device_label,
    k_values, mds_values, ref_N,
    deadline_hz, baseline_speedup_ref,
):
    ours = _ours_family(device_df, policy, k_values, mds_values, ref_N)
    bl_seq = _baseline_curve(device_df, policy, "baseline", mds_values, ref_N)
    bl_mps = _baseline_curve(device_df, policy, "parallel_mps", mds_values, ref_N)

    ours_pts = (
        np.array([(p[1], p[2]) for seq in ours.values() for p in seq])
        if ours else np.zeros((0, 2))
    )
    bl_arr_seq = (
        np.array([(p[1], p[2]) for p in bl_seq]) if bl_seq else np.zeros((0, 2))
    )
    bl_arr_mps = (
        np.array([(p[1], p[2]) for p in bl_mps]) if bl_mps else np.zeros((0, 2))
    )
    bl_pts = (
        np.concatenate([a for a in (bl_arr_seq, bl_arr_mps) if len(a)])
        if (len(bl_arr_seq) or len(bl_arr_mps)) else np.zeros((0, 2))
    )

    edge = MARKER_EDGE
    edge_w = MARKER_EDGE_WIDTH

    if bl_seq:
        xs = [p[1] for p in bl_seq]
        ys = [p[2] for p in bl_seq]
        ax.plot(xs, ys, color=PlotColors.BASELINE_PRIMARY, linewidth=1.6,
                linestyle="-", zorder=2.5)
        ax.scatter(xs, ys, s=48, marker="s",
                   color=PlotColors.BASELINE_PRIMARY,
                   edgecolor=edge, linewidth=edge_w, zorder=2.7)

    if bl_mps:
        xs = [p[1] for p in bl_mps]
        ys = [p[2] for p in bl_mps]
        ax.plot(xs, ys, color=PlotColors.MPS_PRIMARY, linewidth=1.6,
                linestyle="-", zorder=2.5)
        ax.scatter(xs, ys, s=56, marker="^", color=PlotColors.MPS_PRIMARY,
                   edgecolor=edge, linewidth=edge_w, zorder=2.7)

    for k_val in sorted(ours.keys()):
        pts = ours[k_val]
        if not pts:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        color = OURS_K_COLORS.get(int(k_val), PlotColors.OURS_PRIMARY)
        ax.plot(xs, ys, color=color, linewidth=1.8, linestyle="-", zorder=3.5)
        ax.scatter(xs, ys, s=40, color=color,
                   edgecolor=edge, linewidth=edge_w, zorder=3.8)

    # Speedup is reported against the sequential baseline only (matches the
    # ablation bar chart's denominator). Parallel MPS still appears on the
    # plot as its own curve, but it's not used as the reference.
    if len(ours_pts) and len(bl_arr_seq):
        peak_idx = int(np.argmax(ours_pts[:, 1]))
        peak_af, peak_th = ours_pts[peak_idx]
        if baseline_speedup_ref == "max_throughput":
            ref_bl_idx = int(np.argmax(bl_arr_seq[:, 1]))
        else:
            ref_bl_idx = int(np.argmax(bl_arr_seq[:, 0]))
        ref_bl_af, ref_bl_th = bl_arr_seq[ref_bl_idx]
        thr_speedup = peak_th / ref_bl_th if ref_bl_th > 0 else float("nan")

        ax.annotate(
            "",
            xy=(peak_af, peak_th),
            xytext=(ref_bl_af, ref_bl_th),
            arrowprops=dict(
                arrowstyle="-|>", edgecolor=SPEEDUP_COLOR,
                facecolor=SPEEDUP_COLOR,
                lw=1.4, linestyle="-",
                shrinkA=4, shrinkB=6,
                mutation_scale=12,
            ),
            zorder=1.2,  # behind data curves
        )
        # Place the speedup label to the upper-left of the arrow's midpoint,
        # highlighted with a light-green rounded background.
        mid_af = (peak_af + ref_bl_af) / 2
        mid_th = (peak_th + ref_bl_th) / 2
        ax.annotate(
            f"{thr_speedup:.1f}×",
            xy=(mid_af, mid_th),
            xytext=(-14, 10), textcoords="offset points",
            fontsize=13, fontweight="bold",
            color=SPEEDUP_COLOR,
            ha="right", va="bottom", zorder=1.3,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=SPEEDUP_BG,
                edgecolor="none",
            ),
        )

    if deadline_hz is not None and deadline_hz > 0:
        ax.axvline(
            x=deadline_hz, color=PlotColors.NEUTRAL_DARK,
            linestyle=(0, (4, 3)), linewidth=0.8, alpha=0.6, zorder=1.5,
        )

    ax.set_title(device_label, fontweight="bold")
    ax.grid(True, alpha=0.5, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)

    xs_all = (
        np.concatenate([a[:, 0] for a in (ours_pts, bl_pts) if len(a)])
        if (len(ours_pts) or len(bl_pts)) else np.array([1.0])
    )
    ys_all = (
        np.concatenate([a[:, 1] for a in (ours_pts, bl_pts) if len(a)])
        if (len(ours_pts) or len(bl_pts)) else np.array([1.0])
    )
    x_max = _ceil_to_step(float(xs_all.max()) * 1.04, 10.0)
    y_max = _ceil_to_step(float(ys_all.max()) * 1.08, 50.0)
    if device_label in Y_MAX_OVERRIDES:
        y_max = Y_MAX_OVERRIDES[device_label]
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)


def _draw_right_legend(leg_ax, entries, deadline_entry=None,
                       bg_facecolor="#EEEEEE", bg_edgecolor="#D0D0D0"):
    """Right-side vertical legend: marker+line on the left, label beside it.
    Drawn on a light-gray rounded-rect background contained inside leg_ax."""
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")

    # Rounded-rect background as a child of leg_ax so it renders under the
    # legend's own line samples / text (which are added below with higher
    # z-order by draw order).
    bg = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        transform=leg_ax.transAxes,
        facecolor=bg_facecolor, edgecolor=bg_edgecolor, linewidth=0.8,
        zorder=0, clip_on=False,
    )
    leg_ax.add_patch(bg)

    items = list(entries)
    if deadline_entry is not None:
        items.append(deadline_entry)
    n = len(items)
    top, bottom = 0.86, 0.14
    centers = (
        np.linspace(top, bottom, n) if n > 1 else np.array([(top + bottom) / 2])
    )

    sample_x0, sample_x1 = 0.06, 0.34
    sample_mid = (sample_x0 + sample_x1) / 2
    text_x = sample_x1 + 0.06

    for cy, cfg in zip(centers, items):
        if cfg.get("kind") == "line":
            leg_ax.plot(
                [sample_x0, sample_x1], [cy, cy],
                color=cfg["color"],
                linestyle=cfg.get("linestyle", "-"),
                linewidth=cfg.get("linewidth", 1.8),
                solid_capstyle="round",
                clip_on=False, zorder=3,
            )
            if cfg.get("marker"):
                leg_ax.plot(
                    [sample_mid], [cy],
                    marker=cfg["marker"],
                    markersize=cfg.get("markersize", 7),
                    markerfacecolor=cfg["color"],
                    markeredgecolor=cfg.get("edgecolor", "white"),
                    markeredgewidth=cfg.get("edgewidth", 1.2),
                    linestyle="None",
                    clip_on=False, zorder=4,
                )
        leg_ax.text(
            text_x, cy, cfg["label"],
            ha="left", va="center",
            fontsize=9,
            fontweight="bold" if cfg.get("bold") else "normal",
            zorder=5,
        )


def plot_tradeoff(
    results_root_4090: Path, results_root_jetson: Path,
    policy: str, k_values, mds_values, ref_N: int,
    deadline_hz_4090, deadline_hz_jetson,
    baseline_speedup_ref: str, out_path: Path,
):
    df_4090 = _load_metrics(results_root_4090)
    df_jet = _load_metrics(results_root_jetson)

    # Explicit axes placement in figure coordinates so we can set the
    # panel↔panel gap and the panel↔legend gap independently.
    fig = plt.figure(figsize=(8.5, 2.4))

    fig_left = 0.09
    fig_right = 0.985
    fig_top = 0.88
    fig_bottom = 0.22
    panel_gap = 0.065       # gap between the two data panels
    legend_gap = 0.015      # gap between Jetson panel and legend strip
    legend_width = 0.17

    width_total = fig_right - fig_left
    panel_width = (width_total - panel_gap - legend_gap - legend_width) / 2
    height = fig_top - fig_bottom

    ax_4090 = fig.add_axes([fig_left, fig_bottom, panel_width, height])
    ax_jet = fig.add_axes([
        fig_left + panel_width + panel_gap, fig_bottom,
        panel_width, height,
    ])
    legend_left = fig_left + 2 * panel_width + panel_gap + legend_gap
    leg_ax = fig.add_axes([legend_left, fig_bottom, legend_width, height])
    axes = [ax_4090, ax_jet]

    _draw_panel(ax_4090, df_4090, policy, "GeForce RTX 4090",
                k_values, mds_values, ref_N,
                deadline_hz_4090, baseline_speedup_ref)
    _draw_panel(ax_jet, df_jet, policy, "Jetson AGX Thor",
                k_values, mds_values, ref_N,
                deadline_hz_jetson, baseline_speedup_ref)

    axes[0].set_ylabel("Language\nThroughput (tok/s)", labelpad=10)
    for ax in axes:
        ax.set_xlabel("Action Frequency (Hz)")

    entries = [
        dict(kind="line", color=PlotColors.BASELINE_PRIMARY,
             marker="s", markersize=7,
             label="Baseline"),
        dict(kind="line", color=PlotColors.MPS_PRIMARY,
             marker="^", markersize=8,
             label="Parallel"),
    ]
    for k_val in sorted(k_values):
        color = OURS_K_COLORS.get(int(k_val), PlotColors.OURS_PRIMARY)
        entries.append(dict(
            kind="line", color=color,
            marker="o", markersize=7,
            label=f"Ours ($k{{=}}{int(k_val)}$)",
        ))
    deadline_entry = None
    if deadline_hz_4090 or deadline_hz_jetson:
        deadline_entry = dict(
            kind="line", color=PlotColors.NEUTRAL_DARK,
            linestyle=(0, (4, 3)), linewidth=0.9, marker=None,
            label="Deadline",
        )

    # Light-gray rounded-rect background is drawn by _draw_right_legend
    # itself (as a child of leg_ax), so the legend samples/text render on
    # top of it.
    _draw_right_legend(leg_ax, entries, deadline_entry)

    # setup_style() sets savefig.bbox="tight", which would crop the side
    # margins we just configured. Locally disable it so the output PDF
    # keeps the full 8.5"x2.8" figure canvas and its breathing room.
    with plt.rc_context({"savefig.bbox": "standard",
                         "savefig.pad_inches": 0.0}):
        fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root-4090", type=Path,
                        default=Path("experiments/results_eccv"))
    parser.add_argument("--results-root-jetson", type=Path,
                        default=Path("experiments/results_eccv_jetson_pytorch"))
    parser.add_argument("--plot-root-dir", type=Path, default=None)
    parser.add_argument("--policy", type=str, default="pi05_o2_aloha")
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[1, 5, 10])
    parser.add_argument("--mds-values", type=int, nargs="+",
                        default=[5, 10, 15, 20, 30])
    parser.add_argument("--ref-n", type=int, default=10)
    parser.add_argument("--deadline-hz-4090", type=float, default=None)
    parser.add_argument("--deadline-hz-jetson", type=float, default=None)
    parser.add_argument("--baseline-speedup-ref",
                        choices=["max_throughput", "max_hz"],
                        default="max_throughput")
    args = parser.parse_args()

    plot_root = args.plot_root_dir or args.results_root_4090 / "plot"
    plot_dir = plot_root / "plot_tradeoff"
    plot_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    plot_tradeoff(
        results_root_4090=args.results_root_4090,
        results_root_jetson=args.results_root_jetson,
        policy=args.policy,
        k_values=args.k_values,
        mds_values=args.mds_values,
        ref_N=args.ref_n,
        deadline_hz_4090=args.deadline_hz_4090,
        deadline_hz_jetson=args.deadline_hz_jetson,
        baseline_speedup_ref=args.baseline_speedup_ref,
        out_path=plot_dir / f"tradeoff_{args.policy}.pdf",
    )


if __name__ == "__main__":
    main()
