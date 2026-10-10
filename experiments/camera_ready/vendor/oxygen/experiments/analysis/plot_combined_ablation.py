"""Combined ablation bar chart across available devices.

Produces one panel per device with available data. If one result set is
missing, the script still writes a single-panel figure instead of failing.
Panels share the y-axis range and use independent x-axis ranges. Legend is
placed as a single horizontal row at the bottom.

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
    device_label_from_results_dir,
    setup_style,
)

POLICY_ORDER = ["pi05_o2_aloha", "pi05_o2_droid", "pi05_o2_libero"]
POLICY_LABELS = {
    "pi05_o2_aloha": "ALOHA",
    "pi05_o2_droid": "DROID",
    "pi05_o2_libero": "LIBERO",
}


def _read_speedup_csv(results_root: Path, device_name: str) -> pd.DataFrame:
    path = (
        results_root / "analysis" / "compute_speedup_ablation"
        / "speedup_ablation.csv"
    )
    if not path.exists():
        print(f"Skipping {device_name}: missing {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _draw_panel(ax, psub: pd.DataFrame, x_max: float, use_hatch: bool,
                show_yticklabels: bool, device_label: str,
                parallel_on_top: bool = False):
    """Draw one device's horizontal overlapping bar chart on ax.

    When ``parallel_on_top`` is True, the Parallel bar is drawn in front of
    the Baseline bar. This is useful when Parallel is slower than Baseline
    and would otherwise be hidden.
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
    required = {"policy", "num_denoise_steps", "max_decoding_steps"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    sub = df[df["policy"] == policy]
    sub = sub[sub["num_denoise_steps"] == num_denoise_steps]
    sub = sub[sub["max_decoding_steps"].isin(mds_values)]
    return sub.sort_values("max_decoding_steps").reset_index(drop=True)


def _plot_combined_from_specs(
    result_specs: list[dict],
    policy: str,
    num_denoise_steps: int,
    mds_values: list[int],
    plot_dir: Path,
    use_hatch: bool,
):
    panels = []
    for spec in result_specs:
        df = _read_speedup_csv(spec["root"], spec["label"])
        sub = _filter(df, policy, num_denoise_steps, mds_values)
        if sub.empty:
            print(f"Skipping {spec['label']} panel: no rows for policy={policy}")
            continue
        slug = Path(spec["root"]).name.replace("results_", "")
        sub.to_csv(plot_dir / f"ablation_speedup_{slug}_{policy}.csv",
                   index=False)
        panels.append({
            "data": sub,
            "x_max": spec["x_max"],
            "label": spec["label"],
            "parallel_on_top": spec["parallel_on_top"],
        })

    if not panels:
        print(
            f"No ablation data to plot for policy={policy}, "
            f"num_denoise_steps={num_denoise_steps}"
        )
        return

    _draw_combined_panels(panels, policy, plot_dir, use_hatch)


def _draw_combined_panels(
    panels: list[dict],
    policy: str,
    plot_dir: Path,
    use_hatch: bool,
):
    ncols = len(panels)
    fig_width = 4.8 if ncols == 1 else max(8.5, 3.9 * ncols + 1.0)

    fig, axes = plt.subplots(
        1, ncols, figsize=(fig_width, 2.4), sharey=True, squeeze=False,
        gridspec_kw={"wspace": 0.08},
    )
    axes = axes[0]

    for idx, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
        _draw_panel(ax, panel["data"], x_max=panel["x_max"],
                    use_hatch=use_hatch, show_yticklabels=(idx == 0),
                    device_label=panel["label"],
                    parallel_on_top=panel["parallel_on_top"])

    axes[0].set_ylabel("Lang. Decoding\nSteps (Total)")
    for ax in axes:
        ax.set_xlabel("Action Frequency (Hz)")

    handles, labels = axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles, strict=True))
    ordered_labels = [
        "Ours", "Ours w/o Batching", "Parallel", "Baseline",
        "Batching Upper Bound",
    ]
    ordered_handles = [label_to_handle[label] for label in ordered_labels
                       if label in label_to_handle]
    ordered_labels = [label for label in ordered_labels if label in label_to_handle]

    left = 0.16 if ncols == 1 else 0.10
    fig.subplots_adjust(left=left, right=0.985, top=0.86, bottom=0.32,
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
        "--results-root", type=Path, nargs="+",
        default=[
            Path("experiments/results_4090_jax"),
            Path("experiments/results_thor_pytorch"),
        ],
        help="One or more result directories to plot as device panels.",
    )
    parser.add_argument(
        "--device-label", type=str, nargs="+", default=None,
        help="Optional labels matching --results-root order.",
    )
    parser.add_argument("--plot-root-dir", type=Path, default=None)
    parser.add_argument("--policy", type=str, default="pi05_o2_libero")
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument(
        "--max-decoding-steps", type=int, nargs="+",
        default=[5, 10, 15, 20, 30],
    )
    parser.add_argument(
        "--x-max", type=float, nargs="+", default=None,
        help="Optional x-axis maxima matching --results-root order.",
    )
    parser.add_argument(
        "--parallel-on-top", action="store_true",
        help="Draw the Parallel bars in front of Baseline bars for all panels.",
    )
    parser.add_argument("--use-hatch", action="store_true")
    args = parser.parse_args()

    plot_root = args.plot_root_dir or args.results_root[0] / "plot"
    plot_dir = plot_root / "plot_combined_ablation"
    plot_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    labels = args.device_label or [
        device_label_from_results_dir(path) for path in args.results_root
    ]
    if len(labels) != len(args.results_root):
        raise ValueError("--device-label must match --results-root length")
    x_max_values = args.x_max or [None] * len(args.results_root)
    if len(x_max_values) != len(args.results_root):
        raise ValueError("--x-max must match --results-root length")
    result_specs = []
    for root, label, x_max in zip(
        args.results_root, labels, x_max_values, strict=True,
    ):
        result_specs.append({
            "root": root,
            "label": label,
            "x_max": x_max or 80.0,
            "parallel_on_top": args.parallel_on_top,
        })
    _plot_combined_from_specs(
        result_specs=result_specs,
        policy=args.policy,
        num_denoise_steps=args.num_denoise_steps,
        mds_values=args.max_decoding_steps,
        plot_dir=plot_dir,
        use_hatch=args.use_hatch,
    )


if __name__ == "__main__":
    main()
