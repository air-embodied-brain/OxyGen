"""Extract workload sweep results from metrics.csv into a dedicated CSV.

Parses arrival_pattern strings into category + parameter, producing a tidy
table for plotting.  Also joins baseline frame latency (single-request,
same denoise/decode) to enable computing sequential_baseline_ms =
baseline_frame_ms * avg_batch_size.

Output columns:
    category, param_value, param_label, frame_latency_ms,
    language_throughput_tps, avg_batch_size, avg_request_wall_ms,
    baseline_frame_ms, sequential_baseline_ms
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def _parse_pattern(pattern: str) -> dict:
    """Parse an arrival_pattern string into category, param_value, param_label."""
    pattern = pattern.replace(" ", "")

    # uniform_arrivals(rate=..., t_max=...)
    m = re.match(r"uniform_arrivals\(rate=([0-9.]+)", pattern)
    if m:
        rate = float(m.group(1))
        # Convert rate parameter to arrivals/frame
        # rate>=1: 1 request every `rate` frames → 1/rate arrivals/frame
        # rate<1:  round(1/rate) requests per frame
        if rate < 1:
            arrivals_per_frame = round(1 / rate)
        else:
            arrivals_per_frame = 1 / rate
        # Format: integer if whole, else fraction (just the value, no prefix)
        if arrivals_per_frame == int(arrivals_per_frame):
            label = str(int(arrivals_per_frame))
        else:
            from fractions import Fraction
            frac = Fraction(1, int(rate)).limit_denominator(10)
            label = str(frac)
        return {
            "category": "uniform",
            "param_value": arrivals_per_frame,  # sort by arrivals/frame
            "param_label": label,
        }

    # poisson_arrivals(lam=..., t_max=...)
    m = re.match(r"poisson_arrivals\(lam=([0-9.]+)", pattern)
    if m:
        lam = float(m.group(1))
        return {
            "category": "poisson",
            "param_value": lam,
            "param_label": f"{lam}",
        }

    # random_length_arrivals(t_max_values=[...], ...)
    m = re.match(r"random_length_arrivals\(t_max_values=\[([0-9,]+)\]", pattern)
    if m:
        values = [int(x) for x in m.group(1).split(",")]
        # Check for weights
        wm = re.search(r"weights=\[([0-9.,]+)\]", pattern)
        if wm:
            weights = [float(x) for x in wm.group(1).split(",")]
            if len(values) == 2:
                # Label as long/short ratio (integer parts out of 10)
                long_w = weights[1]
                long_part = int(round(long_w * 10))
                short_part = 10 - long_part
                label = f"{long_part}/{short_part}"
                param_val = long_w
            else:
                label = str(weights)
                param_val = weights[0]
        else:
            label = f"U{values}"
            param_val = len(values)
        return {
            "category": "random_length",
            "param_value": param_val,
            "param_label": label,
        }

    return {"category": "unknown", "param_value": 0, "param_label": pattern}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root-dir", type=Path, default=Path("experiments/results"),
    )
    parser.add_argument(
        "--analysis-root-dir", type=Path, default=None,
    )
    parser.add_argument(
        "--num-denoise-steps", type=int, default=10,
    )
    parser.add_argument(
        "--max-decoding-steps", type=int, default=20,
    )
    parser.add_argument(
        "--policy", type=str, default="pi05_o2_libero",
        help="Filter to this policy (default: pi05_o2_libero).",
    )
    args = parser.parse_args()

    analysis_root = args.analysis_root_dir or args.results_root_dir / "analysis"
    df = pd.read_csv(analysis_root / "aggregate_metrics" / "metrics.csv")

    # Filter to CB rows matching the workload sweep's fixed parameters
    cb = df[
        (df["setting"] == "continuous_batching")
        & (df["runner"] == "workload_sweep")
        & (df["steps_per_frame"] == 5)
        & (df["num_denoise_steps"] == args.num_denoise_steps)
        & (df["max_decoding_steps"] == args.max_decoding_steps)
    ].copy()

    if args.policy and args.policy != "all":
        cb = cb[cb["policy"] == args.policy]

    if cb.empty:
        print("No matching continuous_batching rows found.")
        return

    # Get baseline frame latency for the same denoise/decode (batch=1 reference)
    bl = df[
        (df["setting"] == "baseline")
        & (df["num_denoise_steps"] == args.num_denoise_steps)
        & (df["max_decoding_steps"] == args.max_decoding_steps)
    ].copy()

    # Parse each arrival pattern
    parsed = cb["arrival_pattern"].apply(_parse_pattern).apply(pd.Series)
    cb = pd.concat([cb, parsed], axis=1)

    # Join baseline: merge on policy
    if not bl.empty:
        bl_ref = bl[["policy", "frame_latency_ms", "action_frequency_hz"]].rename(
            columns={
                "frame_latency_ms": "baseline_frame_ms",
                "action_frequency_hz": "baseline_action_freq_hz",
            }
        )
        cb = cb.merge(bl_ref, on="policy", how="left")
        # Sequential baseline = baseline per-request latency × concurrent batch size
        cb["sequential_baseline_ms"] = cb["baseline_frame_ms"] * cb["avg_batch_size"]
        # Sequential baseline action freq = baseline freq / avg_batch_size
        cb["sequential_baseline_action_freq_hz"] = cb["baseline_action_freq_hz"] / cb["avg_batch_size"]
    else:
        print("WARNING: no baseline data found for "
              f"denoise={args.num_denoise_steps}, decode={args.max_decoding_steps}")
        cb["baseline_frame_ms"] = float("nan")
        cb["baseline_action_freq_hz"] = float("nan")
        cb["sequential_baseline_ms"] = float("nan")
        cb["sequential_baseline_action_freq_hz"] = float("nan")

    # Select output columns
    out = cb[[
        "category", "param_value", "param_label",
        "frame_latency_ms", "action_frequency_hz", "language_throughput_tps",
        "avg_batch_size", "avg_request_wall_ms",
        "baseline_frame_ms", "baseline_action_freq_hz",
        "sequential_baseline_ms", "sequential_baseline_action_freq_hz",
    ]].copy()

    out = out.sort_values(["category", "param_value"]).reset_index(drop=True)

    # Deduplicate: if old grid-search pattern (e.g. "uniform_arrivals(rate=1)")
    # and new sweep pattern (e.g. "uniform_arrivals(rate=1, t_max=20)") both
    # match the same (category, param_label), keep the last (newer) one.
    out = out.drop_duplicates(
        subset=["category", "param_label"], keep="last"
    ).reset_index(drop=True)

    output_dir = analysis_root / "compute_workload_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "workload_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
