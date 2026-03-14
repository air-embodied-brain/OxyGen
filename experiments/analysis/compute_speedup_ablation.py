"""Compute ablation speedups: Shared KV + CB (ours) vs Shared KV only vs baseline.

Three-way merge on (policy, num_denoise_steps, max_decoding_steps):
  - CB rows: setting=continuous_batching, arrival_pattern contains rate=1, steps_per_frame=5
  - Baseline rows: frame_latency_ms, actions_total_ms, text_total_ms
  - Shared KV rows: frame_latency_ms

Outputs speedup_ablation.csv with shared_kv_speedup, cb_speedup, and action_only_upper_bound.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

MERGE_KEYS = ["policy", "num_denoise_steps", "max_decoding_steps"]


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
        help="Root directory for analysis inputs/outputs (default: {results-root-dir}/analysis)",
    )
    args = parser.parse_args()

    analysis_root_dir = args.analysis_root_dir or args.results_root_dir / "analysis"
    df = pd.read_csv(analysis_root_dir / "aggregate_metrics" / "metrics.csv")

    # --- CB rows: uniform rate=1, steps_per_frame=5 (single representative pattern) ---
    cb = df[
        (df["setting"] == "continuous_batching")
        & (df["runner"] == "grid_search")
        & (df["arrival_pattern"] == "uniform_arrivals(rate=1)")
        & (df["steps_per_frame"] == 5)
    ].copy()
    cb_cols = MERGE_KEYS + ["frame_latency_ms", "action_frequency_hz"]
    cb = cb[cb_cols].rename(columns={"frame_latency_ms": "cb_frame_ms", "action_frequency_hz": "cb_action_freq_hz"})

    # --- Baseline rows ---
    baseline = df[df["setting"] == "baseline"].copy()
    bl_cols = MERGE_KEYS + ["frame_latency_ms", "action_frequency_hz", "actions_total_ms", "text_total_ms"]
    baseline = baseline[bl_cols].rename(columns={
        "frame_latency_ms": "baseline_frame_ms",
        "action_frequency_hz": "baseline_action_freq_hz",
        "actions_total_ms": "baseline_actions_ms",
        "text_total_ms": "baseline_text_ms",
    })

    # --- Shared KV rows ---
    shared_kv = df[df["setting"] == "shared_kv"].copy()
    sk_cols = MERGE_KEYS + ["frame_latency_ms", "action_frequency_hz"]
    shared_kv = shared_kv[sk_cols].rename(
        columns={"frame_latency_ms": "shared_kv_frame_ms", "action_frequency_hz": "shared_kv_action_freq_hz"}
    )

    # --- Parallel MPS rows ---
    parallel_mps = df[df["setting"] == "parallel_mps"].copy()
    mps_cols = MERGE_KEYS + ["frame_latency_ms", "action_frequency_hz"]
    parallel_mps = parallel_mps[mps_cols].rename(
        columns={"frame_latency_ms": "parallel_mps_frame_ms", "action_frequency_hz": "parallel_mps_action_freq_hz"}
    )

    # --- Four-way merge ---
    merged = cb.merge(baseline, on=MERGE_KEYS, how="left")
    merged = merged.merge(shared_kv, on=MERGE_KEYS, how="left")
    merged = merged.merge(parallel_mps, on=MERGE_KEYS, how="left")

    # Warn about unmatched rows
    for col, label in [
        ("baseline_frame_ms", "baseline"),
        ("shared_kv_frame_ms", "shared_kv"),
        ("parallel_mps_frame_ms", "parallel_mps"),
    ]:
        missing = merged[col].isna()
        if missing.any():
            for _, row in merged[missing].iterrows():
                warnings.warn(
                    f"No {label} match for policy={row['policy']} "
                    f"denoise={row['num_denoise_steps']} "
                    f"decode={row['max_decoding_steps']} — skipping"
                )
    merged = merged.dropna(
        subset=["baseline_frame_ms", "shared_kv_frame_ms", "parallel_mps_frame_ms"]
    ).copy()

    # --- Shared KV at decode=5: upper bound for batched inference ---
    sk5 = shared_kv[shared_kv["max_decoding_steps"] == 5].copy()
    sk5 = sk5.rename(columns={
        "shared_kv_frame_ms": "shared_kv_5_frame_ms",
        "shared_kv_action_freq_hz": "shared_kv_5_action_freq_hz",
    })
    sk5 = sk5[["policy", "num_denoise_steps", "shared_kv_5_frame_ms", "shared_kv_5_action_freq_hz"]]

    # --- Compute speedups ---
    merged["parallel_mps_speedup"] = (
        merged["baseline_frame_ms"] / merged["parallel_mps_frame_ms"]
    )
    merged["shared_kv_speedup"] = (
        merged["baseline_frame_ms"] / merged["shared_kv_frame_ms"]
    )
    merged["cb_speedup"] = (
        merged["baseline_frame_ms"] / merged["cb_frame_ms"]
    )
    merged["action_only_upper_bound"] = (
        merged["baseline_frame_ms"]
        / (merged["baseline_frame_ms"] - merged["baseline_text_ms"])
    )
    # Upper bound assuming perfect batching: baseline / shared_kv(decode=5)
    merged = merged.merge(sk5, on=["policy", "num_denoise_steps"], how="left")
    merged["batched_upper_bound"] = (
        merged["baseline_frame_ms"] / merged["shared_kv_5_frame_ms"]
    )

    merged = merged.sort_values(MERGE_KEYS).reset_index(drop=True)

    output_dir = analysis_root_dir / "compute_speedup_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "speedup_ablation.csv"
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
