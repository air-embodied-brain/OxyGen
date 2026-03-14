"""Compare continuous batching to baseline using two matching strategies.

Method 1 (same-config): match CB to baseline by (policy, num_denoise_steps, max_decoding_steps).
Method 2 (truncated):   match CB to baseline where baseline.max_decoding_steps == cb.steps_per_frame.
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

MERGE_KEYS = ["policy", "num_denoise_steps", "max_decoding_steps"]


def _compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    df["action_frequency_speedup"] = (
        df["action_frequency_hz"] / df["baseline_action_frequency_hz"]
    )
    df["language_throughput_speedup"] = (
        df["language_throughput_tps"] / df["baseline_language_throughput_tps"]
    )
    return df


def _warn_unmatched(merged: pd.DataFrame, method: str) -> pd.DataFrame:
    unmatched = merged["baseline_frame_latency_ms"].isna()
    if unmatched.any():
        for _, row in merged[unmatched].iterrows():
            warnings.warn(
                f"[{method}] No baseline match for "
                f"policy={row['policy']} denoise={row['num_denoise_steps']} "
                f"decode={row['max_decoding_steps']} "
                f"steps_per_frame={row.get('steps_per_frame', '')} — skipping"
            )
    return merged[~unmatched].copy()


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

    cb = df[
        (df["setting"] == "continuous_batching")
        & (df["runner"] == "grid_search")
        & (df["arrival_pattern"] == "uniform_arrivals(rate=1)")
    ].copy()

    baseline = df[df["setting"] == "baseline"].copy()

    output_dir = analysis_root_dir / "compute_speedup_cb"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Method 1: same-config ---
    bl1 = baseline[MERGE_KEYS + ["frame_latency_ms", "action_frequency_hz", "language_throughput_tps"]].rename(
        columns={
            "frame_latency_ms": "baseline_frame_latency_ms",
            "action_frequency_hz": "baseline_action_frequency_hz",
            "language_throughput_tps": "baseline_language_throughput_tps",
        }
    )
    m1 = cb.merge(bl1, on=MERGE_KEYS, how="left")
    m1 = _warn_unmatched(m1, "same-config")
    m1 = _compute_speedup(m1)

    out1_cols = [
        "policy", "num_denoise_steps", "max_decoding_steps", "steps_per_frame",
        "frame_latency_ms", "action_frequency_hz", "language_throughput_tps",
        "baseline_frame_latency_ms", "baseline_action_frequency_hz", "baseline_language_throughput_tps",
        "action_frequency_speedup", "language_throughput_speedup",
    ]
    m1 = m1[out1_cols].sort_values(
        ["policy", "num_denoise_steps", "max_decoding_steps", "steps_per_frame"]
    ).reset_index(drop=True)

    p1 = output_dir / "speedup_cb_same_config.csv"
    m1.to_csv(p1, index=False)
    print(f"Wrote {len(m1)} rows to {p1}")

    # --- Method 2: truncated-baseline ---
    cb2 = cb.copy()
    cb2["_bl_decode"] = cb2["steps_per_frame"].apply(
        lambda x: int(x) if pd.notna(x) and float(x) == int(float(x)) else None
    )
    cb2 = cb2.dropna(subset=["_bl_decode"])
    cb2["_bl_decode"] = cb2["_bl_decode"].astype(int)

    bl2 = baseline[
        ["policy", "num_denoise_steps", "max_decoding_steps",
         "frame_latency_ms", "action_frequency_hz", "language_throughput_tps"]
    ].rename(columns={
        "max_decoding_steps": "_bl_decode",
        "frame_latency_ms": "baseline_frame_latency_ms",
        "action_frequency_hz": "baseline_action_frequency_hz",
        "language_throughput_tps": "baseline_language_throughput_tps",
    })

    m2 = cb2.merge(bl2, on=["policy", "num_denoise_steps", "_bl_decode"], how="left")
    m2["baseline_max_decoding_steps"] = m2["_bl_decode"]
    m2 = _warn_unmatched(m2, "truncated")
    m2 = _compute_speedup(m2)

    out2_cols = [
        "policy", "num_denoise_steps", "max_decoding_steps", "steps_per_frame",
        "frame_latency_ms", "action_frequency_hz", "language_throughput_tps",
        "baseline_max_decoding_steps",
        "baseline_frame_latency_ms", "baseline_action_frequency_hz", "baseline_language_throughput_tps",
        "action_frequency_speedup", "language_throughput_speedup",
    ]
    m2 = m2[out2_cols].sort_values(
        ["policy", "num_denoise_steps", "max_decoding_steps", "steps_per_frame"]
    ).reset_index(drop=True)

    p2 = output_dir / "speedup_cb_truncated.csv"
    m2.to_csv(p2, index=False)
    print(f"Wrote {len(m2)} rows to {p2}")

    # --- Unified: both methods side by side ---
    join_keys = ["policy", "num_denoise_steps", "max_decoding_steps", "steps_per_frame"]
    u1 = m1[join_keys + [
        "frame_latency_ms", "action_frequency_hz", "language_throughput_tps",
        "baseline_frame_latency_ms", "baseline_action_frequency_hz", "baseline_language_throughput_tps",
        "action_frequency_speedup", "language_throughput_speedup",
    ]].rename(columns={
        "baseline_frame_latency_ms": "same_baseline_frame_latency_ms",
        "baseline_action_frequency_hz": "same_baseline_action_frequency_hz",
        "baseline_language_throughput_tps": "same_baseline_language_throughput_tps",
        "action_frequency_speedup": "same_action_frequency_speedup",
        "language_throughput_speedup": "same_language_throughput_speedup",
    })
    u2 = m2[join_keys + [
        "baseline_max_decoding_steps",
        "baseline_frame_latency_ms", "baseline_action_frequency_hz", "baseline_language_throughput_tps",
        "action_frequency_speedup", "language_throughput_speedup",
    ]].rename(columns={
        "baseline_frame_latency_ms": "trunc_baseline_frame_latency_ms",
        "baseline_action_frequency_hz": "trunc_baseline_action_frequency_hz",
        "baseline_language_throughput_tps": "trunc_baseline_language_throughput_tps",
        "action_frequency_speedup": "trunc_action_frequency_speedup",
        "language_throughput_speedup": "trunc_language_throughput_speedup",
    })
    unified = u1.merge(u2, on=join_keys, how="outer").sort_values(join_keys).reset_index(drop=True)

    unified_cols = [
        # identifiers
        "policy", "num_denoise_steps", "max_decoding_steps", "steps_per_frame",
        # CB metrics
        "frame_latency_ms", "action_frequency_hz", "language_throughput_tps",
        # same-config baseline
        "same_baseline_frame_latency_ms", "same_baseline_action_frequency_hz", "same_baseline_language_throughput_tps",
        # truncated baseline
        "baseline_max_decoding_steps",
        "trunc_baseline_frame_latency_ms", "trunc_baseline_action_frequency_hz", "trunc_baseline_language_throughput_tps",
        # all speedups grouped together
        "same_action_frequency_speedup", "same_language_throughput_speedup",
        "trunc_action_frequency_speedup", "trunc_language_throughput_speedup",
    ]
    unified = unified[unified_cols]

    p3 = output_dir / "speedup_cb_unified.csv"
    unified.to_csv(p3, index=False)
    print(f"Wrote {len(unified)} rows to {p3}")


if __name__ == "__main__":
    main()
