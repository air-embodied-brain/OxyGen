#!/usr/bin/env python3
"""Compute warmup-free, phase-aligned robust statistics for long-run traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def trimmed_mean(values: list[float], fraction: float = 0.05) -> float:
    ordered = sorted(values)
    trim = math.floor(len(ordered) * fraction)
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return statistics.fmean(retained)


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "trimmed_mean_5pct": trimmed_mean(values),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def aggregate_rows(rows: list[dict]) -> dict:
    frame_ms = [row["frame_ms"] for row in rows]
    manager_ms = [row["manager_total_ms"] for row in rows]
    manager_fraction = [
        row["manager_total_ms"] / row["frame_ms"] for row in rows if row["frame_ms"] > 0
    ]
    return {
        "frame_ms": stats(frame_ms),
        "manager_ms": stats(manager_ms),
        "manager_fraction": stats(manager_fraction),
        "batch_size": stats([row["batch_size"] for row in rows]),
        "active_after": stats([row["active_after"] for row in rows]),
        "n_new": stats([row["n_new"] for row in rows]),
    }


def complete_periods(rows: list[dict], period: int) -> list[list[dict]]:
    by_frame = {row["frame"]: row for row in rows}
    first_period = math.ceil(min(by_frame) / period)
    last_period = math.floor((max(by_frame) + 1) / period) - 1
    result = []
    for period_index in range(first_period, last_period + 1):
        period_rows = [by_frame.get(frame) for frame in range(period_index * period, (period_index + 1) * period)]
        if all(row is not None for row in period_rows):
            result.append(period_rows)
    return result


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2
    y_mean = statistics.fmean(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return numerator / denominator


def period_analysis(rows: list[dict], period: int, edge_periods: int) -> dict:
    periods = complete_periods(rows, period)
    per_period = []
    for period_rows in periods:
        frame_values = [row["frame_ms"] for row in period_rows]
        per_period.append(
            {
                "start_frame": period_rows[0]["frame"],
                "end_frame": period_rows[-1]["frame"],
                "frame_ms_median": statistics.median(frame_values),
                "frame_ms_trimmed_mean_5pct": trimmed_mean(frame_values),
                "frame_ms_p95": percentile(frame_values, 95),
                "frame_ms_p99": percentile(frame_values, 99),
                "batch_mean": statistics.fmean(row["batch_size"] for row in period_rows),
                "manager_ms_trimmed_mean_5pct": trimmed_mean(
                    [row["manager_total_ms"] for row in period_rows]
                ),
            }
        )
    edge_periods = min(edge_periods, len(periods) // 2)
    first_rows = [row for block in periods[:edge_periods] for row in block]
    last_rows = [row for block in periods[-edge_periods:] for row in block]
    complete_rows = [row for block in periods for row in block]
    metric_names = ("frame_ms_median", "frame_ms_trimmed_mean_5pct", "frame_ms_p95")
    return {
        "period_frames": period,
        "complete_period_count": len(periods),
        "complete_frame_range": [complete_rows[0]["frame"], complete_rows[-1]["frame"]],
        "excluded_measured_prefix_frames": periods[0][0]["frame"] - rows[0]["frame"],
        "excluded_measured_suffix_frames": rows[-1]["frame"] - periods[-1][-1]["frame"],
        "complete_periods": aggregate_rows(complete_rows),
        "edge_period_count": edge_periods,
        "first_edge_periods": aggregate_rows(first_rows),
        "last_edge_periods": aggregate_rows(last_rows),
        "per_period_slope_ms": {
            name: linear_slope([block[name] for block in per_period]) for name in metric_names
        },
        "per_period": per_period,
    }


def memory_analysis(rows: list[dict], backend: str, period: int, edge_periods: int) -> dict:
    samples = [row for row in rows if row.get("memory")]
    if backend == "pytorch":
        keys = (
            "allocated_bytes",
            "reserved_bytes",
            "inactive_split_bytes",
            "num_alloc_retries",
            "num_ooms",
        )
    else:
        keys = ("bytes_in_use", "pool_bytes", "bytes_reserved", "num_allocs")
    periods = complete_periods(rows, period)
    edge_periods = min(edge_periods, len(periods) // 2)
    first_start, first_end = periods[0][0]["frame"], periods[edge_periods - 1][-1]["frame"]
    last_start, last_end = periods[-edge_periods][0]["frame"], periods[-1][-1]["frame"]
    first_samples = [row for row in samples if first_start <= row["frame"] <= first_end]
    last_samples = [row for row in samples if last_start <= row["frame"] <= last_end]
    varying_key = "allocated_bytes" if backend == "pytorch" else "bytes_in_use"
    return {
        "sample_count": len(samples),
        **{
            key: {
                "first": samples[0]["memory"][key],
                "last": samples[-1]["memory"][key],
                "min": min(row["memory"][key] for row in samples),
                "max": max(row["memory"][key] for row in samples),
            }
            for key in keys
        },
        "phase_aligned_edge_periods": edge_periods,
        "phase_aligned_first_samples": stats([row["memory"][varying_key] for row in first_samples]),
        "phase_aligned_last_samples": stats([row["memory"][varying_key] for row in last_samples]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("--backend", choices=("jax", "pytorch"), required=True)
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--edge-periods", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_rows = [json.loads(line) for line in args.frames.read_text().splitlines()]
    rows = [row for row in all_rows if not row.get("is_warmup", False) and "frame_ms" in row]
    payload = {
        "source": str(args.frames),
        "backend": args.backend,
        "warmup_excluded_by_is_warmup": len(all_rows) - len(rows),
        "measured": aggregate_rows(rows),
        "phase_aligned": period_analysis(rows, args.period, args.edge_periods),
        "memory": memory_analysis(rows, args.backend, args.period, args.edge_periods),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
