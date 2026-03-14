"""Aggregate JSON benchmark results into a single CSV summary."""

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.analysis.plot_utils import ACTION_HORIZON


def parse_result_file(path: Path) -> dict | None:
    with open(path) as f:
        data = json.load(f)

    setting = data["setting"]
    params = data["params"]

    # Policy name: try checkpoint_config first, fall back to policy_config
    policy = params.get("checkpoint_config") or params.get("policy_config", "")
    num_denoise_steps = params["num_denoise_steps"]
    max_decoding_steps = params["max_decoding_steps"]

    # CB-only fields
    steps_per_frame = params.get("steps_per_frame", "")
    arrival_pattern = params.get("arrival_pattern", "")

    # Detect runner based on arrival_pattern format
    # run_workload_sweep includes t_max= in uniform/poisson patterns
    # or uses random_length_arrivals (which is workload_sweep only)
    runner = ""
    if setting == "continuous_batching" and arrival_pattern:
        if "t_max=" in arrival_pattern or "random_length_arrivals" in arrival_pattern:
            runner = "workload_sweep"
        else:
            runner = "grid_search"

    # Filter non-warmup frames
    frames = [f for f in data["frames"] if not f.get("is_warmup", False)]
    if not frames:
        return None

    # frame_latency_ms = mean of frame_ms
    frame_latency_ms = sum(f["frame_ms"] for f in frames) / len(frames)

    # action_frequency_hz = ACTION_HORIZON * num_frames / total_frame_ms * 1000
    total_frame_ms = sum(f["frame_ms"] for f in frames)
    action_frequency_hz = ACTION_HORIZON * len(frames) / total_frame_ms * 1000

    # language_throughput_tps = mean of (total_tokens_this_frame / frame_ms * 1000)
    tps_values = [
        f["total_tokens_this_frame"] / f["frame_ms"] * 1000
        for f in frames
        if f["frame_ms"] > 0
    ]
    language_throughput_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0

    # CB-only: avg_batch_size and avg_request_wall_ms
    avg_batch_size = ""
    avg_request_wall_ms = ""
    if setting == "continuous_batching":
        avg_batch_size = sum(f["n_total"] for f in frames) / len(frames)
        completed = data.get("completed_requests", [])
        if completed:
            avg_request_wall_ms = (
                sum(r["total_wall_ms"] for r in completed) / len(completed)
            )

    # Baseline-only: per-component timing from policy_timing
    actions_total_ms = ""
    text_total_ms = ""
    if setting == "baseline":
        at_vals = [f["policy_timing"]["actions_total_ms"] for f in frames if "policy_timing" in f]
        tt_vals = [f["policy_timing"]["text_total_ms"] for f in frames if "policy_timing" in f]
        if at_vals:
            actions_total_ms = sum(at_vals) / len(at_vals)
        if tt_vals:
            text_total_ms = sum(tt_vals) / len(tt_vals)

    return {
        "setting": setting,
        "runner": runner,
        "policy": policy,
        "num_denoise_steps": num_denoise_steps,
        "max_decoding_steps": max_decoding_steps,
        "steps_per_frame": steps_per_frame,
        "arrival_pattern": arrival_pattern,
        "frame_latency_ms": frame_latency_ms,
        "action_frequency_hz": action_frequency_hz,
        "language_throughput_tps": language_throughput_tps,
        "avg_batch_size": avg_batch_size,
        "avg_request_wall_ms": avg_request_wall_ms,
        "actions_total_ms": actions_total_ms,
        "text_total_ms": text_total_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Root directory containing JSON result files",
    )
    parser.add_argument(
        "--analysis-root-dir",
        type=Path,
        default=None,
        help="Root directory for analysis outputs (default: {results-root-dir}/analysis)",
    )
    args = parser.parse_args()
    if args.analysis_root_dir is None:
        args.analysis_root_dir = args.results_root_dir / "analysis"

    json_files = sorted(args.results_root_dir.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found under {args.results_root_dir}")
        return

    rows = []
    for path in json_files:
        try:
            row = parse_result_file(path)
            if row is not None:
                rows.append(row)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Warning: skipping {path}: {e}")

    df = pd.DataFrame(rows)

    dedup_cols = [
        "setting", "runner", "policy", "num_denoise_steps",
        "max_decoding_steps", "steps_per_frame", "arrival_pattern",
    ]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")

    df = df.sort_values(
        ["setting", "policy", "num_denoise_steps", "max_decoding_steps"]
    ).reset_index(drop=True)

    output = args.analysis_root_dir / "aggregate_metrics" / "metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} rows to {output}")


if __name__ == "__main__":
    main()
