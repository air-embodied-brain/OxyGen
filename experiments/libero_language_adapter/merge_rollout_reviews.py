#!/usr/bin/env python3
"""Merge per-suite rollout reviews without copying or re-encoding videos."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from experiments.libero_language_adapter import rollout_review


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suites", default=",".join(rollout_review.SUITE_LABELS))
    args = parser.parse_args()

    suites = [value.strip() for value in args.suites.split(",") if value.strip()]
    items: list[dict] = []
    rollouts: list[dict] = []
    events: list[dict] = []
    for suite in suites:
        suite_root = args.root / suite
        suite_items = json.loads((suite_root / "manifest.json").read_text(encoding="utf-8"))
        for item in suite_items:
            merged_item = dict(item)
            merged_item["video"] = f"{suite}/{item['video']}"
            merged_item["poster"] = f"{suite}/{item['poster']}"
            items.append(merged_item)
        rollouts.extend(_jsonl(suite_root / "rollouts.jsonl"))
        events.extend(_jsonl(suite_root / "inference_events.jsonl"))

    batch_histogram = collections.Counter(int(event["policy_timing"]["batch_size"]) for event in events)
    update_histogram = collections.Counter(len(event["language_updates"]) for event in events)
    request_keys = {
        (event["task_suite"], int(event["episode_idx"]), update["request_id"])
        for event in events
        for update in event["language_updates"]
    }
    completed_keys = {
        (event["task_suite"], int(event["episode_idx"]), update["request_id"])
        for event in events
        for update in event["language_updates"]
        if update["is_finished"]
    }
    nonempty_keys = {
        (event["task_suite"], int(event["episode_idx"]), update["request_id"])
        for event in events
        for update in event["language_updates"]
        if update["text"].strip()
    }
    summary = {
        "episodes": len(rollouts),
        "successes": sum(bool(record["success"]) for record in rollouts),
        "exceptions": sum(record["exception"] is not None for record in rollouts),
        "action_replans": len(events),
        "new_request_events": sum(int(event["policy_timing"]["new_requests"]) == 1 for event in events),
        "resumed_request_events": sum(int(event["policy_timing"]["resumed_requests"]) > 0 for event in events),
        "language_requests": len(request_keys),
        "nonempty_language_requests": len(nonempty_keys),
        "completed_language_requests": len(completed_keys),
        "unfinished_at_episode_end": len(request_keys - completed_keys),
        "batch_size_histogram": dict(sorted(batch_histogram.items())),
        "language_updates_per_call_histogram": dict(sorted(update_histogram.items())),
        "by_suite": [
            {
                "suite": suite,
                "episodes": sum(record["task_suite"] == suite for record in rollouts),
                "successes": sum(record["task_suite"] == suite and record["success"] for record in rollouts),
            }
            for suite in suites
        ],
        "request_arrival_protocol": {
            "simulator_control_hz": 20,
            "action_replan_steps": 5,
            "action_replan_and_request_arrival_hz": 4,
            "language_steps_per_frame": 5,
            "max_decoding_steps": 20,
            "request_mode": "new_each_call",
        },
    }
    (args.root / "manifest.json").write_text(json.dumps(items, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.root / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rollout_review._build_html(items, args.root / "index.html")  # noqa: SLF001
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
