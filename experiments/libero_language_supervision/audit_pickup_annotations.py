#!/usr/bin/env python3
"""Audit grasp, pickup, and placement ordering in compiled annotations."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _first_stable(values: Sequence[bool], window: int = 3) -> Optional[int]:
    return next(
        (frame for frame in range(len(values) - window + 1) if all(values[frame : frame + window])),
        None,
    )


def audit(root: Path) -> Dict[str, Any]:
    counts = {
        "demonstrations": 0,
        "placement_goals": 0,
        "spatial_goals": 0,
        "missing_grasp": 0,
        "missing_pickup": 0,
        "invalid_event_order": 0,
        "grasp_language_references": 0,
    }
    grasp_to_pickup: List[int] = []
    issues: List[Dict[str, Any]] = []
    for path in root.rglob("demo_*.jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        counts["demonstrations"] += 1
        auxiliary = {item["id"]: item for item in rows[0]["auxiliary_predicates"]}
        goals = {item["id"]: item for item in rows[0]["goal_predicates"]}
        auxiliary_series = {
            predicate_id: [
                next(item["satisfied"] for item in row["auxiliary_predicates"] if item["id"] == predicate_id)
                for row in rows
            ]
            for predicate_id in auxiliary
        }
        for row in rows:
            language_ids = (
                row["language_components"]["completed_predicate_ids"]
                + row["language_components"]["remaining_predicate_ids"]
            )
            counts["grasp_language_references"] += sum(":grasped(" in value for value in language_ids)

        completion_frames = rows[0]["stable_progress"]["goal_completion_frames"]
        for goal_id, goal in goals.items():
            if len(goal["args"]) < 2 or goal["name"].lower() not in ("on", "in"):
                continue
            object_name, target_name = goal["args"][:2]
            is_spatial = "region" in target_name and any(
                token in target_name for token in ("front", "left", "right", "center")
            )
            if is_spatial:
                counts["spatial_goals"] += 1
                continue
            counts["placement_goals"] += 1
            object_auxiliary = [
                predicate_id
                for predicate_id, predicate in auxiliary.items()
                if predicate["args"] and predicate["args"][0] == object_name
            ]
            grasp_id = next(
                (predicate_id for predicate_id in object_auxiliary if auxiliary[predicate_id]["name"] == "grasped"),
                None,
            )
            pickup_id = next(
                (predicate_id for predicate_id in object_auxiliary if auxiliary[predicate_id]["name"] == "picked_up"),
                None,
            )
            grasp_frame = _first_stable(auxiliary_series[grasp_id]) if grasp_id else None
            pickup_frame = _first_stable(auxiliary_series[pickup_id]) if pickup_id else None
            placement_frame = completion_frames.get(goal_id)
            if grasp_frame is None:
                counts["missing_grasp"] += 1
            if pickup_frame is None:
                counts["missing_pickup"] += 1
            if grasp_frame is not None and pickup_frame is not None:
                grasp_to_pickup.append(pickup_frame - grasp_frame)
            if (
                grasp_frame is None
                or pickup_frame is None
                or placement_frame is None
                or not grasp_frame <= pickup_frame < placement_frame
            ):
                counts["invalid_event_order"] += 1
                issues.append({
                    "file": str(path.relative_to(root)),
                    "object": object_name,
                    "grasp_frame": grasp_frame,
                    "pickup_frame": pickup_frame,
                    "placement_frame": placement_frame,
                })

    sorted_deltas = sorted(grasp_to_pickup)
    report = {
        **counts,
        "grasp_to_pickup_frames": {
            "min": min(sorted_deltas),
            "median": sorted_deltas[len(sorted_deltas) // 2],
            "p95": sorted_deltas[int(0.95 * (len(sorted_deltas) - 1))],
            "max": max(sorted_deltas),
        },
        "issues": issues,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.annotation_root)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    if report["issues"] or report["grasp_language_references"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
