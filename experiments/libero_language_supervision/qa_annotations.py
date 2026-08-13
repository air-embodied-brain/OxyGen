#!/usr/bin/env python3
"""Audit completeness and temporal consistency of generated annotations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict

import h5py


def audit_file(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    issues = []
    if not rows:
        return {"path": str(path), "frames": 0, "issues": ["empty file"]}
    if [row["frame"] for row in rows] != list(range(len(rows))):
        issues.append("non-contiguous frame indices")
    if not any(row["success"] for row in rows):
        issues.append("simulator success is never reached")

    predicate_ids = {item["id"] for item in rows[0]["goal_predicates"] + rows[0]["auxiliary_predicates"]}
    previous_completed = set()
    goal_ids = {item["id"] for item in rows[0]["goal_predicates"]}
    stable_progress = rows[0]["stable_progress"]
    milestone_ids = goal_ids | {
        predicate_id
        for predicate_id, frame in stable_progress["auxiliary_completion_frames"].items()
        if frame is not None
    }
    for row in rows:
        language = row.get("language", {})
        if any(
            not str(language.get(key, "")).strip()
            for key in ("completed", "remaining", "next", "visual_memory")
        ):
            issues.append(f"frame {row['frame']}: missing language component")
            break
        components = row.get("language_components", {})
        completed = set(components.get("completed_predicate_ids", []))
        remaining = set(components.get("remaining_predicate_ids", []))
        if completed & remaining or completed | remaining != milestone_ids:
            issues.append(f"frame {row['frame']}: completed/remaining is not a milestone partition")
            break
        if not previous_completed <= completed:
            issues.append(f"frame {row['frame']}: stable completion regresses")
            break
        previous_completed = completed
        visual_memory = set(components.get("visual_memory_predicate_ids", []))
        current_goal_values = {item["id"]: bool(item["satisfied"]) for item in row["goal_predicates"]}
        if not visual_memory <= goal_ids:
            issues.append(f"frame {row['frame']}: visual memory contains a non-goal predicate")
            break
        if any(not current_goal_values[predicate_id] for predicate_id in visual_memory):
            issues.append(f"frame {row['frame']}: visual memory contains a currently false predicate")
            break
        target_id = components.get("next_target_predicate_id")
        target_frame = components.get("next_target_frame")
        if target_id is not None and target_id not in predicate_ids:
            issues.append(f"frame {row['frame']}: unknown next predicate")
            break
        if target_frame is not None and target_frame <= row["frame"]:
            issues.append(f"frame {row['frame']}: next event is not in the future")
            break
    if not goal_ids <= previous_completed:
        issues.append("terminal stable progress is incomplete")
    return {
        "path": str(path),
        "suite": rows[0].get("suite"),
        "task": rows[0].get("task"),
        "demo": rows[0].get("demo"),
        "frames": len(rows),
        "reached_raw_success": any(row["success"] for row in rows),
        "final_raw_success": bool(rows[-1]["success"]),
        "issues": issues,
        "language_values": {
            component: sorted({row["language"][component] for row in rows})
            for component in ("completed", "remaining", "next", "visual_memory")
        },
    }


def audit_coverage(dataset_root: Path, annotation_root: Path) -> Dict[str, Any]:
    expected = {}
    for dataset_path in sorted(dataset_root.glob("*/*.hdf5")):
        suite = dataset_path.parent.name
        stem = dataset_path.stem
        task = stem[:-5] if stem.endswith("_demo") else stem
        with h5py.File(dataset_path, "r") as dataset:
            for demo_name, demo in dataset["data"].items():
                expected[(suite, task, demo_name)] = len(demo["states"])

    actual = {}
    for path in sorted(annotation_root.glob("*/*/demo_*.jsonl")):
        key = (path.parents[1].name, path.parent.name, path.stem)
        with path.open(encoding="utf-8") as stream:
            actual[key] = sum(1 for line in stream if line.strip())
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(
        {
            key: {"expected": expected[key], "actual": actual[key]}
            for key in set(expected) & set(actual)
            if expected[key] != actual[key]
        }.items()
    )
    return {
        "dataset_files": len(list(dataset_root.glob("*/*.hdf5"))),
        "expected_demos": len(expected),
        "actual_demos": len(actual),
        "expected_frames": sum(expected.values()),
        "actual_frames": sum(actual.values()),
        "missing": [list(key) for key in missing],
        "unexpected": [list(key) for key in unexpected],
        "frame_mismatches": [{"key": list(key), **value} for key, value in mismatched],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(args.annotation_root.glob("*/*/demo_*.jsonl"))
    audits = [audit_file(path) for path in files]
    issue_counts = Counter(issue for audit in audits for issue in audit["issues"])
    catalog: Dict[str, Dict[str, set]] = {}
    for item in audits:
        task_key = f"{item.get('suite')}/{item.get('task')}"
        task_catalog = catalog.setdefault(
            task_key,
            {key: set() for key in ("completed", "remaining", "next", "visual_memory")},
        )
        for component, values in item.get("language_values", {}).items():
            task_catalog[component].update(values)
    report = {
        "files": len(audits),
        "tasks": len({(item.get("suite"), item.get("task")) for item in audits}),
        "frames": sum(item["frames"] for item in audits),
        "files_with_issues": sum(bool(item["issues"]) for item in audits),
        "files_reaching_success": sum(item.get("reached_raw_success", False) for item in audits),
        "files_never_reaching_success": sum(not item.get("reached_raw_success", False) for item in audits),
        "files_with_unsuccessful_final_state": sum(not item.get("final_raw_success", False) for item in audits),
        "issue_counts": dict(issue_counts),
        "issues": [item for item in audits if item["issues"]],
        "language_catalog": {
            task: {component: sorted(values) for component, values in components.items()}
            for task, components in sorted(catalog.items())
        },
    }
    if args.dataset_root is not None:
        report["coverage"] = audit_coverage(args.dataset_root, args.annotation_root)
        coverage = report["coverage"]
        if coverage["missing"] or coverage["unexpected"] or coverage["frame_mismatches"]:
            report["files_with_issues"] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["files_with_issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
