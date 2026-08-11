#!/usr/bin/env python3
"""Render deterministic natural-language labels from predicate JSONL."""

# This module also supports LIBERO's Python 3.8 environment.
# ruff: noqa: UP006, UP035

import argparse
import json
from pathlib import Path
import re
from typing import Dict, List, Sequence

OBJECT_ALIASES = {
    "akita_black_bowl": "black bowl",
}


def humanize(name: str) -> str:
    name = re.sub(r"_\d+$", "", name)
    name = OBJECT_ALIASES.get(name, name)
    name = name.replace("_region", "").replace("_", " ")
    return "the " + name


def describe(predicate: Dict, *, satisfied: bool) -> str:
    name = predicate["name"].lower()
    args = [humanize(value) for value in predicate["args"]]
    if name == "on":
        return f"{args[0]} is on {args[1]}" if satisfied else f"{args[0]} is not yet on {args[1]}"
    if name == "in":
        return f"{args[0]} is in {args[1]}" if satisfied else f"{args[0]} is not yet in {args[1]}"
    if name == "open":
        return f"{args[0]} is open" if satisfied else f"{args[0]} is not yet open"
    if name == "close":
        return f"{args[0]} is closed" if satisfied else f"{args[0]} is not yet closed"
    if name == "turnon":
        return f"{args[0]} is on" if satisfied else f"{args[0]} is not yet turned on"
    if name == "turnoff":
        return f"{args[0]} is off" if satisfied else f"{args[0]} is not yet turned off"
    if name == "up":
        return f"{args[0]} is lifted" if satisfied else f"{args[0]} is not lifted"
    rendered_args = ", ".join(args)
    status = "satisfied" if satisfied else "not satisfied"
    return f"{name}({rendered_args}) is {status}"


def _sentence(parts: Sequence[str]) -> str:
    if not parts:
        return ""
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


def render_status(record: Dict) -> str:
    goals = record["goal_predicates"]
    if record["success"]:
        return _sentence([describe(item, satisfied=True) for item in goals])[:-1] + "; the task is complete."

    lifted = [item for item in record["auxiliary_predicates"] if item["name"] == "up" and item["satisfied"]]
    pending = [item for item in goals if not item["satisfied"]]
    completed = [item for item in goals if item["satisfied"]]
    parts = [describe(item, satisfied=True) for item in completed]
    parts.extend(describe(item, satisfied=True) for item in lifted)
    parts.extend(describe(item, satisfied=False) for item in pending)
    return _sentence(parts)


def render_events(record: Dict) -> List[str]:
    predicates = {item["id"]: item for item in record["goal_predicates"] + record["auxiliary_predicates"]}
    events = [
        describe(predicates[predicate_id_value], satisfied=True).capitalize() + "."
        for predicate_id_value in record["newly_satisfied"]
    ]
    events.extend(
        describe(predicates[predicate_id_value], satisfied=False).capitalize() + "."
        for predicate_id_value in record["newly_unsatisfied"]
    )
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_records = []
    with args.input.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            record["language"] = {
                "status": render_status(record),
                "events": render_events(record),
            }
            output_records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in output_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "frames": len(output_records),
                "event_frames": sum(bool(record["language"]["events"]) for record in output_records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
