#!/usr/bin/env python3
"""Compile stable LIBERO predicate trajectories into language supervision."""

# This module also supports LIBERO's Python 3.8 environment.
# ruff: noqa: UP006, UP007, UP035

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth"}
OBJECT_ALIASES = {
    "akita_black_bowl": "black bowl",
    "bbq_sauce": "BBQ sauce",
    "white_yellow_mug": "yellow and white mug",
    "porcelain_mug": "white mug",
    "flat_stove": "stove",
}

# These names encode spatial regions rather than ordinary objects. They are
# stable across the four official LIBERO suites.
REGION_ALIASES = {
    "main_table_stove_front_region": "area in front of the stove",
    "living_room_table_plate_right_region": "area to the right of the plate",
    "desk_caddy_back_contain_region": "back compartment of the caddy",
}


def _strip_instance(name: str) -> Tuple[str, Optional[int]]:
    match = re.match(r"^(.*)_([0-9]+)(_.*)?$", name)
    if not match:
        return name, None
    return match.group(1) + (match.group(3) or ""), int(match.group(2))


def _base_display_name(name: str) -> str:
    base, _ = _strip_instance(name)
    if base in REGION_ALIASES:
        return REGION_ALIASES[base]
    if base.endswith("_contain_region"):
        base = base[: -len("_contain_region")]
    if base.endswith("_cook_region") or base.endswith("_heating_region"):
        base = base.rsplit("_", 2)[0]
    if base == "wine_rack_top_region":
        return "wine rack"
    if base.endswith("_top_side"):
        return "top of the " + _base_display_name(base[: -len("_top_side")])
    for level in ("top", "middle", "bottom"):
        suffix = "_" + level + "_region"
        if base.endswith(suffix):
            fixture = _base_display_name(base[: -len(suffix)])
            return level + " drawer of the " + fixture
    base = OBJECT_ALIASES.get(base, base)
    return base.replace("_", " ")


class NameResolver:
    """Resolve predicate arguments into concise, unambiguous task-local names."""

    def __init__(
        self,
        predicates: Sequence[Mapping[str, Any]],
        task_instruction: str = "",
        execution_ordinals: Optional[Mapping[str, int]] = None,
    ) -> None:
        names = {str(arg) for item in predicates for arg in item["args"]}
        bases = [_strip_instance(name)[0] for name in names]
        self._counts = Counter(bases)
        self._instruction = task_instruction.lower()
        self._execution_ordinals = dict(execution_ordinals or {})

    @property
    def instruction(self) -> str:
        return self._instruction

    def resolve(self, name: str) -> str:
        base, instance = _strip_instance(name)
        display = _base_display_name(name)

        if base == "cream_cheese" and "cream cheese box" in self._instruction:
            display = "cream cheese box"

        if "left plate" in self._instruction and "right plate" in self._instruction and base == "plate":
            return "the left plate" if instance == 1 else "the right plate"
        ordinal = self._execution_ordinals.get(name)
        if self._counts[base] > 1 and ordinal in ORDINALS:
            return "the " + ORDINALS[ordinal] + " " + display
        if display.startswith(("top of the ", "area ", "back compartment ")):
            return "the " + display
        return "the " + display


def humanize(name: str) -> str:
    """Backward-compatible single-name rendering used by external callers."""
    return NameResolver([], "").resolve(name)


def _describe_state(predicate: Mapping[str, Any], resolver: NameResolver, satisfied: bool) -> str:
    name = str(predicate["name"]).lower()
    args = [resolver.resolve(str(value)) for value in predicate["args"]]
    if name == "on":
        if predicate["args"][0].startswith("cream_cheese") and "in the bowl" in resolver.instruction:
            return f"{args[0]} is in {args[1]}" if satisfied else f"{args[0]} still needs to be placed in {args[1]}"
        if "area " in args[1]:
            return f"{args[0]} is in {args[1]}" if satisfied else f"{args[0]} still needs to be moved to {args[1]}"
        return f"{args[0]} is on {args[1]}" if satisfied else f"{args[0]} still needs to be placed on {args[1]}"
    if name == "in":
        return f"{args[0]} is in {args[1]}" if satisfied else f"{args[0]} still needs to be placed in {args[1]}"
    if name == "open":
        return f"{args[0]} is open" if satisfied else f"{args[0]} still needs to be opened"
    if name == "close":
        return f"{args[0]} is closed" if satisfied else f"{args[0]} still needs to be closed"
    if name == "turnon":
        return f"{args[0]} is on" if satisfied else f"{args[0]} still needs to be turned on"
    if name == "turnoff":
        return f"{args[0]} is off" if satisfied else f"{args[0]} still needs to be turned off"
    if name in ("picked_up", "up"):
        return f"{args[0]} has been picked up" if satisfied else f"{args[0]} still needs to be picked up"
    status = "is complete" if satisfied else "is not complete"
    return f"{name}({', '.join(args)}) {status}"


def _describe_next(predicate: Mapping[str, Any], resolver: NameResolver) -> str:
    name = str(predicate["name"]).lower()
    args = [resolver.resolve(str(value)) for value in predicate["args"]]
    if name in ("picked_up", "up"):
        return f"Pick up {args[0]}."
    if name == "on":
        if predicate["args"][0].startswith("cream_cheese") and "in the bowl" in resolver.instruction:
            return f"Place {args[0]} in {args[1]}."
        # The plate-front task is demonstrated by pushing rather than lifting.
        if "area " in args[1]:
            return f"Move {args[0]} to {args[1]}."
        return f"Place {args[0]} on {args[1]}."
    if name == "in":
        return f"Place {args[0]} in {args[1]}."
    if name == "open":
        return f"Open {args[0]}."
    if name == "close":
        return f"Close {args[0]}."
    if name == "turnon":
        return f"Turn on {args[0]}."
    if name == "turnoff":
        return f"Turn off {args[0]}."
    return f"Complete {name} for {', '.join(args)}."


def _sentence(parts: Sequence[str], empty: str) -> str:
    if not parts:
        return empty
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


def _first_stable_true(values: Sequence[bool], window: int, end: Optional[int] = None) -> Optional[int]:
    stop = len(values) if end is None else min(len(values), end + 1)
    if window < 1:
        raise ValueError("stability window must be positive")
    for frame in range(0, stop - window + 1):
        if all(values[frame : frame + window]):
            return frame
    return None


def _first_stable_false_to_true(values: Sequence[bool], window: int, end: Optional[int] = None) -> Optional[int]:
    """Find a stable positive segment preceded by a stable negative segment."""
    stop = len(values) if end is None else min(len(values), end + 1)
    saw_stable_false = False
    for frame in range(0, stop - window + 1):
        segment = values[frame : frame + window]
        if not any(segment):
            saw_stable_false = True
        elif saw_stable_false and all(segment):
            return frame
    return None


def _final_stable_true(values: Sequence[bool], success_frame: int, window: int) -> Optional[int]:
    """Find the start of the true segment that establishes terminal success."""
    if not values[success_frame]:
        return None
    start = success_frame
    while start > 0 and values[start - 1]:
        start -= 1
    return start


def compile_trajectory(records: Sequence[Dict[str, Any]], stability_window: int = 3) -> List[Dict[str, Any]]:
    if not records:
        return []
    predicates = {
        item["id"]: item
        for item in records[0]["goal_predicates"] + records[0]["auxiliary_predicates"]
    }
    goal_ids = [item["id"] for item in records[0]["goal_predicates"]]
    aux_ids = [item["id"] for item in records[0]["auxiliary_predicates"]]
    series = {
        predicate_id: [
            next(
                item["satisfied"]
                for item in row["goal_predicates"] + row["auxiliary_predicates"]
                if item["id"] == predicate_id
            )
            for row in records
        ]
        for predicate_id in predicates
    }
    success_values = [bool(row["success"]) for row in records]
    success_frame = _first_stable_true(success_values, stability_window)
    if success_frame is None:
        success_frame = next(
            (index for index in range(len(records) - 1, -1, -1) if success_values[index]),
            len(records) - 1,
        )
    completion_frames = {
        predicate_id: _final_stable_true(series[predicate_id], success_frame, stability_window)
        for predicate_id in goal_ids
    }
    auxiliary_frames = {}
    for predicate_id in aux_ids:
        predicate = predicates[predicate_id]
        predicate_name = str(predicate["name"]).lower()
        if predicate_name == "grasped":
            # Preserve grasp provenance in the structured annotation, but do
            # not expose it as a language milestone in the current target set.
            auxiliary_frames[predicate_id] = None
            continue
        if predicate_name == "picked_up":
            object_name = str(predicate["args"][0])
            related_goals = [
                predicates[goal_id]
                for goal_id in goal_ids
                if predicates[goal_id]["args"] and str(predicates[goal_id]["args"][0]) == object_name
            ]
            if related_goals and all(
                len(goal["args"]) > 1 and _base_display_name(str(goal["args"][1])).startswith("area ")
                for goal in related_goals
            ):
                auxiliary_frames[predicate_id] = None
                continue
        if predicate_name in ("picked_up", "up"):
            # Pickup and the legacy Up predicate are useful only after a stable
            # transition from their initial false state.
            frame = _first_stable_false_to_true(series[predicate_id], stability_window, end=success_frame)
        else:
            frame = _first_stable_true(series[predicate_id], stability_window, end=success_frame)
        auxiliary_frames[predicate_id] = frame

    # Auxiliary events are useful only when they precede a related terminal goal.
    events = []
    for predicate_id, frame in auxiliary_frames.items():
        if frame is not None:
            events.append((frame, 0, predicate_id))
    for predicate_id, frame in completion_frames.items():
        if frame is not None:
            events.append((frame, 1, predicate_id))
    events.sort()
    milestone_ids = [predicate_id for _, _, predicate_id in events]
    missing_goal_ids = [predicate_id for predicate_id in goal_ids if predicate_id not in milestone_ids]

    instruction = str(records[0].get("task_instruction", ""))
    pickup_order = sorted(
        (
            frame,
            str(predicates[predicate_id]["args"][0]),
        )
        for predicate_id, frame in auxiliary_frames.items()
        if frame is not None and str(predicates[predicate_id]["name"]).lower() == "picked_up"
    )
    execution_ordinals = {}
    per_base_counts = Counter()
    for _, object_name in pickup_order:
        base, _ = _strip_instance(object_name)
        per_base_counts[base] += 1
        execution_ordinals[object_name] = per_base_counts[base]
    resolver = NameResolver(list(predicates.values()), instruction, execution_ordinals)
    output = []
    for frame, source in enumerate(records):
        completed_ids = [pid for event_frame, _, pid in events if event_frame <= frame]
        remaining_ids = [pid for event_frame, _, pid in events if event_frame > frame] + missing_goal_ids
        completed_goal_ids = [pid for pid in completed_ids if pid in goal_ids]
        remaining_goal_ids = [pid for pid in goal_ids if pid not in completed_goal_ids]
        next_event = next(((event_frame, pid) for event_frame, _, pid in events if event_frame > frame), None)

        row = dict(source)
        row["stable_progress"] = {
            "stability_window": stability_window,
            "success_frame": success_frame,
            "goal_completion_frames": completion_frames,
            "auxiliary_completion_frames": auxiliary_frames,
        }
        row["language_components"] = {
            "completed_predicate_ids": completed_ids,
            "remaining_predicate_ids": remaining_ids,
            "completed_goal_predicate_ids": completed_goal_ids,
            "remaining_goal_predicate_ids": remaining_goal_ids,
            "next_target_predicate_id": next_event[1] if next_event else None,
            "next_target_frame": next_event[0] if next_event else None,
        }
        row["language"] = {
            "completed": _sentence(
                [_describe_state(predicates[pid], resolver, True) for pid in completed_ids],
                "No task step has been completed yet.",
            ),
            "remaining": _sentence(
                [_describe_state(predicates[pid], resolver, False) for pid in remaining_ids],
                "No task step remains.",
            ),
            "next": _describe_next(predicates[next_event[1]], resolver) if next_event else "Task complete.",
        }
        output.append(row)
    return output


def compose_language(record: Mapping[str, Any], components: Sequence[str], separator: str = "\n") -> str:
    """Compose any configured subset without regenerating annotations."""
    allowed = {"completed", "remaining", "next"}
    unknown = set(components) - allowed
    if unknown:
        raise ValueError("Unknown language components: " + ", ".join(sorted(unknown)))
    return separator.join(str(record["language"][component]) for component in components)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stability-window", type=int, default=3)
    parser.add_argument("--components", nargs="+", choices=("completed", "remaining", "next"), default=None)
    parser.add_argument("--separator", default="\n")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    output_records = compile_trajectory(records, stability_window=args.stability_window)
    if args.components:
        for record in output_records:
            record["language_target"] = compose_language(record, args.components, args.separator)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in output_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    print(json.dumps({"input": str(args.input), "output": str(args.output), "frames": len(output_records)}, indent=2))


if __name__ == "__main__":
    main()
