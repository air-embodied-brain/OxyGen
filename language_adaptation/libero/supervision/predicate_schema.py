"""Structured LIBERO predicate records and transition tracking."""

# This module runs in LIBERO's Python 3.8 environment.
# ruff: noqa: UP006, UP035, UP045

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PredicateValue:
    predicate_id: str
    name: str
    args: Tuple[str, ...]
    satisfied: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.predicate_id,
            "name": self.name,
            "args": list(self.args),
            "satisfied": self.satisfied,
        }


def predicate_id(kind: str, index: int, state: Sequence[str]) -> str:
    """Make a stable ID while preserving duplicate predicates and ordering."""
    name = str(state[0]).lower()
    args = ",".join(str(value) for value in state[1:])
    return f"{kind}:{index:02d}:{name}({args})"


def make_predicate_values(
    kind: str,
    states: Iterable[Sequence[str]],
    values: Iterable[bool],
) -> List[PredicateValue]:
    state_list = list(states)
    value_list = list(values)
    if len(state_list) != len(value_list):
        raise ValueError("Predicate states and values must have the same length")
    records = []
    for index, (state, value) in enumerate(zip(state_list, value_list)):  # noqa: B905
        records.append(
            PredicateValue(
                predicate_id=predicate_id(kind, index, state),
                name=str(state[0]).lower(),
                args=tuple(str(arg) for arg in state[1:]),
                satisfied=bool(value),
            )
        )
    return records


def build_frame_record(
    *,
    dataset: str,
    demo: str,
    frame: int,
    goal_predicates: Sequence[PredicateValue],
    auxiliary_predicates: Sequence[PredicateValue],
    previous_values: Optional[Mapping[str, bool]],
    replay_state_l2_error: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    all_predicates = list(goal_predicates) + list(auxiliary_predicates)
    current_values = {item.predicate_id: item.satisfied for item in all_predicates}

    if previous_values is None:
        newly_satisfied = []
        newly_unsatisfied = []
    else:
        newly_satisfied = [
            key for key, value in current_values.items() if value and not previous_values.get(key, False)
        ]
        newly_unsatisfied = [
            key for key, value in current_values.items() if not value and previous_values.get(key, False)
        ]

    completed_goal_count = sum(item.satisfied for item in goal_predicates)
    goal_count = len(goal_predicates)
    record = {
        "dataset": dataset,
        "demo": demo,
        "frame": frame,
        "goal_predicates": [item.as_dict() for item in goal_predicates],
        "auxiliary_predicates": [item.as_dict() for item in auxiliary_predicates],
        "completed_goal_count": completed_goal_count,
        "goal_count": goal_count,
        "goal_progress": completed_goal_count / goal_count if goal_count else 1.0,
        "newly_satisfied": newly_satisfied,
        "newly_unsatisfied": newly_unsatisfied,
        "success": bool(goal_count == completed_goal_count),
    }
    if replay_state_l2_error is not None:
        record["replay_state_l2_error"] = replay_state_l2_error
    return record, current_values
