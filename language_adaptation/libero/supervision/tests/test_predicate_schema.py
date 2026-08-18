from language_adaptation.libero.supervision.predicate_schema import build_frame_record
from language_adaptation.libero.supervision.predicate_schema import make_predicate_values


def test_progress_and_transitions():
    states = [["on", "bowl", "plate"], ["turnon", "stove"]]
    first = make_predicate_values("goal", states, [False, False])
    first_record, previous = build_frame_record(
        dataset="task.hdf5",
        demo="demo_0",
        frame=0,
        goal_predicates=first,
        auxiliary_predicates=[],
        previous_values=None,
    )
    assert first_record["goal_progress"] == 0.0
    assert not first_record["success"]

    second = make_predicate_values("goal", states, [False, True])
    second_record, _ = build_frame_record(
        dataset="task.hdf5",
        demo="demo_0",
        frame=1,
        goal_predicates=second,
        auxiliary_predicates=[],
        previous_values=previous,
    )
    assert second_record["goal_progress"] == 0.5
    assert second_record["newly_satisfied"] == ["goal:01:turnon(stove)"]


def test_auxiliary_predicates_do_not_define_success():
    goals = make_predicate_values("goal", [["on", "bowl", "plate"]], [True])
    auxiliary = make_predicate_values("aux", [["up", "bowl"]], [False])
    record, _ = build_frame_record(
        dataset="task.hdf5",
        demo="demo_0",
        frame=0,
        goal_predicates=goals,
        auxiliary_predicates=auxiliary,
        previous_values=None,
    )
    assert record["success"]
