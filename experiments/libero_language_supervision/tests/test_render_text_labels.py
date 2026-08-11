from render_text_labels import compile_trajectory
from render_text_labels import compose_language
from render_text_labels import humanize


def _predicate(kind, index, name, args, satisfied):
    return {
        "id": f"{kind}:{index:02d}:{name}({','.join(args)})",
        "name": name,
        "args": args,
        "satisfied": satisfied,
    }


def _trajectory(goal_values, up_values):
    records = []
    for frame, (goal_value, up_value) in enumerate(zip(goal_values, up_values)):
        goal = _predicate("goal", 0, "on", ["akita_black_bowl_1", "plate_1"], goal_value)
        up = _predicate("aux", 0, "up", ["akita_black_bowl_1"], up_value)
        records.append({
            "frame": frame,
            "task_instruction": "put the black bowl on the plate",
            "goal_predicates": [goal],
            "auxiliary_predicates": [up],
            "success": goal_value,
        })
    return records


def test_humanize_known_object_and_region():
    assert humanize("akita_black_bowl_1") == "the black bowl"
    assert humanize("basket_1_contain_region") == "the basket"
    assert humanize("wooden_cabinet_1_top_region") == "the top drawer of the wooden cabinet"
    assert humanize("wine_rack_1_top_region") == "the wine rack"


def test_compiler_uses_future_stable_events():
    rows = compile_trajectory(
        _trajectory(
            [False, False, False, False, False, False, False, True, True, True],
            [False, False, False, True, True, True, True, False, False, False],
        ),
        stability_window=3,
    )
    assert rows[0]["language"]["next"] == "Pick up the black bowl."
    assert rows[3]["language"]["next"] == "Place the black bowl on the plate."
    assert rows[0]["language"]["remaining"] == (
        "The black bowl still needs to be picked up; the black bowl still needs to be placed on the plate."
    )
    assert rows[3]["language"]["completed"] == "The black bowl has been picked up."
    assert rows[6]["language"]["completed"] == "The black bowl has been picked up."
    assert rows[7]["language"]["completed"] == (
        "The black bowl has been picked up; the black bowl is on the plate."
    )
    assert rows[7]["language"]["remaining"] == "No task step remains."
    assert rows[7]["language"]["next"] == "Task complete."


def test_terminal_completion_ignores_initially_true_goal():
    records = []
    values = [True, True, False, False, True, True]
    for frame, value in enumerate(values):
        goal = _predicate("goal", 0, "close", ["microwave_1"], value)
        records.append({
            "frame": frame,
            "task_instruction": "put the mug in the microwave and close it",
            "goal_predicates": [goal],
            "auxiliary_predicates": [],
            "success": value and frame >= 4,
        })
    rows = compile_trajectory(records, stability_window=2)
    assert rows[0]["language"]["completed"] == "No task step has been completed yet."
    assert rows[4]["language"]["completed"] == "The microwave is closed."


def test_initially_high_object_is_not_reported_as_picked_up():
    rows = compile_trajectory(
        _trajectory(
            [False, False, False, True, True, True],
            [True, True, True, True, True, False],
        ),
        stability_window=3,
    )
    assert rows[0]["language"]["completed"] == "No task step has been completed yet."
    assert "picked up" not in rows[0]["language"]["remaining"]
    assert rows[0]["language"]["next"] == "Place the black bowl on the plate."


def test_configurable_composition():
    row = {"language": {"completed": "Done.", "remaining": "Left.", "next": "Act."}}
    assert compose_language(row, ["completed", "next"], " ") == "Done. Act."


def test_instruction_override_preserves_natural_relation():
    records = []
    for frame, value in enumerate([False, False, True, True]):
        goal = _predicate("goal", 0, "on", ["cream_cheese_1", "akita_black_bowl_1"], value)
        records.append({
            "frame": frame,
            "task_instruction": "put the cream cheese in the bowl",
            "goal_predicates": [goal],
            "auxiliary_predicates": [],
            "success": value,
        })
    rows = compile_trajectory(records, stability_window=2)
    assert rows[0]["language"]["remaining"] == "The cream cheese still needs to be placed in the black bowl."
    assert rows[0]["language"]["next"] == "Put the cream cheese in the black bowl."
