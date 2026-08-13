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
    for frame, (goal_value, up_value) in enumerate(zip(goal_values, up_values)):  # noqa: B905
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


def _pickup_trajectory(goal_values, grasp_values, pickup_values):
    records = []
    for frame, (goal_value, grasp_value, pickup_value) in enumerate(  # noqa: B905
        zip(goal_values, grasp_values, pickup_values)  # noqa: B905
    ):
        goal = _predicate("goal", 0, "in", ["alphabet_soup_1", "basket_1_contain_region"], goal_value)
        grasped = _predicate("aux", 0, "grasped", ["alphabet_soup_1"], grasp_value)
        picked_up = _predicate("aux", 1, "picked_up", ["alphabet_soup_1"], pickup_value)
        records.append({
            "frame": frame,
            "task_instruction": "pick up the alphabet soup and place it in the basket",
            "goal_predicates": [goal],
            "auxiliary_predicates": [grasped, picked_up],
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
    row = {
        "language": {
            "completed": "Done.",
            "remaining": "Left.",
            "next": "Act.",
            "visual_memory": "Seen.",
        }
    }
    assert compose_language(row, ["completed", "next"], " ") == "Done. Act."
    assert compose_language(row, ["visual_memory"]) == "Seen."


def test_visual_memory_is_causal_stable_and_excludes_pickup():
    rows = compile_trajectory(
        _pickup_trajectory(
            [False, False, False, False, False, True, True, True],
            [False, True, True, True, False, False, False, False],
            [False, False, True, True, True, False, False, False],
        ),
        stability_window=2,
    )
    assert rows[4]["language"]["visual_memory"] == "No relevant task progress is visible yet."
    assert rows[6]["language"]["visual_memory"] == "The alphabet soup is in the basket."
    assert rows[6]["language_components"]["visual_memory_predicate_ids"] == [
        "goal:00:in(alphabet_soup_1,basket_1_contain_region)"
    ]


def test_visual_memory_does_not_claim_initial_or_reverted_state():
    records = []
    values = [True, True, False, False, True, True, False, False]
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
    assert rows[1]["language"]["visual_memory"] == "No relevant task progress is visible yet."
    assert rows[5]["language"]["visual_memory"] == "The microwave is closed."
    assert rows[7]["language"]["visual_memory"] == "No relevant task progress is visible yet."


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
    assert rows[0]["language"]["next"] == "Place the cream cheese in the black bowl."


def test_only_picked_up_splits_pickup_from_container_placement():
    rows = compile_trajectory(
        _pickup_trajectory(
            [False, False, False, False, False, False, True, True],
            [False, True, True, True, True, True, False, False],
            [False, False, False, True, True, True, False, False],
        ),
        stability_window=2,
    )
    assert rows[0]["language"]["next"] == "Pick up the alphabet soup."
    assert rows[1]["language"]["next"] == "Pick up the alphabet soup."
    assert rows[3]["language"]["next"] == "Place the alphabet soup in the basket."
    assert rows[3]["language"]["completed"] == "The alphabet soup has been picked up."
    assert rows[3]["stable_progress"]["auxiliary_completion_frames"]["aux:00:grasped(alphabet_soup_1)"] is None
    assert rows[6]["language"]["next"] == "Task complete."


def test_grasped_is_ignored_for_push_to_spatial_region():
    records = []
    for frame, (goal_value, grasp_value) in enumerate(  # noqa: B905
        zip(  # noqa: B905
            [False, False, False, False, False, False, True, True],
            [False, False, True, True, True, True, False, False],
        )
    ):
        goal = _predicate(
            "goal",
            0,
            "on",
            ["plate_1", "main_table_stove_front_region"],
            goal_value,
        )
        grasped = _predicate("aux", 0, "grasped", ["plate_1"], grasp_value)
        picked_up = _predicate("aux", 1, "picked_up", ["plate_1"], grasp_value)
        records.append({
            "frame": frame,
            "task_instruction": "push the plate to the front of the stove",
            "goal_predicates": [goal],
            "auxiliary_predicates": [grasped, picked_up],
            "success": goal_value,
        })
    rows = compile_trajectory(records, stability_window=2)
    assert rows[0]["language"]["next"] == "Move the plate to the area in front of the stove."
    assert rows[2]["language"]["next"] == "Move the plate to the area in front of the stove."
    assert "picked up" not in rows[2]["language"]["completed"]


def test_repeated_objects_are_named_by_execution_order_not_instance_id():
    records = []
    for frame in range(10):
        goals = [
            _predicate("goal", 0, "on", ["moka_pot_1", "flat_stove_1_cook_region"], frame >= 8),
            _predicate("goal", 1, "on", ["moka_pot_2", "flat_stove_1_cook_region"], frame >= 5),
        ]
        auxiliary = [
            _predicate("aux", 0, "grasped", ["moka_pot_1"], 6 <= frame < 9),
            _predicate("aux", 1, "picked_up", ["moka_pot_1"], 6 <= frame < 9),
            _predicate("aux", 2, "grasped", ["moka_pot_2"], 2 <= frame < 6),
            _predicate("aux", 3, "picked_up", ["moka_pot_2"], 2 <= frame < 6),
        ]
        records.append({
            "frame": frame,
            "task_instruction": "put both moka pots on the stove",
            "goal_predicates": goals,
            "auxiliary_predicates": auxiliary,
            "success": frame >= 8,
        })
    rows = compile_trajectory(records, stability_window=2)
    assert rows[0]["language"]["next"] == "Pick up the first moka pot."
    assert rows[2]["language"]["next"] == "Place the first moka pot on the stove."
    assert rows[5]["language"]["next"] == "Pick up the second moka pot."
    assert rows[6]["language"]["next"] == "Place the second moka pot on the stove."
