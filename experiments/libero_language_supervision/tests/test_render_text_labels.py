from render_text_labels import humanize
from render_text_labels import render_events
from render_text_labels import render_status


def test_humanize_known_object():
    assert humanize("akita_black_bowl_1") == "the black bowl"


def test_status_and_event():
    goal = {
        "id": "goal:00:on(akita_black_bowl_1,plate_1)",
        "name": "on",
        "args": ["akita_black_bowl_1", "plate_1"],
        "satisfied": False,
    }
    lifted = {
        "id": "aux:00:up(akita_black_bowl_1)",
        "name": "up",
        "args": ["akita_black_bowl_1"],
        "satisfied": True,
    }
    record = {
        "goal_predicates": [goal],
        "auxiliary_predicates": [lifted],
        "newly_satisfied": [lifted["id"]],
        "newly_unsatisfied": [],
        "success": False,
    }
    assert render_status(record) == "The black bowl is lifted; the black bowl is not yet on the plate."
    assert render_events(record) == ["The black bowl is lifted."]
