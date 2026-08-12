#!/usr/bin/env python3
"""Replay one raw LIBERO demonstration and annotate predicates per state."""

# LIBERO and robosuite run in the existing Python 3.8 simulator environment.
# ruff: noqa: UP006, UP007, UP035

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from predicate_schema import build_frame_record
from predicate_schema import make_predicate_values


def _decode(value: Any) -> Any:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _find_libero_checkout(explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        candidates = [explicit_path]
    else:
        oxygen_repo = Path(__file__).resolve().parents[2]
        workspace = oxygen_repo.parent
        candidates = [
            oxygen_repo / "third_party" / "libero",
            workspace / "libero_exp" / "openpi_subtask_generation" / "third_party" / "libero",
        ]
    for candidate in candidates:
        if (candidate / "libero" / "libero" / "__init__.py").exists():
            return candidate
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find a LIBERO checkout. Pass --libero-checkout explicitly; checked {checked}")


def _resolve_bddl_path(stored_path: str, bddl_root: Path) -> Path:
    candidate = Path(stored_path)
    if candidate.exists():
        return candidate
    matches = list(bddl_root.rglob(candidate.name))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one local BDDL file named {candidate.name}, found {len(matches)} under {bddl_root}"
        )
    return matches[0]


def _goal_and_auxiliary_states(env: Any) -> Tuple[List[Sequence[str]], List[Sequence[str]]]:
    goal_states = [list(state) for state in env.env.parsed_problem["goal_state"]]
    movable_objects = set(env.env.objects_dict)
    goal_objects = []
    for state in goal_states:
        if len(state) > 1 and state[1] in movable_objects and state[1] not in goal_objects:
            goal_objects.append(state[1])
    auxiliary_states = [["grasped", object_name] for object_name in goal_objects]
    # Container tasks omit opening from the terminal goal, although opening is
    # an observable prerequisite in the demonstration.
    for state in goal_states:
        if len(state) < 3 or str(state[0]).lower() != "in":
            continue
        target = str(state[2])
        if "cabinet" in target and target.endswith("_region"):
            auxiliary_states.append(["open", target])
        elif target.endswith("_heating_region"):
            auxiliary_states.append(["open", target[: -len("_heating_region")]])

    goal_keys = {tuple(str(value).lower() for value in state) for state in goal_states}
    auxiliary_states = [
        state for state in auxiliary_states if tuple(str(value).lower() for value in state) not in goal_keys
    ]
    return goal_states, auxiliary_states


def _evaluate_states(env: Any, states: Sequence[Sequence[str]]) -> List[bool]:
    return [
        _is_grasped(env, str(state[1]))
        if str(state[0]).lower() == "grasped"
        else bool(env.env._eval_predicate(state))  # noqa: SLF001
        for state in states
    ]


def _is_grasped(env: Any, object_name: str) -> bool:
    """Return whether the closed gripper stably contacts the target object."""
    base_env = env.env
    object_geoms = set(base_env.get_object(object_name).contact_geoms)
    gripper_geoms = base_env.robots[0].gripper.important_geoms
    left_geoms = set(gripper_geoms["left_finger"])
    right_geoms = set(gripper_geoms["right_finger"])
    left_contact = False
    right_contact = False
    for contact_index in range(base_env.sim.data.ncon):
        contact = base_env.sim.data.contact[contact_index]
        geom_names = {
            base_env.sim.model.geom_id2name(contact.geom1),
            base_env.sim.model.geom_id2name(contact.geom2),
        }
        if not geom_names & object_geoms:
            continue
        left_contact = left_contact or bool(geom_names & left_geoms)
        right_contact = right_contact or bool(geom_names & right_geoms)
        if left_contact and right_contact:
            return True
    # Concave objects such as bowls may register collision on only one finger.
    # Accept that case only when a finger joint has substantially closed.
    finger_qpos = base_env.sim.data.qpos[base_env.robots[0]._ref_gripper_joint_pos_indexes]  # noqa: SLF001
    return (left_contact or right_contact) and bool(np.min(np.abs(finger_qpos)) < 0.03)


def _rewrite_libero_asset_paths(model_xml: str, asset_root: Path) -> str:
    tree = ET.fromstring(model_xml)
    for tag in ("mesh", "texture", "hfield"):
        for element in tree.findall(f".//{tag}"):
            stored_path = element.get("file")
            if not stored_path or Path(stored_path).exists():
                continue
            path_parts = Path(stored_path).parts
            if "assets" not in path_parts:
                continue
            relative_path = Path(*path_parts[path_parts.index("assets") + 1 :])
            local_path = asset_root / relative_path
            if local_path.exists():
                element.set("file", str(local_path))
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def _prepare_environment(
    env: Any,
    model_xml: Optional[str],
    initial_state: np.ndarray,
    libero_asset_root: Path,
) -> None:
    env.reset()
    if model_xml:
        from libero.libero.utils.utils import postprocess_model_xml

        portable_xml = postprocess_model_xml(model_xml, {})
        portable_xml = _rewrite_libero_asset_paths(portable_xml, libero_asset_root)
        env.reset_from_xml_string(portable_xml)
        env.sim.reset()
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()


def _annotate_state(
    *,
    env: Any,
    dataset_name: str,
    demo_name: str,
    frame: int,
    goal_states: Sequence[Sequence[str]],
    auxiliary_states: Sequence[Sequence[str]],
    previous_values: Optional[Dict[str, bool]],
    replay_error: Optional[float],
) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    goals = make_predicate_values("goal", goal_states, _evaluate_states(env, goal_states))
    auxiliary = make_predicate_values("aux", auxiliary_states, _evaluate_states(env, auxiliary_states))
    return build_frame_record(
        dataset=dataset_name,
        demo=demo_name,
        frame=frame,
        goal_predicates=goals,
        auxiliary_predicates=auxiliary,
        previous_values=previous_values,
        replay_state_l2_error=replay_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--mode", choices=("recorded-states", "action-replay"), default="recorded-states")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--bddl-root", type=Path, default=None)
    parser.add_argument("--libero-checkout", type=Path, default=None)
    parser.add_argument("--use-stored-model-xml", action="store_true")
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    libero_checkout = _find_libero_checkout(args.libero_checkout)
    sys.path.insert(0, str(libero_checkout))
    from libero.libero.envs import OffScreenRenderEnv

    bddl_root = args.bddl_root or libero_checkout / "libero" / "libero" / "bddl_files"
    with h5py.File(args.dataset, "r") as dataset:
        data = dataset["data"]
        if args.demo not in data:
            raise KeyError(f"{args.demo} is not present in {args.dataset}")
        demo = data[args.demo]
        states = np.asarray(demo["states"])
        actions = np.asarray(demo["actions"])
        frame_count = len(states) if args.max_frames is None else min(len(states), args.max_frames)
        if frame_count == 0:
            raise ValueError("No states selected")

        stored_bddl = str(_decode(data.attrs["bddl_file_name"]))
        bddl_path = _resolve_bddl_path(stored_bddl, bddl_root)
        problem_info = json.loads(str(_decode(data.attrs["problem_info"])))
        task_instruction = str(problem_info["language_instruction"])
        model_xml = _decode(demo.attrs.get("model_file"))
        initial_state = np.asarray(demo.attrs.get("init_state", states[0]))

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            use_camera_obs=False,
            camera_heights=64,
            camera_widths=64,
            render_gpu_device_id=args.render_gpu_device_id,
            ignore_done=True,
        )
        try:
            _prepare_environment(
                env,
                model_xml if args.use_stored_model_xml else None,
                initial_state,
                libero_checkout / "libero" / "libero" / "assets",
            )
            goal_states, auxiliary_states = _goal_and_auxiliary_states(env)
            previous_values = None
            records = []

            for frame in range(frame_count):
                replay_error = None
                if args.mode == "recorded-states":
                    env.sim.set_state_from_flattened(states[frame])
                    env.sim.forward()
                else:
                    if frame > 0:
                        env.step(actions[frame - 1])
                    replay_error = float(np.linalg.norm(env.sim.get_state().flatten() - states[frame]))

                record, previous_values = _annotate_state(
                    env=env,
                    dataset_name=args.dataset.name,
                    demo_name=args.demo,
                    frame=frame,
                    goal_states=goal_states,
                    auxiliary_states=auxiliary_states,
                    previous_values=previous_values,
                    replay_error=replay_error,
                )
                record["annotation_mode"] = args.mode
                record["task_instruction"] = task_instruction
                records.append(record)
        finally:
            env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "dataset": str(args.dataset),
        "demo": args.demo,
        "mode": args.mode,
        "used_stored_model_xml": bool(args.use_stored_model_xml),
        "task_instruction": task_instruction,
        "frames": len(records),
        "goal_predicates": records[0]["goal_predicates"],
        "auxiliary_predicates": records[0]["auxiliary_predicates"],
        "final_success": records[-1]["success"],
        "first_success_frame": next((row["frame"] for row in records if row["success"]), None),
    }
    if args.mode == "action-replay":
        errors = [row["replay_state_l2_error"] for row in records]
        summary["replay_state_l2_error"] = {
            "mean": float(np.mean(errors)),
            "max": float(np.max(errors)),
            "final": errors[-1],
        }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
