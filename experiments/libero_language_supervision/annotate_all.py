#!/usr/bin/env python3
"""Annotate all demonstrations in the four standard LIBERO suites."""

# ruff: noqa: UP006, UP007, UP035

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List

import h5py
import numpy as np
from render_text_labels import compile_trajectory
from replay_and_annotate import _annotate_state
from replay_and_annotate import _capture_pickup_references
from replay_and_annotate import _decode
from replay_and_annotate import _find_libero_checkout
from replay_and_annotate import _goal_and_auxiliary_states
from replay_and_annotate import _prepare_environment
from replay_and_annotate import _resolve_bddl_path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def annotate_file(
    dataset_path: Path,
    output_root: Path,
    libero_checkout: Path,
    stability_window: int,
    render_gpu_device_id: int,
) -> Dict[str, Any]:
    suite = dataset_path.parent.name
    task = dataset_path.stem[:-5] if dataset_path.stem.endswith("_demo") else dataset_path.stem
    task_root = output_root / suite / task
    bddl_root = libero_checkout / "libero" / "libero" / "bddl_files"
    asset_root = libero_checkout / "libero" / "libero" / "assets"
    started = time.time()
    summaries = []

    with h5py.File(dataset_path, "r") as dataset:
        data = dataset["data"]
        stored_bddl = str(_decode(data.attrs["bddl_file_name"]))
        bddl_path = _resolve_bddl_path(stored_bddl, bddl_root)
        problem_info = json.loads(str(_decode(data.attrs["problem_info"])))
        instruction = str(problem_info["language_instruction"])
        demo_names = sorted(data.keys(), key=lambda value: int(value.split("_")[-1]))

        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            use_camera_obs=False,
            camera_heights=64,
            camera_widths=64,
            render_gpu_device_id=render_gpu_device_id,
            ignore_done=True,
        )
        try:
            goal_states, auxiliary_states = _goal_and_auxiliary_states(env)

            for demo_name in demo_names:
                output_path = task_root / (demo_name + ".jsonl")
                if output_path.exists() and output_path.with_suffix(".jsonl.summary.json").exists():
                    summaries.append(json.loads(output_path.with_suffix(".jsonl.summary.json").read_text()))
                    continue

                states = np.asarray(data[demo_name]["states"])
                model_xml = _decode(data[demo_name].attrs.get("model_file"))
                _prepare_environment(env, model_xml, states[0], asset_root)
                pickup_references = _capture_pickup_references(env, auxiliary_states)
                previous_values = None
                raw_records = []
                for frame, state in enumerate(states):
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    record, previous_values = _annotate_state(
                        env=env,
                        dataset_name=dataset_path.name,
                        demo_name=demo_name,
                        frame=frame,
                        goal_states=goal_states,
                        auxiliary_states=auxiliary_states,
                        pickup_references=pickup_references,
                        previous_values=previous_values,
                        replay_error=None,
                    )
                    record.update({
                        "annotation_mode": "recorded-states",
                        "suite": suite,
                        "task": task,
                        "task_instruction": instruction,
                    })
                    raw_records.append(record)

                records = compile_trajectory(raw_records, stability_window=stability_window)
                _write_jsonl(output_path, records)
                summary = {
                    "suite": suite,
                    "task": task,
                    "dataset": str(dataset_path),
                    "demo": demo_name,
                    "frames": len(records),
                    "first_success_frame": next((row["frame"] for row in records if row["success"]), None),
                    "final_success": records[-1]["success"],
                    "goal_predicate_count": len(goal_states),
                    "auxiliary_predicate_count": len(auxiliary_states),
                }
                output_path.with_suffix(".jsonl.summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                summaries.append(summary)
        finally:
            env.close()

    file_summary = {
        "suite": suite,
        "task": task,
        "dataset": str(dataset_path),
        "task_instruction": instruction,
        "demos": len(summaries),
        "frames": sum(item["frames"] for item in summaries),
        "successful_demos": sum(item["final_success"] for item in summaries),
        "elapsed_seconds": time.time() - started,
    }
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "manifest.json").write_text(
        json.dumps(file_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return file_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--stability-window", type=int, default=3)
    parser.add_argument("--libero-checkout", type=Path, default=None)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--single-dataset", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    checkout = _find_libero_checkout(args.libero_checkout)
    sys.path.insert(0, str(checkout))
    if args.single_dataset is not None:
        result = annotate_file(
            args.single_dataset,
            args.output_root,
            checkout,
            args.stability_window,
            args.render_gpu_device_id,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    datasets = [path for suite in args.suites for path in sorted((args.dataset_root / suite).glob("*.hdf5"))]
    if not datasets:
        raise FileNotFoundError("No HDF5 files found under selected suites")

    manifests = []
    for index, dataset in enumerate(datasets, 1):
        print(json.dumps({"file": index, "total_files": len(datasets), "dataset": str(dataset)}), flush=True)
        suite = dataset.parent.name
        stem = dataset.stem
        task = stem[:-5] if stem.endswith("_demo") else stem
        manifest_path = args.output_root / suite / task / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with h5py.File(dataset, "r") as raw_dataset:
                expected_demos = len(raw_dataset["data"])
            if manifest.get("demos") == expected_demos:
                manifests.append(manifest)
                continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset-root",
            str(args.dataset_root),
            "--output-root",
            str(args.output_root),
            "--stability-window",
            str(args.stability_window),
            "--libero-checkout",
            str(checkout),
            "--render-gpu-device-id",
            str(args.render_gpu_device_id),
            "--single-dataset",
            str(dataset),
        ]
        subprocess.run(command, check=True, env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    aggregate = {
        "suites": args.suites,
        "files": len(manifests),
        "demos": sum(item["demos"] for item in manifests),
        "frames": sum(item["frames"] for item in manifests),
        "successful_demos": sum(item["successful_demos"] for item in manifests),
        "tasks": manifests,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
