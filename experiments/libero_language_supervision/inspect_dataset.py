#!/usr/bin/env python3
"""Inspect the replay-critical contents of a raw LIBERO HDF5 file."""

import argparse
import json
from pathlib import Path

import h5py


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--demo", default=None)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as dataset:
        data = dataset["data"]
        demos = sorted(data.keys(), key=lambda name: int(name.rsplit("_", 1)[-1]))
        demo_name = args.demo or demos[0]
        demo = data[demo_name]
        attrs = {key: _decode(value) for key, value in data.attrs.items()}
        summary = {
            "path": str(args.dataset),
            "num_demos": len(demos),
            "selected_demo": demo_name,
            "demo_keys": sorted(demo.keys()),
            "actions_shape": list(demo["actions"].shape),
            "states_shape": list(demo["states"].shape),
            "has_model_file": "model_file" in demo.attrs,
            "has_init_state": "init_state" in demo.attrs,
            "bddl_file_name": attrs.get("bddl_file_name"),
            "problem_info": json.loads(attrs["problem_info"]) if "problem_info" in attrs else None,
            "env_args": json.loads(attrs["env_args"]) if "env_args" in attrs else None,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
