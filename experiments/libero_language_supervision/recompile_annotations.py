#!/usr/bin/env python3
"""Recompile language fields from saved raw predicate trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from render_text_labels import compile_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_root", type=Path)
    parser.add_argument("--stability-window", type=int, default=3)
    args = parser.parse_args()

    paths = sorted(args.annotation_root.glob("*/*/demo_*.jsonl"))
    for index, path in enumerate(paths, 1):
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        compiled = compile_trajectory(rows, stability_window=args.stability_window)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in compiled:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(path)
        if index % 100 == 0 or index == len(paths):
            print(json.dumps({"completed": index, "total": len(paths)}), flush=True)


if __name__ == "__main__":
    main()
