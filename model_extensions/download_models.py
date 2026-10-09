"""Download model definitions and tokenizers for random-initialized examples."""

import argparse
import hashlib
import json
from pathlib import Path

GROUPS = {
    "xiaomi": ["XiaomiRobotics/Xiaomi-Robotics-0-LIBERO"],
    "starvla": [
        "Qwen/Qwen3-VL-4B-Instruct",
        "StarVLA/Qwen3VL-PI_v3-Bridge-RT_1",
        "StarVLA/Qwen3VL-GR00T-Bridge-RT-1",
        "StarVLA/Qwen3VL-OFT-Bridge-RT-1",
    ],
    "qwen35": [f"Qwen/Qwen3.5-{size}" for size in ("0.8B", "2B", "4B", "9B")],
}


def verify(directory, files):
    for name, expected in files.items():
        path = directory / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Missing or changed model file: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets")
    parser.add_argument("--size", choices=("0.8B", "2B", "4B", "9B"), help="Download one Qwen3.5 size.")
    parser.add_argument("--verify-only", action="store_true", help="Check existing files without network access.")
    args = parser.parse_args()
    if args.size and args.group != "qwen35":
        parser.error("--size is only used with qwen35")
    manifest = json.loads(Path(__file__).with_name("models.json").read_text())
    repos = GROUPS[args.group]
    if args.size:
        repos = [f"Qwen/Qwen3.5-{args.size}"]
    for repo in repos:
        spec = manifest[repo]
        target = args.output.resolve() / repo.split("/")[-1]
        if not args.verify_only:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=repo, revision=spec["revision"],
                              allow_patterns=list(spec["files"]), local_dir=target)
        verify(target, spec["files"])
        print(target)


if __name__ == "__main__":
    main()
