"""Run an OxyGen model extension in its own environment."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess


def command(args, extra):
    python = args.envs.resolve() / args.model / "bin/python"
    assets = args.assets.resolve()
    output = args.output.resolve()
    if args.model == "xiaomi":
        module = "oxygen_models.xiaomi.check" if args.check else "oxygen_models.xiaomi.benchmark"
        flags = ["--checkpoint", str(assets / "Xiaomi-Robotics-0-LIBERO")]
    elif args.model == "starvla":
        module = "oxygen_models.starvla.benchmark"
        flags = ["--base-vlm", str(assets / "Qwen3-VL-4B-Instruct")]
        for key, name in (("pi", "PI_v3-Bridge-RT_1"), ("groot", "GR00T-Bridge-RT-1"), ("oft", "OFT-Bridge-RT-1")):
            flags += [f"--{key}-checkpoint", str(assets / f"Qwen3VL-{name}")]
    else:
        module = "oxygen_models.qwen35.benchmark"
        flags = ["--model", str(assets / f"Qwen3.5-{args.size}"), "--require-fast-path"]
    flags += ["--output", str(output)]
    if args.smoke:
        if args.model == "xiaomi" and not args.check:
            flags += ["--max-decoding-steps", "5", "--steps-per-frame", "1", "--measured-repeats", "1",
                      "--measured-frames", "2", "--single-measured-frames", "2", "--single-extended-frames", "2"]
        elif args.model == "starvla":
            flags += ["--language-tokens", "5", "--steps-per-frame", "1", "--repeats", "1",
                      "--measured-frames", "2", "--single-measured-frames", "2", "--single-adaptive-frames", "2"]
        elif args.model == "qwen35":
            flags += ["--smoke", "--decode-steps", "5", "--steps-per-frame", "1", "--repeats", "1"]
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.model == "qwen35":
        cmd = [str(python), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", "--module", module]
    else:
        cmd = [str(python), "-m", module]
    return cmd + flags + extra


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("xiaomi", "starvla", "qwen35"))
    parser.add_argument("--envs", type=Path, default=root / ".envs")
    parser.add_argument("--assets", type=Path, default=root / "assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=("0.8B", "2B", "4B", "9B"), default="0.8B")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check", action="store_true", help="Run Xiaomi request/cache checks.")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if args.check and args.model != "xiaomi":
        parser.error("--check is Xiaomi-specific; StarVLA and Qwen3.5 check the runtime during each run")
    cmd = command(args, extra)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    if args.output.exists() or args.output.with_suffix(".jsonl").exists():
        parser.error("Output exists; choose a new --output path")
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, TOKENIZERS_PARALLELISM="false")
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
