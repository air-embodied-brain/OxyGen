import dataclasses
import json
from pathlib import Path

import tyro

from experiments.libero_language_adapter import data


@dataclasses.dataclass(frozen=True)
class Args:
    annotation_root: Path
    dataset_root: Path
    output: Path
    seed: int = 7
    validation_episodes_per_task: int = 5
    uniform_stride: int = 8
    language_target: str = "next"


def main(args: Args) -> None:
    train, validation, manifest = data.build_episode_split(
        args.annotation_root,
        args.dataset_root,
        seed=args.seed,
        validation_episodes_per_task=args.validation_episodes_per_task,
        uniform_stride=args.uniform_stride,
        language_target=args.language_target,
    )
    data.save_split(args.output, train, validation, manifest)
    print(json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
