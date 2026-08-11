from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
import random

import h5py
import jax
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import libero_policy
from openpi.shared import normalize

DISCRETE_STATE_INPUT = False
PROMPT_TOKEN_LEN = 100


@dataclasses.dataclass(frozen=True)
class SampleRef:
    suite: str
    task: str
    demo: str
    frame: int
    instruction: str
    target: str
    annotation_path: str
    dataset_path: str


@dataclasses.dataclass(frozen=True)
class SplitManifest:
    seed: int
    validation_episodes_per_task: int
    uniform_stride: int
    train_episodes: int
    validation_episodes: int
    train_samples: int
    validation_samples: int
    validation_demos: dict[str, list[str]]


def _stable_seed(seed: int, suite: str, task: str) -> int:
    digest = hashlib.sha256(f"{seed}:{suite}:{task}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _load_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sample_frames(rows: Sequence[dict], uniform_stride: int) -> list[int]:
    selected = {0, len(rows) - 1}
    selected.update(range(0, len(rows), uniform_stride))
    previous = None
    for index, row in enumerate(rows):
        target = row["language"]["next"]
        if previous is not None and target != previous:
            selected.add(index)
            selected.add(index - 1)
        previous = target
    return sorted(selected)


def build_episode_split(
    annotation_root: Path,
    dataset_root: Path,
    *,
    seed: int = 7,
    validation_episodes_per_task: int = 5,
    uniform_stride: int = 8,
) -> tuple[list[SampleRef], list[SampleRef], SplitManifest]:
    annotation_paths = sorted(annotation_root.glob("*/*/demo_*.jsonl"))
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in annotation_paths:
        grouped.setdefault((path.parent.parent.name, path.parent.name), []).append(path)
    if len(grouped) != 40:
        raise ValueError(f"Expected 40 LIBERO tasks, found {len(grouped)}")

    train_samples: list[SampleRef] = []
    validation_samples: list[SampleRef] = []
    validation_demos: dict[str, list[str]] = {}
    train_episode_count = 0
    validation_episode_count = 0

    for (suite, task), task_paths in sorted(grouped.items()):
        if len(task_paths) != 50:
            raise ValueError(f"Expected 50 demonstrations for {suite}/{task}, found {len(task_paths)}")
        paths = sorted(task_paths, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
        rng = random.Random(_stable_seed(seed, suite, task))
        validation_names = set(rng.sample([path.stem for path in paths], validation_episodes_per_task))
        validation_demos[f"{suite}/{task}"] = sorted(validation_names)
        dataset_path = dataset_root / suite / f"{task}_demo.hdf5"
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        for path in paths:
            rows = _load_rows(path)
            demo = path.stem
            is_validation = demo in validation_names
            destination = validation_samples if is_validation else train_samples
            if is_validation:
                validation_episode_count += 1
            else:
                train_episode_count += 1
            for frame in _sample_frames(rows, uniform_stride):
                row = rows[frame]
                if row["frame"] != frame or row["demo"] != demo:
                    raise ValueError(f"Misaligned annotation row in {path} at frame {frame}")
                destination.append(
                    SampleRef(
                        suite=suite,
                        task=task,
                        demo=demo,
                        frame=frame,
                        instruction=row["task_instruction"],
                        target=row["language"]["next"],
                        annotation_path=str(path),
                        dataset_path=str(dataset_path),
                    )
                )

    manifest = SplitManifest(
        seed=seed,
        validation_episodes_per_task=validation_episodes_per_task,
        uniform_stride=uniform_stride,
        train_episodes=train_episode_count,
        validation_episodes=validation_episode_count,
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        validation_demos=validation_demos,
    )
    return train_samples, validation_samples, manifest


class LiberoLanguageDataset:
    def __init__(
        self,
        samples: Sequence[SampleRef],
        *,
        norm_stats_path: Path,
        prompt_tokenizer: tokenizer_lib.PaligemmaTokenizer,
        action_dim: int,
        suffix_len: int,
        suffix_seed: str = "Subtask: ",
    ):
        self.samples = list(samples)
        self.suffix_len = suffix_len
        self.suffix_seed = suffix_seed
        self._tokenizer = prompt_tokenizer
        self._files: dict[str, h5py.File] = {}
        norm_stats = normalize.load(norm_stats_path.parent)
        self._transform = transforms.compose(
            [
                libero_policy.LiberoInputs(model_type=model_lib.ModelType.PI05_O2),
                transforms.Normalize(norm_stats, use_quantiles=True),
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(
                    prompt_tokenizer,
                    discrete_state_input=DISCRETE_STATE_INPUT,
                ),
                transforms.PadStatesAndActions(action_dim),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def close(self) -> None:
        for dataset in self._files.values():
            dataset.close()
        self._files.clear()

    def _file(self, path: str) -> h5py.File:
        if path not in self._files:
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def __getitem__(self, index: int):
        sample = self.samples[index]
        demo = self._file(sample.dataset_path)[f"data/{sample.demo}"]
        if sample.frame >= len(demo["actions"]):
            raise IndexError(f"Frame {sample.frame} is outside {sample.dataset_path}/{sample.demo}")
        obs = demo["obs"]
        raw = {
            "observation/image": np.ascontiguousarray(obs["agentview_rgb"][sample.frame][::-1, ::-1]),
            "observation/wrist_image": np.ascontiguousarray(obs["eye_in_hand_rgb"][sample.frame][::-1, ::-1]),
            "observation/state": np.concatenate(
                [obs["ee_states"][sample.frame], obs["gripper_states"][sample.frame]]
            ).astype(np.float32),
            "prompt": sample.instruction,
        }
        transformed = self._transform(raw)
        observation = model_lib.Observation.from_dict(transformed)
        suffix = self._tokenizer.tokenize_language_suffix(
            self.suffix_seed,
            sample.target,
            max_len=self.suffix_len,
        )
        return observation, *suffix


def collate(samples):
    return jax.tree.map(lambda *values: np.stack(values), *samples)


def batches(
    dataset: LiberoLanguageDataset,
    *,
    batch_size: int,
    seed: int,
) -> Iterator:
    rng = np.random.default_rng(seed)
    while True:
        indices = rng.integers(0, len(dataset), size=batch_size)
        yield collate([dataset[int(index)] for index in indices])


def save_split(
    output_path: Path,
    train_samples: Sequence[SampleRef],
    validation_samples: Sequence[SampleRef],
    manifest: SplitManifest,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": dataclasses.asdict(manifest),
        "train": [dataclasses.asdict(sample) for sample in train_samples],
        "validation": [dataclasses.asdict(sample) for sample in validation_samples],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
