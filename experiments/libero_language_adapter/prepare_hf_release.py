"""Stage the selected visual-memory adapter and annotations for Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


DEFAULT_BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transfer(source: Path, destination: Path, *, hardlink: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Release directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _model_card(*, base_checkpoint: str, metrics: dict) -> str:
    evaluation = metrics["evaluation"]
    performance = metrics["performance"]
    return f"""---
library_name: openpi
license: other
license_name: apache-2.0-and-gemma-terms
tags:
- robotics
- vision-language-action
- libero
- lora
---

# OxyGen pi0.5 LIBERO visual-memory suffix LoRA

This rank-16 LoRA is enabled only for the private autoregressive language
suffix. The observation, robot state, and task instruction are encoded by the
frozen base model, and the action expert reads the unchanged root KV cache.

- Base checkpoint: `{base_checkpoint}`
- Suffix seed: `Memory: `
- Maximum suffix length: 28 tokens
- Held-out token accuracy: {evaluation["teacher_forced_token_accuracy"]:.2%}
- Greedy exact match: {evaluation["normalized_exact"]:.2%}
- Greedy word F1: {evaluation["word_f1"]:.2%}
- Root-cache/action maximum absolute difference: 0 / 0
- Adapter overhead: {performance["adapter_overhead_per_median_request_ms"]:.2f} ms
- Shared prefix forward: {performance["prefix_prefill_ms"]:.2f} ms

Use this adapter with the `libero-language-adapter` feature in OxyGen. The
repository documents training, evaluation, continuous batching, and the
blocking openpi baseline.

OxyGen/openpi code is Apache 2.0. Gemma components remain subject to the Gemma
terms distributed with openpi. The base checkpoint is not included here.
"""


def _dataset_card(*, qa: dict) -> str:
    return f"""---
license: cc-by-4.0
task_categories:
- robotics
language:
- en
tags:
- libero
- robot-learning
- visual-memory
---

# LIBERO predicate and visual-memory annotations

This release contains predicate-derived annotations for the complete training
split of LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10.

- Demonstrations: {qa["files"]:,}
- Frames: {qa["frames"]:,}
- Demonstrations reaching simulator success: {qa["files_reaching_success"]:,}
- QA issues: {qa["files_with_issues"]}

Each JSONL row stores raw goal/auxiliary predicates and independent language
fields for completed progress, remaining progress, the next demonstrated
transition, and visual memory. `visual_memory` uses only stable goal predicates
completed in the current observation; it does not use future frames.

The original LIBERO HDF5 demonstrations are not redistributed. Download them
from the official LIBERO dataset to train the adapter or reconstruct images and
robot state. The derived annotations use the official dataset's CC BY 4.0
license; retain LIBERO attribution when redistributing or modifying them.
"""


def stage_release(
    *,
    adapter: Path,
    annotation_root: Path,
    summary_path: Path,
    output_root: Path,
    base_checkpoint: str,
    hardlink: bool,
) -> dict:
    _prepare_output(output_root)
    metrics = _read_json(summary_path)
    qa = _read_json(annotation_root / "qa.json")
    if qa["files"] != 2_000 or qa["frames"] != 338_575 or qa["files_with_issues"] != 0:
        raise ValueError("Annotation QA does not match the selected visual-memory release")

    model_root = output_root / "model"
    dataset_root = output_root / "dataset"
    model_root.mkdir()
    dataset_root.mkdir()

    adapter_destination = model_root / adapter.name
    _transfer(adapter, adapter_destination, hardlink=hardlink)
    adapter_hash = _sha256(adapter_destination)
    adapter_config = {
        "adapter_type": "suffix_lora",
        "alpha": 16,
        "base_checkpoint": base_checkpoint,
        "language_target": "visual_memory",
        "rank": 16,
        "sha256": adapter_hash,
        "suffix_length": metrics["checkpoint"]["suffix_length"],
        "suffix_seed": metrics["checkpoint"]["suffix_seed"],
        "training_step": metrics["checkpoint"]["step"],
    }
    _write_json(model_root / "adapter_config.json", adapter_config)
    _write_json(model_root / "metrics.json", metrics)
    (model_root / "README.md").write_text(
        _model_card(base_checkpoint=base_checkpoint, metrics=metrics),
        encoding="utf-8",
    )

    annotation_paths = sorted(annotation_root.glob("*/*/demo_*.jsonl"))
    if len(annotation_paths) != 2_000:
        raise ValueError(f"Expected 2,000 annotation files, found {len(annotation_paths)}")
    for source in annotation_paths:
        relative = source.relative_to(annotation_root)
        _transfer(source, dataset_root / "annotations" / relative, hardlink=hardlink)
    for name in ("qa.json", "pickup_audit.json"):
        _transfer(annotation_root / name, dataset_root / name, hardlink=hardlink)

    portable_manifest = {
        "annotation_files": len(annotation_paths),
        "frames": qa["frames"],
        "language_fields": ["completed", "remaining", "next", "visual_memory"],
        "qa_issues": qa["files_with_issues"],
        "schema_version": "libero_all_v6",
        "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    }
    _write_json(dataset_root / "manifest.json", portable_manifest)
    (dataset_root / "README.md").write_text(_dataset_card(qa=qa), encoding="utf-8")

    release_manifest = {
        "adapter": {
            "bytes": adapter_destination.stat().st_size,
            "file": f"model/{adapter_destination.name}",
            "sha256": adapter_hash,
        },
        "annotations": portable_manifest,
        "base_checkpoint": base_checkpoint,
        "copy_mode": "hardlink" if hardlink else "copy",
    }
    _write_json(output_root / "release_manifest.json", release_manifest)
    return release_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--hardlink", action="store_true")
    args = parser.parse_args()
    manifest = stage_release(
        adapter=args.adapter,
        annotation_root=args.annotation_root,
        summary_path=args.summary,
        output_root=args.output_root,
        base_checkpoint=args.base_checkpoint,
        hardlink=args.hardlink,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
