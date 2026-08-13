"""Greedy-scan multiple adapter checkpoints without repeatedly loading the base model."""

# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
from pathlib import Path

import flax.nnx as nnx
import numpy as np

from experiments.libero_language_adapter import data
from experiments.libero_language_adapter import evaluate
from experiments.libero_language_adapter import train
from openpi.models import tokenizer as tokenizer_lib


def _adapter_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(path) for path in sorted(glob.glob(value))] if any(c in value for c in "*?[") else [Path(value)]
        paths.extend(matches)
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--suffix-len", type=int, default=20)
    parser.add_argument("--suffix-seed", default="Subtask: ")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_paths = _adapter_paths(args.adapters)
    _, validation_refs, _ = train.load_split(args.split)
    tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=data.PROMPT_TOKEN_LEN)
    dataset = data.LiberoLanguageDataset(
        validation_refs,
        norm_stats_path=args.norm_stats,
        prompt_tokenizer=tokenizer,
        action_dim=32,
        suffix_len=args.suffix_len,
        suffix_seed=args.suffix_seed,
    )
    model_args = train.Args(
        split=args.split,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        output_dir=args.output_dir,
        adapter_type="suffix_lora",
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
        suffix_seed=args.suffix_seed,
    )
    _, model = train.load_model(model_args)
    model_def, state = nnx.split(model)
    seed_tokens = tokenizer.tokenize_language_seed(args.suffix_seed)
    (
        _,
        prefill_incremental,
        _,
        _,
        suffix_step_active,
        _,
        suffix_block_active,
        _,
        _,
        _,
        _,
    ) = evaluate._make_functions(model_def, suffix_len=args.suffix_len, seed_len=len(seed_tokens))
    selected = evaluate._select_samples(validation_refs, per_task=args.samples_per_task, seed=args.seed)
    index_by_ref = {
        (sample.suite, sample.task, sample.demo, sample.frame): index for index, sample in enumerate(validation_refs)
    }
    observations = [
        data.collate([dataset[index_by_ref[(sample.suite, sample.task, sample.demo, sample.frame)]]])[0]
        for sample in selected
    ]

    summaries = []
    for adapter_path in adapter_paths:
        evaluate.apply_adapter(state, adapter_path)
        predictions = []
        for sample, observation in zip(selected, observations, strict=True):
            prediction = evaluate._generate(
                state,
                observation,
                tokenizer,
                prefill_incremental,
                suffix_step_active,
                suffix_block_active,
                suffix_len=args.suffix_len,
                suffix_seed=args.suffix_seed,
            )
            predictions.append(
                {
                    **dataclasses.asdict(sample),
                    "prediction": prediction,
                    "normalized_exact": evaluate.normalize_text(prediction) == evaluate.normalize_text(sample.target),
                    "word_f1": evaluate.word_f1(prediction, sample.target),
                }
            )
        exact = float(np.mean([record["normalized_exact"] for record in predictions]))
        word_f1 = float(np.mean([record["word_f1"] for record in predictions]))
        name = f"{adapter_path.parent.name}__{adapter_path.stem}"
        output_path = args.output_dir / f"{name}.jsonl"
        with output_path.open("w") as stream:
            for record in predictions:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        summary = {
            "adapter": str(adapter_path),
            "samples": len(predictions),
            "normalized_exact": exact,
            "word_f1": word_f1,
            "predictions": str(output_path),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    summaries.sort(key=lambda result: (result["normalized_exact"], result["word_f1"]), reverse=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    dataset.close()


if __name__ == "__main__":
    main()
