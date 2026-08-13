from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import numpy as np
import tyro

from experiments.libero_language_adapter import data
from experiments.libero_language_adapter import evaluate
from experiments.libero_language_adapter import train
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import policy as policy_lib


@dataclasses.dataclass(frozen=True)
class Args:
    split: Path
    checkpoint: Path
    norm_stats: Path
    adapter: Path
    output: Path
    sample_index: int = 0
    rank: int = 16
    alpha: float = 16.0
    seed: int = 7
    suffix_len: int = 20
    suffix_seed: str = "Subtask: "


def main(args: Args) -> None:
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
    sample = validation_refs[args.sample_index]
    observation = dataset[args.sample_index][0]

    model_args = train.Args(
        split=args.split,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        output_dir=args.output.parent,
        adapter_type="suffix_lora",
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
        suffix_seed=args.suffix_seed,
    )
    _, model = train.load_model(model_args)
    graphdef, state = nnx.split(model)
    evaluate.apply_adapter(state, args.adapter)
    model = nnx.merge(graphdef, state)

    policy = policy_lib.Policy(
        model,
        rng=jax.random.key(args.seed),
        sample_kwargs={"num_steps": 10},
        language_seed=args.suffix_seed,
    )
    manager = policy.init_continuous_batching()
    raw_observation = {
        key: jax.tree.map(np.asarray, value) for key, value in observation.to_dict().items() if value is not None
    }
    result = policy.infer_text_actions_continuous_batch(
        [raw_observation],
        cache_manager=manager,
        steps_per_frame=args.suffix_len,
        num_action_steps=10,
        max_decoding_steps=args.suffix_len,
        temperature=0.0,
        PALIGEMMA_EOS_TOKEN=tokenizer.eos_token_id,
    )[0]

    payload = {
        "sample": dataclasses.asdict(sample),
        "target": sample.target,
        "prediction": result["text"].strip(),
        "normalized_exact": evaluate.normalize_text(result["text"]) == evaluate.normalize_text(sample.target),
        "word_f1": evaluate.word_f1(result["text"], sample.target),
        "action_shape": list(np.asarray(result["actions"]).shape),
        "request_finished": bool(result["is_finished"]),
        "active_requests_after_call": len(manager.active_states),
        "timing": result["policy_timing"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
