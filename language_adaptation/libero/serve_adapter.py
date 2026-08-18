from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import flax.nnx as nnx
import jax
import tyro

from language_adaptation.libero import adapter
from language_adaptation.libero import data
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import libero_policy
from openpi.policies import policy as policy_lib
from openpi.serving import websocket_policy_server
from openpi.shared import normalize


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    norm_stats: Path
    rank: int = 16
    alpha: float = 16.0
    adapter: Path | None = None
    port: int = 8011
    seed: int = 7
    stop_on_eos: bool = True
    request_mode: Literal["resume_until_finished", "new_each_call"] = "resume_until_finished"
    steps_per_frame: int = 5
    max_decoding_steps: int = 20
    language_seed: str = "Memory: "
    temperature: float = 0.1
    execution: Literal["oxygen", "blocking_baseline"] = "oxygen"


def main(args: Args) -> None:
    _, model = adapter.load_model(args.checkpoint, rank=args.rank, alpha=args.alpha, seed=args.seed)
    graphdef, state = nnx.split(model)
    if args.adapter is not None:
        adapter.apply(state, args.adapter)
    model = nnx.merge(graphdef, state)

    norm_stats = normalize.load(args.norm_stats.parent)
    tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=data.PROMPT_TOKEN_LEN)
    policy = policy_lib.Policy(
        model,
        rng=jax.random.key(args.seed),
        transforms=[
            libero_policy.LiberoInputs(model_type=model_lib.ModelType.PI05_O2),
            transforms.Normalize(norm_stats, use_quantiles=True),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(
                tokenizer,
                discrete_state_input=data.DISCRETE_STATE_INPUT,
            ),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[
            transforms.Unnormalize(norm_stats, use_quantiles=True),
            libero_policy.LiberoOutputs(),
        ],
        sample_kwargs={"num_steps": 10},
        metadata={
            "policy_seed": args.seed,
            "adapter": None if args.adapter is None else str(args.adapter.resolve()),
            "adapter_rank": args.rank,
            "effective_infer_api": ("continuous_batching" if args.execution == "oxygen" else args.execution),
            "execution": args.execution,
            "text_stop_condition": "eos" if args.stop_on_eos else "fixed_length",
            "language_steps_per_frame": (None if args.execution == "blocking_baseline" else args.steps_per_frame),
            "language_max_decoding_steps": args.max_decoding_steps,
            "language_temperature": args.temperature,
            "language_seed": args.language_seed,
        },
        language_seed=args.language_seed,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
        infer_api="continuous_batching" if args.execution == "oxygen" else args.execution,
        inference_kwargs={
            "max_decoding_steps": args.max_decoding_steps,
            "temperature": args.temperature,
            "PALIGEMMA_EOS_TOKEN": tokenizer.eos_token_id if args.stop_on_eos else -1,
            **({"steps_per_frame": args.steps_per_frame} if args.execution != "blocking_baseline" else {}),
        },
        continuous_batching_request_mode=args.request_mode,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
