from __future__ import annotations

import dataclasses
from pathlib import Path

import flax.nnx as nnx
import tyro

from experiments.libero_language_adapter import data
from experiments.libero_language_adapter import evaluate
from experiments.libero_language_adapter import train
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
    rank: int
    adapter: Path | None = None
    port: int = 8011
    seed: int = 7


def main(args: Args) -> None:
    model_args = train.Args(
        split=Path("unused"),
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        output_dir=Path("unused"),
        rank=args.rank,
        seed=args.seed,
    )
    _, model = train.load_model(model_args)
    graphdef, state = nnx.split(model)
    if args.adapter is not None:
        evaluate.apply_adapter(state, args.adapter)
    model = nnx.merge(graphdef, state)

    norm_stats = normalize.load(args.norm_stats.parent)
    tokenizer = tokenizer_lib.PaligemmaTokenizer(max_len=200)
    policy = policy_lib.Policy(
        model,
        rng=None,
        transforms=[
            libero_policy.LiberoInputs(model_type=model_lib.ModelType.PI05),
            transforms.Normalize(norm_stats, use_quantiles=False),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(
                tokenizer,
                discrete_state_input=data.DISCRETE_STATE_INPUT,
            ),
            transforms.PadStatesAndActions(32),
        ],
        output_transforms=[
            transforms.Unnormalize(norm_stats, use_quantiles=False),
            libero_policy.LiberoOutputs(),
        ],
        sample_kwargs={"num_steps": 10},
        metadata={
            "policy_seed": args.seed,
            "adapter": None if args.adapter is None else str(args.adapter.resolve()),
            "adapter_rank": args.rank,
            "effective_infer_api": "sample_actions",
        },
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
