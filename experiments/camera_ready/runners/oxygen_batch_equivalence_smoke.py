#!/usr/bin/env python3
"""Check heterogeneous-age batched language decode against independent decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

from oxygen_runtime import (
    batched_language_steps,
    init_language_from_prefix,
    language_step,
    prefill_vlm,
    staticize_language_state,
)
from oxygen_sweep import build_inputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def new_state(model, processor, inputs):
    prefix = prefill_vlm(model, dict(inputs))
    return init_language_from_prefix(processor, prefix, stop_on_eos=False)


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    _, _, language_inputs, _, _, _, _ = build_inputs(
        processor, model, device, instruction_repeat=1
    )

    reference_old = new_state(model, processor, language_inputs)
    reference_new = new_state(model, processor, language_inputs)
    batched_old = new_state(model, processor, language_inputs)
    batched_new = new_state(model, processor, language_inputs)
    staticize_language_state(batched_old, 16)
    staticize_language_state(batched_new, 16)

    for _ in range(3):
        language_step(model, reference_old)
    batched_language_steps(model, [batched_old], 3)

    for _ in range(4):
        language_step(model, reference_old)
        language_step(model, reference_new)
    batched_language_steps(model, [batched_old, batched_new], 4)

    result = {
        "old_reference_ids": reference_old.generated_ids,
        "old_batched_ids": batched_old.generated_ids,
        "new_reference_ids": reference_new.generated_ids,
        "new_batched_ids": batched_new.generated_ids,
        "old_exact": reference_old.generated_ids == batched_old.generated_ids,
        "new_exact": reference_new.generated_ids == batched_new.generated_ids,
        "old_next_token_exact": bool(
            torch.equal(reference_old.next_token, batched_old.next_token)
        ),
        "new_next_token_exact": bool(
            torch.equal(reference_new.next_token, batched_new.next_token)
        ),
    }
    if not all(
        result[key]
        for key in ("old_exact", "new_exact", "old_next_token_exact", "new_next_token_exact")
    ):
        raise RuntimeError(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
