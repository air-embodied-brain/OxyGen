#!/usr/bin/env python3
"""Validate request progress and cache integrity under staggered continuous batching."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from oxygen_runtime import (
    batched_language_steps,
    init_language_from_prefix,
    language_step,
    prefill_vlm,
    staticize_language_state,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--max-decoding-steps", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--base-image", type=Path, required=True)
    parser.add_argument("--wrist-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_images(base_image: Path, wrist_image: Path):
    return [
        Image.open(base_image).convert("RGB"),
        Image.open(wrist_image).convert("RGB"),
    ]


def make_prompt() -> str:
    return (
        "<|im_start|>user\n"
        "The following observations are captured from multiple views.\n"
        "# Base View\n<|vision_start|><|image_pad|><|vision_end|>\n"
        "# Left-Wrist View\n<|vision_start|><|image_pad|><|vision_end|>\n"
        "Generate robot actions for the task:\n"
        "Pick up the black bowl between the plate and the ramekin and place it on the plate. "
        "/no_cot<|im_end|>\n"
        "<|im_start|>assistant\n<cot></cot><|im_end|>\n"
    )


def main():
    args = parse_args()
    if args.random_init:
        torch.manual_seed(0)
    device = torch.device(args.device)
    if args.random_init:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
        config._attn_implementation = "flash_attention_2"
        model = AutoModel.from_config(config, trust_remote_code=True).to(dtype=torch.bfloat16)
    else:
        model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True,
            attn_implementation="flash_attention_2", dtype=torch.bfloat16, local_files_only=True)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    images = load_images(args.base_image, args.wrist_image)

    def process(prompt):
        return dict(processor(
            text=[prompt], images=images, videos=None, padding=True,
            return_tensors="pt",
        ).to(device))

    # Match the paper sweep workload: each frame receives the same prompt and
    # tensor shapes, while request ages differ because arrivals are staggered.
    prompts = [make_prompt() for _ in range(args.num_requests)]
    prompt_inputs = [process(prompt) for prompt in prompts]
    prefix_lengths = [int(item["attention_mask"].shape[-1]) for item in prompt_inputs]
    if len(set(prefix_lengths)) != 1:
        raise ValueError(f"Prefix lengths must match for batching: {prefix_lengths}")

    active = {}
    completed = {}
    frame_trace = []
    drain_frames = math.ceil(args.max_decoding_steps / args.steps_per_frame) - 1
    total_frames = args.num_requests + drain_frames
    for frame in range(total_frames):
        if frame < args.num_requests:
            prefix = prefill_vlm(model, dict(prompt_inputs[frame]))
            state = init_language_from_prefix(processor, prefix, stop_on_eos=False)
            staticize_language_state(state, args.max_decoding_steps)
            active[f"req_{frame}"] = state

        request_ids = list(active)
        states = [active[request_id] for request_id in request_ids]
        batched_language_steps(model, states, args.steps_per_frame)
        frame_trace.append({
            "frame": frame,
            "request_ids": request_ids,
            "generated_lengths": {
                request_id: len(active[request_id].generated_ids)
                for request_id in request_ids
            },
        })
        for request_id in request_ids:
            if active[request_id].finished:
                completed[request_id] = active.pop(request_id).generated_ids

    if active:
        raise RuntimeError(f"Requests did not finish: {list(active)}")

    request_results = []
    for index, prompt_inputs_item in enumerate(prompt_inputs):
        prefix = prefill_vlm(model, dict(prompt_inputs_item))
        reference = init_language_from_prefix(processor, prefix, stop_on_eos=False)
        for _ in range(args.max_decoding_steps):
            language_step(model, reference)
        request_id = f"req_{index}"
        actual_ids = completed[request_id]
        text = processor.batch_decode([actual_ids], skip_special_tokens=True)[0]
        exact_match = actual_ids == reference.generated_ids
        readable = (
            len(text.strip()) >= 20
            and any(character.isalpha() for character in text)
            and "<cot></cot>" not in text
        )
        request_results.append({
            "request_id": request_id,
            "prefix_tokens": prefix_lengths[index],
            "generated_token_ids": actual_ids,
            "text": text,
            "exact_match_independent": exact_match,
            "readable_nonempty": readable,
        })
        print(json.dumps(request_results[-1], ensure_ascii=False), flush=True)

    from xiaomi_implementation_audit import implementation_audit
    mechanics = implementation_audit(model, processor, prompt_inputs[0])
    all_lengths_correct = all(len(item["generated_token_ids"]) == args.max_decoding_steps for item in request_results)
    payload = {
        "validation_policy": "performance_implementation_v2",
        "implementation_audit": mechanics,
        "all_request_lengths_correct": all_lengths_correct,
        "checkpoint": args.checkpoint,
        "num_requests": args.num_requests,
        "max_decoding_steps": args.max_decoding_steps,
        "steps_per_frame": args.steps_per_frame,
        "base_image": str(args.base_image),
        "wrist_image": str(args.wrist_image),
        "common_prefix_has_expert_suffix": False,
        "all_exact_match_independent": all(
            item["exact_match_independent"] for item in request_results
        ),
        "all_readable_nonempty": all(
            item["readable_nonempty"] for item in request_results
        ),
        "requests": request_results,
        "frame_trace": frame_trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if not mechanics["passed"] or not all_lengths_correct:
        raise RuntimeError("Continuous batching implementation/progress audit failed")


if __name__ == "__main__":
    main()
