#!/usr/bin/env python3
"""Measure realistic StarVLA expert combinations on one Qwen3-VL-4B prefix."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

XIAOMI_SCRIPTS = str(Path(__file__).resolve().parent)
if XIAOMI_SCRIPTS not in sys.path:
    sys.path.insert(0, XIAOMI_SCRIPTS)
from oxygen_runtime import (  # noqa: E402
    PrefixState,
    _snapshot_cache,
    init_language_from_prefix,
    language_step,
)
from starVLA.model.framework.base_framework import baseframework  # noqa: E402
from starVLA.model.modules.vlm import get_vlm_model  # noqa: E402


class RuntimeAdapter:
    def __init__(self, interface):
        self.vlm = interface
        self.device = interface.model.device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--pi-checkpoint", required=True)
    parser.add_argument("--groot-checkpoint", required=True)
    parser.add_argument("--oft-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--language-tokens", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--base-language-backbone",
        action="store_true",
        help="Replace the PI-finetuned VLM with the base VLM; systems-only for action heads.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def example():
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((22, 66, 86, 132), fill=(190, 35, 35))
    draw.rectangle((134, 62, 207, 145), fill=(35, 70, 180))
    return image, "Move the red block next to the blue block."


def build_inputs(interface, instruction, image, oft_model=None):
    if oft_model is not None:
        tokens = oft_model.action_token * oft_model.chunk_len
        instruction = (
            instruction
            + f" Please predict the next {oft_model.chunk_len} robot actions: "
            + f"<action>{tokens}<action>."
        )
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ],
    }]]
    return interface.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(interface.model.device)


@torch.inference_mode()
def prefill(pi_model, inputs):
    outputs = pi_model.qwen_vl_interface(
        **inputs,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
        logits_to_keep=1,
    )
    outputs.attention_mask = inputs["attention_mask"]
    return PrefixState(
        outputs=outputs,
        action_kv=_snapshot_cache(outputs.past_key_values),
        prefix_length=int(inputs["attention_mask"].shape[-1]),
    )


@torch.inference_mode()
def run_language(adapter, processor, prefix, count):
    state = init_language_from_prefix(processor, prefix, stop_on_eos=False)
    for _ in range(count):
        language_step(adapter, state)
    return state


@torch.inference_mode()
def pi_action(model, prefix, steps):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model.action_model.num_inference_timesteps = steps
    hidden = list(prefix.outputs.hidden_states[-model.num_action_dit_layers:])
    hidden = model._project_vl_hidden_for_action(hidden)
    return model.action_model.predict_action(
        hidden, None, encoder_attention_mask=prefix.outputs.attention_mask.bool()
    )


@torch.inference_mode()
def groot_action(head, prefix, steps):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    head.num_inference_timesteps = steps
    state = torch.zeros((1, 1, 7), device=prefix.outputs.logits.device, dtype=torch.bfloat16)
    return head.predict_action(
        prefix.outputs.hidden_states[-1],
        state,
        encoder_attention_mask=prefix.outputs.attention_mask.bool(),
    )


@torch.inference_mode()
def oft_action(oft_model, prefix, input_ids):
    queries = oft_model._gather_action_token_embeddings(
        prefix.outputs.hidden_states[-1], input_ids, oft_model.action_token_id
    )
    return oft_model.action_model.predict_action(queries)


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = fn()
    torch.cuda.synchronize()
    return output, (time.perf_counter() - started) * 1000, torch.cuda.max_memory_allocated()


def main():
    args = parse_args()
    overrides = [
        f"framework.qwenvl.base_vlm={args.base_vlm}",
        "framework.qwenvl.attn_implementation=sdpa",
    ]
    pi = baseframework.from_pretrained(args.pi_checkpoint, config_overrides=overrides)
    pi = pi.to(torch.bfloat16).to(args.device).eval()

    groot_model = baseframework.from_pretrained(args.groot_checkpoint, config_overrides=overrides)
    groot_head = groot_model.action_model.to(torch.bfloat16).to(args.device).eval()
    del groot_model
    gc.collect()

    oft = baseframework.from_pretrained(args.oft_checkpoint, config_overrides=overrides)
    oft.action_model = oft.action_model.to(torch.bfloat16).to(args.device).eval()
    del oft.qwen_vl_interface
    gc.collect()

    if args.base_language_backbone:
        del pi.qwen_vl_interface
        gc.collect()
        torch.cuda.empty_cache()
        pi.qwen_vl_interface = get_vlm_model(config=pi.config)
        pi.qwen_vl_interface = pi.qwen_vl_interface.to(torch.bfloat16).to(args.device).eval()

    adapter = RuntimeAdapter(pi.qwen_vl_interface)
    image, instruction = example()
    inputs_plain = build_inputs(pi.qwen_vl_interface, instruction, image)
    inputs_oft = build_inputs(pi.qwen_vl_interface, instruction, image, oft)
    processor = pi.qwen_vl_interface.processor

    combinations = {
        2: (inputs_plain, ["language", "pi_v3"]),
        3: (inputs_plain, ["language", "pi_v3", "groot"]),
        4: (inputs_oft, ["language", "pi_v3", "groot", "oft"]),
    }
    records = []
    summaries = []
    for count, (inputs, names) in combinations.items():
        def execute(shared):
            prefixes = {}
            if shared:
                common = prefill(pi, inputs)
                prefixes = {name: common for name in names}
            else:
                prefixes = {name: prefill(pi, inputs) for name in names}
            outputs = {}
            outputs["language"] = run_language(
                adapter, processor, prefixes["language"], args.language_tokens
            )
            outputs["pi_v3"] = pi_action(pi, prefixes["pi_v3"], args.denoise_steps)
            if "groot" in names:
                outputs["groot"] = groot_action(
                    groot_head, prefixes["groot"], args.denoise_steps
                )
            if "oft" in names:
                outputs["oft"] = oft_action(oft, prefixes["oft"], inputs["input_ids"])
            return outputs

        for _ in range(args.warmup):
            execute(False)
            execute(True)
        latency = {"separate": [], "shared": []}
        outputs = {}
        for repeat in range(args.repeats):
            modes = ["separate", "shared"] if repeat % 2 == 0 else ["shared", "separate"]
            for mode in modes:
                result, elapsed, peak = timed(lambda mode=mode: execute(mode == "shared"))
                outputs[mode] = result
                latency[mode].append(elapsed)
                records.append({
                    "expert_count": count,
                    "experts": names,
                    "mode": mode,
                    "repeat": repeat,
                    "latency_ms": elapsed,
                    "peak_allocated_bytes": peak,
                })
        separate_ms = statistics.median(latency["separate"])
        shared_ms = statistics.median(latency["shared"])
        action_checks = {}
        for name in names:
            if name == "language":
                action_checks[name] = {
                    "text": processor.batch_decode(
                        [outputs["shared"][name].generated_ids], skip_special_tokens=True
                    )[0],
                    "token_ids_equal": (
                        outputs["shared"][name].generated_ids
                        == outputs["separate"][name].generated_ids
                    ),
                }
            else:
                left = outputs["separate"][name].float()
                right = outputs["shared"][name].float()
                action_checks[name] = {
                    "shape": list(right.shape),
                    "finite": bool(torch.isfinite(right).all()),
                    "max_abs": float((left - right).abs().max()),
                }
        summaries.append({
            "expert_count": count,
            "experts": names,
            "separate_median_ms": separate_ms,
            "shared_median_ms": shared_ms,
            "speedup": separate_ms / shared_ms,
            "correctness": action_checks,
            "prefix_tokens": int(inputs["input_ids"].shape[-1]),
        })
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "metadata": {
            "base_vlm": args.base_vlm,
            "pi_checkpoint": args.pi_checkpoint,
            "groot_checkpoint": args.groot_checkpoint,
            "oft_checkpoint": args.oft_checkpoint,
            "expert_semantics": {
                "language": "Qwen AR task explanation",
                "pi_v3": "layer-wise high-capacity flow action policy",
                "groot": "lighter last-layer flow action policy",
                "oft": "one-shot low-latency action fallback or proposal policy",
            },
            "quality_scope": "PI_v3 evaluated separately; K=3/4 are not jointly trained",
            "backbone_source": (
                "base_vlm_systems_only_action_heads_are_backbone_mismatched"
                if args.base_language_backbone
                else "pi_v3_finetuned_vlm"
            ),
            "dtype": "bfloat16",
            "attention": "sdpa",
            "compile": False,
            "cuda_graph": False,
        },
        "records": records,
        "summary": summaries,
    }, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
