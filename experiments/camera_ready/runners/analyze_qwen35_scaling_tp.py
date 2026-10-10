#!/usr/bin/env python3
"""Validate and summarize protocol-aligned Qwen3.5 + PI_v3 systems runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


SIZES = ("0.8B", "2B", "4B", "9B")
DECODE_STEPS = (10, 20, 30)
STEPS_PER_FRAME = (1, 5, 10)


def validate_cache_implementation(audit):
    """Check cache mechanics at matched batch size, not cross-batch argmax."""
    import math
    assert audit["input_tokens_exact"], "wrong input token/request association"
    for field in ("prestep_state_max_abs_error",
                  "mixed_age_vs_same_age_batch_max_abs_error",
                  "mixed_age_vs_same_age_rollout_state_max_abs_error"):
        assert audit[field] and all(math.isfinite(v) and v == 0.0 for v in audit[field].values()), field
    assert audit["mixed_age_vs_same_age_rollout_tokens_exact"], "matched-batch rollout mismatch"
    recycle = audit["slot_recycle_isolation"]
    for field in ("recycled_row_vs_fresh_prefix_max_abs_error", "untouched_rows_max_abs_error"):
        assert recycle[field] and all(math.isfinite(v) and v == 0.0 for v in recycle[field].values()), field
    assert audit["after_ages"] == [age + 1 for age in audit["ages"]]
    assert audit["rollout_after_ages"] == [age + audit["rollout_steps"] for age in audit["ages"]]


def load(path: Path):
    return json.loads(path.read_text())


def aggregate(payload):
    groups = defaultdict(list)
    for row in payload["records"]:
        groups[(row["mode"], row["decode_steps"], row["steps_per_frame"])].append(row)

    output = {}
    for key, rows in groups.items():
        output[key] = {
            "latency_ms": statistics.median(row["mean_frame_ms"] for row in rows),
            "repeat_means_ms": [row["mean_frame_ms"] for row in rows],
            "max_frame_cv": max(row.get("frame_cv", 0.0) for row in rows),
            "max_half_trend": max(
                row.get("half_mean_relative_difference", 0.0) for row in rows
            ),
            "frames_per_repeat": [len(row["frame_times_ms"]) for row in rows],
            "peak_allocated_bytes_per_rank": [
                max(values)
                for values in zip(*(row["peak_allocated_bytes_per_rank"] for row in rows))
            ],
        }
        breakdowns = [row.get("breakdown", row.get("mean_stage_ms")) for row in rows]
        if all(breakdowns):
            output[key]["breakdown_ms"] = {
                name: statistics.median(value[name] for value in breakdowns)
                for name in breakdowns[0]
            }
    return output


def expected_groups():
    groups = {("prefill_only", None, None)}
    groups.update(
        (mode, n, None)
        for mode in ("separate", "shared_kv")
        for n in DECODE_STEPS
    )
    groups.update(
        ("continuous_batching", n, k)
        for n in DECODE_STEPS
        for k in STEPS_PER_FRAME
        if n % k == 0
    )
    return groups


def validate(label: str, payload):
    size, tp_text = label.rsplit("-TP", 1)
    assert size in SIZES
    expected_world_size = int(tp_text)
    metadata = payload["metadata"]

    assert metadata["world_size"] == expected_world_size
    assert metadata["backend"] == (
        "native_hf_tp" if expected_world_size > 1 else "single_gpu"
    )
    assert (metadata["tp_plan"] is not None) == (expected_world_size > 1)
    assert metadata["denoise_steps"] == 4
    assert metadata["dtype"] == "bfloat16"
    assert metadata["attention"] == "sdpa"
    assert metadata["sdpa_flash_enabled"] is True
    assert metadata["compile"] is False
    assert metadata["cuda_graph"] is False
    assert metadata["fast_path_required"] is True
    assert metadata["fast_path_available"] is True
    assert metadata["fast_path_packages"] == {
        "flash-linear-attention": "0.5.1",
        "causal-conv1d": "1.6.2.post1",
    }
    assert metadata["fast_path_functions"] == {
        "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn",
        "causal_conv1d_update": "causal_conv1d.causal_conv1d_interface.causal_conv1d_update",
        "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule": (
            "fla.ops.gated_delta_rule.fused_recurrent."
            "fused_recurrent_gated_delta_rule"
        ),
    }
    assert metadata["compiler_state_before"]["inductor_generated_kernel_count"] == 0
    assert metadata["compiler_state_after"]["inductor_generated_kernel_count"] == 0
    assert metadata["action_horizon"] == 16
    assert metadata["action_dim"] == metadata["state_dim"] == 7
    assert metadata["action_num_dit_layers"] == metadata["vlm_num_hidden_layers"]
    assert metadata["baseline_measured_frames"] == 10
    assert metadata["continuous_measured_frames"] == 50
    assert metadata["repeats"] == 3

    audit = metadata["mixed_age_cache_audit"]
    validate_cache_implementation(audit)

    groups = defaultdict(list)
    for row in payload["records"]:
        key = (row["mode"], row["decode_steps"], row["steps_per_frame"])
        groups[key].append(row)
        if row["mode"] == "continuous_batching":
            assert len(row["frame_times_ms"]) == 50
            assert row["completed_requests"] == 50
            assert row["avg_batch_size"] == row["decode_steps"] // row["steps_per_frame"]
            assert row["all_completed_lengths_correct"]
            assert row["all_completed_text_nonempty"]
            assert row["action_finite"]
            assert row["action_shape"] == [1, 16, 7]
        elif row["mode"] in ("separate", "shared_kv"):
            assert len(row["frame_times_ms"]) in (10, 30)
            assert row["extended_to_30_frames"] == (len(row["frame_times_ms"]) == 30)
            assert row["action_finite"]
            assert row["action_shape"] == [1, 16, 7]
            assert len(row["tokens"]) == row["decode_steps"]
            assert row["text"].strip()
        elif row["mode"] == "prefill_only":
            assert len(row["frame_times_ms"]) in (10, 30)
            assert row["extended_to_30_frames"] == (len(row["frame_times_ms"]) == 30)
        else:
            raise AssertionError(f"unexpected mode: {row['mode']}")

    assert set(groups) == expected_groups()
    assert all(
        sorted(row["repeat"] for row in rows) == [0, 1, 2]
        for rows in groups.values()
    )
    assert len(payload["records"]) == 48

    for n in DECODE_STEPS:
        single_rows = [
            row
            for row in payload["records"]
            if row["mode"] in ("separate", "shared_kv") and row["decode_steps"] == n
        ]
        assert len({tuple(row["tokens"]) for row in single_rows}) == 1
    action_checksums = {
        row["action_checksum"]
        for row in payload["records"]
        if row["mode"] in ("separate", "shared_kv", "continuous_batching")
    }
    assert len(action_checksums) == 1


def parse_run(value: str):
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from exc
    if not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payloads = {label: load(path) for label, path in args.run}
    assert len(payloads) == len(args.run), "duplicate run label"
    for label, payload in payloads.items():
        validate(label, payload)

    signatures_by_size = defaultdict(list)
    for label, payload in payloads.items():
        size, _ = label.rsplit("-TP", 1)
        signatures_by_size[size].append(payload["metadata"]["input_signature"])
    assert all(
        all(signature == signatures[0] for signature in signatures)
        for signatures in signatures_by_size.values()
    ), "TP configurations of the same model size did not receive identical inputs"
    prefix_tokens_by_size = {
        size: signatures[0]["input_shapes"]["input_ids"][-1]
        for size, signatures in signatures_by_size.items()
    }

    aggregates = {label: aggregate(payload) for label, payload in payloads.items()}
    rows = []
    for label, values in aggregates.items():
        for n in DECODE_STEPS:
            baseline = values[("separate", n, None)]["latency_ms"]
            shared = values[("shared_kv", n, None)]["latency_ms"]
            for k in STEPS_PER_FRAME:
                if n % k:
                    continue
                oxygen = values[("continuous_batching", n, k)]["latency_ms"]
                rows.append({
                    "configuration": label,
                    "N": n,
                    "k": k,
                    "baseline_ms": baseline,
                    "shared_kv_ms": shared,
                    "oxygen_ms": oxygen,
                    "shared_kv_speedup": baseline / shared,
                    "oxygen_speedup": baseline / oxygen,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    serializable = {
        label: {
            f"{mode}|N={n}|k={k}": value
            for (mode, n, k), value in groups.items()
        }
        for label, groups in aggregates.items()
    }
    (args.output_dir / "aggregated.json").write_text(
        json.dumps(serializable, indent=2) + "\n"
    )

    lines = [
        "# Qwen3.5 + PI_v3 systems configuration scaling and TP",
        "",
        "All values are end-to-end control-frame latency on GD-8 RTX 4090. "
        "Baseline and Shared-KV are the median of three repeat means, each measured "
        "over 10 post-warmup frames or extended to 30 by the frozen stability rule. "
        "OxyGen is the median of three 50-frame steady-state repeat means.",
        "",
        "Each Qwen3.5 size uses its model-default VLM depth and a matching-depth "
        "randomly initialized PI_v3 action head (S=4). These are systems measurements, "
        "not action-quality results.",
        "",
        "All configurations use the same fixed generated observation and the same "
        "task/language-request text. Each model uses its native tokenizer, producing "
        "the following common-prefix lengths: "
        + ", ".join(
            f"{size}={prefix_tokens_by_size[size]} tokens"
            for size in SIZES
            if size in prefix_tokens_by_size
        )
        + ". TP1 and TP2 input signatures are token-exact within each model size.",
        "",
        "|Configuration|N|Baseline|Shared-KV|Shared speedup|OxyGen k=1|k=5|k=10|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {(row["configuration"], row["N"], row["k"]): row for row in rows}
    for label in payloads:
        for n in DECODE_STEPS:
            first = by_key[(label, n, 1)]
            oxygen_cells = []
            for k in STEPS_PER_FRAME:
                if (label, n, k) not in by_key:
                    oxygen_cells.append("-")
                    continue
                row = by_key[(label, n, k)]
                oxygen_cells.append(
                    f'{row["oxygen_ms"]:.1f} ms ({row["oxygen_speedup"]:.2f}x)'
                )
            lines.append(
                f'|{label}|{n}|{first["baseline_ms"]:.1f} ms|'
                f'{first["shared_kv_ms"]:.1f} ms|{first["shared_kv_speedup"]:.2f}x|'
                + "|".join(oxygen_cells)
                + "|"
            )

    lines.extend(["", "## Validation", ""])
    for label, payload in payloads.items():
        metadata = payload["metadata"]
        values = aggregates[label]
        max_cv = max(row["max_frame_cv"] for row in values.values())
        max_trend = max(row["max_half_trend"] for row in values.values())
        memory = [round(value / 2**30, 2) for value in metadata["allocated_after_load_bytes_per_rank"]]
        lines.append(
            f'- {label}: {metadata["vlm_num_hidden_layers"]} VLM/action layers, '
            f'{metadata["action_head_parameter_count"]:,} action-head parameters; '
            f'maximum within-repeat CV {max_cv:.2%}; maximum half-window difference '
            f'{max_trend:.2%}; model+head allocation per rank {memory} GiB.'
        )
        exact_samples = 0
        total_samples = 0
        mismatch_points = set()
        for n in DECODE_STEPS:
            reference = next(
                row["text"]
                for row in payload["records"]
                if row["mode"] == "separate" and row["decode_steps"] == n
            )
            for row in payload["records"]:
                if row["mode"] != "continuous_batching" or row["decode_steps"] != n:
                    continue
                total_samples += 1
                if row["completed_text_sample"] == reference:
                    exact_samples += 1
                else:
                    mismatch_points.add((n, row["steps_per_frame"]))
        lines.append(
            f'- {label}: {exact_samples}/{total_samples} stored continuous text samples '
            f'are text-exact to single-request decoding; non-exact points are '
            f'{sorted(mismatch_points) or "none"}. All requests have the required '
            f'token length and non-empty text, and all actions are finite [1,16,7].'
        )

    tp2_payloads = [
        (label, payload)
        for label, payload in payloads.items()
        if label.endswith("-TP2")
    ]
    if tp2_payloads:
        projection_by_label = {
            label: payload["metadata"]["linear_attention_projection_placement"]
            for label, payload in tp2_payloads
        }
        replicated = all(
            not value["is_dtensor"]
            for projection in projection_by_label.values()
            for value in projection["projections"].values()
        )
        lines.append(
            "- TP=2 uses native Hugging Face tensor parallelism; end-to-end and stage "
            "latencies use rank-max timing. The native TP plan shards full-attention "
            "and MLP projections. "
            + (
                "Qwen3.5 linear-attention projections remain replicated across ranks "
                "in every TP2 run ("
                + ", ".join(
                    f'{label}: {projection["linear_layer_count"]} layers'
                    for label, projection in projection_by_label.items()
                )
                + ")."
                if replicated
                else "Linear-attention projection placement is recorded in each raw file."
            )
        )
        lines.append(
            "- TP2 ranks use GD-8 GPUs 0 and 1. Their reported topology is SYS "
            "(traffic traverses PCIe and the host interconnect), with no NVLink; see "
            "`gpu_topology.txt`."
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- [Protocol audit](PROTOCOL_AUDIT.md)",
        "- [Aggregated breakdown](aggregated.json)",
        "- [Flat metrics](metrics.csv)",
        "- [Benchmark commands](commands.txt)",
        "- [Software versions](software_versions.txt)",
        "- [Script checksums](script_sha256.txt)",
        "- [GPU topology](gpu_topology.txt)",
        "- [Raw logs](logs/)",
        "- [Native 4B/9B decoder diagnostic](diagnostics/README.md)",
    ])
    for label, path in args.run:
        lines.append(f"- [{label} raw]({path.name})")
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")

    audit_lines = [
        "# Protocol audit",
        "",
        "- Backend: PyTorch BF16 eager, SDPA enabled, compile and CUDA Graph disabled.",
        "- Qwen3.5 fast path: flash-linear-attention 0.5.1 and causal-conv1d 1.6.2.post1 required and active.",
        "- Workload: identical observation/task/language-request common prefix; no language-specific suffix.",
        "- Input audit: the logical observation and text are fixed across sizes; native tokenizer differences change the encoded prefix length. TP1/TP2 signatures are token-exact within each size.",
        "- Action: randomly initialized PI_v3 head, model-default matching depth, horizon 16, S=4 denoising steps.",
        "- Sweep: N={10,20,30}, k={1,5,10}; only N divisible by k is reported.",
        "- Baseline and Shared-KV: 2 warmups, 3 repeats, 10 post-warmup frames per repeat, extended to 30 by the frozen 2% stability rule.",
        "- Continuous batching: 2 warmups, 3 repeats, 50 measured frames per repeat.",
        "- Aggregation: mean across measured frames within each repeat, then median of the three repeat means.",
        "- TP=2: native Hugging Face TP; every latency boundary is synchronized and reduced with rank-max.",
        "- TP=2 topology: GD-8 GPUs 0/1 use a SYS path with no NVLink; raw topology is saved in gpu_topology.txt.",
        "- Correctness: finite [1,16,7] actions; fixed-length non-empty language from every completed request; mixed-age cache checked against same-batch-shape controls over 30 steps plus slot recycling.",
    ]
    (args.output_dir / "PROTOCOL_AUDIT.md").write_text("\n".join(audit_lines) + "\n")


if __name__ == "__main__":
    main()
