import json

from experiments.libero_language_adapter import prepare_hf_release


def test_stage_release_writes_portable_assets(tmp_path) -> None:
    adapter = tmp_path / "adapter.npz"
    adapter.write_bytes(b"adapter")
    annotations = tmp_path / "annotations"
    trajectory = annotations / "libero_spatial" / "task" / "demo_0.jsonl"
    trajectory.parent.mkdir(parents=True)
    for index in range(2_000):
        path = trajectory if index == 0 else trajectory.with_name(f"demo_{index}.jsonl")
        path.write_text('{"frame": 0}\n', encoding="utf-8")
    (annotations / "qa.json").write_text(
        json.dumps(
            {
                "files": 2_000,
                "frames": 338_575,
                "files_reaching_success": 2_000,
                "files_with_issues": 0,
            }
        ),
        encoding="utf-8",
    )
    (annotations / "pickup_audit.json").write_text("{}\n", encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "checkpoint": {"step": 1500, "suffix_length": 28, "suffix_seed": "Memory: "},
                "evaluation": {
                    "teacher_forced_token_accuracy": 0.98,
                    "normalized_exact": 0.87,
                    "word_f1": 0.90,
                },
                "performance": {
                    "adapter_overhead_per_median_request_ms": 8.16,
                    "prefix_prefill_ms": 47.66,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = prepare_hf_release.stage_release(
        adapter=adapter,
        annotation_root=annotations,
        summary_path=summary,
        output_root=tmp_path / "release",
        base_checkpoint="base",
        hardlink=False,
    )

    assert manifest["annotations"]["annotation_files"] == 2_000
    assert (tmp_path / "release/model/adapter.npz").read_bytes() == b"adapter"
    portable = json.loads((tmp_path / "release/dataset/manifest.json").read_text())
    assert "source_path" not in portable
    assert len(list((tmp_path / "release/dataset/annotations").rglob("*.jsonl"))) == 2_000
