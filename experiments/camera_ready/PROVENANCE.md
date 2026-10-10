# Rebuttal provenance and migration decisions

The local historical sessions are still present under `/home/lixiangyu/.codex/sessions/2026/07/`.
`provenance/sessions.json` records their full paths, identifiers, origin and SHA-256.
Conversation contents are not published or needed to run the suite.

| Work | Session ID |
| --- | --- |
| Main rebuttal experiment session | `019f920d-e283-7201-8a4b-31d31623081f` |
| Batch-32 fluctuating long run | `019f9d90-2c74-7d53-b26e-c39ef2554f17` |
| Xiaomi protocol-aligned rerun | `019fa2bc-c501-7b82-8ee8-5227b5ed6fdf` |
| StarVLA real expert sweep | `019fa2bd-fc04-7500-a3eb-6397591bc05d` |
| Scaling/TP parent | `019fa2be-31b8-7851-a55d-f2a6c72b0cf5` |
| Qwen3.5 rerun | `019fa4a0-6daf-7210-8658-7c2be74bab32` |
| Qwen3.5 formal runner | `019fa4bd-d11c-7543-a383-78c4a88c1c8f` |
| Aligned full-frame manager | `019fae48-45ad-7a30-aafe-bd33db4d12d8` |

Recovered artifacts are mapped in `provenance/sources.json`. Formal launchers are retained
under `provenance/` for comparison; the runnable manifest is `protocol.json`.

## Source snapshots

- **OxyGen**: unmodified `dcc134a8eceee79718df0511f6f32989f732c248` in `vendor/oxygen`.
  Original submission data used `13bdc15`; rebuttal profiling used the later snapshot, which
  also supports the converted PyTorch backend. The manifest measures that rebuttal implementation.
- **StarVLA**: `c00b8e18c1b5ba6228fd68a8f38acca8df4e938a` package and deployment helpers,
  plus the historical OFT context-sharing change. Compared file by file with the upstream tree:
  `QwenOFT.py` is the only modified included upstream source. Its diff is retained separately.
  The two latest upstream commits are recorded separately in a Git bundle and log; the bundle
  is a history fragment with prerequisites, not a standalone full repository clone.
- **Xiaomi**: runtime and sweep scripts from the snapshot based on
  `fa170bc295a22e0f4e582ff9e5bb104b3c6900d6`. Inference loads custom code from the model
  checkpoint with `trust_remote_code=True` and `local_files_only=True`; those files must be
  transferred and hashed with the checkpoint. The original repository license is retained.
- Only hard-coded sibling Python import directories were relocated in archived runners.
  No attention, denoising, scheduling, timer, or adaptive-repeat logic was rewritten.
- The input RGB images and state come from the exact 07-27 Xiaomi run. StarVLA and Qwen
  synthetic observations are constructed by the original fixed runner code.

## Which historical result is authoritative for each supplement

- Xiaomi: `xiaomi_protocol_aligned_rerun_20260727`, not the earlier 07-25 summary.
- StarVLA: `starvla_4b_multi_expert_fixed_prefix_20260727`, including the fixed prefix
  and the persistent continuous scheduler. Do not use earlier heterogeneous sweep defaults.
- Qwen: `qwen35_scaling_tp_formal_20260728`, not the earlier Qwen2.5 experiment. The
  Qwen2.5-named helper remains because the formal Qwen3.5 script imports its action-head utilities.
- Manager: `kv_manager_fullpath_aligned_20260729/profile_fullpath.py`, not the earlier
  resumed-only microbenchmark.
- Long run: the profiler saved beside `kv_manager_longrun_20260726` results; the earlier
  `remote_edit` version lacks `--arrival-pattern` and `--arrival-scale`.

## Corrections found during recovery

The short-N report already contains paired stage and end-to-end measurements. At N=5,
45.41 ms is the stage saving, while 54.51 ms is the end-to-end saving; the latter closely
matches the original 55.18 ms. This is not an unresolved failure to reproduce prefix reuse.
The suite retains both timing boundaries for diagnosis, but the end-to-end wall-clock
measurement is the primary result and must be used for any manuscript replacement. The
stage-synchronized sum is a diagnostic breakdown only; it is not an alternative speedup
claim.

All camera-ready performance tasks are capability-independent system measurements. They
therefore default to config-only model construction with random parameters. The metadata
directories still provide architecture, tokenizer/processor, custom code, normalization,
and action-schema information needed to exercise the same tensor shapes and execution paths.

The archived JAX fluctuating run samples memory every 5 frames, while PyTorch samples every
10. The formal command manifest preserves those cadences; plot bins consume every sample.
The existing paper caption's blanket “every ten frames” needs correction when adopting results.
Long-run `--frames` counts warmup too: steady 3030−30=3000, fluctuating 3100−100=3000.

## Checkpoint sources

These are recovery locations, not floating revisions to use automatically:

- pi JAX: ModelScope `MaZp001/pi05_libero`, historical revision
  `c27c8531f6748ee8814db1dc55cd62c6b9e862ea`; PyTorch is the historical conversion of this checkpoint.
- Xiaomi: `XiaomiRobotics/Xiaomi-Robotics-0-LIBERO`.
- Shared StarVLA base: `Qwen/Qwen3-VL-4B-Instruct`.
- Experts: `StarVLA/Qwen3VL-PI_v3-Bridge-RT_1` (step 50000),
  `StarVLA/Qwen3VL-GR00T-Bridge-RT-1` (step 20000),
  `StarVLA/Qwen3VL-OFT-Bridge-RT-1` (step 5000).
- Scaling: `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`.

Most historical model directories were not archived with complete content hashes. The source-host
`seal-assets` step fills this gap and detects transfer/version changes. The runner refuses to
claim an exact source-asset match without that manifest. If original weights cannot be recovered,
record immutable downloaded revisions and classify the run as a new asset baseline before comparison.
