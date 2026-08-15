# LIBERO visual-memory suffix LoRA: selected result

The final run uses the reviewed `libero_all_v6` predicate annotations and a
private `Memory: ` suffix. It predicts only stable goal predicates completed in
the current observation; generated memory is not part of the shared prefix and
is not fed back to the action expert.

## Data and training

| Item | Value |
|---|---:|
| LIBERO suites / tasks | 4 / 40 |
| Demonstrations / annotated frames | 2,000 / 338,575 |
| Annotation QA issues | 0 |
| Adapter | rank-16 suffix LoRA, all 18 language layers |
| Selected run | `lr=1e-4`, step 1,500 |
| Suffix seed / maximum length | `Memory: ` / 28 tokens |

Training and evaluation use the same incremental seed-plus-token execution
path as serving. Learning rates `1e-4`, `3e-5`, and `1e-5` were compared on a
fixed 400-sample task/state-balanced greedy validation set.

## Held-out results

| Metric | Result |
|---|---:|
| Incremental teacher-forced token accuracy (5,082 samples) | 98.15% |
| Greedy normalized exact match (400 samples) | 87.25% |
| Greedy word F1 | 90.01% |
| Root KV maximum absolute difference (20 observations) | 0 |
| Fixed-noise action maximum absolute difference (20 observations) | 0 |

The zero differences are exact checks against the adapter-bypassed path. They
confirm that suffix LoRA does not alter the root KV cache consumed by the
action expert or the resulting action tensor.

## Runtime

On one RTX 4090, the median 11-token suffix incurred 8.16 ms of adapter
overhead. The shared prefix forward took 47.66 ms, 5.84 times the adapter
overhead. The adapter therefore does not offset the prefix computation removed
by sharing for this workload.

Three real continuous-batching LIBERO rollouts completed successfully with no
exception. They contained 142 action replans and 142 memory requests, of which
134 finished before episode termination; the largest active language batch was
5. These rollouts are serving regressions and qualitative examples, not a
LIBERO success-rate estimate.

## Known limitation

The single-stage drawer example exposed a label/model failure: the action
succeeded while the language output continued to report no completed memory.
The feature demonstrates train/inference compatibility and action isolation,
not perfect visual-state recognition.

## Release assets

The Git repository keeps code and compact metrics only. The selected adapter
must be distributed as a separate model asset:

| Asset | Value |
|---|---|
| File | `adapter_step_1500.npz` |
| Size | 107 MB |
| SHA256 | `be7e5a8a27bff13176a1e6b42100ead5a4d0bab9d9397784aedb534c1a25af78` |
| Base model | released pi0.5 LIBERO checkpoint |

The annotation release should contain the 2,000 derived JSONL trajectories,
schema/QA metadata, and LIBERO attribution, but not redistribute the raw HDF5
demonstrations. LIBERO code is MIT licensed and the official dataset is CC BY
4.0; a public release must retain the upstream attribution.

Machine-readable metrics are in
[`results/2026-08-13/visual_memory_v1/summary.json`](results/2026-08-13/visual_memory_v1/summary.json).
