# Data sources for the project page

Every measured number on the page comes from one of the three data files below.
Components only format and derive values (ratios, rounding); they hold no
measurements of their own. To update results, change the source, regenerate or
edit the data file, then run `npm run build` and check the page.

Paths are relative to the experiment workspace (`oxygen_ws/`, repository
`xxxxyu/oxygen_ws`) unless they start with `OxyGen-website/`. Paper locations
refer to `NeurIPS-2026---OxyGen/tex/` (camera-ready, revision `30fbe58`, checked 2026-10-09).

The adopted remeasurements are the **first run**, not the reliability repeats,
under `results/dodo_camera_ready_20261008/results/formal-affinity-v1/`
(abbreviated `dodo/` below). They use random parameters on one RTX 4090 with
an i9-13900K and CPU affinity `0-7,10-31`. See the evidence package's `INDEX.md`
and the paper's `DODO_RESULTS_INDEX.md`. These runtime experiments do not
establish policy success. The original π0.5 main curves, Thor, robot, energy,
and LIBERO quality results remain the paper's adopted evidence; the dodo
π0.5 reference run is not a replacement for the main curves.

Historical tensor-parallel runs are a paired TP=1/2 comparison on a separate
eight-RTX-4090 server (PCIe, no NVLink, CPU unspecified). Never combine the new
dodo TP=1 latency with historical TP=2 to calculate a scaling benefit.

## 1. `src/data/paper-figures.ts` (generated)

π0.5 tradeoff curves (LIBERO / DROID / ALOHA × RTX 4090 / Jetson AGX Thor) and
the LIBERO ablation. **Do not edit by hand.** Regenerate:

```bash
cd OxyGen-website/scripts/data
python build_figure_data.py /path/to/oxygen_ws            # writes src/data/paper-figures.ts
python build_figure_data.py /path/to/oxygen_ws --check-pdf  # also compares with the paper PDFs (needs pymupdf)
```

| Device | Raw results | Settings read |
|---|---|---|
| RTX 4090 (JAX) | `results/eccv_submission_raw/results_eccv/` (same as `results_eccv_merged.zip`) | `baseline/`, `parallel_mps/`, `shared_kv/`, `continuous_batching/` × `pi05_o2_{libero,droid,aloha}/` |
| Jetson AGX Thor (PyTorch) | `results/eccv_submission_raw/results_eccv_jetson_pytorch/` (from `results_eccv_jetson_pytorch.zip`) | same layout |

File names: `denoise10_decode{N}.json` (baseline, parallel_mps, shared_kv) and
`denoise10_decode{N}_step{k}_uniform_1.json` (continuous batching), with
N ∈ {5, 10, 15, 20, 30}, k ∈ {1, 5, 10}.

Metrics follow `OxyGen/experiments/analysis/aggregate_metrics.py`: warmup frames
excluded; action frequency = 10 × frames / Σ frame_ms × 1000; language
throughput = mean over frames of tokens / frame_ms × 1000. The ablation uses
k = 5 ("ours"), `shared_kv` ("no batching"), and `shared_kv` at N = 5 as the
one-frame decoding bound.

Last check (`scripts/data/raw_verification.json`): every value matches the
camera-ready figures within 0.05.

**Numbers on the page derived from this file**

| Where | Value | Derivation |
|---|---|---|
| Masthead "One second of inference" | 71 Hz / 213 tok/s vs 19 Hz / 57 tok/s | LIBERO, RTX 4090, N = 30: `k1` vs `baseline` |
| Headline stat "3.7×" and §3 text | 3.7× | same point, k1 Hz / baseline Hz |
| Headline stat "71 Hz … 213 tokens/s" | round(70.9), round(212.9) | same point |
| Fig. 3 frame totals | 524 / 469 / 176 ms | 10 / Hz of ablation RTX 4090 at N = 30: `baseline`, `no_batching`, `ours` |
| Fig. 4 | all curves, 3.7× / 3.3× panel labels | tradeoff, k1 vs baseline at N = 30 per device |
| Fig. 5 | all curves | ablation |
| Table 1, π0.5 row | 3.71× / 2.98× | LIBERO RTX 4090 N = 30, k1 and k5 Hz over baseline Hz |

## 2. `src/data/paper-tables.ts` (paper tables and adopted raw-result summaries)

| Export | Paper location | Underlying evidence |
|---|---|---|
| `xiaomi` | `appendix_models.tex`, `tab:cross-model-results` | `dodo/xiaomi/raw.json`; `dodo/paper/tables/xiaomi_full.csv` (also N=5/15) |
| `starvla` (E = 2/3/4) | `appendix_models.tex`, `tab:cross-model-results` | `dodo/starvla/raw.json`; `dodo/paper/tables/starvla_full.csv` (also N=5/15) |
| `scaling` (Qwen3.5, TP=1) | `appendix_models.tex`, `tab:model-scaling` | `dodo/qwen35_scaling_tp_formal_20260728/formal_*_tp1.json`; `dodo/paper/tables/model_scaling.csv` |
| `tensorParallel` (9B, N=30, TP=1/2) | `appendix_models.tex`, Tensor parallelism paragraph | `results/qwen35_scaling_tp_formal_20260728/aggregated.json` (historical paired runs) |
| `quality` | `appendix.tex`, `tab:quality` | `libero_exp/results/rebuttal_multiseed_summary.{md,csv,json}` |
| `qualityProtocol` (10 seeds, 20,000 rollouts) | `experiments.tex` §Action Quality | same |
| `overhead` (memory, power, energy) | `appendix.tex`, `tab:overhead` | `results/eccv_submission_raw/results_eccv/analysis/overhead/overhead.csv` (from `OxyGen/experiments/run_overhead.py`; the `_mj` columns are actually J, memory column is GiB) |
| `robot` (207.5 + 822.3 vs 198.0 + 195.4 ms, 333 ms window) | `experiments.tex`, `fig:realrobot` | humanoid Jetson AGX Thor measurement; no raw log in this workspace, the paper table is the record |
| `stageProfile` (prefix 44.0 ms, denoise 20.4 ms) | `appendix_systems.tex`, `tab:small-n-profile` and text | `dodo/small_n/profile.json`; `dodo/paper/tables/small_n_profile.csv`, rounded mean over N=1–5 |
| `memoryAdapter` | not in the paper | `OxyGen/language_adaptation/libero/README.md` ("Reference results"); HF model card `xxxxyu/oxygen-pi05-textual-memory-lora` |

Used by: Table 1 and its notes (StarVLA 6.64×; prefix-sharing gains
1.23× → 1.40× → 1.59×; Xiaomi boundary 0.96×; Qwen3.5 k=5 range
3.58–3.79×), Fig. 6 success-rate validation, the cost table and its −78%
energy result, Fig. 7 and its three facts, Fig. 3 stage widths, §5 numbers.

Fig. 3 retains original ablation frame totals; the new synchronized stage
profile only supplies illustrative widths. Its remaining segment includes
decoding and other frame work, not a directly measured decode-only latency.
The caption explicitly distinguishes these measurements.

## 3. `src/data/site.ts` — demo videos

`demos[].oxygenS / baselineS` are wall-clock task-completion times of the
rollouts shown in Fig. 1 (LIBERO-10, one RTX 4090, OxyGen k = 1). They are
printed by `scripts/demo/render_comparison.py --meta` and come from:

| Demo | OxyGen run | Baseline run | Episode |
|---|---|---|---|
| white mug & pudding | `libero_exp/language_adapter_runs/visual_memory_v1_20260813/demo_campaign_v2/oxygen/k1/task6` | `…/blocking_baseline_eos_v2/task6` | 3 |
| alphabet soup & tomato sauce | `…/demo_campaign_v2/oxygen/k1/task0` | `…/blocking_baseline_eos_v2/task0` | 2 |
| two moka pots | `…/demo_campaign_v2/oxygen/k1/task8` | `…/blocking_baseline_eos_v2/task8` | 3 |

The memory lines quoted in §5 ("step 50", "step 185") come from the
`inference_events.jsonl` of the alphabet-soup OxyGen run. Re-rendering is
described in `README.md` ("Demo videos").

## 4. Settings stated in prose

These are experiment configurations rather than measurements, written directly
in the components: 30-token requests (N = 30), k = 5 for Fig. 3/5 and the robot,
k = 1 for the demo and headline, S = 10 for π0.5 (S = 5 Xiaomi; S = 4 StarVLA/Qwen3.5), H = 10, three
224 × 224 views on the robot. If a re-run changes a setting, search the
components for it (`grep -rn "30-token\|k</i> = 5" src`).

## Replacing results after a re-run

1. π0.5 curves: replace the JSON files under the raw-results directories
   (same names), rerun `build_figure_data.py`, rebuild.
2. Table values: edit `src/data/paper-tables.ts`, keeping the paper location in
   sync; update this file if the evidence path changes.
3. Demo: re-render with the scripts in `scripts/demo/`, then update `demos` in
   `src/data/site.ts`.
4. Run `npm run build`, open the page and check the headline stats, Fig. 2–8
   and Table 1.

## Desktop editorial refinement (2026-10-10)

The headline now pairs the original π0.5 speedup and frequency with Xiaomi
5.83× and StarVLA 6.64× (N=30, k=1), each against its own baseline.
Success rate remains in the validation section. Table 1 covers models and
experts; the following Qwen3.5 size panel uses the same `scaling` export
and a common zero-based bar scale. No measurement values were changed.

## Chapter review pass (2026-10-10)

Headline frequency now uses ordinary integer rounding (70.9 → 71 Hz), matching
the inference illustration. Demo task-completion readouts derive from `demos`
and update together with the selected video. Cross-model numbers are unchanged.

The robot figure now separates the measured inference sum (1030 → 393 ms)
from an illustrative overlap timeline. Motion begins at the 198 ms dispatch
point and lasts 333 ms; the 195.4 ms language path follows action inference.
The timeline is a possibility illustrated using measured durations, not a new
logged overlap experiment. Communication overhead is omitted explicitly.
