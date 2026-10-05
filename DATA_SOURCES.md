# Data sources for the project page

Every measured number on the page comes from one of the four data files below.
Components only format and derive values (ratios, rounding); they hold no
measurements of their own. To update results, change the source, regenerate or
edit the data file, then run `npm run build` and check the page.

Paths are relative to the experiment workspace (`oxygen_ws/`, repository
`xxxxyu/oxygen_ws`) unless they start with `OxyGen-website/`. Paper locations
refer to `NeurIPS-2026---OxyGen/tex/` (camera-ready).

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
| Headline stat "70 Hz … over 200 tokens/s" | floor(70.95), floor to 100 of 212.9 | same point |
| Fig. 2 frame totals | 524 / 469 / 176 ms | 10 / Hz of ablation RTX 4090 at N = 30: `baseline`, `no_batching`, `ours` |
| Fig. 3 | all curves, 3.7× / 3.3× panel labels | tradeoff, k1 vs baseline at N = 30 per device |
| Fig. 4 | all curves | ablation |
| Table 1, π0.5 row | 3.71× / 2.98× | LIBERO RTX 4090 N = 30, k1 and k5 Hz over baseline Hz |

## 2. `src/data/paper-tables.ts` (hand-copied from the paper)

| Export | Paper location | Underlying evidence |
|---|---|---|
| `xiaomi` | `appendix_models.tex`, `tab:xiaomi-full` | `results/xiaomi_protocol_aligned_rerun_20260727/` |
| `starvla` (E = 2/3/4) | `appendix_models.tex`, `tab:starvla-full` | `results/starvla_4b_multi_expert_fixed_prefix_20260727/` |
| `scaling` (Qwen3.5, TP) | `appendix_models.tex`, `tab:model-scaling` | `results/qwen35_scaling_tp_formal_20260728/` |
| `quality` | `appendix.tex`, `tab:quality` | `libero_exp/results/rebuttal_multiseed_summary.{md,csv,json}` |
| `qualityProtocol` (10 seeds, 20,000 rollouts) | `experiments.tex` §Action Quality | same |
| `overhead` (memory, power, energy) | `appendix.tex`, `tab:overhead` | `results/eccv_submission_raw/results_eccv/analysis/overhead/overhead.csv` (from `OxyGen/experiments/run_overhead.py`; the `_mj` columns are actually J, memory column is GiB) |
| `robot` (207.5 + 822.3 vs 198.0 + 195.4 ms, 333 ms window) | `experiments.tex`, `fig:realrobot` | humanoid Jetson AGX Thor measurement; no raw log in this workspace, the paper table is the record |
| `stageProfile` (prefix 45.1 ms, denoise 20.4 ms) | `appendix_systems.tex`, `tab:small-n-profile` and text | `results/eccv_submission_raw/small_n_rerun_20260725/` |
| `memoryAdapter` | not in the paper | `OxyGen/language_adaptation/libero/README.md` ("Reference results"); HF model card `xxxxyu/oxygen-pi05-textual-memory-lora` |

Used by: Table 1 and its notes (6.18×, 1.07× → 1.36×, 0.92×), Fig. 5 and the
96.8% headline stat, the cost table and the −78% headline stat, Fig. 6 and its
three facts, Fig. 2 stage widths, §5 numbers.

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
in the components: 30-token requests (N = 30), k = 5 for Fig. 2/4 and the robot,
k = 1 for the demo and headline, S = 10 for π0.5 (S = 5 Xiaomi), H = 10, three
224 × 224 views on the robot. If a re-run changes a setting, search the
components for it (`grep -rn "30-token\|k</i> = 5" src`).

## Replacing results after a re-run

1. π0.5 curves: replace the JSON files under the raw-results directories
   (same names), rerun `build_figure_data.py`, rebuild.
2. Table values: edit `src/data/paper-tables.ts`, keeping the paper location in
   sync; update this file if the evidence path changes.
3. Demo: re-render with the scripts in `scripts/demo/`, then update `demos` in
   `src/data/site.ts`.
4. Run `npm run build`, open the page and check the headline stats, Fig. 2–6
   and Table 1.
