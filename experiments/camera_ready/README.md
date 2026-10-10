# Camera-ready experiment suite

Branch: `camera-ready-experiments`. This is the complete supplementary **performance**
rerun suite behind the camera-ready cross-model and systems sections. Run all single-GPU
measurements sequentially on the target RTX 4090 host. Do not run timing jobs concurrently.
The primary pi0.5/Thor experiments, 10-seed LIBERO rollouts, and robot demonstrations are
separate evaluations; this command does not rerun or replace them.

## Executed version and current paper

The runners, vendored inference code, protocol and environment locks in this branch
match the source actually executed on the dodo host on 2026-10-08, including its
post-preparation fixes. `provenance/executed-dodo-source-lock.json` preserves the
measured source manifest; `provenance/executed-dodo.json` records its digest.
The top-level `source-lock.json` covers this maintained branch, including reviewed
plotting and documentation updates, and therefore has a different digest.

The private `paper_results_20261009` archive maps every adopted paper result to its
raw records and source code. Its `PAPER_RESULT_MAP.json` is the selection authority.
The complete rerun grid intentionally includes unused results. In particular:

- Main pi0.5/Thor speed, ablations, resource measurements and D.4 completion time use
  submission data, not this suite's pi_reference rerun.
- LIBERO success rates use the separate rebuttal ten-seed evaluation.
- Xiaomi, StarVLA, Qwen3.5 single-GPU, short-output breakdown and manager overhead
  use the primary formal-affinity-v1 run.
- Stability uses only fluctuating traces, ordinary means per complete 119-frame
  cycle, and first/last five-cycle pooled means. P99 and steady results are not adopted.
- The TP paragraph uses the historical same-host TP1/TP2 pair. Random-init TP2 is
  rejected; the new single-GPU record must not replace its TP1 denominator.

Regenerate the adopted supplementary figure after measurements:

```bash
python experiments/analysis/plot_camera_ready_appendix.py \
  --results-root /path/to/formal-affinity-v1 \
  --output-dir /path/to/reviewed-figures --only stability
```

Run commands from the repository in its plotting environment. Exported full-grid
CSV/TeX files are analysis products; they do not automatically replace the reviewed
paper's selected rows, ratios, or formatting. Source/asset changes require a new
output directory rather than resuming a measured historical run.

## On the destination host

1. Check out this branch. Install `uv`, the NVIDIA driver, and Liberation Serif fonts.
   The installer uses checksum-verified official Linux x86-64 CPython 3.12 wheels for
   FlashAttention and causal-conv1d. The target driver must support PyTorch CUDA 12.8.
2. Copy the **model metadata directories**, including model code, tokenizer/processor
   files, configs, and action/state schemas. The default configuration uses random
   parameter initialization because every task measures systems behavior rather than
   model quality. Keep trained weights available only when explicitly selecting
   `"random_init": false`; StarVLA still needs the complete run directory in that mode.
3. Create the four isolated environments:

   ```bash
   bash experiments/camera_ready/bootstrap.sh /absolute/path/to/oxygen-envs
   ```

4. Copy `config.example.json` to a local JSON file and set the checkpoint paths, Python paths,
   output directory, device index, and source checkpoint hash manifest. Only these machine paths
   change. The experimental grid and timing protocol live in `protocol.json`.
5. For a checkpoint-backed run, generate the weight hash manifest once:

   ```bash
   python3 experiments/camera_ready/run.py seal-assets --config /path/to/source-config.json
   ```

   Transfer that manifest with the checkpoints. In the default random-init mode no weight
   manifest is needed; preflight instead checks that each metadata directory exists.
6. Inspect the command expansion, then run the whole suite:

   ```bash
   python3 experiments/camera_ready/run.py plan --config /path/to/host.json
   python3 experiments/camera_ready/run.py run --config /path/to/host.json
   ```

The run command checks sources, assets, package versions, backend imports and GPU availability
before measurements. It generates raw results, validates coverage, exports tables and renders
three diagnostic figure PDFs automatically. The reviewed paper retains only the runtime-stability figure; model scaling and manager overhead use tables. The entry point never silently downloads a model
or falls back from the required attention implementation.

Resume after fixing a failure:

```bash
python3 experiments/camera_ready/run.py run --config /path/to/host.json --resume
```

Successful tasks are skipped only after revalidating their data and hashes. Failed jobs restart as
whole tasks; per-attempt logs are retained. Environment/config/source/weight changes require a
new output directory. A failed task does not prevent independent later tasks from finishing, but
any failure prevents the suite from being reported as complete.

## Coverage fixed in code

| Group | Included measurements |
| --- | --- |
| pi0.5 paired reference | Frozen official driver, full LIBERO N/k grid, S=10, Baseline/Shared KV/continuous batching; includes completion-time reconstruction |
| pi0.5 short outputs | N=1,2,3,4,5,10,15,20,30; S=10; both synchronized stages and natural end-to-end Baseline/Shared KV; 2 warmups + 10 measurements |
| Allocation | N=30, k=5; append-expert counts 1/2/4; steady and bursty allocation probes |
| Full-frame manager | JAX and converted PyTorch; N=30, k=5/1/10; same-run natural and stage-synchronized measurements; full shape warmup |
| Long run | Each backend: steady batch 32 (arrivals 5,5,6) and fluctuating batch up to 40; 3,000 measured frames each; request completion traces and memory samples |
| Xiaomi | Original two images/state/task; S=5; N=5/10/15/20/30; k=1/5/10 for every divisible pair; Baseline, Shared KV, continuous batching; 3 repeats; separate cache/progress implementation check |
| StarVLA | Released PI_v3, GR00T, OFT checkpoints; E=2/3/4; identical 116-token prefix; S=4; full N/k sweep; per-mode checks and persistent-scheduler preflight |
| Qwen3.5 | 0.8B/2B/4B/9B; N=10/20/30; k=1/5/10; all three modes plus prefill; random matching PI_v3 heads; native fast path required |
| Native decoder | 4B/9B, prefill batches 1/3/6 and decode batches 1/3/6/30; 3 warmups + 5 repeats |

There are **18 single-GPU tasks**. Optional `include_tp2: true` plus two distinct device indices
adds six tasks: the complete four-size TP=2 sweep and the two native decoder diagnostics.
Random pi initialization uses the same parameter precision as checkpoint loading:
JAX BF16, and PyTorch BF16 with the checkpoint path's selected FP32 exceptions.
JAX initialization/casting takes place on CPU before transfer, avoiding simultaneous
FP32/BF16 copies in the GPU pool. Initialization is outside all measured regions.

Random-init TP=2 is currently rejected because config-only construction does not
apply a TP plan. Use the default single-GPU protocol; do not interpret replicas as
tensor parallelism. Checkpoint-backed TP coverage is deliberately broader than the paper's 9B-only TP discussion. All TP=1 tasks also run on
that host; compare TP=2 only with its same-host TP=1 results. Single-GPU completion explicitly
excludes TP=2 and never fabricates it from the old server data.

The added pi0.5 N=10/15/20/30 and manager k=10 points are extra diagnostics. Original points keep
their historical timings, workloads and aggregation rules. StarVLA's adaptive 10-to-30-frame
rule is preserved: the historical run happened to extend all points, but a new run is not required
to trigger the same extensions. No validator enforces a desirable speedup or monotonic trend.

## Outputs

- `run.json`: resolved configuration, protocol, CPU/GPU/topology, package versions and source/asset locks.
- `status/*.json`, `logs/*`: commands, exit/validation status, raw artifact hashes and attempt logs.
- Raw JSON/JSONL: individual frame timings, repeats, stages, memory and request completions.
- `paper/figures/`: `model_size_scaling.pdf`, `manager_overhead_breakdown.pdf`, `longrun_stability.pdf`
  and PNG previews, using the reviewed figure layouts and shared OxyGen style.
- `paper/tables/`: CSV and LaTeX tabular bodies for the main cross-model table and all relevant
  appendix tables; extra repeat-spread and allocation tables support review.
- `paper/all_metrics.json`: unrounded metrics for updating prose and captions.

The plotter derives annotations and axis ranges from each new run. It never carries over the old
4.5x KV-split label. Font sizes and canvas widths remain fixed. Tables are generated from raw
measurements, not copied from the manuscript. Insert PDFs at text width; review new curves and
update captions/prose before replacing the paper. This suite never edits the LaTeX repo automatically.

To regenerate products from a completed run:

```bash
python3 experiments/camera_ready/run.py report --config /path/to/host.json
```

## Source and environment provenance

See `provenance/sources.json`, `provenance/sessions.json`, and `PROVENANCE.md`.
The inference source snapshots are included so another Codex session does not need the old
workspace, private conversation transcripts, or upstream `main` branches. Historical absolute-path
launchers are retained as evidence under `provenance/`; do not execute them.

The pi package is based on `dcc134a8...`, with its original `uv.lock`. The local
destination preparation adds an environment override for the JAX cache directory;
benchmark computations and timers are unchanged. See the destination section in
`VALIDATION.md` for the random-init and entry-point corrections.
The formal benchmark scripts are preserved, apart from sibling-import path relocation.
StarVLA uses its recorded base revision plus the historical OFT change. Xiaomi's custom Hugging Face
model implementation is in its checkpoint directory and is covered by the asset hash manifest.

Separate environments are intentional:

| Environment | PyTorch | Transformers | Required specialization |
| --- | --- | --- | --- |
| pi | 2.6.0, CUDA 12.4 | 4.53.2 | JAX 0.5.3, Flax 0.10.2, original uv lock |
| xiaomi | 2.8.0, CUDA 12.8 | 4.57.1 | FlashAttention 2.8.3 |
| starvla | 2.8.0, CUDA 12.8 | 4.57.0 | SDPA, eager |
| qwen35 | 2.8.0, CUDA 12.8 | 5.2.0 | FLA 0.5.1, causal-conv1d 1.6.2.post1 |

The historical Qwen environment inherited many unrelated system packages; its full `pip freeze`
is evidence, not an install recipe. The clean environment recipe pins inference dependencies and all resolved transitive packages
in `environments/*.lock.txt`, with Python 3.11.15/3.12.13. The historical cross-model Python
was 3.12.11; this clean rebuild uses a newer patch release and records that difference.
Core framework, CUDA wheel, attention, and benchmark versions are retained. A later environment refinement must start a new result directory.

## Checks before another session runs this

```bash
python3 -m unittest discover -s experiments/camera_ready -p test_suite.py -v
python3 experiments/camera_ready/run.py preflight --config /path/to/host.json
```

Read `VALIDATION.md` for what has actually been checked locally. Do not equate CPU coverage tests
or archive re-analysis with a full new-host GPU run. After the first complete destination run,
retain its asset manifest and environment freezes with the results, and review the generated PDFs.

## Performance-only implementation validation

The destination review approved random-weight performance tests without requiring
cross-batch-size greedy token equality or meaningful generated text. Token equality
and cross-batch errors remain recorded as diagnostics; trained weights are not
required or downloaded for this purpose.

Xiaomi now validates fixed, distinct per-request inputs across 30 decode steps at
matched batch size and varying request ages, comparing logits and full caches
with atol=1e-4/rtol=1e-3. It checks stack/split, request progress, replacement
isolation and a deliberately swapped-cache negative control. The original
five staggered requests must still each complete all 30 tokens.

Qwen retains matched-batch mixed-age/rollout controls, exact input association,
cache-copy/slot-recycle isolation, finite state checks and age progression.
Only singleton-versus-batch greedy argmax equality is no longer a gate. Existing
matched-batch checks are stronger and already pass; their thresholds are unchanged.
These audits are outside measured regions. Model workload, timing and experimental
grids do not change. Formal runs still require complete-grid validation and an
approved, stable host condition.
