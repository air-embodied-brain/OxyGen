# LIBERO language-adapter experiment

This experiment evaluates three ways to restore structured language output on
the released pi0.5 LIBERO action checkpoint:

- `suffix_lora`: standard rank-16 LoRA on every language-layer attention and
  FFN projection, enabled only after the shared observation/task prefix;
- `full_lora`: the same LoRA modules enabled on the complete language sequence;
- `final_mlp`: a 65,536-parameter residual MLP after the final language hidden
  state, used as a low-capacity lower bound.

The suffix-only path builds the observation/task root KV cache with the frozen
base model. The action expert reads that root directly. Language inference
creates a private fixed-size append state, writes the `Subtask: ` seed through
LoRA, and then generates target tokens incrementally through the same LoRA
layers. The root and action path never see adapter-modified states.

## Data

- Supervision: predicate-derived `language.next` labels.
- Coverage: all 40 tasks in LIBERO-Spatial, Object, Goal, and LIBERO-10.
- Split: five held-out episodes per task, for 1,800 training and 200 validation
  episodes (47,773 and 5,277 sampled frames in the final v6 annotations).
- Prefix: the released pi0.5 LIBERO observation, state, and task-prompt protocol.
- Suffix: `Subtask: <next label> EOS`, padded to 20 tokens without truncation.

Build the deterministic split:

```bash
python -m experiments.libero_language_adapter.build_split \
  --annotation-root /path/to/libero_all_v6 \
  --dataset-root /path/to/raw_libero \
  --output /path/to/split.json
```

## Training

The standard comparison uses batch size 2, 2,000 steps, 100 warmup steps,
AdamW, cosine decay, gradient clipping at 1.0, and learning rates `1e-4` and
`3e-4`. LoRA uses rank 16 and alpha 16 on all 18 language layers, including
Q/K/V/O and gated/up/down FFN projections. Frozen parameters are BF16; adapter
and optimizer states are FP32.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
python -m experiments.libero_language_adapter.train \
  --split /path/to/split.json \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --adapter-type suffix_lora --rank 16 --alpha 16 \
  --learning-rate 3e-4 --steps 2000 --warmup-steps 100 \
  --output-dir /path/to/run
```

Standard teacher forcing exposes a BF16 shape-sensitive gap between full-suffix
training and block-plus-token incremental inference. The current selected model
therefore uses the exact deployed incremental path for both training and
evaluation. Training samples a task uniformly, then a target/predicate within
that task, so long episodes and frequent completion labels do not dominate.

Starting from the first-round suffix-LoRA checkpoint, run the balanced sweep:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
python -m experiments.libero_language_adapter.train \
  --split /path/to/split.json \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/norm_stats.json \
  --adapter-type suffix_lora --rank 16 --alpha 16 \
  --init-adapter /path/to/first_round_adapter_step_2000.npz \
  --loss-mode incremental --sampling-mode task_target_balanced \
  --validation-sampling-mode task_target_balanced \
  --validation-samples-per-task 10 \
  --learning-rate 1e-4 --steps 2000 --warmup-steps 100 \
  --eval-every 250 --output-dir /path/to/run
```

Run `1e-4`, `3e-5`, and `1e-5` as independent jobs. Scan their saved
checkpoints on the same 400-sample, task/target-stratified incremental greedy
set before choosing the final checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.libero_language_adapter.scan_checkpoints \
  --split /path/to/split.json \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/norm_stats.json \
  --adapters '/path/to/run/adapter_step_*.npz' \
  --samples-per-task 10 --output-dir /path/to/checkpoint_scan
```

## Evaluation

`evaluate.py` reports full held-out teacher-forced metrics, greedy incremental
generation across 80 samples, root/action isolation with fixed diffusion noise,
and adapter latency relative to the duplicate prefix forward it replaces.
`verify_serving.py` calls the actual continuous-batching policy and checks that
one request returns both actions and adapter-generated language.
`benchmark_throughput.py` compares isolated execution and OxyGen with identical
LoRA-on and LoRA-bypassed paths under a fixed-length paper-style workload.
`rollout_review.py` runs deterministic LIBERO rollouts, records every raw policy
response, and builds a lazy-loading review page with the generated text overlaid
on the corresponding action chunk. Videos default to 10 FPS, or 0.5x the
20 Hz simulator control rate. The latest three language requests are shown as a
rolling buffer; incremental updates replace the current row in place.
`merge_rollout_reviews.py` combines independently run per-suite reviews into one
page without copying or re-encoding their videos.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform \
python -m experiments.libero_language_adapter.evaluate \
  --split /path/to/split.json \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/norm_stats.json \
  --adapter /path/to/adapter_step_1500.npz \
  --adapter-type suffix_lora --rank 16 --alpha 16 \
  --teacher-forced-samples 5075 --teacher-forced-loss-mode incremental \
  --samples-per-task 10 --action-invariance-samples 20 \
  --output-dir /path/to/evaluation
```

Run the controlled end-to-end throughput comparison:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python -m experiments.libero_language_adapter.benchmark_throughput \
  --split /path/to/split.json --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/norm_stats.json --adapter /path/to/adapter.npz \
  --max-decoding-steps 20 --steps-per-frame 5 \
  --warmup-frames 8 --measured-frames 40 --repeats 3 \
  --output /path/to/throughput_summary.json
```

For qualitative review, start `serve_adapter.py` with the selected adapter, then
run the real simulator client. The server stops text on tokenizer EOS; the
fixed-length decoding used by the paper's performance sweeps is unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.libero_language_adapter.serve_adapter \
  --checkpoint /path/to/pi05_libero --norm-stats /path/to/norm_stats.json \
  --rank 16 --adapter /path/to/adapter_step_1500.npz --port 8011 \
  --request-mode new_each_call --steps-per-frame 5 --max-decoding-steps 20

python -m experiments.libero_language_adapter.rollout_review \
  --host 0.0.0.0 --port 8011 --task-id 0 --episodes 0,1,2,3,4 \
  --replan-steps 5 --seed 7 --video-fps 10 --source-control-hz 20 \
  --output-root /path/to/review
```

With `new_each_call`, every action replan creates one language request from the
same current observation. The server advances that request and every unfinished
older request in one continuous batch, while generating actions only for the
new observation. Thus `replan_steps=5` at LIBERO's 20 Hz control rate produces
both action replans and language-request arrivals at 4 Hz. The saved inference
events contain every request's incremental text and the actual batch size.

For a wall-clock rollout, pass `--wall-clock-timeline --video-fps 100
--source-control-hz 20`. Each simulator state then occupies 50 ms of video,
while every blocking model call is represented by a frozen environment frame
for its measured client round-trip time, quantized to 10 ms. The server resets
the policy RNG on each client connection, so a warmup rollout can compile all
shapes without changing the action noise used by the formal episode.

`run_wallclock_sweep.sh` runs one OxyGen or isolated-baseline point. The
isolated baseline uses separate action/language prefix forwards and advances
each active language request sequentially, while retaining the same request
arrivals, LoRA, EOS handling, and token budget. `build_rollout_comparison.py`
builds the two-column, five-row task pages.

An existing review can be rerendered from its saved videos and response logs
without rerunning the policy or simulator:

```bash
python -m experiments.libero_language_adapter.rollout_review \
  --rerender-from /path/to/old_review --output-root /path/to/new_review \
  --video-fps 10 --source-control-hz 20
```

See [RESULTS.md](RESULTS.md) for the findings and the
[current aggregate summary](results/2026-08-11/balanced_incremental_v2/aggregate_summary.json)
for machine-readable metrics. Large checkpoints, datasets, and review videos
remain local and are not tracked by Git.
