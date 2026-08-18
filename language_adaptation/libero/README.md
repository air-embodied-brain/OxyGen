# LIBERO textual-memory adaptation

This feature trains pi0.5 to produce observation-grounded textual memory while
keeping the action expert frozen. For each observation, the model can produce a
short memory such as:

```text
Memory: The alphabet soup is in the basket.
```

The observation, robot state, and task instruction form one full-attention
prefix evaluated with frozen base weights. The action expert consumes this
root KV cache. A rank-16 LoRA is active only for the private `Memory: ` seed
and the autoregressive suffix, so training cannot alter the shared prefix or
action output.

## What is included

- deterministic episode-level train/validation splitting over raw LIBERO HDF5;
- suffix-LoRA training through the same incremental path used at inference;
- held-out text metrics, root-KV and action invariance checks, and adapter
  latency measurement;
- OxyGen continuous batching and an isolated baseline with separate action and
  language prefix forwards.

Predicate extraction and text-label construction are documented in
[`supervision`](supervision/README.md).
Generated annotations, checkpoints, and videos are intentionally kept outside
Git.

The selected derived annotations are available from the
[LIBERO textual-memory dataset release](https://huggingface.co/datasets/xxxxyu/libero-textual-memory-annotations),
and the trained adapter is available from the
[pi0.5 model release](https://huggingface.co/xxxxyu/oxygen-pi05-textual-memory-lora).

## Required assets

Prepare these paths before running the experiment:

1. raw HDF5 demonstrations for the four standard LIBERO suites;
2. annotations produced by the supervision pipeline;
3. the released pi0.5 LIBERO checkpoint, including its `norm_stats.json`.

The code assumes the raw LIBERO layout below:

```text
raw_libero/
  libero_spatial/*_demo.hdf5
  libero_object/*_demo.hdf5
  libero_goal/*_demo.hdf5
  libero_10/*_demo.hdf5
```

## Build the split

The default split holds out five demonstrations per task. Frames are sampled
every eight simulator steps, with additional samples immediately around each
change in the textual-memory target.

```bash
uv run python -m language_adaptation.libero.build_split \
  --annotation-root /path/to/annotations \
  --dataset-root /path/to/raw_libero \
  --output /path/to/textual_memory_split.json
```

The split records its random seed, held-out episodes, and source paths. Keep it
with the trained adapter so evaluation uses the same episode partition.

## Train

The defaults reproduce the selected configuration: rank 16, learning rate
`1e-4`, 1,500 steps, a 28-token suffix, and task/target-balanced sampling.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
uv run python -m language_adaptation.libero.train \
  --split /path/to/textual_memory_split.json \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --output-dir /path/to/textual_memory_run
```

Only LoRA parameters in the 18 language layers are optimized. Each checkpoint
is a standalone `.npz` adapter; frozen base weights are not copied into the run
directory. Training teacher-forces the seed and target one token at a time
after a frozen root prefill. This matches serving and prevents the adapter from
entering the root KV cache.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform \
uv run python -m language_adaptation.libero.evaluate \
  --split /path/to/textual_memory_split.json \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --adapter /path/to/textual_memory_run/adapter_step_1500.npz \
  --teacher-forced-samples 5082 \
  --samples-per-task 10 \
  --output-dir /path/to/textual_memory_eval
```

Evaluation reports token accuracy, greedy exact match and word F1, then repeats
the same observations with the adapter zeroed. With fixed diffusion noise, the
root KV cache and action tensor must be exactly equal between the enabled and
zeroed-adapter paths. It also compares enabled and disabled suffix latency with
the shared prefix forward, checking that adapter overhead does not erase the
computation saved by prefix reuse.

## Serve

Start OxyGen with one new memory request at every action replan. Unfinished
requests are resumed together in the next language batch.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m language_adaptation.libero.serve_adapter \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --adapter /path/to/textual_memory_run/adapter_step_1500.npz \
  --request-mode new_each_call \
  --steps-per-frame 1 \
  --execution oxygen \
  --port 8011
```

For an isolated baseline under the same model and decoding settings, change
`--execution oxygen` to `--execution blocking_baseline`. The baseline computes
separate prefixes for action and language and finishes the language request
before returning the action chunk.

The server returns the action chunk as usual. `language_updates` contains the
new tokens, full text, request ID, and completion status for every memory
request advanced during that call.

## Reference result

The selected run used all 2,000 demonstrations and 338,575 annotated frames.
The results below are a compact record of that run; generated JSON summaries
remain local so the repository does not become an experiment-results archive.

| Metric | Value |
| --- | ---: |
| Held-out teacher-forced token accuracy | 98.15% |
| Held-out greedy exact match | 87.25% |
| Held-out greedy word F1 | 90.01% |
| Root KV max difference with adapter zeroed | 0 |
| Fixed-noise action max difference with adapter zeroed | 0 |
| Median 11-token suffix LoRA overhead (RTX 4090) | 8.16 ms |
| Shared prefix forward (RTX 4090) | 47.66 ms |

These measurements establish feasibility for this workload; they are not a
full LIBERO success-rate result. Three successful qualitative rollouts are
also reported in the local evaluation record, but are not a success-rate
estimate.

## Tests

```bash
uv run pytest --import-mode=importlib -q \
  language_adaptation/libero/supervision/tests \
  src/openpi/models/lora_test.py \
  src/openpi/models/tokenizer_test.py \
  src/openpi/policies/policy_test.py \
  src/openpi/serving/websocket_policy_server_test.py
```
