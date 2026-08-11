# LIBERO suffix-only language adapter

This experiment trains a language-only residual adapter on the standard pi0.5
LIBERO action checkpoint. The action prefix, action expert, and all released
parameters remain frozen. The adapter is active only while processing causal
language suffix tokens appended to the same root KV cache used by the action
expert.

## Data protocol

- Target: `language.next` from the predicate-derived LIBERO annotations.
- Split: deterministic episode-level 45/5 train/validation split for every one
  of the 40 tasks (1,800/200 episodes). No frames from a validation episode
  occur in training.
- Sampling: retain frame 0, the final frame, every eighth frame, and both sides
  of every `next`-label transition. The current split contains 45,935 training
  and 5,075 validation frames.
- Prefix: the released pi0.5 LIBERO prompt format, including its
  `discrete_state_input=False` setting. Raw HDF5 images are rotated 180 degrees
  to match the official RLDS data and rollout preprocessing. State is retained
  in the observation and normalized with the checkpoint statistics, but is not
  tokenized into the prefix.
- Suffix: `Subtask: ` followed by the target and EOS. The final seed token
  predicts the first target token. A length of 20 covers the dataset maximum of
  18 tokens without truncation.

Build the split:

```bash
PYTHONPATH=src:. python -m experiments.libero_language_adapter.build_split \
  --annotation-root /path/to/libero_all_v3 \
  --dataset-root /path/to/raw_libero \
  --output /path/to/split.json
```

## Training

Only parameters matching `suffix_lora` are optimized. Released parameters are
kept in BF16; adapter parameters and AdamW optimizer state remain FP32. The
initial adapter output projection is zero, so enabling an untrained adapter is
identical to the original language path.

The initial sweep uses rank 8/16 and learning rate 1e-4/3e-4, with batch size 2,
2,000 steps, 100 warmup steps, cosine decay, AdamW (`b1=0.9`, `b2=0.95`, zero
weight decay), and gradient clipping at 1.0. These settings follow openpi's
parameter-efficient fine-tuning defaults while using a higher learning rate for
the newly initialized adapter only.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 PYTHONPATH=src:. \
  uv run python -m experiments.libero_language_adapter.train \
  --split /path/to/split.json \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  --output-dir /path/to/run \
  --rank 8 \
  --learning-rate 1e-4 \
  --steps 2000 \
  --warmup-steps 100
```

## Evaluation

`evaluate.py` performs greedy generation on held-out episodes, checks exact
action invariance with fixed diffusion noise, and profiles root-prefix latency
against the incremental per-token adapter cost. The quality samples are drawn
across all 40 tasks and favor distinct target states within each task.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 PYTHONPATH=src:. \
  uv run python -m experiments.libero_language_adapter.evaluate \
  --split /path/to/split.json \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/norm_stats.json \
  --adapter /path/to/adapter_step_2000.npz \
  --output-dir /path/to/evaluation \
  --rank 8 \
  --teacher-forced-samples 5075 \
  --action-invariance-samples 20
```

For simulator evaluation, `serve_adapter.py` loads the released checkpoint and
the selected adapter but exposes the unchanged 10-step action policy. Omitting
`--adapter` gives the paired original-checkpoint baseline with the same model
construction and preprocessing.

The formal run, including the held-out language metrics, action-isolation
check, runtime profile, and simulator rollout, is documented in
[RESULTS.md](RESULTS.md). The selected 2.3 MB adapter is stored at
[`checkpoints/r8_lr3e4_step2000.npz`](checkpoints/r8_lr3e4_step2000.npz); the
full pi0.5 checkpoint and LIBERO HDF5 data remain external.
