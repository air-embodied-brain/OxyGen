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
  episodes (45,935 and 5,075 sampled frames).
- Prefix: the released pi0.5 LIBERO observation, state, and task-prompt protocol.
- Suffix: `Subtask: <next label> EOS`, padded to 20 tokens without truncation.

Build the deterministic split:

```bash
python -m experiments.libero_language_adapter.build_split \
  --annotation-root /path/to/libero_all_v3 \
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
training and block-plus-token incremental inference. The selected suffix model
therefore receives a short 500-step continuation through the exact deployed
incremental path; no LoRA targets are removed.

## Evaluation

`evaluate.py` reports full held-out teacher-forced metrics, greedy incremental
generation across 80 samples, root/action isolation with fixed diffusion noise,
and adapter latency relative to the duplicate prefix forward it replaces.
`verify_serving.py` calls the actual continuous-batching policy and checks that
one request returns both actions and adapter-generated language.

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_ALLOCATOR=platform \
python -m experiments.libero_language_adapter.evaluate \
  --split /path/to/split.json \
  --checkpoint /path/to/pi05_libero \
  --norm-stats /path/to/norm_stats.json \
  --adapter /path/to/adapter_step_400.npz \
  --adapter-type suffix_lora --rank 16 --alpha 16 \
  --teacher-forced-samples 5075 --action-invariance-samples 20 \
  --output-dir /path/to/evaluation
```

See [RESULTS.md](RESULTS.md) for the findings and
[`aggregate_summary.json`](results/2026-08-11/three_way_accuracy/aggregate_summary.json)
for machine-readable metrics. Large checkpoints and datasets remain local and
are not tracked by Git.
