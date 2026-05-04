# OxyGen

This repository contains code for paper [*OxyGen: Unified KV Cache Management for Vision-Language-Action Models under Multi-Task Parallelism*](https://arxiv.org/abs/2603.14371). It is built atop [openpi](https://github.com/Physical-Intelligence/openpi).

OxyGen optimizes multi-task inference for Mixture-of-Transformers (MoT) Vision-Language-Action (VLA) models through a unified KV cache management, with cross-task KV sharing and cross-frame continuous batching.

## Requirements

- Ubuntu 22.04 LTS (or later)
- NVIDIA GPU (>8 GB VRAM for inference; >16 GB recommended for full experiments)
- Python 3.11, managed by [uv](https://docs.astral.sh/uv/)

Tested environment:

| Component | Version |
|-----------|---------|
| CPU | Intel i9-13900K |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| OS | Ubuntu 24.04.2 LTS |
| NVIDIA driver | 580.126.09 |
| CUDA | 13.0 |

Official checkpoints from openpi (downloaded automatically to `~/.cache/openpi`):

- pi05-DROID: `gs://openpi-assets/checkpoints/pi05_droid`
- pi05-LIBERO: `gs://openpi-assets/checkpoints/pi05_libero`
- pi05-BASE: `gs://openpi-assets/checkpoints/pi05_base`

## Installation

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` is needed to pull the [LeRobot](https://github.com/huggingface/lerobot) dependency without downloading large LFS payloads.

An optional Docker workflow is documented in [`docs/docker.md`](docs/docker.md).

## Experiments

### Running experiments

Run the full set of experiments from the paper (results saved to `experiments/results_eccv`):

```bash
# Main experiments: all settings and benchmarks
uv run python -m experiments.run_experiments \
    --settings baseline shared_kv continuous_batching parallel_mps \
    --policies pi05_o2_aloha pi05_o2_droid pi05_o2_libero \
    --results-dir experiments/results_eccv --gpu 0

# Memory and energy measurements (LIBERO only)
uv run python -m experiments.run_overhead \
    --gpu 0 --results-dir experiments/results_eccv --policy pi05_o2_libero

# Workload sweep (LIBERO only)
uv run python -m experiments.run_workload_sweep \
    --gpu 0 --results-dir experiments/results_eccv --policy pi05_o2_libero
```

The first run takes roughly 3+ hours (checkpoint download + JAX compilation). Subsequent runs are faster. Experiments require exclusive GPU access.

To run a smaller subset, specify settings or policies via CLI args. For example:

```bash
# Baseline on LIBERO
uv run python -m experiments.run_experiments \
    --settings baseline --policies pi05_o2_libero --gpu 0

# Continuous batching ("Ours") on LIBERO
uv run python -m experiments.run_experiments \
    --settings continuous_batching --policies pi05_o2_libero --gpu 0
```

### PyTorch checkpoints

If a checkpoint directory contains `model.safetensors`, `create_trained_policy` loads the PyTorch backend automatically. You can run the continuous batching experiment against a local PyTorch checkpoint with:

```bash
uv run python -m experiments.run_experiments \
    --settings continuous_batching --policies pi05_o2_libero --gpu 0 \
    --checkpoint-dir /path/to/pytorch/checkpoint --pytorch-device cuda:0
```

The PyTorch backend currently supports action inference, text inference, and continuous batching for `pi05_o2_*` configs. New requests are batched for prefill and action generation; active text states use a fixed-size PyTorch KV cache and are decoded as one batch.

For performance benchmarking, enable PyTorch compilation:

```bash
OPENPI_TORCH_COMPILE=1 uv run python -m experiments.run_experiments \
    --settings baseline shared_kv continuous_batching \
    --policies pi05_o2_libero --gpu 0 \
    --checkpoint-dir /path/to/pytorch/checkpoint --pytorch-device cuda:0
```

`OPENPI_TORCH_COMPILE=1` compiles PI05 denoising, VLM prefill, and language decoding. To keep prefill eager while compiling the other hot paths, set `OPENPI_TORCH_COMPILE_PREFILL=0`. Continuous batching uses a fixed-size PyTorch text KV cache so active requests are decoded as one batch.

Other settings: `shared_kv` ("Ours w/o Batching" in the ablation) and `parallel_mps` ("Parallel").

### Analysis and plotting

Generate all paper figures and tables (no GPU needed):

```bash
uv run python -m experiments.analysis.aggregate_metrics \
    --results-root-dir experiments/results_eccv
uv run python -m experiments.analysis.compute_speedup_cb \
    --results-root-dir experiments/results_eccv
uv run python -m experiments.analysis.compute_speedup_ablation \
    --results-root-dir experiments/results_eccv
uv run python -m experiments.analysis.compute_workload_sweep \
    --results-root-dir experiments/results_eccv \
    --policy pi05_o2_libero --num-denoise-steps 10 --max-decoding-steps 30
uv run python -m experiments.analysis.plot_all \
    --results-root-dir experiments/results_eccv \
    --steps-per-frame 5 --num-denoise-steps 10 --policy all
```

Outputs:
- `experiments/results_eccv/plot/` — paper figures (PDF) and summary tables (CSV)
- `experiments/analysis/overhead/overhead.csv` — memory and energy measurements

### CLI reference

`experiments.run_experiments` supports:

| Flag | Description |
|------|-------------|
| `--settings` | Subset of `baseline shared_kv continuous_batching parallel_mps` |
| `--policies` | One or more policy config names |
| `--results-dir` | Output root (default: `experiments/results`) |
| `--gpu` | GPU id (default: `0`) |
| `--prompt` | Text prompt for synthetic observations |
| `--checkpoint-dir` | Override checkpoint directory; `model.safetensors` selects PyTorch |
| `--pytorch-device` | PyTorch device override, e.g. `cuda:0` |

## Acknowledgments

This codebase is built on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence. We thank the openpi team for open-sourcing their code and models.

We use [openpi_subtask_generation](https://github.com/BrunoFANG1/openpi_subtask_generation), a community reproduction of pi05 subtask prediction, for the initial language decoding implementation. We thank @BrunoFANG1 and other contributors for open-sourcing their code.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details. The Gemma model components are subject to [additional terms](LICENSE_GEMMA.txt).

## Citation

If you find this project useful, please cite our [paper](https://arxiv.org/abs/2603.14371):

```bibtex
@article{li2026oxygen,
  title={OxyGen: Unified KV Cache Management for Vision-Language-Action Models under Multi-Task Parallelism},
  author={Li, Xiangyu and Tang, Huaizhi and Ding, Xin and Wang, Weijun and Cao, Ting and Liu, Yunxin},
  journal={arXiv preprint arXiv:2603.14371},
  year={2026}
}
```
