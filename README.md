# OxyGen

This repository contains code for paper [*OxyGen: Unified KV Cache Management for Vision-Language-Action Models under Multi-Task Parallelism*](https://arxiv.org/abs/2603.14371). It is built atop [openpi](https://github.com/Physical-Intelligence/openpi).

OxyGen optimizes multi-task inference for Mixture-of-Transformers (MoT) Vision-Language-Action (VLA) models (e.g., pi0.5) through unified KV cache management, with cross-task KV sharing and cross-frame continuous batching.

## News

- **2026-05:** Added PyTorch inference support with primary `torch.compile` optimization, verified on NVIDIA A100 and Jetson AGX Thor.

## Requirements

- Ubuntu 22.04 LTS or later
- NVIDIA GPU (>8 GB VRAM for inference; >16 GB recommended for full experiments)
- Python 3.11, managed by [uv](https://docs.astral.sh/uv/)

Tested environment for the JAX backend:

| Component | Version |
| --- | --- |
| CPU | Intel i9-13900K |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| OS | Ubuntu 24.04.2 LTS |
| NVIDIA driver | 580.126.09 |
| CUDA | 13.0 |

Official checkpoints from openpi are downloaded automatically to `~/.cache/openpi` when needed:

- pi05-DROID: `gs://openpi-assets/checkpoints/pi05_droid`
- pi05-LIBERO: `gs://openpi-assets/checkpoints/pi05_libero`
- pi05-BASE: `gs://openpi-assets/checkpoints/pi05_base`

## Installation

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` is needed to pull the [LeRobot](https://github.com/huggingface/lerobot) dependency without downloading large LFS payloads. An optional Docker workflow is documented in [`docs/docker.md`](docs/docker.md).

## Quick Test

The command below loads the pi0.5 base checkpoint through the `pi05_o2_arx` config, creates one synthetic observation, and prints the generated language. The first run downloads `pi05_base` to `~/.cache/openpi`.

```bash
uv run python - <<'PY'
from experiments.common.setup import create_policy
from experiments.common.workload import create_synthetic_observation

policy = create_policy("pi05_o2_arx")
obs = create_synthetic_observation("pick up the red cup", seed=0, policy_config="pi05_o2_arx")
out = policy.infer_text(obs, max_decoding_steps=20)
print(out["text"])
PY
```

For full benchmark commands, random-initialized latency runs, PyTorch support, Jetson AGX Thor setup, analysis, and plotting, see [`experiments/Experiments.md`](experiments/Experiments.md).

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
