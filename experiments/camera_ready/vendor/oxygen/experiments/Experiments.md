# Experiments

This directory benchmarks frame-level latency for OxyGen under multi-task Vision-Language-Action inference. Given one camera observation and one prompt, the policy produces:

- actions through flow-matching denoising (`num_denoise_steps`)
- language tokens through autoregressive decoding (`max_decoding_steps`)

The experiment code compares four scheduling strategies:

| Setting               | Meaning                                                                             |
| --------------------- | ----------------------------------------------------------------------------------- |
| `baseline`            | Run `infer_actions()` and `infer_text()` sequentially for each request.             |
| `shared_kv`           | Run one shared prefill and reuse its KV cache for action and text heads.            |
| `continuous_batching` | OxyGen path: shared KV plus batched incremental text decode across active requests. |
| `parallel_mps`        | Run action and text workers concurrently with NVIDIA MPS.                           |

`parallel_mps` is primarily a server-GPU baseline. On Jetson Thor, the usual PyTorch sweep uses `baseline shared_kv continuous_batching`.

## Environment Setup

### Server GPU

From the repository root:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Use `uv run python -m ...` for server commands.

### Jetson AGX Thor

For Jetson AGX Thor, follow the [Jetson AI Lab OpenPi-on-Thor guide](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) through the container setup and PyTorch checkpoint verification steps. Stop before the ONNX, TensorRT, and NVFP4 quantization sections; OxyGen's PyTorch experiments run directly in PyTorch. Inside that container, run OxyGen commands with `python -m ...` rather than `uv run python -m ...`, unless you intentionally created a uv environment in the container.

### PyTorch Backend

`create_trained_policy()` selects the PyTorch backend when the checkpoint directory contains `model.safetensors`, or when `--pytorch-device` is provided.

You can benchmark PyTorch with random initialization and no checkpoint download:

```bash
OPENPI_TORCH_COMPILE=1 uv run python -m experiments.run_experiments \
    --settings baseline shared_kv continuous_batching \
    --policies pi05_o2_libero \
    --checkpoint-dir /tmp/oxygen_random_init_pytorch_checkpoint \
    --pytorch-device cuda:0 \
    --results-dir experiments/results_a100_pytorch \
    --gpu 0
```

Convert a JAX checkpoint once only if you need meaningful outputs:

```bash
python scripts/convert_jax_model_to_pytorch.py \
    --config-name pi05_o2_aloha \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
    --output-path ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
```

For real-checkpoint benchmark runs, enable PyTorch compilation:

```bash
OPENPI_TORCH_COMPILE=1 uv run python -m experiments.run_experiments \
    --settings baseline shared_kv continuous_batching \
    --policies pi05_o2_libero \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \
    --pytorch-device cuda:0 \
    --results-dir experiments/results_a100_pytorch \
    --gpu 0
```

Compilation flags:

| Env var                                   | Default when `OPENPI_TORCH_COMPILE=1` | Description                                          |
| ----------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| `OPENPI_TORCH_COMPILE_DENOISE`            | `1`                                   | Compile repeated action denoising.                   |
| `OPENPI_TORCH_COMPILE_PREFILL`            | `1`                                   | Compile VLM/language prefill.                        |
| `OPENPI_TORCH_COMPILE_TEXT_DECODE`        | `1`                                   | Compile one-token language decode.                   |
| `OPENPI_TORCH_COMPILE_DENOISE_CUDAGRAPHS` | `0`                                   | Opt into Inductor CUDA graphs for denoising.         |
| `OPENPI_TORCH_COMPILE_MODE`               | `reduce-overhead`                     | Denoising compile mode when CUDA graphs are enabled. |

Jetson Thor stable mode:

```bash
OPENPI_TORCH_COMPILE=1 \
OPENPI_TORCH_COMPILE_TEXT_DECODE=0 \
python -m experiments.run_experiments \
    --settings baseline shared_kv continuous_batching \
    --policies pi05_o2_aloha pi05_o2_droid pi05_o2_libero \
    --checkpoint-dir /tmp/oxygen_random_init_pytorch_checkpoint \
    --pytorch-device cuda:0 \
    --results-dir experiments/results_thor_pytorch \
    --gpu 0
```

On Thor, compiled one-token text decode has produced invalid sampling probabilities in some runs, so the stable path compiles denoise and prefill but keeps text decode eager.

## Running Experiments

### Result and Checkpoint Conventions

Example results directories are separated by device/backend:

| Directory                          | Intended use                                      |
| ---------------------------------- | ------------------------------------------------- |
| `experiments/results_4090_jax`     | RTX 4090 server run with the default JAX backend. |
| `experiments/results_a100_pytorch` | A100 server run with PyTorch checkpoints.         |
| `experiments/results_thor_pytorch` | Jetson AGX Thor run inside the container.         |
| `experiments/results`              | Small local/debug runs.                           |

Raw JSON files are written under:

```text
{results_dir}/{setting}/{policy_config}/denoise{N}_decode{M}.json
{results_dir}/continuous_batching/{policy_config}/denoise{N}_decode{M}_step{K}_{arrival_slug}.json
```

Analysis output is written under `{results_dir}/analysis/`; plots are written under `{results_dir}/plot/`.

Latency experiments do not use model output quality. To avoid downloading checkpoints, pass a nonexistent checkpoint path and let OxyGen initialize random weights:

```bash
--checkpoint-dir /tmp/oxygen_random_init_checkpoint
```

For PyTorch benchmark runs, also pass `--pytorch-device` so the nonexistent path selects the PyTorch backend:

```bash
--checkpoint-dir /tmp/oxygen_random_init_pytorch_checkpoint --pytorch-device cuda:0
```

Random-init outputs are garbage by design, but the benchmark still uses the requested policy config, tensor shapes, decoding lengths, and scheduling path. Use real checkpoints only when you need meaningful actions or language tokens.

### Full Server/JAX Run

```bash
uv run python -m experiments.run_experiments \
    --settings baseline shared_kv continuous_batching parallel_mps \
    --policies pi05_o2_aloha pi05_o2_droid pi05_o2_libero \
    --checkpoint-dir /tmp/oxygen_random_init_checkpoint \
    --results-dir experiments/results_4090_jax \
    --gpu 0

uv run python -m experiments.run_overhead \
    --gpu 0 \
    --results-dir experiments/results_4090_jax \
    --policy pi05_o2_libero \
    --checkpoint-dir /tmp/oxygen_random_init_checkpoint

uv run python -m experiments.run_workload_sweep \
    --gpu 0 \
    --results-dir experiments/results_4090_jax \
    --policy pi05_o2_libero \
    --checkpoint-dir /tmp/oxygen_random_init_checkpoint
```

### Full Jetson Thor/PyTorch Run

Run inside the container:

```bash
export PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH

OPENPI_TORCH_COMPILE=1 \
OPENPI_TORCH_COMPILE_TEXT_DECODE=0 \
python -m experiments.run_experiments \
    --settings baseline shared_kv continuous_batching \
    --policies pi05_o2_aloha pi05_o2_droid pi05_o2_libero \
    --checkpoint-dir /tmp/oxygen_random_init_pytorch_checkpoint \
    --pytorch-device cuda:0 \
    --results-dir experiments/results_thor_pytorch \
    --gpu 0
```

### Small Debug Run

```bash
uv run python -m experiments.run_experiments \
    --settings continuous_batching \
    --policies pi05_o2_libero \
    --num-denoise-steps 10 \
    --max-decoding-steps 10 \
    --steps-per-frame 5 \
    --total-frames 12 \
    --checkpoint-dir /tmp/oxygen_random_init_checkpoint \
    --results-dir experiments/results \
    --gpu 0
```

### Analysis and Plotting

Run these commands after each result directory is populated:

```bash
uv run python -m experiments.analysis.aggregate_metrics \
    --results-root-dir experiments/results_4090_jax
uv run python -m experiments.analysis.compute_speedup_cb \
    --results-root-dir experiments/results_4090_jax
uv run python -m experiments.analysis.compute_speedup_ablation \
    --results-root-dir experiments/results_4090_jax
uv run python -m experiments.analysis.compute_workload_sweep \
    --results-root-dir experiments/results_4090_jax \
    --policy pi05_o2_libero \
    --num-denoise-steps 10 \
    --max-decoding-steps 30
uv run python -m experiments.analysis.plot_all \
    --results-root-dir experiments/results_4090_jax \
    --steps-per-frame 5 \
    --num-denoise-steps 10 \
    --policy all
```

Inside the Jetson container, use the same module names with `python -m`:

```bash
python -m experiments.analysis.aggregate_metrics \
    --results-root-dir experiments/results_thor_pytorch
python -m experiments.analysis.compute_speedup_cb \
    --results-root-dir experiments/results_thor_pytorch
python -m experiments.analysis.compute_speedup_ablation \
    --results-root-dir experiments/results_thor_pytorch
python -m experiments.analysis.plot_all \
    --results-root-dir experiments/results_thor_pytorch \
    --steps-per-frame 5 \
    --num-denoise-steps 10 \
    --policy all
```

For device-comparison figures, pass one or more result roots in display order. Known result-directory names are mapped to readable labels in `experiments/analysis/plot_utils.py`; unknown names fall back to the directory name. Use `--device-label` when you want paper-style labels for custom result directories, and pass one `--x-max` value per result root when drawing the combined ablation figure.

```bash
uv run python -m experiments.analysis.plot_tradeoff \
    --results-root experiments/results_my_gpu_jax experiments/results_my_edge_pytorch \
    --device-label "My GPU" "My Edge Device" \
    --policy pi05_o2_aloha

uv run python -m experiments.analysis.plot_combined_ablation \
    --results-root experiments/results_my_gpu_jax experiments/results_my_edge_pytorch \
    --device-label "My GPU" "My Edge Device" \
    --x-max 80 40 \
    --policy pi05_o2_libero
```

If one result directory or one policy is missing, these scripts skip the missing panel and still write a figure for the available data. For the combined ablation figure, add `--parallel-on-top` if the Parallel bar is slower than Baseline and should be drawn in front.

## Reference

### Metrics

Runners record raw per-frame data:

| Field                           | Meaning                                                         |
| ------------------------------- | --------------------------------------------------------------- |
| `frame_ms`                      | Wall-clock frame time.                                          |
| `total_tokens_this_frame`       | Language tokens generated by all active requests in that frame. |
| `n_new`, `n_resumed`, `n_total` | Request counts for continuous batching.                         |
| `policy_timing`                 | Backend-specific timing breakdown.                              |
| `is_warmup`                     | Warmup frames excluded by analysis scripts.                     |
| `completed_requests`            | Continuous-batching request latency records.                    |

Analysis scripts compute:

| Metric                    | Meaning                                                            |
| ------------------------- | ------------------------------------------------------------------ |
| `frame_latency_ms`        | Mean non-warmup frame latency.                                     |
| `action_frequency_hz`     | `ACTION_HORIZON / frame_latency`.                                  |
| `language_throughput_tps` | Mean language tokens per second.                                   |
| `avg_batch_size`          | Mean active batch size for continuous batching.                    |
| `avg_request_wall_ms`     | Mean request wall time for completed continuous-batching requests. |

### CLI Reference

`experiments.run_experiments`:

| Flag                   | Default                    | Description                                                          |
| ---------------------- | -------------------------- | -------------------------------------------------------------------- |
| `--settings`           | all settings               | Any subset of `baseline shared_kv continuous_batching parallel_mps`. |
| `--policies`           | `pi05_o2_droid`            | Policy config names to sweep.                                        |
| `--results-dir`        | `experiments/results`      | Raw result output directory.                                         |
| `--gpu`                | `0`                        | GPU index; sets `CUDA_VISIBLE_DEVICES`.                              |
| `--checkpoint-dir`     | unset                      | Override checkpoint directory. `model.safetensors` selects PyTorch.  |
| `--pytorch-device`     | unset                      | PyTorch device such as `cuda:0` or `cpu`.                            |
| `--num-denoise-steps`  | default sweep              | Override denoise-step values.                                        |
| `--max-decoding-steps` | default sweep              | Override decode-step values.                                         |
| `--steps-per-frame`    | default sweep              | Continuous-batching text tokens per frame.                           |
| `--total-frames`       | `50`                       | Continuous-batching simulation length.                               |
| `--arrival-pattern`    | `uniform_arrivals(rate=1)` | Continuous-batching request arrival pattern.                         |
| `--num-measured-runs`  | `3`                        | Measured runs for non-CB settings.                                   |
| `--warmup-runs`        | `1`                        | Warmup runs for non-CB settings.                                     |

Default sweeps are defined in `experiments/run_experiments.py`. For continuous batching, invalid combinations such as `max_decoding_steps % steps_per_frame != 0` are skipped with a warning.

### Implementation Notes

- `run_experiments.py` launches each setting in a subprocess when multiple settings are requested, which releases GPU memory between settings.
- `parallel_mps` starts and stops the NVIDIA MPS daemon automatically when run through `run_experiments.py`.
- PyTorch continuous batching uses a fixed-size text KV cache and advances active requests in one batched decode call.
- Inductor-compiled prefill is for performance benchmarking. Use `OPENPI_TORCH_COMPILE_PREFILL=0` for stricter eager-prefill parity checks.
