# Experiments

Frame-level latency benchmarking for four VLA (Vision-Language-Action) execution strategies. Given a camera observation, the policy produces:

- **Actions** (robot motor commands) via flow-matching denoising (`num_denoise_steps` iterations)
- **Language** (text) via autoregressive token decoding (`max_decoding_steps` tokens)

All four strategies produce the same outputs but differ in how inference calls are scheduled on the GPU.

## Directory structure

```
experiments/
├── run_experiments.py              # Top-level CLI orchestrator
├── common/
│   ├── setup.py                    # create_policy(), setup_jax_cache(), collect_metadata()
│   └── workload.py                 # create_synthetic_observation(), arrival pattern generators
├── baseline/
│   ├── runner.py                   # run_baseline() — sequential infer_actions + infer_text
│   └── grid_search.py             # Grid search over (policy_config, denoise, decode)
├── shared_kv/
│   ├── runner.py                   # run_shared_kv() — single prefill, shared KV cache
│   └── grid_search.py             # Grid search over (policy_config, denoise, decode)
├── continuous_batching/
│   ├── runner.py                   # run_continuous_batching() — batched incremental text
│   └── grid_search.py             # Grid search over (policy_config, denoise, decode, steps_per_frame)
├── parallel_mps/
│   ├── runner.py                   # MPSWorkerPool — two processes on one GPU via NVIDIA MPS
│   └── grid_search.py             # Grid search + MPS daemon lifecycle
└── results/                        # Raw JSON results (gitignored)
```

## The four settings

| Setting | Key idea | Inference function |
|---|---|---|
| `baseline` | Sequential isolated execution | `policy.infer_actions()` then `policy.infer_text()` called separately, one request at a time |
| `shared_kv` | Single prefill shared by text and action heads | `policy.infer_text_actions_shared_kv()`, one request at a time |
| `continuous_batching` | Shared KV + incremental batched text | `policy.infer_text_actions_continuous_batch()`: new requests get prefill+actions, resumed requests get incremental text generation. Multiple concurrent requests per frame |
| `parallel_mps` | Two worker processes on the same GPU via NVIDIA MPS | `infer_text` and `infer_actions` run simultaneously in separate processes (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45` each) |

## Metrics

Runners record raw data per frame. Analysis scripts compute derived metrics.

**Raw data recorded by runners (per frame):**
- `frame_ms`: wall time of the frame
- `total_tokens_this_frame`: total language tokens generated across all active requests this frame
- `n_new`, `n_resumed`, `n_total`: request counts
- `policy_timing`: internal breakdown from the inference function (`prefill_actions_ms`, `text_gen_ms`, etc.)
- For continuous batching: `total_wall_ms` per completed request (arrival frame to finish frame)

**Derived metrics (computed by analysis scripts):**
- Frame latency (ms): `avg(frame_ms)` over steady-state frames. Lower = higher action frequency.
- Language throughput (tokens/s): `total_tokens_this_frame / frame_ms * 1000`, averaged over steady-state frames.
- For continuous batching: avg batch size, avg request wall latency.

## Quick start

### Run all settings on GPU 0

```bash
uv run python -m experiments.run_experiments --gpu 0
```

### Run a single setting with a small search space

```bash
uv run python -m experiments.run_experiments \
    --settings baseline \
    --policies pi05_o2_droid \
    --gpu 0
```

### Run PyTorch backend experiments

Pass a checkpoint directory containing `model.safetensors` to select the PyTorch backend. For benchmark runs, enable PyTorch compilation:

```bash
OPENPI_TORCH_COMPILE=1 uv run python -m experiments.run_experiments \
    --settings baseline shared_kv continuous_batching \
    --policies pi05_o2_arx \
    --gpu 0 \
    --checkpoint-dir /home/lixiangyu/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch \
    --pytorch-device cuda:0 \
    --num-denoise-steps 10 \
    --max-decoding-steps 5 \
    --steps-per-frame 1 \
    --total-frames 8 \
    --arrival-pattern 'uniform_arrivals(rate=1,t_max=5)'
```

`OPENPI_TORCH_COMPILE=1` compiles the PI05 denoising step, VLM prefill, and one-token language decode. The individual switches are:

| Env var | Default when `OPENPI_TORCH_COMPILE=1` | Description |
|---|---|---|
| `OPENPI_TORCH_COMPILE_DENOISE` | `1` | Compile the repeated action denoising step |
| `OPENPI_TORCH_COMPILE_PREFILL` | `1` | Compile VLM/language prefill |
| `OPENPI_TORCH_COMPILE_TEXT_DECODE` | `1` | Compile one-token autoregressive language decode |
| `OPENPI_TORCH_COMPILE_MODE` | `reduce-overhead` | Mode passed to `torch.compile` for denoising |

Set a component flag to `0` to disable it, for example:

```bash
OPENPI_TORCH_COMPILE=1 OPENPI_TORCH_COMPILE_PREFILL=0 uv run python -m experiments.run_experiments ...
```

The PyTorch continuous batching path uses a fixed-size text KV cache and advances all active requests in one batched decode call. New requests still run prefill and action generation as a batch.

Note: Inductor-compiled prefill is intended for performance benchmarking. It can shift BF16 logits relative to eager prefill, especially when the top tokens are very close. Use `OPENPI_TORCH_COMPILE_PREFILL=0` for stricter eager-prefill parity checks.

### Run a single setting's grid_search directly (for debugging)

```python
from experiments.common.setup import create_policy, setup_jax_cache
from experiments.baseline.grid_search import run_grid_search
from pathlib import Path

setup_jax_cache()
policy = create_policy("pi05_o2_droid")

results = run_grid_search(
    policy=policy,
    search_space={
        "policy_config": ["pi05_o2_droid"],
        "num_denoise_steps": [10],
        "max_decoding_steps": [5],
    },
    fixed_params={"prompt": "pick the red cup"},
    results_dir=Path("experiments/results"),
    num_measured_runs=3,
    warmup_runs=1,
)
```

### parallel_mps: MPS daemon required

When calling `parallel_mps/grid_search.py` directly (not via `run_experiments.py`), you must manage the MPS daemon yourself:

```python
from experiments.parallel_mps.grid_search import start_mps, stop_mps, run_grid_search

start_mps(gpu_id=0)
try:
    results = run_grid_search(
        policy=None,  # ignored — workers create their own policies
        search_space={...},
        fixed_params={"prompt": "pick the red cup"},
        results_dir=Path("experiments/results"),
    )
finally:
    stop_mps()
```

`run_experiments.py` handles `start_mps` / `stop_mps` automatically.

## CLI reference (`run_experiments.py`)

| Flag | Default | Description |
|---|---|---|
| `--settings` | all four | Which settings to run. Choices: `baseline`, `shared_kv`, `continuous_batching`, `parallel_mps` |
| `--policies` | `pi05_o2_droid` | Policy config names to sweep (space-separated) |
| `--results-dir` | `experiments/results` | Root directory for saving result JSON files |
| `--gpu` | `0` | GPU index (`CUDA_VISIBLE_DEVICES` is set internally) |
| `--prompt` | `"pick the red cup"` | Prompt string for synthetic observations |

Example with multiple policies:

```bash
uv run python -m experiments.run_experiments \
    --settings baseline shared_kv \
    --policies pi05_o2_droid pi05_o2_aloha pi05_o2_libero \
    --gpu 1
```

## Grid search design

Each setting folder exposes a `grid_search.py` with a common interface:

```python
def run_grid_search(
    policy,                    # Already-initialized Policy object (None for parallel_mps)
    search_space: dict,        # Parameter name -> list of values (cartesian product)
    fixed_params: dict,        # Parameters held constant (e.g. prompt)
    results_dir: Path,         # Where to save raw JSON results
    num_measured_runs: int = 3,  # Repeated measurements per grid point
    warmup_runs: int = 1,       # Discarded warmup runs per grid point
) -> list[dict]:
```

Note: `continuous_batching` does not accept `num_measured_runs`/`warmup_runs` — it uses `total_frames` and auto-computed `warmup_frames` instead.

### Measurement protocol

- Each grid point runs `warmup_runs` iterations (discarded) + `num_measured_runs` iterations (recorded).
- For continuous_batching: skip the first `warmup_frames` frames (ramp-up before steady state). No wind-down phase — measurement runs until `total_frames` and stops.
- Record wall-clock time and internal `policy_timing` breakdown for each call.
- Use `jax.block_until_ready()` on outputs before stopping the timer (JAX ops are async).
- Skip invalid parameter combinations (e.g., `max_decoding_steps % steps_per_frame != 0`) with a warning log — no error.

## Search space configuration

### Default search spaces

Defined as constants in `run_experiments.py`:

```python
# Shared across all settings
DEFAULT_SEARCH_SPACE = {
    "policy_config": ["pi05_o2_droid"],       # overridden by --policies
    "num_denoise_steps": [5, 10, 15, 20],
    "max_decoding_steps": [1, 2, 3, 4, 5, 10, 15, 20],
}

# Extra axes for continuous_batching only
CB_SEARCH_SPACE_EXTRA = {
    "steps_per_frame": [1, 2, 5],
}

CB_FIXED_PARAMS = {
    "total_frames": 50,
    "arrival_pattern": "uniform_arrivals(rate=1)",
}
```

### Full search spaces per setting (from design spec)

**baseline / shared_kv / parallel_mps** all share the same axes:
```python
search_space = {
    "policy_config": ["pi05_o2_droid", "pi05_o2_aloha", "pi05_o2_libero"],
    "num_denoise_steps": [5, 10, 15, 20],
    "max_decoding_steps": [1, 2, 3, 4, 5, 10, 15, 20],
}
```

**continuous_batching** adds `steps_per_frame`:
```python
search_space = {
    "policy_config": ["pi05_o2_droid", "pi05_o2_aloha", "pi05_o2_libero"],
    "num_denoise_steps": [10],
    "max_decoding_steps": [5, 10, 15, 25, 30],
    "steps_per_frame": [3, 5],
}
fixed_params = {
    "total_frames": 30,
    "arrival_pattern": "uniform_arrivals(rate=1)",
}
# Constraint: max_decoding_steps % steps_per_frame == 0 (invalid combos skipped)
```

### Customization

- **Policies**: use `--policies` on the CLI.
- **Other axes**: edit the constants in `run_experiments.py` directly.

## Result format

Each grid point saves a JSON file. Example (baseline):

```json
{
    "setting": "baseline",
    "params": {
        "policy_config": "pi05_o2_droid",
        "num_denoise_steps": 10,
        "max_decoding_steps": 20
    },
    "frames": [
        {
            "frame_idx": 0,
            "frame_ms": 45.2,
            "total_tokens_this_frame": 20,
            "n_new": 1,
            "n_resumed": 0,
            "n_total": 1,
            "policy_timing": {"actions_wall_ms": 25.1, "text_wall_ms": 18.3},
            "is_warmup": true
        }
    ],
    "gpu_monitor": [],
    "metadata": {
        "timestamp": "20260224_120000",
        "gpu": "NVIDIA A100-SXM4-80GB",
        "jax_version": "0.4.x",
        "num_measured_runs": 3,
        "warmup_runs": 1
    }
}
```

For `continuous_batching`, the result also includes `completed_requests`:

```json
{
    "setting": "continuous_batching",
    "params": {
        "policy_config": "pi05_o2_droid",
        "num_denoise_steps": 10,
        "max_decoding_steps": 20,
        "steps_per_frame": 5,
        "arrival_pattern": "uniform_arrivals(rate=1)"
    },
    "frames": [
        {
            "frame_idx": 0,
            "frame_ms": 52.3,
            "total_tokens_this_frame": 5,
            "n_new": 1,
            "n_resumed": 0,
            "n_total": 1,
            "policy_timing": {"new_requests": 1, "resumed_requests": 0, "batch_size": 1},
            "is_warmup": true
        }
    ],
    "completed_requests": [
        {"request_id": "r0", "arrival_frame": 0, "finish_frame": 3, "total_wall_ms": 135.0}
    ],
    "gpu_monitor": [],
    "metadata": {
        "timestamp": "20260224_120000",
        "gpu": "NVIDIA A100-SXM4-80GB",
        "jax_version": "0.4.x",
        "total_frames": 50,
        "warmup_frames": 4
    }
}
```

### Field notes

- `is_warmup`: frames where this is `true` should be excluded from analysis.
- `completed_requests`: only present in `continuous_batching` results. Each entry has `request_id`, `arrival_frame`, `finish_frame`, `total_wall_ms`. The other three settings do not include this field.
- `gpu_monitor`: TODO — reserved for future `--gpu-monitor` CLI flag. Currently always `[]`.
- For `baseline`/`shared_kv`/`parallel_mps`: each "frame" = one full inference call. `total_tokens_this_frame` = `max_decoding_steps`.

### File naming convention

```
{results_dir}/{setting}/{policy_config}/denoise{N}_decode{M}.json
```

For `continuous_batching`, the filename includes extra parameters:

```
{results_dir}/continuous_batching/{policy_config}/denoise{N}_decode{M}_step{K}_{arrival_slug}.json
```

Where `{arrival_slug}` is derived from the arrival pattern string (e.g., `uniform1`, `poisson2.0`, `bursty4_8`).

## Setting-specific notes

### parallel_mps

- Uses NVIDIA MPS (Multi-Process Service) to run two worker processes on the same GPU, each with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`.
- `run_experiments.py` calls `start_mps(gpu_id)` before and `stop_mps()` after the grid search automatically.
- When calling `parallel_mps/grid_search.py` directly, you must call `start_mps()` / `stop_mps()` yourself (see Quick start above).
- The `policy` argument to `run_grid_search()` is ignored — each worker process creates its own policy internally via `MPSWorkerPool`.
- Workers are grouped by `policy_config` to avoid redundant policy loading. The pool is created once per policy config and reused across all grid points for that config.
- `start_mps()` cleans up any existing MPS processes before starting, sets `CUDA_MPS_PIPE_DIRECTORY` and `CUDA_MPS_LOG_DIRECTORY` to user-specific `/tmp` paths.
- GPU device remapping: `start_mps()` starts the daemon with the real GPU ID (e.g. `CUDA_VISIBLE_DEVICES=5`). Once MPS virtualizes that GPU, it appears as device 0 to child processes. Each worker sets `CUDA_VISIBLE_DEVICES=0` before importing JAX so it addresses the correct MPS-virtualized device.

### continuous_batching

- Extra search axis: `steps_per_frame` — how many text tokens are generated per request per frame.
- Extra fixed params: `total_frames` (simulation length), `arrival_pattern` (string like `"uniform_arrivals(rate=1)"`).
- `warmup_frames` is auto-computed as `max_decoding_steps // steps_per_frame` (the ramp-up period before steady state).
- Before the grid search, batch-size warmup is performed: JAX is pre-compiled for batch sizes 1 through the estimated peak batch size. Peak batch size is estimated by simulating the arrival pattern for `total_frames` frames (run 5 times for stochastic patterns like poisson, +1 safety margin).
- Each grid point runs the full simulation twice — once for warmup (discarded), once for measurement.
- Output includes `completed_requests` with per-request wall latency (`arrival_frame`, `finish_frame`, `total_wall_ms`).
- Constraint: `max_decoding_steps % steps_per_frame == 0` must hold. Invalid combinations are skipped with a warning log.

## Common utilities

### `common/setup.py`

| Function | Description |
|---|---|
| `setup_jax_cache(cache_dir=None)` | Enable persistent JAX compilation cache (default: `~/.cache/jax_compilation_cache`) |
| `create_policy(policy_config)` | Download checkpoint via `openpi.shared.download` and create an initialized Policy object via `create_trained_policy()` |
| `resolve_policy_checkpoint(policy_config)` | Map config name to GCS checkpoint path and download. `"droid"` → `pi05_droid`, `"libero"` → `pi05_libero`, else → `pi05_base` |
| `collect_metadata()` | Return dict with `timestamp`, `gpu` name (from `nvidia-smi`), `jax_version` |

### `common/workload.py`

| Function | Description |
|---|---|
| `create_synthetic_observation(prompt, seed, policy_config)` | Create one synthetic observation dict. Platform auto-detected from config name substring (`arx`/`droid`/`aloha`/`libero`) |
| `generate_workload(num_requests, prompt, seed, policy_config)` | Generate a list of synthetic observations |

### Arrival pattern generators

All arrival functions return `ArrivalsFn = Callable[[int], list[int]]` — given a frame index, returns a list of `t_max` values for new requests arriving that frame.

| Function | Description |
|---|---|
| `uniform_arrivals(rate=1, t_max=32)` | One new request every `rate` frames |
| `bursty_arrivals(burst_size=4, burst_every=8, t_max=32)` | Burst of `burst_size` requests every `burst_every` frames |
| `poisson_arrivals(lam=1.0, t_max=32, seed=42)` | Poisson-distributed arrivals with mean rate `lam` per frame |
| `variable_length_arrivals(t_max_values, rate=1)` | One request per `rate` frames, cycling through `t_max_values` deterministically |
| `random_length_arrivals(t_max_values, weights=None, rate=1, seed=42)` | One request per `rate` frames, `t_max` sampled from `t_max_values` with optional `weights` |

Arrival patterns are specified as strings in `fixed_params["arrival_pattern"]` (e.g., `"uniform_arrivals(rate=1)"`) and parsed by the continuous batching runner at runtime.

---

## Full Experiments Guide

This section provides a complete end-to-end workflow for reproducing all ECCV paper experiments, from running raw experiments to generating final paper figures. All commands use `experiments/results_eccv` as the results directory to avoid overwriting existing results.

### Prerequisites

- Check available GPUs with `nvidia-smi` before running experiments
- All commands assume you're in the repository root directory
- Use `uv run python -m` to execute scripts with the correct environment

### Workflow Overview

```
Experiments → JSON files → aggregate_metrics.py → metrics.csv
                                                      ↓
                              ┌───────────────────────┼───────────────────────┐
                              ↓                       ↓                       ↓
                    compute_speedup_cb.py  compute_speedup_ablation.py  compute_workload_sweep.py
                              ↓                       ↓                       ↓
                    speedup_cb_unified.csv  speedup_ablation.csv  workload_sweep.csv
                              ↓                       ↓                       ↓
                    plot_speedup_cb.py      plot_speedup_ablation.py  plot_workload_sweep.py
                              ↓                       ↓                       ↓
                    [4 heatmap/line PDFs]   [ablation_speedup.pdf]  [workload_sweep.pdf]
                                                      ↓
                                          copy_plots_to_paper.py → Paper figures directory
```

---

## Section A: Running Experiments

### A1. Main experiments (all policies and settings)

Run all four settings (baseline, shared_kv, continuous_batching, parallel_mps) across all three policies:

```bash
uv run python -m experiments.run_experiments \
    --settings baseline shared_kv continuous_batching parallel_mps \
    --policies pi05_o2_aloha pi05_o2_droid pi05_o2_libero \
    --results-dir experiments/results_eccv \
    --gpu 0  # Change GPU ID if needed (check nvidia-smi for available GPUs)
```

**Notes:**
- Check `nvidia-smi` for available GPUs and modify `--gpu` value if needed
- Results are saved to `experiments/results_eccv/{setting}/{policy}/`
- **Warning:** Re-running will overwrite existing results. Back up `experiments/results_eccv` if needed.
- This is the longest-running step (several hours depending on hardware)

### A2. Overhead measurements

Measure memory and energy overhead for the continuous batching setting:

```bash
uv run python -m experiments.run_overhead \
    --gpu 0 \  # Change GPU ID if needed
    --results-dir experiments/results_eccv \
    --policy pi05_o2_libero
```

**Notes:**
- Uses default policy `pi05_o2_libero` for overhead measurements
- Output: `experiments/results_eccv/analysis/overhead/overhead.csv`

### A3. Workload sweep

Test continuous batching under different arrival patterns:

```bash
uv run python -m experiments.run_workload_sweep \
    --gpu 0 \  # Change GPU ID if needed
    --results-dir experiments/results_eccv \
    --policy pi05_o2_libero
```

**Notes:**
- Tests three arrival patterns: uniform, poisson, random_length
- Output: `experiments/results_eccv/continuous_batching/pi05_o2_libero/denoise10_decode30_step5_*.json`
- Uses fixed configuration: 10 denoise steps, 30 max decoding steps, 5 steps per frame

---

## Section B: Analysis and Plotting Pipeline

Run these steps in order. Each step depends on outputs from previous steps.

### B1. Aggregate raw JSON results into CSV

**Command:**
```bash
uv run python -m experiments.analysis.aggregate_metrics \
    --results-root-dir experiments/results_eccv
```

**Output:** `experiments/results_eccv/analysis/aggregate_metrics/metrics.csv`

**Description:** Parses all raw JSON result files and aggregates them into a single CSV with columns for setting, policy, parameters, and computed metrics (frame latency, throughput, etc.). This is the foundation for all subsequent analysis.

### B2. Compute speedup metrics (continuous batching vs baseline)

**Command:**
```bash
uv run python -m experiments.analysis.compute_speedup_cb \
    --results-root-dir experiments/results_eccv
```

**Outputs:**
- `experiments/results_eccv/analysis/compute_speedup_cb/speedup_cb_unified.csv`
- `experiments/results_eccv/analysis/compute_speedup_cb/speedup_cb_same_config.csv`
- `experiments/results_eccv/analysis/compute_speedup_cb/speedup_cb_truncated.csv`

**Description:** Computes speedup of continuous batching over baseline for matching configurations. Generates three variants: unified (all comparisons), same_config (exact parameter matches), and truncated (baseline truncated to match CB's max_decoding_steps).

**Depends on:** `metrics.csv` from B1

### B3. Compute ablation speedup metrics

**Command:**
```bash
uv run python -m experiments.analysis.compute_speedup_ablation \
    --results-root-dir experiments/results_eccv
```

**Output:** `experiments/results_eccv/analysis/compute_speedup_ablation/speedup_ablation.csv`

**Description:** Computes speedup for ablation study comparing baseline → shared_kv → continuous_batching progression.

**Depends on:** `metrics.csv` from B1

### B4. Compute workload sweep metrics

**Command:**
```bash
uv run python -m experiments.analysis.compute_workload_sweep \
    --results-root-dir experiments/results_eccv \
    --policy pi05_o2_libero \
    --num-denoise-steps 10 \
    --max-decoding-steps 30
```

**Output:** `experiments/results_eccv/analysis/compute_workload_sweep/workload_sweep.csv`

**Description:** Aggregates metrics from workload sweep experiments (different arrival patterns).

**Depends on:** `metrics.csv` from B1

### B5. Generate all plots (recommended)

**Command:**
```bash
uv run python -m experiments.analysis.plot_all \
    --results-root-dir experiments/results_eccv \
    --steps-per-frame 5 \
    --num-denoise-steps 10 \
    --policy all
```

**Outputs (6 PDFs in `experiments/results_eccv/plot/`):**
- `plot_speedup_cb/e2e_speedup_heatmap_same_spf5.pdf` - Heatmap of speedup for same-config comparisons
- `plot_speedup_cb/e2e_speedup_heatmap_trunc_spf5.pdf` - Heatmap of speedup for truncated comparisons
- `plot_speedup_cb/e2e_latency_throughput_vs_decoding_steps.pdf` - Line plots showing latency/throughput vs max_decoding_steps
- `plot_speedup_cb/e2e_latency_throughput_vs_steps_per_frame.pdf` - Line plots showing latency/throughput vs steps_per_frame
- `plot_speedup_ablation/ablation_speedup.pdf` - Bar chart showing ablation study results
- `plot_workload_sweep/workload_sweep.pdf` - Plot comparing performance under different arrival patterns

**Description:** Runs all three plotting scripts in sequence to generate all paper figures.

**Depends on:** `speedup_cb_unified.csv` from B2, `speedup_ablation.csv` from B3, `workload_sweep.csv` from B4

### B5a. Generate speedup plots individually (optional)

If you need to regenerate specific plots, you can run the individual plotting scripts:

**Command:**
```bash
uv run python -m experiments.analysis.plot_speedup_cb \
    --results-root-dir experiments/results_eccv \
    --plot-type all \
    --steps-per-frame 5 \
    --num-denoise-steps 10
```

**Depends on:** `speedup_cb_unified.csv` from B2

### B5b. Generate ablation plot individually (optional)

**Command:**
```bash
uv run python -m experiments.analysis.plot_speedup_ablation \
    --results-root-dir experiments/results_eccv \
    --policy all \
    --num-denoise-steps 10
```

**Depends on:** `speedup_ablation.csv` from B3

### B5c. Generate workload sweep plot individually (optional)

**Command:**
```bash
uv run python -m experiments.analysis.plot_workload_sweep \
    --results-root-dir experiments/results_eccv
```

**Depends on:** `workload_sweep.csv` from B4

---

## Section C: Copy Figures to Paper Directory (Optional)

After generating all plots, copy them to the paper figures directory:

**Command:**
```bash
uv run python -m experiments.analysis.copy_plots_to_paper \
    --results-root-dir experiments/results_eccv \
    --paper-figures-dir ECCV-2026---MoT-VLA-Inference/figures/experiments
```

**Description:** Copies all 6 generated PDFs to the paper figures directory. Creates the target directory if it doesn't exist.

**Files copied:**
- `e2e_speedup_heatmap_same_spf5.pdf`
- `e2e_speedup_heatmap_trunc_spf5.pdf`
- `e2e_latency_throughput_vs_decoding_steps.pdf`
- `e2e_latency_throughput_vs_steps_per_frame.pdf`
- `ablation_speedup.pdf`
- `workload_sweep.pdf`

---

## Quick Reference: Complete Pipeline

For convenience, here's the complete sequence of commands to run all experiments and generate all figures:

```bash
# Step A: Run experiments (change --gpu 0 to another GPU ID if needed)
uv run python -m experiments.run_experiments --settings baseline shared_kv continuous_batching parallel_mps --policies pi05_o2_aloha pi05_o2_droid pi05_o2_libero --results-dir experiments/results_eccv --gpu 0
uv run python -m experiments.run_overhead --gpu 0 --results-dir experiments/results_eccv --policy pi05_o2_libero
uv run python -m experiments.run_workload_sweep --gpu 0 --results-dir experiments/results_eccv --policy pi05_o2_libero

# Step B: Analysis and plotting (no GPU needed)
uv run python -m experiments.analysis.aggregate_metrics --results-root-dir experiments/results_eccv
uv run python -m experiments.analysis.compute_speedup_cb --results-root-dir experiments/results_eccv
uv run python -m experiments.analysis.compute_speedup_ablation --results-root-dir experiments/results_eccv
uv run python -m experiments.analysis.compute_workload_sweep --results-root-dir experiments/results_eccv --policy pi05_o2_libero --num-denoise-steps 10 --max-decoding-steps 30
uv run python -m experiments.analysis.plot_all --results-root-dir experiments/results_eccv --steps-per-frame 5 --num-denoise-steps 10 --policy all

# Step C: Copy to paper (optional)
uv run python -m experiments.analysis.copy_plots_to_paper --results-root-dir experiments/results_eccv --paper-figures-dir ECCV-2026---MoT-VLA-Inference/figures/experiments
```
