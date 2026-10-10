# Camera-ready appendix plots

Run the complete portable suite via `experiments/camera_ready/run.py`; see its README.
For existing aggregated data:

```bash
.venv/bin/python -m experiments.analysis.plot_camera_ready_appendix --results-root /path/to/results --output-dir /path/to/figures
```

Outputs are PDF and PNG. All canvases are 8.5 inches wide; heights are 2.9 inches
(manager), 3.25 (scaling), and 3.35 (long run). Include at full text width without
cropping to preserve the reviewed font scale. Fonts use shared `setup_style()`
with Times New Roman or Liberation Serif, and STIX math. Missing fonts fail explicitly.

Input paths relative to `--results-root`:

- `kv_manager_fullpath_aligned_20260729/aggregate.json`: synchronized full-frame
  JAX costs at k=5/1; stack + split + bookkeeping form the manager total.
- `qwen35_scaling_tp_formal_20260728/aggregated.json`: full-precision median-of-repeat-means
  for 0.8B/2B/4B/9B, N=10/30, Baseline, prefix reuse only, and k=1/5/10.
- `eccv_submission_raw/kv_manager_longrun_20260726/raw/bs32/{jax,pytorch}_fluctuating_bs32/frames.jsonl`:
  all 3,000 post-warmup frames. Complete 119-frame cycles supply 5%-trimmed latency means;
  50-frame bins supply memory means and min/max envelopes, using every recorded sample.

Scaling: Baseline red, prefix reuse gold, k=1 dark blue, k=5 blue, k=10 light blue.
Long run: JAX blue, PyTorch red; lighter dashed lines/bands show memory on the right axis.
Numeric annotations and axis ranges are derived from inputs; new measurements cannot
retain an archived 4.5x split annotation or be silently clipped by fixed old limits.
The camera-ready reporter constructs these input schemas from newly measured raw data.
