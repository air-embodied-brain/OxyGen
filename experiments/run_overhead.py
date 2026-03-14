"""Measure memory and energy overhead for baseline vs continuous batching.

This script measures peak GPU memory, power draw, and derived energy metrics
for representative settings (baseline + continuous batching with different batch sizes).
Results are saved to a CSV file that can be used to populate LaTeX tables.

Usage:
    uv run python -m experiments.run_overhead --gpu 0 --results-dir experiments/results_eccv
"""

import argparse
import csv
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import jax

from experiments.baseline.runner import run_baseline
from experiments.common.setup import collect_metadata, create_policy, setup_jax_cache
from experiments.common.workload import create_synthetic_observation
from experiments.continuous_batching.runner import run_continuous_batching
from experiments.shared_kv.runner import run_shared_kv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GpuMonitor:
    """Background thread that polls nvidia-smi for power and optionally JAX for memory.

    When ``jax_device`` is provided, actual GPU memory is read from
    ``device.memory_stats()['bytes_in_use']`` (live JAX allocations).
    When ``jax_device`` is *None* (e.g. MPS case where the parent has no
    model loaded), only power is polled via nvidia-smi; memory must be
    supplied externally.
    """

    def __init__(self, gpu_id: int = 0, interval_s: float = 0.05, jax_device=None):
        """Initialize GPU monitor.

        Args:
            gpu_id: GPU device ID to monitor.
            interval_s: Polling interval in seconds (default 50ms for finer granularity).
            jax_device: Optional JAX device whose ``memory_stats()`` will be
                polled for live memory usage.  Pass ``None`` to skip JAX
                memory polling (power-only mode).
        """
        self.gpu_id = gpu_id
        self.interval_s = interval_s
        self.jax_device = jax_device
        self.samples = []          # (timestamp, power_w)
        self.memory_samples = []   # (timestamp, bytes_in_use) — only when jax_device set
        self._stop_event = threading.Event()
        self._thread = None

    def _poll_loop(self):
        """Background polling loop."""
        while not self._stop_event.is_set():
            try:
                # Power from nvidia-smi (always)
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_id}",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    power_w = float(result.stdout.strip())
                    self.samples.append((time.monotonic(), power_w))

                # Memory from JAX (when available)
                if self.jax_device is not None:
                    stats = self.jax_device.memory_stats()
                    if stats:
                        self.memory_samples.append(
                            (time.monotonic(), stats["bytes_in_use"])
                        )
            except Exception as e:
                logger.warning(f"GPU monitoring error: {e}")
            time.sleep(self.interval_s)

    def start(self):
        """Start monitoring in background thread."""
        self.samples = []
        self.memory_samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        mode = "power+memory (JAX)" if self.jax_device is not None else "power-only"
        logger.info(f"GPU monitor started (GPU {self.gpu_id}, {mode}, interval={self.interval_s}s)")

    def stop(self) -> dict:
        """Stop monitoring and return statistics.

        Returns:
            Dict with peak_memory_gb, peak_power_w, avg_power_w, num_samples.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        if not self.samples:
            logger.warning("No GPU samples collected")
            return {
                "peak_memory_gb": 0.0,
                "peak_power_w": 0.0,
                "avg_power_w": 0.0,
                "num_samples": 0,
            }

        power_values = [s[1] for s in self.samples]
        peak_power_w = max(power_values)
        avg_power_w = sum(power_values) / len(power_values)

        # JAX memory stats (if available)
        if self.memory_samples:
            mem_values = [s[1] for s in self.memory_samples]
            peak_memory_gb = max(mem_values) / (1024**3)
        else:
            peak_memory_gb = 0.0

        logger.info(
            f"GPU monitor stopped: {len(self.samples)} power samples, "
            f"{len(self.memory_samples)} memory samples, "
            f"peak_mem={peak_memory_gb:.2f}GB, peak_power={peak_power_w:.1f}W, "
            f"avg_power={avg_power_w:.1f}W"
        )

        return {
            "peak_memory_gb": peak_memory_gb,
            "peak_power_w": peak_power_w,
            "avg_power_w": avg_power_w,
            "num_samples": len(self.samples),
        }


def measure_shared_kv(policy, policy_config: str, prompt: str, num_denoise_steps: int,
                     max_decoding_steps: int, gpu_monitor: GpuMonitor) -> dict:
    """Measure shared_kv (single request with shared KV cache) overhead.

    Args:
        policy: Initialized policy object.
        policy_config: Policy config name.
        prompt: Text prompt for synthetic observations.
        num_denoise_steps: Denoising steps for actions.
        max_decoding_steps: Max text tokens per request.
        gpu_monitor: GPU monitor instance.

    Returns:
        Dict with measurement results including variance statistics.
    """
    logger.info("=== Measuring Shared KV (w/o batching) ===")

    # Extended warmup: 5 frames to stabilize GPU temperature
    logger.info("Warmup (5 frames)...")
    for i in range(5):
        obs = create_synthetic_observation(prompt, seed=i, policy_config=policy_config)
        run_shared_kv(policy, obs, num_denoise_steps=num_denoise_steps, max_decoding_steps=max_decoding_steps)

    # Measured frames: 30 frames for stable statistics
    logger.info("Starting GPU monitoring...")
    gpu_monitor.start()

    measured_frames = []
    for i in range(30):
        obs = create_synthetic_observation(prompt, seed=i + 100, policy_config=policy_config)
        result = run_shared_kv(policy, obs, num_denoise_steps=num_denoise_steps, max_decoding_steps=max_decoding_steps)
        measured_frames.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"  Frame {i+1}/30: {result['frame_ms']:.1f}ms")

    gpu_stats = gpu_monitor.stop()

    # Compute metrics with variance
    frame_latencies = [f["frame_ms"] for f in measured_frames]
    frame_latency_ms = sum(frame_latencies) / len(frame_latencies)
    frame_latency_std = (sum((x - frame_latency_ms) ** 2 for x in frame_latencies) / len(frame_latencies)) ** 0.5

    energy_per_frame_mj = gpu_stats["avg_power_w"] * frame_latency_ms / 1000.0
    # For shared_kv: 1 request per frame, 1 frame per request, so energy/request = energy/frame
    energy_per_request_mj = energy_per_frame_mj

    logger.info(f"  Avg frame latency: {frame_latency_ms:.1f} ± {frame_latency_std:.1f} ms")
    logger.info(f"  Avg power: {gpu_stats['avg_power_w']:.1f}W")
    logger.info(f"  Energy/request: {energy_per_request_mj:.1f} mJ")

    return {
        "label": "Ours w/o batching",
        "avg_batch_size": 1.0,
        "peak_memory_gb": gpu_stats["peak_memory_gb"],
        "peak_power_w": gpu_stats["peak_power_w"],
        "avg_power_w": gpu_stats["avg_power_w"],
        "frame_latency_ms": frame_latency_ms,
        "frame_latency_std": frame_latency_std,
        "energy_per_frame_mj": energy_per_frame_mj,
        "energy_per_request_mj": energy_per_request_mj,
    }


def measure_baseline(policy, policy_config: str, prompt: str, num_denoise_steps: int,
                     max_decoding_steps: int, gpu_monitor: GpuMonitor) -> dict:
    """Measure baseline (isolated execution) overhead.

    Args:
        policy: Initialized policy object.
        policy_config: Policy config name.
        prompt: Text prompt for synthetic observations.
        num_denoise_steps: Denoising steps for actions.
        max_decoding_steps: Max text tokens per request.
        gpu_monitor: GPU monitor instance.

    Returns:
        Dict with measurement results including variance statistics.
    """
    logger.info("=== Measuring Baseline (isolated execution) ===")

    # Extended warmup: 5 frames to stabilize GPU temperature
    logger.info("Warmup (5 frames)...")
    for i in range(5):
        obs = create_synthetic_observation(prompt, seed=i, policy_config=policy_config)
        run_baseline(policy, obs, num_denoise_steps=num_denoise_steps, max_decoding_steps=max_decoding_steps)

    # Measured frames: 30 frames for stable statistics
    logger.info("Starting GPU monitoring...")
    gpu_monitor.start()

    measured_frames = []
    for i in range(30):
        obs = create_synthetic_observation(prompt, seed=i + 100, policy_config=policy_config)
        result = run_baseline(policy, obs, num_denoise_steps=num_denoise_steps, max_decoding_steps=max_decoding_steps)
        measured_frames.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"  Frame {i+1}/30: {result['frame_ms']:.1f}ms")

    gpu_stats = gpu_monitor.stop()

    # Compute metrics with variance
    frame_latencies = [f["frame_ms"] for f in measured_frames]
    frame_latency_ms = sum(frame_latencies) / len(frame_latencies)
    frame_latency_std = (sum((x - frame_latency_ms) ** 2 for x in frame_latencies) / len(frame_latencies)) ** 0.5

    energy_per_frame_mj = gpu_stats["avg_power_w"] * frame_latency_ms / 1000.0
    # For baseline: 1 request per frame, 1 frame per request, so energy/request = energy/frame
    energy_per_request_mj = energy_per_frame_mj

    logger.info(f"  Avg frame latency: {frame_latency_ms:.1f} ± {frame_latency_std:.1f} ms")
    logger.info(f"  Avg power: {gpu_stats['avg_power_w']:.1f}W")
    logger.info(f"  Energy/request: {energy_per_request_mj:.1f} mJ")

    return {
        "label": "Baseline",
        "avg_batch_size": 1.0,
        "peak_memory_gb": gpu_stats["peak_memory_gb"],
        "peak_power_w": gpu_stats["peak_power_w"],
        "avg_power_w": gpu_stats["avg_power_w"],
        "frame_latency_ms": frame_latency_ms,
        "frame_latency_std": frame_latency_std,
        "energy_per_frame_mj": energy_per_frame_mj,
        "energy_per_request_mj": energy_per_request_mj,
    }


def measure_parallel_mps(policy_config: str, prompt: str, num_denoise_steps: int,
                         max_decoding_steps: int, gpu_id: int, gpu_monitor: GpuMonitor) -> dict:
    """Measure parallel_mps (naive parallelization via CUDA MPS) overhead.

    Args:
        policy_config: Policy config name.
        prompt: Text prompt for synthetic observations.
        num_denoise_steps: Denoising steps for actions.
        max_decoding_steps: Max text tokens per request.
        gpu_id: GPU device ID (needed for MPS daemon).
        gpu_monitor: GPU monitor instance.

    Returns:
        Dict with measurement results including variance statistics.
    """
    import multiprocessing
    from experiments.parallel_mps.runner import MPSWorkerPool
    from experiments.parallel_mps.grid_search import start_mps, stop_mps

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    logger.info("=== Measuring Parallel (MPS) ===")

    # Start MPS daemon
    start_mps(gpu_id=gpu_id)

    try:
        # Create worker pool
        with MPSWorkerPool(policy_config, prompt) as pool:
            # Extended warmup: 5 frames to stabilize GPU temperature
            logger.info("Warmup (5 frames)...")
            for i in range(5):
                pool.run_frame(num_denoise_steps=num_denoise_steps, max_decoding_steps=max_decoding_steps)

            # Measured frames: 30 frames for stable statistics
            logger.info("Starting GPU monitoring...")
            gpu_monitor.start()

            measured_frames = []
            for i in range(30):
                result = pool.run_frame(num_denoise_steps=num_denoise_steps, max_decoding_steps=max_decoding_steps)
                measured_frames.append(result)
                if (i + 1) % 10 == 0:
                    logger.info(f"  Frame {i+1}/30: {result['frame_ms']:.1f}ms")

            gpu_stats = gpu_monitor.stop()
    finally:
        # Always stop MPS daemon
        stop_mps()

    # Compute metrics with variance
    frame_latencies = [f["frame_ms"] for f in measured_frames]
    frame_latency_ms = sum(frame_latencies) / len(frame_latencies)
    frame_latency_std = (sum((x - frame_latency_ms) ** 2 for x in frame_latencies) / len(frame_latencies)) ** 0.5

    energy_per_frame_mj = gpu_stats["avg_power_w"] * frame_latency_ms / 1000.0
    # For parallel_mps: 1 request per frame, 1 frame per request, so energy/request = energy/frame
    energy_per_request_mj = energy_per_frame_mj

    # Memory from worker reports (sum of both workers' bytes_in_use)
    memory_values = [f["memory_bytes_in_use"] for f in measured_frames]
    peak_memory_gb = max(memory_values) / (1024**3) if memory_values else 0.0

    logger.info(f"  Avg frame latency: {frame_latency_ms:.1f} ± {frame_latency_std:.1f} ms")
    logger.info(f"  Avg power: {gpu_stats['avg_power_w']:.1f}W")
    logger.info(f"  Peak memory (JAX, both workers): {peak_memory_gb:.2f} GB")
    logger.info(f"  Energy/request: {energy_per_request_mj:.1f} mJ")

    return {
        "label": "Parallel (MPS)",
        "avg_batch_size": 1.0,
        "peak_memory_gb": peak_memory_gb,
        "peak_power_w": gpu_stats["peak_power_w"],
        "avg_power_w": gpu_stats["avg_power_w"],
        "frame_latency_ms": frame_latency_ms,
        "frame_latency_std": frame_latency_std,
        "energy_per_frame_mj": energy_per_frame_mj,
        "energy_per_request_mj": energy_per_request_mj,
    }


def measure_continuous_batching(policy, policy_config: str, prompt: str, num_denoise_steps: int,
                                max_decoding_steps: int, steps_per_frame: int, arrival_rate: float,
                                label: str, gpu_monitor: GpuMonitor) -> dict:
    """Measure continuous batching overhead.

    Args:
        policy: Initialized policy object.
        policy_config: Policy config name.
        prompt: Text prompt for synthetic observations.
        num_denoise_steps: Denoising steps for actions.
        max_decoding_steps: Max text tokens per request.
        steps_per_frame: Tokens generated per frame.
        arrival_rate: Request arrival rate (requests per frame).
        label: Human-readable label for this configuration.
        gpu_monitor: GPU monitor instance.

    Returns:
        Dict with measurement results including variance statistics.
    """
    logger.info(f"=== Measuring {label} ===")

    # Compute warmup frames: need to ramp up to steady state
    # Steady-state batch size ≈ (max_decoding_steps / steps_per_frame) / arrival_rate
    # Note: uniform_arrivals(rate=R) means 1 request every R frames
    expected_batch = int((max_decoding_steps / steps_per_frame) / arrival_rate)
    warmup_frames = max(15, expected_batch * 3)  # Increased warmup for thermal stability
    total_frames = warmup_frames + 60  # 60 measured frames for stable statistics

    arrival_pattern = f"uniform_arrivals(rate={arrival_rate}, t_max={max_decoding_steps})"

    logger.info(f"Running {total_frames} frames ({warmup_frames} warmup, 60 measured)...")
    logger.info(f"Expected steady-state batch size: ~{expected_batch}")

    # Start GPU monitoring
    gpu_monitor.start()

    result = run_continuous_batching(
        policy,
        policy_config=policy_config,
        prompt=prompt,
        num_denoise_steps=num_denoise_steps,
        max_decoding_steps=max_decoding_steps,
        steps_per_frame=steps_per_frame,
        total_frames=total_frames,
        warmup_frames=warmup_frames,
        arrival_pattern=arrival_pattern,
    )

    gpu_stats = gpu_monitor.stop()

    # Filter to steady-state frames only
    steady_frames = [f for f in result["frames"] if not f["is_warmup"]]

    if not steady_frames:
        logger.error("No steady-state frames collected!")
        return None

    # Compute metrics with variance
    frame_latencies = [f["frame_ms"] for f in steady_frames]
    batch_sizes = [f["n_total"] for f in steady_frames]

    frame_latency_ms = sum(frame_latencies) / len(frame_latencies)
    frame_latency_std = (sum((x - frame_latency_ms) ** 2 for x in frame_latencies) / len(frame_latencies)) ** 0.5
    avg_batch_size = sum(batch_sizes) / len(batch_sizes)

    energy_per_frame_mj = gpu_stats["avg_power_w"] * frame_latency_ms / 1000.0

    # Energy per request: each request takes (max_decoding_steps / steps_per_frame) frames,
    # but each frame processes avg_batch_size requests in parallel
    frames_per_request = max_decoding_steps / steps_per_frame
    energy_per_request_mj = energy_per_frame_mj * frames_per_request / avg_batch_size

    logger.info(f"  Steady-state: {len(steady_frames)} frames, avg_batch={avg_batch_size:.1f}, "
                f"frame_latency={frame_latency_ms:.1f} ± {frame_latency_std:.1f} ms")

    return {
        "label": label,
        "avg_batch_size": avg_batch_size,
        "peak_memory_gb": gpu_stats["peak_memory_gb"],
        "peak_power_w": gpu_stats["peak_power_w"],
        "avg_power_w": gpu_stats["avg_power_w"],
        "frame_latency_ms": frame_latency_ms,
        "frame_latency_std": frame_latency_std,
        "energy_per_frame_mj": energy_per_frame_mj,
        "energy_per_request_mj": energy_per_request_mj,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure memory and energy overhead")
    parser.add_argument("--gpu", type=int, required=True, help="GPU device ID")
    parser.add_argument("--results-dir", type=str, required=True, help="Results directory")
    parser.add_argument("--policy", type=str, default="pi05_o2_libero", help="Policy config name")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger.info(f"Using GPU {args.gpu}")

    # Common parameters
    policy_config = args.policy
    prompt = "Pick up the object and place it in the container."
    num_denoise_steps = 10
    max_decoding_steps = 20
    arrival_rate = 1.0  # Fixed: 1 request per frame for fair comparison

    # GPU monitor (power-only for MPS — parent has no JAX model loaded)
    power_only_monitor = GpuMonitor(gpu_id=args.gpu, interval_s=0.05, jax_device=None)

    # 1. Parallel (MPS) — must run BEFORE loading the parent policy, because
    #    MPS workers each need 45% GPU memory for their own model copies and
    #    the parent policy alone occupies ~60 GB.
    parallel_mps_result = measure_parallel_mps(
        policy_config, prompt, num_denoise_steps, max_decoding_steps, args.gpu, power_only_monitor
    )

    # Now set up JAX and load the parent policy for remaining measurements
    setup_jax_cache()
    metadata = collect_metadata()
    logger.info(f"GPU: {metadata['gpu']}, JAX: {metadata['jax_version']}")

    logger.info(f"Creating policy: {args.policy}")
    policy = create_policy(args.policy)

    # GPU monitor with JAX memory polling (for non-MPS measurements)
    jax_device = jax.local_devices()[0]
    gpu_monitor = GpuMonitor(gpu_id=args.gpu, interval_s=0.05, jax_device=jax_device)

    # 2. Baseline
    baseline_result = measure_baseline(
        policy, policy_config, prompt, num_denoise_steps, max_decoding_steps, gpu_monitor
    )

    # 3. Shared KV (w/o batching)
    shared_kv_result = measure_shared_kv(
        policy, policy_config, prompt, num_denoise_steps, max_decoding_steps, gpu_monitor
    )

    # Assemble results in table order: Baseline, Parallel (MPS), Ours w/o batch, ...
    results = []
    results.append(baseline_result)
    results.append(parallel_mps_result)
    results.append(shared_kv_result)

    # 4. Continuous batching with steps_per_frame=10 → batch≈2
    cb_batch2_result = measure_continuous_batching(
        policy, policy_config, prompt, num_denoise_steps, max_decoding_steps,
        steps_per_frame=10, arrival_rate=arrival_rate, label="Ours (batch≈2)", gpu_monitor=gpu_monitor
    )
    if cb_batch2_result:
        results.append(cb_batch2_result)

    # 5. Continuous batching with steps_per_frame=5 → batch≈4
    cb_batch4_result = measure_continuous_batching(
        policy, policy_config, prompt, num_denoise_steps, max_decoding_steps,
        steps_per_frame=5, arrival_rate=arrival_rate, label="Ours (batch≈4)", gpu_monitor=gpu_monitor
    )
    if cb_batch4_result:
        results.append(cb_batch4_result)

    # Save results
    output_dir = Path(args.results_dir) / "analysis" / "overhead"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "overhead.csv"

    with open(output_file, "w", newline="") as f:
        fieldnames = [
            "label",
            "avg_batch_size",
            "peak_memory_gb",
            "peak_power_w",
            "avg_power_w",
            "frame_latency_ms",
            "frame_latency_std",
            "energy_per_frame_mj",
            "energy_per_request_mj",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Results saved to {output_file}")

    # Print summary table
    print("\n" + "="*90)
    print("OVERHEAD MEASUREMENT RESULTS")
    print("="*90)
    print(f"{'Setting':<25} {'Batch':<8} {'Peak Mem':<12} {'Avg Pwr':<12} {'Latency (ms)':<20} {'E/Req':<12}")
    print(f"{'':25} {'':8} {'(GB)':12} {'(W)':12} {'mean ± std':20} {'(mJ)':12}")
    print("-"*90)
    for r in results:
        latency_str = f"{r['frame_latency_ms']:.1f} ± {r['frame_latency_std']:.1f}"
        print(f"{r['label']:<25} {r['avg_batch_size']:<8.1f} {r['peak_memory_gb']:<12.2f} "
              f"{r['avg_power_w']:<12.1f} {latency_str:<20} {r['energy_per_request_mj']:<12.1f}")
    print("="*90)


if __name__ == "__main__":
    main()
