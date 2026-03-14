"""Parallel MPS runner: two worker processes on the same GPU via NVIDIA MPS.

Each worker runs one inference type (actions or text) simultaneously.
Frame latency = max(action_time, text_time).
"""

import logging
import multiprocessing
import os
import queue
import time

logger = logging.getLogger(__name__)

PALIGEMMA_EOS_TOKEN = -1  # No early stop for latency profiling


def _worker_fn(task_type, policy_config, prompt, input_queue, output_queue):
    """Worker process: initializes its own policy and serves inference requests.

    Args:
        task_type: "action" or "text".
        policy_config: Policy config name for policy creation.
        prompt: Prompt string (used to create synthetic observation once).
        input_queue: Receives RUN/STOP commands from parent.
        output_queue: Sends ready/done/error signals back to parent.
    """
    # After MPS is started with the real GPU ID (e.g. CUDA_VISIBLE_DEVICES=5),
    # the daemon virtualizes that GPU as a single device at index 0. Child
    # processes must address it as device 0, otherwise JAX/CUDA won't find it.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Set memory fraction so two workers fit on one GPU
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

    try:
        # Late imports — each worker initializes its own JAX context
        import jax
        from experiments.common.setup import create_policy
        from experiments.common.workload import create_synthetic_observation

        logger.info("[%s] Initializing policy '%s'...", task_type, policy_config)
        policy = create_policy(policy_config)

        obs = create_synthetic_observation(
            prompt, seed=42, policy_config=policy_config,
        )

        logger.info("[%s] Ready.", task_type)
        output_queue.put({"status": "ready", "task_type": task_type})

        while True:
            try:
                cmd = input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if cmd["type"] == "STOP":
                break

            if cmd["type"] == "RUN":
                params = cmd["params"]
                t0 = time.monotonic()

                if task_type == "action":
                    out = policy.infer_actions(
                        obs, num_steps=params["num_denoise_steps"],
                    )
                    jax.block_until_ready(out["actions"])
                    policy_timing = out["policy_timing"]
                elif task_type == "text":
                    out = policy.infer_text(
                        obs,
                        max_decoding_steps=params["max_decoding_steps"],
                        PALIGEMMA_EOS_TOKEN=PALIGEMMA_EOS_TOKEN,
                    )
                    jax.block_until_ready(out["tokens"])
                    policy_timing = out["policy_timing"]
                else:
                    raise ValueError(f"Unknown task_type: {task_type}")

                t1 = time.monotonic()

                # Report actual JAX memory usage
                device = jax.local_devices()[0]
                mem_stats = device.memory_stats()
                mem_bytes = mem_stats["bytes_in_use"] if mem_stats else 0

                output_queue.put({
                    "status": "done",
                    "task_type": task_type,
                    "wall_time": t1 - t0,
                    "policy_timing": policy_timing,
                    "memory_bytes_in_use": mem_bytes,
                })

    except Exception as e:
        import traceback
        traceback.print_exc()
        output_queue.put({"status": "error", "task_type": task_type, "message": str(e)})


class MPSWorkerPool:
    """Manages two MPS worker processes (action + text) on the same GPU.

    Usage::

        with MPSWorkerPool(policy_config, prompt) as pool:
            result = pool.run_frame(num_denoise_steps=10, max_decoding_steps=5)
    """

    def __init__(self, policy_config: str, prompt: str):
        self.policy_config = policy_config
        self.prompt = prompt
        self._action_proc = None
        self._text_proc = None
        self._q_in_action = None
        self._q_out_action = None
        self._q_in_text = None
        self._q_out_text = None

    def start(self):
        """Spawn action and text worker processes, wait until both are ready.

        Workers are started sequentially to avoid OOM during concurrent
        checkpoint loading — orbax's device_put needs temporary GPU buffers
        that can exceed the per-process memory fraction when two loads overlap.
        """
        self._q_in_action = multiprocessing.Queue()
        self._q_out_action = multiprocessing.Queue()
        self._q_in_text = multiprocessing.Queue()
        self._q_out_text = multiprocessing.Queue()

        self._action_proc = multiprocessing.Process(
            target=_worker_fn,
            name="ActionWorker",
            args=("action", self.policy_config, self.prompt,
                  self._q_in_action, self._q_out_action),
        )
        self._text_proc = multiprocessing.Process(
            target=_worker_fn,
            name="TextWorker",
            args=("text", self.policy_config, self.prompt,
                  self._q_in_text, self._q_out_text),
        )

        # Start workers one at a time so checkpoint loading doesn't overlap.
        for proc, q_out, name in [
            (self._action_proc, self._q_out_action, "action"),
            (self._text_proc, self._q_out_text, "text"),
        ]:
            proc.start()
            self._wait_for_worker(proc, q_out, name)

        logger.info("Both workers ready.")

    def _wait_for_worker(self, proc, q_out, name):
        """Block until a single worker signals ready or errors out."""
        while True:
            if not proc.is_alive():
                raise RuntimeError(f"Worker [{name}] died during initialization.")
            try:
                msg = q_out.get(timeout=1.0)
                if msg["status"] == "ready":
                    logger.info("Worker [%s] ready.", name)
                    return
                elif msg["status"] == "error":
                    raise RuntimeError(
                        f"Worker [{name}] failed: {msg['message']}"
                    )
            except queue.Empty:
                continue

    def run_frame(self, *, num_denoise_steps: int, max_decoding_steps: int) -> dict:
        """Dispatch one frame to both workers in parallel.

        Returns:
            Dict matching baseline frame format.
        """
        params = {
            "num_denoise_steps": num_denoise_steps,
            "max_decoding_steps": max_decoding_steps,
        }

        t0 = time.monotonic()
        self._q_in_action.put({"type": "RUN", "params": params})
        self._q_in_text.put({"type": "RUN", "params": params})

        res_action = self._q_out_action.get()
        res_text = self._q_out_text.get()
        t1 = time.monotonic()

        for res in (res_action, res_text):
            if res["status"] == "error":
                raise RuntimeError(
                    f"Worker [{res['task_type']}] error: {res['message']}"
                )

        action_wall = res_action["wall_time"]
        text_wall = res_text["wall_time"]

        # Merge policy_timing with prefixed keys
        policy_timing = {
            "actions_wall_ms": action_wall * 1000.0,
            "text_wall_ms": text_wall * 1000.0,
        }
        for k, v in res_action["policy_timing"].items():
            policy_timing[f"actions_{k}"] = v
        for k, v in res_text["policy_timing"].items():
            policy_timing[f"text_{k}"] = v

        return {
            "frame_ms": max(action_wall, text_wall) * 1000.0,
            "total_tokens_this_frame": max_decoding_steps,
            "n_new": 1,
            "n_resumed": 0,
            "n_total": 1,
            "policy_timing": policy_timing,
            "memory_bytes_in_use": (
                res_action.get("memory_bytes_in_use", 0)
                + res_text.get("memory_bytes_in_use", 0)
            ),
        }

    def stop(self):
        """Send STOP to both workers and join them."""
        for q in (self._q_in_action, self._q_in_text):
            if q is not None:
                q.put({"type": "STOP"})

        for proc, name in [
            (self._action_proc, "action"),
            (self._text_proc, "text"),
        ]:
            if proc is not None:
                proc.join(timeout=10)
                if proc.is_alive():
                    logger.warning("Worker [%s] hung, terminating.", name)
                    proc.terminate()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
