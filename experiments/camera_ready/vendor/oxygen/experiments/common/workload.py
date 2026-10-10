"""Generate synthetic workload for benchmarking.

Migrated from benchmark/continuous_batching/workload.py — this is now the
canonical location. benchmark/ re-exports from here for backwards compatibility.
"""

import math
import random
import numpy as np
from typing import Callable, Dict, Any


# ---------------------------------------------------------------------------
# Platform-specific observation factories
# ---------------------------------------------------------------------------

def _create_arx_obs(prompt: str, rng: np.random.RandomState) -> Dict[str, Any]:
    """ARX format: CHW uint8 images, float32[14] state."""
    return {
        "image": {
            "left_wrist_view": rng.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "face_view": rng.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist_view": rng.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        },
        "state": rng.randn(14).astype(np.float32),
        "prompt": prompt,
    }


def _create_droid_obs(prompt: str, rng: np.random.RandomState) -> Dict[str, Any]:
    """DROID format: HWC uint8 images, float32[7] joint + float32[1] gripper."""
    return {
        "observation/exterior_image_1_left": rng.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": rng.randn(7).astype(np.float32),
        "observation/gripper_position": rng.randn(1).astype(np.float32),
        "prompt": prompt,
    }


def _create_aloha_obs(prompt: str, rng: np.random.RandomState) -> Dict[str, Any]:
    """ALOHA format: CHW uint8 images, float32[14] state."""
    return {
        "images": {
            "cam_high": rng.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": rng.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": rng.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": rng.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
        },
        "state": rng.randn(14).astype(np.float32),
        "prompt": prompt,
    }


def _create_libero_obs(prompt: str, rng: np.random.RandomState) -> Dict[str, Any]:
    """LIBERO format: HWC uint8 images, float32[8] state."""
    return {
        "observation/image": rng.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": rng.randn(8).astype(np.float32),
        "prompt": prompt,
    }


_OBSERVATION_FACTORIES: Dict[str, Callable[[str, np.random.RandomState], Dict[str, Any]]] = {
    "arx": _create_arx_obs,
    "droid": _create_droid_obs,
    "aloha": _create_aloha_obs,
    "libero": _create_libero_obs,
}


def _resolve_factory(policy_config: str) -> Callable[[str, np.random.RandomState], Dict[str, Any]]:
    """Match a policy config name to an observation factory."""
    for key, factory in _OBSERVATION_FACTORIES.items():
        if key in policy_config:
            return factory
    raise ValueError(
        f"Cannot resolve platform from policy config {policy_config!r}. "
        f"Expected one of {list(_OBSERVATION_FACTORIES.keys())} as a substring."
    )


def create_synthetic_observation(
    prompt: str,
    seed: int | None = None,
    policy_config: str = "pi05_o2_arx",
    # Deprecated alias — benchmark/ still uses this kwarg name
    checkpoint_config: str | None = None,
) -> Dict[str, Any]:
    """Create a synthetic observation for benchmarking."""
    if checkpoint_config is not None:
        policy_config = checkpoint_config
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    factory = _resolve_factory(policy_config)
    return factory(prompt, rng)


def generate_workload(
    num_requests: int,
    prompt: str,
    seed: int = 42,
    policy_config: str = "pi05_o2_arx",
    # Deprecated alias — benchmark/ still uses this kwarg name
    checkpoint_config: str | None = None,
) -> list[Dict[str, Any]]:
    """Generate a list of synthetic observations."""
    if checkpoint_config is not None:
        policy_config = checkpoint_config
    return [
        create_synthetic_observation(prompt, seed=seed + i, policy_config=policy_config)
        for i in range(num_requests)
    ]


# ---------------------------------------------------------------------------
# Arrival pattern generators for non-steady-state benchmark
# ---------------------------------------------------------------------------

ArrivalsFn = Callable[[int], list[int]]
"""Callable(frame_idx) -> list of T_max values for new requests this frame."""


def uniform_arrivals(rate: float = 1, t_max: int = 32) -> ArrivalsFn:
    """Uniform arrival pattern.

    Args:
        rate: Arrival rate parameter. IMPORTANT: This is the INTER-ARRIVAL TIME, not frequency.
            - rate >= 1: One new request every `rate` frames
              (e.g., rate=1 → 1 req/frame, rate=2 → 1 req every 2 frames)
            - rate < 1: Multiple requests per frame
              (e.g., rate=0.5 → 2 req/frame, rate=0.25 → 4 req/frame)
            - Steady-state batch size: B* ≈ (T_max / k) / rate
              where T_max is max tokens per request, k is steps_per_frame
        t_max: Maximum decoding steps (tokens) per request.

    Returns:
        Callable that takes frame_idx and returns list of t_max values for new arrivals.

    Examples:
        >>> fn = uniform_arrivals(rate=1, t_max=20)  # 1 request per frame
        >>> fn(0)  # [20]
        >>> fn = uniform_arrivals(rate=2, t_max=20)  # 1 request every 2 frames
        >>> fn(0)  # [20]
        >>> fn(1)  # []
        >>> fn = uniform_arrivals(rate=0.5, t_max=20)  # 2 requests per frame
        >>> fn(0)  # [20, 20]
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if rate >= 1:
        period = int(rate)
        def fn(frame_idx: int) -> list[int]:
            return [t_max] if frame_idx % period == 0 else []
    else:
        count = round(1 / rate)
        def fn(frame_idx: int) -> list[int]:
            return [t_max] * count
    return fn


def bursty_arrivals(burst_size: int = 4, burst_every: int = 8, t_max: int = 32) -> ArrivalsFn:
    """Burst of `burst_size` requests every `burst_every` frames."""
    def fn(frame_idx: int) -> list[int]:
        return [t_max] * burst_size if frame_idx % burst_every == 0 else []
    return fn


def variable_length_arrivals(t_max_values: list[int], rate: int = 1) -> ArrivalsFn:
    """One new request per `rate` frames, cycling through `t_max_values`."""
    def fn(frame_idx: int) -> list[int]:
        if frame_idx % rate == 0:
            return [t_max_values[(frame_idx // rate) % len(t_max_values)]]
        return []
    return fn


def poisson_arrivals(lam: float = 1.0, t_max: int = 32, seed: int = 42) -> ArrivalsFn:
    """Poisson-distributed arrivals with mean rate `lam` requests per frame."""
    rng = random.Random(seed)

    def fn(frame_idx: int) -> list[int]:
        # Knuth's algorithm for exact Poisson sampling
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            p *= rng.random()
            if p < L:
                break
            k += 1
        return [t_max] * k
    return fn


def random_length_arrivals(
    t_max_values: list[int],
    weights: list[float] | None = None,
    rate: int = 1,
    seed: int = 42,
) -> ArrivalsFn:
    """One new request per `rate` frames; T_max sampled from `t_max_values` with `weights`.

    Unlike `variable_length_arrivals` (which cycles deterministically), this
    samples T_max independently each frame, introducing genuine randomness in
    request length. `weights` defaults to uniform if not provided.
    """
    rng = random.Random(seed)
    population = t_max_values
    cum_weights: list[float] | None = None
    if weights is not None:
        total = sum(weights)
        cum_weights = []
        acc = 0.0
        for w in weights:
            acc += w / total
            cum_weights.append(acc)

    def fn(frame_idx: int) -> list[int]:
        if frame_idx % rate != 0:
            return []
        if cum_weights is None:
            t = rng.choice(population)
        else:
            r = rng.random()
            t = population[next(i for i, c in enumerate(cum_weights) if r <= c)]
        return [t]
    return fn
