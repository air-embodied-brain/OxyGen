"""Explicit random-init selection without altering the frozen pi implementation."""
from pathlib import Path


def pi_checkpoint(path, random_init):
    if not random_init:
        return path
    # Frozen pi selects random initialization only for a missing local path.
    sentinel = Path(path) / "__camera_ready_random_init_no_weights__"
    if sentinel.exists():
        raise RuntimeError(f"Random-init sentinel must not exist: {sentinel}")
    return str(sentinel)
