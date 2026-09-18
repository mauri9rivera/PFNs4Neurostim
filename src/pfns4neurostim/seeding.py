"""Reproducible seeding across ``random``, NumPy and torch (CLAUDE.md section 4)."""
from __future__ import annotations

import random

import numpy as np

__all__ = ["set_seed", "rng_for"]


def set_seed(seed: int = 42) -> None:
    """Set all three random seeds, plus CUDA when available.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415 - optional at import time, required in practice
    except ImportError:  # pragma: no cover - torch is a hard dependency in runs
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_for(*parts: object, base_seed: int = 42) -> np.random.Generator:
    """Build a reproducible generator keyed by experiment coordinates.

    The seed is derived from the string form of ``parts`` via NumPy's
    ``SeedSequence`` entropy mixing, so a (dataset, subject, emg, knob, level,
    rep) cell always gets the same stream regardless of iteration order — which
    is what makes a resumed or reordered sweep reproduce bit for bit. Python's
    ``hash`` is deliberately avoided: it is salted per process.

    Args:
        *parts: Coordinates identifying the cell.
        base_seed: Experiment-wide base seed.

    Returns:
        A seeded :class:`numpy.random.Generator`.
    """
    key = "|".join(str(p) for p in parts)
    entropy = [base_seed] + [b for b in key.encode("utf-8")]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def seed_for(*parts: object, base_seed: int = 42) -> int:
    """Derive a deterministic 32-bit seed from experiment coordinates.

    Args:
        *parts: Coordinates identifying the cell.
        base_seed: Experiment-wide base seed.

    Returns:
        A seed in ``[0, 2**31)``, stable across processes and platforms.
    """
    return int(rng_for(*parts, base_seed=base_seed).integers(0, 2**31 - 1))
