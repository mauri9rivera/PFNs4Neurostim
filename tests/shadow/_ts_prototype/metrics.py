"""Three co-primary regret outcomes in range-normalised ground-truth units.

Phase-3 target: ``src/pfns4neurostim/evaluation/metrics.py``.

All regrets are divided by the channel's GT range (max - min), which makes them invariant to
any affine rescaling of y (P0.9), so GP and PFN runs are directly comparable.
"""
from __future__ import annotations

import numpy as np


def regret_curves(
    y_gt: np.ndarray, queried_idx: np.ndarray, recommended_idx: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute recommended-site, best-queried simple, and cumulative regret curves.

    Args:
        y_gt: Ground-truth per-site means, shape [M].
        queried_idx: Pool indices queried in order (incl. initial design), shape [T].
        recommended_idx: argmax-posterior-mean recommendation after each observation
            count, shape [T_rec] (aligned with the tail of ``queried_idx``).

    Returns:
        Dict with ``'recommended'`` [T_rec], ``'best_queried'`` [T], ``'cumulative'`` [T],
        all normalised by the GT range.

    Raises:
        RuntimeError: If the GT range is zero or inputs are non-finite.
    """
    y_gt = np.asarray(y_gt, dtype=np.float64)
    rng_ = float(y_gt.max() - y_gt.min())
    if not np.isfinite(y_gt).all() or rng_ <= 0:
        raise RuntimeError("regret_curves: GT must be finite with positive range.")
    opt = y_gt.max()
    inst = (opt - y_gt[np.asarray(queried_idx, dtype=int)]) / rng_           # [T]
    return {
        "recommended": (opt - y_gt[np.asarray(recommended_idx, dtype=int)]) / rng_,  # [T_rec]
        "best_queried": np.minimum.accumulate(inst),                           # [T]
        "cumulative": np.cumsum(inst),                                         # [T]
    }
