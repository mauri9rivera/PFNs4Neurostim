"""Corrected predictive summaries (P0.11).

Phase-3 target: ``src/pfns4neurostim/models/pfn/summaries.py``.

Fixes relative to ``src/utils/gpbo_utils.std_from_quantiles`` and
``TabPFNSurrogate.predict``: the point estimate is the distribution *mean* (not the median),
the 5/95 pair uses divisor 2 * Phi^-1(0.95) = 3.290 (not 4.390), and nothing is swallowed.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
PAIR_DIVISORS: tuple[tuple[int, int, float], ...] = tuple(
    (i, len(QUANTILE_LEVELS) - 1 - i, float(2 * norm.ppf(QUANTILE_LEVELS[-1 - i])))
    for i in range(3)
)  # ((0, 6, 3.290), (1, 5, 2.563), (2, 4, 1.349))


def std_from_quantiles(quantiles: np.ndarray) -> np.ndarray:
    """Gaussian-equivalent SD from symmetric quantile pairs (corrected divisors).

    Args:
        quantiles: Quantiles at ``QUANTILE_LEVELS``, shape [7, M].

    Returns:
        SD estimate, shape [M].

    Raises:
        RuntimeError: On NaN/Inf quantiles or wrong shape.
    """
    q = np.asarray(quantiles, dtype=np.float64)
    if q.shape[0] != len(QUANTILE_LEVELS):
        raise RuntimeError(f"std_from_quantiles expects {len(QUANTILE_LEVELS)} rows, got {q.shape[0]}.")
    if not np.isfinite(q).all():
        raise RuntimeError("std_from_quantiles received NaN/Inf quantiles.")
    est = [(q[hi] - q[lo]) / div for lo, hi, div in PAIR_DIVISORS]
    return np.maximum(np.mean(est, axis=0), 1e-12)
