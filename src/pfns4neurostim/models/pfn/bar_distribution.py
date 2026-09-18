"""Bucketized-regression adapter: read a classifier's class probabilities as a distribution.

Two Hyp 0 benchmark models — **TabPFN v1** and **TabFlex** — are classification
models. To use them as regression surrogates, the standardized response is
discretized into K bins and the predicted class probabilities are read as a bar
(piecewise-uniform) distribution over the response axis, giving mean, standard
deviation, quantiles and samples.

This is an *adaptation*, not the models' native behaviour, and it costs
resolution: no prediction can be sharper than one bin. Every table and caption
must label these rows **"classification-head adaptation"** (decision, 2026-09-16)
so the comparison is not read as like-for-like against TabPFN v2.5, whose bar
distribution is native and learned.

Binning follows the same convention as TabPFN's own bar distribution: bin borders
are quantiles of the training responses, so bins carry roughly equal mass and the
resolution follows the data rather than a uniform grid.
"""
from __future__ import annotations

import numpy as np

__all__ = ["BarDistribution", "quantile_borders"]


def quantile_borders(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute bin borders at equally spaced quantiles of ``y``.

    Args:
        y: Training responses, shape [n].
        n_bins: Number of bins K.

    Returns:
        Borders, shape [K + 1], strictly increasing.

    Raises:
        ValueError: If ``n_bins`` < 2 or ``y`` has fewer than two distinct values.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}.")
    y = np.asarray(y, dtype=np.float64).ravel()
    if np.unique(y).size < 2:
        raise ValueError("quantile_borders: y has fewer than two distinct values.")
    borders = np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1))
    # Ties collapse borders; nudge them apart so every bin has positive width.
    span = float(y.max() - y.min())
    eps = max(span, 1.0) * 1e-6
    for i in range(1, borders.size):
        if borders[i] <= borders[i - 1]:
            borders[i] = borders[i - 1] + eps
    return borders


class BarDistribution:
    """A piecewise-uniform distribution defined by bin borders and bin probabilities.

    Args:
        borders: Bin borders, shape [K + 1], strictly increasing.

    Raises:
        ValueError: If the borders are not strictly increasing.
    """

    def __init__(self, borders: np.ndarray) -> None:
        borders = np.asarray(borders, dtype=np.float64).ravel()
        if borders.size < 3 or np.any(np.diff(borders) <= 0):
            raise ValueError("BarDistribution: borders must be strictly increasing, length >= 3.")
        self.borders = borders                                  # [K + 1]
        self.centers = 0.5 * (borders[:-1] + borders[1:])       # [K]
        self.widths = np.diff(borders)                          # [K]

    @property
    def n_bins(self) -> int:
        """Number of bins K."""
        return int(self.centers.size)

    def _check(self, probs: np.ndarray) -> np.ndarray:
        """Validate and normalize a probability matrix.

        Args:
            probs: Class probabilities, shape [N, K].

        Returns:
            Row-normalized probabilities, shape [N, K].

        Raises:
            ValueError: On a bin-count mismatch.
            RuntimeError: On non-finite or non-positive-mass rows.
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[None, :]
        if probs.shape[1] != self.n_bins:
            raise ValueError(
                f"BarDistribution has {self.n_bins} bins but received {probs.shape[1]} class probabilities."
            )
        if not np.isfinite(probs).all():
            raise RuntimeError("BarDistribution: non-finite class probabilities.")
        totals = probs.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise RuntimeError("BarDistribution: a row of class probabilities sums to zero.")
        return probs / totals

    def mean(self, probs: np.ndarray) -> np.ndarray:
        """Predictive mean per row.

        Args:
            probs: Class probabilities, shape [N, K].

        Returns:
            Means, shape [N].
        """
        return self._check(probs) @ self.centers

    def std(self, probs: np.ndarray) -> np.ndarray:
        """Predictive standard deviation per row.

        Includes the within-bin variance (a uniform bin contributes
        ``width^2 / 12``), so a confident single-bin prediction still has the
        spread its resolution implies rather than zero.

        Args:
            probs: Class probabilities, shape [N, K].

        Returns:
            Standard deviations, shape [N].
        """
        p = self._check(probs)                                   # [N, K]
        mean = p @ self.centers                                  # [N]
        second = p @ (self.centers**2 + self.widths**2 / 12.0)   # [N]
        return np.sqrt(np.maximum(second - mean**2, 0.0))

    def quantile(self, probs: np.ndarray, q: float) -> np.ndarray:
        """Predictive quantile per row, interpolated within the containing bin.

        Args:
            probs: Class probabilities, shape [N, K].
            q: Quantile level in (0, 1).

        Returns:
            Quantile values, shape [N].

        Raises:
            ValueError: If ``q`` is outside (0, 1).
        """
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile level must be in (0, 1), got {q}.")
        p = self._check(probs)                                   # [N, K]
        cdf = np.cumsum(p, axis=1)                               # [N, K]
        idx = np.argmax(cdf >= q, axis=1)                        # [N]
        rows = np.arange(p.shape[0])
        below = np.where(idx > 0, cdf[rows, np.maximum(idx - 1, 0)], 0.0)   # [N]
        mass = np.maximum(p[rows, idx], 1e-12)                   # [N]
        frac = np.clip((q - below) / mass, 0.0, 1.0)             # [N]
        return self.borders[idx] + frac * self.widths[idx]

    def sample(self, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one sample per row (bin by probability, then uniform within it).

        Args:
            probs: Class probabilities, shape [N, K].
            rng: Seeded generator.

        Returns:
            Samples, shape [N].
        """
        p = self._check(probs)                                   # [N, K]
        cdf = np.cumsum(p, axis=1)                               # [N, K]
        u = rng.random(p.shape[0])[:, None]                      # [N, 1]
        idx = np.argmax(cdf >= u, axis=1)                        # [N]
        return self.borders[idx] + rng.random(p.shape[0]) * self.widths[idx]

    def digitize(self, y: np.ndarray) -> np.ndarray:
        """Map continuous responses to bin labels, for fitting the classifier.

        Args:
            y: Responses, shape [n].

        Returns:
            Bin indices in ``[0, K)``, shape [n].
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        return np.clip(np.digitize(y, self.borders[1:-1]), 0, self.n_bins - 1)

    def expand(self, probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Re-expand probabilities over only the observed classes to all K bins.

        A classifier fitted on a context that never contained some bin will not
        predict that class at all, so its probability matrix is narrower than K.

        Args:
            probs: Class probabilities over ``classes``, shape [N, C].
            classes: Bin indices the classifier knows, shape [C].

        Returns:
            Probabilities over all bins, shape [N, K].
        """
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[None, :]
        full = np.zeros((probs.shape[0], self.n_bins), dtype=np.float64)   # [N, K]
        full[:, np.asarray(classes, dtype=int)] = probs
        return full
