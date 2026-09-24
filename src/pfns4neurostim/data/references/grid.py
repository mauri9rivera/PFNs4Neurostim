"""Shared standardization and on-grid reference banks (task #6 Steps 0-1, P0.6).

Every source compared in the placement analysis — a neurostim ground-truth map, a prior
dataset, a noise dataset — passes through :func:`standardize_source`: X MinMax-scaled to
[0, 1]^D and y z-scored *per dataset*. The legacy noise bank was U[0, 1] in y, which made
"distance to noise" partly a distance in scale (audit F9).

References live **on the real electrode grid** at matched n:

* prior datasets are sampled densely (``n_dense`` random points, TabPFN v1 prior bag via
  ``libs/tabpfn-v1-prior``), MinMax-scaled, and their y interpolated onto the grid
  (linear inside the convex hull, nearest-neighbour outside). A random ``holdout_frac`` of
  the dense points is held out to measure the interpolation error; a dataset is kept only
  if its standardized RMSE is below the pre-declared ``rmse_threshold``, and the rejection
  rate is reported;
* noise is drawn directly on the grid.

Both are re-standardized on the grid, so every map in every bank has mean 0 and SD 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

__all__ = [
    "standardize_source",
    "standardize_map",
    "interpolate_to_grid",
    "GridBank",
    "prior_grid_bank",
    "noise_grid_bank",
]


def standardize_map(y: np.ndarray) -> np.ndarray:
    """z-score one response vector.

    Args:
        y: Responses, shape [N].

    Returns:
        ``(y - mean) / sd``, shape [N].

    Raises:
        RuntimeError: For a constant or non-finite vector (fail fast).
    """
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        raise RuntimeError("standardize_map: non-finite responses.")
    sd = float(y.std())
    if sd == 0.0:
        raise RuntimeError("standardize_map: constant responses cannot be standardized.")
    return (y - y.mean()) / sd


def standardize_source(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The one preprocessing every placement source passes through.

    Args:
        X: Inputs, shape [N, D].
        y: Responses, shape [N].

    Returns:
        ``(X MinMax-scaled to [0, 1]^D, y z-scored)``; a constant X axis maps to 0.
    """
    X = np.asarray(X, dtype=np.float64)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / span, standardize_map(y)


def interpolate_to_grid(
    X_dense: np.ndarray,
    y_dense: np.ndarray,
    grid_X: np.ndarray,
    *,
    holdout_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Interpolate a dense sample onto the grid and measure the interpolation error.

    Args:
        X_dense: Dense inputs in [0, 1]^D, shape [n, D].
        y_dense: Standardized responses, shape [n].
        grid_X: Grid coordinates in [0, 1]^D, shape [N, D].
        holdout_frac: Fraction of dense points held out for the error estimate.
        rng: Generator for the hold-out split.

    Returns:
        ``(y_grid [N], holdout_rmse)``; the RMSE is in standardized units because
        ``y_dense`` is standardized.
    """
    n = len(X_dense)
    n_hold = max(1, int(round(holdout_frac * n)))
    perm = rng.permutation(n)
    hold, fit = perm[:n_hold], perm[n_hold:]

    def _interp(Xf: np.ndarray, yf: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        lin = LinearNDInterpolator(Xf, yf)(Xq)                        # NaN outside the hull
        out = np.asarray(lin, dtype=np.float64)
        miss = ~np.isfinite(out)
        if miss.any():
            out[miss] = NearestNDInterpolator(Xf, yf)(Xq[miss])
        return out

    pred = _interp(X_dense[fit], y_dense[fit], X_dense[hold])
    rmse = float(np.sqrt(np.mean((pred - y_dense[hold]) ** 2)))
    return _interp(X_dense, y_dense, grid_X), rmse


@dataclass
class GridBank:
    """A bank of standardized maps on one grid.

    Attributes:
        maps: Standardized responses, shape [K, N].
        kind: ``'prior'`` or ``'noise'``.
        n_requested: Datasets attempted.
        rmse: Hold-out interpolation RMSE of every attempted dataset (prior only).
        meta: Provenance (threshold, prior type, seeds).
    """

    maps: np.ndarray
    kind: str
    n_requested: int
    rmse: list[float] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rejection_rate(self) -> float:
        """Fraction of attempted datasets rejected by the interpolation-error rule."""
        return 1.0 - len(self.maps) / self.n_requested if self.n_requested else 0.0


def prior_grid_bank(
    grid_X: np.ndarray,
    n_datasets: int,
    *,
    n_dense: int = 1024,
    holdout_frac: float = 0.1,
    rmse_threshold: float = 0.5,
    prior_type: str = "prior_bag",
    seed: int = 0,
    sampler: Callable[[int, int, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> GridBank:
    """On-grid prior references (task #6 Step 1).

    Args:
        grid_X: Grid coordinates (already MinMax-scaled), shape [N, D].
        n_datasets: Prior datasets to attempt.
        n_dense: Dense sample size per dataset.
        holdout_frac: Hold-out fraction for the interpolation error.
        rmse_threshold: Pre-declared maximum standardized hold-out RMSE.
        prior_type: TabPFN v1 prior component (``'prior_bag'`` is the canonical one).
        seed: Base seed; dataset ``i`` uses ``seed + i``.
        sampler: ``(n_features, n_samples, seed) -> (X, y)``; defaults to the TabPFN v1
            prior. Injected in tests.

    Returns:
        The kept maps plus every attempted dataset's RMSE.

    Raises:
        RuntimeError: If every dataset is rejected.
    """
    if sampler is None:
        from .prior import generate_tabpfn_prior_dataset  # noqa: PLC0415 - loads libs/ prior

        def sampler(d: int, n: int, s: int) -> tuple[np.ndarray, np.ndarray]:
            return generate_tabpfn_prior_dataset(n_features=d, n_samples=n, seed=s, prior_type=prior_type)

    D = grid_X.shape[1]
    maps: list[np.ndarray] = []
    rmses: list[float] = []
    for i in range(n_datasets):
        Xd, yd = sampler(D, n_dense, seed + i)
        try:
            Xd, yd = standardize_source(Xd, yd)
        except RuntimeError:
            rmses.append(float("inf"))
            continue
        y_grid, rmse = interpolate_to_grid(
            Xd, yd, grid_X, holdout_frac=holdout_frac, rng=np.random.default_rng(seed + i)
        )
        rmses.append(rmse)
        if rmse <= rmse_threshold and np.std(y_grid) > 0:
            maps.append(standardize_map(y_grid))
    if not maps:
        raise RuntimeError(
            f"prior_grid_bank: all {n_datasets} datasets exceeded rmse_threshold={rmse_threshold}."
        )
    return GridBank(
        maps=np.stack(maps), kind="prior", n_requested=n_datasets, rmse=rmses,
        meta={"rmse_threshold": rmse_threshold, "prior_type": prior_type, "n_dense": n_dense,
              "holdout_frac": holdout_frac, "seed": seed},
    )


def noise_grid_bank(grid_X: np.ndarray, n_datasets: int, *, seed: int = 0) -> GridBank:
    """Noise references drawn directly on the grid (i.i.d. uniform, then standardized).

    Args:
        grid_X: Grid coordinates, shape [N, D].
        n_datasets: Number of noise maps.
        seed: Seed.

    Returns:
        The bank.
    """
    rng = np.random.default_rng(seed)
    maps = np.stack([standardize_map(rng.uniform(size=len(grid_X))) for _ in range(n_datasets)])
    return GridBank(maps=maps, kind="noise", n_requested=n_datasets, meta={"seed": seed})
