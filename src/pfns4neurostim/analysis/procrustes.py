"""Procrustes shape metrics on Gram matrices: NBS, Bures, shape distance, disparity (BETA Phase 1, 2026-10-21).

**Why Gram matrices.** For centred linear kernels ``K = X X'`` and ``L = Y Y'`` of two row-matched point clouds, the
orthogonal Procrustes problem depends on the clouds only through their kernels (Harvey, Larsen & Williams 2024,
"Duality of Bures and shape distances")::

    NBS(K, L)       = tr((K^1/2 L K^1/2)^1/2) / sqrt(tr K tr L)          normalised Bures similarity, in [0, 1]
    bures(K, L)^2   = tr K + tr L - 2 tr((K^1/2 L K^1/2)^1/2)              = min_R ||X - Y R||_F^2 (R orthogonal)
    shape_distance  = arccos(NBS)                                         angular Procrustes distance, a metric
    disparity       = 1 - NBS^2                                           ``scipy.spatial.procrustes``'s value

with ``tr((K^1/2 L K^1/2)^1/2) = ||A' B||_*`` (nuclear norm) for any factors ``K = A A'``, ``L = B B'``. Two
consequences the old cloud-only module could not offer:

* **No feature map is needed.** A posterior covariance -- C_PFN, Sigma_MLL, Sigma_fixed -- is a valid argument, and
  so is a 192-d layer Gram against a 32-d belief embedding: only the row count (the sites) must match. The width
  restriction of the pre-BETA ``procrustes_disparity`` is gone (BETA §2.6).
* **Bures is the 2-Wasserstein distance between N(0, K) and N(0, L).** On *uncentred* covariances it is the
  scale-aware comparison of two zero-mean Gaussian posteriors on the grid (BETA β1b), which CKA and NBS, both scale
  invariant, cannot make.

**What Procrustes adds next to CKA, stated correctly.** Centred linear CKA and Procrustes share their invariance
group (translation, isotropic scale, orthogonal maps), so an anisotropic stretch changes both. They differ in how they
weight the cross-covariance spectrum: CKA is a Frobenius (sum-of-squares) alignment dominated by the leading
directions, NBS a nuclear-norm (plain sum) quantity in which every direction counts (Ding et al. 2021). Neither
orders the other (``tests/analysis/test_procrustes.py``).

**Caveats.** (1) The rotation quotient is a choice: the prediction head is not rotation-equivariant, so every number
here is a representation-similarity readout reported beside output-space readouts, never a claim about the
prediction. (2) NBS has no zero baseline and is biased high at n < d (96 sites, 192 dims); BETA reads it only
against the smooth-field surrogate (D3d). The shuffled-site null of :func:`gram_null` permutes the site
correspondence of one kernel; like the CKA permutation null it destroys spatial smoothness, so it is a
**descriptive** chance level only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .cka import double_center

__all__ = [
    "GramNull",
    "gram_factor",
    "nbs_factors",
    "psd_project",
    "nbs",
    "bures",
    "shape_distance",
    "disparity",
    "gram_null",
    "STATISTICS",
    "MIN_ROWS",
    "EIG_REL_TOL",
    "NEGATIVE_TOL",
]

#: Fewest sites a shape metric is computed on. Below 3 the best rigid alignment of two centred clouds is exact for any
#: pair (2 centred points differ by a rotation and a scale alone), so every statistic is trivial.
MIN_ROWS: int = 3

#: Eigenvalues below ``EIG_REL_TOL x`` the largest one are treated as 0 when a Gram is factorised (rounding noise of a
#: PSD matrix). Negative eigenvalues beyond it are *clipped* and their mass is reported by :func:`psd_project`.
EIG_REL_TOL: float = 1e-12

#: Relative tolerance on quantities that are non-negative in exact arithmetic (a squared Bures distance, ``1 - NBS``):
#: anything more negative than this times the scale is a bug, not rounding, and raises.
NEGATIVE_TOL: float = 1e-8

#: Statistics :func:`gram_null` can draw a shuffled-site null for.
STATISTICS: tuple[str, ...] = ("nbs", "shape_distance", "disparity")


@dataclass(frozen=True)
class GramNull:
    """One shape statistic and its shuffled-site null (descriptive chance level).

    Attributes:
        statistic: One of :data:`STATISTICS`.
        observed: The statistic on the matched sites.
        null_mean: Mean over the shuffled-site replicates.
        null_sd: Standard deviation over the replicates.
        null_q05: 5th percentile of the replicates.
        null_q95: 95th percentile of the replicates.
        n_perm: Replicates drawn.
        n_rows: Sites.
    """

    statistic: str
    observed: float
    null_mean: float
    null_sd: float
    null_q05: float
    null_q95: float
    n_perm: int
    n_rows: int

    def as_row(self) -> dict[str, Any]:
        """Flat, JSON-safe mapping for a tidy CSV row."""
        return {k: (v if isinstance(v, str) else int(v) if k.startswith("n_") else float(v))
                for k, v in asdict(self).items()}


def _check(K: np.ndarray, name: str) -> np.ndarray:
    """``K`` as a finite, square float64 matrix with at least :data:`MIN_ROWS` rows, symmetrised.

    Raises:
        RuntimeError: On NaN/Inf.
        ValueError: On a non-square matrix or too few rows.
    """
    A = np.asarray(K, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"procrustes: {name} must be a square [n, n] Gram, got shape {A.shape}.")
    if A.shape[0] < MIN_ROWS:
        raise ValueError(f"procrustes: {name} needs n >= {MIN_ROWS} rows, got {A.shape[0]}.")
    if not np.isfinite(A).all():
        bad = int((~np.isfinite(A)).sum())
        raise RuntimeError(f"procrustes: {name} holds {bad} non-finite value(s) of {A.size}; refusing to proceed.")
    return 0.5 * (A + A.T)


def psd_project(K: np.ndarray) -> tuple[np.ndarray, float]:
    """Nearest PSD matrix in Frobenius norm (negative eigenvalues set to 0), and the clipped eigenvalue mass.

    The implied posterior covariance C_PFN is estimated by quadrature and need not be PSD (BETA §4); the clipped mass
    ``sum |lambda_-| / sum |lambda|`` is reported beside every statistic computed on the projection.

    Args:
        K: Symmetric (or nearly symmetric) matrix, shape [n, n]; symmetrised first.

    Returns:
        ``(K_psd [n, n], clipped_mass in [0, 1])``.
    """
    A = _check(K, "K")
    lam, V = np.linalg.eigh(A)                                             # [n], [n, n]
    total = float(np.abs(lam).sum())
    clipped = 0.0 if total == 0 else float(np.abs(lam[lam < 0]).sum() / total)
    return (V * np.clip(lam, 0.0, None)[None, :]) @ V.T, clipped


def gram_factor(K: np.ndarray, *, center: bool) -> np.ndarray:
    """A factor ``F`` with ``F F' = K_+`` (``K`` double-centred first when ``center``; negative part clipped).

    Args:
        K: Gram matrix, shape [n, n].
        center: Double-centre before factorising (the centred kernel of the shape metrics).

    Returns:
        ``F``, shape [n, r] with ``r`` the numerical rank.
    """
    A = _check(K, "K")
    if center:
        A = double_center(A)
    lam, V = np.linalg.eigh(0.5 * (A + A.T))                               # [n], [n, n]
    top = float(lam.max()) if lam.size else 0.0
    keep = lam > EIG_REL_TOL * max(top, 0.0)
    return V[:, keep] * np.sqrt(lam[keep])[None, :]                        # [n, r]


def _nuclear(Fk: np.ndarray, Fl: np.ndarray) -> float:
    """``||Fk' Fl||_*`` = ``tr((K^1/2 L K^1/2)^1/2)`` for ``K = Fk Fk'``, ``L = Fl Fl'``."""
    if Fk.shape[1] == 0 or Fl.shape[1] == 0:
        return 0.0
    return float(np.linalg.svd(Fk.T @ Fl, compute_uv=False).sum())


def _factors(K: np.ndarray, L: np.ndarray, center: bool) -> tuple[np.ndarray, np.ndarray]:
    """Factors of two Grams over the same sites (shape check included)."""
    if np.shape(K) != np.shape(L):
        raise ValueError(f"procrustes: Grams must be over the same sites, got {np.shape(K)} vs {np.shape(L)}.")
    return gram_factor(K, center=center), gram_factor(L, center=center)


def nbs_factors(Fk: np.ndarray, Fl: np.ndarray) -> float:
    """NBS from precomputed factors (``K = Fk Fk'``, ``L = Fl Fl'``; see :func:`gram_factor`).

    Lets a caller factor each Gram of a cell once and pair it with many others (BETA's representation runner).

    Args:
        Fk: Factor of the first (centred) Gram, shape [n, r_k].
        Fl: Factor of the second (centred) Gram, shape [n, r_l].

    Returns:
        NBS in [0, 1].

    Raises:
        RuntimeError: On a degenerate (zero-trace) side, or a value above 1 beyond tolerance.
    """
    tk, tl = float((Fk ** 2).sum()), float((Fl ** 2).sum())
    if tk <= 0 or tl <= 0:
        raise RuntimeError(f"procrustes: zero-trace Gram after centring ({tk:.3g}, {tl:.3g}); no shape to compare.")
    value = _nuclear(Fk, Fl) / np.sqrt(tk * tl)
    if value > 1.0 + NEGATIVE_TOL:
        raise RuntimeError(f"procrustes: NBS {value:.6f} > 1 beyond tolerance (Cauchy-Schwarz violated).")
    return float(min(value, 1.0))


def nbs(K: np.ndarray, L: np.ndarray, *, center: bool = True) -> float:
    """Normalised Bures similarity of two Grams over the same sites (Harvey, Larsen & Williams 2024).

    Implements ``NBS = tr((K^1/2 L K^1/2)^1/2) / sqrt(tr K tr L)``; on centred linear kernels it equals the cosine of
    the Procrustes angle between the two clouds, and ``1 - NBS^2`` equals ``scipy.spatial.procrustes``'s disparity.

    Args:
        K: Gram matrix, shape [n, n] (any width behind it, or a covariance).
        L: Gram matrix, shape [n, n].
        center: Double-centre both first (the shape metric; default). ``False`` compares uncentred kernels.

    Returns:
        NBS in [0, 1]; 1 iff the clouds agree up to translation, scale and an orthogonal map.

    Raises:
        ValueError: On mismatched or non-square shapes, or fewer than :data:`MIN_ROWS` sites.
        RuntimeError: On non-finite input or a zero-trace (degenerate) side.
    """
    return nbs_factors(*_factors(K, L, center))


def bures(K: np.ndarray, L: np.ndarray, *, center: bool = False) -> float:
    """Bures distance of two PSD Grams: the 2-Wasserstein distance between N(0, K) and N(0, L).

    ``bures^2 = tr K + tr L - 2 tr((K^1/2 L K^1/2)^1/2)``; for linear kernels of row-matched clouds it equals the
    unnormalised Procrustes distance ``min_R ||X - Y R||_F`` (centred clouds when ``center``). Scale-aware, so it
    sees what NBS and CKA cannot: a posterior that has the right shape at the wrong size.

    Args:
        K: PSD Gram / covariance, shape [n, n] (negative eigenvalues are clipped; see :func:`psd_project`).
        L: PSD Gram / covariance, shape [n, n].
        center: Double-centre both first (default ``False``: the uncentred posterior covariances of β1b).

    Returns:
        The distance, in the units of ``sqrt(K)``.

    Raises:
        ValueError: On mismatched shapes or too few sites.
        RuntimeError: On non-finite input or a squared distance negative beyond tolerance.
    """
    Fk, Fl = _factors(K, L, center)
    tk, tl = float((Fk ** 2).sum()), float((Fl ** 2).sum())
    sq = tk + tl - 2.0 * _nuclear(Fk, Fl)
    if sq < -NEGATIVE_TOL * max(tk + tl, 1.0):
        raise RuntimeError(f"bures: squared distance {sq:.3e} is negative beyond tolerance.")
    return float(np.sqrt(max(sq, 0.0)))


def shape_distance(K: np.ndarray, L: np.ndarray, *, center: bool = True) -> float:
    """Angular Procrustes shape distance ``arccos(NBS)`` (Williams et al. 2021): a true metric, in [0, pi/2].

    Args:
        K: Gram matrix, shape [n, n].
        L: Gram matrix, shape [n, n].
        center: Double-centre both first (default).

    Returns:
        The angle in radians.
    """
    return float(np.arccos(np.clip(nbs(K, L, center=center), 0.0, 1.0)))


def disparity(K: np.ndarray, L: np.ndarray, *, center: bool = True) -> float:
    """Procrustes disparity ``1 - NBS^2``: ``scipy.spatial.procrustes``'s third value, from Gram matrices.

    Args:
        K: Gram matrix, shape [n, n].
        L: Gram matrix, shape [n, n].
        center: Double-centre both first (default; scipy centres its clouds).

    Returns:
        The disparity in [0, 1].
    """
    return float(1.0 - nbs(K, L, center=center) ** 2)


def _from_nbs(statistic: str, value: float) -> float:
    """Map an NBS value to one of :data:`STATISTICS`."""
    if statistic == "nbs":
        return value
    if statistic == "shape_distance":
        return float(np.arccos(np.clip(value, 0.0, 1.0)))
    return float(1.0 - value ** 2)


def gram_null(
    K: np.ndarray,
    L: np.ndarray,
    *,
    statistic: str,
    n_perm: int,
    rng: np.random.Generator,
    center: bool = True,
) -> GramNull:
    """A shape statistic of two Grams and its shuffled-site null (rows and columns of ``L`` permuted jointly).

    The null keeps both kernels' marginal geometry (spectrum, trace) and destroys only the site matching, which is
    "what two unrelated n-site clouds reach" -- at 96 sites in 192 dimensions far from the no-similarity value, the
    n < d bias. It also destroys spatial smoothness, so it is descriptive only (BETA D4; the smooth-field surrogate is
    the chance level). Each replicate re-indexes ``L``'s factor and costs one ``[r_K, r_L]`` SVD.

    Args:
        K: Gram matrix, shape [n, n].
        L: Gram matrix, shape [n, n].
        statistic: One of :data:`STATISTICS`.
        n_perm: Replicates (>= 1), from a config key.
        rng: Generator keyed by the cell (:func:`pfns4neurostim.seeding.rng_for`); consumed: ``n_perm`` permutations.
        center: Double-centre both first (default).

    Returns:
        The :class:`GramNull` record.

    Raises:
        ValueError: On an unknown statistic or ``n_perm < 1`` (and see :func:`nbs`).
    """
    if statistic not in STATISTICS:
        raise ValueError(f"gram_null: unknown statistic {statistic!r}; known {STATISTICS}.")
    if int(n_perm) < 1:
        raise ValueError(f"gram_null: n_perm must be >= 1, got {n_perm}.")
    Fk, Fl = _factors(K, L, center)
    n = Fk.shape[0]
    observed = _from_nbs(statistic, nbs_factors(Fk, Fl))
    null = np.array([_from_nbs(statistic, nbs_factors(Fk, Fl[rng.permutation(n)]))
                     for _ in range(int(n_perm))])                          # [n_perm]
    return GramNull(statistic=statistic, observed=observed, null_mean=float(null.mean()),
                    null_sd=float(null.std(ddof=1)) if len(null) > 1 else 0.0,
                    null_q05=float(np.quantile(null, 0.05)), null_q95=float(np.quantile(null, 0.95)),
                    n_perm=int(n_perm), n_rows=int(n))
