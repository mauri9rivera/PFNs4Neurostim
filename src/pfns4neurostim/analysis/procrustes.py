"""Orthogonal Procrustes disparity with a shuffled-site null: the complement of the CKA analysis.

**Why this exists next to** :mod:`pfns4neurostim.analysis.cka`, **stated correctly.** The motivating claim was
that CKA is blind to anisotropy and Procrustes is not. That is **false as stated**, and the tests record why:
centred linear CKA and Procrustes have the *same* invariance group -- translation, isotropic rescaling and any
orthogonal map -- so a general anisotropic stretch changes the Gram matrix and CKA *does* register it
(``tests/analysis/test_procrustes.py::test_sensitive_to_anisotropic_stretch``; a 1->6 gain ramp moves debiased
CKA by 0.20 and the disparity by only 0.06).

The real difference is the **norm each puts on the cross-covariance spectrum**. Linear CKA is a Frobenius
(sum-of-squares) alignment of Gram matrices, so it is dominated by the leading directions; the Procrustes
residual is ``1 - (sum_i sigma_i) ** 2``, a **nuclear-norm** (plain sum) quantity, so it weights every
direction's alignment equally. Neither dominates the other: a reweighting that inflates the 95 weak directions
of a 96-site cloud while leaving the dominant one alone moves the disparity by 0.17 and debiased CKA by < 0.01
(``test_spectrum_reweighting_that_cka_is_nearly_blind_to``), and the gain-ramp case above goes the other way.
Procrustes therefore adds a channel that is sensitive to change spread across the weakly-populated directions,
which CKA's Frobenius weighting discounts -- and *that* is the complementarity worth reporting.

The context in which that matters: each transformer block of the probed checkpoint ends in a *parameter-free
post-LayerNorm*, which puts every token on a sqrt(192) sphere, so the per-site norm of a representation change
is a pure rotation magnitude and the headline mechanism readouts (``rho_shape``, ``||dZ||``) are direction-blind.
Procrustes and CKA are both direction-aware; they disagree about which directions count.

**CAVEAT 1 -- the rotation quotient is a choice, not a free win.** The prediction head has fixed weights and is
**not** rotation-equivariant (``Linear 192->384 -> GELU -> Linear 384->5000``), so a global rotation of all
tokens genuinely changes the model's output. Procrustes declares that rotation irrelevant. That is defensible for
a *"did the geometry change shape?"* question and **indefensible for any claim about the prediction**. Every
number this module produces is therefore a representation-similarity readout reported *alongside*
output-space readouts, never a substitute for one, and must not carry a claim about what the model predicts.

**CAVEAT 2 -- the n < d regime, and the control that fixes it.** The NHP array has 96 sites and the
representation is 192-dimensional. Two centred 96-row clouds each span at most a 95-dimensional subspace of a
192-dimensional space, so an orthogonal map can align them far better than chance and the raw disparity is
**biased low**. (The reduction in :func:`procrustes_disparity` makes this explicit: the statistic only ever sees
a ``[r, r]`` matrix with ``r <= min(n, d)``.) The legacy M14 implementation sidestepped the bias by subsampling
300 rows, which a 96-site grid cannot do. The fix here is a **shuffled-site null**, and it is part of the same
primitive -- :func:`procrustes_with_null` is the only public entry point that returns a disparity, so a raw
disparity can never be reported without the null it must be read against. The null permutes the row
correspondence of one cloud (destroying the site matching while leaving both clouds' marginal geometry
untouched), which is exactly "what disparity do two unrelated 96-row clouds in 192 dimensions reach".

**Pre-declared headline statistic:** ``disparity_minus_null`` = ``disparity - null_mean``, the *difference*. It is
chosen over the ratio and the percentile because (i) it mirrors CKA's ``cka_minus_null``, so a Procrustes panel
and a CKA panel over the same pairs are read on the same kind of axis; (ii) the ratio is unstable when the null
mean sits close to its ceiling (disparity is bounded in [0, 1] and the null is typically near the top of that
range, so small absolute moves become large ratio moves); and (iii) the percentile saturates -- with ``n_perm``
replicates it cannot resolve anything finer than ``1 / (n_perm + 1)``, and the measured disparities are expected
to sit far below the whole null distribution. The ratio and the percentile are still returned and written, as
secondary readouts.

Relation to the legacy code: :func:`pfns4neurostim.analysis.id_ood.compute_procrustes_disparity` is *not* reused.
It returns ``nan`` on non-finite input (against the project's fail-fast rule), subsamples with a hardcoded
``RandomState`` seed, and has no null -- the three things that make a disparity uninterpretable here. The
statistic computed below is the same quantity as ``scipy.spatial.procrustes`` (asserted in the tests) by a
reduction that makes the permutation null affordable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "ProcrustesNull",
    "standardize_cloud",
    "procrustes_disparity",
    "procrustes_with_null",
    "HEADLINE_STATISTIC",
    "MIN_ROWS",
    "NEGATIVE_DISPARITY_TOL",
]

#: The pre-declared headline statistic (see the module docstring); the others are secondary.
HEADLINE_STATISTIC: str = "disparity_minus_null"

#: Fewest rows a disparity is computed on. Below 3 the best rigid alignment is exact for any pair of clouds
#: (2 centred points in any dimension differ by a rotation and a scale alone), so the statistic is identically 0.
MIN_ROWS: int = 3

#: ``1 - nuclear_norm ** 2`` is non-negative in exact arithmetic; a float64 SVD may undershoot by ~1e-15.
#: Anything below this is a bug, not rounding, and raises.
NEGATIVE_DISPARITY_TOL: float = 1e-8


@dataclass(frozen=True)
class ProcrustesNull:
    """One Procrustes comparison and its shuffled-site null.

    Attributes:
        disparity: Orthogonal Procrustes disparity in [0, 1] of the matched clouds.
        null_mean: Mean disparity over the shuffled-site replicates.
        null_sd: Standard deviation over the replicates.
        null_q05: 5th percentile of the replicates (the low tail the measurement is compared against).
        disparity_minus_null: ``disparity - null_mean`` -- the :data:`HEADLINE_STATISTIC`.
        disparity_ratio: ``disparity / null_mean`` (secondary).
        null_percentile: Fraction of replicates at or below ``disparity`` (secondary; resolution
            ``1 / (n_perm + 1)``).
        n_perm: Replicates drawn.
        n_rows: Rows (matched sites) per cloud.
        n_dim: Representation dimension.
        n_rank: Rank of the reduction, ``min(n_rows - 1, n_dim)``; ``n_rank < n_dim`` *is* the n < d regime.
    """

    disparity: float
    null_mean: float
    null_sd: float
    null_q05: float
    disparity_minus_null: float
    disparity_ratio: float
    null_percentile: float
    n_perm: int
    n_rows: int
    n_dim: int
    n_rank: int

    def as_row(self) -> dict[str, Any]:
        """Return the result as a flat, JSON-safe mapping for a tidy CSV row."""
        return {k: (int(v) if k.startswith("n_") else float(v)) for k, v in asdict(self).items()}


def _check_finite(Z: np.ndarray, name: str) -> np.ndarray:
    """Return ``Z`` as float64, raising on NaN/Inf (no silent propagation).

    Args:
        Z: Candidate array.
        name: Argument name, for the message.

    Returns:
        The array as float64, shape unchanged.

    Raises:
        RuntimeError: If any entry is NaN or Inf.
    """
    A = np.asarray(Z, dtype=np.float64)
    if not np.all(np.isfinite(A)):
        bad = int((~np.isfinite(A)).sum())
        raise RuntimeError(f"procrustes: {name} holds {bad} non-finite value(s) of {A.size}; refusing to proceed.")
    return A


def standardize_cloud(Z: np.ndarray) -> np.ndarray:
    """Centre a point cloud on its centroid and scale it to unit Frobenius norm.

    This is the standardization ``scipy.spatial.procrustes`` applies, isolated so the permutation null can reuse
    it: permuting rows changes neither the centroid nor the Frobenius norm, so a row permutation of the
    standardized cloud *is* the standardization of the row-permuted cloud.

    Args:
        Z: Point cloud, shape [n, d].

    Returns:
        The standardized cloud, shape [n, d].

    Raises:
        RuntimeError: On non-finite input, or a cloud whose centred Frobenius norm is 0 (all rows identical,
            so no shape to compare).
        ValueError: For fewer than :data:`MIN_ROWS` rows or a non-2-D array.
    """
    A = _check_finite(Z, "cloud")
    if A.ndim != 2:
        raise ValueError(f"standardize_cloud expects a 2-D [n, d] cloud, got shape {A.shape}.")
    if A.shape[0] < MIN_ROWS:
        raise ValueError(f"standardize_cloud needs n >= {MIN_ROWS} rows, got {A.shape[0]}.")
    A = A - A.mean(axis=0, keepdims=True)                                  # [n, d]
    norm = float(np.linalg.norm(A))
    if norm <= 0.0:
        raise RuntimeError("standardize_cloud: zero Frobenius norm after centring (every row identical).")
    return A / norm                                                        # [n, d]


def _reduced_factor(Z_std: np.ndarray) -> np.ndarray:
    """Factor a standardized cloud as ``Z = F V'`` with ``V`` orthonormal, returning ``F``.

    Singular values are invariant under multiplication by a matrix with orthonormal columns, so for two
    standardized clouds ``X = F_x V_x'`` and ``Y = F_y V_y'`` the singular values of ``X' Y`` equal those of
    ``F_x' F_y``. ``F`` is ``[n, r]`` with ``r <= min(n, d)``, which turns every permutation replicate into an
    ``[r, r]`` SVD instead of a ``[d, d]`` one -- the reduction that makes the null affordable, and a direct
    statement of Caveat 2 (at ``n = 96``, ``d = 192`` only 95 of the 192 directions are ever populated).

    Args:
        Z_std: Standardized cloud, shape [n, d].

    Returns:
        ``F``, shape [n, r].
    """
    U, S, _ = np.linalg.svd(Z_std, full_matrices=False)                    # [n, r], [r], [r, d]
    return U * S[None, :]                                                  # [n, r]


def _disparity_from_factors(Fx: np.ndarray, Fy: np.ndarray) -> float:
    """Disparity ``1 - (sum of singular values of Fx' Fy) ** 2`` of two unit-norm reduced factors.

    Args:
        Fx: Reduced factor of the first standardized cloud, shape [n, rx].
        Fy: Reduced factor of the second standardized cloud, shape [n, ry].

    Returns:
        The disparity in [0, 1].

    Raises:
        RuntimeError: If the value falls below ``-``:data:`NEGATIVE_DISPARITY_TOL` (a bug, not rounding).
    """
    sv = np.linalg.svd(Fx.T @ Fy, compute_uv=False)                        # [min(rx, ry)]
    value = 1.0 - float(sv.sum()) ** 2
    if value < -NEGATIVE_DISPARITY_TOL:
        raise RuntimeError(f"procrustes: disparity {value:.3e} is negative beyond tolerance; clouds not unit-norm?")
    return max(value, 0.0)


def procrustes_disparity(X: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal Procrustes disparity between two row-matched point clouds.

    Both clouds are centred and scaled to unit Frobenius norm, the orthogonal map ``R`` (rotation or reflection)
    and scale ``s`` minimizing ``||X - s Y R||_F`` are solved in closed form, and the residual
    ``1 - (sum_i sigma_i(X' Y)) ** 2`` is returned. Identical to ``scipy.spatial.procrustes``'s third return
    value (asserted in ``tests/analysis/test_procrustes.py``); computed through the reduction of
    :func:`_reduced_factor` so that :func:`procrustes_with_null`'s replicates are cheap.

    Row ``i`` of ``X`` must be the same site as row ``i`` of ``Y``: the statistic is only meaningful under that
    correspondence, and destroying it is exactly what the null of :func:`procrustes_with_null` does.

    **Prefer** :func:`procrustes_with_null`. A bare disparity is uninterpretable at ``n < d`` (Caveat 2 of the
    module docstring); this function exists for the tests and as the primitive the null is built from.

    Args:
        X: First cloud, shape [n, d].
        Y: Second cloud, shape [n, d].

    Returns:
        The disparity in [0, 1]; 0 iff the clouds agree up to translation, scale and an orthogonal map.

    Raises:
        ValueError: On mismatched shapes, a non-2-D array, or fewer than :data:`MIN_ROWS` rows.
        RuntimeError: On non-finite input or a degenerate (zero-norm) cloud.
    """
    if np.asarray(X).shape != np.asarray(Y).shape:
        raise ValueError(f"procrustes_disparity needs matching shapes, got {np.shape(X)} vs {np.shape(Y)}.")
    return _disparity_from_factors(_reduced_factor(standardize_cloud(X)), _reduced_factor(standardize_cloud(Y)))


def procrustes_with_null(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> ProcrustesNull:
    """Procrustes disparity of two row-matched clouds **and** its shuffled-site null, in one primitive.

    The null permutes the row correspondence of ``Y``, which destroys the site matching while leaving both
    clouds' marginal geometry (centroid, Frobenius norm, spectrum) untouched. It answers the question a raw
    disparity cannot at ``n < d``: how well can an orthogonal map align two *unrelated* ``n``-row clouds in ``d``
    dimensions? See Caveat 2 in the module docstring; the disparity and the null are returned together by
    construction so that one can never be reported without the other.

    Args:
        X: First cloud, shape [n, d] (the baseline).
        Y: Second cloud, shape [n, d], row-matched to ``X``.
        n_perm: Shuffled-site replicates; must be >= 1. Comes from a config key, never a literal
            (``procrustes.n_perm``). The percentile readout resolves ``1 / (n_perm + 1)``.
        rng: Generator, keyed by the cell's coordinates via :func:`pfns4neurostim.seeding.rng_for`
            (never a bare ``default_rng(0)``). Consumed: ``n_perm`` permutations.

    Returns:
        The :class:`ProcrustesNull` record, headline field :data:`HEADLINE_STATISTIC`.

    Raises:
        ValueError: On mismatched shapes, too few rows, or ``n_perm < 1``.
        RuntimeError: On non-finite input or a degenerate cloud.
    """
    if int(n_perm) < 1:
        raise ValueError(f"procrustes_with_null: n_perm must be >= 1, got {n_perm}.")
    if np.asarray(X).shape != np.asarray(Y).shape:
        raise ValueError(f"procrustes_with_null needs matching shapes, got {np.shape(X)} vs {np.shape(Y)}.")
    Xs, Ys = standardize_cloud(X), standardize_cloud(Y)                    # [n, d] each
    Fx, Fy = _reduced_factor(Xs), _reduced_factor(Ys)                      # [n, rx], [n, ry]
    n_rows, n_dim = Xs.shape
    observed = _disparity_from_factors(Fx, Fy)
    # A row permutation of the standardized cloud is the standardization of the row-permuted cloud, so the
    # reduced factor can simply be re-indexed: no re-standardization and no second [n, d] SVD per replicate.
    null = np.empty(int(n_perm), dtype=np.float64)                         # [n_perm]
    for i in range(int(n_perm)):
        null[i] = _disparity_from_factors(Fx, Fy[rng.permutation(n_rows)])
    null_mean = float(null.mean())
    return ProcrustesNull(
        disparity=observed,
        null_mean=null_mean,
        null_sd=float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        null_q05=float(np.quantile(null, 0.05)),
        disparity_minus_null=observed - null_mean,
        disparity_ratio=observed / null_mean if null_mean > 0 else float("inf"),
        null_percentile=float((null <= observed).mean()),
        n_perm=int(n_perm),
        n_rows=int(n_rows),
        n_dim=int(n_dim),
        n_rank=int(min(n_rows - 1, n_dim)),
    )
