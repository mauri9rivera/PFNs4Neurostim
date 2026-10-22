"""Debiased centred kernel alignment, geometry deconfounding and the smooth-field surrogate (task #5, P0.5; BETA).

CKA compares two *pictures of the same sites*: rows correspond (site i in one matrix is
site i in the other). The estimator is the **debiased** CKA built on the unbiased HSIC
estimator (Song et al. 2012, Eq. 5)::

    HSIC_1(K, L) = [tr(K~ L~) + (1'K~1)(1'L~1) / ((n-1)(n-2)) - 2/(n-2) 1'K~L~1] / (n(n-3))
    CKA(K, L)    = HSIC_1(K, L) / sqrt(HSIC_1(K, K) HSIC_1(L, L))

with ``K~`` = ``K`` with its diagonal set to 0. The biased estimator is inflated when the
number of sites (32-96) is far below the embedding dimension (192); the debiased one is not
(Kornblith et al. 2019; Murphy et al. 2024). ``biased_linear_cka`` is the pre-restructure
estimator, kept only for comparison.

**Geometry deconfounding (BETA D3d, 2026-10-19; Cui et al. 2022).** Every kernel over the electrode grid inherits the
grid's geometry, so unrelated smooth fields reach a debiased CKA of 0.2-0.9 with one another. :class:`Deconfounder`
regresses the off-diagonal entries of a double-centred Gram on those of a set of confound Grams -- RBF kernels on the
coordinates at several bandwidths (:func:`geometry_confounds`), plus GP-fixed's posterior covariance for covariance
targets -- in ONE joint least-squares fit, and keeps the residual. The same confound set is applied to the
representation and to the target, and the deconfounded CKA is the debiased CKA of the two residuals
(:func:`cka_deconfounded`). Only the off-diagonal entries enter: the unbiased HSIC zeroes the diagonals anyway.

**Chance level (BETA D3d).** :func:`smooth_field_surrogate` draws a representation of independent GP fields on the
same sites; run through the identical pipeline it gives the CKA unrelated smooth fields reach. The site-permutation
null (:func:`cka_null_draws`) destroys spatial smoothness, so it is kept as a *descriptive* chance level only.

Target kernels over the same N sites (:func:`target_kernels`): ``K_X`` = RBF on coordinates (median bandwidth; the
confound itself), ``K_GT`` = RBF on the standardized ground-truth response (median |dy| bandwidth; a secondary,
accuracy reference), ``K_GT_linear`` = ``y y'``.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "biased_linear_cka",
    "linear_gram",
    "centered_linear_gram",
    "double_center",
    "hsic_unbiased",
    "cka_debiased",
    "cka_or_none",
    "rbf_gram",
    "median_distance",
    "target_kernels",
    "geometry_confounds",
    "Deconfounder",
    "deconfound",
    "cka_deconfounded",
    "smooth_field_surrogate",
    "effective_rank",
    "permutation_null",
    "cka_null_draws",
    "permutation_p",
    "PERM_CHUNK",
    "MIN_RESIDUAL_FRAC",
]

#: Permutations evaluated per vectorised chunk of :func:`cka_null_draws`. A memory bound, not a statistical
#: setting: one chunk holds ``PERM_CHUNK x N x N`` float64 (19 MB at N = 96), whatever ``n_perm`` is.
PERM_CHUNK: int = 256

#: Default share of a Gram's off-diagonal energy that must survive deconfounding for the residual to be read. Below it
#: the Gram lies in the span of its confounds (a pure-geometry representation), so there is no signal beyond the
#: confounds and :func:`cka_deconfounded` returns 0 with ``degenerate = True`` instead of dividing rounding noise.
#: The runners pass ``representation.deconfound.min_residual_frac``.
MIN_RESIDUAL_FRAC: float = 1e-10


def biased_linear_cka(X: Any, Y: Any) -> float:
    """Pre-restructure *biased* linear CKA on torch activations [n, d] (comparison only)."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = (Y.T @ X).norm() ** 2
    hsic_xx = (X.T @ X).norm()
    hsic_yy = (Y.T @ Y).norm()
    return (hsic_xy / (hsic_xx * hsic_yy + 1e-10)).item()


def linear_gram(Z: np.ndarray) -> np.ndarray:
    """Linear Gram matrix ``Z Z'`` of site representations.

    Args:
        Z: Representations, shape [N, d].

    Returns:
        Gram matrix, shape [N, N].
    """
    Z = np.asarray(Z, dtype=np.float64)
    return Z @ Z.T


def centered_linear_gram(Z: np.ndarray) -> np.ndarray:
    """Linear Gram of the column-centred representation: ``(Z - 1 z_bar')(Z - 1 z_bar')'``.

    A vector (a mean map, an SD map) is treated as a one-column representation, so its Gram is the outer product of
    the centred map -- the BETA location / spread targets (D3c).

    Args:
        Z: Representations, shape [N, d] or [N].

    Returns:
        Gram matrix, shape [N, N].
    """
    Z = np.asarray(Z, dtype=np.float64).reshape(len(Z), -1)               # [N, d]
    Zc = Z - Z.mean(axis=0, keepdims=True)                                 # [N, d]
    return Zc @ Zc.T


def double_center(K: np.ndarray) -> np.ndarray:
    """Double-centre a Gram matrix: ``H K H`` with ``H = I - 11'/n``.

    Args:
        K: Gram matrix, shape [N, N].

    Returns:
        The centred matrix, shape [N, N]; for a linear Gram it is the Gram of the column-centred representation.
    """
    K = np.asarray(K, dtype=np.float64)
    row = K.mean(axis=1, keepdims=True)                                    # [N, 1]
    col = K.mean(axis=0, keepdims=True)                                    # [1, N]
    return K - row - col + K.mean()


def hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al. 2012, Eq. 5).

    Args:
        K: Gram matrix, shape [N, N].
        L: Gram matrix, shape [N, N].

    Returns:
        The estimate (may be slightly negative under independence).

    Raises:
        ValueError: For fewer than 4 sites.
    """
    n = K.shape[0]
    if n < 4:
        raise ValueError(f"hsic_unbiased needs n >= 4 sites, got {n}.")
    Kt = K.copy()
    Lt = L.copy()
    np.fill_diagonal(Kt, 0.0)
    np.fill_diagonal(Lt, 0.0)
    one = np.ones(n)
    term1 = float(np.sum(Kt * Lt.T))                               # tr(K~ L~)
    term2 = float(one @ Kt @ one) * float(one @ Lt @ one) / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * float(one @ Kt @ Lt @ one)
    return (term1 + term2 - term3) / (n * (n - 3))


def cka_debiased(K: np.ndarray, L: np.ndarray) -> float:
    """Debiased CKA between two Gram matrices over the same sites.

    Args:
        K: Gram matrix, shape [N, N] (e.g. ``linear_gram(Z)``).
        L: Gram matrix, shape [N, N].

    Returns:
        CKA in roughly [-small, 1]; exactly 1 for ``K`` proportional to ``L``.

    Raises:
        RuntimeError: If a self-HSIC is non-positive (degenerate representation).
    """
    kk, ll = hsic_unbiased(K, K), hsic_unbiased(L, L)
    if kk <= 0 or ll <= 0:
        raise RuntimeError(f"cka_debiased: non-positive self-HSIC ({kk:.3g}, {ll:.3g}).")
    return float(hsic_unbiased(K, L) / np.sqrt(kk * ll))


def cka_or_none(K: np.ndarray, L: np.ndarray) -> float | None:
    """Debiased CKA, or ``None`` when either side is degenerate (non-positive self-HSIC).

    The explicit ``None`` (an empty CSV cell, flagged by the caller) replaces the exception for readouts that are
    degenerate *by construction* -- the block-0 label token is constant across sites (BETA §2.5, pitfall 3) -- so a
    whole cell is not lost to one pre-information block. Every other failure still raises.

    Args:
        K: Gram matrix, shape [N, N].
        L: Gram matrix, shape [N, N].

    Returns:
        The CKA, or ``None``.
    """
    kk, ll = hsic_unbiased(K, K), hsic_unbiased(L, L)
    if kk <= 0 or ll <= 0:
        return None
    value = float(hsic_unbiased(K, L) / np.sqrt(kk * ll))
    if not np.isfinite(value):
        raise RuntimeError(f"cka_or_none: non-finite CKA ({value}) from finite self-HSICs ({kk:.3g}, {ll:.3g}).")
    return value


def median_distance(V: np.ndarray) -> float:
    """Median pairwise Euclidean distance (the median heuristic bandwidth).

    Args:
        V: Points, shape [N, F] (or [N] for scalars).

    Returns:
        The median over i < j; raises if 0.
    """
    V = np.asarray(V, dtype=np.float64).reshape(len(V), -1)
    d = np.sqrt(np.maximum(((V[:, None] - V[None]) ** 2).sum(-1), 0.0))
    med = float(np.median(d[np.triu_indices(len(V), 1)]))
    if med <= 0:
        raise RuntimeError("median_distance: zero median; bandwidth undefined.")
    return med


def rbf_gram(V: np.ndarray, bandwidth: float) -> np.ndarray:
    """RBF Gram ``exp(-||v_i - v_j||^2 / (2 bw^2))``.

    Args:
        V: Points, shape [N, F] or [N].
        bandwidth: Bandwidth.

    Returns:
        Gram matrix, shape [N, N].
    """
    V = np.asarray(V, dtype=np.float64).reshape(len(V), -1)
    d2 = np.maximum(((V[:, None] - V[None]) ** 2).sum(-1), 0.0)
    return np.exp(-d2 / (2.0 * bandwidth ** 2))


def target_kernels(
    X: np.ndarray,
    y_gt: np.ndarray,
    gp_posterior_cov: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Coordinate and ground-truth kernels over the same sites.

    Args:
        X: Coordinates in [0, 1]^D, shape [N, D].
        y_gt: Ground-truth responses, shape [N] (standardized here).
        gp_posterior_cov: A GP latent posterior covariance on the same sites, shape [N, N]; returned as ``K_GP``
            when given (the retired CKA (a) target; kept for callers that compare against one).

    Returns:
        ``{'K_X', 'K_GT', 'K_GT_linear'[, 'K_GP']}``.
    """
    y = (np.asarray(y_gt, dtype=np.float64) - np.mean(y_gt)) / np.std(y_gt)
    out = {
        "K_X": rbf_gram(X, median_distance(X)),
        "K_GT": rbf_gram(y, median_distance(y)),
        "K_GT_linear": np.outer(y, y),
    }
    if gp_posterior_cov is not None:
        out["K_GP"] = np.asarray(gp_posterior_cov, dtype=np.float64)
    return out


def geometry_confounds(X: np.ndarray, multipliers: Sequence[float]) -> list[np.ndarray]:
    """RBF kernels on the coordinates at ``multipliers x`` the median inter-site distance (BETA D3d).

    Args:
        X: Site coordinates, shape [N, D].
        multipliers: Bandwidth multipliers (``representation.deconfound.bandwidth_multipliers``; 0.25, 0.5, 1, 2).

    Returns:
        One Gram per multiplier, each [N, N].

    Raises:
        ValueError: For an empty or non-positive multiplier list.
    """
    mult = [float(m) for m in multipliers]
    if not mult or any(m <= 0 for m in mult):
        raise ValueError(f"geometry_confounds: multipliers must be positive and non-empty, got {list(multipliers)}.")
    med = median_distance(X)
    return [rbf_gram(X, m * med) for m in mult]


def _offdiag(K: np.ndarray) -> np.ndarray:
    """Off-diagonal entries of a square matrix in row-major order, shape [N (N - 1)]."""
    n = K.shape[0]
    return np.asarray(K, dtype=np.float64)[~np.eye(n, dtype=bool)]


class Deconfounder:
    """Joint least-squares removal of confound Grams from the off-diagonals of double-centred Grams (Cui et al. 2022).

    For a Gram ``K`` the residual is ``R = HKH - b0 - sum_j b_j H C_j H`` on the off-diagonal entries, with
    ``(b0, b)`` the ordinary least-squares fit of ``offdiag(HKH)`` on an intercept and ``offdiag(H C_j H)``, and
    ``R``'s diagonal set to 0 (the unbiased HSIC never reads it). The design is factorised once, so every Gram of a
    cell -- each block, readout and surrogate -- is residualised by one projection.

    Args:
        confounds: Confound Grams, each [N, N] (geometry at every bandwidth; GP-fixed's covariance for covariance
            targets). An empty list removes the intercept only.

    Raises:
        RuntimeError: If a confound is non-finite.
        ValueError: On mismatched shapes or fewer than 4 sites.
    """

    def __init__(self, confounds: Sequence[np.ndarray]) -> None:
        mats = [np.asarray(C, dtype=np.float64) for C in confounds]
        if not mats:
            raise ValueError("Deconfounder needs at least one confound Gram (an intercept alone is not a confound).")
        n = mats[0].shape[0]
        if n < 4:
            raise ValueError(f"Deconfounder needs n >= 4 sites, got {n}.")
        for C in mats:
            if C.shape != (n, n):
                raise ValueError(f"Deconfounder: confound shape {C.shape} != ({n}, {n}).")
            if not np.isfinite(C).all():
                raise RuntimeError("Deconfounder: a confound Gram holds NaN/Inf.")
        self.n = n
        self.n_confounds = len(mats)
        design = np.column_stack([np.ones(n * (n - 1))] + [_offdiag(double_center(C)) for C in mats])   # [n(n-1), 1+c]
        Q, _ = np.linalg.qr(design)                                         # [n(n-1), r] orthonormal basis
        self._Q = Q
        self._mask = ~np.eye(n, dtype=bool)

    def residual(self, K: np.ndarray) -> np.ndarray:
        """The deconfounded Gram: off-diagonal residual of ``HKH``, zero diagonal.

        Args:
            K: Gram matrix, shape [N, N].

        Returns:
            Residual matrix, shape [N, N].

        Raises:
            RuntimeError: On non-finite input.
            ValueError: On a shape mismatch.
        """
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (self.n, self.n):
            raise ValueError(f"Deconfounder.residual: shape {K.shape} != ({self.n}, {self.n}).")
        if not np.isfinite(K).all():
            raise RuntimeError("Deconfounder.residual: Gram holds NaN/Inf.")
        v = _offdiag(double_center(K))                                     # [n(n-1)]
        r = v - self._Q @ (self._Q.T @ v)                                   # [n(n-1)]
        out = np.zeros((self.n, self.n))
        out[self._mask] = r
        return out

    def residual_fraction(self, K: np.ndarray) -> float:
        """Share of ``offdiag(HKH)``'s squared norm that survives the projection (0 = entirely confound)."""
        v = _offdiag(double_center(np.asarray(K, dtype=np.float64)))
        total = float(v @ v)
        if total <= 0:
            return 0.0
        r = v - self._Q @ (self._Q.T @ v)
        return float(r @ r) / total


def deconfound(K: np.ndarray, confounds: Sequence[np.ndarray]) -> np.ndarray:
    """One Gram deconfounded on a set of confound Grams (see :class:`Deconfounder`).

    Args:
        K: Gram matrix, shape [N, N].
        confounds: Confound Grams, each [N, N].

    Returns:
        The residual matrix, shape [N, N] (zero diagonal).
    """
    return Deconfounder(confounds).residual(K)


def cka_deconfounded(
    K: np.ndarray,
    L: np.ndarray,
    deconfounder: Deconfounder,
    *,
    min_residual_frac: float = MIN_RESIDUAL_FRAC,
) -> dict[str, Any]:
    """Debiased CKA of two Grams after removing the same confounds from both (BETA D3d; Cui et al. 2022).

    Args:
        K: Representation Gram, shape [N, N].
        L: Target Gram, shape [N, N].
        deconfounder: The confound set, applied identically to ``K`` and ``L``.
        min_residual_frac: Below this surviving share of either side's off-diagonal energy the side is "entirely
            confound": the value is then 0 (no alignment beyond the confounds) and ``degenerate`` is ``True``.

    Returns:
        ``{'value', 'residual_frac_rep', 'residual_frac_target', 'degenerate'}``; ``value`` is ``None`` only when a
        residual is non-degenerate by the energy test but has a non-positive self-HSIC (reported, never silent).
    """
    fk, fl = deconfounder.residual_fraction(K), deconfounder.residual_fraction(L)
    out: dict[str, Any] = {"residual_frac_rep": fk, "residual_frac_target": fl, "degenerate": False}
    if fk < float(min_residual_frac) or fl < float(min_residual_frac):
        return {**out, "value": 0.0, "degenerate": True}
    return {**out, "value": cka_or_none(deconfounder.residual(K), deconfounder.residual(L))}


def smooth_field_surrogate(
    X: np.ndarray,
    lengthscale: float,
    n_fields: int,
    rng: np.random.Generator,
    *,
    jitter: float,
) -> np.ndarray:
    """A surrogate representation: ``n_fields`` independent unit-variance RBF-GP fields on the sites (BETA D3d).

    Unrelated to any target by construction yet as smooth as the grid allows, so its CKA against a target is the
    chance level of smooth fields (Burt et al. 2020: a site-permutation null destroys exactly this smoothness).

    Args:
        X: Site coordinates in [0, 1]^D, shape [N, D].
        lengthscale: RBF lengthscale of the fields (normalised coordinates).
        n_fields: Fields (columns), the representation width.
        rng: Generator.
        jitter: Diagonal added before the Cholesky factorisation.

    Returns:
        Surrogate representation, shape [N, n_fields].

    Raises:
        ValueError: On a non-positive lengthscale or field count.
    """
    if float(lengthscale) <= 0 or int(n_fields) < 1:
        raise ValueError(f"smooth_field_surrogate: lengthscale > 0 and n_fields >= 1 required, got "
                         f"{lengthscale}, {n_fields}.")
    X = np.asarray(X, dtype=np.float64)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)                   # [N, N]
    K = np.exp(-0.5 * d2 / float(lengthscale) ** 2) + float(jitter) * np.eye(len(X))
    Lc = np.linalg.cholesky(K)                                              # [N, N]
    return Lc @ rng.standard_normal((len(X), int(n_fields)))                # [N, n_fields]


def effective_rank(K: np.ndarray) -> float:
    """Participation ratio ``(sum lambda)^2 / sum lambda^2`` of a double-centred Gram's eigenvalues.

    1 for a rank-one target (a mean-map outer product), up to N - 1 for an isotropic cloud; the audit's replacement
    for the "remove PC1" remedy on belief targets (BETA §2.5 amendment, pitfall 2).

    Args:
        K: Gram matrix, shape [N, N].

    Returns:
        The effective rank (0 for an all-zero Gram).
    """
    lam = np.clip(np.linalg.eigvalsh(0.5 * (double_center(K) + double_center(K).T)), 0.0, None)   # [N]
    s2 = float((lam ** 2).sum())
    return 0.0 if s2 == 0 else float(lam.sum() ** 2 / s2)


def cka_null_draws(
    Ks: Sequence[np.ndarray],
    L: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed debiased CKA of every ``K`` against ``L``, and its site-permutation null draws.

    A *descriptive* chance level since BETA D4 / D3d: it destroys the spatial smoothness every grid kernel has, so it
    reports ~0 where unrelated smooth fields reach 0.2-0.9 (the smooth-field surrogate is the chance level).

    The null permutes the rows *and* columns of ``L`` jointly. Its permutations depend on the target
    alone, so ONE set of ``n_perm`` permutations is drawn and shared by every ``K`` -- every layer and
    readout of a cell. That is what makes the null cheap: the permuted
    ``L~`` stack is gathered once per chunk and each ``K`` costs one contraction over it.
    Sharing permutations across layers correlates their p-values;
    Bonferroni over layers stays valid under any dependence.

    The unbiased HSIC (Song et al. 2012, Eq. 5) of ``K`` with a permuted ``L`` needs only three terms:
    ``tr(K~ L~_p) = sum(K~ * (L~')_p)``, the invariant ``(1'K~1)(1'L~1)`` and
    ``1'K~ L~_p 1 = colsum(K~) . rowsum(L~)[p]``, so no matrix product is formed per permutation.

    Args:
        Ks: Gram matrices, each [N, N] (need not be symmetric).
        L: Target Gram matrix, [N, N].
        n_perm: Permutations. With a single ``K`` the draws match :func:`permutation_null`'s for the same
            ``rng`` state (same permutation sequence).
        rng: Generator (consumed: ``n_perm`` permutations).

    Returns:
        ``(observed, null)`` of shapes [k] and [k, n_perm].

    Raises:
        ValueError: For fewer than 4 sites.
        RuntimeError: If a self-HSIC is non-positive (degenerate representation).
    """
    L = np.asarray(L, dtype=np.float64)
    n = L.shape[0]
    if n < 4:
        raise ValueError(f"cka_null_draws needs n >= 4 sites, got {n}.")
    ll = hsic_unbiased(L, L)
    if ll <= 0:
        raise RuntimeError(f"cka_null_draws: non-positive target self-HSIC ({ll:.3g}).")
    Lt_T = L.T.copy()
    np.fill_diagonal(Lt_T, 0.0)                                        # (L~)'  [N, N]
    l_row = Lt_T.sum(axis=0)                                           # rowsum(L~) = colsum((L~)')  [N]
    l_tot = float(l_row.sum())
    Kts, k_col, kk, obs = [], [], [], []
    for K in Ks:
        K = np.asarray(K, dtype=np.float64)
        Kt = K.copy()
        np.fill_diagonal(Kt, 0.0)
        k_self = hsic_unbiased(K, K)
        if k_self <= 0:
            raise RuntimeError(f"cka_null_draws: non-positive self-HSIC ({k_self:.3g}).")
        Kts.append(Kt)
        k_col.append(Kt.sum(axis=0))                                   # colsum(K~)  [N]
        kk.append(k_self)
        obs.append(hsic_unbiased(K, L) / np.sqrt(k_self * ll))
    Kt_stack = np.stack(Kts)                                           # [k, N, N]
    k_col_m = np.stack(k_col)                                          # [k, N]
    k_tot = Kt_stack.sum(axis=(1, 2))                                  # [k]
    denom = np.sqrt(np.asarray(kk) * ll)                               # [k]
    term2 = k_tot * l_tot / ((n - 1) * (n - 2))                        # [k]
    perms = np.stack([rng.permutation(n) for _ in range(int(n_perm))]) # [P, N]
    null = np.empty((len(Ks), int(n_perm)))
    for start in range(0, int(n_perm), PERM_CHUNK):
        P = perms[start:start + PERM_CHUNK]                            # [c, N]
        Lp = Lt_T[P[:, :, None], P[:, None, :]]                        # [c, N, N]  ((L~')_p)
        term1 = np.einsum("kij,cij->kc", Kt_stack, Lp)                 # [k, c]
        term3 = 2.0 / (n - 2) * (k_col_m @ l_row[P].T)                 # [k, c]
        hsic = (term1 + term2[:, None] - term3) / (n * (n - 3))        # [k, c]
        null[:, start:start + PERM_CHUNK] = hsic / denom[:, None]
    return np.asarray(obs, dtype=np.float64), null


def permutation_null(
    K: np.ndarray,
    L: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Site-permutation null: permute rows *and* columns of ``L`` together (descriptive only, see above).

    Args:
        K: Gram matrix, shape [N, N].
        L: Gram matrix, shape [N, N].
        n_perm: Permutations.
        rng: Generator.

    Returns:
        ``(observed, null_mean, p)`` with ``p = (1 + #{null >= observed}) / (n_perm + 1)``.
    """
    obs, null = cka_null_draws([K], L, n_perm, rng)
    return float(obs[0]), float(null[0].mean()), permutation_p(float(obs[0]), null[0])


def permutation_p(observed: float, null: np.ndarray) -> float:
    """One-sided permutation p-value ``(1 + #{null >= observed}) / (n_perm + 1)``.

    Args:
        observed: Observed statistic.
        null: Null draws, shape [n_perm].

    Returns:
        The p-value.
    """
    return (1.0 + float((np.asarray(null) >= observed).sum())) / (len(null) + 1.0)
