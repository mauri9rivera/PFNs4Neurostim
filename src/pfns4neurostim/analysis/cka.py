"""Debiased centred kernel alignment and CKA target kernels (task #5, P0.5).

CKA compares two *pictures of the same sites*: rows correspond (site i in one matrix is
site i in the other). The estimator is the **debiased** CKA built on the unbiased HSIC
estimator (Song et al. 2012, Eq. 5)::

    HSIC_1(K, L) = [tr(K~ L~) + (1'K~1)(1'L~1) / ((n-1)(n-2)) - 2/(n-2) 1'K~L~1] / (n(n-3))
    CKA(K, L)    = HSIC_1(K, L) / sqrt(HSIC_1(K, K) HSIC_1(L, L))

with ``K~`` = ``K`` with its diagonal set to 0. The biased estimator is inflated when the
number of sites (32-96) is far below the embedding dimension (192); the debiased one is not
(Kornblith et al. 2019; Murphy et al. 2024). ``biased_linear_cka`` is the pre-restructure
estimator, kept only for comparison.

Target kernels over the same N sites (P0.5): ``K_X`` = RBF on coordinates (median
bandwidth; a *control*, since coordinates are the input), ``K_GT`` = RBF on the
standardized ground-truth response (median |dy| bandwidth; linear ``y y'`` as sensitivity),
``K_GP`` = latent posterior covariance of the MLL-tuned GP fitted on the same context.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "biased_linear_cka",
    "linear_gram",
    "hsic_unbiased",
    "cka_debiased",
    "rbf_gram",
    "median_distance",
    "target_kernels",
    "permutation_null",
    "cka_null_draws",
    "permutation_p",
    "PERM_CHUNK",
]

#: Permutations evaluated per vectorised chunk of :func:`cka_null_draws`. A memory bound, not a statistical
#: setting: one chunk holds ``PERM_CHUNK x N x N`` float64 (19 MB at N = 96), whatever ``n_perm`` is.
PERM_CHUNK: int = 256


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
    """The P0.5 target kernels over the same sites.

    Args:
        X: Coordinates in [0, 1]^D, shape [N, D].
        y_gt: Ground-truth responses, shape [N] (standardized here).
        gp_posterior_cov: Latent posterior covariance of the MLL GP on the same context,
            shape [N, N]; omitted when ``None``.

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


def cka_null_draws(
    Ks: Sequence[np.ndarray],
    L: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed debiased CKA of every ``K`` against ``L``, and its site-permutation null draws.

    The null permutes the rows *and* columns of ``L`` jointly. Its permutations depend on the target
    alone, so ONE set of ``n_perm`` permutations is drawn and shared by every ``K`` -- every layer and
    decoder stage of a cell, every feature token. That is what makes the null cheap: the permuted
    ``L~`` stack is gathered once per chunk and each ``K`` costs one contraction over it, instead of a
    fresh gather per (layer, permutation) as before 2026-10-06 (~10x faster on the NHP grid, and the
    95 %-of-cost term of CKA (a)). Sharing permutations across layers correlates their p-values;
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
    """Site-permutation null: permute rows *and* columns of ``L`` together.

    Args:
        K: Gram matrix, shape [N, N].
        L: Gram matrix, shape [N, N].
        n_perm: Permutations (1000 in the protocol).
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
