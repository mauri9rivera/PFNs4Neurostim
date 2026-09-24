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

from typing import Any

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
]


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
    obs = cka_debiased(K, L)
    kk = hsic_unbiased(K, K)
    ll = hsic_unbiased(L, L)       # invariant to a joint row/column permutation
    null = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(len(L))
        null[b] = hsic_unbiased(K, L[np.ix_(perm, perm)]) / np.sqrt(kk * ll)
    p = (1.0 + float((null >= obs).sum())) / (n_perm + 1.0)
    return obs, float(null.mean()), p
