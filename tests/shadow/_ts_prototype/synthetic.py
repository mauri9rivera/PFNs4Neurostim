"""Synthetic fixtures: GP-prior functions on grids and heteroscedastic toys."""
from __future__ import annotations

import numpy as np

from .preprocessing import SharedData, shared_preprocess


def grid(n_per_dim: int, dim: int) -> np.ndarray:
    """Regular grid in [0, 1]^dim, shape [n_per_dim**dim, dim]."""
    axes = [np.linspace(0.0, 1.0, n_per_dim)] * dim
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, dim)


def rbf_kernel(A: np.ndarray, B: np.ndarray, lengthscale: float, outputscale: float) -> np.ndarray:
    """ARD-free RBF kernel, shape [n, m]."""
    d2 = ((A[:, None, :] - B[None, :, :]) / lengthscale) ** 2
    return outputscale * np.exp(-0.5 * d2.sum(-1))


def sample_gp_functions(
    X: np.ndarray, n: int, lengthscale: float, outputscale: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw latent functions from a zero-mean GP prior.

    Args:
        X: Inputs, shape [M, D].
        n: Number of functions.
        lengthscale: RBF lengthscale.
        outputscale: RBF variance.
        rng: Explicit Generator.

    Returns:
        Functions, shape [n, M].
    """
    K = rbf_kernel(X, X, lengthscale, outputscale) + 1e-8 * np.eye(X.shape[0])
    L = np.linalg.cholesky(K)
    return rng.standard_normal((n, X.shape[0])) @ L.T


def gp_problem(
    X: np.ndarray, f: np.ndarray, noise_sd: np.ndarray | float, n_trials: int,
    rng: np.random.Generator, name: str = "synthetic",
) -> SharedData:
    """Wrap a latent function as a noisy multi-trial problem in shared units.

    Args:
        X: Inputs [M, D].
        f: Latent function values [M].
        noise_sd: Scalar or per-site noise SD [M].
        n_trials: Trials per site.
        rng: Explicit Generator.
        name: Identifier.

    Returns:
        SharedData (GT = latent f, rescaled with the trial statistics).
    """
    sd = np.broadcast_to(np.asarray(noise_sd, dtype=np.float64), f.shape)
    trials = f[:, None] + sd[:, None] * rng.standard_normal((f.size, n_trials))
    return shared_preprocess(X, trials, gt_mean=f, name=name)
