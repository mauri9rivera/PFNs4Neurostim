"""Joint Thompson sampling fallback of ``GPSurrogate.predict_ts`` (2026-11-18).

A fitted GP-MLL with long lengthscales produced a float32 latent posterior covariance that no jitter up to 1e-3 could
factor (NHP, M = 96: lambda_min -3.9e-3 against lambda_max 1.6), so every ts_joint cell failed. The fallback recomputes
the posterior in float64 closed form and factors it through a clipped eigendecomposition (``_psd_factor``).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pfns4neurostim.models.gp.surrogates import GPSurrogate, _psd_factor


def _fitted_gp() -> tuple[GPSurrogate, np.ndarray]:
    """A small CPU GP on a smooth 2-D surface, with a 12-point query grid."""
    rng = np.random.RandomState(0)
    X = rng.rand(15, 2).astype(np.float32)                                  # [15, 2]
    y = (np.sin(3.0 * X[:, 0]) + np.cos(2.0 * X[:, 1])).astype(np.float32)  # [15]
    gp = GPSurrogate(device="cpu")
    gp.fit(X, y)
    return gp, rng.rand(12, 2).astype(np.float32)                           # [12, 2]


def test_psd_factor_reproduces_a_psd_matrix() -> None:
    a = torch.randn(6, 6, dtype=torch.float64)
    cov = a @ a.T                                                           # [6, 6] PSD
    f = _psd_factor(cov, psd_tolerance=1e-3)                                # [6, 6]
    assert torch.allclose(f @ f.T, cov, atol=1e-10)


def test_psd_factor_clips_round_off_negatives() -> None:
    v = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))[0]          # [5, 5] orthonormal
    lam = torch.tensor([2.0, 1.0, 0.5, 1e-3, -1e-6], dtype=torch.float64)   # one round-off negative
    cov = v @ torch.diag(lam) @ v.T                                         # [5, 5]
    f = _psd_factor(cov, psd_tolerance=1e-3)
    assert torch.allclose(f @ f.T, v @ torch.diag(lam.clamp(min=0.0)) @ v.T, atol=1e-10)


def test_psd_factor_rejects_a_genuinely_indefinite_matrix() -> None:
    cov = torch.diag(torch.tensor([1.0, -0.5], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="indefinite beyond round-off"):
        _psd_factor(cov, psd_tolerance=1e-3)


def test_predict_ts_falls_back_to_the_float64_posterior(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every Cholesky refused, draws come from the float64 closed-form posterior N(mean, Sigma)."""
    gp, Xq = _fitted_gp()
    mean_ref, _ = gp.predict_frozen(Xq)                                     # [M]
    cov_ref = gp.posterior_covariance(Xq)                                   # [M, M]

    def refuse(*args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("cholesky refused (test)")

    monkeypatch.setattr(torch.linalg, "cholesky", refuse)
    g = torch.Generator().manual_seed(42)
    draws = np.stack([gp.predict_ts(Xq, generator=g) for _ in range(4000)])  # [S, M]
    assert np.isfinite(draws).all()
    sd = np.sqrt(np.clip(np.diag(cov_ref), 1e-12, None))                    # [M]
    # Monte Carlo error of the mean is sd / sqrt(4000) ~ 0.016 sd: 0.1 sd is a loose, non-flaky bound.
    assert np.all(np.abs(draws.mean(0) - mean_ref) <= 0.1 * sd + 1e-6)
    emp = np.cov(draws.T)                                                   # [M, M]
    scale = float(np.max(np.diag(cov_ref)))
    assert np.max(np.abs(emp - cov_ref)) <= 0.1 * scale + 1e-8
