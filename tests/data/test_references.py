"""Standardization and on-grid reference banks (task #6 Steps 0-1)."""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.data.references.grid import (
    interpolate_to_grid,
    noise_grid_bank,
    prior_grid_bank,
    standardize_source,
)


def _grid(side: int = 8) -> np.ndarray:
    ax = np.linspace(0, 1, side)
    return np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)


def _smooth_sampler(d: int, n: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(s)
    X = rng.uniform(size=(n, d))
    return X, np.sin(4 * X[:, 0]) + np.cos(3 * X[:, 1])


def _rough_sampler(d: int, n: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(s)
    return rng.uniform(size=(n, d)), rng.normal(size=n)


def test_standardize_source() -> None:
    X, y = standardize_source(np.array([[2.0, 5.0], [4.0, 5.0], [3.0, 5.0]]), np.array([1.0, 2.0, 3.0]))
    assert X.min() == 0.0 and X[:, 0].max() == 1.0 and (X[:, 1] == 0).all()
    assert y.mean() == pytest.approx(0.0) and y.std() == pytest.approx(1.0)


def test_every_bank_is_standardized() -> None:
    G = _grid()
    for bank in (noise_grid_bank(G, 10, seed=0), prior_grid_bank(G, 5, n_dense=256, sampler=_smooth_sampler)):
        np.testing.assert_allclose(bank.maps.mean(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(bank.maps.std(axis=1), 1.0, atol=1e-10)


def test_interpolation_error_small_for_smooth_function() -> None:
    X, y = standardize_source(*_smooth_sampler(2, 1024, 0))
    y_grid, rmse = interpolate_to_grid(X, y, _grid(), holdout_frac=0.1, rng=np.random.default_rng(0))
    assert rmse < 0.1 and np.isfinite(y_grid).all()


def test_rough_datasets_are_rejected_and_rate_reported() -> None:
    G = _grid()
    bank = prior_grid_bank(G, 4, n_dense=256, rmse_threshold=0.5, sampler=_smooth_sampler)
    assert bank.rejection_rate == 0.0 and len(bank.rmse) == 4
    with pytest.raises(RuntimeError, match="rmse_threshold"):
        prior_grid_bank(G, 3, n_dense=256, rmse_threshold=0.5, sampler=_rough_sampler)


@pytest.mark.slow
def test_real_prior_bag_bank_on_nhp_like_grid() -> None:
    bank = prior_grid_bank(_grid(10), 5, n_dense=1024, rmse_threshold=0.5, seed=0)
    assert bank.maps.shape[1] == 100 and 0.0 <= bank.rejection_rate < 1.0
