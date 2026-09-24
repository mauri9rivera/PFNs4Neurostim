"""Closed-form GP posterior update (task #9 Step 0)."""
from __future__ import annotations

import gpytorch
import numpy as np
import pytest
import torch

from pfns4neurostim.models.gp.surrogates import GPSurrogate, NaiveGPSurrogate


@pytest.fixture
def fitted() -> tuple[GPSurrogate, np.ndarray, np.ndarray, np.ndarray]:
    """An MLL-fitted GP on 15 noisy points of a smooth 2D function, plus a 49-site grid."""
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(15, 2))
    y = np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1]) + 0.1 * rng.normal(size=15)
    ax = np.linspace(0, 1, 7)
    Q = np.stack(np.meshgrid(ax, ax), -1).reshape(-1, 2)
    gp = GPSurrogate(n_opt_steps=60)
    gp.fit(X, y)
    return gp, X, y, Q


def test_predict_frozen_matches_gpytorch(fitted) -> None:
    gp, _, _, Q = fitted
    m_ref, s_ref = gp.predict(Q)
    m, s = gp.predict_frozen(Q)
    np.testing.assert_allclose(m, m_ref, atol=2e-4)
    np.testing.assert_allclose(s, s_ref, atol=2e-4)


def test_frozen_update_equals_direct_conditioning(fitted) -> None:
    """dmu and dsd equal the posterior on C + {(x*, y*)} minus the posterior on C."""
    gp, X, y, Q = fitted
    xs = np.array([[0.3, 0.7], [0.9, 0.1]])
    ys = np.array([2.0, -1.0])
    dm, ds = gp.posterior_update(xs, ys, Q)
    m0, s0 = gp.predict_frozen(Q)
    for p in range(2):
        m1, s1 = gp.predict_frozen(Q, X_train=np.vstack([X, xs[p]]), y_train=np.append(y, ys[p]))
        np.testing.assert_allclose(dm[p], m1 - m0, atol=1e-10)
        np.testing.assert_allclose(ds[p], s1 - s0, atol=1e-10)


def test_frozen_update_matches_gpytorch_reconditioning(fitted) -> None:
    """Same numbers from gpytorch with the training set swapped and hyperparameters kept."""
    gp, X, y, Q = fitted
    xs, ys = np.array([[0.5, 0.5]]), np.array([1.5])
    dm, _ = gp.posterior_update(xs, ys, Q)
    m0, _ = gp.predict(Q)
    Xa = torch.tensor(np.vstack([X, xs]), dtype=torch.float32)
    ya = torch.tensor(np.append(y, ys), dtype=torch.float32)
    gp._model.set_train_data(Xa, ya, strict=False)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
        m1 = gp._model(torch.tensor(Q, dtype=torch.float32)).mean.numpy()
    np.testing.assert_allclose(dm[0], m1 - m0, atol=5e-4)


def test_variance_update_is_value_independent(fitted) -> None:
    """H10.4 reference: a GP's SD change does not depend on y*."""
    gp, _, _, Q = fitted
    xs = np.repeat([[0.4, 0.4]], 3, axis=0)
    _, ds = gp.posterior_update(xs, np.array([-3.0, 0.0, 3.0]), Q)
    np.testing.assert_allclose(ds[0], ds[2])
    assert (ds <= 1e-12).all()


def test_update_is_linear_in_surprise(fitted) -> None:
    gp, _, _, Q = fitted
    xs = np.repeat([[0.2, 0.8]], 3, axis=0)
    mu = gp.predict_frozen(xs[:1])[0][0]
    dm, _ = gp.posterior_update(xs, mu + np.array([1.0, 2.0, -1.0]), Q)
    np.testing.assert_allclose(dm[1], 2 * dm[0], atol=1e-12)
    np.testing.assert_allclose(dm[2], -dm[0], atol=1e-12)


def test_refit_arm_records_hyperparameters(fitted) -> None:
    gp, _, _, Q = fitted
    dm, ds = gp.posterior_update(np.array([[0.5, 0.5]]), np.array([3.0]), Q, refit=True)
    assert dm.shape == ds.shape == (1, len(Q))
    assert len(gp.last_refit_hyperparameters) == 1
    # The refit leaves the original surrogate untouched.
    np.testing.assert_allclose(gp.predict_frozen(Q)[0], gp.predict_frozen(Q)[0])


def test_naive_gp_keeps_pinned_hyperparameters() -> None:
    gp = NaiveGPSurrogate(lengthscale=0.3, outputscale=2.0, noise=0.05)
    gp.fit(np.random.default_rng(0).uniform(size=(5, 2)), np.arange(5.0))
    hp = gp.hyperparameters()
    np.testing.assert_allclose(hp["lengthscale"], [0.3, 0.3], rtol=1e-5)
    assert hp["outputscale"] == pytest.approx(2.0, rel=1e-5)
    assert hp["noise"] == pytest.approx(0.05, rel=1e-4)


def test_update_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        GPSurrogate().posterior_update(np.zeros((1, 2)), np.zeros(1), np.zeros((3, 2)))
