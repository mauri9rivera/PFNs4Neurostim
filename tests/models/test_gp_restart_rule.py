"""GP-MLL restarts are ranked by the train-mode marginal likelihood, not the evaluation-mode loss (2026-10-27)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pfns4neurostim.models.gp.surrogates import GPSurrogate
from pfns4neurostim.models.registry import model_version

N, D = 30, 2


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.random((N, D))
    y = np.sin(4 * X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.standard_normal(N)
    return X, y


def _nlml_per_datum(X: np.ndarray, y: np.ndarray, hp: dict) -> float:
    """Closed-form negative log marginal likelihood per datum of the fitted RBF GP."""
    a = X / hp["lengthscale"]
    sq = ((a[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    K = hp["outputscale"] * np.exp(-0.5 * sq) + hp["noise"] * np.eye(len(y))
    r = y - hp["mean"]
    _, logdet = np.linalg.slogdet(K)
    return float((0.5 * r @ np.linalg.solve(K, r) + 0.5 * logdet + 0.5 * len(y) * math.log(2 * math.pi)) / len(y))


class _Recorder(GPSurrogate):
    """Records the ranking value of every feasible start."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.losses: list[float] = []

    def _fit_candidate(self, *a, **kw):
        out = super()._fit_candidate(*a, **kw)
        self.losses.append(out[0])
        return out


@pytest.mark.parametrize("seed", [0, 1])
def test_reported_loss_is_the_marginal_likelihood_of_the_fitted_hyperparameters(seed: int) -> None:
    X, y = _data(seed)
    gp = GPSurrogate(optimizer="lbfgs", dtype="float64", n_opt_steps=60, lr=1.0, n_restarts=1, noise_floor=1e-3)
    gp.fit(X, y)
    assert gp.fit_diagnostics()["gp_mll_final"] == pytest.approx(_nlml_per_datum(X, y, gp.hyperparameters()), abs=1e-6)


def test_selected_restart_has_the_lowest_train_mode_nlml() -> None:
    X, y = _data(2)
    gp = _Recorder(optimizer="lbfgs", dtype="float64", n_opt_steps=60, lr=1.0, n_restarts=8, noise_floor=1e-3, restart_seed=3)
    gp.fit(X, y)
    assert len(gp.losses) >= 2
    assert gp.fit_diagnostics()["gp_mll_final"] == pytest.approx(min(gp.losses), abs=1e-12)
    assert gp.fit_diagnostics()["gp_mll_final"] == pytest.approx(_nlml_per_datum(X, y, gp.hyperparameters()), abs=1e-6)


def test_more_restarts_never_worsen_the_marginal_likelihood() -> None:
    X, y = _data(4)
    best = []
    for k in (1, 3, 8):
        gp = GPSurrogate(optimizer="lbfgs", dtype="float64", n_opt_steps=60, lr=1.0, n_restarts=k, noise_floor=1e-3, restart_seed=5)
        gp.fit(X, y)
        best.append(gp.fit_diagnostics()["gp_mll_final"])
    assert best[1] <= best[0] + 1e-9 and best[2] <= best[1] + 1e-9


def test_gp_mll_version_string_names_the_restart_rule() -> None:
    assert "train-mode NLML" in model_version("gp_mll")


def test_every_gp_mll_belief_identity_carries_the_restart_rule() -> None:
    """The representation (BETA) cells use GP-MLL refit beliefs: a cell fitted under the old rule must never be served."""
    from pfns4neurostim.experiments import mechanism as M

    p = {"gp_params": {"n_restarts": 5}, "fixed_gp_params": {"lengthscale": 1.0}}
    assert M._representation_gp_params("gp_mll_refit", p)["restart_rule"] == M.GP_RESTART_RULE
    assert "restart_rule" not in M._representation_gp_params("gp_fixed", p)
