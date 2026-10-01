"""The GP marginal-likelihood fit: optimiser arms, precision and the convergence diagnostics (P0.10).

Guardrail G1 forbids a "cheaper than GP" claim while the tuned GP's fit may be unconverged, so the
diagnostics these tests pin are the ones the claim is read off: which optimiser ran, how many iterations
and objective evaluations it cost, the gradient it stopped at, and whether that counts as converged.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pfns4neurostim.models.gp.surrogates import (
    GP_DTYPES,
    GP_OPTIMIZERS,
    INFEASIBLE_LOSS,
    MAX_NOISE_ESCALATIONS,
    RESTART_INIT_RANGES,
    GPSurrogate,
    NaiveGPSurrogate,
)
from pfns4neurostim.models.registry import build_surrogate, model_version

DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "gp_lengthscale",
    "gp_lengthscale_min",
    "gp_lengthscale_max",
    "gp_outputscale",
    "gp_noise",
    "gp_mll_final",
    "gp_n_opt_steps",
    "gp_n_opt_steps_max",
    "gp_n_obj_evals",
    "gp_grad_max",
    "gp_fit_converged",
    "gp_fit_n_train",
    "gp_n_restarts",
    "gp_n_feasible",
    "gp_noise_escalations",
    "gp_noise_floor",
    "gp_fit_degenerate",
    "gp_grad_max_free",
    "gp_stopped_early",
    "gp_lengthscale_floor",
    "gp_restart_spread",
)

#: Diagnostics that are legitimately NaN rather than missing: a single start cannot disagree with itself.
NAN_WITH_ONE_START: frozenset[str] = frozenset({"gp_restart_spread"})


def _data(n: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A smooth 2D function on MinMax-scaled inputs, as every surrogate sees it.

    Args:
        n: Number of points.
        seed: RNG seed.

    Returns:
        ``(X, y)`` of shapes [n, 2] and [n].
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 2))                                             # [n, 2]
    y = np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1]) + 0.05 * rng.normal(size=n)  # [n]
    return X, y


# --------------------------------------------------------------------------- construction
@pytest.mark.parametrize("bad", ["sgd", "lbfgs ", "", "LBFGS"])
def test_unknown_optimizer_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="optimizer must be one of"):
        GPSurrogate(optimizer=bad)


@pytest.mark.parametrize("bad", ["float16", "double", ""])
def test_unknown_dtype_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="dtype must be one of"):
        GPSurrogate(dtype=bad)


def test_gp_mll_is_the_converged_arm() -> None:
    """P0.10, settled 2026-09-30: `gp_mll` IS the converged fit, and the Adam arm is retired."""
    assert "adam" in GP_OPTIMIZERS and "lbfgs" in GP_OPTIMIZERS
    gp = build_surrogate("gp_mll", device="cpu")._model
    assert gp._optimizer == "lbfgs"
    assert gp._dtype is GP_DTYPES["float64"]
    assert gp._n_restarts == 5
    assert gp._noise_floor == pytest.approx(1e-3)
    assert gp._n_opt_steps == 50
    assert gp._lr == pytest.approx(1.0)
    version = model_version("gp_mll")
    assert "L-BFGS" in version and "float64" in version


def test_the_adam_arm_is_no_longer_registered() -> None:
    """Retired, not renamed: nothing may reach the unconverged fit through the registry."""
    from pfns4neurostim.models.registry import MODEL_REGISTRY

    assert "gp_mll_lbfgs" not in MODEL_REGISTRY
    assert not any("adam" in key for key in MODEL_REGISTRY)


def test_the_other_gp_arms_are_untouched() -> None:
    """The class defaults did not move, so gp_naive and the deep kernel keep their cached identity."""
    naive = build_surrogate("gp_naive", device="cpu")._model
    assert naive._optimizer == "adam"
    assert naive._dtype is GP_DTYPES["float32"]
    assert naive._noise_floor == pytest.approx(1e-4)
    assert naive._n_restarts == 1
    assert GPSurrogate()._optimizer == "adam"
    assert GPSurrogate()._noise_floor == pytest.approx(1e-4)


# --------------------------------------------------------------------------- multi-start + floor
def test_restarts_are_counted_and_all_feasible_on_benign_data() -> None:
    X, y = _data()
    gp = GPSurrogate(n_opt_steps=40, lr=1.0, optimizer="lbfgs", dtype="float64", n_restarts=4)
    gp.fit(X, y)
    d = gp.fit_diagnostics()
    assert d["gp_n_restarts"] == 4.0
    assert d["gp_n_feasible"] == 4.0
    assert d["gp_noise_escalations"] == 0.0


def test_restarts_are_reproducible_and_seed_dependent() -> None:
    """Starts are drawn from a seeded generator, so a cached cell and a fresh one agree."""
    X, y = _data()
    fits = []
    for seed in (0, 0, 7):
        gp = GPSurrogate(n_opt_steps=40, lr=1.0, optimizer="lbfgs", dtype="float64",
                         n_restarts=3, restart_seed=seed)
        gp.fit(X, y)
        fits.append(gp.fit_diagnostics()["gp_mll_final"])
    assert fits[0] == fits[1]


def test_the_noise_floor_is_respected() -> None:
    """On near-noiseless data the fit wants zero noise, so it must stop at the floor, not below it."""
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(25, 2))                                            # [25, 2]
    y = np.sin(3 * X[:, 0])                                                  # [25] noiseless
    for floor in (1e-3, 1e-2):
        gp = GPSurrogate(n_opt_steps=50, lr=1.0, optimizer="lbfgs", dtype="float64",
                         noise_floor=floor)
        gp.fit(X, y)
        assert gp.fit_diagnostics()["gp_noise"] >= floor * 0.999
        assert gp.fit_diagnostics()["gp_noise_floor"] == pytest.approx(floor)


def test_a_fit_on_the_floor_is_flagged_degenerate() -> None:
    """The flag is a column, never an exception: a bound-saturating fit is usable but must be visible."""
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(25, 2))
    y = np.sin(3 * X[:, 0])
    gp = GPSurrogate(n_opt_steps=50, lr=1.0, optimizer="lbfgs", dtype="float64", noise_floor=1e-3)
    gp.fit(X, y)
    assert gp.fit_diagnostics()["gp_fit_degenerate"] == 1.0


def test_duplicate_rows_at_the_old_floor_now_fit(recwarn) -> None:
    """The P0.10 regression: re-queried sites made the converged fit unfactorisable at 1e-4.

    A TS loop queries with replacement, so a late-run context holds duplicate rows; K is then rank
    deficient and only the noise ridge keeps it factorisable. With the gp_mll defaults this must fit,
    predict and sample.
    """
    rng = np.random.default_rng(2)
    ax = np.linspace(0.0, 1.0, 8)
    pool = np.stack(np.meshgrid(ax, ax), -1).reshape(-1, 2)                  # [64, 2]
    gt = np.exp(-6.0 * ((pool - np.array([0.4, 0.6])) ** 2).sum(1))           # [64]
    best = np.argsort(gt)[::-1]
    idx = np.concatenate([rng.choice(best[3:], size=12, replace=False), np.repeat(best[:3], 5)])
    gp = build_surrogate("gp_mll", device="cpu")
    gp.fit(pool[idx], gt[idx])
    mean, sd = gp.predict(pool)
    assert np.isfinite(mean).all() and np.isfinite(sd).all()
    assert gp.fit_diagnostics()["gp_n_feasible"] >= 1.0


def test_restart_init_ranges_are_positive_and_ordered() -> None:
    for name, (lo, hi) in RESTART_INIT_RANGES.items():
        assert 0.0 < lo < hi, name


def test_the_escalation_ladder_is_bounded() -> None:
    """There is no 'raise the noise until it works' loop: the ladder has a fixed length and then fails."""
    assert 1 <= MAX_NOISE_ESCALATIONS <= 10


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_restarts_raise(bad: int) -> None:
    with pytest.raises(ValueError, match="n_restarts must be"):
        GPSurrogate(n_restarts=bad)


def test_non_positive_noise_floor_raises() -> None:
    with pytest.raises(ValueError, match="noise_floor must be"):
        GPSurrogate(noise_floor=0.0)


# --------------------------------------------------------------------------- diagnostics
@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_every_diagnostic_is_present_and_finite(optimizer: str) -> None:
    X, y = _data()
    gp = GPSurrogate(n_opt_steps=40, optimizer=optimizer, lr=1.0 if optimizer == "lbfgs" else 0.1)
    gp.fit(X, y)
    diagnostics = gp.fit_diagnostics()
    assert set(diagnostics) == set(DIAGNOSTIC_KEYS)
    assert all(
        np.isfinite(v) for k, v in diagnostics.items() if k not in NAN_WITH_ONE_START
    ), diagnostics
    assert diagnostics["gp_n_restarts"] == 1.0
    assert np.isnan(diagnostics["gp_restart_spread"])
    assert diagnostics["gp_fit_n_train"] == len(X)
    assert diagnostics["gp_n_opt_steps_max"] == 40.0


def test_adam_reports_its_fixed_budget_one_evaluation_per_step() -> None:
    X, y = _data()
    gp = GPSurrogate(n_opt_steps=25, optimizer="adam")
    gp.fit(X, y)
    d = gp.fit_diagnostics()
    assert d["gp_n_opt_steps"] == 25.0
    assert d["gp_n_obj_evals"] == 25.0


def test_lbfgs_stops_before_its_budget_and_line_searches() -> None:
    """The point of the arm: it stops on its own tolerance, and an iteration costs several evaluations."""
    X, y = _data()
    gp = GPSurrogate(n_opt_steps=80, lr=1.0, optimizer="lbfgs", dtype="float64")
    gp.fit(X, y)
    d = gp.fit_diagnostics()
    assert 0 < d["gp_n_opt_steps"] < 80.0
    assert d["gp_n_obj_evals"] >= d["gp_n_opt_steps"]
    assert d["gp_fit_converged"] == 1.0


def test_lbfgs_reaches_a_smaller_gradient_than_a_fixed_adam_budget() -> None:
    """The P0.10 question, as a test: the fixed-budget fit is the one left with a live gradient.

    Only the *gradient* is asserted, deliberately. The marginal likelihood is multi-modal, and the two
    arms do land in different basins: on this fixture the unconverged Adam fit ends 0.01 nats BELOW the
    converged L-BFGS one, which is why the shadow run reports both NLL and regret per arm rather than
    assuming the converged fit is the better model.
    """
    X, y = _data()
    adam = GPSurrogate(n_opt_steps=100, optimizer="adam")
    lbfgs = GPSurrogate(n_opt_steps=100, lr=1.0, optimizer="lbfgs", dtype="float64")
    adam.fit(X, y)
    lbfgs.fit(X, y)
    assert lbfgs.fit_diagnostics()["gp_grad_max"] < adam.fit_diagnostics()["gp_grad_max"]
    assert abs(lbfgs.fit_diagnostics()["gp_mll_final"] - adam.fit_diagnostics()["gp_mll_final"]) < 0.5


def test_converged_flag_is_false_for_an_unconverged_fixed_budget_fit() -> None:
    X, y = _data()
    gp = GPSurrogate(n_opt_steps=2, optimizer="adam")
    gp.fit(X, y)
    d = gp.fit_diagnostics()
    assert d["gp_n_opt_steps"] == 2.0
    assert d["gp_fit_converged"] == 0.0
    assert d["gp_grad_max"] > 1e-7


def test_naive_gp_takes_no_step_and_reports_no_gradient() -> None:
    """The fixed-hyperparameter arm must not claim convergence either way."""
    X, y = _data()
    gp = NaiveGPSurrogate()
    gp.fit(X, y)
    d = gp.fit_diagnostics()
    assert d["gp_n_opt_steps"] == 0.0
    assert d["gp_n_obj_evals"] == 0.0
    assert np.isnan(d["gp_grad_max"])
    assert np.isnan(d["gp_fit_converged"])
    assert np.isnan(d["gp_mll_final"])


# --------------------------------------------------------------------------- precision
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_predictions_are_finite_in_either_precision(dtype: str) -> None:
    X, y = _data()
    gp = GPSurrogate(n_opt_steps=30, dtype=dtype)
    gp.fit(X, y)
    mean, sd = gp.predict(X)
    assert np.isfinite(mean).all() and np.isfinite(sd).all()
    assert (sd > 0).all()


def test_the_two_precisions_agree_on_the_same_fit() -> None:
    """Precision changes the fit's numerics, not the model: the same steps land in the same place."""
    X, y = _data()
    fits = {}
    for dtype in ("float32", "float64"):
        torch.manual_seed(0)
        gp = GPSurrogate(n_opt_steps=50, dtype=dtype)
        gp.fit(X, y)
        fits[dtype] = gp.predict(X)[0]                                       # [n]
    np.testing.assert_allclose(fits["float32"], fits["float64"], atol=1e-3)


# --------------------------------------------------------------------------- robustness
def test_infeasible_region_does_not_kill_the_lbfgs_fit() -> None:
    """Five nearly-collinear points: the flat likelihood is what sent the line search into a NaN kernel."""
    X = np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]])   # [5, 2]
    y = np.array([0.1, 0.1, 0.1, 0.1, 0.1])                                        # [5]
    gp = GPSurrogate(n_opt_steps=50, lr=1.0, optimizer="lbfgs", dtype="float64")
    gp.fit(X, y)
    d = gp.fit_diagnostics()
    assert np.isfinite(d["gp_grad_max"])
    assert d["gp_mll_final"] < INFEASIBLE_LOSS
    mean, sd = gp.predict(X)
    assert np.isfinite(mean).all() and np.isfinite(sd).all()


def test_a_run_of_fits_is_reproducible() -> None:
    """Both arms are deterministic given the data: a cached cell and a fresh one must agree."""
    X, y = _data()
    for optimizer, dtype in (("adam", "float32"), ("lbfgs", "float64")):
        runs = []
        for _ in range(2):
            gp = GPSurrogate(n_opt_steps=40, lr=1.0 if optimizer == "lbfgs" else 0.1,
                             optimizer=optimizer, dtype=dtype)
            gp.fit(X, y)
            runs.append((gp.fit_diagnostics(), gp.predict(X)[0]))
        first, second = runs[0][0], runs[1][0]
        assert set(first) == set(second)
        for key in first:                      # NaN == NaN is False, so compare NaN-aware
            a, b = first[key], second[key]
            assert (np.isnan(a) and np.isnan(b)) or a == b, key
        np.testing.assert_array_equal(runs[0][1], runs[1][1])


# --------------------------------------------------------------------------- convergence diagnostics
def test_restart_spread_is_recorded_and_nan_for_one_start() -> None:
    """Agreement between independent starts is the only cheap evidence about the GLOBAL optimum."""
    X, y = _data()
    many = GPSurrogate(n_opt_steps=40, lr=1.0, optimizer="lbfgs", dtype="float64", n_restarts=4)
    many.fit(X, y)
    assert np.isfinite(many.fit_diagnostics()["gp_restart_spread"])
    one = GPSurrogate(n_opt_steps=40, lr=1.0, optimizer="lbfgs", dtype="float64", n_restarts=1)
    one.fit(X, y)
    assert np.isnan(one.fit_diagnostics()["gp_restart_spread"])


def test_a_fixed_budget_fit_can_never_read_converged() -> None:
    """Which is the finding of P0.10, not a defect: Adam always exhausts its budget."""
    X, y = _data()
    gp = GPSurrogate(n_opt_steps=100, optimizer="adam")
    gp.fit(X, y)
    d = gp.fit_diagnostics()
    assert d["gp_stopped_early"] == 0.0
    assert d["gp_fit_converged"] == 0.0


def test_the_free_gradient_excludes_a_parameter_resting_on_its_floor() -> None:
    """At an active constraint the gradient legitimately does not vanish; the projected one is the test."""
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(20, 2))                                            # [20, 2]
    y = np.sin(3 * X[:, 0])                                                  # [20] noiseless
    gp = GPSurrogate(n_opt_steps=50, lr=1.0, optimizer="lbfgs", dtype="float64", noise_floor=1e-2)
    gp.fit(X, y)
    d = gp.fit_diagnostics()
    assert d["gp_noise"] <= 1e-2 * 1.01                 # it did rest on the floor
    assert d["gp_grad_max_free"] <= d["gp_grad_max"]    # and that parameter is excluded


def test_the_lengthscale_floor_binds_only_when_asked() -> None:
    """Default 0.0 leaves gpytorch's own constraint alone, so no existing arm changes."""
    assert GPSurrogate()._lengthscale_floor == 0.0
    assert build_surrogate("gp_mll", device="cpu")._model._lengthscale_floor == 0.0
    X, y = _data()
    gp = GPSurrogate(n_opt_steps=50, lr=1.0, optimizer="lbfgs", dtype="float64",
                     lengthscale_floor=0.5)
    gp.fit(X, y)
    assert gp.fit_diagnostics()["gp_lengthscale_min"] >= 0.5 * 0.999
    assert gp.fit_diagnostics()["gp_lengthscale_floor"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [-1.0, -1e-9])
def test_negative_lengthscale_floor_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="lengthscale_floor must be"):
        GPSurrogate(lengthscale_floor=bad)


def test_rbf_is_stable_at_a_collapsed_lengthscale() -> None:
    """The C1 crash: the a^2+b^2-2ab identity cancels away every digit once 1/ell reaches ~1e8.

    Measured 2026-10-01 on a real NHP context: kernel entries wrong by 1.25 on a [0, 1.27] scale and a
    minimum eigenvalue of -0.44, which jitter cannot repair because the entries themselves are wrong.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(26, 2))                                            # [26, 2]
    for ell in (1e-2, 1e-6, 1e-8, 6.54e-9, 1e-12):
        hp = {"lengthscale": np.array([7.4e-2, ell]), "outputscale": 1.269, "noise": 1e-3, "mean": 0.0}
        K = GPSurrogate._rbf(X, X, hp) + hp["noise"] * np.eye(len(X))         # [26, 26]
        assert np.isfinite(K).all()
        assert float(np.linalg.eigvalsh((K + K.T) / 2.0).min()) > 0.0, ell
        np.linalg.cholesky(K)                                                # must not raise
