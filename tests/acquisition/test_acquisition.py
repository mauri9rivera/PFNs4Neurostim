"""Unit tests for the acquisition registry, schemas, closed forms and schedules.

Covers task #1 Step 4 (schema validation, unknown-param error, schedule values,
EI/PI closed forms) and task #4 Step 6 (``ts_marginal`` for both families,
``ts_joint`` GP-only with a clear error for PFNs).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from pfns4neurostim.acquisition import build_acquisition, masked_argmax
from pfns4neurostim.acquisition.base import BOState, acquire
from pfns4neurostim.acquisition.registry import ACQUISITION_REGISTRY, available_acquisitions
from pfns4neurostim.acquisition.schedules import Schedule, build_schedule


class _FakeSurrogate:
    """A surrogate with a fixed, known posterior, for exact closed-form checks.

    Args:
        mean: Predictive means, shape [N].
        std: Predictive standard deviations, shape [N].
        family: ``'gp'`` (joint available) or ``'pfn'`` (marginals only).
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, family: str = "gp") -> None:
        self._mean = np.asarray(mean, dtype=np.float64)
        self._std = np.asarray(std, dtype=np.float64)
        self.family = family
        self.key = f"fake_{family}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No-op: the posterior is fixed."""

    def predict_marginals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the fixed posterior."""
        return self._mean, self._std

    @property
    def supports_joint(self) -> bool:
        """Only the GP-like fake has a joint posterior."""
        return self.family == "gp"

    def sample_marginal(self, X: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray:
        """Independent Gaussian draws from the fixed marginals."""
        return rng.normal(self._mean, np.sqrt(temperature) * self._std)

    def sample_joint(self, X: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> np.ndarray:
        """A 'joint' draw; raises for the PFN-like fake, as the real adapter does."""
        if not self.supports_joint:
            raise NotImplementedError("ts_joint requires a joint posterior")
        return rng.normal(self._mean, np.sqrt(temperature) * self._std)


def _state(observed: tuple[int, ...] = (0,), step: int = 0, n_steps: int = 10) -> BOState:
    """Build a loop state with the given observations."""
    return BOState(observed, tuple(float(i) for i in observed), step, n_steps, n_dims=2)


class TestRegistrySchema:
    """P0.2: the registry is the single definition of what a type accepts."""

    def test_expected_types_are_registered(self) -> None:
        assert set(available_acquisitions()) == {
            "ei", "pi", "ucb", "ts_marginal", "ts_joint", "greedy", "random"
        }

    def test_unknown_type_raises_with_options(self) -> None:
        with pytest.raises(KeyError, match="ts_marginal"):
            build_acquisition("not_a_thing")

    def test_unknown_param_raises(self) -> None:
        with pytest.raises(ValueError, match="does not declare"):
            build_acquisition("ei", {"kappa": 2.0})

    def test_unknown_schedule_key_raises(self) -> None:
        with pytest.raises(ValueError, match="schedules has key"):
            build_acquisition("ei", {"xi": 0.0}, {"kappa": {"kind": "linear", "start": 1, "end": 0}})

    def test_greedy_and_random_take_no_params(self) -> None:
        for name in ("greedy", "random"):
            build_acquisition(name)
            with pytest.raises(ValueError, match="does not declare"):
                build_acquisition(name, {"xi": 1.0})

    def test_resolved_params_are_logged_per_step(self) -> None:
        _, params = build_acquisition("ucb", {"kappa": 2.0})
        assert params.resolved(_state()) == {"kappa": 2.0}

    def test_ts_joint_declares_it_needs_a_joint_posterior(self) -> None:
        assert ACQUISITION_REGISTRY["ts_joint"].needs_joint is True
        assert ACQUISITION_REGISTRY["ts_marginal"].needs_joint is False


class TestClosedForms:
    """EI and PI must match their textbook expressions exactly."""

    def test_ei_matches_the_analytic_expression(self) -> None:
        mean = np.array([0.0, 1.0, 2.0, 0.5])
        std = np.array([1.0, 0.5, 2.0, 0.1])
        spec, params = build_acquisition("ei", {"xi": 0.0})
        got = spec.score_fn(_FakeSurrogate(mean, std), np.zeros((4, 2)), _state((0,)), None, params)

        incumbent = mean[0]           # best predicted value among observed sites
        gap = mean - incumbent
        z = gap / std
        want = gap * stats.norm.cdf(z) + std * stats.norm.pdf(z)
        np.testing.assert_allclose(got, want, rtol=1e-10)

    def test_ei_is_non_negative(self) -> None:
        rng = np.random.default_rng(0)
        mean, std = rng.normal(size=50), rng.uniform(0.1, 2.0, size=50)
        spec, params = build_acquisition("ei")
        got = spec.score_fn(_FakeSurrogate(mean, std), np.zeros((50, 2)), _state((3,)), None, params)
        assert (got >= -1e-12).all()

    def test_ei_xi_reduces_the_score(self) -> None:
        mean, std = np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0])
        spec, p0 = build_acquisition("ei", {"xi": 0.0})
        _, p1 = build_acquisition("ei", {"xi": 0.5})
        args = (_FakeSurrogate(mean, std), np.zeros((3, 2)), _state((0,)), None)
        assert (spec.score_fn(*args, p1) <= spec.score_fn(*args, p0) + 1e-12).all()

    def test_pi_matches_the_analytic_expression(self) -> None:
        mean = np.array([0.0, 1.0, 2.0])
        std = np.array([1.0, 0.5, 2.0])
        spec, params = build_acquisition("pi", {"xi": 0.0})
        got = spec.score_fn(_FakeSurrogate(mean, std), np.zeros((3, 2)), _state((0,)), None, params)
        np.testing.assert_allclose(got, stats.norm.cdf((mean - mean[0]) / std), rtol=1e-10)

    def test_ucb_is_mean_plus_kappa_std(self) -> None:
        mean, std = np.array([0.0, 1.0]), np.array([2.0, 0.5])
        spec, params = build_acquisition("ucb", {"kappa": 3.0})
        got = spec.score_fn(_FakeSurrogate(mean, std), np.zeros((2, 2)), _state(), None, params)
        np.testing.assert_allclose(got, mean + 3.0 * std)

    def test_greedy_is_the_mean(self) -> None:
        mean, std = np.array([0.0, 1.0]), np.array([5.0, 0.1])
        spec, params = build_acquisition("greedy")
        got = spec.score_fn(_FakeSurrogate(mean, std), np.zeros((2, 2)), _state(), None, params)
        np.testing.assert_allclose(got, mean)

    def test_zero_variance_ei_is_the_raw_improvement(self) -> None:
        mean, std = np.array([0.0, 1.0, -1.0]), np.zeros(3)
        spec, params = build_acquisition("ei")
        got = spec.score_fn(_FakeSurrogate(mean, std), np.zeros((3, 2)), _state((0,)), None, params)
        np.testing.assert_allclose(got, [0.0, 1.0, 0.0])


class TestThompson:
    """Task #4 Step 6: marginal for both families, joint for the GP only."""

    def test_ts_marginal_works_for_both_families(self) -> None:
        mean, std = np.zeros(10), np.ones(10)
        spec, params = build_acquisition("ts_marginal", {"temperature": 1.0})
        for family in ("gp", "pfn"):
            got = spec.score_fn(
                _FakeSurrogate(mean, std, family), np.zeros((10, 2)), _state(),
                np.random.default_rng(0), params,
            )
            assert got.shape == (10,) and np.isfinite(got).all()

    def test_ts_joint_raises_for_a_pfn(self) -> None:
        spec, params = build_acquisition("ts_joint")
        with pytest.raises(NotImplementedError, match="joint posterior"):
            spec.score_fn(
                _FakeSurrogate(np.zeros(5), np.ones(5), "pfn"), np.zeros((5, 2)),
                _state(), np.random.default_rng(0), params,
            )

    def test_temperature_scales_the_spread(self) -> None:
        mean, std = np.zeros(4000), np.ones(4000)
        spec, hot = build_acquisition("ts_marginal", {"temperature": 4.0})
        _, cold = build_acquisition("ts_marginal", {"temperature": 1.0})
        args = (_FakeSurrogate(mean, std), np.zeros((4000, 2)), _state())
        s_hot = spec.score_fn(*args, np.random.default_rng(0), hot)
        s_cold = spec.score_fn(*args, np.random.default_rng(0), cold)
        assert s_hot.std() == pytest.approx(2.0 * s_cold.std(), rel=0.1)

    def test_sampling_is_deterministic_under_a_seed(self) -> None:
        spec, params = build_acquisition("ts_marginal")
        args = (_FakeSurrogate(np.zeros(20), np.ones(20)), np.zeros((20, 2)), _state())
        a = spec.score_fn(*args, np.random.default_rng(7), params)
        b = spec.score_fn(*args, np.random.default_rng(7), params)
        np.testing.assert_allclose(a, b)


class TestSelection:
    """Masked, randomly tie-broken argmax (defect D6)."""

    def test_observed_sites_are_never_reselected(self) -> None:
        values = np.array([10.0, 9.0, 8.0])
        assert masked_argmax(values, (0,), np.random.default_rng(0)) == 1

    def test_ties_are_broken_at_random_not_by_index(self) -> None:
        values = np.ones(6)
        picks = {masked_argmax(values, (), np.random.default_rng(s)) for s in range(40)}
        assert len(picks) > 1, "tie-break collapsed to a single index (D6 regression)"

    def test_non_finite_surface_raises(self) -> None:
        with pytest.raises(RuntimeError, match="non-finite"):
            masked_argmax(np.array([np.nan, np.inf * -1, np.nan]), (), np.random.default_rng(0))

    def test_exhausted_pool_raises(self) -> None:
        with pytest.raises(RuntimeError, match="already been queried"):
            masked_argmax(np.zeros(2), (0, 1), np.random.default_rng(0))

    def test_acquire_returns_index_values_and_params(self) -> None:
        spec, params = build_acquisition("ucb", {"kappa": 1.0})
        res = acquire(
            spec.score_fn, _FakeSurrogate(np.array([0.0, 5.0]), np.ones(2)),
            np.zeros((2, 2)), _state(()), np.random.default_rng(0), params,
        )
        assert res.index == 1
        assert res.values.shape == (2,)
        assert res.params == {"kappa": 1.0}


class TestSchedules:
    """Schedule kinds and their endpoints."""

    def test_constant_holds(self) -> None:
        sch = Schedule("constant", start=3.0)
        assert [sch.value(t, 10) for t in (0, 5, 9)] == [3.0, 3.0, 3.0]

    def test_linear_hits_both_endpoints(self) -> None:
        sch = Schedule("linear", start=10.0, end=0.0)
        assert sch.value(0, 11) == pytest.approx(10.0)
        assert sch.value(10, 11) == pytest.approx(0.0)
        assert sch.value(5, 11) == pytest.approx(5.0)

    def test_cosine_is_monotone_between_endpoints(self) -> None:
        sch = Schedule("cosine", start=7.5, end=0.6)
        values = [sch.value(t, 21) for t in range(21)]
        assert values[0] == pytest.approx(7.5)
        assert values[-1] == pytest.approx(0.6)
        assert all(b <= a + 1e-12 for a, b in zip(values, values[1:]))

    def test_auto_dim_grows_with_dimensionality(self) -> None:
        sch = Schedule("auto_dim")
        assert sch.value(0, 20, n_dims=5) > sch.value(0, 20, n_dims=2)

    def test_auto_dim_respects_its_floor(self) -> None:
        sch = Schedule("auto_dim", floor=3.0)
        assert sch.value(0, 20, n_dims=1) >= 3.0

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schedule kind"):
            Schedule("exponential")

    def test_build_schedule_rejects_unknown_arguments(self) -> None:
        with pytest.raises(ValueError, match="unknown key"):
            build_schedule({"kind": "linear", "start": 1.0, "finish": 0.0})

    def test_schedule_drives_the_acquisition_parameter(self) -> None:
        _, params = build_acquisition(
            "ucb", {"kappa": 2.0}, {"kappa": {"kind": "linear", "start": 8.0, "end": 0.0}}
        )
        assert params.value("kappa", _state(step=0, n_steps=9)) == pytest.approx(8.0)
        assert params.value("kappa", _state(step=8, n_steps=9)) == pytest.approx(0.0)
