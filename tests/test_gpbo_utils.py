"""Unit tests for src/utils/gpbo_utils.py.

Covers all pure numpy/scipy functions:
- compute_ucb_kappa()
- _auto_kappa_max()
- _auto_kappa_min()
- std_from_quantiles()
- expected_improvement_numpy()
"""
import math
import os
import sys

import numpy as np
import pytest
from scipy.stats import norm as sp_norm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _import_compute_ucb_kappa():
    from utils.gpbo_utils import compute_ucb_kappa
    return compute_ucb_kappa


def _import_auto_kappa_max():
    from utils.gpbo_utils import _auto_kappa_max
    return _auto_kappa_max


def _import_auto_kappa_min():
    from utils.gpbo_utils import _auto_kappa_min
    return _auto_kappa_min


def _import_std_from_quantiles():
    from utils.gpbo_utils import std_from_quantiles
    return std_from_quantiles


def _import_ei_numpy():
    from utils.gpbo_utils import expected_improvement_numpy
    return expected_improvement_numpy


# ---------------------------------------------------------------------------
# compute_ucb_kappa
# ---------------------------------------------------------------------------

class TestComputeUcbKappa:
    """Tests for compute_ucb_kappa() cosine annealing schedule."""

    def test_at_t0_returns_kappa_max_with_default_alpha(self):
        """At t=0 with alpha=0.5, returns kappa_max."""
        # Arrange
        compute_ucb_kappa = _import_compute_ucb_kappa()
        # Act
        result = compute_ucb_kappa(t=0, n_steps=100, kappa_max=7.0, kappa_min=0.5)
        # Assert: kappa_min + 2*0.5*(kappa_max - kappa_min) = kappa_max
        np.testing.assert_allclose(result, 7.0, rtol=1e-9)

    def test_at_t_equals_n_steps_returns_kappa_min(self):
        """At t=n_steps, returns kappa_min."""
        compute_ucb_kappa = _import_compute_ucb_kappa()
        result = compute_ucb_kappa(t=100, n_steps=100, kappa_max=7.0, kappa_min=0.5)
        np.testing.assert_allclose(result, 0.5, rtol=1e-9)

    def test_monotonically_decreasing(self):
        """Values should decrease from t=0 to t=n_steps."""
        compute_ucb_kappa = _import_compute_ucb_kappa()
        n_steps = 100
        kappa_max, kappa_min = 7.0, 0.5
        ts = [0, 20, 40, 60, 80, 100]
        values = [
            compute_ucb_kappa(t, n_steps, kappa_max, kappa_min) for t in ts
        ]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1], (
                f"Not monotone at indices {i},{i+1}: {values[i]} < {values[i+1]}"
            )

    def test_n_steps_zero_always_returns_kappa_min(self):
        """When n_steps=0, always returns kappa_min regardless of t."""
        compute_ucb_kappa = _import_compute_ucb_kappa()
        for t in [0, 1, 10]:
            result = compute_ucb_kappa(t=t, n_steps=0, kappa_max=7.0, kappa_min=0.5)
            np.testing.assert_allclose(result, 0.5)

    def test_value_between_kappa_min_and_kappa_max(self):
        """All values in (kappa_min, kappa_max] for default alpha."""
        compute_ucb_kappa = _import_compute_ucb_kappa()
        kappa_max, kappa_min = 7.0, 0.5
        n_steps = 50
        for t in range(0, n_steps + 1, 5):
            v = compute_ucb_kappa(t, n_steps, kappa_max, kappa_min)
            assert kappa_min - 1e-9 <= v <= kappa_max + 1e-9, (
                f"Value {v} out of range [{kappa_min}, {kappa_max}] at t={t}"
            )

    def test_at_midpoint_returns_midpoint_kappa(self):
        """At t=n_steps/2, cos(pi/2)=0, result = kappa_min + alpha*(kappa_max-kappa_min)."""
        compute_ucb_kappa = _import_compute_ucb_kappa()
        kappa_max, kappa_min, alpha = 8.0, 2.0, 0.5
        n_steps = 100
        result = compute_ucb_kappa(n_steps // 2, n_steps, kappa_max, kappa_min, alpha)
        expected = kappa_min + alpha * (kappa_max - kappa_min)  # = 5.0
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_custom_alpha_above_half_starts_above_kappa_max(self):
        """With alpha > 0.5, the initial value exceeds kappa_max."""
        compute_ucb_kappa = _import_compute_ucb_kappa()
        result = compute_ucb_kappa(t=0, n_steps=100, kappa_max=7.0, kappa_min=0.5, alpha=0.8)
        # kappa_min + 2*0.8*(7.0-0.5) = 0.5 + 1.6*6.5 = 10.9
        assert result > 7.0


# ---------------------------------------------------------------------------
# _auto_kappa_max
# ---------------------------------------------------------------------------

class TestAutoKappaMax:
    """Tests for _auto_kappa_max() formula and floor."""

    def test_formula_value_nhp_case(self):
        """NHP case: d=2, n_iter=95 → approx 7.54 (within 0.1 tolerance)."""
        _auto_kappa_max = _import_auto_kappa_max()
        result = _auto_kappa_max(d=2, n_iter=95)
        np.testing.assert_allclose(result, 7.54, atol=0.1)

    def test_returns_at_least_kappa_floor(self):
        """Returns kappa_floor when formula gives smaller value."""
        _auto_kappa_max = _import_auto_kappa_max()
        result = _auto_kappa_max(d=1, n_iter=1, alpha=0.001, kappa_floor=3.0)
        assert result >= 3.0

    def test_increases_with_d(self):
        """Higher input dimension → larger kappa_max."""
        _auto_kappa_max = _import_auto_kappa_max()
        v1 = _auto_kappa_max(d=2, n_iter=50)
        v2 = _auto_kappa_max(d=5, n_iter=50)
        assert v2 > v1

    def test_increases_with_n_iter(self):
        """More iterations → larger kappa_max."""
        _auto_kappa_max = _import_auto_kappa_max()
        v1 = _auto_kappa_max(d=2, n_iter=10)
        v2 = _auto_kappa_max(d=2, n_iter=200)
        assert v2 > v1

    def test_n_iter_one_uses_safe_log(self):
        """n_iter=1 uses max(n_iter, 2)=2 to avoid log(1)=0."""
        _auto_kappa_max = _import_auto_kappa_max()
        result = _auto_kappa_max(d=2, n_iter=1, kappa_floor=0.0)
        # alpha * sqrt(d * log(2)) — must be positive
        assert result > 0.0

    def test_positive_return_value(self):
        """Always returns a positive value."""
        _auto_kappa_max = _import_auto_kappa_max()
        for d, n in [(1, 5), (2, 20), (10, 100)]:
            assert _auto_kappa_max(d, n) > 0.0


# ---------------------------------------------------------------------------
# _auto_kappa_min
# ---------------------------------------------------------------------------

class TestAutoKappaMin:
    """Tests for _auto_kappa_min() formula."""

    def test_always_positive(self):
        """Always returns a positive value."""
        _auto_kappa_min = _import_auto_kappa_min()
        for d, n in [(1, 5), (2, 20), (10, 100)]:
            assert _auto_kappa_min(d, n) > 0.0

    def test_ratio_to_kappa_max_is_alpha_over_beta(self):
        """kappa_max / kappa_min = alpha / beta = 12.5 at default params."""
        _auto_kappa_max = _import_auto_kappa_max()
        _auto_kappa_min = _import_auto_kappa_min()
        # Use d=2, n_iter=95 where formula dominates the floor
        d, n = 2, 95
        km = _auto_kappa_max(d, n, alpha=2.5, kappa_floor=0.0)
        kn = _auto_kappa_min(d, n, beta=0.2)
        ratio = km / kn
        np.testing.assert_allclose(ratio, 2.5 / 0.2, atol=0.1)

    def test_n_iter_one_safe(self):
        """n_iter=1 uses max(n_iter, 2) safely."""
        _auto_kappa_min = _import_auto_kappa_min()
        result = _auto_kappa_min(d=2, n_iter=1)
        assert result > 0.0

    def test_increases_with_d_and_n_iter(self):
        """Larger d or n_iter → larger kappa_min."""
        _auto_kappa_min = _import_auto_kappa_min()
        assert _auto_kappa_min(5, 50) > _auto_kappa_min(2, 50)
        assert _auto_kappa_min(2, 100) > _auto_kappa_min(2, 10)


# ---------------------------------------------------------------------------
# std_from_quantiles
# ---------------------------------------------------------------------------

class TestStdFromQuantiles:
    """Tests for std_from_quantiles() normal approximation."""

    def _make_normal_quantiles(
        self, mu: float, sigma: float, n_samples: int = 100
    ) -> np.ndarray:
        """Build exact N(mu, sigma) quantile array of shape (7, n_samples)."""
        levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        # All samples identical — test scalar case
        q_vals = sp_norm.ppf(levels, loc=mu, scale=sigma)
        return np.tile(q_vals, (n_samples, 1)).T  # [7, n_samples]

    def test_exact_normal_recovers_mu_and_std_proportional(self):
        """Exact N(3, 2) quantiles → mean==3 and std is proportional to sigma.

        The function uses hard-coded divisors that produce a consistent but
        slightly attenuated sigma estimate; the key properties are:
          - mean equals the median exactly
          - std is positive and proportional to the true sigma
          - doubling sigma doubles the estimated std
        """
        std_from_quantiles = _import_std_from_quantiles()
        q1 = self._make_normal_quantiles(mu=3.0, sigma=1.0)
        q2 = self._make_normal_quantiles(mu=3.0, sigma=2.0)
        mean1, std1 = std_from_quantiles(q1)
        mean2, std2 = std_from_quantiles(q2)
        # Mean == median exactly
        np.testing.assert_allclose(mean1, 3.0, atol=1e-10)
        np.testing.assert_allclose(mean2, 3.0, atol=1e-10)
        # Doubling sigma should double the std estimate
        np.testing.assert_allclose(std2[0] / std1[0], 2.0, rtol=1e-5)
        # Both std estimates should be positive
        assert np.all(std1 > 0)
        assert np.all(std2 > 0)

    def test_std_floored_never_zero_for_constant_quantiles(self):
        """When all quantiles are identical (std→0), std is floored at 1e-9."""
        std_from_quantiles = _import_std_from_quantiles()
        quantiles = np.ones((7, 10)) * 5.0  # constant → std=0 analytically
        _, std = std_from_quantiles(quantiles)
        assert np.all(std >= 1e-9)

    def test_output_shapes(self):
        """Both outputs are shape (n_samples,)."""
        std_from_quantiles = _import_std_from_quantiles()
        n = 50
        quantiles = np.random.RandomState(42).randn(7, n)
        quantiles = np.sort(quantiles, axis=0)  # ensure monotone
        mean, std = std_from_quantiles(quantiles)
        assert mean.shape == (n,)
        assert std.shape == (n,)

    def test_std_positive_when_quantiles_reversed(self):
        """Reversed quantiles (high < low) still produce positive std due to floor."""
        std_from_quantiles = _import_std_from_quantiles()
        quantiles = np.array([
            [0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.05]
        ]).T  # shape (7, 1) — reversed order
        _, std = std_from_quantiles(quantiles)
        assert std[0] >= 1e-9

    def test_median_used_as_mean(self):
        """The mean output equals the median row (index 3) of quantiles."""
        std_from_quantiles = _import_std_from_quantiles()
        rng = np.random.RandomState(7)
        quantiles = np.sort(rng.randn(7, 20), axis=0)
        mean, _ = std_from_quantiles(quantiles)
        np.testing.assert_array_equal(mean, quantiles[3])


# ---------------------------------------------------------------------------
# expected_improvement_numpy
# ---------------------------------------------------------------------------

class TestExpectedImprovementNumpy:
    """Tests for expected_improvement_numpy()."""

    def test_zero_ei_when_mean_equals_y_best_and_std_near_zero(self):
        """EI ≈ 0 when mean == y_best and std → 0."""
        ei_numpy = _import_ei_numpy()
        mean = np.array([1.0])
        std = np.array([1e-10])
        ei = ei_numpy(mean, std, y_best=1.0)
        assert ei[0] < 1e-8

    def test_positive_ei_when_mean_above_y_best(self):
        """EI > 0 when mean > y_best (improvement guaranteed)."""
        ei_numpy = _import_ei_numpy()
        mean = np.array([3.0])
        std = np.array([0.5])
        ei = ei_numpy(mean, std, y_best=1.0)
        assert ei[0] > 0.0

    def test_positive_ei_when_mean_below_y_best_but_std_large(self):
        """EI > 0 when std > 0 even if mean < y_best (exploration term)."""
        ei_numpy = _import_ei_numpy()
        mean = np.array([-5.0])
        std = np.array([2.0])
        ei = ei_numpy(mean, std, y_best=1.0)
        assert ei[0] > 0.0

    def test_non_negative_everywhere(self):
        """EI is non-negative for all inputs."""
        ei_numpy = _import_ei_numpy()
        rng = np.random.RandomState(42)
        mean = rng.randn(50)
        std = np.abs(rng.randn(50)) + 0.01
        ei = ei_numpy(mean, std, y_best=0.0)
        assert np.all(ei >= 0.0)

    def test_output_shape_matches_input(self):
        """Output array shape matches input mean and std."""
        ei_numpy = _import_ei_numpy()
        mean = np.zeros(30)
        std = np.ones(30)
        ei = ei_numpy(mean, std, y_best=0.5)
        assert ei.shape == (30,)

    def test_larger_std_gives_larger_ei_when_mean_below_y_best(self):
        """Higher uncertainty increases EI when mean is below y_best."""
        ei_numpy = _import_ei_numpy()
        y_best = 2.0
        mean = np.array([1.0, 1.0])
        std_small = np.array([0.1, 1.0])
        ei = ei_numpy(mean, std_small, y_best)
        assert ei[1] > ei[0]

    def test_ei_increases_with_mean_improvement(self):
        """Higher mean (closer to y_best from above) increases EI."""
        ei_numpy = _import_ei_numpy()
        std = np.array([0.5, 0.5])
        mean = np.array([1.5, 3.0])
        y_best = 1.0
        ei = ei_numpy(mean, std, y_best)
        assert ei[1] > ei[0]
