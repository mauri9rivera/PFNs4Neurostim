"""Unit tests for the range-normalized metrics and the S9 robustness statistics."""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.evaluation import metrics, robustness


class TestRegretMetrics:
    """Regrets are range-normalized and therefore scale-invariant (P0.9)."""

    def test_perfect_recommendation_has_zero_regret(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        out = metrics.regret_metrics(y, [0, 3], recommended_index=3)
        assert out["recommended_regret"] == pytest.approx(0.0)
        assert out["best_queried_regret"] == pytest.approx(0.0)

    def test_worst_recommendation_has_unit_regret(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        out = metrics.regret_metrics(y, [1, 2], recommended_index=0)
        assert out["recommended_regret"] == pytest.approx(1.0)

    def test_invariant_to_affine_rescaling(self) -> None:
        """The whole point of P0.9: units must not change the numbers."""
        y = np.array([0.0, 1.0, 2.0, 5.0])
        base = metrics.regret_metrics(y, [0, 1], recommended_index=1)
        scaled = metrics.regret_metrics(3.7 * y - 12.0, [0, 1], recommended_index=1)
        for key in base:
            assert base[key] == pytest.approx(scaled[key])

    def test_cumulative_regret_accumulates(self) -> None:
        y = np.array([0.0, 1.0, 2.0])
        one = metrics.regret_metrics(y, [0], recommended_index=2)["cumulative_regret"]
        two = metrics.regret_metrics(y, [0, 0], recommended_index=2)["cumulative_regret"]
        assert two == pytest.approx(2.0 * one)

    def test_degenerate_range_raises(self) -> None:
        with pytest.raises(RuntimeError, match="degenerate"):
            metrics.regret_metrics(np.ones(5), [0, 1], recommended_index=1)


class TestAccuracyAndIdentification:
    """R-squared, Spearman, hit rates and optimum distance."""

    def test_perfect_prediction(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        out = metrics.surrogate_accuracy(y, y)
        assert out["r2"] == pytest.approx(1.0)
        assert out["spearman"] == pytest.approx(1.0)

    def test_non_finite_prediction_raises(self) -> None:
        with pytest.raises(RuntimeError, match="non-finite"):
            metrics.surrogate_accuracy(np.arange(4.0), np.array([0.0, np.nan, 2.0, 3.0]))

    def test_identification_hits_and_distance(self) -> None:
        y = np.array([0.0, 3.0, 2.0, 1.0])
        coords = np.array([[0, 0], [0, 3], [0, 2], [0, 1]])
        best = metrics.identification_metrics(y, 1, coords)
        assert best["top1_hit"] == 1.0 and best["opt_distance"] == pytest.approx(0.0)
        third = metrics.identification_metrics(y, 3, coords)
        assert third["top1_hit"] == 0.0
        assert third["top3_hit"] == 1.0
        assert third["opt_distance"] == pytest.approx(2.0)


class TestCalibration:
    """Coverage, ECE, NLL and CRPS behave as the S9 row expects."""

    def test_well_calibrated_predictions_hit_nominal_coverage(self) -> None:
        rng = np.random.default_rng(0)
        mu = np.zeros(20000)
        sigma = np.ones(20000)
        y = rng.normal(size=20000)
        out = metrics.calibration_metrics(y, mu, sigma, gt_range=1.0)
        assert out["coverage_90"] == pytest.approx(0.9, abs=0.02)
        assert out["coverage_50"] == pytest.approx(0.5, abs=0.02)
        assert out["ece"] < 0.02

    def test_overconfident_predictions_undercover(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.normal(size=5000)
        out = metrics.calibration_metrics(y, np.zeros(5000), np.full(5000, 0.25), gt_range=1.0)
        assert out["coverage_90"] < 0.6
        assert out["ece"] > 0.1

    def test_non_positive_sigma_raises(self) -> None:
        with pytest.raises(RuntimeError, match="non-positive"):
            metrics.calibration_metrics(np.zeros(3), np.zeros(3), np.zeros(3), gt_range=1.0)


class TestRobustness:
    """Breakdown point, degradation AUC, CVaR and relative robustness."""

    def test_cvar_is_the_upper_tail_mean(self) -> None:
        values = np.arange(10.0)
        assert robustness.cvar(values, alpha=0.1) == pytest.approx(9.0)
        assert robustness.cvar(values, alpha=0.5) == pytest.approx(np.mean(np.arange(5.0, 10.0)))

    def test_cvar_requires_finite_values(self) -> None:
        with pytest.raises(ValueError, match="no finite values"):
            robustness.cvar([np.nan, np.nan])

    def test_degradation_auc_of_a_constant_is_that_constant(self) -> None:
        assert robustness.degradation_auc([0.0, 1.0, 2.0], [0.4, 0.4, 0.4]) == pytest.approx(0.4)

    def test_degradation_auc_is_invariant_to_x_scaling(self) -> None:
        """Normalizing x makes knobs with different units comparable."""
        a = robustness.degradation_auc([0.0, 1.0, 2.0], [0.0, 0.5, 1.0])
        b = robustness.degradation_auc([0.0, 10.0, 20.0], [0.0, 0.5, 1.0])
        assert a == pytest.approx(b)

    def test_degradation_auc_needs_two_points(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            robustness.degradation_auc([1.0], [0.5])

    def test_no_breakdown_when_models_agree(self) -> None:
        rng = np.random.default_rng(0)
        levels = [1.0, 2.0, 3.0]
        model = {lv: rng.normal(0.5, 0.01, size=30) for lv in levels}
        ref = {lv: rng.normal(0.5, 0.01, size=30) for lv in levels}
        out = robustness.breakdown_point(model, ref, margin=0.1)
        assert np.isnan(out["breakdown_level"])
        assert out["breakdown_reason"] == "none"
        assert all(out["equivalent"])

    def test_breakdown_found_at_the_first_divergent_level(self) -> None:
        rng = np.random.default_rng(1)
        model = {
            1.0: rng.normal(0.50, 0.01, size=30),
            2.0: rng.normal(0.52, 0.01, size=30),
            3.0: rng.normal(0.90, 0.01, size=30),   # clearly worse than the reference
        }
        ref = {lv: rng.normal(0.50, 0.01, size=30) for lv in (1.0, 2.0, 3.0)}
        out = robustness.breakdown_point(model, ref, margin=0.05)
        assert out["breakdown_level"] == 3.0
        assert out["breakdown_reason"] == "tost"

    def test_severity_direction_is_respected(self) -> None:
        """With achieved SNR the ladder is walked from high (mild) to low (severe)."""
        rng = np.random.default_rng(2)
        model = {
            10.0: rng.normal(0.50, 0.01, size=30),
            0.0: rng.normal(0.50, 0.01, size=30),
            -10.0: rng.normal(0.95, 0.01, size=30),
        }
        ref = {lv: rng.normal(0.50, 0.01, size=30) for lv in (10.0, 0.0, -10.0)}
        out = robustness.breakdown_point(model, ref, margin=0.05, severity_ascending=False)
        assert out["breakdown_level"] == -10.0
        assert out["levels_tested"] == [10.0, 0.0, -10.0]

    def test_mismatched_levels_raise(self) -> None:
        with pytest.raises(ValueError, match="level sets differ"):
            robustness.breakdown_point({1.0: np.zeros(5)}, {2.0: np.zeros(5)}, margin=0.1)

    def test_relative_robustness_crossing(self) -> None:
        out = robustness.relative_robustness([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
        assert out["crossing_x"] == pytest.approx(1.0)
        assert out["gap_mean"] == pytest.approx(0.0)

    def test_bootstrap_ci_brackets_the_mean(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(5.0, 1.0, size=200)
        lo, hi = bootstrap = robustness.bootstrap_ci(values, rng=rng)
        assert lo < np.mean(values) < hi
        assert len(bootstrap) == 2

    def test_bootstrap_ci_of_a_single_value_is_nan(self) -> None:
        lo, hi = robustness.bootstrap_ci([1.0])
        assert np.isnan(lo) and np.isnan(hi)
