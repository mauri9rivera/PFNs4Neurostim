"""Tests for the two A8 / A9 post-hoc metrics (task plan #17).

Both are derived from artefacts a finished run already stored, so their correctness is entirely a matter of
definition: A8 must state its target on the same ground-truth range the regrets use and must **censor** rather
than drop a run that never reaches it, and A9 must be a right-continuous step function of cumulative time that
starts only once a step has actually finished.
"""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.evaluation.metrics import anytime_regret_curve, queries_to_target


@pytest.fixture()
def y_gt() -> np.ndarray:
    """Five sites spanning a range of 10, so 90 % of the range is the value 9."""
    return np.array([0.0, 2.0, 5.0, 7.0, 10.0])


class TestQueriesToTarget:
    def test_target_is_on_the_ground_truth_range(self, y_gt: np.ndarray) -> None:
        """90 % of a [0, 10] range is 9: a query of 8.9 misses, 9.0 reaches."""
        assert np.isnan(queries_to_target(y_gt, [0.0, 8.9], fractions=(0.9,))["queries_to_target_90"])
        assert queries_to_target(y_gt, [0.0, 9.0], fractions=(0.9,))["queries_to_target_90"] == 2.0

    def test_index_is_one_based_and_counts_the_first_hit(self, y_gt: np.ndarray) -> None:
        got = queries_to_target(y_gt, [10.0, 10.0, 0.0], fractions=(0.9, 0.95))
        assert got["queries_to_target_90"] == 1.0 and got["queries_to_target_95"] == 1.0

    def test_a_run_that_never_reaches_is_censored_not_zero_or_budget(self, y_gt: np.ndarray) -> None:
        """NaN, not the budget: reporting the budget would make a failure look like a slow success."""
        got = queries_to_target(y_gt, [0.0, 2.0, 5.0], fractions=(0.9,))
        assert np.isnan(got["queries_to_target_90"])

    def test_a_harder_target_is_never_reached_earlier(self, y_gt: np.ndarray) -> None:
        got = queries_to_target(y_gt, [0.0, 9.2, 9.6, 10.0])
        assert got["queries_to_target_90"] <= got["queries_to_target_95"]

    def test_the_reference_mask_sets_the_range(self, y_gt: np.ndarray) -> None:
        """Under K6 failure the optimum and range are taken over survivors, as for every other metric."""
        alive = np.array([True, True, True, True, False])      # drops the site worth 10
        # Survivor range is [0, 7], so 90 % is 6.3 -- a query of 7.0 reaches it although it misses on the
        # full grid, where the target would be 9.
        assert queries_to_target(y_gt, [7.0], reference=alive, fractions=(0.9,))["queries_to_target_90"] == 1.0
        assert np.isnan(queries_to_target(y_gt, [7.0], fractions=(0.9,))["queries_to_target_90"])

    def test_degenerate_range_fails_fast(self) -> None:
        with pytest.raises(RuntimeError, match="degenerate"):
            queries_to_target(np.ones(4), [1.0, 1.0])

    def test_column_names_match_the_tidy_schema(self, y_gt: np.ndarray) -> None:
        from pfns4neurostim.evaluation.results import ALL_COLUMNS

        for key in queries_to_target(y_gt, [10.0]):
            assert key in ALL_COLUMNS


class TestAnytimeRegretCurve:
    def test_value_at_a_grid_time_is_the_last_finished_step(self) -> None:
        got = anytime_regret_curve([1.0, 1.0, 2.0], [0.9, 0.5, 0.1], np.array([1.0, 1.5, 2.0, 4.0]))
        assert got.tolist() == [0.9, 0.9, 0.5, 0.1]

    def test_before_the_first_step_finishes_the_value_is_undefined(self) -> None:
        got = anytime_regret_curve([2.0], [0.4], np.array([0.5, 1.9, 2.0]))
        assert np.isnan(got[0]) and np.isnan(got[1]) and got[2] == 0.4

    def test_the_curve_is_monotone_even_when_regret_rises(self) -> None:
        """Best-so-far: a later recommendation that is worse must not raise the curve."""
        got = anytime_regret_curve([1.0, 1.0, 1.0], [0.5, 0.9, 0.8], np.array([1.0, 2.0, 3.0]))
        assert got.tolist() == [0.5, 0.5, 0.5]

    def test_a_leading_pre_run_regret_value_is_dropped(self) -> None:
        """``recommended_regret_per_step`` has T+1 entries (the value before any query); they must line up."""
        with_lead = anytime_regret_curve([1.0, 1.0], [1.0, 0.6, 0.2], np.array([1.0, 2.0]))
        without = anytime_regret_curve([1.0, 1.0], [0.6, 0.2], np.array([1.0, 2.0]))
        assert with_lead.tolist() == without.tolist()

    def test_mismatched_lengths_fail_fast(self) -> None:
        with pytest.raises(ValueError, match="regret values against"):
            anytime_regret_curve([1.0, 1.0], [0.5], np.array([1.0]))
