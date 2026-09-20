"""Unit tests for the K5 outlier and K6 sparsity knobs (task #10 Step 8)."""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.data import snr, stress
from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.data.stress import KnobNotApplicable


def _channel(
    n_sites: int = 40,
    n_trials: int = 8,
    noise: float = 0.3,
    n_invalid: int = 12,
    seed: int = 0,
) -> ChannelData:
    """Build a synthetic channel with a controllable pool of invalid trials.

    Args:
        n_sites: Number of sites N.
        n_trials: Trials per site R.
        noise: Within-site trial standard deviation.
        n_invalid: How many artefact trials to place in ``Y_invalid``.
        seed: RNG seed.

    Returns:
        A :class:`ChannelData` with artefacts an order of magnitude larger than
        the clean trials, so contamination is detectable.
    """
    rng = np.random.default_rng(seed)
    y_gt = rng.normal(size=n_sites)                                    # [N]
    Y = y_gt[:, None] + noise * rng.normal(size=(n_sites, n_trials))   # [N, R]
    X = rng.random((n_sites, 2))                                       # [N, 2]
    ch2xy = np.stack([np.arange(n_sites) // 8, np.arange(n_sites) % 8], axis=1)

    Y_invalid = np.full_like(Y, np.nan)
    if n_invalid:
        rows = rng.integers(0, n_sites, size=n_invalid)
        cols = rng.integers(0, n_trials, size=n_invalid)
        Y_invalid[rows, cols] = 50.0 + rng.normal(size=n_invalid)      # obvious artefacts
    return ChannelData(
        "nhp", 1, 0, X, Y, y_gt, ch2xy, (n_sites // 8, 8), Y_invalid=Y_invalid
    )


class TestK5Outliers:
    """Contamination with real lab-flagged artefacts (Demo 2)."""

    def test_registered_and_implemented(self) -> None:
        assert "k5_outliers" in stress.available_knobs(implemented_only=True)

    def test_zero_level_is_the_identity(self) -> None:
        ch = _channel()
        out = stress.build_knob("k5_outliers").apply(ch, 0.0, np.random.default_rng(0))
        np.testing.assert_allclose(out.Y_trials, ch.Y_trials)

    def test_contamination_replaces_the_requested_fraction(self) -> None:
        ch = _channel()
        out = stress.build_knob("k5_outliers").apply(ch, 0.25, np.random.default_rng(0))
        changed = ~np.isclose(out.Y_trials, ch.Y_trials)
        assert changed.sum() == pytest.approx(0.25 * ch.Y_trials.size, rel=0.02)

    def test_replaced_values_come_from_the_artefact_pool(self) -> None:
        """Demo 2 contaminates with the channel's *own* rejected trials."""
        ch = _channel()
        out = stress.build_knob("k5_outliers").apply(ch, 0.2, np.random.default_rng(0))
        changed = ~np.isclose(out.Y_trials, ch.Y_trials)
        donors = ch.Y_invalid[np.isfinite(ch.Y_invalid)]
        assert np.isin(out.Y_trials[changed], donors).all()

    def test_ground_truth_and_shape_are_untouched(self) -> None:
        ch = _channel()
        out = stress.build_knob("k5_outliers").apply(ch, 0.4, np.random.default_rng(0))
        np.testing.assert_allclose(out.y_gt, ch.y_gt)
        assert out.Y_trials.shape == ch.Y_trials.shape

    def test_contamination_lowers_achieved_snr(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k5_outliers")
        clean = snr.achieved_snr_db(knob.apply(ch, 0.0, np.random.default_rng(0)))
        dirty = snr.achieved_snr_db(knob.apply(ch, 0.3, np.random.default_rng(0)))
        assert dirty < clean

    def test_achieved_reports_contamination_and_donor_count(self) -> None:
        """Donor scarcity must be visible in the output, not hidden."""
        ch = _channel(n_invalid=7)
        knob = stress.build_knob("k5_outliers")
        out = knob.achieved(knob.apply(ch, 0.2, np.random.default_rng(0)))
        assert out["achieved_contamination"] == pytest.approx(0.2, abs=0.02)
        assert out["n_donor_trials"] == 7.0

    def test_channel_without_artefacts_raises_knob_not_applicable(self) -> None:
        ch = _channel(n_invalid=0)
        with pytest.raises(KnobNotApplicable, match="no lab-flagged"):
            stress.build_knob("k5_outliers").apply(ch, 0.2, np.random.default_rng(0))

    def test_heavy_tail_source_works_without_artefacts(self) -> None:
        """The synthetic fallback is what makes artefact-free channels sweepable."""
        ch = _channel(n_invalid=0)
        knob = stress.K5OutlierKnob([0.0, 0.2], source="heavy_tail")
        out = knob.apply(ch, 0.2, np.random.default_rng(0))
        changed = ~np.isclose(out.Y_trials, ch.Y_trials)
        assert changed.sum() > 0
        assert np.isfinite(out.Y_trials).all()

    def test_unknown_source_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown source"):
            stress.K5OutlierKnob([0.1], source="gaussian")

    def test_level_outside_unit_interval_raises(self) -> None:
        with pytest.raises(ValueError, match=r"in \[0, 1\]"):
            stress.K5OutlierKnob([1.5])

    def test_does_not_mutate_its_input(self) -> None:
        ch = _channel()
        before = ch.Y_trials.copy()
        stress.build_knob("k5_outliers").apply(ch, 0.3, np.random.default_rng(0))
        np.testing.assert_allclose(ch.Y_trials, before)

    def test_deterministic_under_seed(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k5_outliers")
        a = knob.apply(ch, 0.25, np.random.default_rng(3)).Y_trials
        b = knob.apply(ch, 0.25, np.random.default_rng(3)).Y_trials
        np.testing.assert_allclose(a, b)


class TestK6Budget:
    """Sparsity through the BO budget: the data is untouched."""

    def test_level_is_the_budget(self) -> None:
        knob = stress.build_knob("k6_budget")
        assert knob.budget_for(20.0, 50) == 20
        assert knob.budget_for(50.0, 10) == 50

    def test_data_is_unchanged(self) -> None:
        ch = _channel()
        out = stress.build_knob("k6_budget").apply(ch, 20.0, np.random.default_rng(0))
        np.testing.assert_allclose(out.Y_trials, ch.Y_trials)
        np.testing.assert_allclose(out.y_gt, ch.y_gt)
        assert out.n_queryable == ch.n_queryable

    def test_achieved_reports_the_budget(self) -> None:
        knob = stress.build_knob("k6_budget")
        out = knob.achieved(knob.apply(_channel(), 30.0, np.random.default_rng(0)))
        assert out["achieved_budget"] == 30.0

    def test_fractional_level_raises(self) -> None:
        """A budget is a query count, not a fraction of the grid (P0.3)."""
        with pytest.raises(ValueError, match="positive integer"):
            stress.build_knob("k6_budget").apply(_channel(), 12.5, np.random.default_rng(0))


class TestK6Dropout:
    """Sparsity through electrode dropout: sites are masked, never deleted."""

    def test_zero_level_keeps_every_site(self) -> None:
        ch = _channel()
        out = stress.build_knob("k6_dropout").apply(ch, 0.0, np.random.default_rng(0))
        assert out.n_queryable == ch.n_sites

    def test_dropout_masks_the_requested_fraction(self) -> None:
        ch = _channel(n_sites=40)
        out = stress.build_knob("k6_dropout").apply(ch, 0.25, np.random.default_rng(0))
        assert out.n_queryable == 30
        assert out.n_sites == 40, "sites must be masked, not deleted"

    def test_ground_truth_still_spans_every_site(self) -> None:
        """Regret must stay measured against the true optimum, dropped or not."""
        ch = _channel(n_sites=40)
        out = stress.build_knob("k6_dropout").apply(ch, 0.5, np.random.default_rng(0))
        np.testing.assert_allclose(out.y_gt, ch.y_gt)
        assert out.gt_range == pytest.approx(ch.gt_range)
        assert out.best_site == ch.best_site

    def test_achieved_reports_dropout_and_remaining_sites(self) -> None:
        ch = _channel(n_sites=40)
        knob = stress.build_knob("k6_dropout")
        out = knob.achieved(knob.apply(ch, 0.25, np.random.default_rng(0)))
        assert out["achieved_dropout"] == pytest.approx(0.25)
        assert out["n_queryable"] == 30.0

    def test_too_much_dropout_raises_knob_not_applicable(self) -> None:
        ch = _channel(n_sites=4)
        with pytest.raises(KnobNotApplicable, match="at least two"):
            stress.build_knob("k6_dropout").apply(ch, 0.9, np.random.default_rng(0))

    def test_level_of_one_raises(self) -> None:
        with pytest.raises(ValueError, match=r"in \[0, 1\)"):
            stress.build_knob("k6_dropout").apply(_channel(), 1.0, np.random.default_rng(0))

    def test_deterministic_under_seed(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k6_dropout")
        a = knob.apply(ch, 0.3, np.random.default_rng(5)).queryable
        b = knob.apply(ch, 0.3, np.random.default_rng(5)).queryable
        np.testing.assert_array_equal(a, b)


class TestDropoutConstrainsTheLoop:
    """A masked electrode must be unqueryable *and* unrecommendable."""

    def test_loop_never_queries_a_dropped_site(self) -> None:
        from pfns4neurostim.evaluation.bo_runner import run_channel_bo

        ch = _channel(n_sites=40, n_invalid=0)
        stressed = stress.build_knob("k6_dropout").apply(ch, 0.5, np.random.default_rng(0))
        allowed = set(stressed.queryable_indices.tolist())
        res = run_channel_bo(
            "gp_naive", stressed, acq_fn="ei", budget=10, n_init=3, seed=2, device="cpu"
        )
        assert set(res.trajectory["observed_indices"]) <= allowed
        assert set(res.trajectory["best_rec_indices"]) <= allowed

    def test_budget_above_the_queryable_count_raises(self) -> None:
        from pfns4neurostim.evaluation.bo_runner import run_channel_bo

        ch = _channel(n_sites=40, n_invalid=0)
        stressed = stress.build_knob("k6_dropout").apply(ch, 0.5, np.random.default_rng(0))
        with pytest.raises(ValueError, match="queryable site"):
            run_channel_bo("gp_naive", stressed, budget=30, n_init=3, device="cpu")
