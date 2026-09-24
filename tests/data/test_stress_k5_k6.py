"""Unit and behaviour tests for the K5 outlier and K6 sparsity knobs (restructure 2026-09-23)."""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.data import snr, stress
from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.data.stress import KnobNotApplicable


def _channel(n_sites: int = 40, n_trials: int = 8, noise: float = 0.3, seed: int = 0) -> ChannelData:
    """Build a synthetic channel with Gaussian trial noise around a random ground truth.

    Args:
        n_sites: Number of sites N.
        n_trials: Trials per site R.
        noise: Within-site trial standard deviation.
        seed: RNG seed.

    Returns:
        A :class:`ChannelData`.
    """
    rng = np.random.default_rng(seed)
    y_gt = rng.normal(size=n_sites)                                    # [N]
    Y = y_gt[:, None] + noise * rng.normal(size=(n_sites, n_trials))   # [N, R]
    X = rng.random((n_sites, 2))                                       # [N, 2]
    ch2xy = np.stack([np.arange(n_sites) // 8, np.arange(n_sites) % 8], axis=1)
    return ChannelData("nhp", 1, 0, X, Y, y_gt, ch2xy, (n_sites // 8, 8))


class TestK5Outliers:
    """Epsilon-contamination of all trial slots with a Student-t heavy tail."""

    def test_registered_and_implemented(self) -> None:
        assert "k5_outliers" in stress.available_knobs(implemented_only=True)

    def test_zero_level_is_the_identity(self) -> None:
        ch = _channel()
        out = stress.build_knob("k5_outliers").apply(ch, 0.0, np.random.default_rng(0))
        np.testing.assert_allclose(out.Y_trials, ch.Y_trials)

    @pytest.mark.parametrize("n_trials", [8, 20])
    def test_epsilon_is_a_fraction_of_all_slots_whatever_the_repetition_count(self, n_trials: int) -> None:
        """The same epsilon contaminates the same share on an 8- and a 20-repetition dataset."""
        ch = _channel(n_sites=40, n_trials=n_trials)
        out = stress.build_knob("k5_outliers").apply(ch, 0.1, np.random.default_rng(0))
        changed = ~np.isclose(out.Y_trials, ch.Y_trials)
        assert changed.sum() == round(0.1 * 40 * n_trials)
        assert stress.build_knob("k5_outliers").achieved(out)["achieved_contamination"] == pytest.approx(0.1)

    def test_slots_are_spread_over_sites_not_one_per_site(self) -> None:
        """Contamination is sampled over slots, so some sites get several artefacts and some none."""
        ch = _channel(n_sites=40, n_trials=8)
        out = stress.build_knob("k5_outliers").apply(ch, 0.2, np.random.default_rng(1))
        per_site = (~np.isclose(out.Y_trials, ch.Y_trials)).sum(axis=1)
        assert per_site.max() >= 2 and (per_site == 0).any()

    def test_contaminant_is_heavy_tailed_at_the_configured_scale(self) -> None:
        """Replaced values follow y_gt + scale * sigma_noise * t(df): the standardized residuals are t(3)."""
        ch = _channel(n_sites=400, n_trials=20, noise=0.3)
        knob = stress.build_knob("k5_outliers", df=3.0, scale=5.0)
        out = knob.apply(ch, 0.5, np.random.default_rng(0))
        changed = ~np.isclose(out.Y_trials, ch.Y_trials)
        rows = np.nonzero(changed)[0]
        sigma = np.sqrt(snr.noise_power(ch))
        z = (out.Y_trials[changed] - ch.y_gt[rows]) / (5.0 * sigma)
        # Student-t(3) has excess kurtosis far above a Gaussian's 0, and median |t| ~ 0.765.
        assert np.median(np.abs(z)) == pytest.approx(0.765, rel=0.1)
        kurt = np.mean((z - z.mean()) ** 4) / np.var(z) ** 2 - 3.0
        assert kurt > 2.0

    def test_ground_truth_nan_mask_and_input_are_untouched(self) -> None:
        ch = _channel()
        Y = ch.Y_trials.copy()
        Y[0, 0] = np.nan
        ch = ch.with_trials(Y)
        before = ch.Y_trials.copy()
        out = stress.build_knob("k5_outliers").apply(ch, 0.3, np.random.default_rng(0))
        np.testing.assert_allclose(out.y_gt, ch.y_gt)
        np.testing.assert_array_equal(np.isnan(out.Y_trials), np.isnan(Y))
        np.testing.assert_allclose(ch.Y_trials, before)

    def test_contamination_lowers_achieved_snr(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k5_outliers")
        assert snr.achieved_snr_db(knob.apply(ch, 0.2, np.random.default_rng(0))) < snr.achieved_snr_db(ch)

    def test_no_real_artefact_source_remains(self) -> None:
        with pytest.raises(TypeError):
            stress.build_knob("k5_outliers", source="invalid")

    @pytest.mark.parametrize("bad", [{"df": 0.0}, {"scale": -1.0}])
    def test_non_positive_df_or_scale_raises(self, bad: dict) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            stress.build_knob("k5_outliers", **bad)

    def test_level_outside_unit_interval_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            stress.build_knob("k5_outliers", [1.5])

    def test_deterministic_under_seed(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k5_outliers")
        a = knob.apply(ch, 0.2, np.random.default_rng(3)).Y_trials
        b = knob.apply(ch, 0.2, np.random.default_rng(3)).Y_trials
        np.testing.assert_allclose(a, b)


class TestK6Budget:
    """K6 sparsity via the BO budget."""

    def test_level_is_the_budget(self) -> None:
        knob = stress.build_knob("k6_budget", [10, 20])
        assert knob.budget_for(10, 96) == 10
        assert knob.budget_for(20, 96) == 20

    def test_data_is_unchanged(self) -> None:
        ch = _channel()
        out = stress.build_knob("k6_budget").apply(ch, 20, np.random.default_rng(0))
        np.testing.assert_allclose(out.Y_trials, ch.Y_trials)
        np.testing.assert_allclose(out.y_gt, ch.y_gt)

    def test_achieved_reports_the_budget(self) -> None:
        knob = stress.build_knob("k6_budget")
        out = knob.achieved(knob.apply(_channel(), 30, np.random.default_rng(0)))
        assert out["achieved_budget"] == 30.0

    def test_fractional_level_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            stress.build_knob("k6_budget").apply(_channel(), 12.5, np.random.default_rng(0))


class TestK6Failure:
    """K6 sparsity via electrode failure at random times during the run."""

    def test_zero_level_fails_nothing(self) -> None:
        ch = _channel()
        out = stress.build_knob("k6_failure").apply(ch, 0.0, np.random.default_rng(0))
        assert out.survivors.all()

    def test_failure_fraction_and_times(self) -> None:
        ch = _channel(n_sites=40)
        out = stress.build_knob("k6_failure").apply(ch, 0.25, np.random.default_rng(0))
        failing = np.isfinite(out.failure_time)
        assert failing.sum() == 10
        assert ((out.failure_time[failing] >= 0.0) & (out.failure_time[failing] < 1.0)).all()
        assert out.survivors.sum() == 30

    def test_ground_truth_and_trials_untouched(self) -> None:
        ch = _channel()
        out = stress.build_knob("k6_failure").apply(ch, 0.5, np.random.default_rng(0))
        np.testing.assert_allclose(out.y_gt, ch.y_gt)
        np.testing.assert_allclose(out.Y_trials, ch.Y_trials)

    def test_y_end_reads_the_dead_value_at_failed_electrodes(self) -> None:
        ch = _channel()
        out = stress.build_knob("k6_failure", dead_value=0.0).apply(ch, 0.5, np.random.default_rng(0))
        np.testing.assert_allclose(out.y_end[~out.survivors], 0.0)
        np.testing.assert_allclose(out.y_end[out.survivors], ch.y_gt[out.survivors])

    def test_is_failed_follows_the_failure_time(self) -> None:
        ch = _channel()
        out = stress.build_knob("k6_failure").apply(ch, 0.5, np.random.default_rng(0))
        i = int(np.flatnonzero(np.isfinite(out.failure_time))[0])
        t = float(out.failure_time[i])
        assert not out.is_failed(i, t - 1e-9) and out.is_failed(i, t)
        j = int(np.flatnonzero(out.survivors)[0])
        assert not out.is_failed(j, 0.999)

    def test_achieved_reports_failure_and_survivors(self) -> None:
        knob = stress.build_knob("k6_failure")
        out = knob.achieved(knob.apply(_channel(n_sites=40), 0.25, np.random.default_rng(0)))
        assert out["achieved_failure"] == pytest.approx(0.25)
        assert out["n_survivors"] == 30.0

    def test_too_much_failure_raises_knob_not_applicable(self) -> None:
        ch = _channel(n_sites=8)
        with pytest.raises(KnobNotApplicable, match="at least two"):
            stress.build_knob("k6_failure").apply(ch, 0.9, np.random.default_rng(0))

    def test_level_of_one_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            stress.build_knob("k6_failure", [1.0])

    def test_deterministic_under_seed(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k6_failure")
        a = knob.apply(ch, 0.3, np.random.default_rng(5)).failure_time
        b = knob.apply(ch, 0.3, np.random.default_rng(5)).failure_time
        np.testing.assert_array_equal(a, b)


class TestFailureInTheLoop:
    """Shadow tests: a failed electrode returns the dead value and scoring uses survivors."""

    @staticmethod
    def _run(channel: ChannelData, budget: int = 30, seed: int = 2) -> dict:
        from pfns4neurostim.evaluation.bo_runner import run_channel_bo

        return run_channel_bo(
            "gp_naive", channel, acq_fn="ei", budget=budget, n_init=3, seed=seed, device="cpu"
        )

    def test_queries_after_failure_return_the_dead_value(self) -> None:
        ch = _channel(n_sites=40)
        stressed = stress.build_knob("k6_failure").apply(ch, 0.5, np.random.default_rng(0))
        res = self._run(stressed, budget=40)
        traj = res.trajectory
        for t, (i, obs, real) in enumerate(
            zip(traj["observed_indices"], traj["observed_values"], traj["real_values"])
        ):
            if stressed.is_failed(i, t / 40):
                assert obs == 0.0 and real == 0.0
            else:
                assert real == pytest.approx(ch.y_gt[i])

    def test_dead_electrodes_stay_queryable(self) -> None:
        """Every electrode fails at time 0 except two: the loop still queries dead ones."""
        ch = _channel(n_sites=40)
        failure_time = np.zeros(40)
        failure_time[[3, 7]] = np.inf
        dead = ChannelData(
            ch.dataset, ch.subject, ch.emg, ch.X_pool, ch.Y_trials, ch.y_gt, ch.ch2xy,
            ch.grid_shape, failure_time=failure_time,
        )
        res = self._run(dead)
        assert set(res.trajectory["observed_indices"]) - {3, 7}

    def test_regret_is_scored_on_survivors(self) -> None:
        """Killing the global optimum moves the reference to the best surviving electrode."""
        ch = _channel(n_sites=40)
        best = int(np.argmax(ch.y_gt))
        failure_time = np.full(40, np.inf)
        failure_time[best] = 0.0
        killed = ChannelData(
            ch.dataset, ch.subject, ch.emg, ch.X_pool, ch.Y_trials, ch.y_gt, ch.ch2xy,
            ch.grid_shape, failure_time=failure_time,
        )
        res = self._run(killed)
        alive = killed.survivors
        y_star = ch.y_gt[alive].max()
        rng_alive = np.ptp(ch.y_gt[alive])
        rec = int(res.trajectory["best_rec_indices"][-1])
        rec_value = 0.0 if rec == best else ch.y_gt[rec]
        assert res.row["recommended_regret"] == pytest.approx((y_star - rec_value) / rng_alive)
        assert res.trajectory["gt_range"] == pytest.approx(rng_alive)
        assert res.row["best_queried_regret"] >= 0.0

    def test_no_failure_matches_the_nominal_run(self) -> None:
        """At epsilon = 0 the knob is the identity on every reported metric."""
        ch = _channel(n_sites=40)
        stressed = stress.build_knob("k6_failure").apply(ch, 0.0, np.random.default_rng(0))
        a, b = self._run(ch), self._run(stressed)
        assert a.trajectory["observed_indices"] == b.trajectory["observed_indices"]
        assert a.row["recommended_regret"] == pytest.approx(b.row["recommended_regret"])
