"""Unit tests for the stress-knob framework and the two K2 SNR knobs (restructure 2026-09-23)."""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.data import snr, stress
from pfns4neurostim.data.channels import ChannelData


def _channel(n_sites: int = 40, n_trials: int = 8, noise: float = 0.3, seed: int = 0) -> ChannelData:
    """Build a small synthetic channel with a known noise level.

    Args:
        n_sites: Number of sites N.
        n_trials: Trials per site R.
        noise: Within-site trial standard deviation.
        seed: RNG seed.

    Returns:
        A :class:`ChannelData` whose trials are the GT plus Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    y_gt = rng.normal(size=n_sites)                                     # [N]
    Y = y_gt[:, None] + noise * rng.normal(size=(n_sites, n_trials))    # [N, R]
    X = rng.random((n_sites, 2))                                        # [N, 2]
    ch2xy = np.stack([np.arange(n_sites) // 8, np.arange(n_sites) % 8], axis=1)
    return ChannelData("nhp", 1, 0, X, Y, y_gt, ch2xy, (n_sites // 8, 8))


class TestRegistry:
    """The knob registry and its declared placeholders."""

    def test_implemented_knobs(self) -> None:
        """K1 (Demo 1), both K2 settings, K5 and both K6 variants are live; K7 is still declared."""
        assert stress.available_knobs(implemented_only=True) == [
            "k1_decoy", "k2_channel", "k2_global", "k5_outliers", "k6_budget", "k6_failure"
        ]

    def test_every_knob_is_registered(self) -> None:
        for name in ("k1_decoy", "k2_channel", "k2_global", "k5_outliers", "k6_budget", "k6_failure", "k7_shuffle"):
            assert name in stress.KNOB_REGISTRY

    def test_seed_key_defaults_to_name_and_survives_the_k2_rename(self) -> None:
        """Seeds (hence cached cells) must not change when a knob is renamed."""
        assert stress.KNOB_REGISTRY["k2_channel"].seed_key == "k2_snr"
        for name in ("k2_global", "k5_outliers", "k6_budget", "k6_failure", "k1_decoy"):
            assert stress.KNOB_REGISTRY[name].seed_key == name

    def test_remaining_placeholders_raise(self) -> None:
        """K7 (spatial shuffle) is the only declared placeholder left."""
        for name in ("k7_shuffle",):
            with pytest.raises(NotImplementedError, match="not implemented"):
                stress.build_knob(name)

    def test_unknown_knob_raises_with_options(self) -> None:
        with pytest.raises(KeyError, match="k2_channel"):
            stress.build_knob("does_not_exist")

    def test_levels_override(self) -> None:
        knob = stress.build_knob("k2_channel", [1.0, 2.0])
        assert knob.levels == (1.0, 2.0)

    def test_empty_levels_raise(self) -> None:
        with pytest.raises(ValueError, match="empty level ladder"):
            stress.build_knob("k2_channel", [])


class TestK2ChannelKnob:
    """Algebra of the K2-channel residual-amplification knob."""

    def test_nominal_level_is_the_identity(self) -> None:
        ch = _channel()
        out = stress.build_knob("k2_channel").apply(ch, 1.0, np.random.default_rng(0))
        np.testing.assert_allclose(out.Y_trials, ch.Y_trials)

    def test_ground_truth_is_untouched(self) -> None:
        ch = _channel()
        out = stress.build_knob("k2_channel").apply(ch, 3.0, np.random.default_rng(0))
        np.testing.assert_allclose(out.y_gt, ch.y_gt)
        np.testing.assert_allclose(out.X_pool, ch.X_pool)

    def test_does_not_mutate_its_input(self) -> None:
        ch = _channel()
        before = ch.Y_trials.copy()
        stress.build_knob("k2_channel").apply(ch, 4.0, np.random.default_rng(0))
        np.testing.assert_allclose(ch.Y_trials, before)

    @pytest.mark.parametrize("alpha", [0.5, 1.5, 2.0, 4.0])
    def test_snr_shifts_by_minus_20_log10_alpha(self, alpha: float) -> None:
        """The defining identity of the knob: alpha scales noise power by alpha^2."""
        ch = _channel()
        knob = stress.build_knob("k2_channel")
        base = snr.achieved_snr_db(ch)
        stressed = knob.apply(ch, alpha, np.random.default_rng(0))
        assert snr.achieved_snr_db(stressed) - base == pytest.approx(
            -20.0 * np.log10(alpha), abs=1e-9
        )

    def test_achieved_reports_snr(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k2_channel")
        out = knob.achieved(knob.apply(ch, 2.0, np.random.default_rng(0)))
        assert set(out) == {"achieved_snr_db"}
        assert np.isfinite(out["achieved_snr_db"])

    def test_stress_provenance_is_recorded(self) -> None:
        out = stress.build_knob("k2_channel").apply(_channel(), 2.0, np.random.default_rng(0))
        assert out.stress == {"knob": "k2_channel", "level": 2.0}

    def test_nan_mask_is_preserved(self) -> None:
        ch = _channel()
        Y = ch.Y_trials.copy()
        Y[0, 0] = np.nan
        Y[5, 2] = np.nan
        ch = ch.with_trials(Y)
        out = stress.build_knob("k2_channel").apply(ch, 2.0, np.random.default_rng(0))
        np.testing.assert_array_equal(np.isnan(out.Y_trials), np.isnan(Y))

    def test_deterministic_under_repeated_application(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k2_channel")
        a = knob.apply(ch, 2.5, np.random.default_rng(0)).Y_trials
        b = knob.apply(ch, 2.5, np.random.default_rng(999)).Y_trials
        np.testing.assert_allclose(a, b)

    @pytest.mark.parametrize("alpha", [0.0, -1.0])
    def test_non_positive_alpha_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            stress.build_knob("k2_channel").apply(_channel(), alpha, np.random.default_rng(0))

    def test_budget_is_unchanged_by_a_data_knob(self) -> None:
        assert stress.build_knob("k2_channel").budget_for(4.0, 50) == 50


class TestK2GlobalKnob:
    """K2-global: one absolute Gaussian noise level on every trial of every channel."""

    def test_zero_level_is_the_identity(self) -> None:
        ch = _channel()
        out = stress.build_knob("k2_global").apply(ch, 0.0, np.random.default_rng(0))
        np.testing.assert_allclose(out.Y_trials, ch.Y_trials)

    def test_added_noise_has_the_level_as_standard_deviation(self) -> None:
        ch = _channel(n_sites=400, n_trials=20)
        out = stress.build_knob("k2_global").apply(ch, 0.8, np.random.default_rng(0))
        added = (out.Y_trials - ch.Y_trials).ravel()
        assert np.std(added) == pytest.approx(0.8, rel=0.03)
        assert np.mean(added) == pytest.approx(0.0, abs=0.02)

    def test_same_level_is_the_same_absolute_noise_on_every_channel(self) -> None:
        """Not proportional to the floor: the same absolute noise costs a clean channel more dB."""
        knob = stress.build_knob("k2_global")
        clean, noisy = _channel(noise=0.1, seed=1), _channel(noise=1.0, seed=1)
        drop_clean = snr.achieved_snr_db(clean) - snr.achieved_snr_db(knob.apply(clean, 1.0, np.random.default_rng(0)))
        drop_noisy = snr.achieved_snr_db(noisy) - snr.achieved_snr_db(knob.apply(noisy, 1.0, np.random.default_rng(0)))
        assert drop_clean > drop_noisy + 5.0

    def test_reaches_negative_snr(self) -> None:
        """SNR < 0 dB is a valid, reported regime (restructure 2026-09-23)."""
        knob = stress.build_knob("k2_global")
        out = knob.achieved(knob.apply(_channel(), 3.0, np.random.default_rng(0)))
        assert out["achieved_snr_db"] < 0.0

    def test_nan_mask_ground_truth_and_input_untouched(self) -> None:
        ch = _channel()
        Y = ch.Y_trials.copy()
        Y[1, 1] = np.nan
        ch = ch.with_trials(Y)
        out = stress.build_knob("k2_global").apply(ch, 1.0, np.random.default_rng(0))
        np.testing.assert_array_equal(np.isnan(out.Y_trials), np.isnan(Y))
        np.testing.assert_allclose(out.y_gt, ch.y_gt)
        np.testing.assert_allclose(ch.Y_trials, Y)

    def test_negative_level_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            stress.build_knob("k2_global", [-0.5])


class TestCalibration:
    """dB ladders relative to each channel's floor are solved exactly (K2-channel only)."""

    @pytest.mark.parametrize("target", [12.0, 0.0, -6.0, -18.0, -30.0])
    def test_k2_channel_hits_every_target_exactly(self, target: float) -> None:
        ch = _channel()
        (cal,) = stress.calibrate_levels(stress.build_knob("k2_channel"), ch, [target], np.random.default_rng(0))
        assert cal.achieved_db == pytest.approx(target, abs=1e-9)
        assert cal.level == pytest.approx(10.0 ** (-target / 20.0))

    def test_ladder_can_cross_below_zero_db_absolute(self) -> None:
        ch = _channel(noise=0.3)
        floor = snr.achieved_snr_db(ch)
        knob = stress.build_knob("k2_channel")
        (cal,) = stress.calibrate_levels(knob, ch, [-(floor + 5.0)], np.random.default_rng(0))
        assert snr.achieved_snr_db(knob.apply(ch, cal.level, np.random.default_rng(cal.seed))) < 0.0

    @pytest.mark.parametrize("name", ["k2_global", "k5_outliers", "k6_failure"])
    def test_knobs_without_exact_inversion_refuse_db_targets(self, name: str) -> None:
        with pytest.raises(ValueError, match="no exact dB inversion"):
            stress.calibrate_levels(stress.build_knob(name), _channel(), [-3.0], np.random.default_rng(0))


class TestSNR:
    """Achieved-SNR estimation."""

    def test_less_noise_means_higher_snr(self) -> None:
        quiet = snr.achieved_snr_db(_channel(noise=0.1))
        loud = snr.achieved_snr_db(_channel(noise=1.0))
        assert quiet > loud

    def test_per_site_snr_shape_and_nan_policy(self) -> None:
        ch = _channel()
        Y = ch.Y_trials.copy()
        Y[3, 1:] = np.nan            # one valid trial left -> variance undefined
        values = snr.per_site_snr_db(ch.with_trials(Y))
        assert values.shape == (ch.n_sites,)
        assert np.isnan(values[3])
        assert np.isfinite(values[0])

    def test_zero_noise_raises_rather_than_returning_inf(self) -> None:
        ch = _channel(noise=0.0)
        with pytest.raises(RuntimeError, match="non-positive power"):
            snr.achieved_snr_db(ch)


class TestChannelValidation:
    """ChannelData fails fast on malformed inputs (CLAUDE.md section 4)."""

    def test_shape_mismatch_raises(self) -> None:
        ch = _channel()
        with pytest.raises(ValueError, match="y_gt has shape"):
            ChannelData(
                "nhp", 1, 0, ch.X_pool, ch.Y_trials, ch.y_gt[:-1], ch.ch2xy, ch.grid_shape
            )

    def test_non_finite_ground_truth_raises(self) -> None:
        ch = _channel()
        bad = ch.y_gt.copy()
        bad[0] = np.nan
        with pytest.raises(RuntimeError, match="non-finite"):
            ChannelData("nhp", 1, 0, ch.X_pool, ch.Y_trials, bad, ch.ch2xy, ch.grid_shape)

    def test_site_with_no_valid_trial_raises(self) -> None:
        ch = _channel()
        Y = ch.Y_trials.copy()
        Y[2, :] = np.nan
        with pytest.raises(RuntimeError, match="no valid trial"):
            ch.with_trials(Y)

    def test_gt_range_and_best_site(self) -> None:
        ch = _channel()
        assert ch.gt_range == pytest.approx(float(ch.y_gt.max() - ch.y_gt.min()))
        assert ch.best_site == int(np.argmax(ch.y_gt))
