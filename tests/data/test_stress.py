"""Unit tests for the stress-knob framework and the K2 SNR knob (task #10 Step 1)."""
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
        """K2, K5 and both K6 variants are live; K1 and K7 are still declared."""
        assert stress.available_knobs(implemented_only=True) == [
            "k2_snr", "k5_outliers", "k6_budget", "k6_dropout"
        ]

    def test_every_knob_is_registered(self) -> None:
        for name in ("k1_decoy", "k2_snr", "k5_outliers", "k6_budget", "k6_dropout", "k7_shuffle"):
            assert name in stress.KNOB_REGISTRY

    def test_remaining_placeholders_raise(self) -> None:
        """K1 needs the Demo 1 generator; K7 is the spatial-shuffle knob."""
        for name in ("k1_decoy", "k7_shuffle"):
            with pytest.raises(NotImplementedError, match="not implemented"):
                stress.build_knob(name)

    def test_unknown_knob_raises_with_options(self) -> None:
        with pytest.raises(KeyError, match="k2_snr"):
            stress.build_knob("does_not_exist")

    def test_levels_override(self) -> None:
        knob = stress.build_knob("k2_snr", [1.0, 2.0])
        assert knob.levels == (1.0, 2.0)

    def test_empty_levels_raise(self) -> None:
        with pytest.raises(ValueError, match="empty level ladder"):
            stress.build_knob("k2_snr", [])


class TestK2Knob:
    """Algebra of the K2 residual-amplification knob."""

    def test_nominal_level_is_the_identity(self) -> None:
        ch = _channel()
        out = stress.build_knob("k2_snr").apply(ch, 1.0, np.random.default_rng(0))
        np.testing.assert_allclose(out.Y_trials, ch.Y_trials)

    def test_ground_truth_is_untouched(self) -> None:
        ch = _channel()
        out = stress.build_knob("k2_snr").apply(ch, 3.0, np.random.default_rng(0))
        np.testing.assert_allclose(out.y_gt, ch.y_gt)
        np.testing.assert_allclose(out.X_pool, ch.X_pool)

    def test_does_not_mutate_its_input(self) -> None:
        ch = _channel()
        before = ch.Y_trials.copy()
        stress.build_knob("k2_snr").apply(ch, 4.0, np.random.default_rng(0))
        np.testing.assert_allclose(ch.Y_trials, before)

    @pytest.mark.parametrize("alpha", [0.5, 1.5, 2.0, 4.0])
    def test_snr_shifts_by_minus_20_log10_alpha(self, alpha: float) -> None:
        """The defining identity of the knob: alpha scales noise power by alpha^2."""
        ch = _channel()
        knob = stress.build_knob("k2_snr")
        base = snr.achieved_snr_db(ch)
        stressed = knob.apply(ch, alpha, np.random.default_rng(0))
        assert snr.achieved_snr_db(stressed) - base == pytest.approx(
            -20.0 * np.log10(alpha), abs=1e-9
        )

    def test_achieved_reports_snr(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k2_snr")
        out = knob.achieved(knob.apply(ch, 2.0, np.random.default_rng(0)))
        assert set(out) == {"achieved_snr_db"}
        assert np.isfinite(out["achieved_snr_db"])

    def test_stress_provenance_is_recorded(self) -> None:
        out = stress.build_knob("k2_snr").apply(_channel(), 2.0, np.random.default_rng(0))
        assert out.stress == {"knob": "k2_snr", "level": 2.0}

    def test_nan_mask_is_preserved(self) -> None:
        ch = _channel()
        Y = ch.Y_trials.copy()
        Y[0, 0] = np.nan
        Y[5, 2] = np.nan
        ch = ch.with_trials(Y)
        out = stress.build_knob("k2_snr").apply(ch, 2.0, np.random.default_rng(0))
        np.testing.assert_array_equal(np.isnan(out.Y_trials), np.isnan(Y))

    def test_deterministic_under_repeated_application(self) -> None:
        ch = _channel()
        knob = stress.build_knob("k2_snr")
        a = knob.apply(ch, 2.5, np.random.default_rng(0)).Y_trials
        b = knob.apply(ch, 2.5, np.random.default_rng(999)).Y_trials
        np.testing.assert_allclose(a, b)

    @pytest.mark.parametrize("alpha", [0.0, -1.0])
    def test_non_positive_alpha_raises(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            stress.build_knob("k2_snr").apply(_channel(), alpha, np.random.default_rng(0))

    def test_budget_is_unchanged_by_a_data_knob(self) -> None:
        assert stress.build_knob("k2_snr").budget_for(4.0, 50) == 50


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
