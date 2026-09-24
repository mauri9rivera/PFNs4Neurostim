"""Tests for the Demo 1 synthetic generator (roadmap S0) and the K1 decoy knob (S1)."""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.data import stress
from pfns4neurostim.data import synthetic_neurostim as sn
from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.data.snr import achieved_snr_db
from pfns4neurostim.data.stress import KnobNotApplicable


def _grid(nx: int = 10, ny: int = 10) -> np.ndarray:
    """Integer electrode coordinates of an ``nx x ny`` array, shape [N, 2]."""
    return np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), axis=-1).reshape(-1, 2)


def _params(**overrides) -> sn.GeneratorParams:
    """A two-hotspot map on a 10 x 10 array."""
    base = dict(
        ch2xy=_grid(),
        grid_shape=(10, 10),
        hotspots=(
            sn.Hotspot(np.array([6.0, 3.0]), np.array([1.5, 1.0]), 3.0, 0.4),
            sn.Hotspot(np.array([2.0, 7.0]), np.array([1.0, 1.0]), 1.0),
        ),
        baseline=0.5,
        saturation=3.0,
        noise_cv=0.5,
        n_trials=12,
    )
    base.update(overrides)
    return sn.GeneratorParams(**base)


class TestGenerator:
    """Exact ground truth, positive heteroscedastic trials, the shared preprocessing contract."""

    def test_ground_truth_is_the_noise_free_map(self) -> None:
        params = _params()
        ch = sn.generate_neurostim_map(params, np.random.default_rng(0))
        np.testing.assert_allclose(ch.to_raw(ch.y_gt), sn.mean_map(params), rtol=1e-6)
        assert ch.demo == "demo1" and ch.meta["generator"] is params

    def test_trials_are_positive_with_mean_mu_and_sd_cv_mu(self) -> None:
        params = _params(n_trials=4000, noise_cv=0.5)
        ch = sn.generate_neurostim_map(params, np.random.default_rng(1))
        raw = ch.to_raw(ch.Y_trials.reshape(-1)).reshape(ch.Y_trials.shape)
        mu = sn.mean_map(params)
        assert (raw > 0).all()
        np.testing.assert_allclose(raw.mean(axis=1), mu, rtol=0.05)
        np.testing.assert_allclose(raw.std(axis=1) / mu, 0.5, rtol=0.1)

    def test_preprocessing_contract_is_the_in_vivo_one(self) -> None:
        ch = sn.generate_neurostim_map(_params(), np.random.default_rng(0))
        assert ch.X_pool.min() == pytest.approx(0.0) and ch.X_pool.max() == pytest.approx(1.0)
        assert np.nanmean(ch.Y_trials) == pytest.approx(0.0, abs=1e-6)   # float32 trials, as in vivo
        assert np.nanstd(ch.Y_trials) == pytest.approx(1.0, rel=1e-6)

    def test_saturation_caps_the_response(self) -> None:
        mu = sn.mean_map(_params(saturation=1.0))
        assert mu.max() < 0.5 + 1.0

    def test_rotation_turns_the_long_axis(self) -> None:
        spot = sn.Hotspot(np.array([5.0, 5.0]), np.array([3.0, 0.5]), 1.0, 0.0)
        turned = sn.Hotspot(spot.center, spot.lengthscale, 1.0, np.pi / 2)
        along_x = np.array([[8.0, 5.0]])
        assert sn.hotspot_drive(along_x, spot)[0] > 0.5 > sn.hotspot_drive(along_x, turned)[0]

    def test_deterministic_under_seed(self) -> None:
        a = sn.generate_neurostim_map(_params(), np.random.default_rng(3)).Y_trials
        b = sn.generate_neurostim_map(_params(), np.random.default_rng(3)).Y_trials
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("bad", [{"baseline": 0.0}, {"saturation": 0.0}, {"n_trials": 1}, {"hotspots": ()}])
    def test_invalid_parameters_raise(self, bad: dict) -> None:
        with pytest.raises(ValueError):
            _params(**bad)


class TestFit:
    """The nominal anchor recovers a known map and reproduces its SNR."""

    def test_fit_recovers_a_generated_map(self) -> None:
        truth = _params(noise_cv=0.3, n_trials=20)
        ch = sn.generate_neurostim_map(truth, np.random.default_rng(0))
        fitted = sn.fit_generator_to_channel(ch, n_hotspots=2)
        mu, fit = sn.mean_map(truth), sn.mean_map(fitted)
        r2 = 1.0 - np.sum((mu - fit) ** 2) / np.sum((mu - mu.mean()) ** 2)
        assert r2 > 0.95

    def test_fitted_twin_matches_the_channel_snr(self) -> None:
        ch = sn.generate_neurostim_map(_params(noise_cv=0.8, n_trials=20), np.random.default_rng(0))
        twin = sn.generate_neurostim_map(sn.fit_generator_to_channel(ch, n_hotspots=2), np.random.default_rng(5))
        assert achieved_snr_db(twin) == pytest.approx(achieved_snr_db(ch), abs=1.0)

    def test_synthetic_channels_keep_identity_and_label_the_source(self) -> None:
        ch = sn.generate_neurostim_map(_params(), np.random.default_rng(0), dataset="nhp", subject=3, emg=2)
        (twin,) = list(sn.synthetic_channels([ch], n_hotspots=1, seed=42))
        assert (twin.dataset, twin.subject, twin.emg) == ("synthetic_nhp", 3, 2)
        assert twin.meta["source_label"] == ch.label

    def test_fit_needs_raw_units(self) -> None:
        ch = sn.generate_neurostim_map(_params(), np.random.default_rng(0))
        bare = ChannelData(ch.dataset, 0, 0, ch.X_pool, ch.Y_trials, ch.y_gt, ch.ch2xy, ch.grid_shape)
        with pytest.raises(ValueError, match="scaler_y"):
            sn.fit_generator_to_channel(bare)


class TestMetaFeatures:
    """Map statistics of the S0 validation figure."""

    def test_morans_i_of_a_smooth_and_a_checkerboard_map(self) -> None:
        coords = _grid()
        smooth = coords[:, 0].astype(float)
        checker = ((coords[:, 0] + coords[:, 1]) % 2).astype(float)
        assert sn.morans_i(smooth, coords) > 0.8
        assert sn.morans_i(checker, coords) == pytest.approx(-1.0)

    def test_heteroscedastic_noise_gives_a_high_mean_sd_correlation(self) -> None:
        ch = sn.generate_neurostim_map(_params(n_trials=30), np.random.default_rng(0))
        feats = sn.meta_features(ch)
        assert set(feats) == {"morans_i", "skewness", "cv", "mean_sd_corr"}
        assert feats["mean_sd_corr"] > 0.8


class TestK1Decoy:
    """A second hotspot at fixed separation; direction and noise shared across levels."""

    def _channel(self) -> ChannelData:
        return sn.generate_neurostim_map(_params(), np.random.default_rng(0), dataset="synthetic_nhp")

    def test_zero_ratio_adds_no_decoy_and_no_basin(self) -> None:
        out = stress.build_knob("k1_decoy").apply(self._channel(), 0.0, np.random.default_rng(0))
        assert len(out.meta["generator"].hotspots) == 2 and not out.meta["decoy_basin"].any()

    def test_decoy_has_the_requested_ratio_and_separation(self) -> None:
        ch = self._channel()
        knob = stress.build_knob("k1_decoy", separation=3.0)
        out = knob.apply(ch, 0.8, np.random.default_rng(0))
        primary, *_, decoy = out.meta["generator"].hotspots
        assert decoy.amplitude == pytest.approx(0.8 * primary.amplitude)
        anchor = ch.ch2xy[int(np.argmax(sn.hotspot_drive(ch.ch2xy, primary)))]
        assert np.linalg.norm(decoy.center - anchor) == pytest.approx(3.0)
        assert out.meta["decoy_basin"].any()
        assert knob.achieved(out)["amplitude_ratio"] == 0.8

    def test_levels_share_direction_and_noise(self) -> None:
        """Common random numbers: only the decoy amplitude differs between two levels."""
        ch = self._channel()
        knob = stress.build_knob("k1_decoy")
        a = knob.apply(ch, 0.5, np.random.default_rng(1))
        b = knob.apply(ch, 0.9, np.random.default_rng(2))
        np.testing.assert_allclose(a.meta["generator"].hotspots[-1].center, b.meta["generator"].hotspots[-1].center)
        np.testing.assert_array_equal(a.meta["decoy_basin"], b.meta["decoy_basin"])

    def test_true_optimum_stays_on_the_primary(self) -> None:
        ch = self._channel()
        out = stress.build_knob("k1_decoy").apply(ch, 0.95, np.random.default_rng(0))
        assert not out.meta["decoy_basin"][out.best_site]

    def test_in_vivo_channel_is_not_applicable(self) -> None:
        ch = self._channel()
        real = ChannelData("nhp", 1, 0, ch.X_pool, ch.Y_trials, ch.y_gt, ch.ch2xy, ch.grid_shape)
        with pytest.raises(KnobNotApplicable, match="Demo 1"):
            stress.build_knob("k1_decoy").apply(real, 0.5, np.random.default_rng(0))

    def test_infeasible_separation_is_not_applicable(self) -> None:
        with pytest.raises(KnobNotApplicable, match="inside the array"):
            stress.build_knob("k1_decoy", separation=50.0).apply(self._channel(), 0.5, np.random.default_rng(0))

    @pytest.mark.parametrize("bad", [{"levels": [1.0]}, {"separation": 0.0}])
    def test_invalid_configuration_raises(self, bad: dict) -> None:
        levels = bad.pop("levels", None)
        with pytest.raises(ValueError):
            stress.build_knob("k1_decoy", levels, **bad)

    def test_decoy_capture_is_reported_per_run(self) -> None:
        from pfns4neurostim.evaluation.bo_runner import run_channel_bo

        out = stress.build_knob("k1_decoy").apply(self._channel(), 0.9, np.random.default_rng(0))
        res = run_channel_bo("gp_naive", out, acq_fn="ei", budget=12, n_init=3, seed=0, device="cpu")
        rec = res.trajectory["best_rec_indices"][-1]
        assert res.row["decoy_capture"] == float(out.meta["decoy_basin"][rec])
