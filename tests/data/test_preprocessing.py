"""Tests for the native preprocessing contract (sprint 2026-09-20, Task 1)."""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.data.preprocessing import (
    DEFAULT_NORMALIZATION,
    NORMALIZATIONS,
    get_normalization,
    preprocess_channel,
    valid_site_mask,
)
from pfns4neurostim.evaluation import metrics


def _subject(n_sites: int = 12, n_emg: int = 2, n_reps: int = 6, seed: int = 0) -> dict:
    """Synthetic raw subject: 2D grid, one dead site at EMG 0, a few flagged trials."""
    rng = np.random.default_rng(seed)
    ch2xy = np.stack([np.arange(n_sites) % 4 + 1, np.arange(n_sites) // 4 + 1], axis=1)  # [N, 2]
    resp = rng.normal(5.0, 2.0, size=(n_sites, n_emg, n_reps))                            # [N, E, R]
    isvalid = np.ones_like(resp, dtype=int)
    isvalid[3, 0, :] = 0            # site 3 has no valid trial at EMG 0
    isvalid[5, 0, 1] = 0            # one flagged trial elsewhere
    isvalid[7, 0, 2] = 0
    resp_mean = np.nan_to_num(np.nanmean(np.where(isvalid == 1, resp, np.nan), axis=-1))
    return {
        "ch2xy": ch2xy,
        "sorted_resp": resp,
        "sorted_respMean": resp_mean,
        "sorted_isvalid": isvalid,
    }


class TestRegistry:
    def test_default_is_the_contract(self) -> None:
        assert DEFAULT_NORMALIZATION == "pfn"
        mode = get_normalization("pfn")
        assert (mode.x, mode.y) == ("minmax", "zscore")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown normalization"):
            get_normalization("nope")


class TestPreprocessChannel:
    def test_contract_units(self) -> None:
        pre = preprocess_channel(_subject(), emg=1)
        assert pre.X_pool.min(axis=0) == pytest.approx(0.0)
        assert pre.X_pool.max(axis=0) == pytest.approx(1.0)
        valid = pre.Y_trials[np.isfinite(pre.Y_trials)]
        assert valid.mean() == pytest.approx(0.0, abs=1e-6)
        assert valid.std() == pytest.approx(1.0, abs=1e-6)

    def test_dead_site_dropped_and_mask_consistent(self) -> None:
        data = _subject()
        pre = preprocess_channel(data, emg=0)
        assert not pre.site_mask[3]
        assert pre.X_pool.shape[0] == int(pre.site_mask.sum()) == pre.y_gt.shape[0]
        np.testing.assert_array_equal(pre.site_mask, valid_site_mask(data, 0))

    def test_invalid_trials_nan_in_trials_finite_in_bank(self) -> None:
        pre = preprocess_channel(_subject(), emg=0)
        assert pre.Y_invalid is not None
        # Every trial slot is exactly one of: valid (finite in Y_trials) or flagged (finite in bank).
        both = np.isfinite(pre.Y_trials) & np.isfinite(pre.Y_invalid)
        neither = ~np.isfinite(pre.Y_trials) & ~np.isfinite(pre.Y_invalid)
        assert not both.any() and not neither.any()
        assert int(np.isfinite(pre.Y_invalid).sum()) == 2

    def test_no_validity_flags(self) -> None:
        data = _subject()
        del data["sorted_isvalid"]
        pre = preprocess_channel(data, emg=0)
        assert pre.Y_invalid is None and pre.site_mask.all()

    def test_all_invalid_raises(self) -> None:
        data = _subject()
        data["sorted_isvalid"][:, 1, :] = 0
        with pytest.raises(RuntimeError, match="no valid trials"):
            preprocess_channel(data, emg=1)

    @pytest.mark.parametrize("name", sorted(NORMALIZATIONS))
    def test_modes_scale_as_declared(self, name: str) -> None:
        pre = preprocess_channel(_subject(), emg=1, normalization=name)
        mode = NORMALIZATIONS[name]
        raw_x = _subject()["ch2xy"].astype(float)
        if mode.x == "raw":
            np.testing.assert_allclose(pre.X_pool, raw_x)
        else:
            assert pre.X_pool.max() == pytest.approx(1.0)
        valid = pre.Y_trials[np.isfinite(pre.Y_trials)]
        if mode.y == "minmax":
            assert valid.min() == pytest.approx(0.0) and valid.max() == pytest.approx(1.0)
        else:
            assert valid.mean() == pytest.approx(0.0, abs=1e-6)

    def test_matches_legacy_reference_in_default_mode(self) -> None:
        """Pin the native implementation to the frozen pre-restructure numbers."""
        legacy = pytest.importorskip("pfns4neurostim.data.legacy_io")
        data = _subject()
        X, Y, _, y_gt, _ = legacy.preprocess_neural_data(data, emg_idx=0, normalization="pfn")
        pre = preprocess_channel(data, emg=0)
        np.testing.assert_allclose(pre.X_pool, X, atol=1e-9)
        np.testing.assert_allclose(pre.Y_trials, Y, atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(pre.y_gt, y_gt, atol=1e-6)

    def test_regret_identical_under_both_y_scalers(self) -> None:
        """Locks in the affine-invariance argument: the y scaler cannot move regret."""
        data = _subject()
        z = preprocess_channel(data, emg=1, normalization="pfn")
        m = preprocess_channel(data, emg=1, normalization="minmax_x_minmax_y")
        queried, rec = [0, 4, 9], 4
        rz = metrics.regret_metrics(z.y_gt, queried, recommended_index=rec)
        rm = metrics.regret_metrics(m.y_gt, queried, recommended_index=rec)
        for key in rz:
            assert rz[key] == pytest.approx(rm[key])


class TestConfigWiring:
    def test_dataset_config_rejects_unknown_normalization(self) -> None:
        from pfns4neurostim.config import DatasetConfig

        with pytest.raises(ValueError, match="Unknown normalization"):
            DatasetConfig(name="nhp", subjects=(1,), normalization="bogus")

    def test_override_reaches_resolved_config(self) -> None:
        from pfns4neurostim.config import load_experiment_config, resolved_dict

        cfg = load_experiment_config(
            "configs/experiment/preproc_ab_nhp.yaml", ["dataset.normalization=raw_x_zscore_y"]
        )
        assert cfg.dataset.normalization == "raw_x_zscore_y"
        assert resolved_dict(cfg)["dataset"]["normalization"] == "raw_x_zscore_y"
