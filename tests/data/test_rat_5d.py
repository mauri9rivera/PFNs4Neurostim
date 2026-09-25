"""Tests for the native 5d_rat loader (cohort ``noOutliers-2026-09-25``).

Two kinds of test live here. The **cohort contract** tests need no data files: they
assert the properties that make a result traceable (a cohort stamp exists, the subject
table is the index, the frozen loader no longer serves this dataset). The **loader**
tests read ``data/5d_rat`` and are skipped where it is absent, as on CI.

Replaces ``tests/test_5d_rat_data.py``, which tested the branch of
``legacy_io.load_data`` that this cohort retired.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.io

from pfns4neurostim.data import splits
from pfns4neurostim.data.loaders import rat_5d

DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)
SUBJECT_IDS = tuple(range(len(rat_5d.SUBJECTS)))


def _require_data(subject: int) -> str:
    """Skip unless one subject's raw file is present.

    Args:
        subject: Subject index.

    Returns:
        The subject's directory.
    """
    directory = rat_5d.subject_dir(subject, DATA_ROOT)
    if not os.path.isfile(os.path.join(directory, rat_5d._MAT_NAME)):
        pytest.skip(f"data/5d_rat/{rat_5d.SUBJECTS[subject].key} not available")
    return directory


class TestCohortContract:
    """Properties that make a 5d_rat result traceable, checked without data files."""

    def test_the_cohort_is_stamped(self) -> None:
        """Without a stamp, results from two cohorts are indistinguishable."""
        assert rat_5d.COHORT
        from pfns4neurostim.data import loaders

        assert loaders.data_cohort("5d_rat") == rat_5d.COHORT
        assert loaders.data_cohort("nhp") is None

    def test_subject_table_defines_the_indices(self) -> None:
        """rData03 left the cohort, so 5d_rat has five subjects, not six."""
        assert [s.key for s in rat_5d.SUBJECTS] == [
            "rCer1.5", "BCI00", "rCer1.12", "rCer1.14", "rCer1.15"
        ]
        assert splits.ALL_SUBJECTS["5d_rat"] == SUBJECT_IDS

    def test_an_index_outside_the_cohort_names_the_cohort(self) -> None:
        """A config left on the old six-subject list must say what changed."""
        with pytest.raises(IndexError, match="not in cohort"):
            rat_5d.subject_dir(len(rat_5d.SUBJECTS), DATA_ROOT)

    def test_the_frozen_loader_no_longer_serves_5d_rat(self) -> None:
        """One loader per dataset: legacy_io must not keep a second 5d_rat path."""
        from pfns4neurostim.data import legacy_io

        assert not hasattr(legacy_io, "_sort_valid_5drat_reps")
        with pytest.raises(ValueError, match="5d_rat moved"):
            legacy_io.load_data("5d_rat", 0)

    def test_a_missing_file_points_at_the_archive(self) -> None:
        with pytest.raises(FileNotFoundError, match="datasets_noOutliers"):
            rat_5d.load_subject(0, data_root=os.path.join(DATA_ROOT, "does-not-exist"))


class TestLoader:
    """The loader on the real files."""

    @pytest.mark.parametrize("subject", SUBJECT_IDS)
    def test_shapes_are_consistent(self, subject: int) -> None:
        _require_data(subject)
        data = rat_5d.load_subject(subject, DATA_ROOT)
        n_cond, n_emgs, n_reps = data["sorted_resp"].shape
        assert data["ch2xy"].shape == (n_cond, 5)
        assert data["sorted_isvalid"].shape == (n_cond, n_emgs, n_reps)
        assert data["sorted_respMean"].shape == (n_cond, n_emgs)
        assert data["sorted_respSD"].shape == (n_cond, n_emgs)
        assert len(data["emgs"]) == n_emgs
        assert data["nChan"] == data["DimSearchSpace"] == n_cond
        assert data["grid_shape"] is None
        assert data["data_cohort"] == rat_5d.COHORT

    @pytest.mark.parametrize("subject", SUBJECT_IDS)
    def test_kept_emgs_are_exactly_the_labs_valid_ones(self, subject: int) -> None:
        """``flag_valid_emg`` replaces the hand-maintained valid_emg_idx lists."""
        directory = _require_data(subject)
        mat = scipy.io.loadmat(os.path.join(directory, rat_5d._MAT_NAME))
        flags = np.asarray(mat["flag_valid_emg"]).ravel().astype(bool)
        names = rat_5d.SUBJECTS[subject].emg_names

        data = rat_5d.load_subject(subject, DATA_ROOT)
        assert data["emgs"] == [n for n, keep in zip(names, flags) if keep]

    @pytest.mark.parametrize("subject", SUBJECT_IDS)
    def test_validity_is_valid_own_equal_to_one(self, subject: int) -> None:
        """Codes 0 (noisy baseline) and -1 (response outlier) are both invalid."""
        directory = _require_data(subject)
        mat = scipy.io.loadmat(os.path.join(directory, rat_5d._MAT_NAME))
        flags = np.asarray(mat["flag_valid_emg"]).ravel().astype(bool)
        expected = (mat["valid_own"][np.flatnonzero(flags)] == 1).transpose(1, 0, 2)

        data = rat_5d.load_subject(subject, DATA_ROOT)
        assert np.array_equal(data["sorted_isvalid"].astype(bool), expected)

    @pytest.mark.parametrize("subject", SUBJECT_IDS)
    def test_response_axes_are_not_transposed(self, subject: int) -> None:
        """Advanced indexing on two separated axes silently reorders the result."""
        directory = _require_data(subject)
        mat = scipy.io.loadmat(os.path.join(directory, rat_5d._MAT_NAME))
        flags = np.asarray(mat["flag_valid_emg"]).ravel().astype(bool)
        peak = mat["emg_response"][..., 0]                       # [R, E_raw, C]

        data = rat_5d.load_subject(subject, DATA_ROOT)
        for e, raw_e in enumerate(np.flatnonzero(flags)):
            assert np.allclose(data["sorted_resp"][:, e, :], peak[:, raw_e, :].T)

    @pytest.mark.parametrize("subject", SUBJECT_IDS)
    def test_mean_ignores_invalid_repetitions(self, subject: int) -> None:
        _require_data(subject)
        data = rat_5d.load_subject(subject, DATA_ROOT)
        masked = np.ma.masked_where(data["sorted_isvalid"] == 0, data["sorted_resp"])
        assert np.allclose(
            data["sorted_respMean"], np.ma.filled(masked.mean(axis=-1), 0.0)
        )
        # Every site keeps at least one valid repetition, or the channel would have a
        # fill value of 0.0 standing where a measurement should be.
        assert (data["sorted_isvalid"].sum(axis=-1) > 0).all()


class TestChannelIntegration:
    """The 5-D search space through the shared channel pipeline."""

    def test_channel_carries_the_cohort_and_a_5d_pool(self) -> None:
        _require_data(0)
        from pfns4neurostim.data.channels import load_channel

        channel = load_channel("5d_rat", 0, 0, data_root=DATA_ROOT)
        assert channel.X_pool.shape[1] == 5
        assert channel.X_pool.min() >= 0.0 and channel.X_pool.max() <= 1.0
        assert channel.Y_trials.shape[0] == channel.X_pool.shape[0]
        assert channel.meta["data_cohort"] == rat_5d.COHORT
        assert channel.meta["subject_key"] == "rCer1.5"

    def test_the_cohort_enters_the_cell_identity(self) -> None:
        """Replacing the raw files must invalidate this dataset's cached cells."""
        _require_data(0)
        from pfns4neurostim.config import AcquisitionConfig, load_experiment_config
        from pfns4neurostim.data.channels import load_channel
        from pfns4neurostim.experiments._cells import cell_identity

        cfg = load_experiment_config("configs/experiment/hyp_a_5d_rat.yaml")
        channel = load_channel("5d_rat", 0, 0, data_root=DATA_ROOT)
        identity = cell_identity(
            cfg, channel, "gp_naive", AcquisitionConfig(type="ts_marginal"),
            experiment="bo_benchmark", rep=0, seed=0, budget=10,
        )
        assert identity["data_cohort"] == rat_5d.COHORT

    def test_nhp_identity_has_no_cohort_key(self) -> None:
        """Adding the stamp must not invalidate the datasets that did not change."""
        from pfns4neurostim.config import AcquisitionConfig, load_experiment_config
        from pfns4neurostim.data.channels import ChannelData
        from pfns4neurostim.experiments._cells import cell_identity

        rng = np.random.default_rng(0)
        coords = np.stack(np.meshgrid(np.arange(4), np.arange(4)), -1).reshape(-1, 2)
        channel = ChannelData(
            dataset="nhp", subject=0, emg=0,
            X_pool=coords / 3.0, Y_trials=rng.normal(size=(16, 3)),
            y_gt=rng.normal(size=16), ch2xy=coords, grid_shape=(4, 4),
        )
        cfg = load_experiment_config("configs/experiment/hyp_a_nhp.yaml")
        identity = cell_identity(
            cfg, channel, "gp_naive", AcquisitionConfig(type="ts_marginal"),
            experiment="bo_benchmark", rep=0, seed=0, budget=10,
        )
        assert "data_cohort" not in identity
