"""Unit tests for the 5d_rat dataset integration in src/utils/data_utils.py.

Covers:
- load_data('5d_rat', m_i) for the subjects with real data files (m_i 0-2)
- preprocess_neural_data() / augment_maps() on the 5D (non-grid) search space
- _apply_aug_transform() 2D-only guard for h_flip/v_flip/d_flip
- 5d_rat split-constant sanity (TRAIN ∪ HELD_OUT == ALL, no overlap)
"""
import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


@pytest.fixture(autouse=True)
def _chdir_project_root(monkeypatch):
    """load_data() resolves './data' relative to the working directory."""
    monkeypatch.chdir(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# load_data('5d_rat', m_i)
# ---------------------------------------------------------------------------

class TestLoadData5dRat:
    """Tests for load_data('5d_rat', m_i) on subjects with real .mat files."""

    @pytest.mark.parametrize("m_i", [0, 1, 2, 3, 4, 5])
    def test_returns_expected_keys(self, m_i):
        from utils.data_utils import load_data
        data = load_data('5d_rat', m_i)
        expected_keys = {
            'emgs', 'nChan', 'sorted_isvalid', 'sorted_resp', 'sorted_respMean',
            'sorted_respSD', 'ch2xy', 'grid_shape', 'DimSearchSpace',
        }
        assert expected_keys.issubset(data.keys())

    @pytest.mark.parametrize("m_i", [0, 1, 2, 3, 4, 5])
    def test_grid_shape_is_none(self, m_i):
        from utils.data_utils import load_data
        data = load_data('5d_rat', m_i)
        assert data['grid_shape'] is None

    @pytest.mark.parametrize("m_i", [0, 1, 2, 3, 4, 5])
    def test_ch2xy_is_5d(self, m_i):
        from utils.data_utils import load_data
        data = load_data('5d_rat', m_i)
        assert data['ch2xy'].ndim == 2
        assert data['ch2xy'].shape[1] == 5

    @pytest.mark.parametrize("m_i", [0, 1, 2, 3, 4, 5])
    def test_sorted_resp_shape_consistent(self, m_i):
        from utils.data_utils import load_data
        data = load_data('5d_rat', m_i)
        n_cond, n_emgs, n_reps = data['sorted_resp'].shape
        assert data['ch2xy'].shape[0] == n_cond
        assert data['sorted_respMean'].shape == (n_cond, n_emgs)
        assert data['sorted_respSD'].shape == (n_cond, n_emgs)
        assert data['sorted_isvalid'].shape == (n_cond, n_emgs, n_reps)
        assert len(data['emgs']) == n_emgs

    @pytest.mark.parametrize("m_i", [0, 1, 2, 3, 4, 5])
    def test_dim_search_space_equals_n_cond(self, m_i):
        from utils.data_utils import load_data
        data = load_data('5d_rat', m_i)
        assert data['DimSearchSpace'] == data['sorted_resp'].shape[0]
        assert data['nChan'] == data['sorted_resp'].shape[0]

    @pytest.mark.parametrize("m_i, expected_n_emgs", [(0, 1), (1, 1), (2, 4), (3, 4), (4, 5), (5, 3)])
    def test_emg_count_matches_valid_idx(self, m_i, expected_n_emgs):
        """Per-subject EMG validity filtering yields the expected channel count."""
        from utils.data_utils import load_data
        data = load_data('5d_rat', m_i)
        assert len(data['emgs']) == expected_n_emgs
        assert data['sorted_resp'].shape[1] == expected_n_emgs

    def test_unknown_dataset_type_raises(self):
        from utils.data_utils import load_data
        with pytest.raises(ValueError):
            load_data('not_a_real_dataset', 0)


# ---------------------------------------------------------------------------
# preprocess_neural_data / augment_maps on 5D data
# ---------------------------------------------------------------------------

class TestPipeline5dRat:
    """Tests that the generic [N, D] pipeline runs on 5D 5d_rat data."""

    def test_preprocess_neural_data_produces_5d_x(self):
        from utils.data_utils import load_data, preprocess_neural_data
        data = load_data('5d_rat', 1)
        X_train, Y_train, X_test, Y_test, scaler_y = preprocess_neural_data(data, emg_idx=0)
        assert X_train.shape[1] == 5
        assert X_test.shape[1] == 5
        assert len(Y_test) == X_test.shape[0]

    def test_augment_maps_runs_with_none_and_y_shift(self):
        from utils.data_utils import load_data, augment_maps
        data = load_data('5d_rat', 1)
        pairs = augment_maps(
            data, emg_idx=0, n_augmentations=3, seed=42,
            aug_transforms=('none', 'y_shift'),
        )
        assert len(pairs) == 3
        for X, y in pairs:
            assert X.shape[1] == 5
            assert len(y) == X.shape[0]


# ---------------------------------------------------------------------------
# _apply_aug_transform 2D-only guard
# ---------------------------------------------------------------------------

class TestApplyAugTransformDimGuard:
    """_apply_aug_transform raises for 2D-only spatial flips on non-2D X."""

    @pytest.mark.parametrize("transform", ["h_flip", "v_flip", "d_flip"])
    def test_raises_for_5d_input(self, transform):
        from utils.data_utils import _apply_aug_transform
        rng = np.random.RandomState(0)
        X = rng.rand(10, 5)
        y = rng.randn(10)
        with pytest.raises(ValueError):
            _apply_aug_transform(X, y, transform, rng)

    @pytest.mark.parametrize("transform", ["none", "y_shift"])
    def test_does_not_raise_for_5d_input(self, transform):
        from utils.data_utils import _apply_aug_transform
        rng = np.random.RandomState(0)
        X = rng.rand(10, 5)
        y = rng.randn(10)
        X_out, y_out = _apply_aug_transform(X, y, transform, rng)
        assert X_out.shape == X.shape
        assert y_out.shape == y.shape

    @pytest.mark.parametrize("transform", ["h_flip", "v_flip", "d_flip"])
    def test_does_not_raise_for_2d_input(self, transform):
        from utils.data_utils import _apply_aug_transform
        rng = np.random.RandomState(0)
        X = rng.rand(10, 2)
        y = rng.randn(10)
        X_out, y_out = _apply_aug_transform(X, y, transform, rng)
        assert X_out.shape == X.shape
        assert y_out.shape == y.shape


# ---------------------------------------------------------------------------
# Split-constant sanity
# ---------------------------------------------------------------------------

class TestSplitConstants5dRat:
    """5d_rat split constants are present and self-consistent."""

    def test_keys_present(self):
        from utils.data_utils import HELD_OUT_SUBJECTS, TRAIN_SUBJECTS, ALL_SUBJECTS
        assert '5d_rat' in HELD_OUT_SUBJECTS
        assert '5d_rat' in TRAIN_SUBJECTS
        assert '5d_rat' in ALL_SUBJECTS

    def test_train_and_held_out_partition_all(self):
        from utils.data_utils import HELD_OUT_SUBJECTS, TRAIN_SUBJECTS, ALL_SUBJECTS
        train = set(TRAIN_SUBJECTS['5d_rat'])
        held_out = set(HELD_OUT_SUBJECTS['5d_rat'])
        all_subj = set(ALL_SUBJECTS['5d_rat'])
        assert train.isdisjoint(held_out)
        assert train | held_out == all_subj
