"""Unit tests for core functions in src/utils/data_utils.py.

Does NOT re-test generate_experiment_tag() or create_run_dir() (covered in
test_step1_tag_config.py).

Covers:
- _topographic_metadata()
- augment_maps()
- preprocess_neural_data()
- write_run_config()
- save_results() / load_results() round-trip
- aggregate_results()
"""
import os
import pickle
import sys
import tempfile
from typing import Dict

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_subject_data(
    n_chan: int = 10,
    n_emgs: int = 3,
    n_reps: int = 5,
    seed: int = 0,
) -> Dict:
    """Build a minimal subject_data dict without loading real .mat files.

    Args:
        n_chan: Number of electrode channels.
        n_emgs: Number of EMG channels.
        n_reps: Number of repetitions.
        seed: Random seed.

    Returns:
        Dict matching the structure returned by load_data().
    """
    rng = np.random.RandomState(seed)
    ch2xy = np.zeros((n_chan, 2), dtype=np.float64)
    ch2xy[:, 0] = np.arange(n_chan) % 5
    ch2xy[:, 1] = np.arange(n_chan) // 5
    sorted_respMean = rng.randn(n_chan, n_emgs).astype(np.float64)
    sorted_respSD = np.abs(rng.randn(n_chan, n_emgs)).astype(np.float64) + 0.1
    sorted_isvalid = np.ones((n_chan, n_emgs, n_reps), dtype=np.int32)
    sorted_resp = sorted_respMean[:, :, np.newaxis] + rng.randn(n_chan, n_emgs, n_reps) * 0.1
    return {
        'ch2xy': ch2xy,
        'sorted_respMean': sorted_respMean,
        'sorted_respSD': sorted_respSD,
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
    }


def _make_results_dict() -> Dict:
    """Build a minimal results_dict for save/load round-trip tests."""
    rng = np.random.RandomState(42)
    return {
        'gp': [{
            'dataset': 'nhp',
            'subject': 1,
            'emg': 0,
            'r2': rng.randn(5).tolist(),
            'times': [0.1] * 5,
            'values': rng.rand(5, 10).tolist(),
            'y_test': rng.rand(20),
        }],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def subject_data():
    """Standard synthetic subject_data fixture."""
    return _make_subject_data()


@pytest.fixture
def results_dict():
    """Standard minimal results dict fixture."""
    return _make_results_dict()


# ---------------------------------------------------------------------------
# _topographic_metadata
# ---------------------------------------------------------------------------

class TestTopographicMetadata:
    """Tests for _topographic_metadata() pure utility function."""

    def test_float_ch2xy_is_cast_to_int64(self):
        """Float coordinates are rounded and cast to int64."""
        from utils.data_utils import _topographic_metadata
        ch2xy = np.array([[1.6, 2.4], [3.1, 4.9]], dtype=np.float64)
        maps = np.zeros((5, 6, 1))
        ch2xy_int, _ = _topographic_metadata(ch2xy, maps)
        assert ch2xy_int.dtype == np.int64

    def test_grid_shape_matches_map_first_two_dims(self):
        """grid_shape equals (maps.shape[0], maps.shape[1]) as ints."""
        from utils.data_utils import _topographic_metadata
        ch2xy = np.array([[0.0, 0.0]])
        maps = np.zeros((8, 7, 3))
        _, grid_shape = _topographic_metadata(ch2xy, maps)
        assert grid_shape == (8, 7)
        assert isinstance(grid_shape[0], int)
        assert isinstance(grid_shape[1], int)

    def test_shape_invariant_ch2xy_preserved(self):
        """Returned ch2xy_int has the same shape as input ch2xy."""
        from utils.data_utils import _topographic_metadata
        ch2xy = np.random.RandomState(0).rand(15, 2) * 10
        maps = np.zeros((10, 10))
        ch2xy_int, _ = _topographic_metadata(ch2xy, maps)
        assert ch2xy_int.shape == ch2xy.shape

    def test_already_integer_input_passes_through(self):
        """Integer input is returned unchanged (apart from dtype cast)."""
        from utils.data_utils import _topographic_metadata
        ch2xy = np.array([[2, 3], [4, 1]], dtype=np.int32)
        maps = np.zeros((6, 6))
        ch2xy_int, _ = _topographic_metadata(ch2xy, maps)
        np.testing.assert_array_equal(ch2xy_int, ch2xy)
        assert ch2xy_int.dtype == np.int64

    def test_rounding_applied_correctly(self):
        """Values are rounded before cast: 0.5 → 0 or 1 (numpy rounds to even)."""
        from utils.data_utils import _topographic_metadata
        ch2xy = np.array([[0.4, 1.6]], dtype=np.float64)
        maps = np.zeros((3, 3))
        ch2xy_int, _ = _topographic_metadata(ch2xy, maps)
        # rint(0.4)=0, rint(1.6)=2
        assert ch2xy_int[0, 0] == 0
        assert ch2xy_int[0, 1] == 2


# ---------------------------------------------------------------------------
# augment_maps
# ---------------------------------------------------------------------------

class TestAugmentMaps:
    """Tests for augment_maps() data augmentation."""

    def test_returns_list_of_length_n_augmentations(self, subject_data):
        """Returns exactly n_augmentations tuples."""
        from utils.data_utils import augment_maps
        pairs = augment_maps(subject_data, emg_idx=0, n_augmentations=7, seed=42)
        assert len(pairs) == 7

    def test_each_element_is_xy_tuple(self, subject_data):
        """Each element is a (X, y) tuple."""
        from utils.data_utils import augment_maps
        pairs = augment_maps(subject_data, emg_idx=0, n_augmentations=3, seed=42)
        for X, y in pairs:
            assert X.ndim == 2
            assert X.shape[1] == 2
            assert len(y) == len(X)

    def test_x_values_in_unit_range(self, subject_data):
        """MinMax-scaled X values lie in [0, 1]."""
        from utils.data_utils import augment_maps
        pairs = augment_maps(subject_data, emg_idx=0, n_augmentations=5, seed=42)
        for X, _ in pairs:
            assert np.all(X >= 0.0 - 1e-9)
            assert np.all(X <= 1.0 + 1e-9)

    def test_deterministic_with_same_seed(self, subject_data):
        """Same seed produces identical augmented pairs."""
        from utils.data_utils import augment_maps
        pairs1 = augment_maps(subject_data, emg_idx=0, n_augmentations=4, seed=42)
        pairs2 = augment_maps(subject_data, emg_idx=0, n_augmentations=4, seed=42)
        for (X1, y1), (X2, y2) in zip(pairs1, pairs2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_produce_different_outputs(self, subject_data):
        """Different seeds produce different augmented pairs."""
        from utils.data_utils import augment_maps
        pairs1 = augment_maps(subject_data, emg_idx=0, n_augmentations=4, seed=1)
        pairs2 = augment_maps(subject_data, emg_idx=0, n_augmentations=4, seed=99)
        # At least one pair should differ
        any_diff = any(
            not np.allclose(y1, y2) for (_, y1), (_, y2) in zip(pairs1, pairs2)
        )
        assert any_diff

    def test_zero_augmentations_returns_empty_list(self, subject_data):
        """n_augmentations=0 returns an empty list."""
        from utils.data_utils import augment_maps
        pairs = augment_maps(subject_data, emg_idx=0, n_augmentations=0, seed=42)
        assert pairs == []

    def test_channels_with_all_zero_isvalid_excluded(self):
        """Channels where sorted_isvalid is all zero are dropped from coords."""
        from utils.data_utils import augment_maps
        n_chan, n_emgs, n_reps = 10, 3, 5
        data = _make_subject_data(n_chan=n_chan, n_emgs=n_emgs, n_reps=n_reps)
        # Invalidate first 3 channels for emg_idx=0
        data['sorted_isvalid'][:3, 0, :] = 0
        pairs = augment_maps(data, emg_idx=0, n_augmentations=2, seed=42)
        # Should have fewer X rows than n_chan
        for X, y in pairs:
            assert len(X) <= n_chan - 3


# ---------------------------------------------------------------------------
# preprocess_neural_data
# ---------------------------------------------------------------------------

class TestPreprocessNeuralData:
    """Tests for preprocess_neural_data() preprocessing pipeline."""

    def test_returns_five_tuple(self, subject_data):
        """Returns a 5-tuple (X_train, Y_train, X_test, Y_test, scaler_y)."""
        from utils.data_utils import preprocess_neural_data
        result = preprocess_neural_data(subject_data, emg_idx=0)
        assert len(result) == 5

    def test_x_train_equals_x_test(self, subject_data):
        """X_train and X_test are the same array (API compatibility)."""
        from utils.data_utils import preprocess_neural_data
        X_train, Y_train, X_test, Y_test, _ = preprocess_neural_data(subject_data, emg_idx=0)
        np.testing.assert_array_equal(X_train, X_test)

    def test_y_train_shape(self, subject_data):
        """Y_train has shape (n_valid_sites, n_reps)."""
        from utils.data_utils import preprocess_neural_data
        n_reps = subject_data['sorted_isvalid'].shape[2]
        X_train, Y_train, _, _, _ = preprocess_neural_data(subject_data, emg_idx=0)
        assert Y_train.ndim == 2
        assert Y_train.shape[1] == n_reps

    def test_y_test_shape(self, subject_data):
        """Y_test has shape (n_valid_sites,)."""
        from utils.data_utils import preprocess_neural_data
        X_train, _, _, Y_test, _ = preprocess_neural_data(subject_data, emg_idx=0)
        assert Y_test.ndim == 1
        assert len(Y_test) == X_train.shape[0]

    def test_x_train_in_unit_range_for_pfn_normalization(self, subject_data):
        """X_train values are in [0, 1] for normalization='pfn'."""
        from utils.data_utils import preprocess_neural_data
        X_train, _, _, _, _ = preprocess_neural_data(subject_data, emg_idx=0, normalization='pfn')
        assert np.all(X_train >= 0.0 - 1e-9)
        assert np.all(X_train <= 1.0 + 1e-9)

    def test_raises_runtime_error_when_all_sites_invalid(self):
        """Raises RuntimeError with 'no valid trials' when all sites are invalid."""
        from utils.data_utils import preprocess_neural_data
        data = _make_subject_data()
        data['sorted_isvalid'][:, 0, :] = 0  # all channels invalid for emg 0
        with pytest.raises(RuntimeError, match="no valid trials"):
            preprocess_neural_data(data, emg_idx=0)

    def test_nan_in_y_train_at_invalid_positions(self):
        """NaN in Y_train at positions where sorted_isvalid == 0."""
        from utils.data_utils import preprocess_neural_data
        data = _make_subject_data(n_chan=10, n_reps=5)
        # Invalidate rep 0 for all channels at emg_idx=1
        data['sorted_isvalid'][:, 1, 0] = 0
        _, Y_train, _, _, _ = preprocess_neural_data(data, emg_idx=1)
        # Column 0 of Y_train should be all NaN
        assert np.all(np.isnan(Y_train[:, 0]))

    def test_y_test_values_are_finite(self, subject_data):
        """Y_test contains only finite values for fully-valid data."""
        from utils.data_utils import preprocess_neural_data
        _, _, _, Y_test, _ = preprocess_neural_data(subject_data, emg_idx=0)
        assert np.all(np.isfinite(Y_test))


# ---------------------------------------------------------------------------
# write_run_config
# ---------------------------------------------------------------------------

class TestWriteRunConfig:
    """Tests for write_run_config() config serialization."""

    def test_creates_config_yaml_in_run_dir(self, tmp_path):
        """Creates config.yaml inside run_dir."""
        from utils.data_utils import write_run_config
        run_dir = str(tmp_path)
        write_run_config(run_dir, {'epochs': 50, 'lr': 1e-5})
        assert os.path.isfile(os.path.join(run_dir, 'config.yaml'))

    def test_round_trips_config_dict(self, tmp_path):
        """Loaded YAML matches the written config dict."""
        from utils.data_utils import write_run_config
        config = {'epochs': 50, 'dataset': 'nhp', 'n_reps': 30}
        path = write_run_config(str(tmp_path), config)
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded['epochs'] == 50
        assert loaded['dataset'] == 'nhp'
        assert loaded['n_reps'] == 30

    def test_float_scientific_notation_survives_round_trip(self, tmp_path):
        """Floats like 1e-5 survive YAML round-trip."""
        from utils.data_utils import write_run_config
        config = {'lr': 1e-5, 'weight_decay': 1e-4}
        path = write_run_config(str(tmp_path), config)
        with open(path) as f:
            loaded = yaml.safe_load(f)
        np.testing.assert_allclose(loaded['lr'], 1e-5, rtol=1e-6)
        np.testing.assert_allclose(loaded['weight_decay'], 1e-4, rtol=1e-6)

    def test_returns_path_string(self, tmp_path):
        """Returns the path to the written file as a string."""
        from utils.data_utils import write_run_config
        path = write_run_config(str(tmp_path), {'a': 1})
        assert isinstance(path, str)
        assert path.endswith('config.yaml')


# ---------------------------------------------------------------------------
# save_results / load_results round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadResultsRoundTrip:
    """Tests for save_results() and load_results() persistence."""

    def test_returns_pkl_and_csv_tuple(self, tmp_path, results_dict):
        """save_results returns a (pkl_path, csv_path) tuple."""
        from utils.data_utils import save_results
        result = save_results(
            results_dict, 'optimization', output_dir=str(tmp_path)
        )
        assert len(result) == 2
        pkl_path, csv_path = result
        assert pkl_path.endswith('.pkl')
        assert csv_path.endswith('.csv')

    def test_both_files_exist_after_save(self, tmp_path, results_dict):
        """Both pkl and csv files are written to disk."""
        from utils.data_utils import save_results
        pkl_path, csv_path = save_results(
            results_dict, 'optimization', output_dir=str(tmp_path)
        )
        assert os.path.isfile(pkl_path)
        assert os.path.isfile(csv_path)

    def test_load_results_returns_same_model_keys(self, tmp_path, results_dict):
        """load_results returns dict with same model keys as saved."""
        from utils.data_utils import save_results, load_results
        pkl_path, _ = save_results(
            results_dict, 'optimization', output_dir=str(tmp_path)
        )
        loaded = load_results(pkl_path)
        assert set(loaded.keys()) & set(results_dict.keys()) == set(results_dict.keys())

    def test_csv_has_model_column(self, tmp_path, results_dict):
        """Summary CSV contains a 'model' column."""
        import csv as csv_mod
        from utils.data_utils import save_results
        _, csv_path = save_results(
            results_dict, 'optimization', output_dir=str(tmp_path)
        )
        with open(csv_path, newline='') as f:
            reader = csv_mod.DictReader(f)
            header = reader.fieldnames
        assert 'model' in header

    def test_csv_has_mean_final_regret_for_optimization_with_values(self, tmp_path, results_dict):
        """CSV has mean_final_regret column for optimization type with values key."""
        import csv as csv_mod
        from utils.data_utils import save_results
        _, csv_path = save_results(
            results_dict, 'optimization', output_dir=str(tmp_path)
        )
        with open(csv_path, newline='') as f:
            reader = csv_mod.DictReader(f)
            header = reader.fieldnames
        assert 'mean_final_regret' in header

    def test_metadata_stored_in_pickle(self, tmp_path, results_dict):
        """metadata argument is stored as _metadata key in pickle."""
        from utils.data_utils import save_results, load_results
        metadata = {'family': 'optimization', 'dataset': 'nhp'}
        pkl_path, _ = save_results(
            results_dict, 'optimization',
            output_dir=str(tmp_path), metadata=metadata,
        )
        loaded = load_results(pkl_path)
        assert '_metadata' in loaded
        assert loaded['_metadata']['family'] == 'optimization'

    def test_different_tags_produce_different_filenames(self, tmp_path, results_dict):
        """Different tag arguments yield different file names."""
        from utils.data_utils import save_results
        pkl1, _ = save_results(
            results_dict, 'optimization', output_dir=str(tmp_path), tag='run_a'
        )
        pkl2, _ = save_results(
            results_dict, 'optimization', output_dir=str(tmp_path), tag='run_b'
        )
        assert pkl1 != pkl2


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------

class TestAggregateResults:
    """Tests for aggregate_results() aggregation logic."""

    def test_returns_empty_dataframe_when_runs_dir_missing(self, tmp_path):
        """Returns empty DataFrame when runs_dir doesn't exist."""
        import pandas as pd
        from utils.data_utils import aggregate_results
        missing_dir = os.path.join(str(tmp_path), 'nonexistent')
        df = aggregate_results('optimization', 'nhp', 'optimization', runs_dir=missing_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_returns_empty_dataframe_when_no_matching_dirs(self, tmp_path):
        """Returns empty DataFrame when no {dataset}-{family}-* dirs exist."""
        import pandas as pd
        from utils.data_utils import aggregate_results
        # Create a dir that does NOT match the prefix
        os.makedirs(os.path.join(str(tmp_path), 'rat-other-12345'), exist_ok=True)
        df = aggregate_results('optimization', 'nhp', 'optimization', runs_dir=str(tmp_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_raises_value_error_for_invalid_result_type(self, tmp_path):
        """Raises ValueError for unrecognised result_type."""
        from utils.data_utils import aggregate_results
        with pytest.raises(ValueError, match="result_type"):
            aggregate_results('optimization', 'nhp', 'bad_type', runs_dir=str(tmp_path))

    def test_returns_dataframe_with_tag_and_family_columns(self, tmp_path):
        """Returns DataFrame with 'tag' and 'family' columns when valid run exists."""
        import pandas as pd
        from utils.data_utils import save_results, aggregate_results
        # Create a fake run directory matching the expected prefix
        tag = 'nhp-optimization-ab123'
        run_dir = os.path.join(str(tmp_path), tag)
        results_subdir = os.path.join(run_dir, 'results')
        os.makedirs(results_subdir, exist_ok=True)

        # Build a minimal results pkl with the correct filename pattern
        rng = np.random.RandomState(0)
        results_dict = {
            'gp': [{
                'dataset': 'nhp', 'subject': 1, 'emg': 0,
                'r2': [0.5, 0.6, 0.7],
                'times': [0.1, 0.1, 0.1],
                'values': rng.rand(3, 5).tolist(),
                'y_test': rng.rand(10),
            }]
        }
        pkl_path = os.path.join(results_subdir, 'nhp_optimization_ab123_20260101_000000.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(results_dict, f)

        df = aggregate_results('optimization', 'nhp', 'optimization', runs_dir=str(tmp_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'tag' in df.columns
        assert 'family' in df.columns
