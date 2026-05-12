"""Integration tests for the BO pipeline using only synthetic data.

End-to-end smoke tests: synthetic data → BO loop → save → load → verify.
No GPU, no real data files, no model loading.

Covers:
- TestBoPipelineIntegration.test_full_pipeline_no_data_loading
- TestBoPipelineIntegration.test_save_load_round_trip
- TestBoPipelineIntegration.test_write_run_config_round_trip
- TestBoPipelineIntegration.test_preprocess_neural_data_runs_on_synthetic
"""
import os
import sys
from typing import Dict, List, Tuple

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
# Dummy surrogate (no GPU / model loading)
# ---------------------------------------------------------------------------

class _DummySurrogate:
    """Constant-prediction surrogate for integration smoke tests.

    Conforms to the SurrogateModel protocol without requiring any ML framework.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store mean of observed y as constant prediction."""
        self._mean = float(y.mean()) if len(y) > 0 else 0.0

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return constant mean and unit std."""
        n = X.shape[0]
        return np.full(n, self._mean), np.ones(n) * 0.1

    def predict_ucb(
        self, X: np.ndarray, kappa: float, t: int, n_steps: int
    ) -> np.ndarray:
        """Return mean + kappa * std."""
        mean, std = self.predict(X)
        return mean + kappa * std


# ---------------------------------------------------------------------------
# Synthetic subject_data helper (shared with test_data_utils_core.py)
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
    sorted_resp = (
        sorted_respMean[:, :, np.newaxis]
        + rng.randn(n_chan, n_emgs, n_reps) * 0.1
    )
    return {
        'ch2xy': ch2xy,
        'sorted_respMean': sorted_respMean,
        'sorted_respSD': sorted_respSD,
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
    }


# ---------------------------------------------------------------------------
# Integration test class
# ---------------------------------------------------------------------------

class TestBoPipelineIntegration:
    """End-to-end BO pipeline smoke tests using synthetic data and dummy surrogate."""

    def test_full_pipeline_no_data_loading(self):
        """BO loop runs on synthetic pool and returns expected keys/shapes."""
        from utils.bo_loops import run_bo_loop

        np.random.seed(42)
        N = 50
        X_pool = np.random.rand(N, 2)
        y_pool = np.random.rand(N, 5)
        X_test = X_pool.copy()
        y_test = y_pool.mean(axis=1)

        result = run_bo_loop(
            model=_DummySurrogate(),
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            n_init=5,
            budget=15,
        )

        assert 'observed_indices' in result
        assert len(result['observed_indices']) == 15
        assert 'y_pred' in result
        assert result['y_pred'].shape == (N,)
        assert 'times' in result
        assert all(t >= 0 for t in result['times'])

    def test_save_load_round_trip(self, tmp_path):
        """save_results then load_results returns consistent dict."""
        from utils.data_utils import save_results, load_results

        np.random.seed(42)
        rng = np.random.RandomState(42)
        results_dict = {
            'gp': [{
                'dataset': 'nhp',
                'subject': 1,
                'emg': 0,
                'r2': [0.5, 0.6, 0.7],
                'times': [0.1, 0.2, 0.1],
                'values': rng.rand(3, 10).tolist(),
                'y_test': rng.rand(20),
            }]
        }

        pkl_path, csv_path = save_results(
            results_dict, 'optimization', output_dir=str(tmp_path)
        )

        # Verify files exist
        assert os.path.isfile(pkl_path)
        assert os.path.isfile(csv_path)

        # Verify round-trip
        loaded = load_results(pkl_path)
        assert 'gp' in loaded
        assert len(loaded['gp']) == 1
        assert loaded['gp'][0]['subject'] == 1

        # CSV has rows
        import csv
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1

    def test_write_run_config_round_trip(self, tmp_path):
        """write_run_config creates config.yaml with correct values."""
        from utils.data_utils import create_run_dir, write_run_config

        run_dir = create_run_dir(
            'dummy', base_dir=str(tmp_path), tag='test-00000'
        )

        config = {'epochs': 50, 'lr': 1e-5, 'dataset': 'nhp'}
        write_run_config(run_dir, config)

        config_path = os.path.join(run_dir, 'config.yaml')
        assert os.path.isfile(config_path)

        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded['epochs'] == 50
        np.testing.assert_allclose(loaded['lr'], 1e-5, rtol=1e-6)
        assert loaded['dataset'] == 'nhp'

    def test_preprocess_neural_data_runs_on_synthetic(self):
        """preprocess_neural_data processes synthetic subject_data without error."""
        from utils.data_utils import preprocess_neural_data

        subject_data = _make_subject_data(n_chan=12, n_emgs=4, n_reps=6, seed=7)
        X_train, Y_train, X_test, Y_test, scaler_y = preprocess_neural_data(
            subject_data, emg_idx=0
        )

        # Shape consistency
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_train.shape[0] == len(Y_test)
        assert X_train.shape[1] == 2  # 2D electrode coords

        # Finite values where expected
        assert np.all(np.isfinite(X_train))
        assert np.all(np.isfinite(Y_test))

        # X values are scaled to [0, 1]
        assert np.all(X_train >= 0.0 - 1e-9)
        assert np.all(X_train <= 1.0 + 1e-9)

    def test_bo_loop_values_track_pool(self):
        """BO loop's observed indices are all valid and values come from y_pool rows."""
        from utils.bo_loops import run_bo_loop

        np.random.seed(0)
        N = 30
        X_pool = np.random.rand(N, 2)
        y_pool = np.random.rand(N, 3)  # 3 reps
        X_test = X_pool.copy()
        y_test = y_pool.mean(axis=1)

        result = run_bo_loop(
            model=_DummySurrogate(),
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            n_init=3,
            budget=10,
        )

        # All observed indices are valid pool indices
        assert all(0 <= idx < N for idx in result['observed_indices'])
        # Budget is respected (total observations == budget)
        assert len(result['observed_indices']) == 10
        # Note: revisits are allowed by design (run_bo_loop docstring), so
        # we do NOT assert uniqueness of observed_indices.

    def test_bo_loop_real_values_are_finite(self):
        """BO loop's 'real_values' list contains only finite floats."""
        from utils.bo_loops import run_bo_loop

        np.random.seed(3)
        N = 25
        X_pool = np.random.rand(N, 2)
        y_pool = np.random.rand(N, 4)
        X_test = X_pool.copy()
        y_test = y_pool.mean(axis=1)

        result = run_bo_loop(
            model=_DummySurrogate(),
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            n_init=4,
            budget=12,
        )

        if 'real_values' in result:
            assert all(np.isfinite(v) for v in result['real_values'])
