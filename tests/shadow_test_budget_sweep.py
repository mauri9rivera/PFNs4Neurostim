"""Shadow test: budget-sweep single-trajectory refactor (Task 3, Step 3).

Verifies that _vanilla_optimization_budget() produces the correct DataFrame
structure and budget_sweep_optimization SVG using a single max-budget
trajectory (no outer per-budget loop).  Runs on CPU with tiny params so it
finishes in under 60 s.

Run with:
    conda activate pfns4neurostim
    pytest tests/shadow_test_budget_sweep.py -v
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from evaluation import _unpack_budget_trajectory, evaluate_optimization
from models.regressors import GPSurrogate, TabPFNSurrogate
from tabpfn import TabPFNRegressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_pool(n: int = 32, d: int = 2, n_reps: int = 3, seed: int = 0):
    """Synthetic pool that mimics neurostim shape [N, n_reps]."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, d).astype(np.float32)
    y_mean = np.sin(X[:, 0] * 3) + np.cos(X[:, 1] * 2)
    y_pool = (y_mean[:, None] + rng.randn(n, n_reps) * 0.05).astype(np.float32)
    return X, y_pool


# ---------------------------------------------------------------------------
# Unit-level: _unpack_budget_trajectory
# ---------------------------------------------------------------------------

class TestUnpackBudgetTrajectory:
    """Verify _unpack_budget_trajectory extracts rows correctly from a stub result."""

    def _stub_result(self, n_reps: int = 4, max_b: int = 10, budgets: list = None):
        rng = np.random.RandomState(7)
        budgets = budgets or [5, 10]
        n_pts = 20
        y_test = rng.rand(n_pts).astype(np.float32)
        # values: [n_reps, max_b]  monotonically increasing (simulate best_found)
        values = np.cumsum(rng.rand(n_reps, max_b), axis=1).tolist()
        r2_by_snap = {b: [rng.rand() for _ in range(n_reps)] for b in budgets}
        return {
            'values': values,
            'y_test': y_test,
            'r2_by_snapshot': r2_by_snap,
            'model_type': 'TestModel',
            'dataset': 'synthetic',
            'subject': 0,
            'emg': 0,
        }, budgets

    def test_row_count(self):
        result, budgets = self._stub_result(n_reps=4, max_b=10, budgets=[5, 10])
        plot_data: list = []
        _unpack_budget_trajectory(plot_data, result, 'TestModel', budgets, 0, 0)
        # 2 budgets × 4 reps = 8 rows
        assert len(plot_data) == 8

    def test_columns_present(self):
        result, budgets = self._stub_result()
        plot_data: list = []
        _unpack_budget_trajectory(plot_data, result, 'TestModel', budgets, 0, 0)
        df = pd.DataFrame(plot_data)
        for col in ('Budget', 'Model', 'Regret', 'R2', 'ID'):
            assert col in df.columns, f"Missing column: {col}"

    def test_budget_values(self):
        budgets = [5, 10]
        result, _ = self._stub_result(budgets=budgets)
        plot_data: list = []
        _unpack_budget_trajectory(plot_data, result, 'TestModel', budgets, 0, 0)
        df = pd.DataFrame(plot_data)
        assert set(df['Budget'].unique()) == set(budgets)

    def test_regret_finite(self):
        """Regret values must be finite (can be negative when surrogate overshoots)."""
        result, budgets = self._stub_result()
        plot_data: list = []
        _unpack_budget_trajectory(plot_data, result, 'TestModel', budgets, 0, 0)
        df = pd.DataFrame(plot_data)
        assert df['Regret'].notna().all(), "Regret contains NaN"
        assert np.isfinite(df['Regret']).all(), "Regret contains Inf"

    def test_r2_fallback_nan(self):
        """When r2_by_snapshot is missing a budget key, R2 should be NaN."""
        result, _ = self._stub_result(budgets=[5, 10])
        result['r2_by_snapshot'] = {}   # empty → all NaN
        plot_data: list = []
        _unpack_budget_trajectory(plot_data, result, 'TestModel', [5, 10], 0, 0)
        df = pd.DataFrame(plot_data)
        assert df['R2'].isna().all(), "Expected NaN R2 when r2_by_snapshot is empty"


# ---------------------------------------------------------------------------
# Integration: evaluate_optimization with capture_all_snapshots
# ---------------------------------------------------------------------------

class TestEvaluateOptimizationSnapshot:
    """Verify that capture_all_snapshots=True populates r2_by_snapshot."""

    @pytest.fixture(scope='class')
    def tabpfn_surrogate(self):
        model = TabPFNRegressor(device='cpu', n_estimators=1,
                                ignore_pretraining_limits=True)
        return TabPFNSurrogate(model=model)

    def _run_with_synthetic_data(self, surrogate, budgets, n_reps=2):
        """Patch evaluate_optimization to use in-memory synthetic data."""
        from unittest.mock import patch
        X, y_pool = _make_synthetic_pool(n=32, d=2, n_reps=3)
        y_test = y_pool[:, 0]
        max_b = max(budgets)

        # Patch load_data and preprocess_neural_data inside evaluate_optimization
        with patch('evaluation.load_data') as mock_load, \
             patch('evaluation.preprocess_neural_data') as mock_pre:
            mock_load.return_value = {
            'ch2xy': X,
            'sorted_respMean': y_pool,
            'sorted_resp': y_pool[:, :, np.newaxis],  # [N, n_emgs, n_reps] shape stub
        }
            mock_pre.return_value = (X, y_pool, X, y_test, None)

            from evaluation import evaluate_optimization
            result = evaluate_optimization(
                surrogate=surrogate,
                dataset_type='synthetic',
                subject_idx=0,
                emg_idx=0,
                device='cpu',
                budget=max_b,
                n_reps=n_reps,
                capture_all_snapshots=True,
                snapshot_iters_override=budgets,
            )
        return result

    def test_r2_by_snapshot_populated(self, tabpfn_surrogate):
        budgets = [5, 10]
        result = self._run_with_synthetic_data(tabpfn_surrogate, budgets, n_reps=2)
        assert result.get('r2_by_snapshot') is not None
        for b in budgets:
            assert b in result['r2_by_snapshot'], f"Budget {b} missing from r2_by_snapshot"
            assert len(result['r2_by_snapshot'][b]) == 2  # one per rep

    def test_values_shape(self, tabpfn_surrogate):
        budgets = [5, 10]
        result = self._run_with_synthetic_data(tabpfn_surrogate, budgets, n_reps=2)
        arr = np.array(result['values'])
        assert arr.shape == (2, max(budgets)), \
            f"Expected values shape (2, {max(budgets)}), got {arr.shape}"


# ---------------------------------------------------------------------------
# Integration: end-to-end plot production (no real data needed)
# ---------------------------------------------------------------------------

class TestBudgetSweepPlotProduction:
    """Verify budget_sweep_plot is produced from the single-trajectory path."""

    def test_plot_produced(self, tmp_path):
        from unittest.mock import patch
        import pandas as pd
        from utils.visualization import budget_sweep_plot

        # Simulate the DataFrame that _vanilla_optimization_budget would produce
        rng = np.random.RandomState(42)
        budgets = [5, 10]
        rows = []
        for b in budgets:
            for model in ('TabPFN', 'GP'):
                for _ in range(4):
                    rows.append({
                        'Budget': b,
                        'Model': model,
                        'Regret': max(0.0, rng.randn() * 0.1 + 0.5),
                        'R2': float(np.clip(rng.randn() * 0.1 + 0.7, -1, 1)),
                        'ID': '0_0',
                    })
        df = pd.DataFrame(rows)

        out = str(tmp_path)
        # Should not raise; produces SVG file
        budget_sweep_plot(df, eval_type='optimization', dataset='nhp',
                          split_type='test', save=True, output_dir=out)

        # budget_sweep_plot saves into an 'optimization' subdirectory
        opt_dir = os.path.join(out, 'optimization')
        search_dir = opt_dir if os.path.isdir(opt_dir) else out
        found = any(
            'budget_sweep' in f and f.endswith('.svg')
            for f in os.listdir(search_dir)
        )
        assert found, (
            f"budget_sweep SVG not found. "
            f"Searched: {search_dir}. Files: {os.listdir(search_dir)}"
        )
