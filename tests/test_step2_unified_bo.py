"""Unit tests for Step 2 -- Unified BO Loop & Evaluation.

Covers:
- SurrogateModel protocol (runtime_checkable isinstance checks)
- GPSurrogate constructor interface and pre-fit guard
- TabPFNSurrogate constructor interface and delegation
- run_bo_loop() logic, return keys, error cases, snapshot behaviour
- evaluate_optimization() function signature existence

All tests are low-cost: no GPU, no real model loading, no dataset I/O.
Heavy dependencies (TabPFN, ExactGP, gpytorch) are mocked where needed.
"""
import ast
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    np.random.seed(42)
    return np.random.RandomState(42)


@pytest.fixture
def small_pool(rng):
    """Small synthetic pool: 20 points in 2D, 5 repetitions each.

    X_test and y_test have the same number of rows as X_pool because
    run_bo_loop indexes y_test using pool indices.
    """
    N = 20
    X_pool = rng.rand(N, 2).astype(np.float64)
    y_pool = rng.rand(N, 5).astype(np.float64)
    X_test = X_pool.copy()       # same locations as pool
    y_test = y_pool.mean(axis=1)  # mean across reps
    return X_pool, y_pool, X_test, y_test


# ---------------------------------------------------------------------------
# Mock surrogates for testing (no ML dependencies)
# ---------------------------------------------------------------------------

class _DummySurrogate:
    """Minimal SurrogateModel returning constant predictions."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._mean_val = float(y.mean()) if len(y) > 0 else 0.0

    def predict(self, X: np.ndarray):
        n = X.shape[0]
        return np.full(n, self._mean_val), np.ones(n) * 0.1

    def predict_ucb(self, X: np.ndarray, kappa: float, t: int, n_steps: int):
        mean, std = self.predict(X)
        return mean + kappa * std


class _NoPredictUCBSurrogate:
    """Surrogate without predict_ucb -- tests fallback path."""

    def fit(self, X, y):
        self._val = float(y.mean())

    def predict(self, X):
        return np.full(X.shape[0], self._val), np.ones(X.shape[0]) * 0.1


class _NaNUCBSurrogate:
    """Surrogate that always returns NaN UCB values."""

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.full(X.shape[0], np.nan), np.ones(X.shape[0])

    def predict_ucb(self, X, kappa, t, n_steps):
        return np.full(X.shape[0], np.nan)


class _MissingPredictSurrogate:
    def fit(self, X, y):
        pass


class _MissingFitSurrogate:
    def predict(self, X):
        return np.zeros(X.shape[0]), np.ones(X.shape[0])


# ---------------------------------------------------------------------------
# SurrogateModel protocol
# ---------------------------------------------------------------------------

class TestSurrogateModelProtocol:

    def _import(self):
        from models.regressors import SurrogateModel
        return SurrogateModel

    def test_full_implementation_satisfies(self):
        SM = self._import()
        assert isinstance(_DummySurrogate(), SM)

    def test_missing_predict_does_not_satisfy(self):
        SM = self._import()
        assert not isinstance(_MissingPredictSurrogate(), SM)

    def test_missing_fit_does_not_satisfy(self):
        SM = self._import()
        assert not isinstance(_MissingFitSurrogate(), SM)

    def test_plain_object_does_not_satisfy(self):
        SM = self._import()
        assert not isinstance(object(), SM)

    def test_protocol_is_runtime_checkable(self):
        SM = self._import()
        assert getattr(SM, "_is_protocol", False)


# ---------------------------------------------------------------------------
# GPSurrogate
# ---------------------------------------------------------------------------

class TestGPSurrogateConstructor:

    def _import(self):
        from models.regressors import GPSurrogate
        return GPSurrogate

    def test_default_construction(self):
        GP = self._import()
        s = GP()
        assert s._device == "cpu"
        assert s._n_opt_steps == 50
        assert s._model is None

    def test_custom_construction(self):
        GP = self._import()
        s = GP(device="cpu", n_opt_steps=10, lr=0.05, kappa_0=3.0, kappa_min=0.5)
        assert s._n_opt_steps == 10
        assert abs(s._lr - 0.05) < 1e-9
        assert abs(s._kappa_0 - 3.0) < 1e-9

    def test_predict_before_fit_raises(self):
        GP = self._import()
        s = GP()
        with pytest.raises(RuntimeError, match="before fit"):
            s.predict(np.random.rand(5, 2))

    def test_fit_raises_on_nan_x(self):
        GP = self._import()
        s = GP()
        with pytest.raises(RuntimeError, match="NaN"):
            s.fit(np.array([[1.0, np.nan]]), np.array([1.0]))

    def test_fit_raises_on_nan_y(self):
        GP = self._import()
        s = GP()
        with pytest.raises(RuntimeError, match="NaN"):
            s.fit(np.array([[1.0, 2.0]]), np.array([np.nan]))

    def test_satisfies_protocol(self):
        from models.regressors import GPSurrogate, SurrogateModel
        assert isinstance(GPSurrogate(), SurrogateModel)


# ---------------------------------------------------------------------------
# TabPFNSurrogate
# ---------------------------------------------------------------------------

class TestTabPFNSurrogateConstructor:

    def test_stores_model_reference(self):
        from models.regressors import TabPFNSurrogate
        mock = MagicMock()
        s = TabPFNSurrogate(mock)
        assert s._model is mock

    def test_default_kappa_values(self):
        from models.regressors import TabPFNSurrogate
        s = TabPFNSurrogate(MagicMock())
        assert abs(s._kappa_0 - 2.5) < 1e-9
        assert abs(s._kappa_min - 0.5) < 1e-9

    def test_custom_kappa(self):
        from models.regressors import TabPFNSurrogate
        s = TabPFNSurrogate(MagicMock(), kappa_0=5.0, kappa_min=1.0)
        assert abs(s._kappa_0 - 5.0) < 1e-9

    def test_fit_delegates_to_inner_model(self):
        from models.regressors import TabPFNSurrogate
        mock = MagicMock()
        s = TabPFNSurrogate(mock)
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        s.fit(X, y)
        mock.fit.assert_called_once()
        np.testing.assert_array_equal(mock.fit.call_args[0][0], X)
        np.testing.assert_array_equal(mock.fit.call_args[0][1], y)

    def test_fit_raises_on_nan_x(self):
        from models.regressors import TabPFNSurrogate
        s = TabPFNSurrogate(MagicMock())
        with pytest.raises(RuntimeError, match="NaN"):
            s.fit(np.array([[1.0, np.nan]]), np.array([1.0]))

    def test_fit_raises_on_nan_y(self):
        from models.regressors import TabPFNSurrogate
        s = TabPFNSurrogate(MagicMock())
        with pytest.raises(RuntimeError, match="NaN"):
            s.fit(np.array([[1.0, 2.0]]), np.array([np.nan]))

    def test_satisfies_protocol(self):
        from models.regressors import TabPFNSurrogate, SurrogateModel
        assert isinstance(TabPFNSurrogate(MagicMock()), SurrogateModel)


# ---------------------------------------------------------------------------
# run_bo_loop
# ---------------------------------------------------------------------------

class TestRunBoLoop:

    def test_returns_all_required_keys(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        result = run_bo_loop(
            model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=8,
        )
        for key in ("observed_indices", "observed_values", "real_values",
                     "times", "y_pred", "snapshots"):
            assert key in result, f"Missing key: {key}"

    def test_observed_indices_length_equals_budget(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        result = run_bo_loop(
            model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=8,
        )
        assert len(result["observed_indices"]) == 8

    def test_y_pred_shape_matches_x_test(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        result = run_bo_loop(
            model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=8,
        )
        assert result["y_pred"].shape == (X_test.shape[0],)

    def test_times_length_equals_n_steps(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        result = run_bo_loop(
            model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=8,
        )
        assert len(result["times"]) == 8 - 3

    def test_raises_when_budget_le_n_init(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        with pytest.raises(ValueError, match="budget"):
            run_bo_loop(
                model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
                X_test=X_test, y_test=y_test, n_init=10, budget=5,
            )

    def test_raises_when_budget_equals_n_init(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        with pytest.raises(ValueError, match="budget"):
            run_bo_loop(
                model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
                X_test=X_test, y_test=y_test, n_init=8, budget=8,
            )

    def test_raises_on_all_nan_ucb(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        with pytest.raises(RuntimeError, match="non-finite"):
            run_bo_loop(
                model=_NaNUCBSurrogate(), X_pool=X_pool, y_pool=y_pool,
                X_test=X_test, y_test=y_test, n_init=3, budget=6,
            )

    def test_no_repeated_queries(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        result = run_bo_loop(
            model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=10,
        )
        assert len(set(result["observed_indices"])) == len(result["observed_indices"])

    def test_snapshots_none_when_not_requested(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        result = run_bo_loop(
            model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=6,
            snapshot_iters=None,
        )
        assert result["snapshots"] is None

    def test_snapshots_dict_when_requested(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        budget = 8
        result = run_bo_loop(
            model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=budget,
            snapshot_iters=[4, 6, budget],
        )
        assert isinstance(result["snapshots"], dict)
        assert budget in result["snapshots"]

    def test_snapshot_predictions_correct_shape(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        budget = 8
        result = run_bo_loop(
            model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=budget,
            snapshot_iters=[budget],
        )
        assert result["snapshots"][budget].shape == (X_test.shape[0],)

    def test_accepts_surrogate_without_predict_ucb(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        result = run_bo_loop(
            model=_NoPredictUCBSurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=6,
        )
        assert result["y_pred"].shape == (X_test.shape[0],)

    def test_real_values_length_matches_observed(self, small_pool):
        from utils.bo_loops import run_bo_loop
        X_pool, y_pool, X_test, y_test = small_pool
        np.random.seed(42)
        result = run_bo_loop(
            model=_DummySurrogate(), X_pool=X_pool, y_pool=y_pool,
            X_test=X_test, y_test=y_test, n_init=3, budget=8,
        )
        assert len(result["real_values"]) == len(result["observed_indices"])


# ---------------------------------------------------------------------------
# _snapshot_iters helper
# ---------------------------------------------------------------------------

class TestSnapshotIters:

    def test_always_includes_budget(self):
        from utils.bo_loops import _snapshot_iters
        assert 20 in _snapshot_iters(budget=20, n_init=5)

    def test_result_is_sorted(self):
        from utils.bo_loops import _snapshot_iters
        result = _snapshot_iters(budget=64, n_init=4)
        assert result == sorted(result)

    def test_all_values_between_n_init_and_budget(self):
        from utils.bo_loops import _snapshot_iters
        result = _snapshot_iters(budget=30, n_init=5)
        for v in result:
            assert 5 < v <= 30

    def test_single_step_budget(self):
        from utils.bo_loops import _snapshot_iters
        assert _snapshot_iters(budget=6, n_init=5) == [6]


# ---------------------------------------------------------------------------
# evaluate_optimization -- signature inspection only (no data loading)
# ---------------------------------------------------------------------------

class TestEvaluateOptimizationSignature:

    def _parse_eval(self):
        eval_path = os.path.join(_SRC_DIR, "evaluation.py")
        with open(eval_path) as f:
            return ast.parse(f.read())

    def test_function_exists(self):
        tree = self._parse_eval()
        names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert "evaluate_optimization" in names

    def test_has_surrogate_parameter(self):
        tree = self._parse_eval()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_optimization":
                args = [a.arg for a in node.args.args]
                assert "surrogate" in args
                return
        pytest.fail("evaluate_optimization not found")

    def test_has_budget_parameter(self):
        tree = self._parse_eval()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_optimization":
                args = [a.arg for a in node.args.args]
                assert "budget" in args
                return
        pytest.fail("evaluate_optimization not found")
