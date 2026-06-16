"""Tests for Thompson Sampling acquisition function.

Organized in three levels:
  Level 1 — Shape/type contracts (no GPU, synthetic data)
  Level 2 — Integration with run_bo_loop
  Level 3 — Config/CLI contract
"""
import os
import sys

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

from models.regressors import GPSurrogate, TabPFNSurrogate
from utils.bo_loops import run_bo_loop
from tabpfn import TabPFNRegressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_data():
    """Small synthetic 2-D regression dataset for surrogate tests."""
    rng = np.random.RandomState(42)
    N, D = 30, 2
    X_pool = rng.rand(N, D).astype(np.float32)
    y_means = np.sin(X_pool[:, 0] * 3) + np.cos(X_pool[:, 1] * 2)
    # y_pool: [N, 5 reps] with small noise
    y_pool = y_means[:, None] + rng.randn(N, 5) * 0.1
    y_test = y_means.copy()
    return X_pool, y_pool.astype(np.float32), X_pool.copy(), y_test.astype(np.float32)


@pytest.fixture(scope='module')
def fitted_gp(synthetic_data):
    """GPSurrogate fitted on first 10 pool points."""
    X_pool, y_pool, X_test, y_test = synthetic_data
    gp = GPSurrogate(device='cpu')
    gp.fit(X_pool[:10], y_test[:10])
    return gp, X_pool[10:20]   # (surrogate, 10-point query pool)


@pytest.fixture(scope='module')
def fitted_tabpfn(synthetic_data):
    """TabPFNSurrogate fitted on first 10 pool points (CPU)."""
    X_pool, y_pool, X_test, y_test = synthetic_data
    base = TabPFNRegressor(device='cpu', n_estimators=1, ignore_pretraining_limits=True)
    surrogate = TabPFNSurrogate(model=base)
    surrogate.fit(X_pool[:10], y_test[:10])
    return surrogate, X_pool[10:20]   # (surrogate, 10-point query pool)


# ---------------------------------------------------------------------------
# Level 1 — Shape and type contracts (no GPU)
# ---------------------------------------------------------------------------

def test_gp_surrogate_predict_ts_shape(fitted_gp):
    """predict_ts returns a float ndarray of shape (N,)."""
    gp, X_query = fitted_gp
    result = gp.predict_ts(X_query)
    assert isinstance(result, np.ndarray), "predict_ts should return np.ndarray"
    assert result.shape == (len(X_query),), (
        f"Expected shape ({len(X_query)},), got {result.shape}"
    )
    assert np.issubdtype(result.dtype, np.floating), (
        f"Expected float dtype, got {result.dtype}"
    )
    assert np.isfinite(result).all(), "predict_ts values should be finite"


def test_tabpfn_surrogate_predict_ts_shape(fitted_tabpfn):
    """predict_ts returns a float ndarray of shape (N,) for TabPFNSurrogate (CPU)."""
    surrogate, X_query = fitted_tabpfn
    result = surrogate.predict_ts(X_query, temperature=1.0)
    assert isinstance(result, np.ndarray), "predict_ts should return np.ndarray"
    assert result.shape == (len(X_query),), (
        f"Expected shape ({len(X_query)},), got {result.shape}"
    )
    assert np.issubdtype(result.dtype, np.floating), (
        f"Expected float dtype, got {result.dtype}"
    )
    assert np.isfinite(result).all(), "predict_ts values should be finite"


def test_ts_samples_in_range(fitted_gp, fitted_tabpfn, synthetic_data):
    """TS samples lie within [y_min - tol, y_max + tol] for both surrogates."""
    X_pool, _, _, y_test = synthetic_data
    y_train = y_test[:10]

    # GP: generous range since the posterior can extrapolate
    gp, _ = fitted_gp
    gp_samples = gp.predict_ts(X_pool[:10])
    gp_tol = 3.0 * (y_test.max() - y_test.min())
    assert (gp_samples >= y_test.min() - gp_tol).all(), "GP TS sample below lower bound"
    assert (gp_samples <= y_test.max() + gp_tol).all(), "GP TS sample above upper bound"

    # TabPFN: bar-distribution borders extend beyond the training range by a
    # data-dependent scaling factor (typically ~1.5-2x y_range on each side).
    # We use a generous tolerance of 3 * y_range to cover the actual border extent.
    surrogate, _ = fitted_tabpfn
    tabpfn_samples = surrogate.predict_ts(X_pool[:10], temperature=1.0)
    y_range = float(y_train.max() - y_train.min())
    tab_tol = 3.0 * y_range + 1.0   # generous: 3x training range + 1 unit slack
    assert (tabpfn_samples >= y_train.min() - tab_tol).all(), (
        "TabPFN TS sample below lower bound"
    )
    assert (tabpfn_samples <= y_train.max() + tab_tol).all(), (
        "TabPFN TS sample above upper bound"
    )


def test_ts_temperature_monotonic(synthetic_data):
    """Variance of TabPFN TS samples increases with temperature from 0.1 → 10.0."""
    X_pool, _, _, y_test = synthetic_data
    temperatures = [0.1, 10.0]

    base = TabPFNRegressor(device='cpu', n_estimators=1, ignore_pretraining_limits=True)
    surrogate = TabPFNSurrogate(model=base)
    surrogate.fit(X_pool[:10], y_test[:10])

    # Average variance over several seeds for robustness
    agg_var: dict[float, list[float]] = {t: [] for t in temperatures}
    for seed in range(8):
        np.random.seed(seed)
        for temp in temperatures:
            samples = surrogate.predict_ts(X_pool, temperature=temp)
            agg_var[temp].append(float(np.var(samples)))

    var_low = float(np.mean(agg_var[0.1]))
    var_high = float(np.mean(agg_var[10.0]))
    assert var_low <= var_high, (
        f"Expected variance to increase with temperature: "
        f"var@0.1={var_low:.6f}, var@10.0={var_high:.6f}"
    )


# ---------------------------------------------------------------------------
# Level 2 — Integration with run_bo_loop
# ---------------------------------------------------------------------------

def test_run_bo_loop_ts_completes(synthetic_data):
    """run_bo_loop with acq_fn='ts' completes and returns required keys."""
    X_pool, y_pool, X_test, y_test = synthetic_data
    base = TabPFNRegressor(device='cpu', n_estimators=1, ignore_pretraining_limits=True)
    surrogate = TabPFNSurrogate(model=base)
    np.random.seed(42)
    result = run_bo_loop(
        model=surrogate,
        X_pool=X_pool,
        y_pool=y_pool,
        X_test=X_test,
        y_test=y_test,
        n_init=5,
        budget=12,
        acq_fn='ts',
        ts_temperature=1.0,
    )
    required_keys = {'observed_indices', 'observed_values', 'real_values', 'y_pred'}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )
    assert len(result['observed_indices']) == 12, (
        f"Expected 12 observations, got {len(result['observed_indices'])}"
    )
    assert result['y_pred'].shape == (len(X_test),), (
        f"y_pred shape mismatch: {result['y_pred'].shape}"
    )


def test_run_bo_loop_ucb_regression(synthetic_data):
    """run_bo_loop with acq_fn='ucb' produces identical results with fixed seed."""
    X_pool, y_pool, X_test, y_test = synthetic_data

    kwargs = dict(
        X_pool=X_pool, y_pool=y_pool, X_test=X_test, y_test=y_test,
        n_init=5, budget=10, acq_fn='ucb', kappa_schedule=2.0,
    )

    np.random.seed(7)
    gp1 = GPSurrogate(device='cpu')
    result1 = run_bo_loop(model=gp1, **kwargs)

    np.random.seed(7)
    gp2 = GPSurrogate(device='cpu')
    result2 = run_bo_loop(model=gp2, **kwargs)

    assert result1['observed_indices'] == result2['observed_indices'], (
        "UCB results should be deterministic with the same seed"
    )
    np.testing.assert_array_almost_equal(
        result1['y_pred'], result2['y_pred'],
        err_msg="UCB final predictions should be identical with same seed",
    )


def test_run_bo_loop_invalid_acq_fn(synthetic_data):
    """run_bo_loop raises ValueError for unsupported acq_fn."""
    X_pool, y_pool, X_test, y_test = synthetic_data
    gp = GPSurrogate(device='cpu')
    with pytest.raises(ValueError, match="Unknown acq_fn"):
        run_bo_loop(
            model=gp,
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            n_init=3,
            budget=8,
            acq_fn='ei',
        )


# ---------------------------------------------------------------------------
# Level 3 — Config / CLI contract
# ---------------------------------------------------------------------------

def test_acq_fn_ts_yaml_loading(tmp_path):
    """YAML with acq_fn: ts, ts_temperature: 2.0 is parsed to the correct types."""
    cfg_dict = {
        'dataset': 'nhp',
        'mode': ['optimization'],
        'budget': 50,
        'n_reps': 5,
        'kappa_schedule': 0.0,
        'acq_fn': 'ts',
        'ts_temperature': 2.0,
        'device': 'cpu',
        'seed': 42,
    }
    config_path = tmp_path / 'test_ts.yaml'
    config_path.write_text(yaml.dump(cfg_dict))

    from vanilla_benchmark import _load_yaml_config
    cfg = _load_yaml_config(str(config_path))

    assert cfg['acq_fn'] == 'ts', f"Expected 'ts', got {cfg['acq_fn']!r}"
    assert cfg['ts_temperature'] == 2.0, (
        f"Expected ts_temperature=2.0, got {cfg['ts_temperature']}"
    )
    assert isinstance(cfg['ts_temperature'], float), (
        f"ts_temperature should be float, got {type(cfg['ts_temperature'])}"
    )


def test_acq_fn_ucb_auto_schedule_yaml(tmp_path):
    """YAML with acq_fn: ucb, kappa_schedule: 0.0 signals the auto cosine schedule."""
    cfg_dict = {
        'dataset': 'nhp',
        'mode': ['optimization'],
        'budget': 100,
        'n_reps': 30,
        'kappa_schedule': 0.0,
        'acq_fn': 'ucb',
        'ts_temperature': 1.0,
        'device': 'cuda',
        'seed': 42,
    }
    config_path = tmp_path / 'test_ucb_auto.yaml'
    config_path.write_text(yaml.dump(cfg_dict))

    from vanilla_benchmark import _load_yaml_config
    cfg = _load_yaml_config(str(config_path))

    assert cfg['acq_fn'] == 'ucb', f"Expected 'ucb', got {cfg['acq_fn']!r}"
    # kappa_schedule=0.0 is the sentinel for auto cosine annealing in run_bo_loop
    assert cfg['kappa_schedule'] == 0.0, (
        f"Expected kappa_schedule=0.0 (auto), got {cfg['kappa_schedule']}"
    )


def test_tabpfn_ts_logit_cache_populated(fitted_tabpfn):
    """After predict_ts, _logit_cache is populated and usable by predict()."""
    surrogate, X_query = fitted_tabpfn
    # Ensure cache is clear before the test
    surrogate._logit_cache = None

    _ = surrogate.predict_ts(X_query, temperature=1.0)

    assert surrogate._logit_cache is not None, (
        "_logit_cache should be set after predict_ts"
    )
    X_ref, logits, criterion = surrogate._logit_cache
    assert X_ref.shape == X_query.shape, (
        f"Cached X_ref shape {X_ref.shape} != X_query shape {X_query.shape}"
    )
    assert logits.dim() == 2, (
        f"Expected 2-D logits [M, n_bins], got shape {logits.shape}"
    )
    assert logits.shape[0] == len(X_query), (
        f"logits first dim {logits.shape[0]} != len(X_query) {len(X_query)}"
    )
    # Verify that predict() can use the cache (no second forward pass)
    mean, std = surrogate.predict(X_query)
    assert mean.shape == (len(X_query),), f"predict() mean shape mismatch: {mean.shape}"
    assert std.shape == (len(X_query),), f"predict() std shape mismatch: {std.shape}"
    assert np.isfinite(mean).all(), "predict() mean from cache should be finite"
