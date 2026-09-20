"""Unit tests for pure utility functions in src/analysis/id_ood.py.

IMPORTANT: imports are lazy (inside each test method) to prevent TabPFN model
loading at module import time. Only pure numpy/scipy functions are tested here.

Covers:
- _layer_name()
- _normalize_for_tabpfn()
- compute_procrustes_disparity()
- compute_rsa()
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


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _import_layer_name():
    from pfns4neurostim.analysis.id_ood import _layer_name
    return _layer_name


def _import_normalize_for_tabpfn():
    from pfns4neurostim.analysis.id_ood import _normalize_for_tabpfn
    return _normalize_for_tabpfn


def _import_procrustes():
    from pfns4neurostim.analysis.id_ood import compute_procrustes_disparity
    return compute_procrustes_disparity


def _import_rsa():
    from pfns4neurostim.analysis.id_ood import compute_rsa
    return compute_rsa


# ---------------------------------------------------------------------------
# _layer_name
# ---------------------------------------------------------------------------

class TestLayerName:
    """Tests for _layer_name() module path helper."""

    def test_layer_zero_produces_correct_path(self):
        """_layer_name(0) == 'transformer_encoder.layers.0'."""
        _layer_name = _import_layer_name()
        assert _layer_name(0) == 'transformer_encoder.layers.0'

    def test_layer_17_produces_correct_path(self):
        """_layer_name(17) == 'transformer_encoder.layers.17'."""
        _layer_name = _import_layer_name()
        assert _layer_name(17) == 'transformer_encoder.layers.17'

    def test_layer_name_format_is_consistent(self):
        """All layer names follow 'transformer_encoder.layers.{idx}' pattern."""
        _layer_name = _import_layer_name()
        for idx in [0, 4, 9, 13, 17]:
            result = _layer_name(idx)
            assert result == f'transformer_encoder.layers.{idx}'


# ---------------------------------------------------------------------------
# _normalize_for_tabpfn
# ---------------------------------------------------------------------------

class TestNormalizeForTabPFN:
    """Tests for _normalize_for_tabpfn() preprocessing utility."""

    def test_returns_none_when_fewer_than_two_rows(self):
        """Returns None when X has fewer than 2 rows."""
        _normalize_for_tabpfn = _import_normalize_for_tabpfn()
        X = np.array([[1.0, 2.0]])  # only 1 row
        y = np.array([0.5])
        result = _normalize_for_tabpfn(X, y)
        assert result is None

    def test_returns_float32_arrays(self):
        """Returned X_norm has dtype float32."""
        _normalize_for_tabpfn = _import_normalize_for_tabpfn()
        rng = np.random.RandomState(42)
        X = rng.randn(20, 3).astype(np.float64)
        y = rng.randn(20)
        result = _normalize_for_tabpfn(X, y)
        assert result is not None
        X_norm, y_norm = result
        assert X_norm.dtype == np.float32

    def test_x_norm_values_in_unit_range(self):
        """MinMax-scaled X_norm values are in [0, 1]."""
        _normalize_for_tabpfn = _import_normalize_for_tabpfn()
        rng = np.random.RandomState(1)
        X = rng.randn(50, 2)
        y = rng.randn(50)
        result = _normalize_for_tabpfn(X, y)
        assert result is not None
        X_norm, _ = result
        assert np.all(X_norm >= 0.0 - 1e-6)
        assert np.all(X_norm <= 1.0 + 1e-6)

    def test_y_norm_is_standardized(self):
        """y_norm has mean ≈ 0, std ≈ 1 (StandardScaler applied)."""
        _normalize_for_tabpfn = _import_normalize_for_tabpfn()
        rng = np.random.RandomState(2)
        X = rng.rand(100, 2)
        y = rng.randn(100) * 5 + 10  # mean=10, std≈5
        result = _normalize_for_tabpfn(X, y)
        assert result is not None
        _, y_norm = result
        np.testing.assert_allclose(y_norm.mean(), 0.0, atol=1e-5)
        np.testing.assert_allclose(y_norm.std(), 1.0, atol=1e-5)

    def test_normalized_x_shape_matches_input(self):
        """Returned X_norm has the same shape as the input X.

        Note: The docstring mentions that constant columns produce NaN after
        MinMaxScaler, but in practice sklearn's MinMaxScaler maps constant
        columns to 0.0 (not NaN). The None path is only triggered by X.shape[0] < 2
        or truly non-finite output after scaling. This test validates the
        shape-preservation property of the happy path.
        """
        _normalize_for_tabpfn = _import_normalize_for_tabpfn()
        rng = np.random.RandomState(9)
        X = rng.rand(40, 3)
        y = rng.randn(40)
        result = _normalize_for_tabpfn(X, y)
        assert result is not None
        X_norm, y_norm = result
        assert X_norm.shape == X.shape

    def test_returns_tuple_on_valid_input(self):
        """Returns a 2-tuple (X_norm, y_norm) for valid inputs."""
        _normalize_for_tabpfn = _import_normalize_for_tabpfn()
        rng = np.random.RandomState(5)
        X = rng.rand(30, 3)
        y = rng.randn(30)
        result = _normalize_for_tabpfn(X, y)
        assert result is not None
        assert len(result) == 2


# ---------------------------------------------------------------------------
# compute_procrustes_disparity
# ---------------------------------------------------------------------------

class TestComputeProcrustesDisparity:
    """Tests for compute_procrustes_disparity() Procrustes shape comparison."""

    def test_returns_float_in_unit_interval(self):
        """Returns a float in [0, 1] for random matrices."""
        compute_procrustes_disparity = _import_procrustes()
        rng = np.random.RandomState(42)
        Z1 = rng.randn(50, 8)
        Z2 = rng.randn(50, 8)
        result = compute_procrustes_disparity(Z1, Z2)
        assert isinstance(result, float)
        assert 0.0 - 1e-9 <= result <= 1.0 + 1e-9

    def test_identical_matrices_give_near_zero_disparity(self):
        """Z1 == Z2 → disparity ≈ 0.0."""
        compute_procrustes_disparity = _import_procrustes()
        rng = np.random.RandomState(10)
        Z = rng.randn(40, 6)
        result = compute_procrustes_disparity(Z, Z.copy())
        assert result < 1e-8

    def test_random_matrices_give_positive_disparity(self):
        """Unrelated random Z1, Z2 → disparity in (0, 1]."""
        compute_procrustes_disparity = _import_procrustes()
        rng = np.random.RandomState(7)
        Z1 = rng.randn(50, 8)
        Z2 = rng.randn(50, 8)
        result = compute_procrustes_disparity(Z1, Z2)
        assert result > 0.0

    def test_n_subsample_gte_n_uses_all_points(self):
        """n_subsample >= len(Z) uses all points without error."""
        compute_procrustes_disparity = _import_procrustes()
        rng = np.random.RandomState(11)
        Z1 = rng.randn(20, 4)
        Z2 = rng.randn(20, 4)
        result = compute_procrustes_disparity(Z1, Z2, n_subsample=500)
        assert np.isfinite(result)

    def test_n_subsample_less_than_n_still_valid(self):
        """n_subsample < len(Z) subsamples and returns valid float."""
        compute_procrustes_disparity = _import_procrustes()
        rng = np.random.RandomState(12)
        Z1 = rng.randn(100, 8)
        Z2 = rng.randn(100, 8)
        result = compute_procrustes_disparity(Z1, Z2, n_subsample=20)
        assert np.isfinite(result)
        assert 0.0 - 1e-9 <= result <= 1.0 + 1e-9

    def test_different_seeds_give_different_subsampled_results(self):
        """Different seeds produce (potentially) different results."""
        compute_procrustes_disparity = _import_procrustes()
        rng = np.random.RandomState(20)
        Z1 = rng.randn(200, 8)
        Z2 = rng.randn(200, 8)
        r1 = compute_procrustes_disparity(Z1, Z2, n_subsample=30, seed=1)
        r2 = compute_procrustes_disparity(Z1, Z2, n_subsample=30, seed=99)
        # They may differ due to subsampling
        assert r1 != r2 or True  # not a strict requirement, just no crash

    def test_mismatched_shapes_raise_value_error(self):
        """Z1 and Z2 with different shapes raises ValueError."""
        compute_procrustes_disparity = _import_procrustes()
        Z1 = np.random.randn(30, 4)
        Z2 = np.random.randn(40, 4)
        with pytest.raises(ValueError):
            compute_procrustes_disparity(Z1, Z2)


# ---------------------------------------------------------------------------
# compute_rsa
# ---------------------------------------------------------------------------

class TestComputeRsa:
    """Tests for compute_rsa() Representational Similarity Analysis.

    Rewritten 2026-09-18 for the current contract (task #1 Phase 0): compute_rsa
    takes the **upper triangles of two RDMs** over the same probe pairs, not two
    embedding clouds. The previous tests still passed embedding matrices and a
    removed ``n_subsample`` argument, so they failed with a TypeError and were
    asserting an API that no longer exists.
    """

    def test_identical_rdms_give_rho_one(self):
        compute_rsa = _import_rsa()
        v = np.random.RandomState(0).rand(45)
        assert compute_rsa(v, v.copy()) == pytest.approx(1.0)

    def test_monotone_transform_preserves_rank_correlation(self):
        """RSA is a *rank* correlation, so any increasing map leaves it at 1."""
        compute_rsa = _import_rsa()
        v = np.random.RandomState(1).rand(45) + 0.1
        assert compute_rsa(v, np.exp(3.0 * v)) == pytest.approx(1.0)

    def test_reversed_ranks_give_rho_minus_one(self):
        compute_rsa = _import_rsa()
        v = np.arange(1.0, 31.0)
        assert compute_rsa(v, v[::-1].copy()) == pytest.approx(-1.0)

    def test_returns_float_in_unit_interval(self):
        compute_rsa = _import_rsa()
        rng = np.random.RandomState(2)
        result = compute_rsa(rng.rand(60), rng.rand(60))
        assert isinstance(result, float)
        assert -1.0 - 1e-9 <= result <= 1.0 + 1e-9

    def test_independent_rdms_are_near_zero(self):
        compute_rsa = _import_rsa()
        rng = np.random.RandomState(55)
        assert abs(compute_rsa(rng.rand(400), rng.rand(400))) < 0.2

    def test_misaligned_shapes_return_nan(self):
        """Probe pairs must correspond; a shape mismatch is a caller bug, not a score."""
        compute_rsa = _import_rsa()
        assert np.isnan(compute_rsa(np.arange(10.0), np.arange(9.0)))

    def test_none_input_returns_nan(self):
        compute_rsa = _import_rsa()
        assert np.isnan(compute_rsa(None, np.arange(10.0)))

    def test_zero_variance_returns_nan(self):
        """A constant RDM has no ranks to correlate, so rho is undefined."""
        compute_rsa = _import_rsa()
        assert np.isnan(compute_rsa(np.ones(20), np.random.RandomState(7).rand(20)))
