"""Unit tests for src/models/regressors.py.

Covers:
- linear_cka(): scale-invariance, self-similarity, orthogonal matrices
- _make_finetuned_regressor(): factory return types
- GPSurrogate.predict_ucb(): shape and kappa semantics (mocked predict)

Does NOT re-test SurrogateModel protocol, GPSurrogate/TabPFNSurrogate
constructors or run_bo_loop (covered in test_step2_unified_bo.py).
"""
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _import_linear_cka():
    from models.regressors import linear_cka
    return linear_cka


def _import_make_finetuned_regressor():
    from models.regressors import _make_finetuned_regressor
    return _make_finetuned_regressor


def _import_gp_surrogate():
    from models.regressors import GPSurrogate
    return GPSurrogate


# ---------------------------------------------------------------------------
# linear_cka
# ---------------------------------------------------------------------------

class TestLinearCka:
    """Tests for linear_cka() linear Centered Kernel Alignment."""

    def test_identical_matrices_give_cka_one(self):
        """CKA(X, X) == 1.0 for any non-zero X."""
        linear_cka = _import_linear_cka()
        torch.manual_seed(42)
        X = torch.randn(20, 8)
        result = linear_cka(X, X)
        assert abs(result - 1.0) < 1e-5

    def test_output_is_python_float(self):
        """Returns a Python float, not a tensor."""
        linear_cka = _import_linear_cka()
        X = torch.randn(10, 4)
        Y = torch.randn(10, 4)
        result = linear_cka(X, Y)
        assert isinstance(result, float)

    def test_scale_invariance(self):
        """CKA(X, Y) == CKA(alpha*X, Y) for any scalar alpha != 0."""
        linear_cka = _import_linear_cka()
        torch.manual_seed(7)
        X = torch.randn(15, 6)
        Y = torch.randn(15, 6)
        cka1 = linear_cka(X, Y)
        cka2 = linear_cka(3.0 * X, Y)
        assert abs(cka1 - cka2) < 1e-5

    def test_result_in_unit_interval_for_nontrivial_matrices(self):
        """Result in [0, 1] for general random matrices."""
        linear_cka = _import_linear_cka()
        torch.manual_seed(99)
        X = torch.randn(30, 10)
        Y = torch.randn(30, 10)
        result = linear_cka(X, Y)
        assert 0.0 - 1e-6 <= result <= 1.0 + 1e-6

    def test_orthogonal_matrices_give_near_zero_cka(self):
        """CKA between very different orthogonal projections should be small."""
        linear_cka = _import_linear_cka()
        n, d = 50, 4
        # Construct two matrices with orthogonal column spaces
        A = torch.zeros(n, d)
        A[:d, :] = torch.eye(d)  # [d×d identity] padded with zeros
        B = torch.zeros(n, d)
        B[d:2*d, :] = torch.eye(d)  # shifted identity
        result = linear_cka(A, B)
        # After mean-centering, these matrices are mostly unrelated
        assert result < 0.5  # not a tight bound, but should be far below 1.0

    def test_small_matrix_returns_finite_float(self):
        """For n=2, d=3, returns a finite float."""
        linear_cka = _import_linear_cka()
        X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Y = torch.tensor([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])
        result = linear_cka(X, Y)
        assert np.isfinite(result)

    def test_symmetry(self):
        """CKA(X, Y) == CKA(Y, X) (symmetric kernel)."""
        linear_cka = _import_linear_cka()
        torch.manual_seed(13)
        X = torch.randn(20, 5)
        Y = torch.randn(20, 5)
        assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-5


# ---------------------------------------------------------------------------
# _make_finetuned_regressor
# ---------------------------------------------------------------------------

class TestMakeFinetunedRegressor:
    """Tests for _make_finetuned_regressor() factory function.

    Only performs isinstance and attribute checks — no .fit() calls.
    """

    def test_silence_diagnostics_true_returns_finetuned_regressor(self):
        """silence_diagnostics=True, use_lora=False → FinetunedTabPFNRegressor."""
        _make_finetuned_regressor = _import_make_finetuned_regressor()
        from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
        model = _make_finetuned_regressor(silence_diagnostics=True, use_lora=False)
        assert isinstance(model, FinetunedTabPFNRegressor)

    def test_silence_diagnostics_false_returns_gradient_monitored_regressor(self):
        """silence_diagnostics=False, use_lora=False → GradientMonitoredRegressor."""
        _make_finetuned_regressor = _import_make_finetuned_regressor()
        from models.regressors import GradientMonitoredRegressor
        model = _make_finetuned_regressor(silence_diagnostics=False, use_lora=False)
        assert isinstance(model, GradientMonitoredRegressor)

    def test_use_lora_true_returns_lora_regressor(self):
        """use_lora=True → LoRAFinetunedRegressor."""
        _make_finetuned_regressor = _import_make_finetuned_regressor()
        from models.regressors import LoRAFinetunedRegressor
        model = _make_finetuned_regressor(use_lora=True)
        assert isinstance(model, LoRAFinetunedRegressor)

    def test_lora_regressor_has_lora_rank_attribute(self):
        """LoRAFinetunedRegressor stores the requested lora_rank."""
        _make_finetuned_regressor = _import_make_finetuned_regressor()
        model = _make_finetuned_regressor(use_lora=True, lora_rank=16)
        assert model._lora_rank == 16

    def test_lora_regressor_has_lora_target_attribute(self):
        """LoRAFinetunedRegressor stores the requested lora_target."""
        _make_finetuned_regressor = _import_make_finetuned_regressor()
        model = _make_finetuned_regressor(use_lora=True, lora_target='decoder_dict')
        assert model._lora_target == 'decoder_dict'

    def test_gradient_monitored_regressor_has_diagnostics_list(self):
        """GradientMonitoredRegressor initialises _diagnostics_ as an empty list."""
        _make_finetuned_regressor = _import_make_finetuned_regressor()
        model = _make_finetuned_regressor(silence_diagnostics=False, use_lora=False)
        assert hasattr(model, '_diagnostics_')
        assert isinstance(model._diagnostics_, list)


# ---------------------------------------------------------------------------
# GPSurrogate.predict_ucb (mocked predict)
# ---------------------------------------------------------------------------

class TestGPSurrogatePredictUcb:
    """Tests for GPSurrogate.predict_ucb() using a mocked predict method."""

    def test_predict_ucb_returns_correct_shape(self):
        """predict_ucb returns shape (M,)."""
        GPSurrogate = _import_gp_surrogate()
        s = GPSurrogate()
        X = np.random.rand(5, 2)
        with patch.object(s, 'predict', return_value=(np.zeros(5), np.ones(5) * 0.1)):
            result = s.predict_ucb(X, kappa=2.0, t=0, n_steps=10)
        assert result.shape == (5,)

    def test_kappa_zero_ucb_equals_mean(self):
        """With kappa=0, UCB output equals the predicted mean."""
        GPSurrogate = _import_gp_surrogate()
        s = GPSurrogate()
        mean_vals = np.array([1.0, 2.0, 3.0])
        std_vals = np.array([0.5, 0.5, 0.5])
        X = np.random.rand(3, 2)
        with patch.object(s, 'predict', return_value=(mean_vals, std_vals)):
            result = s.predict_ucb(X, kappa=0.0, t=0, n_steps=10)
        np.testing.assert_allclose(result, mean_vals)

    def test_kappa_one_std_half_ucb_equals_mean_plus_half(self):
        """With kappa=1 and std=0.5, UCB = mean + 0.5."""
        GPSurrogate = _import_gp_surrogate()
        s = GPSurrogate()
        mean_vals = np.array([1.0, 2.0])
        std_vals = np.array([0.5, 0.5])
        X = np.random.rand(2, 2)
        with patch.object(s, 'predict', return_value=(mean_vals, std_vals)):
            result = s.predict_ucb(X, kappa=1.0, t=0, n_steps=10)
        expected = mean_vals + 0.5
        np.testing.assert_allclose(result, expected)

    def test_larger_kappa_gives_larger_ucb(self):
        """Larger kappa exploration coefficient yields larger UCB values."""
        GPSurrogate = _import_gp_surrogate()
        s = GPSurrogate()
        mean_vals = np.array([0.0, 0.0])
        std_vals = np.array([1.0, 1.0])
        X = np.random.rand(2, 2)
        with patch.object(s, 'predict', return_value=(mean_vals, std_vals)):
            ucb_low = s.predict_ucb(X, kappa=1.0, t=0, n_steps=10)
            ucb_high = s.predict_ucb(X, kappa=5.0, t=0, n_steps=10)
        assert np.all(ucb_high > ucb_low)
