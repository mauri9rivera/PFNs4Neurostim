"""Capture layer of :mod:`pfns4neurostim.analysis.embeddings` (moved from the Sprint #3 module, 2026-10-15).

``capture_block_states``, ``capture_head_states`` and ``logit_lens`` hook a fitted ``FrozenTabPFN``; they load the
real ``tabpfn==6.3.2`` checkpoint on CPU and are marked ``slow``. Only the logit-lens argument check runs without it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pfns4neurostim.analysis import embeddings as E


# ---------------------------------------------------------------------------
# logit_lens: argument validation (no checkpoint needed)
# ---------------------------------------------------------------------------
class TestLogitLensValidation:
    def test_mismatched_sign_sequences_raise_value_error(self) -> None:
        plus = [np.zeros((3, 192)) for _ in range(4)]
        minus = [np.zeros((3, 192)) for _ in range(3)]
        with pytest.raises(ValueError, match="4 blocks for \\+c but 3 for -c"):
            E.logit_lens(None, plus, minus)


# ===========================================================================
# Hook-based functions: need the real tabpfn==6.3.2 checkpoint (CPU, slow)
# ===========================================================================
@pytest.fixture(scope="module")
def toy_context() -> tuple[np.ndarray, np.ndarray]:
    """A tiny, non-constant 2D context: ``(X_ctx [6, 2], y_ctx [6])``."""
    rng = np.random.default_rng(0)
    X = rng.random((6, 2))
    y = rng.random(6)
    return X, y


@pytest.fixture(scope="module")
def engine(toy_context: tuple[np.ndarray, np.ndarray]):
    """A ``FrozenTabPFN`` fitted on :func:`toy_context`, CPU only."""
    from pfns4neurostim.models.pfn.tabpfn import FrozenTabPFN

    X, y = toy_context
    eng = FrozenTabPFN(device="cpu", random_state=0)
    eng.fit(X, y)
    return eng


class TestCaptureBlockStates:
    pytestmark = pytest.mark.slow

    def test_shapes_and_requested_layers_are_all_present(self, engine) -> None:
        X_query = np.random.default_rng(1).random((5, 2))
        n_blocks = len(engine._model.transformer_encoder.layers)
        out = E.capture_block_states(engine, X_query, layers=[0, 5, n_blocks - 1])
        assert out.layers == (0, 5, n_blocks - 1)
        assert len(out.feature_tokens) == len(out.label_token) == 3
        for feat, lab in zip(out.feature_tokens, out.label_token):
            assert feat.shape[0] == 5
            assert lab.shape == (5, feat.shape[2])

    def test_none_layers_captures_every_block(self, engine) -> None:
        X_query = np.random.default_rng(2).random((3, 2))
        n_blocks = len(engine._model.transformer_encoder.layers)
        out = E.capture_block_states(engine, X_query)
        assert out.layers == tuple(range(n_blocks))

    def test_deterministic_given_the_same_inputs(self, engine) -> None:
        X_query = np.random.default_rng(3).random((4, 2))
        a = E.capture_block_states(engine, X_query, layers=[0, 3])
        b = E.capture_block_states(engine, X_query, layers=[0, 3])
        for fa, fb in zip(a.feature_tokens, b.feature_tokens):
            np.testing.assert_array_equal(fa, fb)
        for la, lb in zip(a.label_token, b.label_token):
            np.testing.assert_array_equal(la, lb)

    def test_injecting_an_extra_row_changes_the_captured_state(self, engine) -> None:
        X_query = np.random.default_rng(4).random((4, 2))
        base = E.capture_block_states(engine, X_query, layers=[1])
        extra = E.capture_block_states(
            engine, X_query, layers=[1], X_extra=np.array([[0.5, 0.5]]), y_extra=[1.0]
        )
        assert not np.allclose(base.feature_tokens[0], extra.feature_tokens[0])

    def test_invalid_layer_index_raises(self, engine) -> None:
        X_query = np.random.default_rng(5).random((2, 2))
        with pytest.raises(IndexError):
            E.capture_block_states(engine, X_query, layers=[999])


class TestCaptureHeadStates:
    pytestmark = pytest.mark.slow

    def test_shapes(self, engine) -> None:
        X_query = np.random.default_rng(6).random((4, 2))
        heads = E.capture_head_states(engine, X_query)
        assert heads.h1.shape == (4, 192)
        assert heads.h2.shape == (4, 384)
        assert heads.h3.shape[0] == 4

    def test_decoder_applied_to_h1_reproduces_h3(self, engine) -> None:
        """H1 and H3 are the decoder's own input and output in one forward pass, so applying
        ``decoder_dict['standard']`` to H1 must give H3 (float32 round-trip only)."""
        X_query = np.random.default_rng(7).random((4, 2))
        heads = E.capture_head_states(engine, X_query)
        decoder = engine._model.decoder_dict["standard"]
        with torch.no_grad():
            out = decoder(torch.as_tensor(heads.h1, dtype=torch.float32, device=engine.device))
        np.testing.assert_allclose(out.cpu().numpy(), heads.h3, atol=1e-4)


class TestLogitLens:
    pytestmark = pytest.mark.slow

    X_EXTRA = np.array([[0.5, 0.5]])

    @staticmethod
    def _surprises(toy_context: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
        y_mean = float(np.mean(toy_context[1]))
        return y_mean + 0.4, y_mean - 0.4

    def _label_tokens(self, engine, toy_context, X_query):
        y_plus, y_minus = self._surprises(toy_context)
        plus = E.capture_block_states(engine, X_query, X_extra=self.X_EXTRA, y_extra=[y_plus])
        minus = E.capture_block_states(engine, X_query, X_extra=self.X_EXTRA, y_extra=[y_minus])
        return plus.label_token, minus.label_token

    def _real_antisymmetric_update(self, engine, toy_context, X_query) -> np.ndarray:
        """The model's actual ``(mean_+ - mean_-) / 2`` from two real forward passes."""
        y_plus, y_minus = self._surprises(toy_context)
        Qt = engine.transform_x(X_query)
        Xe = engine.transform_x(self.X_EXTRA).unsqueeze(0)              # [1, 1, F]
        means = []
        for y_val in (y_plus, y_minus):
            ye = engine.transform_y(torch.tensor([[y_val]], dtype=torch.float32, device=engine.device))
            mean, _ = engine.moments(engine.forward(Xe, ye, Qt))        # [1, q]
            means.append(mean[0])
        return 0.5 * (means[0] - means[1])                              # [q]

    def test_shape_is_blocks_by_query_sites_and_finite(self, engine, toy_context) -> None:
        X_query = np.random.default_rng(11).random((5, 2))
        plus, minus = self._label_tokens(engine, toy_context, X_query)
        lens = E.logit_lens(engine, plus, minus)
        n_blocks = len(engine._model.transformer_encoder.layers)
        assert lens.shape == (n_blocks, 5)
        assert np.isfinite(lens).all()

    def test_last_block_equals_the_models_real_antisymmetrised_prediction_change(
        self, engine, toy_context
    ) -> None:
        """Decoding block 17's label token is exactly what the real forward pass does, so the
        lens value at the final block must equal the real antisymmetrised update (float32
        round-trip only; measured agreement ~1.5e-5)."""
        X_query = np.random.default_rng(12).random((5, 2))
        plus, minus = self._label_tokens(engine, toy_context, X_query)
        lens = E.logit_lens(engine, plus, minus)
        real = self._real_antisymmetric_update(engine, toy_context, X_query)
        assert np.abs(real).max() > 0  # the injected row actually moves the prediction
        np.testing.assert_allclose(lens[-1], real, atol=1e-4)

    def test_lens_is_antisymmetric_under_swapping_the_two_signs(self, engine, toy_context) -> None:
        X_query = np.random.default_rng(13).random((4, 2))
        plus, minus = self._label_tokens(engine, toy_context, X_query)
        np.testing.assert_allclose(
            E.logit_lens(engine, plus, minus), -E.logit_lens(engine, minus, plus), atol=1e-12
        )

    def test_identical_sign_sequences_give_a_zero_update(self, engine, toy_context) -> None:
        X_query = np.random.default_rng(14).random((3, 2))
        plus, _ = self._label_tokens(engine, toy_context, X_query)
        np.testing.assert_array_equal(E.logit_lens(engine, plus, plus), 0.0)
