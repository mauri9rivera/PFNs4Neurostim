"""``FrozenTabPFN`` precision flag and train-mode fingerprint replay (F6-E Step 2).

Both features were added for sprint #3's measurement gate (F6-0 A1 / A4): a ``precision``
flag so a CUDA arm can hold the device fixed while precision changes, and ``transform_x``'s
``as_train`` replay so an injected update-rule probe row can get the fingerprint hash a real
refit on ``context U {x}`` would assign it, instead of the test-mode default, which (per
``tabpfn/preprocessing/steps/add_fingerprint_features_step.py``) hashes ``row + 2*salt`` with
no collision check at all -- so an injected row is indistinguishable, in fingerprint space,
from any other row with the same raw content transformed the same way (e.g. the anchor's own
query-grid row).

These tests load the real ``tabpfn==6.3.2`` checkpoint and run on CPU only (no ``gpu`` marker);
marked ``slow`` for the same reason as ``tests/models/test_y_affine_invariance.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pfns4neurostim.models.pfn.tabpfn import FrozenTabPFN

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def toy_context() -> tuple[np.ndarray, np.ndarray]:
    """A tiny, non-constant 2D context: ``(X_ctx [6, 2], y_ctx [6])``."""
    rng = np.random.default_rng(0)
    X = rng.random((6, 2))
    y = rng.random(6)
    return X, y


@pytest.fixture(scope="module")
def fitted(toy_context: tuple[np.ndarray, np.ndarray]) -> FrozenTabPFN:
    """A ``FrozenTabPFN`` fitted on :func:`toy_context` with the default ``precision='auto'``."""
    X, y = toy_context
    model = FrozenTabPFN(device="cpu", random_state=0)
    model.fit(X, y)
    return model


def _without_fingerprint_step(random_state: int = 0) -> FrozenTabPFN:
    """A ``FrozenTabPFN`` whose regressor has ``AddFingerprintFeaturesStep`` disabled.

    ``FrozenTabPFN.__init__`` does not expose ``inference_config`` (it only forwards
    ``device``, ``random_state``, ``max_batch_tokens`` and ``precision`` to the wrapped
    ``TabPFNRegressor``), so the only public-API lever that controls the step
    (``inference_config_.FINGERPRINT_FEATURE``, read at ``tabpfn/regressor.py:644``) is not
    reachable through ``FrozenTabPFN`` alone. This swaps in an otherwise identically
    configured ``TabPFNRegressor`` before ``fit()`` -- the same technique
    ``tests/models/test_pfn_wrappers.py`` uses to stand in for an upstream model.
    """
    from tabpfn import TabPFNRegressor  # noqa: PLC0415 - test-local import

    model = FrozenTabPFN(device="cpu", random_state=random_state)
    model._reg = TabPFNRegressor(
        device="cpu",
        n_estimators=1,
        softmax_temperature=1.0,
        ignore_pretraining_limits=True,
        fit_mode="fit_preprocessors",
        random_state=random_state,
        inference_precision="auto",
        inference_config={"FINGERPRINT_FEATURE": False},
    )
    return model


# ---------------------------------------------------------------------------
# precision: "auto" vs "fp32"
# ---------------------------------------------------------------------------
class TestPrecision:
    """On CPU, autocast is already off regardless of ``precision`` (``forward`` only enables
    ``torch.autocast`` when ``self._autocast and self.device.type == 'cuda'``), so the two
    settings must be a no-op here -- the thing worth pinning down, since it is easy to get
    backwards and assume the flag does something on every device.
    """

    def test_fp32_is_a_no_op_on_cpu(self, toy_context: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = toy_context
        X_query = np.random.default_rng(1).random((5, 2))

        auto = FrozenTabPFN(device="cpu", random_state=0, precision="auto")
        auto.fit(X, y)
        fp32 = FrozenTabPFN(device="cpu", random_state=0, precision="fp32")
        fp32.fit(X, y)

        # Autocast is off for both on CPU -- the shared premise the rest of the test checks.
        assert auto._autocast is False
        assert fp32._autocast is False

        Xt_auto = auto.transform_x(X_query)
        Xt_fp32 = fp32.transform_x(X_query)
        assert torch.equal(Xt_auto, Xt_fp32)

        logp_auto = auto.forward(None, None, Xt_auto)
        logp_fp32 = fp32.forward(None, None, Xt_fp32)
        assert torch.equal(logp_auto, logp_fp32)

        mean_auto, sd_auto = auto.moments(logp_auto)
        mean_fp32, sd_fp32 = fp32.moments(logp_fp32)
        np.testing.assert_array_equal(mean_auto, mean_fp32)
        np.testing.assert_array_equal(sd_auto, sd_fp32)

    def test_invalid_precision_string_raises_at_construction(self) -> None:
        with pytest.raises(ValueError, match="precision"):
            FrozenTabPFN(device="cpu", precision="fp16")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# transform_x(as_train=...)
# ---------------------------------------------------------------------------
class TestTransformXAsTrain:
    def test_as_train_changes_only_the_fingerprint_column(self, fitted: FrozenTabPFN) -> None:
        X_query = np.random.default_rng(2).random((4, 2))
        out_test = fitted.transform_x(X_query, as_train=False).cpu().numpy()
        out_train = fitted.transform_x(X_query, as_train=True).cpu().numpy()

        assert out_test.shape == out_train.shape
        differing_cols = np.where(~np.isclose(out_test, out_train).all(axis=0))[0]
        assert differing_cols.shape == (1,), (
            f"expected exactly one differing column (the fingerprint), got {differing_cols}"
        )

    def test_as_train_reproduces_the_pinned_train_mode_hash(self, fitted: FrozenTabPFN) -> None:
        """Independently replays the fitted pipeline's own steps (not ``transform_x`` itself)
        and calls the fingerprint step's ``_transform(..., is_test=False)`` directly on the
        reconstructed ``context U probe`` array, mirroring what a real refit's fingerprint
        step would do. A divergence here (e.g. a wrong concatenation order, or routing through
        ``is_test=True``) would change the *numbers*, not just "whether it differs from
        test-mode", which the other tests in this class cannot catch.
        """
        from tabpfn.preprocessing.steps import AddFingerprintFeaturesStep

        X_ctx, _ = (fitted._X_ctx_raw, None)
        probe = np.random.default_rng(3).random((3, 2))

        steps = list(fitted._member.cpu_preprocessor.steps)
        fp_idx = next(i for i, s in enumerate(steps) if isinstance(s, AddFingerprintFeaturesStep))

        ctx_pre, probe_pre = X_ctx.astype(np.float64), probe.astype(np.float64)
        for step in steps[:fp_idx]:
            ctx_pre, _ = step.transform(ctx_pre)
            probe_pre, _ = step.transform(probe_pre)
        combined_fp = steps[fp_idx]._transform(                       # noqa: SLF001 - pinned internal
            np.concatenate([ctx_pre, probe_pre], axis=0), is_test=False
        )
        expected = combined_fp[len(ctx_pre):]
        for step in steps[fp_idx + 1:]:
            expected, _ = step.transform(expected)

        actual = fitted.transform_x(probe, as_train=True).cpu().numpy()
        np.testing.assert_allclose(expected, actual, atol=1e-5)

    def test_as_train_resolves_a_hash_collision_against_the_context(self, fitted: FrozenTabPFN) -> None:
        """A probe row that exactly duplicates a context row's raw content forces the
        fingerprint step's collision-avoidance counter (``add_fingerprint_features_step.py``
        ``hash_counter`` / ``seen_hashes``) to fire in train mode, giving the probe a hash
        that differs from the context row it duplicates -- exactly what a real refit would do,
        and exactly what the test-mode path (no collision tracking at all) cannot.
        """
        dup_row = fitted._X_ctx_raw[2:3].copy()

        out_test = fitted.transform_x(dup_row, as_train=False).cpu().numpy()
        out_train = fitted.transform_x(dup_row, as_train=True).cpu().numpy()
        fp_col = int(np.where(~np.isclose(out_test, out_train))[1][0])

        ctx_fingerprint = fitted.X_ctx.cpu().numpy()[2, fp_col]
        train_fingerprint = out_train[0, fp_col]
        assert not np.isclose(train_fingerprint, ctx_fingerprint), (
            "train-mode fingerprint for a duplicated row must differ from the context "
            "row it duplicates once the collision-avoidance counter fires"
        )

    def test_as_train_demonstrates_the_test_mode_duplicate(self, fitted: FrozenTabPFN) -> None:
        """The bug ``as_train`` exists to avoid: test mode's fingerprint is a pure function of
        the row's own content and the model's fixed salt, with no reference to context at all
        -- so the *same* raw point transformed twice in test mode (once as a readout-grid
        query, once as an injected probe) is bit-identical, and train mode is not.
        """
        x_star = np.random.default_rng(4).random((1, 2))
        as_query = fitted.transform_x(x_star, as_train=False).cpu().numpy()
        as_probe = fitted.transform_x(x_star, as_train=False).cpu().numpy()
        np.testing.assert_array_equal(as_query, as_probe)

        as_train_mode = fitted.transform_x(x_star, as_train=True).cpu().numpy()
        assert not np.allclose(as_train_mode, as_query)

    def test_as_train_before_fit_raises(self) -> None:
        unfitted = FrozenTabPFN(device="cpu")
        with pytest.raises(RuntimeError, match="used before fit"):
            unfitted.transform_x(np.zeros((1, 2)), as_train=True)

    def test_as_train_without_a_fingerprint_step_raises(self) -> None:
        model = _without_fingerprint_step()
        X = np.random.default_rng(0).random((6, 2))
        y = np.random.default_rng(0).random(6)
        model.fit(X, y)
        with pytest.raises(RuntimeError, match="no AddFingerprintFeaturesStep"):
            model.transform_x(np.zeros((1, 2)), as_train=True)


# ---------------------------------------------------------------------------
# decode_logits: the post-head chain factored out of forward()
# ---------------------------------------------------------------------------
class TestDecodeLogits:
    """``forward`` = raw decoder logits -> ``decode_logits``. Feeding the raw logits (captured by
    hooking ``decoder_dict['standard']``, i.e. independently of ``forward``'s own post-processing)
    back through ``decode_logits`` must reproduce ``forward`` to float32 round-trip precision."""

    @staticmethod
    def _forward_and_raw_logits(model: FrozenTabPFN, X_query: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        captured: list[torch.Tensor] = []

        def hook(_mod, _inp, out) -> None:
            captured.append(out.detach().float().cpu().clone())          # [q, b, bars]

        handle = model._model.decoder_dict["standard"].register_forward_hook(hook)
        try:
            logp = model.forward(None, None, model.transform_x(X_query))   # [1, q, bars]
        finally:
            handle.remove()
        assert len(captured) == 1
        return logp, captured[0].transpose(0, 1).contiguous()              # [1, q, bars]

    def test_raw_logits_through_decode_logits_reproduce_forward(self, fitted: FrozenTabPFN) -> None:
        X_query = np.random.default_rng(5).random((6, 2))
        logp, raw = self._forward_and_raw_logits(fitted, X_query)
        redecoded = fitted.decode_logits(raw.to(fitted.device))
        assert redecoded.shape == logp.shape
        finite = torch.isfinite(logp)
        assert torch.equal(finite, torch.isfinite(redecoded)), "-inf (zero-probability) pattern must match"
        np.testing.assert_allclose(
            redecoded[finite].cpu().numpy(), logp[finite].cpu().numpy(), atol=1e-5
        )

    def test_moments_of_the_redecoded_logits_match_forward(self, fitted: FrozenTabPFN) -> None:
        X_query = np.random.default_rng(6).random((5, 2))
        logp, raw = self._forward_and_raw_logits(fitted, X_query)
        mean_f, sd_f = fitted.moments(logp)
        mean_d, sd_d = fitted.moments(fitted.decode_logits(raw.to(fitted.device)))
        np.testing.assert_allclose(mean_d, mean_f, atol=1e-5)
        np.testing.assert_allclose(sd_d, sd_f, atol=1e-5)

    def test_accepts_arbitrary_leading_dimensions(self, fitted: FrozenTabPFN) -> None:
        n_bars = fitted._znorm_borders.numel() - 1
        out = fitted.decode_logits(torch.randn(7, n_bars))
        assert out.shape == (7, n_bars)
        out3 = fitted.decode_logits(torch.randn(2, 3, n_bars))
        assert out3.shape == (2, 3, n_bars)

    def test_output_is_a_normalised_log_distribution(self, fitted: FrozenTabPFN) -> None:
        n_bars = fitted._znorm_borders.numel() - 1
        out = fitted.decode_logits(torch.randn(4, n_bars))
        np.testing.assert_allclose(out.exp().sum(dim=-1).numpy(), 1.0, atol=1e-4)

    def test_does_not_mutate_its_input(self, fitted: FrozenTabPFN) -> None:
        n_bars = fitted._znorm_borders.numel() - 1
        logits = torch.randn(3, n_bars)
        before = logits.clone()
        fitted.decode_logits(logits)
        assert torch.equal(logits, before)

    def test_nan_logits_raise(self, fitted: FrozenTabPFN) -> None:
        n_bars = fitted._znorm_borders.numel() - 1
        logits = torch.randn(2, n_bars)
        logits[0, 3] = float("nan")
        with pytest.raises(RuntimeError, match="NaN"):
            fitted.decode_logits(logits)

    def test_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="used before fit"):
            FrozenTabPFN(device="cpu").decode_logits(torch.zeros(1, 4))
