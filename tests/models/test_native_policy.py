"""End-to-end BO models (native policy): generic mechanism, capability routing, PFNs4BO.

An end-to-end BO model owns its query decision. It is integrated through one capability
(``policy_scores``), one acquisition type (``native``) and one registry flag
(``native_policy``), so the shared loop never names a model. These tests pin that design:

* the mechanism is exercised with a toy model (fast, no weights), including
  seed determinism and the routing matrix of ``_supported``;
* a structural test asserts the loop and the routing code contain no model key;
* the real PFNs4BO checkpoint runs end to end on the 20-site fixture (skipped when the backend
  or the vendored checkpoint is absent).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from pfns4neurostim.acquisition.registry import ACQUISITION_REGISTRY, build_acquisition
from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.evaluation import bo_loop
from pfns4neurostim.evaluation.bo_loop import run_bo_loop
from pfns4neurostim.evaluation.bo_runner import run_channel_bo
from pfns4neurostim.experiments.bo_benchmark import _supported
from pfns4neurostim.models.pfn.wrappers import PFNS4BO_DEFAULT_CHECKPOINT
from pfns4neurostim.models.protocol import NativePolicy, SurrogateAdapter
from pfns4neurostim.models.registry import MODEL_REGISTRY

N_SITES = 20
BUDGET = 9
N_INIT = 3


class _ToyNativeModel:
    """A model that scores the pool by closeness to a hidden hotspot (its own 'policy')."""

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._mean = float(np.mean(y))

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full(X.shape[0], self._mean), np.ones(X.shape[0])

    def policy_scores(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return -np.abs(X[:, 0] - 0.7) + 1e-3 * rng.random(X.shape[0])


class _PlainModel:
    """A surrogate with no native policy."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(X.shape[0]), np.ones(X.shape[0])


@pytest.fixture(scope="module")
def channel() -> ChannelData:
    """The 20-site, 6-trial synthetic channel used by the integration tests."""
    rng = np.random.default_rng(0)
    coords = np.stack(np.meshgrid(np.arange(5), np.arange(4)), axis=-1).reshape(-1, 2)  # [20, 2]
    x = coords / np.array([4.0, 3.0])                                                    # [20, 2]
    y_gt = np.exp(-((x[:, 0] - 0.7) ** 2 + (x[:, 1] - 0.3) ** 2) / 0.1)                  # [20]
    y_gt = (y_gt - y_gt.mean()) / y_gt.std()
    Y = y_gt[:, None] + 0.25 * rng.normal(size=(N_SITES, 6))                              # [20, 6]
    return ChannelData("nhp", 1, 0, x, Y, y_gt, coords, (5, 4))


def _run(model: object, channel: ChannelData, seed: int = 0) -> list[int]:
    spec, params = build_acquisition("native")
    adapter = SurrogateAdapter(model, key="toy", family="pfn")
    traj = run_bo_loop(
        adapter, channel, spec, params, budget=BUDGET, n_init=N_INIT, rng=np.random.default_rng(seed)
    )
    return traj.observed_indices


class TestNativePolicyMechanism:
    """The generic path, with a toy model (no weights)."""

    def test_capability_is_detected_structurally(self) -> None:
        assert isinstance(_ToyNativeModel(), NativePolicy)
        assert not isinstance(_PlainModel(), NativePolicy)
        assert SurrogateAdapter(_ToyNativeModel(), "toy", "pfn").has_native_policy
        assert not SurrogateAdapter(_PlainModel(), "plain", "pfn").has_native_policy

    def test_loop_follows_the_model_policy(self, channel: ChannelData) -> None:
        idx = _run(_ToyNativeModel(), channel)
        assert len(idx) == BUDGET
        # After the random initial design the policy chases the hotspot column (x0 = 0.75).
        assert all(abs(channel.X_pool[i, 0] - 0.7) <= 0.2 for i in idx[N_INIT:])

    def test_deterministic_under_seed(self, channel: ChannelData) -> None:
        assert _run(_ToyNativeModel(), channel, seed=3) == _run(_ToyNativeModel(), channel, seed=3)

    def test_plain_surrogate_is_refused(self, channel: ChannelData) -> None:
        with pytest.raises(NotImplementedError, match="owns its query decision"):
            _run(_PlainModel(), channel)

    def test_malformed_scores_fail_fast(self) -> None:
        class _Bad(_ToyNativeModel):
            def policy_scores(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
                return np.full(X.shape[0], np.nan)

        adapter = SurrogateAdapter(_Bad(), "bad", "pfn")
        with pytest.raises(RuntimeError, match="finite"):
            adapter.policy_scores(np.zeros((4, 2)), np.random.default_rng(0))


class TestRouting:
    """Capability flags, not model names, decide who serves what."""

    def test_native_models_serve_only_native_and_vice_versa(self) -> None:
        native_models = [k for k, s in MODEL_REGISTRY.items() if s.native_policy]
        native_acqs = [k for k, s in ACQUISITION_REGISTRY.items() if s.needs_native_policy]
        assert native_models and native_acqs
        for model in MODEL_REGISTRY:
            for acq in ACQUISITION_REGISTRY:
                expected_native = model in native_models and acq in native_acqs
                if model in native_models or acq in native_acqs:
                    assert _supported(model, acq) is expected_native, (model, acq)

    def test_existing_routing_is_unchanged(self) -> None:
        assert _supported("random", "random") and not _supported("gp_mll", "random")
        assert _supported("gp_mll", "ts_joint") and not _supported("tabpfn_v2_5", "ts_joint")
        assert _supported("tabpfn_v2_5", "ts_marginal") and not _supported("random", "ts_marginal")

    def test_registry_flag_matches_declared_support(self) -> None:
        for key, spec in MODEL_REGISTRY.items():
            if spec.native_policy:
                assert set(spec.supports) == {
                    a for a, s in ACQUISITION_REGISTRY.items() if s.needs_native_policy
                }, key

    def test_loop_and_routing_name_no_model(self) -> None:
        """No model key appears as a string literal in the loop or in the routing function."""
        acq_names = set(ACQUISITION_REGISTRY)
        model_keys = set(MODEL_REGISTRY) - acq_names        # 'random' is also an acquisition name
        sources = {
            "bo_loop.py": Path(bo_loop.__file__).read_text(encoding="utf-8"),
            "_supported": inspect.getsource(_supported),
        }
        for where, text in sources.items():
            for key in model_keys:
                assert not re.search(rf"""["']{re.escape(key)}["']""", text), f"{where} names {key!r}"
            assert "family" not in text or where == "_supported"   # the loop never branches on family


def _pfns4bo_available() -> bool:
    try:
        import pfns4bo  # noqa: F401
    except Exception:
        return False
    from pfns4neurostim.models.pfn import wrappers

    return (wrappers._REPO_ROOT / PFNS4BO_DEFAULT_CHECKPOINT).is_file()


@pytest.mark.skipif(not _pfns4bo_available(), reason="pfns4bo backend or vendored checkpoint missing")
class TestPFNs4BO:
    """The real checkpoint through the shared runner."""

    def test_end_to_end_on_the_tiny_fixture(self, channel: ChannelData) -> None:
        res = run_channel_bo(
            "pfns4bo", channel, acq_fn="native", budget=BUDGET, n_init=N_INIT, seed=1, device="cpu"
        )
        assert res.row["acq_type"] == "native"
        assert len(res.trajectory["observed_indices"]) == BUDGET
        assert len(res.trajectory["recommended_regret_per_step"]) == BUDGET - N_INIT + 1
        assert np.isfinite(res.row["r2"]) and np.isfinite(res.row["coverage_90"])

    def test_policy_is_deterministic(self, channel: ChannelData) -> None:
        kwargs = dict(acq_fn="native", budget=BUDGET, n_init=N_INIT, seed=2, device="cpu")
        a = run_channel_bo("pfns4bo", channel, **kwargs)
        b = run_channel_bo("pfns4bo", channel, **kwargs)
        assert a.trajectory["observed_indices"] == b.trajectory["observed_indices"]

    def test_plain_acquisition_is_refused_for_a_native_model(self, channel: ChannelData) -> None:
        assert not _supported("pfns4bo", "ei")
        assert not _supported("gp_mll", "native")
