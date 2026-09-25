"""Contract tests for the bucketized adapter and the external-model wrappers (task #8 Step 4).

The bar distribution is tested exactly — it is our code. The wrappers are tested
for their *contract*: registered, correctly labelled, and failing with an
actionable message when their backend is absent or the wrong version. Wrappers
whose backend is installed get a real fit/predict check via ``skipif``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.models.pfn import external
from pfns4neurostim.models.pfn.bar_distribution import BarDistribution, quantile_borders
from pfns4neurostim.models.registry import MODEL_REGISTRY, build_surrogate

EXTERNAL_KEYS = ("pfns4bo", "tabpfn_v1", "tabfm", "mitra", "tabflex", "tabicl")


class TestBarDistribution:
    """The bucketized-regression adapter, which the classifier models depend on."""

    def test_borders_are_strictly_increasing(self) -> None:
        y = np.random.default_rng(0).normal(size=200)
        borders = quantile_borders(y, 16)
        assert borders.size == 17
        assert (np.diff(borders) > 0).all()

    def test_tied_values_still_give_positive_width_bins(self) -> None:
        y = np.array([0.0] * 50 + [1.0] * 50)
        borders = quantile_borders(y, 10)
        assert (np.diff(borders) > 0).all()

    def test_constant_response_raises(self) -> None:
        with pytest.raises(ValueError, match="two distinct values"):
            quantile_borders(np.ones(20), 8)

    def test_point_mass_mean_is_the_bin_centre(self) -> None:
        bar = BarDistribution(np.array([0.0, 1.0, 2.0, 3.0]))
        probs = np.array([[0.0, 1.0, 0.0]])
        assert bar.mean(probs)[0] == pytest.approx(1.5)

    def test_point_mass_std_is_the_within_bin_spread(self) -> None:
        """A single-bin prediction is not certain: its resolution is the bin width."""
        bar = BarDistribution(np.array([0.0, 1.0, 2.0, 3.0]))
        probs = np.array([[0.0, 1.0, 0.0]])
        assert bar.std(probs)[0] == pytest.approx(1.0 / np.sqrt(12.0))

    def test_uniform_mass_recovers_the_range_mean(self) -> None:
        bar = BarDistribution(np.linspace(0.0, 4.0, 5))
        probs = np.full((1, 4), 0.25)
        assert bar.mean(probs)[0] == pytest.approx(2.0)

    def test_quantiles_are_monotone_and_inside_the_support(self) -> None:
        bar = BarDistribution(np.linspace(-2.0, 2.0, 9))
        probs = np.full((1, 8), 1.0 / 8.0)
        qs = [bar.quantile(probs, q)[0] for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
        assert all(a < b for a, b in zip(qs, qs[1:]))
        assert -2.0 <= qs[0] and qs[-1] <= 2.0
        assert qs[2] == pytest.approx(0.0, abs=1e-9)

    def test_samples_follow_the_mass(self) -> None:
        bar = BarDistribution(np.array([0.0, 1.0, 2.0]))
        probs = np.tile([0.9, 0.1], (4000, 1))
        draws = bar.sample(probs, np.random.default_rng(0))
        assert (draws < 1.0).mean() == pytest.approx(0.9, abs=0.02)

    def test_digitize_round_trips_into_valid_bins(self) -> None:
        y = np.random.default_rng(1).normal(size=100)
        bar = BarDistribution(quantile_borders(y, 8))
        labels = bar.digitize(y)
        assert labels.min() >= 0 and labels.max() < 8

    def test_expand_places_probabilities_at_the_right_bins(self) -> None:
        bar = BarDistribution(np.linspace(0.0, 3.0, 4))
        full = bar.expand(np.array([[0.3, 0.7]]), np.array([0, 2]))
        np.testing.assert_allclose(full, [[0.3, 0.0, 0.7]])

    def test_probability_count_mismatch_raises(self) -> None:
        bar = BarDistribution(np.linspace(0.0, 3.0, 4))
        with pytest.raises(ValueError, match="3 bins"):
            bar.mean(np.ones((1, 5)))

    def test_zero_mass_row_raises(self) -> None:
        bar = BarDistribution(np.linspace(0.0, 3.0, 4))
        with pytest.raises(RuntimeError, match="sums to zero"):
            bar.mean(np.zeros((1, 3)))


class TestExternalRegistration:
    """Every Hyp 0 model is registered, labelled and reachable by name."""

    @pytest.mark.parametrize("key", EXTERNAL_KEYS)
    def test_model_is_registered_with_a_version_string(self, key: str) -> None:
        assert key in MODEL_REGISTRY
        assert MODEL_REGISTRY[key].version

    @pytest.mark.parametrize("key", ("tabpfn_v1", "tabflex"))
    def test_classifier_models_are_labelled_as_adaptations(self, key: str) -> None:
        """The adaptation must be visible in the version string every table prints."""
        assert "classification-head adaptation" in MODEL_REGISTRY[key].version
        assert external.EXTERNAL_SPECS[key].route == "classification-head adaptation"

    @pytest.mark.parametrize("key", ("pfns4bo", "mitra", "tabicl"))
    def test_native_models_are_marked_native(self, key: str) -> None:
        """TabICL v2 ships a native regressor, so it is not a bucketized model."""
        assert external.EXTERNAL_SPECS[key].route == "native"

    def test_python_version_gate_is_reported_before_the_import(self) -> None:
        """A Python-version mismatch must be named, not surface as a SyntaxError."""
        import sys

        for key in ("tabicl", "tabfm"):
            spec = external.EXTERNAL_SPECS[key]
            assert spec.python_min is not None
            if sys.version_info[:2] < spec.python_min:
                with pytest.raises(ImportError, match="needs Python >="):
                    external.require_backend(key)

    @pytest.mark.parametrize(
        "key, env",
        (("pfns4bo", "main"), ("tabpfn_v1", "v1"), ("tabfm", "bench"), ("tabicl", "bench"),
         ("tabflex", "main"), ("mitra", "mitra")),
    )
    def test_every_model_declares_the_environment_it_runs_in(self, key: str, env: str) -> None:
        """Model -> env is data in one place, so no script has to name an env by hand."""
        spec = external.EXTERNAL_SPECS[key]
        assert spec.env == env
        expected = "pfns4neurostim" if env == "main" else f"pfns4neurostim-{env}"
        assert spec.conda_env == expected == external.conda_env_for(key)

    def test_wrong_environment_names_the_environment_to_activate(self) -> None:
        """A model that cannot run here must say where it does run, not just that it failed."""
        for key in ("tabpfn_v1", "tabfm", "tabicl"):
            ok, reason = external._backend_ok(external.EXTERNAL_SPECS[key])
            if not ok and "not installed" not in reason:
                assert external.EXTERNAL_SPECS[key].conda_env in reason

    def test_availability_reports_every_model(self) -> None:
        avail = external.availability()
        assert set(avail) == set(EXTERNAL_KEYS)
        assert all(isinstance(v, bool) for v in avail.values())

    def test_tabpfn_v1_is_not_satisfied_by_v2(self) -> None:
        """v1 and v2.5 share the module name; importability alone is not enough."""
        import tabpfn  # noqa: F401 - the v2.5 package is installed in this env

        assert external.availability()["tabpfn_v1"] is False
        with pytest.raises(ImportError, match="major version"):
            external.require_backend("tabpfn_v1")

    @pytest.mark.parametrize("key", EXTERNAL_KEYS)
    def test_unavailable_backend_says_how_to_get_it(self, key: str) -> None:
        """Either the pip extra, or the submodule to initialise - never a bare failure."""
        if external.availability()[key]:
            pytest.skip(f"{key} backend is available in this environment")
        with pytest.raises(ImportError, match=r"pip install -e|git submodule update"):
            build_surrogate(key)

    #: Models whose wrapper body is still outstanding (task #8 Step 3).
    PENDING = ("mitra",)

    @pytest.mark.parametrize("key", PENDING)
    def test_pending_wrapper_explains_what_is_outstanding(self, key: str) -> None:
        if not external.availability()[key]:
            pytest.skip(f"{key} backend is not available")
        surrogate = build_surrogate(key)
        with pytest.raises(NotImplementedError, match="not implemented yet|needs an environment"):
            surrogate.fit(np.zeros((4, 2)), np.arange(4.0))

    def test_implemented_wrappers_are_not_in_the_pending_list(self) -> None:
        """TabFlex, TabICL and TabFM since 2026-09-20; PFNs4BO 2026-09-23; TabPFN v1 2026-09-25."""
        assert set(self.PENDING).isdisjoint({"tabflex", "tabicl", "tabfm", "pfns4bo", "tabpfn_v1"})

    def test_tabfm_refuses_a_single_ensemble_member(self) -> None:
        """Its only uncertainty signal is ensemble spread, which is zero for one member."""
        from pfns4neurostim.models.pfn.wrappers import TabFMSurrogate

        with pytest.raises((ValueError, ImportError), match="n_estimators >= 2|needs Python"):
            TabFMSurrogate(n_estimators=1)


class _StubTabFM:
    """Stand-in for upstream ``TabFMRegressor`` after ``fit``: members live in a z-scale of the context.

    Mirrors upstream exactly where the wrapper depends on it: ``_predict_internal`` returns [E, N] in the
    internal scale, ``_combine_predictions`` averages then inverse-transforms, ``_compute_oof_preds_scaled``
    returns ([E, n], val_idx).
    """

    def __init__(self, y: np.ndarray, members_scaled: np.ndarray, oof_scaled: np.ndarray) -> None:
        self.mu, self.sd = float(np.mean(y)), float(np.std(y))
        self.members_scaled, self.oof_scaled = members_scaled, oof_scaled

    def _inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return np.asarray(y_scaled) * self.sd + self.mu

    def _combine_predictions(self, scaled: np.ndarray) -> np.ndarray:
        return self._inverse_transform_y(np.mean(scaled, axis=0))

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.members_scaled

    def _compute_oof_preds_scaled(self, cv: int = 5) -> tuple[np.ndarray, None]:
        return self.oof_scaled, None


class TestTabFMPredictiveOutputs:
    """The TabFM wrapper returns upstream's raw-scale prediction and a spread + out-of-fold SD (2026-09-25)."""

    @staticmethod
    def _wrapper(predictive_sd: str, y: np.ndarray, members: np.ndarray, oof: np.ndarray):
        from pfns4neurostim.models.pfn.wrappers import TabFMSurrogate

        w = TabFMSurrogate.__new__(TabFMSurrogate)       # skip the backend check: the stub stands in for it
        w.predictive_sd, w.oof_folds = predictive_sd, 5
        w._model = _StubTabFM(y, members, oof)
        w._oof_sd = w._oof_residual_sd(y) if predictive_sd == "spread_oof" else 0.0
        return w

    def test_mean_is_upstreams_raw_scale_prediction(self) -> None:
        y = np.array([10.0, 12.0, 14.0, 20.0])
        members = np.array([[0.0, 1.0], [0.2, 1.2]])        # [E=2, N=2] in the internal z-scale
        w = self._wrapper("spread", y, members, np.zeros((2, 4)))
        mean, std = w._predict_backend(np.zeros((2, 1)))
        np.testing.assert_allclose(mean, w._model._combine_predictions(members))
        assert mean[0] > 10.0                             # raw scale, not the internal z-scale
        np.testing.assert_allclose(std, np.std(members * w._model.sd, axis=0, ddof=1))

    def test_oof_residual_widens_the_sd(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        members = np.array([[0.0, 0.0], [0.0, 0.0]])        # no spread at all
        oof_scaled = np.tile(((y - y.mean()) / y.std() + 0.5)[None, :], (2, 1))   # OOF off by 0.5 internal sd
        w = self._wrapper("spread_oof", y, members, oof_scaled)
        _, std = w._predict_backend(np.zeros((2, 1)))
        np.testing.assert_allclose(std, 0.5 * y.std())    # RMSE of the OOF residual, in the raw scale

    @pytest.mark.parametrize("bad", [{"predictive_sd": "bogus"}, {"oof_folds": 1}])
    def test_invalid_options_raise(self, bad: dict) -> None:
        from pfns4neurostim.models.pfn.wrappers import TabFMSurrogate

        with pytest.raises((ValueError, ImportError), match="predictive_sd|oof_folds|needs Python"):
            TabFMSurrogate(**bad)


class TestQuantileMoments:
    """The quantile-integration used by the TabICL wrapper."""

    def test_normal_quantiles_recover_mean_and_sd(self) -> None:
        from scipy import stats

        from pfns4neurostim.models.pfn.wrappers import _QUANTILE_ALPHAS, _moments_from_quantiles

        q = stats.norm.ppf(np.asarray(_QUANTILE_ALPHAS), loc=2.0, scale=3.0)[None, :]
        mean, std = _moments_from_quantiles(q)
        assert mean[0] == pytest.approx(2.0, abs=0.02)
        assert std[0] == pytest.approx(3.0, rel=0.10)

    def test_degenerate_quantiles_give_near_zero_spread(self) -> None:
        from pfns4neurostim.models.pfn.wrappers import _moments_from_quantiles

        from pfns4neurostim.models.pfn.wrappers import _QUANTILE_ALPHAS

        q = np.full((1, len(_QUANTILE_ALPHAS)), 5.0)
        mean, std = _moments_from_quantiles(q)
        assert mean[0] == pytest.approx(5.0)
        assert std[0] < 1e-5

    def test_shape_mismatch_raises(self) -> None:
        from pfns4neurostim.models.pfn.wrappers import _moments_from_quantiles

        with pytest.raises(ValueError, match="quantiles for"):
            _moments_from_quantiles(np.zeros((2, 3)))


@pytest.mark.slow
class TestTabFlexRuns:
    """TabFlex actually fits and predicts (downloads weights on first use)."""

    def test_fit_predict_on_a_toy_grid(self) -> None:
        if not external.availability()["tabflex"]:
            pytest.skip("TabFlex backend unavailable (libs/ticl not initialised)")
        try:
            import ticl.prediction.tabpfn  # noqa: F401 - needs wandb/mlflow/interpret
        except ImportError as exc:
            pytest.skip(f"ticl dependencies missing: {exc}")
        import socket

        try:
            socket.gethostbyname("amuellermothernet.blob.core.windows.net")
        except OSError:
            pytest.skip("TabFlex weight host does not resolve (upstream microsoft/ticl issue #27)")
        rng = np.random.default_rng(0)
        X = rng.random((40, 2))
        y = np.exp(-((X[:, 0] - 0.7) ** 2 + (X[:, 1] - 0.3) ** 2) / 0.1)
        surrogate = build_surrogate("tabflex", device="cpu", n_bins=8)
        surrogate.fit(X[:20], y[:20])
        mean, std = surrogate.predict_marginals(X)
        assert mean.shape == (40,) and std.shape == (40,)
        assert np.isfinite(mean).all() and (std > 0).all()

class TestTabPFNv1Adaptation:
    """The v1 classification-head adaptation, which is testable without the v1 backend."""

    def test_bin_count_is_capped_at_the_architectural_class_ceiling(self) -> None:
        """v1 emits at most 10 classes; asking for more must fail at construction."""
        from pfns4neurostim.models.pfn.wrappers import TabPFNv1Surrogate

        assert TabPFNv1Surrogate.MAX_CLASSES == 10
        with pytest.raises(ValueError, match="exceeds the backend's ceiling of 10"):
            TabPFNv1Surrogate(n_bins=32)

    def test_default_bin_count_is_ten(self) -> None:
        """The default must be v1's ceiling, not the 32 used for backends without one."""
        import inspect

        from pfns4neurostim.models.pfn.wrappers import TabPFNv1Surrogate

        assert inspect.signature(TabPFNv1Surrogate.__init__).parameters["n_bins"].default == 10

    def test_fit_hook_passes_v1_overwrite_warning(self) -> None:
        """v1 refuses a context over its soft limit unless the flag is set."""
        from pfns4neurostim.models.pfn.wrappers import TabPFNv1Surrogate

        recorded: dict[str, object] = {}

        class _Classifier:
            def fit(self, X, y, overwrite_warning=False):  # noqa: ANN001, N803 - mirrors the v1 API
                recorded["overwrite_warning"] = overwrite_warning

        TabPFNv1Surrogate._fit_classifier(
            object.__new__(TabPFNv1Surrogate), _Classifier(), np.zeros((4, 2)), np.arange(4)
        )
        assert recorded["overwrite_warning"] is True

    def test_version_string_states_the_adaptation_and_its_resolution(self) -> None:
        """Guardrail G3: the bin count travels with every v1 row."""
        version = MODEL_REGISTRY["tabpfn_v1"].version
        assert "classification-head adaptation" in version
        assert "10" in version


class TestExternalErrors:
    """Failures of the external layer itself."""

    def test_unknown_external_key_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown external model"):
            external.require_backend("not_a_model")
