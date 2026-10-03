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

EXTERNAL_KEYS = ("pfns4bo", "tabfm", "mitra", "tabflex", "tabicl", "tabpfn_v3_5", "causilo")


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

    @pytest.mark.parametrize("key", ("tabflex",))
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
        (("pfns4bo", "main"), ("tabfm", "bench"), ("tabicl", "bench"),
         ("tabflex", "main"), ("mitra", "mitra"), ("tabpfn_v3_5", "latest"), ("causilo", "latest")),
    )
    def test_every_model_declares_the_environment_it_runs_in(self, key: str, env: str) -> None:
        """Model -> env is data in one place, so no script has to name an env by hand."""
        spec = external.EXTERNAL_SPECS[key]
        assert spec.env == env
        expected = "pfns4neurostim" if env == "main" else f"pfns4neurostim-{env}"
        assert spec.conda_env == expected == external.conda_env_for(key)

    def test_wrong_environment_names_the_environment_to_activate(self) -> None:
        """A model that cannot run here must say where it does run, not just that it failed."""
        for key in ("tabpfn_v3_5", "tabfm", "tabicl"):
            ok, reason = external._backend_ok(external.EXTERNAL_SPECS[key])
            if not ok and "not installed" not in reason:
                assert external.EXTERNAL_SPECS[key].conda_env in reason

    def test_availability_reports_every_model(self) -> None:
        avail = external.availability()
        assert set(avail) == set(EXTERNAL_KEYS)
        assert all(isinstance(v, bool) for v in avail.values())

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
        """TabFlex, TabICL and TabFM since 2026-09-20; PFNs4BO 2026-09-23; TabPFN-3.5 and Causilo 2026-10-03."""
        assert set(self.PENDING).isdisjoint({"tabflex", "tabicl", "tabfm", "pfns4bo", "tabpfn_v3_5", "causilo"})

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

class TestExternalErrors:
    """Failures of the external layer itself."""

    def test_unknown_external_key_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown external model"):
            external.require_backend("not_a_model")

class TestLatestEnvironmentModels:
    """TabPFN-3.5 and Causilo (added 2026-10-03): both live in the ``latest`` environment."""

    def test_tabpfn_v3_5_needs_a_newer_tabpfn_than_the_main_env_pins(self) -> None:
        """Same module name as TabPFN-2.5: importability alone must not make it 'available'."""
        spec = external.EXTERNAL_SPECS["tabpfn_v3_5"]
        assert spec.module == "tabpfn" and spec.major_min == 9 and spec.env == "latest"
        import importlib.metadata as md

        if int(md.version("tabpfn").split(".")[0]) < 9:
            ok, reason = external._backend_ok(spec)
            # Either gate may fire first (Python < 3.10 in the py3.9 main env, else the version gate); both name the env.
            assert not ok and spec.conda_env in reason
            assert "needs Python >=" in reason or "major version >= 9" in reason
            with pytest.raises(ImportError, match=r"needs Python >=|major version"):
                external.require_backend("tabpfn_v3_5")

    def test_both_models_share_one_environment(self) -> None:
        envs = {external.EXTERNAL_SPECS[k].conda_env for k in ("tabpfn_v3_5", "causilo")}
        assert envs == {"pfns4neurostim-latest"}

    def test_version_strings_name_the_checkpoint_and_ensemble_size(self) -> None:
        assert "n_estimators=1" in MODEL_REGISTRY["tabpfn_v3_5"].version
        assert "n_estimators=1" in MODEL_REGISTRY["causilo"].version

    def test_causilo_integrates_its_quantile_function(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """mean/std come from the quantile function; Thompson draws invert it."""
        from scipy.stats import norm

        from pfns4neurostim.models.pfn import wrappers

        class _Regressor:
            def __init__(self, device: str, n_estimators: int, **kw: object) -> None:
                self.args = (device, n_estimators)

            def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803
                self.fitted = True

            def predict(self, X: np.ndarray, output_type: str, quantiles: list) -> np.ndarray:  # noqa: N803
                assert output_type == "quantiles"
                mu = X[:, 0:1]                                       # [N, 1]
                return mu + 0.5 * norm.ppf(np.asarray(quantiles))[None, :]   # [N, Q]

        class _Backend:
            CausiloRegressor = _Regressor

        monkeypatch.setattr(external, "require_backend", lambda key: _Backend)   # the base class calls this one
        monkeypatch.setattr(wrappers, "require_backend", lambda key: _Backend)
        model = wrappers.CausiloSurrogate(device="cpu")
        model.fit(np.zeros((4, 2)), np.arange(4.0))
        X = np.array([[0.0, 0.0], [2.0, 0.0]])
        mean, std = model.predict_marginals(X)
        np.testing.assert_allclose(mean, [0.0, 2.0], atol=1e-2)
        assert (std > 0.4).all() and (std < 0.55).all()               # N(mu, 0.5^2), truncated to +-2.3 SD
        draws = np.stack([model.sample_marginal(X, np.random.default_rng(s)) for s in range(400)])
        assert draws.mean(axis=0) == pytest.approx([0.0, 2.0], abs=0.15)

    def test_causilo_rejects_a_changed_upstream_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pfns4neurostim.models.pfn import wrappers

        class _Regressor:
            def __init__(self, **kw: object) -> None: ...
            def fit(self, X, y) -> None: ...  # noqa: ANN001, N803
            def predict(self, X, **kw):  # noqa: ANN001, ANN201, N803
                return np.zeros((X.shape[0], 999))                    # the raw grid, not the requested levels

        class _Backend:
            CausiloRegressor = _Regressor

        monkeypatch.setattr(external, "require_backend", lambda key: _Backend)
        monkeypatch.setattr(wrappers, "require_backend", lambda key: _Backend)
        model = wrappers.CausiloSurrogate()
        model.fit(np.zeros((4, 2)), np.arange(4.0))
        with pytest.raises(RuntimeError, match="expected"):
            model.predict_marginals(np.zeros((3, 2)))


class TestOwnPredictiveSampling:
    """Thompson draws come from a model's OWN predictive, not a Gaussian fitted to its mean and std (2026-10-03).

    Until then ``SurrogateAdapter.sample_marginal`` looked only for ``predict_ts_marginal`` / ``predict_ts``, so the
    ``sample_marginal`` methods of TabICL, Causilo and the bucketized classifiers were never called.
    """

    def test_adapter_prefers_the_wrapped_models_own_sampler(self) -> None:
        from pfns4neurostim.models.protocol import SurrogateAdapter

        class _Model:
            calls: list = []

            def fit(self, X, y) -> None: ...  # noqa: ANN001, N803
            def predict(self, X):  # noqa: ANN001, ANN201, N803
                return np.zeros(len(X)), np.ones(len(X))
            def sample_marginal(self, X, rng, temperature=1.0):  # noqa: ANN001, ANN201, N803
                _Model.calls.append(temperature)
                return np.full(len(X), 7.0)

        adapter = SurrogateAdapter(_Model(), key="stub", family="pfn")
        draws = adapter.sample_marginal(np.zeros((5, 2)), np.random.default_rng(0), temperature=2.0)
        np.testing.assert_allclose(draws, 7.0)                # not a N(0, 1) draw
        assert _Model.calls == [2.0]

    def test_adapter_still_falls_back_to_a_gaussian_without_any_sampler(self) -> None:
        from pfns4neurostim.models.protocol import SurrogateAdapter

        class _Model:
            def fit(self, X, y) -> None: ...  # noqa: ANN001, N803
            def predict(self, X):  # noqa: ANN001, ANN201, N803
                return np.full(len(X), 3.0), np.full(len(X), 0.01)

        draws = SurrogateAdapter(_Model(), key="stub", family="pfn").sample_marginal(
            np.zeros((200, 2)), np.random.default_rng(0)
        )
        assert abs(draws.mean() - 3.0) < 0.01

    def test_inverse_cdf_draws_reproduce_a_skewed_predictive(self) -> None:
        """A lognormal predictive: the Gaussian summary has the right mean/std but the wrong shape."""
        from scipy.stats import lognorm

        from pfns4neurostim.models.pfn import wrappers

        levels = wrappers._midpoint_levels(200)
        q = lognorm.ppf(np.asarray(levels), s=0.8)[None, :].repeat(4000, axis=0)        # [4000, 200]
        draws = wrappers._inverse_cdf_draws(q, levels, np.random.default_rng(0), 1.0)    # [4000]
        truth = lognorm.rvs(s=0.8, size=200_000, random_state=1)
        for level in (0.1, 0.5, 0.9):
            assert np.quantile(draws, level) == pytest.approx(np.quantile(truth, level), rel=0.08)
        assert np.quantile(draws, 0.9) > 2.0 * np.quantile(draws, 0.1)                    # skewed, unlike a Gaussian draw

    def test_temperature_scales_the_spread_around_the_mean(self) -> None:
        from pfns4neurostim.models.pfn import wrappers

        levels = wrappers._midpoint_levels(100)
        q = np.tile(np.linspace(-1.0, 1.0, 100), (3000, 1))
        cold = wrappers._inverse_cdf_draws(q, levels, np.random.default_rng(0), 0.25)
        hot = wrappers._inverse_cdf_draws(q, levels, np.random.default_rng(0), 4.0)
        assert hot.std() == pytest.approx(4.0 * cold.std(), rel=0.05)
        with pytest.raises(ValueError, match="temperature"):
            wrappers._inverse_cdf_draws(q, levels, np.random.default_rng(0), 0.0)

    def test_tabicl_asks_for_the_dense_grid_when_sampling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sampling must not reuse the 15-level summary grid, which collapses the outer 2 % of mass."""
        from scipy.stats import norm

        from pfns4neurostim.models.pfn import wrappers

        seen: dict[str, int] = {}

        class _Regressor:
            def __init__(self, device: str, n_estimators: int, **kw: object) -> None: ...
            def fit(self, X, y) -> None: ...  # noqa: ANN001, N803
            def predict(self, X, output_type: str, alphas: list):  # noqa: ANN001, ANN201, N803
                seen["n_levels"] = len(alphas)
                return X[:, 0:1] + norm.ppf(np.asarray(alphas))[None, :]

        class _Backend:
            TabICLRegressor = _Regressor

        monkeypatch.setattr(external, "require_backend", lambda key: _Backend)
        monkeypatch.setattr(wrappers, "require_backend", lambda key: _Backend)
        model = wrappers.TabICLSurrogate(n_sample_levels=120)
        model.fit(np.zeros((4, 2)), np.arange(4.0))
        draws = np.stack([model.sample_marginal(np.array([[0.0, 0.0], [5.0, 0.0]]), np.random.default_rng(s)) for s in range(300)])
        assert seen["n_levels"] == 120
        assert draws.mean(axis=0) == pytest.approx([0.0, 5.0], abs=0.2)
        assert draws.std(axis=0)[0] == pytest.approx(1.0, abs=0.15)

    def test_tabicl_version_string_changed_so_old_gaussian_cells_cannot_be_served(self) -> None:
        assert "own quantile predictive" in MODEL_REGISTRY["tabicl"].version
