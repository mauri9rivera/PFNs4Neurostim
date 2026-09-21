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
    PENDING = ("pfns4bo", "tabpfn_v1", "mitra")

    @pytest.mark.parametrize("key", PENDING)
    def test_pending_wrapper_explains_what_is_outstanding(self, key: str) -> None:
        if not external.availability()[key]:
            pytest.skip(f"{key} backend is not available")
        surrogate = build_surrogate(key)
        with pytest.raises(NotImplementedError, match="not implemented yet|needs an environment"):
            surrogate.fit(np.zeros((4, 2)), np.arange(4.0))

    def test_implemented_wrappers_are_not_in_the_pending_list(self) -> None:
        """TabFlex, TabICL and TabFM have real bodies as of 2026-09-20."""
        assert set(self.PENDING).isdisjoint({"tabflex", "tabicl", "tabfm"})

    def test_tabfm_refuses_a_single_ensemble_member(self) -> None:
        """Its only uncertainty signal is ensemble spread, which is zero for one member."""
        from pfns4neurostim.models.pfn.wrappers import TabFMSurrogate

        with pytest.raises((ValueError, ImportError), match="n_estimators >= 2|needs Python"):
            TabFMSurrogate(n_estimators=1)


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

    def test_unknown_external_key_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown external model"):
            external.require_backend("not_a_model")
