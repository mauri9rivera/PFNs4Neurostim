"""A bar-capable surrogate is scored over its bar distribution; every other surrogate keeps the Gaussian forms."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("tabpfn")
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution  # noqa: E402

from pfns4neurostim.acquisition import build_acquisition  # noqa: E402
from pfns4neurostim.acquisition.base import BOState  # noqa: E402
from pfns4neurostim.models.pfn.bar_acquisition import bar_ei, bar_pi, bar_ucb  # noqa: E402
from pfns4neurostim.models.registry import MODEL_REGISTRY  # noqa: E402

N_BARS, SITES = 200, 6


class _BarSurrogate:
    """Fixed bar distributions with the surface the registry reads (``predict_marginals``, ``predict_bars``)."""

    def __init__(self, form: str = "bars") -> None:
        inner = torch.linspace(-3.0, 3.0, N_BARS - 1)
        self.crit = FullSupportBarDistribution(torch.cat([torch.tensor([-50.0]), inner, torch.tensor([50.0])]))
        centre = torch.linspace(-1.0, 1.0, SITES)[:, None]
        p = torch.exp(-0.5 * ((torch.linspace(-3, 3, N_BARS)[None] - centre) / 0.8) ** 2) + 1e-4
        self.logits = (p / p.sum(-1, keepdim=True)).log()
        self.has_bar_predictive = form == "bars"
        mean = self.crit.mean(self.logits).numpy().astype(np.float64)
        self._m, self._s = mean, np.sqrt(self.crit.variance(self.logits).numpy().astype(np.float64))
        self.family, self.key = "pfn", "fake_bar"

    def predict_marginals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._m, self._s

    def predict_bars(self, X: np.ndarray) -> tuple[FullSupportBarDistribution, torch.Tensor]:
        return self.crit, self.logits


def _state() -> BOState:
    return BOState((0, 1), (0.1, 0.4), 0, 10, n_dims=2, fit_values=(0.1, 0.4))


X = np.zeros((SITES, 2))


def test_ei_pi_ucb_use_the_bar_forms_for_a_bar_surrogate() -> None:
    s, st, rng = _BarSurrogate(), _state(), np.random.default_rng(0)
    spec, p = build_acquisition("ei", {"xi": 0.0})
    assert np.allclose(spec.score_fn(s, X, st, rng, p), bar_ei(s.crit, s.logits, 0.4))
    spec, p = build_acquisition("pi", {"xi": 0.0})
    best_pred = float(np.max(s._m[[0, 1]]))
    assert np.allclose(spec.score_fn(s, X, st, rng, p), bar_pi(s.crit, s.logits, best_pred))
    spec, p = build_acquisition("ucb", {"kappa": 2.0})
    assert np.allclose(spec.score_fn(s, X, st, rng, p), bar_ucb(s.crit, s.logits, 2.0))


def test_moments_form_keeps_the_gaussian_versions() -> None:
    s, st, rng = _BarSurrogate(form="moments"), _state(), np.random.default_rng(0)
    spec, p = build_acquisition("ucb", {"kappa": 2.0})
    assert np.allclose(spec.score_fn(s, X, st, rng, p), s._m + 2.0 * s._s)


def test_only_the_bar_capable_model_changes_its_identity() -> None:
    assert MODEL_REGISTRY["tabpfn_v2_5"].bar_acquisition
    assert not any(spec.bar_acquisition for k, spec in MODEL_REGISTRY.items() if k != "tabpfn_v2_5")
