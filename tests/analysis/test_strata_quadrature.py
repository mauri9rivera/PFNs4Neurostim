"""Deterministic equal-mass strata and the tail measures of a bar-distribution predictive (amendment 2026-10-28)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("tabpfn")
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution  # noqa: E402

from pfns4neurostim.analysis import bayes_update as B  # noqa: E402
from pfns4neurostim.analysis.context import draw_context  # noqa: E402
from pfns4neurostim.data.synthetic_neurostim import gp_channel  # noqa: E402

N_BARS = 300
FIXED = {"lengthscale": 0.2, "outputscale": 1.0, "noise": 0.09}


def _criterion() -> FullSupportBarDistribution:
    inner = torch.linspace(-3.0, 3.0, N_BARS - 1)
    return FullSupportBarDistribution(torch.cat([torch.tensor([-60.0]), inner, torch.tensor([60.0])]))


def _logits(tail: float = 1e-3) -> torch.Tensor:
    x = torch.linspace(-3.0, 3.0, N_BARS)
    p = torch.exp(-0.5 * ((x - 0.3) / 0.7) ** 2)
    p = p / p.sum() * (1.0 - 2.0 * tail)
    p[0], p[-1] = tail, tail
    return p.log()[None, :]                                                        # [1, N_BARS]


def _predictive(measure: str) -> B.CriterionPredictive:
    return B.CriterionPredictive([(_criterion(), _logits())], tail_measure=measure)


def _wmean_var(y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m = float(w @ y)
    return m, float(w @ (y - m) ** 2)


class TestGaussStrata:
    def test_mean_is_exact_and_variance_approaches_one(self) -> None:
        for K, floor in ((16, 0.97), (64, 0.995)):
            y, w = B.gauss_strata_nodes(2.0, 3.0, K)
            m, v = _wmean_var(y, w)
            assert m == pytest.approx(2.0, abs=1e-12)
            assert floor < v / 9.0 < 1.0                                             # within-stratum variance is the missing part

    def test_one_stratum_is_the_mean(self) -> None:
        y, w = B.gauss_strata_nodes(1.5, 2.0, 1)
        assert y == pytest.approx([1.5]) and w == pytest.approx([1.0])


class TestTailMeasures:
    def test_halfnormal_moments_are_the_criterions_own(self) -> None:
        crit, lg = _criterion(), _logits()
        mean, var = B.bar_moments(crit, lg, "halfnormal")
        assert mean[0] == pytest.approx(float(crit.mean(lg)), rel=1e-5, abs=1e-6)
        assert var[0] == pytest.approx(float(crit.variance(lg)), rel=1e-4)

    def test_measures_are_ordered_and_central_drops_the_edges(self) -> None:
        v = {m: B.bar_moments(_criterion(), _logits(), m)[1][0] for m in B.TAIL_MEASURES}
        assert v["central"] < v["uniform"] < v["halfnormal"]
        p = torch.softmax(_logits()[0].double(), -1)
        b = _criterion().borders.double()
        pc = p[1:-1] / p[1:-1].sum()
        lo, hi = b[1:-2], b[2:-1]
        m1 = (pc * (lo + hi) / 2).sum()
        m2 = (pc * (lo ** 2 + lo * hi + hi ** 2) / 3).sum()
        assert v["central"] == pytest.approx(float(m2 - m1 ** 2), rel=1e-9)

    def test_unknown_measure_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tail_measure"):
            B.CriterionPredictive([(_criterion(), _logits())], tail_measure="gaussian")

    def test_take_keeps_the_measure(self) -> None:
        pred = B.CriterionPredictive([(_criterion(), _logits())] * 2, tail_measure="central")
        assert pred.take([1]).tail_measure == "central"


class TestStrataNodes:
    @pytest.mark.parametrize("measure", B.TAIL_MEASURES)
    def test_nodes_reproduce_the_measure_mean_and_nearly_its_variance(self, measure: str) -> None:
        pred = _predictive(measure)
        y, w = pred.strata_nodes(0, 64)
        assert w.sum() == pytest.approx(1.0)
        m, v = _wmean_var(y, w)
        assert m == pytest.approx(float(pred.mean[0, 0]), abs=2e-5)
        assert 0.97 < v / float(pred.var[0, 0]) <= 1.0 + 1e-6

    def test_variance_converges_with_the_number_of_strata(self) -> None:
        pred = _predictive("central")
        err = [abs(1.0 - _wmean_var(*pred.strata_nodes(0, K))[1] / float(pred.var[0, 0])) for K in (8, 32, 128)]
        assert err[0] > err[1] > err[2]

    def test_central_nodes_stay_inside_the_interior(self) -> None:
        y, w = _predictive("central").strata_nodes(0, 32)
        b = _criterion().borders
        assert y.min() > float(b[1]) and y.max() < float(b[-2]) and np.allclose(w, 1 / 32)

    def test_halfnormal_tail_nodes_sit_beyond_the_uniform_ones(self) -> None:
        yh, _ = _predictive("halfnormal").strata_nodes(0, 32)
        yu, _ = _predictive("uniform").strata_nodes(0, 32)
        assert yh.max() > yu.max() > 3.0 and yh.min() < yu.min() < -3.0


class TestStrataOnAGp:
    @pytest.fixture(scope="class")
    def setup(self):
        ch = gp_channel(10, 0.2, 0.3, 10, np.random.default_rng(1))
        ctx = draw_context(ch, 25, np.random.default_rng(0))
        eng = B.GPFixedEngine(**FIXED)
        eng.fit(ctx.X, ctx.y)
        anchors = np.concatenate([np.setdiff1d(np.arange(ch.n_sites), ctx.sites)[:3], ctx.sites[:2]])
        return ch, eng, anchors

    def test_moment_matched_readouts_are_exact_and_raw_ones_nearly(self, setup) -> None:
        ch, eng, anchors = setup
        rows, C, Cc = B.coherence(eng, ch.X_pool, anchors, K=64, eps=0.01, method="strata")
        kC = eng.posterior_covariance(ch.X_pool)[anchors]
        np.testing.assert_allclose(Cc, kC, atol=1e-8)                                # exact for any node set (affine refit)
        np.testing.assert_allclose(C, kC, atol=2e-3)                                 # raw: only the within-stratum variance is missing
        for r in rows:
            assert r["max_abs_drift_c"] < 1e-8 and r["max_abs_total_var_dev_c"] < 1e-8
            assert 0.99 < r["measure_var_ratio"] <= 1.0 + 1e-9

    def test_a_predictive_without_a_stratum_rule_is_refused(self) -> None:
        with pytest.raises(NotImplementedError):
            B.Predictive().strata_nodes(0, 8)
