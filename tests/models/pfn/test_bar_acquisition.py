"""Closed-form EI / PI / UCB over a ``FullSupportBarDistribution`` (``models.pfn.bar_acquisition``)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import ndtr

from pfns4neurostim.models.pfn.bar_acquisition import (
    bar_ei, bar_pi, bar_ucb, merge_zero_width_buckets, outer_bucket_share, ucb_level,
)
from pfns4neurostim.models.pfn.tabpfn import bar_quantiles

pytest.importorskip("tabpfn")
from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution  # noqa: E402

N_BARS = 300
DUP = 150                       # index of the duplicated border: one zero-width bucket, as in the TabPFN-2.5 checkpoint
N_MC = 400_000


def _criterion(duplicate: bool = True) -> FullSupportBarDistribution:
    """Borders of a standard-normal-ish support with wide outer buckets and (optionally) one duplicated border."""
    inner = torch.linspace(-3.0, 3.0, N_BARS - 1)                                   # [N_BARS - 1]
    borders = torch.cat([torch.tensor([-60.0]), inner, torch.tensor([60.0])])       # [N_BARS + 1]
    if duplicate:
        borders[DUP + 1] = borders[DUP]
    return FullSupportBarDistribution(borders)


def _logits(n_sites: int, seed: int = 0, tail_mass: float = 1e-3) -> torch.Tensor:
    """Logits of ``n_sites`` bell-shaped bar distributions with a given mass in each outer bucket."""
    g = torch.Generator().manual_seed(seed)
    centre = torch.linspace(-1.0, 1.0, n_sites)[:, None]                            # [S, 1]
    x = torch.linspace(-3.0, 3.0, N_BARS)[None, :]                                  # [1, N_BARS]
    p = torch.exp(-0.5 * ((x - centre) / 0.7) ** 2) + 0.01 * torch.rand(n_sites, N_BARS, generator=g)
    p = p / p.sum(-1, keepdim=True) * (1.0 - 2.0 * tail_mass)
    p[:, 0], p[:, -1] = tail_mass, tail_mass
    return p.log()                                                                  # [S, N_BARS]


def _sample(crit: FullSupportBarDistribution, logits: torch.Tensor, n: int, seed: int) -> np.ndarray:
    """Draws from the distribution the criterion defines: uniform inside interior buckets, half-normal outer tails."""
    rng = np.random.default_rng(seed)
    p = torch.softmax(logits.double(), -1).numpy()
    b = crit.borders.double().numpy()
    idx = rng.choice(len(p), size=n, p=p / p.sum())
    y = b[idx] + (b[idx + 1] - b[idx]) * rng.random(n)
    left = crit.halfnormal_with_p_weight_before(crit.bucket_widths[0])
    right = crit.halfnormal_with_p_weight_before(crit.bucket_widths[-1])
    for side, tail in ((0, left), (len(p) - 1, right)):
        sel = idx == side
        h = np.abs(rng.normal(size=int(sel.sum()))) * float(tail.scale)
        y[sel] = b[1] - h if side == 0 else b[-2] + h
    return y


def test_merge_removes_the_zero_width_bucket_and_keeps_the_mass() -> None:
    crit, lg = _criterion(), _logits(5)
    merged, lp = merge_zero_width_buckets(crit, lg)
    assert merged.borders.numel() == crit.borders.numel() - 1
    assert bool((merged.borders[1:] > merged.borders[:-1]).all())
    assert torch.allclose(torch.softmax(lp.double(), -1).sum(-1), torch.ones(5, dtype=torch.float64))
    # the duplicated bucket's mass sits on the next bucket
    p0, p1 = torch.softmax(lg.double(), -1), torch.softmax(lp.double(), -1)
    assert torch.allclose(p1[:, DUP], p0[:, DUP] + p0[:, DUP + 1], atol=1e-9)


def test_upstream_ei_is_nan_with_a_zero_width_bucket_and_bar_ei_is_not() -> None:
    crit, lg = _criterion(), _logits(5)
    assert bool(torch.isnan(crit.ei(lg, 0.3)).all())                                # the upstream defect this module repairs
    assert np.isfinite(bar_ei(crit, lg, 0.3)).all()


def test_no_zero_width_bucket_leaves_the_criterion_untouched() -> None:
    crit, lg = _criterion(duplicate=False), _logits(3)
    merged, _ = merge_zero_width_buckets(crit, lg)
    assert merged is crit


def test_merge_refuses_a_zero_width_edge_bucket() -> None:
    crit = _criterion(duplicate=False)
    crit.borders[1] = crit.borders[0]
    with pytest.raises(ValueError, match="edge bucket"):
        merge_zero_width_buckets(crit, _logits(2))


@pytest.mark.parametrize("incumbent", [-0.5, 0.4, 2.0])
def test_bar_ei_and_pi_match_monte_carlo(incumbent: float) -> None:
    crit, lg = _criterion(), _logits(3)
    merged, lp = merge_zero_width_buckets(crit, lg)
    ei, pi = bar_ei(crit, lg, incumbent), bar_pi(crit, lg, incumbent)
    for s in range(3):
        y = _sample(merged, lp[s], N_MC, seed=s)
        assert ei[s] == pytest.approx(np.mean(np.maximum(y - incumbent, 0.0)), rel=0.03, abs=2e-3)
        assert pi[s] == pytest.approx(np.mean(y > incumbent), abs=4e-3)


def test_bar_ei_is_monotone_in_the_incumbent_and_non_negative() -> None:
    crit, lg = _criterion(), _logits(4)
    grid = [bar_ei(crit, lg, f) for f in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    assert all((a >= b - 1e-9).all() for a, b in zip(grid[:-1], grid[1:]))
    assert (grid[-1] >= 0.0).all()


@pytest.mark.parametrize("kappa", [0.0, 1.0, 2.0])
def test_bar_ucb_is_the_phi_kappa_quantile(kappa: float) -> None:
    crit, lg = _criterion(duplicate=False), _logits(4)
    q = bar_ucb(crit, lg, kappa)
    assert q.shape == (4,)
    ref = bar_quantiles(crit, lg, np.array([float(ndtr(kappa))]))[:, 0]
    assert np.allclose(q, ref)
    assert np.allclose(q, crit.ucb(lg, 0.0, rest_prob=float(ndtr(-kappa)), maximize=True).numpy(), atol=1e-3)


def test_ucb_level_is_clipped_below_one() -> None:
    assert ucb_level(50.0) < 1.0
    assert ucb_level(-50.0) > 0.0


def test_outer_bucket_share_flags_a_quantile_inside_the_outer_bucket() -> None:
    crit = _criterion(duplicate=False)
    heavy = _logits(6, tail_mass=1e-3)               # right outer bucket holds 1e-3 > 1 - Phi(2)? no: 1e-3 < 0.0228 but > Phi(-5)
    assert outer_bucket_share(crit, heavy, 5.0) == 1.0
    assert outer_bucket_share(crit, heavy, 1.0) == 0.0
