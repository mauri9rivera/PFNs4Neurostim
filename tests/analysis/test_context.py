"""Tests for the shared Hyp C context draw (task plan #19 Step 1).

The whole value of this module is a pair of invariants: the *sites* must not depend on the channel or the
stress level (that is what lets every analysis share the work that depends only on the sites), and the
*trials* must depend on both (that is the part which genuinely differs per channel). Everything else here
guards the identity used as the sharing key.
"""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.analysis.context import context_for, context_sites, grid_id
from pfns4neurostim.data.channels import ChannelData

SEED = 42


def _channel(label: str, n_sites: int = 12, n_reps: int = 4, offset: float = 0.0) -> ChannelData:
    """A minimal channel on a 1D grid, with distinguishable per-site trial banks."""
    rng = np.random.default_rng(abs(hash(label)) % 2**32)
    X = np.linspace(0.0, 1.0, n_sites)[:, None]                          # [N, 1]
    y_gt = np.sin(3.0 * X[:, 0]) + offset                                # [N]
    trials = y_gt[:, None] + 0.1 * rng.normal(size=(n_sites, n_reps))    # [N, R]
    subject, emg = int(label.split("-")[0]), int(label.split("-")[1])
    return ChannelData(
        dataset="test", subject=subject, emg=emg, X_pool=X, y_gt=y_gt, Y_trials=trials,
        ch2xy=np.arange(n_sites)[:, None], grid_shape=(n_sites,), normalization="pfn",
    )


class TestGridId:
    def test_same_coordinates_give_the_same_id(self) -> None:
        a, b = _channel("0-0"), _channel("1-1")
        assert grid_id(a.X_pool) == grid_id(b.X_pool)

    def test_different_coordinates_give_different_ids(self) -> None:
        assert grid_id(np.zeros((4, 2))) != grid_id(np.ones((4, 2)))

    def test_id_is_stable_across_calls_and_layout(self) -> None:
        X = np.random.default_rng(0).random((7, 3))
        assert grid_id(X) == grid_id(np.asarray(X, order="F"))


class TestSitesAreShared:
    def test_sites_do_not_depend_on_the_channel(self) -> None:
        """The property every cross-channel reuse rests on."""
        a, b = _channel("0-0"), _channel("3-2", offset=5.0)
        for t in (2, 5, 11):
            assert np.array_equal(
                context_sites(a.X_pool, t, 0, base_seed=SEED),
                context_sites(b.X_pool, t, 0, base_seed=SEED),
            )

    def test_different_draws_give_different_sites(self) -> None:
        first = context_sites(_channel("0-0").X_pool, 5, 0, base_seed=SEED)
        second = context_sites(_channel("0-0").X_pool, 5, 1, base_seed=SEED)
        assert not np.array_equal(first, second)

    def test_sites_are_sorted_distinct_and_in_range(self) -> None:
        sites = context_sites(_channel("0-0").X_pool, 7, 3, base_seed=SEED)
        assert sites.size == 7
        assert np.array_equal(sites, np.sort(sites)) and np.unique(sites).size == 7
        assert sites.min() >= 0 and sites.max() < 12

    def test_a_different_base_seed_moves_the_sites(self) -> None:
        assert not np.array_equal(
            context_sites(_channel("0-0").X_pool, 5, 0, base_seed=SEED),
            context_sites(_channel("0-0").X_pool, 5, 0, base_seed=SEED + 1),
        )

    @pytest.mark.parametrize("t", [0, 1, 13])
    def test_out_of_range_context_size_fails_fast(self, t: int) -> None:
        with pytest.raises(ValueError, match="must be in"):
            context_sites(_channel("0-0").X_pool, t, 0, base_seed=SEED)


class TestTrialsArePerChannel:
    def test_two_channels_share_sites_but_not_trials(self) -> None:
        a, b = _channel("0-0"), _channel("3-2", offset=5.0)
        ca = context_for(a, 6, 0, base_seed=SEED)
        cb = context_for(b, 6, 0, base_seed=SEED)
        assert np.array_equal(ca.sites, cb.sites)
        assert not np.allclose(ca.y, cb.y)

    def test_the_stress_level_changes_the_trials_not_the_sites(self) -> None:
        ch = _channel("0-0")
        nominal = context_for(ch, 6, 0, base_seed=SEED, level=1.0)
        stressed = context_for(ch, 6, 0, base_seed=SEED, level=4.0)
        assert np.array_equal(nominal.sites, stressed.sites)
        assert not np.allclose(nominal.y, stressed.y)

    def test_the_draw_is_reproducible(self) -> None:
        ch = _channel("0-0")
        first = context_for(ch, 6, 2, base_seed=SEED, level=1.0)
        again = context_for(ch, 6, 2, base_seed=SEED, level=1.0)
        assert np.array_equal(first.sites, again.sites) and np.allclose(first.y, again.y)

    def test_x_matches_the_sites(self) -> None:
        ch = _channel("0-0")
        ctx = context_for(ch, 5, 1, base_seed=SEED)
        assert np.allclose(ctx.X, ch.X_pool[ctx.sites])

    def test_every_drawn_value_is_a_real_trial_of_its_site(self) -> None:
        ch = _channel("0-0")
        ctx = context_for(ch, 8, 0, base_seed=SEED)
        for site, value in zip(ctx.sites, ctx.y):
            assert value in set(ch.Y_trials[site].tolist())
