"""Tests for the K7 spatial-shuffle knob (roadmap S7, task plan #10 Step 14).

K7 removes the one assumption every surrogate here relies on -- that nearby electrodes give similar responses
-- and *nothing else*. That "nothing else" is most of what is tested below: the response marginals, the
ground-truth range and the achieved SNR must come through untouched, or a drop in performance could not be
attributed to the loss of spatial structure rather than to a harder problem.
"""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.data.snr import achieved_snr_db
from pfns4neurostim.data.stress import KnobNotApplicable, build_knob
from pfns4neurostim.data.synthetic_neurostim import morans_i

LEVELS = (0.0, 0.1, 0.25, 0.5, 1.0)


def _channel(side: int = 8, n_reps: int = 6, seed: int = 0) -> ChannelData:
    """A smooth 2D bump on a ``side x side`` grid with heteroscedastic trials."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)          # [N, 2]
    centre = np.array([side / 3.0, side / 2.0])
    y_gt = np.exp(-((coords - centre) ** 2).sum(axis=1) / (2.0 * (side / 4.0) ** 2))   # [N]
    # A tiny monotone ramp breaks the ties a radially symmetric bump has on a grid. Needed because one test
    # identifies where a response came from by its value, which is only unambiguous when the values are
    # distinct; it does not otherwise change what is being tested (the ramp is 1e-4 of the response range).
    y_gt = y_gt + 1e-4 * np.arange(y_gt.size) / max(y_gt.size - 1, 1)
    trials = y_gt[:, None] + (0.15 * y_gt[:, None] + 0.02) * rng.normal(size=(y_gt.size, n_reps))
    return ChannelData(
        dataset="test", subject=0, emg=0, X_pool=coords / max(side - 1, 1), y_gt=y_gt,
        Y_trials=trials, ch2xy=coords.astype(int), grid_shape=(side, side), normalization="pfn",
    )


@pytest.fixture()
def channel() -> ChannelData:
    ch = _channel()
    assert np.unique(ch.y_gt).size == ch.n_sites, "the fixture must have distinct responses"
    return ch


@pytest.fixture()
def knob():
    return build_knob("k7_shuffle", LEVELS)


class TestTheKnobIsRegisteredAndValidated:
    def test_it_is_implemented_and_alters_the_map(self, knob) -> None:
        assert knob.implemented and knob.alters == "map"

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_a_fraction_outside_the_unit_interval_fails_fast(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be in"):
            build_knob("k7_shuffle", (bad,))

    def test_a_channel_with_one_site_cannot_be_shuffled(self) -> None:
        one = _channel(side=1)
        with pytest.raises(KnobNotApplicable, match="at least two sites"):
            build_knob("k7_shuffle", (1.0,)).apply(one, 1.0, np.random.default_rng(0))


class TestOnlyTheBindingChanges:
    def test_the_nominal_level_reproduces_the_channel_exactly(self, channel, knob) -> None:
        got = knob.apply(channel, 0.0, np.random.default_rng(0))
        assert np.array_equal(got.y_gt, channel.y_gt)
        assert np.array_equal(np.nan_to_num(got.Y_trials), np.nan_to_num(channel.Y_trials))

    @pytest.mark.parametrize("f", [0.1, 0.25, 0.5, 1.0])
    def test_the_response_multiset_is_preserved(self, channel, knob, f: float) -> None:
        """A permutation, not a perturbation: the same responses, at other coordinates."""
        got = knob.apply(channel, f, np.random.default_rng(0))
        assert np.allclose(np.sort(got.y_gt), np.sort(channel.y_gt))

    @pytest.mark.parametrize("f", [0.1, 0.25, 0.5, 1.0])
    def test_the_ground_truth_range_is_unchanged(self, channel, knob, f: float) -> None:
        """Regret is divided by this range (P0.9), so the metric stays comparable across the ladder."""
        got = knob.apply(channel, f, np.random.default_rng(0))
        assert np.ptp(got.y_gt) == pytest.approx(np.ptp(channel.y_gt))

    @pytest.mark.parametrize("f", [0.1, 0.25, 0.5, 1.0])
    def test_the_achieved_snr_is_unchanged(self, channel, knob, f: float) -> None:
        """A site's trials move with its mean, so the noise structure is carried along, not rebuilt."""
        got = knob.apply(channel, f, np.random.default_rng(0))
        assert achieved_snr_db(got) == pytest.approx(achieved_snr_db(channel), abs=1e-9)

    def test_the_coordinates_never_move(self, channel, knob) -> None:
        got = knob.apply(channel, 1.0, np.random.default_rng(0))
        assert np.array_equal(got.X_pool, channel.X_pool)
        assert np.array_equal(got.ch2xy, channel.ch2xy)

    def test_a_site_keeps_its_own_trial_bank_with_its_mean(self, channel, knob) -> None:
        """The pairing of a mean with its trials is what makes the noise model travel intact."""
        got = knob.apply(channel, 1.0, np.random.default_rng(0))
        for site in range(channel.n_sites):
            # Exact equality, not isclose: the knob COPIES values, so the match is bit-for-bit, and a
            # tolerance would make neighbouring responses ambiguous sources (they are 1e-6 apart here).
            sources = np.flatnonzero(channel.y_gt == got.y_gt[site])
            assert sources.size == 1, "the fixture's responses must identify their site uniquely"
            assert np.allclose(got.Y_trials[site], channel.Y_trials[int(sources[0])])


class TestTheLadderIsNestedAndExact:
    def test_the_shuffled_set_grows_monotonically(self, channel, knob) -> None:
        """Common random numbers: level f re-pairs a superset of level f' < f (review finding R13)."""
        moved = []
        for f in (0.1, 0.25, 0.5, 1.0):
            got = knob.apply(channel, f, np.random.default_rng(0))
            moved.append(set(np.flatnonzero(~np.isclose(got.y_gt, channel.y_gt)).tolist()))
        for smaller, larger in zip(moved, moved[1:]):
            assert smaller <= larger

    @pytest.mark.parametrize("f", [0.25, 0.5, 1.0])
    def test_exactly_the_requested_fraction_moves(self, channel, knob, f: float) -> None:
        """The cycle has no fixed points, so the achieved fraction is exact, not an expectation."""
        got = knob.apply(channel, f, np.random.default_rng(0))
        assert knob.achieved(got)["achieved_shuffle"] == pytest.approx(f)
        assert (~np.isclose(got.y_gt, channel.y_gt)).sum() == int(round(f * channel.n_sites))

    def test_a_fraction_too_small_to_re_pair_is_reported_as_no_stress(self, knob) -> None:
        """Rather than claiming a severity the run does not realize (review finding R2)."""
        small = _channel(side=2)                       # 4 sites: f = 0.25 selects one
        got = build_knob("k7_shuffle", (0.25,)).apply(small, 0.25, np.random.default_rng(0))
        assert knob.achieved(got)["achieved_shuffle"] == 0.0
        assert np.array_equal(got.y_gt, small.y_gt)

    def test_the_draw_does_not_depend_on_the_passed_generator(self, channel, knob) -> None:
        """Common random numbers come from a channel-keyed stream, so the ladder is paired."""
        a = knob.apply(channel, 0.5, np.random.default_rng(0))
        b = knob.apply(channel, 0.5, np.random.default_rng(999))
        assert np.array_equal(a.y_gt, b.y_gt)


class TestTheAchievedStructure:
    def test_moran_i_falls_with_the_shuffle_fraction(self, channel, knob) -> None:
        """The knob's own readout: how much spatial structure survived."""
        got = [knob.achieved(knob.apply(channel, f, np.random.default_rng(0)))["achieved_moran_i"]
               for f in (0.0, 0.25, 0.5, 1.0)]
        assert got[0] > got[1] > got[2]
        assert abs(got[-1]) < 0.3 * got[0]

    def test_full_shuffle_destroys_the_structure_the_nominal_map_has(self, channel, knob) -> None:
        nominal = morans_i(channel.y_gt, channel.ch2xy)
        shuffled = knob.achieved(knob.apply(channel, 1.0, np.random.default_rng(0)))["achieved_moran_i"]
        assert nominal > 0.3 and abs(shuffled) < 0.3

    def test_the_stress_record_names_the_knob_and_the_count(self, channel, knob) -> None:
        got = knob.apply(channel, 0.5, np.random.default_rng(0))
        assert got.stress["knob"] == "k7_shuffle"
        assert got.stress["level"] == 0.5
        assert got.stress["n_shuffled"] == int(round(0.5 * channel.n_sites))
