"""Tests for the causal (online) y scaler used inside the BO loop.

The scaler exists so the y transform can be fitted the way a live experiment would fit it -- from the
observations collected so far -- rather than from every trial of the channel. The invariants below are the
ones the rest of the pipeline relies on: a lossless round trip (predictions are mapped back before any
metric is computed), a safe degenerate case (a flat initial design must not produce inf/NaN targets), and
the affine-invariance that keeps regret comparable across modes.
"""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.data.preprocessing import (
    ONLINE_SCALE_FLOOR,
    ONLINE_Y_MODES,
    OnlineYScaler,
)


@pytest.fixture()
def y() -> np.ndarray:
    """A small, deliberately asymmetric observation vector."""
    return np.array([1.0, 2.0, 5.0, 3.0, 2.5])


class TestModes:
    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            OnlineYScaler("robust")

    @pytest.mark.parametrize("mode", ONLINE_Y_MODES)
    def test_every_declared_mode_constructs(self, mode: str) -> None:
        assert OnlineYScaler(mode).mode == mode

    def test_none_is_inactive_and_an_exact_identity(self, y: np.ndarray) -> None:
        s = OnlineYScaler("none").fit(y)
        assert not s.active
        assert np.array_equal(s.transform(y), y)
        assert np.array_equal(s.inverse_mean(y), y)
        assert np.array_equal(s.inverse_std(y), y)


class TestTransform:
    def test_minmax_maps_observations_onto_the_unit_interval(self, y: np.ndarray) -> None:
        t = OnlineYScaler("minmax").fit(y).transform(y)
        assert t.min() == pytest.approx(0.0)
        assert t.max() == pytest.approx(1.0)

    def test_zscore_standardizes_observations(self, y: np.ndarray) -> None:
        t = OnlineYScaler("zscore").fit(y).transform(y)
        assert t.mean() == pytest.approx(0.0)
        assert t.std() == pytest.approx(1.0)

    @pytest.mark.parametrize("mode", ["minmax", "zscore"])
    def test_round_trip_is_lossless(self, y: np.ndarray, mode: str) -> None:
        """Predictions are inverse-transformed before scoring, so this must be exact."""
        s = OnlineYScaler(mode).fit(y)
        assert np.allclose(s.inverse_mean(s.transform(y)), y)

    @pytest.mark.parametrize("mode", ["minmax", "zscore"])
    def test_std_carries_the_scale_but_not_the_offset(self, y: np.ndarray, mode: str) -> None:
        s = OnlineYScaler(mode).fit(y)
        std = np.array([0.5, 1.0])
        assert np.allclose(s.inverse_std(std), std * s.scale)

    def test_fit_is_causal(self, y: np.ndarray) -> None:
        """Only the observations passed in may influence the transform.

        The prefix deliberately excludes the channel's maximum (``y[2] == 5``), which is the whole point:
        at query 2 a live run cannot know that a larger response exists.
        """
        early = OnlineYScaler("minmax").fit(y[:2])
        late = OnlineYScaler("minmax").fit(y)
        assert (early.offset, early.scale) != (late.offset, late.scale)
        assert early.scale == pytest.approx(float(np.ptp(y[:2])))
        assert late.scale == pytest.approx(float(np.ptp(y)))


class TestDegenerateAndInvalidInput:
    @pytest.mark.parametrize("mode", ["minmax", "zscore"])
    def test_flat_observations_fall_back_to_unit_scale(self, mode: str) -> None:
        """A flat initial design must not divide by ~0 and produce inf/NaN targets."""
        s = OnlineYScaler(mode).fit(np.full(4, 2.0))
        assert s.scale == 1.0
        assert np.isfinite(s.transform(np.full(4, 2.0))).all()

    def test_scale_floor_is_respected(self) -> None:
        s = OnlineYScaler("minmax").fit(np.array([0.0, ONLINE_SCALE_FLOOR / 10.0]))
        assert s.scale == 1.0

    def test_single_observation_is_safe(self) -> None:
        s = OnlineYScaler("minmax").fit(np.array([7.0]))
        assert s.scale == 1.0
        assert np.isfinite(s.transform(np.array([7.0]))).all()

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_observations_fail_fast(self, bad: float) -> None:
        with pytest.raises(RuntimeError, match="non-finite"):
            OnlineYScaler("zscore").fit(np.array([1.0, bad, 3.0]))


class TestAffineInvariance:
    @pytest.mark.parametrize("mode", ["minmax", "zscore"])
    def test_argmax_of_the_transform_is_preserved(self, y: np.ndarray, mode: str) -> None:
        """Acquisition ranks candidates in the scaled space, so the ordering must not move."""
        t = OnlineYScaler(mode).fit(y).transform(y)
        assert np.argmax(t) == np.argmax(y)
        assert np.array_equal(np.argsort(t), np.argsort(y))

    @pytest.mark.parametrize("mode", ["minmax", "zscore"])
    def test_scale_is_positive(self, y: np.ndarray, mode: str) -> None:
        """A negative scale would invert the ranking and silently flip every recommendation."""
        assert OnlineYScaler(mode).fit(y).scale > 0.0
