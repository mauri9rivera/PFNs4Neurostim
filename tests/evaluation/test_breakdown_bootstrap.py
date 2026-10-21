"""Channel bootstrap of the pooled breakdown point (2026-10-21).

The CI must be a CI of the plotted pooled point: every resample is the pooled rule of
:func:`breakdown_point` re-run on channels drawn with replacement, and the interval always contains the point.
"""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.evaluation.robustness import bootstrap_breakdown, breakdown_point

LEVELS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)
MARGIN: float = 0.05


def _channels(n_ch: int, n_rep: int, slope: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Per-channel level -> values for a model that falls behind the reference by ``slope * level``."""
    rng = np.random.default_rng(seed)
    model, ref = [], []
    for _ in range(n_ch):
        offset = rng.normal(0.0, 0.05)
        ref.append({lv: 0.2 + 0.05 * rng.normal(size=n_rep) for lv in LEVELS})
        model.append({lv: ref[-1][lv] + slope * lv + offset + 0.05 * rng.normal(size=n_rep) for lv in LEVELS})
    return model, ref


def _pooled(model: list[dict], ref: list[dict], idx: np.ndarray) -> float:
    """breakdown_point on the concatenation of the channels ``idx`` (the reference implementation)."""
    cat = lambda chans: {lv: np.concatenate([chans[i][lv] for i in idx]) for lv in LEVELS}  # noqa: E731
    return breakdown_point(cat(model), cat(ref), margin=MARGIN, test="noninferiority")["breakdown_level"]


def _level(boot: dict, rank: int) -> float:
    return boot["levels"][rank] if rank < len(boot["levels"]) else float("nan")


class TestBootstrapBreakdown:
    def test_point_equals_breakdown_point(self) -> None:
        model, ref = _channels(12, 6, slope=0.5)
        boot = bootstrap_breakdown(model, ref, margin=MARGIN)
        np.testing.assert_equal(_level(boot, boot["point_rank"]), _pooled(model, ref, np.arange(12)))

    def test_each_resample_is_the_pooled_rule_on_resampled_channels(self) -> None:
        model, ref = _channels(10, 5, slope=0.4, seed=3)
        n_boot = 25
        boot = bootstrap_breakdown(model, ref, margin=MARGIN, n_boot=n_boot, rng=np.random.default_rng(7))
        draws = np.random.default_rng(7).integers(0, 10, size=(n_boot, 10))   # the same draws, replayed
        for b in range(n_boot):
            np.testing.assert_equal(_level(boot, int(boot["ranks"][b])), _pooled(model, ref, draws[b]))

    @pytest.mark.parametrize("slope, seed", [(0.0, 0), (0.3, 1), (0.6, 2), (1.5, 3), (0.45, 4)])
    def test_ci_brackets_the_point(self, slope: float, seed: int) -> None:
        model, ref = _channels(8, 4, slope=slope, seed=seed)
        boot = bootstrap_breakdown(model, ref, margin=MARGIN)
        mild, harsh = boot["ci_rank"]
        assert mild <= boot["point_rank"] <= harsh
        assert 0.0 <= boot["none_fraction"] <= 1.0

    def test_a_model_that_never_breaks_down_reports_no_breakdown(self) -> None:
        model, ref = _channels(10, 6, slope=-0.5)
        boot = bootstrap_breakdown(model, ref, margin=MARGIN)
        assert boot["point_rank"] == len(LEVELS) and boot["ci_rank"][1] == len(LEVELS)
        assert boot["none_fraction"] > 0.5

    def test_severity_descending_orders_the_ladder(self) -> None:
        model, ref = _channels(6, 4, slope=0.5)
        boot = bootstrap_breakdown(model, ref, margin=MARGIN, severity_ascending=False)
        assert boot["levels"] == sorted(LEVELS, reverse=True)

    def test_mismatched_inputs_raise(self) -> None:
        model, ref = _channels(4, 3, slope=0.1)
        with pytest.raises(ValueError, match="channel lists"):
            bootstrap_breakdown(model, ref[:3], margin=MARGIN)
        ref[0] = {lv: v for lv, v in ref[0].items() if lv != 0.4} | {9.0: np.zeros(3)}
        with pytest.raises(ValueError, match="level sets differ"):
            bootstrap_breakdown(model, ref, margin=MARGIN)
