"""Bounded shrink/move index and log total-variance ratio of OMEGA Part A (user, 2026-10-23)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pfns4neurostim.analysis import bayes_update as B


@pytest.mark.parametrize("r, kappa", [(1.0, 0.0), (0.0, -1.0), (-3.0, -1.0), (3.0, 0.5), (1.0e9, 1.0 - 2.0e-9)])
def test_bounded_index_values(r: float, kappa: float) -> None:
    assert B.bounded_variance_index(r) == pytest.approx(kappa, abs=1e-9)


def test_bounded_index_is_monotone_and_bounded() -> None:
    r = np.linspace(-5.0, 50.0, 1001)
    k = np.array([B.bounded_variance_index(float(v)) for v in r])
    assert np.all(np.diff(k) >= 0.0)
    assert k.min() >= -1.0 and k.max() < 1.0


def test_log_ratio_and_none_and_failures() -> None:
    assert B.log_variance_ratio(1.0) == 0.0
    assert B.log_variance_ratio(2.0) == pytest.approx(-B.log_variance_ratio(0.5))
    assert B.bounded_variance_index(None) is None and B.log_variance_ratio(None) is None
    with pytest.raises(ValueError):
        B.log_variance_ratio(0.0)
    with pytest.raises(ValueError):
        B.bounded_variance_index(math.inf)


def test_pool_medians_transform_each_anchor_first() -> None:
    rows = [{"engine": "e", "pool": "unobserved", "shrink_move_ratio": r, "total_var_ratio": t}
            for r, t in ((0.0, 0.5), (3.0, 2.0))]
    rows.append({"engine": "e", "pool": "observed", "shrink_move_ratio": None, "total_var_ratio": None})
    out = B.pool_variance_readouts(rows)
    # median of the per-anchor indices (-1, 0.5), not the index of the median ratio (1.5 -> 0.2)
    assert out[("e", "unobserved")]["shrink_move_index"] == pytest.approx(-0.25)
    assert out[("e", "unobserved")]["log_total_var_ratio"] == pytest.approx(0.0)
    assert out[("e", "observed")]["shrink_move_index"] is None
    assert out[("e", B.POOLED_ANCHORS)]["shrink_move_index"] == pytest.approx(-0.25)
