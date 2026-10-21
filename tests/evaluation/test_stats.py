"""Statistics helpers of :mod:`pfns4neurostim.evaluation.stats` that have no other test module."""
from __future__ import annotations

import numpy as np
import pytest

from pfns4neurostim.evaluation.stats import icc_oneway


def test_icc_oneway() -> None:
    rng = np.random.default_rng(0)
    target = rng.normal(size=(20, 1))
    assert icc_oneway(np.repeat(target, 5, axis=1)) == pytest.approx(1.0)
    assert icc_oneway(target + 0.01 * rng.normal(size=(20, 5))) > 0.99
    assert abs(icc_oneway(rng.normal(size=(200, 5)))) < 0.1
