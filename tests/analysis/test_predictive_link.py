"""Predictive link (task #9 Step 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.analysis.predictive_link import channel_mechanism_table, predictive_link


def _frame(slope: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subj in range(6):
        offset = rng.normal(scale=5.0)            # large subject intercepts
        for emg in range(5):
            x = rng.normal()
            rows.append({"dataset": "nhp", "subject": subj, "emg": emg, "x": x,
                         "regret": offset + slope * x + 0.5 * rng.normal()})
    return pd.DataFrame(rows)


def test_within_slope_recovers_truth_despite_intercepts() -> None:
    res = predictive_link(_frame(2.0), "x", "regret", n_boot=300)
    assert res["slope"] == pytest.approx(2.0, abs=0.3)
    assert res["ci_low"] < res["slope"] < res["ci_high"] and res["ci_low"] > 1.0
    assert res["n_clusters"] == 6


def test_null_slope_ci_covers_zero() -> None:
    res = predictive_link(_frame(0.0, seed=3), "x", "regret", n_boot=300)
    assert res["ci_low"] < 0.0 < res["ci_high"]


def test_too_few_channels_raises() -> None:
    with pytest.raises(ValueError, match=">= 3"):
        predictive_link(_frame(1.0).head(2), "x", "regret")


def test_channel_table_adds_adaptivity() -> None:
    cell = pd.DataFrame([
        {"dataset": "nhp", "subject": 1, "emg": 0, "engine": "tabpfn_v2_5", "level": lv,
         "achieved_snr_db": snr, "rho_shape_median": 0.5, "saturation_index_median": 0.1,
         "ell_hat_median": 0.2 + 0.01 * snr, "ell_cv_anchor": 0.3}
        for lv, snr in ((1.0, 0.0), (2.0, -6.0), (4.0, -12.0))
    ])
    out = channel_mechanism_table(cell)
    assert len(out) == 1
    assert out.loc[0, "ell_adaptivity"] == pytest.approx(0.01)
