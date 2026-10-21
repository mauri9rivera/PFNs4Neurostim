"""Predictive link (M8)."""
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


def test_channel_table_takes_per_channel_medians_of_the_requested_metrics() -> None:
    cells = pd.DataFrame([
        {"dataset": "nhp", "subject": 1, "emg": 0, "level": lv, "drift_abs": d, "delta_crps": -0.01, "other": 9.0}
        for lv, d in ((1.0, 0.1), (1.0, 0.3), (1.0, 0.2), (2.0, 5.0))
    ])
    out = channel_mechanism_table(cells, ["drift_abs", "delta_crps", "missing"], level=1.0)
    assert len(out) == 1 and list(out.columns) == ["dataset", "subject", "emg", "drift_abs", "delta_crps"]
    assert out.loc[0, "drift_abs"] == pytest.approx(0.2)
    assert channel_mechanism_table(cells, ["drift_abs"]).loc[0, "drift_abs"] == pytest.approx(0.25)
