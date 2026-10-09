"""Tests for the tidy-table TOST helpers (evaluation/stats.py: holm_adjust, paired_unit_values, tost_table)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.evaluation.stats import holm_adjust, paired_unit_values, tost_table


def test_holm_adjust_matches_hand_computation() -> None:
    p = np.array([0.01, 0.04, 0.03])
    # sorted 0.01, 0.03, 0.04 -> x3, x2, x1 = 0.03, 0.06, 0.04 -> running max 0.03, 0.06, 0.06
    np.testing.assert_allclose(holm_adjust(p), [0.03, 0.06, 0.06])
    assert holm_adjust(np.array([0.9, 0.8])).max() == 1.0


def _frame(shift: float, noise: float, n_channels: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_channels):
        base = rng.uniform(0.1, 0.4)
        for rep in range(3):
            rows.append({"dataset": "d", "subject": c // 3, "emg": c % 3, "rep": rep, "model": "b", "m": base})
            rows.append({"dataset": "d", "subject": c // 3, "emg": c % 3, "rep": rep, "model": "a",
                         "m": base + shift + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def test_paired_unit_values_averages_reps_and_counts_unpaired() -> None:
    frame = _frame(0.0, 0.0)
    frame = frame[~((frame["model"] == "a") & (frame["subject"] == 0) & (frame["emg"] == 0))]
    a, b, dropped = paired_unit_values(frame, "a", "b", "m")
    assert a.shape == b.shape == (11,) and dropped == 1
    np.testing.assert_allclose(a, b)


def test_paired_unit_values_rejects_nan() -> None:
    frame = _frame(0.0, 0.0)
    frame.loc[0, "m"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        paired_unit_values(frame, "a", "b", "m")


def test_tost_table_equivalent_vs_shifted() -> None:
    close = tost_table(_frame(0.005, 0.005), [("a", "b")], ["m"], margin=0.05, by=["dataset"]).iloc[0]
    far = tost_table(_frame(0.2, 0.005), [("a", "b")], ["m"], margin=0.05, by=["dataset"]).iloc[0]
    assert close["n"] == 12 and close["equivalent_holm"]
    assert not far["equivalent_holm"] and far["ci_lo"] > 0.05


def test_tost_table_subject_level_unit() -> None:
    out = tost_table(_frame(0.0, 0.001), [("a", "b")], ["m"], margin=0.05, by=["dataset"], unit=("subject",)).iloc[0]
    assert out["n"] == 4 and out["unit"] == "subject"
