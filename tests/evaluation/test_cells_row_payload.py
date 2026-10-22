"""Tests for pfns4neurostim.experiments._cells.row_from_payload on rows cached under an older tidy schema."""
from __future__ import annotations

import math
from dataclasses import asdict

import pytest

from pfns4neurostim.evaluation.cache import row_payload
from pfns4neurostim.evaluation.results import TidyRow
from pfns4neurostim.experiments._cells import RETIRED_ROW_FIELDS, row_from_payload


def _row() -> TidyRow:
    """A minimal valid tidy row."""
    return TidyRow(
        run_tag="old-tag", experiment="stress_sweep", dataset="nhp", subject=0, emg=1, model="gp_mll",
        model_version="gpytorch", acq_type="ts_marginal", knob="k5_outliers", level=0.1, gt_mode="full_mean", rep=3,
    )


def test_round_trip_restamps_the_tag() -> None:
    """A current-schema payload rebuilds the same row under the requesting run's tag."""
    row = row_from_payload(row_payload(_row(), {}), "new-tag")
    assert row.run_tag == "new-tag"
    assert {**asdict(row), "run_tag": "old-tag"} == asdict(_row())


@pytest.mark.parametrize("null", [None, math.nan])
def test_retired_column_with_null_is_dropped(null: object) -> None:
    """A row cached before a column was retired still loads when that column is null."""
    payload = row_payload(_row(), {})
    for name in RETIRED_ROW_FIELDS:
        payload["row"][name] = null
    assert asdict(row_from_payload(payload, "old-tag")) == asdict(_row())


def test_retired_column_with_value_raises() -> None:
    """A retired column that holds a measurement is never dropped silently."""
    payload = row_payload(_row(), {})
    payload["row"]["decoy_capture"] = 0.4
    with pytest.raises(ValueError, match="decoy_capture"):
        row_from_payload(payload, "old-tag")


def test_unknown_column_still_raises() -> None:
    """Only the listed retired columns are tolerated; any other unknown key fails as before."""
    payload = row_payload(_row(), {})
    payload["row"]["not_a_column"] = None
    with pytest.raises(TypeError, match="not_a_column"):
        row_from_payload(payload, "old-tag")
