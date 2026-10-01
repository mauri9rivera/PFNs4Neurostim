"""Nothing a runner computes may be lost on the way to the CSV.

Four metrics have now been produced by the BO runner and then silently dropped before reaching
``tidy.csv``: ``online_y_scaler`` and the ``gp_*`` fit diagnostics (two separate causes, both fixed
2026-09-30), and ``queries_to_target_90`` / ``_95``, which were declared in the schema on 2026-09-18,
produced on 2026-09-30 and still absent from every row of two finished runs because
:func:`~pfns4neurostim.experiments._rows.build_row` is an explicit field-by-field mapping that never
listed them.

Each of those bugs was invisible: the column existed, the figures that recomputed the quantity post-hoc
looked right, and only an empty column in a finished run gave it away. These tests close the loop
structurally -- one on the mapping, one on a real run -- so the next producer that is added without a
mapping fails here instead of after an experiment.
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import numpy as np
import pytest

from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.evaluation.bo_runner import run_channel_bo
from pfns4neurostim.evaluation.results import METRIC_COLUMNS, PROVENANCE_COLUMNS, TidyRow
from pfns4neurostim.experiments import _rows

#: Schema columns that no ``build_row`` call site can fill, with the reason. A column may be listed here
#: only because it is *structurally* unavailable, never because it happens to be unset today.
EXEMPT: dict[str, str] = {
    "decoy_capture": "K1 only: set from channel.meta when the decoy knob built the channel",
    "achieved_snr_db": "stress sweeps only: comes from the knob's achieved() mapping",
    "split_id": "split-half ground truth only",
    "gt_r_half": "split-half ground truth only",
    "gt_reliability": "split-half ground truth only",
    "ece": "needs a predictive distribution; None for models without one",
    "nll": "needs a predictive distribution; None for models without one",
    "crps": "needs a predictive distribution; None for models without one",
    "coverage_50": "needs a predictive distribution; None for models without one",
    "coverage_90": "needs a predictive distribution; None for models without one",
    "spearman": "undefined on a constant prediction vector",
    "queries_to_target_90": "right-censored: NaN when a run never reaches the target",
    "queries_to_target_95": "right-censored: NaN when a run never reaches the target",
}


def test_build_row_maps_every_schema_field() -> None:
    """Every ``TidyRow`` field is named in ``build_row``, so adding a column forces a mapping."""
    source = inspect.getsource(_rows.build_row)
    passed = set(re.findall(r"^\s{8}(\w+)=", source, re.M))
    missing = {f.name for f in dataclasses.fields(TidyRow)} - passed
    assert not missing, (
        f"build_row never sets {sorted(missing)}: the column will be empty in every row. Map it, or -- if "
        "it genuinely cannot be filled there -- say so in the docstring of the producer."
    )


@pytest.fixture
def channel() -> ChannelData:
    """A tiny deterministic 2D channel: a 6x6 grid with one smooth hotspot and 3 trials per site."""
    ax = np.linspace(0.0, 1.0, 6)
    ch2xy = np.stack(np.meshgrid(ax, ax), -1).reshape(-1, 2)                       # [36, 2]
    y_gt = np.exp(-8.0 * ((ch2xy - np.array([0.3, 0.7])) ** 2).sum(1))             # [36]
    y_gt = (y_gt - y_gt.mean()) / y_gt.std()                                       # [36] z-scored
    rng = np.random.default_rng(0)
    trials = y_gt[:, None] + 0.1 * rng.normal(size=(len(y_gt), 3))                 # [36, 3]
    return ChannelData(
        dataset="synthetic", subject=0, emg=0, X_pool=ch2xy, Y_trials=trials, y_gt=y_gt,
        ch2xy=ch2xy, grid_shape=(6, 6),
    )


def test_a_real_run_fills_every_non_exempt_metric(channel: ChannelData) -> None:
    """The functional half: run the loop and check the row, not the mapping."""
    result = run_channel_bo(
        "gp_naive", channel, acq_fn="ts_marginal", acq_params={"temperature": 1.0},
        budget=14, n_init=4, seed=0, device="cpu",
    )
    row = _rows.build_row(
        result, channel, run_tag="t", experiment="bo_benchmark", model="gp_naive",
        acq_type="ts_marginal", rep=0,
    ).to_dict()
    empty = [c for c in METRIC_COLUMNS if c not in EXEMPT and row.get(c) is None]
    assert not empty, f"the run computed nothing for {empty}; a producer or a mapping is missing"
    for column in PROVENANCE_COLUMNS:
        assert row.get(column) is not None, f"provenance column {column} is empty"


def test_the_runner_produces_every_key_the_mapping_reads(channel: ChannelData) -> None:
    """The other direction: a key the runner computes but ``build_row`` ignores is a dropped metric."""
    result = run_channel_bo(
        "gp_naive", channel, acq_fn="ts_marginal", acq_params={"temperature": 1.0},
        budget=14, n_init=4, seed=0, device="cpu",
    )
    source = inspect.getsource(_rows.build_row)
    # ``row.get("k")``, ``row.get("k", default)`` and ``row["k"]`` all count as read.
    read = set(re.findall(r'row\.get\("(\w+)"', source)) | set(re.findall(r'row\["(\w+)"\]', source))
    # Keys the row dict carries for other consumers: the model and acquisition identity come to build_row
    # as arguments, and the pool shape is used by the cache payload, not by the schema.
    structural = {"model", "model_version", "acq_type", "n_sites", "n_dims"}
    dropped = set(result.row) - read - structural - set(result.diagnostics)
    assert not dropped, (
        f"the runner computes {sorted(dropped)} and nothing reads them: either map them in build_row or "
        "move them to BOResult.diagnostics, which rides the non-schema extras channel into the CSV."
    )


def test_queries_to_target_reaches_the_row(channel: ChannelData) -> None:
    """The specific regression: A8's two columns, on a channel whose optimum is reachable in 14 queries."""
    result = run_channel_bo(
        "gp_naive", channel, acq_fn="ts_marginal", acq_params={"temperature": 1.0},
        budget=14, n_init=4, seed=0, device="cpu",
    )
    row = _rows.build_row(
        result, channel, run_tag="t", experiment="bo_benchmark", model="gp_naive",
        acq_type="ts_marginal", rep=0,
    ).to_dict()
    for column in ("queries_to_target_90", "queries_to_target_95"):
        assert column in row
        assert row[column] == result.row[column], f"{column} did not survive build_row"
