"""Tests for pfns4neurostim.evaluation.results (P0.2 acquisition logging +
tidy result schema).

Covers:
- TidyRow schema validation (missing key column raises, non-finite metric
  raises, None -> NaN allowed).
- Acquisition block flattening (P0.2 schema).
- config.yaml round-trip (model_version + acquisition block present).
- results.csv / trajectories.pkl round-trip.
- discover_run_dirs archive-skipping behaviour.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from pfns4neurostim.evaluation.results import (
    ALL_COLUMNS,
    KEY_COLUMNS,
    METRIC_COLUMNS,
    TidyRow,
    create_run_dir,
    discover_run_dirs,
    flatten_acquisition_block,
    make_trajectory_key,
    read_config,
    read_results_csv,
    read_trajectories,
    rows_to_dataframe,
    write_config,
    write_results_csv,
    write_trajectories,
)


def _make_row(**overrides: object) -> TidyRow:
    """Build a minimal valid TidyRow, with any field overridden."""
    defaults = dict(
        run_tag="nhp-bo_benchmark-abcde",
        experiment="bo_benchmark",
        dataset="nhp",
        subject=1,
        emg=3,
        model="tabpfn_v2_5",
        model_version="TabPFN v2.5",
        acq_type="ucb",
        knob=None,
        level=None,
        gt_mode="full_mean",
        rep=0,
    )
    defaults.update(overrides)
    return TidyRow(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TidyRow schema validation
# ---------------------------------------------------------------------------


class TestTidyRowSchema:
    def test_minimal_row_constructs(self) -> None:
        row = _make_row()
        assert row.run_tag == "nhp-bo_benchmark-abcde"
        assert row.acq_type == "ucb"

    def test_missing_key_column_raises(self) -> None:
        # 'dataset' (a key column) omitted entirely -> dataclass TypeError.
        with pytest.raises(TypeError):
            TidyRow(
                run_tag="t",
                experiment="e",
                # dataset missing
                subject=0,
                emg=0,
                model="m",
                model_version="v",
                acq_type="ucb",
                knob=None,
                level=None,
                gt_mode="full_mean",
                rep=0,
            )  # type: ignore[call-arg]

    def test_none_metric_allowed_and_becomes_nan(self) -> None:
        row = _make_row(r2=None, ece=None)
        d = row.to_dict()
        assert math.isnan(d["r2"])
        assert math.isnan(d["ece"])

    def test_explicit_nan_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _make_row(r2=float("nan"))

    def test_explicit_inf_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            _make_row(cumulative_regret=float("inf"))

    def test_finite_metric_preserved(self) -> None:
        row = _make_row(r2=0.87, final_regret=0.05)
        d = row.to_dict()
        assert d["r2"] == pytest.approx(0.87)
        assert d["final_regret"] == pytest.approx(0.05)

    def test_co_primary_regret_columns_present(self) -> None:
        assert "final_regret" in METRIC_COLUMNS
        assert "cumulative_regret" in METRIC_COLUMNS
        assert "recommended_regret" in METRIC_COLUMNS
        assert "best_queried_regret" in METRIC_COLUMNS

    def test_all_columns_is_key_then_metric(self) -> None:
        assert ALL_COLUMNS == KEY_COLUMNS + METRIC_COLUMNS

    def test_to_dict_has_all_columns(self) -> None:
        row = _make_row()
        d = row.to_dict()
        assert set(d.keys()) == set(ALL_COLUMNS)


# ---------------------------------------------------------------------------
# Acquisition block flattening
# ---------------------------------------------------------------------------


class TestFlattenAcquisitionBlock:
    def test_ucb_with_cosine_kappa_schedule(self) -> None:
        block = {
            "type": "ucb",
            "params": {"kappa": 2.0},
            "schedules": {"kappa": {"kind": "cosine", "start": 7.5, "end": 0.6}},
        }
        flat = flatten_acquisition_block(block)
        assert flat["acq_type"] == "ucb"
        assert flat["acq_param_kappa"] == 2.0
        assert flat["acq_schedule_kappa"] == "cosine(end=0.6,start=7.5)"

    def test_ts_with_no_schedule(self) -> None:
        block = {"type": "ts", "params": {"temperature": 1.0, "n_candidates": None}}
        flat = flatten_acquisition_block(block)
        assert flat == {
            "acq_type": "ts",
            "acq_param_n_candidates": None,
            "acq_param_temperature": 1.0,
        }

    def test_unknown_top_level_key_raises(self) -> None:
        block = {"type": "ucb", "params": {"kappa": 2.0}, "bogus": 1}
        with pytest.raises(ValueError, match="Unknown acquisition config key"):
            flatten_acquisition_block(block)

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValueError, match="'type'"):
            flatten_acquisition_block({"params": {}})

    def test_missing_params_raises(self) -> None:
        with pytest.raises(ValueError, match="'params'"):
            flatten_acquisition_block({"type": "ucb"})

    def test_schedule_for_undeclared_param_raises(self) -> None:
        block = {
            "type": "ucb",
            "params": {"kappa": 2.0},
            "schedules": {"not_a_param": {"kind": "linear", "start": 0, "end": 1}},
        }
        with pytest.raises(ValueError, match="not_a_param"):
            flatten_acquisition_block(block)

    def test_deterministic_column_order_independent_of_dict_order(self) -> None:
        block_a = {
            "type": "ucb",
            "params": {"kappa": 2.0, "alpha": 1.0},
        }
        block_b = {
            "params": {"alpha": 1.0, "kappa": 2.0},
            "type": "ucb",
        }
        assert list(flatten_acquisition_block(block_a)) == list(
            flatten_acquisition_block(block_b)
        )

    def test_greedy_empty_params(self) -> None:
        flat = flatten_acquisition_block({"type": "greedy", "params": {}})
        assert flat == {"acq_type": "greedy"}


# ---------------------------------------------------------------------------
# rows_to_dataframe
# ---------------------------------------------------------------------------


class TestRowsToDataframe:
    def test_broadcasts_acquisition_columns(self) -> None:
        rows = [_make_row(rep=0), _make_row(rep=1)]
        acquisition = {"type": "ucb", "params": {"kappa": 2.0}}
        df = rows_to_dataframe(rows, acquisition=acquisition)
        assert (df["acq_param_kappa"] == 2.0).all()
        assert len(df) == 2

    def test_acq_type_mismatch_raises(self) -> None:
        rows = [_make_row(acq_type="ucb")]
        acquisition = {"type": "ts", "params": {"temperature": 1.0}}
        with pytest.raises(ValueError, match="does not match"):
            rows_to_dataframe(rows, acquisition=acquisition)

    def test_no_acquisition_block_still_works(self) -> None:
        rows = [_make_row()]
        df = rows_to_dataframe(rows)
        assert list(df.columns) == list(ALL_COLUMNS)


# ---------------------------------------------------------------------------
# Run-directory I/O
# ---------------------------------------------------------------------------


class TestRunDirIO:
    def test_create_run_dir(self, tmp_path: object) -> None:
        run_dir = create_run_dir(str(tmp_path), "nhp-bo_benchmark-abcde")
        assert os.path.isdir(run_dir)
        assert run_dir.endswith("nhp-bo_benchmark-abcde")

    def test_config_round_trip_contains_model_version_and_acquisition(
        self, tmp_path: object
    ) -> None:
        run_dir = create_run_dir(str(tmp_path), "tag")
        acquisition = {
            "type": "ucb",
            "params": {"kappa": 2.0},
            "schedules": {"kappa": {"kind": "cosine", "start": 7.5, "end": 0.6}},
        }
        resolved_config = {
            "model_version": "TabPFN v2.5",
            "acquisition": acquisition,
            "dataset": "nhp",
        }
        path = write_config(run_dir, resolved_config)
        assert os.path.exists(path)
        loaded = read_config(run_dir)
        assert loaded == resolved_config
        assert loaded["model_version"] == "TabPFN v2.5"
        assert loaded["acquisition"] == acquisition

    def test_write_config_missing_model_version_raises(self, tmp_path: object) -> None:
        run_dir = create_run_dir(str(tmp_path), "tag")
        with pytest.raises(ValueError, match="model_version"):
            write_config(run_dir, {"acquisition": {"type": "ucb", "params": {}}})

    def test_write_config_missing_acquisition_raises(self, tmp_path: object) -> None:
        run_dir = create_run_dir(str(tmp_path), "tag")
        with pytest.raises(ValueError, match="acquisition"):
            write_config(run_dir, {"model_version": "TabPFN v2.5"})

    def test_csv_round_trip_lossless_for_full_rows(self, tmp_path: object) -> None:
        # All ints present (no missing budget/n_init/seed) so dtype is
        # preserved exactly, per the documented dtype-widening caveat.
        run_dir = create_run_dir(str(tmp_path), "tag")
        rows = [
            _make_row(rep=0, r2=0.5, budget=100, n_init=10, seed=42),
            _make_row(rep=1, r2=0.6, budget=100, n_init=10, seed=43),
        ]
        df = rows_to_dataframe(rows)
        path = write_results_csv(run_dir, df)
        assert path == os.path.join(run_dir, "results", "results.csv")
        loaded = read_results_csv(path)
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)
        # Value-level exactness for the int columns and float metrics.
        assert list(loaded["budget"]) == [100, 100]
        assert list(loaded["seed"]) == [42, 43]
        assert list(loaded["rep"]) == [0, 1]
        assert loaded["r2"].tolist() == pytest.approx([0.5, 0.6])
        assert list(loaded["run_tag"]) == list(df["run_tag"])

    def test_csv_round_trip_missing_metric_is_nan(self, tmp_path: object) -> None:
        run_dir = create_run_dir(str(tmp_path), "tag")
        rows = [_make_row(rep=0, r2=None)]
        df = rows_to_dataframe(rows)
        path = write_results_csv(run_dir, df)
        loaded = read_results_csv(path)
        assert np.isnan(loaded["r2"].iloc[0])

    def test_trajectories_round_trip(self, tmp_path: object) -> None:
        run_dir = create_run_dir(str(tmp_path), "tag")
        row = _make_row(rep=0)
        key = make_trajectory_key(row)
        assert key == tuple(getattr(row, c) for c in KEY_COLUMNS)
        trajectories = {
            key: {
                "regret": [1.0, 0.5, 0.1],
                "times": [0.01, 0.02, 0.015],
                "recommendations": [3, 3, 7],
            }
        }
        path = write_trajectories(run_dir, trajectories)
        loaded = read_trajectories(path)
        assert loaded == trajectories

    def test_trajectories_key_columns_mismatch_raises(self, tmp_path: object) -> None:
        import pickle

        run_dir = create_run_dir(str(tmp_path), "tag")
        path = os.path.join(run_dir, "trajectories.pkl")
        with open(path, "wb") as f:
            pickle.dump({"key_columns": ("stale", "columns"), "trajectories": {}}, f)
        with pytest.raises(ValueError, match="no longer matches"):
            read_trajectories(path)


# ---------------------------------------------------------------------------
# discover_run_dirs
# ---------------------------------------------------------------------------


class TestDiscoverRunDirs:
    def test_skips_archive_by_default(self, tmp_path: object) -> None:
        runs_root = os.path.join(str(tmp_path), "runs")
        os.makedirs(os.path.join(runs_root, "nhp-bo_benchmark-aaaaa"))
        archive_root = os.path.join(str(tmp_path), "archive", "2026-09-16", "runs")
        os.makedirs(os.path.join(archive_root, "nhp-bo_benchmark-zzzzz"))

        found = discover_run_dirs(runs_root)
        names = [os.path.basename(p) for p in found]
        assert names == ["nhp-bo_benchmark-aaaaa"]

    def test_include_archive_true_finds_archived_when_pointed_at_it(
        self, tmp_path: object
    ) -> None:
        archive_root = os.path.join(str(tmp_path), "archive", "2026-09-16", "runs")
        os.makedirs(os.path.join(archive_root, "nhp-bo_benchmark-zzzzz"))

        # Pointed directly at the archive subtree: skipped unless opted in.
        assert discover_run_dirs(archive_root) == []
        found = discover_run_dirs(archive_root, include_archive=True)
        assert len(found) == 1
        assert os.path.basename(found[0]) == "nhp-bo_benchmark-zzzzz"

    def test_nonexistent_runs_root_returns_empty(self, tmp_path: object) -> None:
        assert discover_run_dirs(os.path.join(str(tmp_path), "does_not_exist")) == []

    def test_family_filter(self, tmp_path: object) -> None:
        runs_root = os.path.join(str(tmp_path), "runs")
        os.makedirs(os.path.join(runs_root, "nhp-vanilla-benchmark-11111"))
        os.makedirs(os.path.join(runs_root, "nhp-stress-sweep-22222"))

        found = discover_run_dirs(runs_root, family="vanilla-benchmark")
        names = [os.path.basename(p) for p in found]
        assert names == ["nhp-vanilla-benchmark-11111"]
