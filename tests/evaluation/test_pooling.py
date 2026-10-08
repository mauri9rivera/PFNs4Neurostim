"""Tests for cross-run pooling (evaluation/pooling.py, task plan A8)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from pfns4neurostim.evaluation.pooling import channel_summary, discover_tidy_runs, load_run, pool_runs


def _write_run(path: str, config: dict, tidy: pd.DataFrame) -> str:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    tidy.to_csv(os.path.join(path, "tidy.csv"), index=False)
    return path


def _tidy(model: str = "gp_mll", **extra: object) -> pd.DataFrame:
    rows = [{"subject": s, "emg": e, "rep": r, "model": model, "acq_label": "ts_marginal",
             "recommended_regret": 0.1 * (s + 1) + 0.01 * r, **extra}
            for s in (0, 1) for e in (0,) for r in (0, 1)]
    return pd.DataFrame(rows)


@pytest.fixture()
def output_root(tmp_path) -> str:
    root = str(tmp_path / "output")
    _write_run(os.path.join(root, "benchmark", "nhp", "hyp-a-nhp"),
               {"family": "hyp-a", "tag": "nhp", "online_y_scaler": "minmax"}, _tidy(online_y_scaler="minmax"))
    _write_run(os.path.join(root, "stress", "k5", "nhp", "old-0924"),
               {"family": "stress-k5", "tag": "nhp"}, _tidy(model="tabpfn_v2_5", knob="k5", level=1.0))
    _write_run(os.path.join(root, "benchmark", "nhp", "hyp-a-nhp", "_stale_cohort"),
               {"family": "hyp-a", "tag": "old"}, _tidy())
    _write_run(os.path.join(root, "archive", "x", "run"), {"family": "x", "tag": "x"}, _tidy())
    return root


def test_discover_skips_stale_and_archive(output_root: str) -> None:
    runs = discover_tidy_runs([output_root])
    rel = [os.path.relpath(r, output_root).replace(os.sep, "/") for r in runs]
    assert rel == ["benchmark/nhp/hyp-a-nhp", "stress/k5/nhp/old-0924"]


def test_discover_missing_root_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_tidy_runs([str(tmp_path / "nope")])


def test_pool_stamps_provenance_and_unions_columns(output_root: str) -> None:
    pooled = pool_runs(discover_tidy_runs([output_root]), output_root)
    assert list(pooled.columns[:4]) == ["run_dir", "family", "tag", "online_y_scaler"]
    assert len(pooled) == 8
    old = pooled[pooled["run_dir"] == "stress/k5/nhp/old-0924"]
    assert set(old["online_y_scaler"]) == {"none"}          # pre-2026-09-27 run: offline scaling
    assert pooled.loc[pooled["family"] == "hyp-a", "knob"].isna().all()


def test_load_run_takes_scaler_from_config(tmp_path) -> None:
    tidy = _tidy(online_y_scaler="minmax")
    tidy.loc[0, "online_y_scaler"] = "none"                 # a cached row written before the field existed
    run = _write_run(str(tmp_path / "r"), {"family": "f", "tag": "t", "online_y_scaler": "minmax"}, tidy)
    with pytest.warns(UserWarning, match="1 rows carry a stale"):
        out = load_run(run, str(tmp_path))
    assert set(out["online_y_scaler"]) == {"minmax"}


def test_pool_empty_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        pool_runs([], str(tmp_path))


def test_channel_summary_mean_ci() -> None:
    out = channel_summary(_tidy(), ["model"], "recommended_regret", robust=False).iloc[0]
    per_channel = np.array([0.105, 0.205])                  # reps averaged within each channel first
    assert out["n_channels"] == 2 and out["n_reps"] == 2
    assert out["recommended_regret"] == pytest.approx(per_channel.mean())
    assert out["recommended_regret_ci"] == pytest.approx(1.96 * per_channel.std(ddof=1) / np.sqrt(2))


def test_channel_summary_robust_quantiles() -> None:
    out = channel_summary(_tidy(), ["model"], "recommended_regret", robust=True).iloc[0]
    assert out["recommended_regret"] == pytest.approx(0.155)
    assert out["recommended_regret_q25"] < out["recommended_regret"] < out["recommended_regret_q75"]


def test_channel_summary_fails_on_nan() -> None:
    frame = _tidy()
    frame.loc[0, "recommended_regret"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        channel_summary(frame, ["model"], "recommended_regret", robust=False)
