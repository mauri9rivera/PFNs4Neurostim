"""Kill a sweep mid-grid, re-run, and get every cell (cache resume)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pfns4neurostim.config import load_experiment_config
from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.evaluation.cache import CellStore
from pfns4neurostim.experiments import stress_sweep

OVERRIDES = [
    "models=[gp_mll,gp_naive]", "n_reps=1", "budget=8", "n_init=3", "device=cpu",
    "knob.levels=[1.0,2.0]",
]


@pytest.fixture()
def channel() -> ChannelData:
    rng = np.random.default_rng(0)
    coords = np.stack(np.meshgrid(np.arange(5), np.arange(4)), axis=-1).reshape(-1, 2)
    x = coords / np.array([4.0, 3.0])
    y_gt = np.exp(-((x[:, 0] - 0.7) ** 2 + (x[:, 1] - 0.3) ** 2) / 0.1)
    y_gt = (y_gt - y_gt.mean()) / y_gt.std()
    Y = y_gt[:, None] + 0.25 * rng.normal(size=(20, 6))
    return ChannelData("nhp", 1, 0, x, Y, y_gt, coords, (5, 4))


@pytest.fixture()
def cfg(tmp_path: Any, monkeypatch: pytest.MonkeyPatch, channel: ChannelData) -> Any:
    monkeypatch.setattr(stress_sweep, "_channels", lambda _cfg, _shard=None: [channel])
    return load_experiment_config(
        "configs/experiment/stress_k2_nhp.yaml", OVERRIDES + [f"output_root={tmp_path}"]
    )


def _rows(cfg: Any, store: CellStore) -> list[Any]:
    return stress_sweep.build_tidy_rows(cfg, "tag", progress=False, store=store)[0]


def test_rerun_is_all_hits_and_identical(cfg: Any) -> None:
    first_store = CellStore(cfg.cell_cache_root)
    first = _rows(cfg, first_store)
    second_store = CellStore(cfg.cell_cache_root)
    second = _rows(cfg, second_store)
    assert (first_store.computed, second_store.computed, second_store.hits) == (4, 0, 4)
    assert [r.recommended_regret for r in first] == [r.recommended_regret for r in second]


def test_killed_run_resumes(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    real = stress_sweep._compute_cell
    calls = {"n": 0}

    def dies_on_third(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt  # not an Exception: simulates a kill, must propagate
        return real(*a, **k)

    monkeypatch.setattr(stress_sweep, "_compute_cell", dies_on_third)
    with pytest.raises(KeyboardInterrupt):
        _rows(cfg, CellStore(cfg.cell_cache_root))
    monkeypatch.setattr(stress_sweep, "_compute_cell", real)
    store = CellStore(cfg.cell_cache_root)
    assert len(_rows(cfg, store)) == 4
    assert (store.hits, store.computed) == (2, 2)


def test_failed_cell_does_not_kill_the_grid(cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    real = stress_sweep._compute_cell

    def flaky(*a: Any, **k: Any) -> Any:
        if a[7] == "gp_naive":  # model argument
            raise RuntimeError("model exploded")
        return real(*a, **k)

    monkeypatch.setattr(stress_sweep, "_compute_cell", flaky)
    store = CellStore(cfg.cell_cache_root)
    rows = _rows(cfg, store)
    assert {r.model for r in rows} == {"gp_mll"} and len(store.failures) == 2
    with pytest.raises(RuntimeError, match="model exploded"):
        store.raise_if_failed()


def test_bo_benchmark_runner_resumes_and_uses_family_tag_dir(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, channel: ChannelData, capsys: Any
) -> None:
    from pfns4neurostim.experiments import bo_benchmark

    monkeypatch.setattr(bo_benchmark, "iter_channels", lambda *a, **k: [channel])
    args = [
        "models=[gp_mll]", "n_reps=2", "budget=8", "n_init=3", "device=cpu",
        f"output_root={tmp_path}",
    ]
    out = bo_benchmark.run_bo_benchmark("configs/experiment/hyp_a_nhp.yaml", args)
    assert out.endswith("hyp-a-nhp") or out.replace("\\", "/").endswith("benchmark/nhp/hyp-a-nhp")
    capsys.readouterr()
    bo_benchmark.run_bo_benchmark("configs/experiment/hyp_a_nhp.yaml", args)
    text = capsys.readouterr().out
    assert "cells: 2 cached, 0 computed, 0 failed" in text


def test_per_model_device_override_wins_and_is_logged(channel: ChannelData) -> None:
    """model_params.<model>.device overrides the experiment device (P0.12)."""
    from pfns4neurostim.evaluation.bo_runner import run_channel_bo
    from pfns4neurostim.experiments._rows import build_row

    res = run_channel_bo(
        "gp_naive", channel, acq_fn="ei", budget=6, n_init=3, seed=0,
        device="cuda", model_params={"device": "cpu"},
    )
    assert res.row["device"] == "cpu"
    row = build_row(res, channel, run_tag="t", experiment="bo_benchmark", model="gp_naive",
                    acq_type="ei", rep=0, acq_label="ei")
    assert row.device == "cpu" and row.acq_label == "ei"


def test_count_channels_uses_explicit_emgs(tmp_path: Any) -> None:
    from pfns4neurostim.experiments._cells import count_channels

    cfg2 = load_experiment_config(
        "configs/experiment/stress_k2_nhp.yaml",
        ["dataset.subjects=[0,3]", "dataset.emgs=[0,1,2]", f"output_root={tmp_path}"],
    )
    assert count_channels(cfg2) == 6


def test_random_acquisition_is_served_only_by_the_random_baseline() -> None:
    from pfns4neurostim.experiments.bo_benchmark import _supported

    assert _supported("random", "random")
    assert not _supported("random", "ts_marginal")
    assert not _supported("gp_mll", "random")
    assert not _supported("tabpfn_v2_5", "random")
    assert _supported("gp_mll", "ts_marginal") and _supported("tabpfn_v2_5", "ts_marginal")
    assert _supported("gp_mll", "ts_joint") and not _supported("tabpfn_v2_5", "ts_joint")


def test_shard_runs_get_distinct_run_dirs_and_share_the_cache(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, channel: ChannelData
) -> None:
    """Two shards of one config write to different run dirs; a final --only-cached merges them."""
    from pfns4neurostim.experiments import bo_benchmark

    calls: list[Any] = []

    def fake_iter(*a: Any, **k: Any) -> list[ChannelData]:
        calls.append(k.get("shard"))
        return [channel] if k.get("shard") in (None, (0, 2)) else []

    monkeypatch.setattr(bo_benchmark, "iter_channels", fake_iter)
    args = ["models=[gp_naive]", "n_reps=1", "budget=6", "n_init=3", "device=cpu", f"output_root={tmp_path}"]
    d0 = bo_benchmark.run_bo_benchmark("configs/experiment/hyp_a_nhp.yaml", args, shard=(0, 2))
    assert d0.replace("\\", "/").endswith("hyp-a-nhp-shard0of2")
    merged = bo_benchmark.run_bo_benchmark(
        "configs/experiment/hyp_a_nhp.yaml", args, only_cached=True
    )
    assert merged.replace("\\", "/").endswith("hyp-a-nhp")
    assert (0, 2) in calls and None in calls


def test_empty_shard_is_a_clean_noop(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, channel: ChannelData, capsys: Any
) -> None:
    """More lanes than channels leaves some shards empty: they must succeed, not raise."""
    from pfns4neurostim.experiments import bo_benchmark

    monkeypatch.setattr(bo_benchmark, "iter_channels", lambda *a, **k: [])
    args = ["models=[gp_naive]", "n_reps=1", "budget=6", "n_init=3", "device=cpu", f"output_root={tmp_path}"]
    bo_benchmark.run_bo_benchmark("configs/experiment/hyp_a_nhp.yaml", args, shard=(5, 8))
    assert "owns no channels" in capsys.readouterr().out
    # An unsharded run that finds nothing is still an error.
    with pytest.raises(RuntimeError, match="produced no rows"):
        bo_benchmark.run_bo_benchmark("configs/experiment/hyp_a_nhp.yaml", args)


def test_empty_shard_noop_in_stress_sweep(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, cfg: Any, capsys: Any
) -> None:
    monkeypatch.setattr(stress_sweep, "_channels", lambda _cfg, _shard=None: [])
    stress_sweep.run_stress_sweep(
        "configs/experiment/stress_k2_nhp.yaml",
        OVERRIDES + [f"output_root={tmp_path}"], shard=(5, 8),
    )
    assert "owns no channels" in capsys.readouterr().out
