"""Shadow and integration tests for task #10 Steps 9-10.

* The S5 regime surface reads a budget-B run off longer runs' per-step traces; the shadow
  test checks that claim against real shorter-budget runs (same seed).
* Demo 1 sweeps run on fitted synthetic twins; split-half ground truth runs repetition i
  on instance i mod 2R; both through the real runner with the GP models.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.config import load_experiment_config
from pfns4neurostim.data import synthetic_neurostim as sn
from pfns4neurostim.data.channels import ChannelData
from pfns4neurostim.evaluation.bo_runner import run_channel_bo
from pfns4neurostim.evaluation.cache import CellStore
from pfns4neurostim.experiments import stress_sweep
from pfns4neurostim.visualization import stress as stress_figs

OVERRIDES = ["models=[gp_mll,gp_naive]", "n_reps=4", "budget=10", "n_init=3", "device=cpu"]


def _synthetic_real_channel(subject: int = 1, emg: int = 0) -> ChannelData:
    """A 'recorded' channel: a generated map relabelled as NHP, with its scaler kept."""
    coords = np.stack(np.meshgrid(np.arange(6), np.arange(5), indexing="ij"), axis=-1).reshape(-1, 2)
    params = sn.GeneratorParams(
        ch2xy=coords, grid_shape=(6, 5),
        hotspots=(sn.Hotspot(np.array([4.0, 2.0]), np.array([1.2, 1.0]), 2.0),),
        baseline=0.5, saturation=3.0, noise_cv=0.4, n_trials=6,
    )
    ch = sn.generate_neurostim_map(params, np.random.default_rng(subject * 10 + emg), dataset="nhp",
                                   subject=subject, emg=emg)
    meta = {k: v for k, v in ch.meta.items() if k != "generator"}   # an in-vivo channel has no generator
    return ChannelData("nhp", subject, emg, ch.X_pool, ch.Y_trials, ch.y_gt, ch.ch2xy, ch.grid_shape, meta=meta)


class TestBudgetFromTraces:
    """S5 costs no compute only if a truncated trace equals a shorter run."""

    @pytest.mark.parametrize("short", [5, 8])
    def test_per_step_regret_equals_a_real_shorter_run(self, short: int) -> None:
        ch = _synthetic_real_channel()
        long = run_channel_bo("gp_naive", ch, acq_fn="ei", budget=14, n_init=3, seed=7, device="cpu")
        brief = run_channel_bo("gp_naive", ch, acq_fn="ei", budget=short, n_init=3, seed=7, device="cpu")
        per_step = long.trajectory["recommended_regret_per_step"][short - 3]
        assert per_step == pytest.approx(brief.row["recommended_regret"])

    def test_regime_surface_indexes_the_trace_by_budget(self) -> None:
        trace = np.linspace(1.0, 0.0, 8)                       # regret after 3..10 observations
        frame = pd.DataFrame(
            [{"model": m, "level": lv, "subject": 1, "emg": e, "rep": r, "n_init": 3,
              "recommended_regret_per_step": trace * (1 + lv) * (1.5 if m == "gp_naive" else 1.0)}
             for m in ("gp_mll", "gp_naive") for lv in (1.0, 2.0) for e in (0, 1) for r in (0, 1, 2)]
        )
        df = pd.DataFrame({"level": [1.0, 2.0], "achieved_snr_db": [5.0, -1.0], "knob": "k2_channel"})
        surf = stress_figs.regime_surface(frame, df, knob="k2_channel", model="gp_mll", n_budgets=8)
        assert list(surf["budgets"]) == list(range(4, 11))
        # Rows follow the plotted axis (achieved SNR ascending); regret at budget B is trace[B - n_init].
        np.testing.assert_allclose(surf["value"][1], trace[1:] * 2.0)
        diff = stress_figs.regime_surface(frame, df, knob="k2_channel", model="gp_naive", reference="gp_mll", margin=0.05)
        assert diff["noninferior"][:, -1].all() and not diff["noninferior"][:, 0].any()


@pytest.fixture()
def patched(monkeypatch: pytest.MonkeyPatch) -> list[ChannelData]:
    channels = [_synthetic_real_channel(1, 0), _synthetic_real_channel(1, 1)]
    monkeypatch.setattr(stress_sweep, "iter_channels", lambda *a, **k: iter(channels))
    return channels


def _rows(cfg: Any, tmp_path: Any) -> tuple[list[Any], list[dict]]:
    rows, _, extras = stress_sweep.build_tidy_rows(cfg, "tag", progress=False, store=CellStore(str(tmp_path)))
    return rows, extras


def test_demo1_sweep_runs_on_fitted_twins(patched: list[ChannelData], tmp_path: Any) -> None:
    cfg = load_experiment_config(
        "configs/experiment/stress_k1_decoy_nhp.yaml",
        OVERRIDES + ["knob.levels=[0.0,0.8]", "knob.params.separation=2.0", f"output_root={tmp_path}"],
    )
    rows, _ = _rows(cfg, tmp_path)
    assert {r.dataset for r in rows} == {"synthetic_nhp"} and {r.demo for r in rows} == {"demo1"}
    assert len(rows) == 2 * 2 * 2 * 4                        # channels x levels x models x reps
    captured = [r.decoy_capture for r in rows if r.level == 0.8]
    assert all(c in (0.0, 1.0) for c in captured)
    assert all(r.decoy_capture == 0.0 for r in rows if r.level == 0.0)


def test_split_half_reps_cycle_through_instances(patched: list[ChannelData], tmp_path: Any) -> None:
    cfg = load_experiment_config(
        "configs/experiment/stress_k2_global_nhp.yaml",
        OVERRIDES + ["knob.levels=[0.0,0.5]", "gt_mode=split_half", "gt_n_splits=2", f"output_root={tmp_path}"],
    )
    rows, _ = _rows(cfg, tmp_path)
    by_rep = {(r.subject, r.emg, r.level, r.model, r.rep): r.split_id for r in rows}
    assert {v for k, v in by_rep.items() if k[:4] == (1, 0, 0.0, "gp_mll")} == {0, 1, 2, 3}
    assert all(r.gt_mode == "split_half" and r.gt_reliability is not None for r in rows)


def test_split_half_is_refused_on_demo1(patched: list[ChannelData], tmp_path: Any) -> None:
    cfg = load_experiment_config(
        "configs/experiment/stress_k1_decoy_nhp.yaml", OVERRIDES + ["gt_mode=split_half", f"output_root={tmp_path}"]
    )
    with pytest.raises(ValueError, match="meaningless on Demo 1"):
        _rows(cfg, tmp_path)
