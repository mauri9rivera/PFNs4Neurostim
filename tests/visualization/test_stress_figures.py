"""Stress-figure rules of 2026-09-23: SNR clip, robust R^2 summary, knob wording, no overlapping panels."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pfns4neurostim.visualization import stress as stress_figs
from pfns4neurostim.visualization import style
from pfns4neurostim.visualization import traces

MODELS: tuple[str, ...] = ("gp_mll", "gp_naive", "tabpfn_v2_5")
N_CHANNELS: int = 6
N_REPS: int = 3
LEVELS: tuple[float, ...] = (1.0, 2.0, 4.0)


def _sweep(snr_of: "callable", r2_outlier: bool = False) -> pd.DataFrame:
    """Synthetic K2-like frame; ``snr_of(channel, level)`` gives the achieved SNR of a cell."""
    rng = np.random.default_rng(0)
    rows = []
    for ch in range(N_CHANNELS):
        for level in LEVELS:
            for model in MODELS:
                for rep in range(N_REPS):
                    r2 = 0.5 + 0.05 * rng.normal()
                    if r2_outlier and model == "gp_naive" and ch == 0:
                        r2 = -50.0
                    rows.append({
                        "demo": "demo2", "knob": "k2_channel", "level": level, "subject": ch, "emg": 0, "rep": rep,
                        "model": model, "recommended_regret": 0.1 + 0.02 * level + 0.02 * rng.normal(),
                        "exploration_score": 0.8, "r2": r2, "achieved_snr_db": snr_of(ch, level), "n_init": 5,
                    })
    return pd.DataFrame(rows)


class TestClipLowSnr:
    def test_drops_cells_below_threshold_and_counts_them(self) -> None:
        df = _sweep(lambda ch, lv: 10.0 - 3.0 * lv - ch)  # falls with level and with channel index
        kept, counts = stress_figs.clip_low_snr(df, 0.0, min_channels=1)
        cell = kept.groupby(["subject", "level"])["achieved_snr_db"].mean()
        assert (cell >= 0.0).all()
        by_level = counts.set_index("level")["n_channels_kept"].to_dict()
        assert by_level[1.0] > by_level[2.0] >= by_level[4.0]
        assert (counts["n_channels_total"] == N_CHANNELS).all()

    def test_none_keeps_everything(self) -> None:
        df = _sweep(lambda ch, lv: -5.0)
        kept, counts = stress_figs.clip_low_snr(df, None)
        assert len(kept) == len(df)
        assert counts["kept"].all()

    def test_level_with_too_few_channels_is_dropped_with_a_warning_naming_it(self) -> None:
        df = _sweep(lambda ch, lv: 5.0 if lv < 4.0 else (1.0 if ch == 0 else -1.0))
        with pytest.warns(UserWarning, match=r"level 4 keeps 1 of 6"):
            kept, counts = stress_figs.clip_low_snr(df, 0.0, min_channels=3)
        assert 4.0 not in set(kept["level"])
        assert set(kept["level"]) == {1.0, 2.0}
        assert not counts.set_index("level").loc[4.0, "kept"]

    def test_tidy_frame_is_never_modified(self) -> None:
        df = _sweep(lambda ch, lv: -1.0 * lv)
        before = df.copy()
        with pytest.warns(UserWarning):
            stress_figs.clip_low_snr(df, 0.0)
        pd.testing.assert_frame_equal(df, before)

    def test_unknown_snr_is_kept(self) -> None:
        df = _sweep(lambda ch, lv: np.nan)
        kept, _ = stress_figs.clip_low_snr(df, 0.0)
        assert len(kept) == len(df)


class TestRobustSummary:
    def test_r2_uses_median_so_one_outlier_channel_cannot_drag_the_curve(self) -> None:
        df = _sweep(lambda ch, lv: 10.0, r2_outlier=True)
        sub = df[df["model"] == "gp_naive"]
        _, centre, lower, upper = stress_figs._model_band(sub, "level", "r2")
        assert np.all(centre > 0.3), "the median must ignore the single -50 channel"
        assert np.all(lower <= centre) and np.all(centre <= upper)
        mean_curve = sub.groupby("level")["r2"].mean().to_numpy()
        assert np.all(mean_curve < 0.0), "the plain mean is the thing that gets dragged down"

    def test_regret_keeps_mean_and_ci(self) -> None:
        df = _sweep(lambda ch, lv: 10.0)
        sub = df[df["model"] == "tabpfn_v2_5"]
        _, centre, lower, upper = stress_figs._model_band(sub, "level", "recommended_regret")
        assert np.allclose((lower + upper) / 2, centre)


class TestRenderAll:
    def test_writes_channel_counts_and_figures(self, tmp_path) -> None:
        df = _sweep(lambda ch, lv: 12.0 - 2.0 * lv)
        written = stress_figs.render_all(df, str(tmp_path), knob="k2_channel", dataset="nhp")
        names = {os.path.basename(p) for p in written}
        assert {"channel_counts.csv", "degradation_invivo.svg", "outcomes_invivo.svg", "robustness.csv"} <= names
        counts = pd.read_csv(tmp_path / "channel_counts.csv")
        assert {"level", "n_channels_total", "n_channels_kept", "kept"} <= set(counts.columns)

    def test_negative_snr_is_plotted_by_default(self, tmp_path) -> None:
        """Since the 2026-09-23 restructure SNR < 0 dB is a regime, not a cell to drop."""
        df = _sweep(lambda ch, lv: -3.0 - lv)
        written = stress_figs.render_all(df, str(tmp_path), knob="k2_channel", dataset="nhp")
        assert "degradation_invivo.svg" in {os.path.basename(p) for p in written}

    def test_opt_in_clip_can_still_drop_everything(self, tmp_path) -> None:
        df = _sweep(lambda ch, lv: -3.0)
        with pytest.warns(UserWarning, match="left no level"):
            written = stress_figs.render_all(df, str(tmp_path), knob="k2_channel", dataset="nhp", min_snr_db=0.0)
        assert [os.path.basename(p) for p in written] == ["channel_counts.csv"]

    def test_x_axis_wording_for_k5_and_k6(self, tmp_path) -> None:
        """The figure's x label is the knob's own wording, read back from the saved SVG."""
        for knob, expected in (("k5_outliers", "Contamination"), ("k6_failure", "Failed electrodes")):
            df = _sweep(lambda ch, lv: 10.0).assign(knob=knob)
            out = tmp_path / knob
            stress_figs.render_all(df, str(out), knob=knob, dataset="nhp", min_snr_db=None)
            assert expected in (out / "outcomes_invivo.svg").read_text(encoding="utf-8")


class TestTraceHelpers:
    def test_channel_band_r2_is_median_iqr(self) -> None:
        rows = []
        for ch in range(5):
            for rep in range(2):
                curve = np.full(4, -40.0 if ch == 0 else 0.5)
                rows.append({"subject": ch, "emg": 0, "rep": rep, "r2_per_step": curve})
        centre, lower, upper, n = traces._channel_band(pd.DataFrame(rows), "r2_per_step")
        assert n == 5
        assert np.allclose(centre, 0.5)
        assert np.all(lower <= centre) and np.all(centre <= upper)

    def test_bounded_field_is_mean_with_symmetric_ci(self) -> None:
        rows = [{"subject": ch, "emg": 0, "rep": 0, "recommended_regret_per_step": np.full(3, 0.1 * (ch + 1))}
                for ch in range(4)]
        centre, lower, upper, _ = traces._channel_band(pd.DataFrame(rows), "recommended_regret_per_step")
        assert np.allclose((lower + upper) / 2, centre)

    def test_r2_floor_lives_in_style(self) -> None:
        assert style.R2_AXIS_FLOOR == -1.0
        assert not hasattr(traces, "R2_AXIS_FLOOR")
