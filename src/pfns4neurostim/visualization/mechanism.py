"""Hyp C mechanism figures: OMEGA, placement (MMD / sliced W2), CKA (b) placement, BETA representation.

The legacy ID/OOD dashboard that used to live under this name (its own hardcoded palette)
is in :mod:`pfns4neurostim.legacy_code.visualization_id_ood`; everything here takes geometry,
colours, labels and axis strings from :mod:`visualization.style` (CLAUDE.md §7).
Each ``render_*`` function reads only the CSVs of a run directory, so ``--replot`` rebuilds
every figure without re-running an analysis.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.bayes_update import POOLED_ANCHORS
from ..analysis.representation import SITE_POOL_ALL, channel_summary
from . import style

__all__ = [
    "arm_style",
    "render_bayes_update",
    "render_predictive_link",
    "render_placement",
    "render_cka",
    "render_representation",
]

#: Legend title over the stress-level series of the placement trajectory, so the lines are not an
#: unlabelled ladder of colours. The series were labelled by the raw K2 level (``1 / 2 / 4``) until
#: 2026-10-12; they are now the **absolute achieved SNR** of the level, the same quantity and the same
#: numbers as the x-axis of every Hyp B stress figure, so a reader can place a mechanism panel against a
#: degradation curve (user decision 2026-10-12).
PLACEMENT_LEVEL_LEGEND_TITLE: str = "Achieved SNR (dB)"

#: Half-width of the floor / ceiling reference bands, in placement units.
REFERENCE_BAND_HALF_WIDTH: float = 0.05


def _level_snr_labels(df: pd.DataFrame) -> dict[float, str]:
    """Map each stress level of a mechanism frame to its achieved-SNR legend entry.

    Every mechanism cell records ``achieved_snr_db`` of the stressed channel it ran on, so the stress ladder
    can be reported in the same units as the Hyp B figures instead of as raw knob levels. The level that
    applies no stress is named "default", which is the per-dataset anchor the two hypotheses share.

    Args:
        df: Mechanism frame carrying ``level`` and, ideally, ``achieved_snr_db`` and ``knob``.

    Returns:
        Mapping level -> legend entry. Falls back to the knob's own symbol and the raw level when the frame
        carries no SNR column, so a missing column is visible rather than silently mislabelled.
    """
    from ..data.stress import KNOB_REGISTRY  # noqa: PLC0415 - avoids a data -> visualization import cycle

    levels = sorted(float(level) for level in df["level"].unique())
    knob = str(df["knob"].iloc[0]) if "knob" in df.columns and len(df) else ""
    cls = KNOB_REGISTRY.get(knob)
    nominal = cls.nominal_in(levels) if cls is not None else None
    if "achieved_snr_db" not in df.columns:
        symbol = style.KNOB_SYMBOLS.get(knob, "level")
        return {level: f"{symbol} = {level:g}" for level in levels}
    snr = df.groupby("level")["achieved_snr_db"].mean()
    return {
        level: style.snr_series_label(
            float(snr.loc[level]), is_default=nominal is not None and np.isclose(level, nominal)
        )
        for level in levels
    }


def _has_rows(path: str) -> bool:
    """Whether a CSV exists and holds at least one data row (an empty frame still writes a newline, CRLF on Windows)."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip()) >= 2


def _reference_bands(ax, orientation: str, *, label_it: bool, lines_only: bool = False) -> None:
    """Draw the prior-floor (p = 0) and noise-ceiling (p = 1) reference bands.

    Uses :data:`style.REFERENCE_FLOOR_COLOR` / :data:`style.REFERENCE_CEILING_COLOR` rather than the model
    anchors, which mean TabPFN-2.5 and GP-MLL everywhere else (fixed 2026-09-27), and labels both so they
    reach the legend.

    Args:
        ax: Target axes.
        orientation: ``'v'`` for vertical bands (placement on the x-axis) or ``'h'`` for horizontal ones.
        label_it: Attach legend labels (set on one panel only, so the shared legend holds one entry each).
        lines_only: Draw single reference lines instead of filled bands (for a trajectory panel, where a
            band would hide the curves).
    """
    span = ax.axhspan if orientation == "h" else ax.axvspan
    line = ax.axhline if orientation == "h" else ax.axvline
    for value, color, linestyle, name in (
        (0.0, style.REFERENCE_FLOOR_COLOR, style.REFERENCE_FLOOR_LINESTYLE, "Prior floor (p = 0)"),
        (1.0, style.REFERENCE_CEILING_COLOR, style.REFERENCE_CEILING_LINESTYLE, "Noise ceiling (p = 1)"),
    ):
        label = name if label_it else None
        if lines_only:
            line(value, color=color, lw=style.LINE_WIDTH, ls=linestyle, label=label, zorder=0)
            continue
        half = REFERENCE_BAND_HALF_WIDTH
        span(value - half, value + half, color=color, alpha=style.REFERENCE_BAND_ALPHA, lw=0,
             label=label, zorder=0)
        line(value, color=color, lw=style.LINE_WIDTH / 2, ls=linestyle, zorder=0)


#: The engines of the update analysis that are tabpfn_v2_5 run differently, as shades of the TabPFN colour.
_TABPFN_VARIANTS: tuple[str, ...] = ("tabpfn_v2_5", "tabpfn_v2_5_t1")
#: Update-analysis engine -> model key of the palette (one identity per reference GP).
_ENGINE_MODEL: dict[str, str] = {"gp_bayes": "gp_bayes", "gp_mll_refit": "gp_mll", "gp_fixed": "gp_naive",
                                 "gp_mll_frozen": "gp_mll_frozen"}


def arm_style(engine: str) -> style.ModelStyle:
    """Style of an update-analysis engine.

    Args:
        engine: Engine name: a TabPFN variant (:data:`_TABPFN_VARIANTS`), ``gp_bayes``, ``gp_mll_refit``,
            ``gp_fixed``, or any model key of the palette.

    Returns:
        The model style. TabPFN variants are shades of the TabPFN colour (the deployed ``tabpfn_v2_5`` is the
        anchor); ``gp_mll_refit`` is GP-MLL and ``gp_fixed`` is GP-fixed.
    """
    if engine in _TABPFN_VARIANTS[1:]:
        return style.series_style("tabpfn_v2_5", engine, list(_TABPFN_VARIANTS))
    st = style.model_style(_ENGINE_MODEL.get(engine, engine))
    return dataclasses.replace(st, label=style.ENGINE_LABELS[engine]) if engine in style.ENGINE_LABELS else st


#: Engines left out of every Hyp C figure; their rows stay in the CSVs. The T 1.0 temperature control is visually
#: indistinguishable from the deployed T 0.9, the default the benchmark runs (user, 2026-10-22).
FIGURE_EXCLUDED_ENGINES: tuple[str, ...] = ("tabpfn_v2_5_t1",)


def _shown(df: pd.DataFrame, col: str = "engine") -> pd.DataFrame:
    """Rows of the engines the figures draw (:data:`FIGURE_EXCLUDED_ENGINES` removed)."""
    return df[~df[col].isin(FIGURE_EXCLUDED_ENGINES)] if col in df else df


def _channel_band(df: pd.DataFrame, col: str, x: str) -> pd.DataFrame:
    """Median over channels per ``x`` with its within-subject channel-bootstrap interval (BETA D4).

    Args:
        df: Rows holding ``subject``, ``emg``, ``draw``, ``x`` and ``col`` (one row per cell, or several that the
            within-channel median reduces).
        col: Value column.
        x: Column on the x-axis.

    Returns:
        Columns ``x``, ``median``, ``ci_low``, ``ci_high``, sorted by ``x``.
    """
    sub = df.dropna(subset=[col])
    if sub.empty:
        return pd.DataFrame(columns=[x, "median", "ci_low", "ci_high"])
    out = channel_summary(sub, [col], [x], n_boot=style.HYPC_BAND_N_BOOT, ci=style.HYPC_BAND_CI,
                          seed=style.HYPC_BAND_SEED)
    return out.sort_values(x)[[x, "median", "ci_low", "ci_high"]]


# ---------------------------------------------------------------------------
# bayes_update (OBJECTIVE OMEGA, 2026-10-15): Omega-1 ... Omega-5
# ---------------------------------------------------------------------------
#: Part A headline metrics of Omega-1, one column each, with the value every coherent Bayesian model scores.
OMEGA_COHERENCE_METRICS: dict[str, float] = {"drift_abs": 0.0, "total_var_ratio": 1.0, "cov_asymmetry": 0.0,
                                            "latent_frac": float("nan"), "update_share": float("nan"),
                                            "drift_abs_anchor": 0.0}
#: Cell keys of the per-anchor rows, to reduce them to one row per cell, engine and pool.
_OMEGA_CELL_KEYS: tuple[str, ...] = ("dataset", "subject", "emg", "knob", "level", "context_t", "draw", "engine")


def _anchor_readouts(anchors: pd.DataFrame) -> pd.DataFrame:
    """Per-cell readouts that need every anchor, from ``coherence.csv`` (M15, user 2026-10-23).

    ``drift_abs`` and ``total_var_ratio`` are medians over the influenced sites I and are undefined at an anchor where
    the update moves no site by ``eps s^2``, so a cell's value rests on its anchors with a non-empty I only. This
    returns, per cell, engine and pool (and over every anchor as :data:`POOLED_ANCHORS`): ``update_share``, the share
    of anchors with a non-empty I, and ``drift_abs_anchor``, the median over anchors of the drift at x' itself, which
    is defined everywhere.

    Args:
        anchors: ``coherence.csv`` rows (one per anchor), with ``pool``, ``n_influenced`` and ``drift_abs_anchor``.

    Returns:
        One row per cell key, engine and pool with the two readouts.
    """
    keys = [k for k in _OMEGA_CELL_KEYS if k in anchors]
    a = anchors.assign(update_share=(anchors["n_influenced"] > 0).astype(float))
    parts = [a.groupby([*keys, "pool"])[["update_share", "drift_abs_anchor"]].agg(
                 {"update_share": "mean", "drift_abs_anchor": "median"}).reset_index(),
             a.groupby(keys)[["update_share", "drift_abs_anchor"]].agg(
                 {"update_share": "mean", "drift_abs_anchor": "median"}).reset_index().assign(pool=POOLED_ANCHORS)]
    return pd.concat(parts, ignore_index=True)
#: Part B headline contrasts of Omega-3, with the value of a perfect match to the reference.
OMEGA_CONTRAST_METRICS: dict[str, float] = {"transfer_slope": 1.0, "footprint_r": 1.0, "gap_change": 0.0,
                                           "delta_crps": 0.0}
#: Panel height / width of the Omega line grids, and of the exemplar maps.
_OMEGA_PANEL_ASPECT: float = 0.8
_OMEGA_MAP_ASPECT: float = 0.7


def _nominal(df: pd.DataFrame) -> pd.DataFrame:
    """Rows at the nominal (lowest) stress level -- the headline operating point."""
    return df[np.isclose(df["level"], df["level"].min())]


def _median_band(ax: Any, df: pd.DataFrame, col: str, st: style.ModelStyle, x: str = "context_t") -> None:
    """Median over channels vs ``x`` with its channel-bootstrap interval, in one engine's style.

    Engines of :data:`style.HYPC_LINE_ONLY_ENGINES` get the median line alone.

    Args:
        ax: Target axes.
        df: One engine's rows (``subject``, ``emg``, ``draw``, ``engine``, ``x``, ``col``).
        col: Value column.
        st: Engine style.
        x: Column on the x-axis.
    """
    band = _channel_band(df, col, x)
    if band.empty:
        return
    ax.plot(band[x], band["median"], color=st.color, ls=st.linestyle, marker=st.marker, label=st.label,
            zorder=st.zorder)
    engines = set(df["engine"]) if "engine" in df else set()
    if not engines & set(style.HYPC_LINE_ONLY_ENGINES):
        ax.fill_between(band[x], band["ci_low"], band["ci_high"], color=st.color, alpha=style.BAND_ALPHA, lw=0)


def _bayes_value(ax: Any, value: float) -> None:
    """The value every coherent Bayesian model scores, as a thin neutral line."""
    if np.isfinite(value):
        ax.axhline(value, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":", zorder=0)


def _context_axis(ax: Any, ts: Sequence[int]) -> None:
    """Context-size axis with ticks at the measured sizes only."""
    ax.set_xticks(list(ts), [str(int(t)) for t in ts])
    ax.set_xlabel(style.axis_label("context_t"))


#: Panels per row of the pooled coherence figure.
_COHERENCE_COLUMNS: int = 2


#: Supplementary readouts of the variance identities: the bounded shrink / move index and the log total-variance ratio
#: (user, 2026-10-23), both 0 for a Bayesian update.
OMEGA_VARIANCE_INDEX_METRICS: dict[str, float] = {"shrink_move_index": 0.0, "log_total_var_ratio": 0.0}


def _omega_variance_index(pooled: pd.DataFrame, ts: Sequence[int], run_dir: str) -> list[str]:
    """Supplementary figure ``coherence_variance_index.svg``: the bounded / log companions of the variance ratios.

    Args:
        pooled: Pooled-anchor ``coherence_cell.csv`` rows at the nominal level.
        ts: Context sizes on the x-axis.
        run_dir: Output directory.

    Returns:
        Written paths (none when the run predates the columns).
    """
    metrics = [m for m in OMEGA_VARIANCE_INDEX_METRICS if m in pooled and pooled[m].notna().any()]
    if not metrics:
        return []
    fig, axes = style.figure("double", nrows=1, ncols=len(metrics), squeeze=False,
                             panel_aspect=_OMEGA_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
    for ax, metric in zip(axes[0], metrics):
        for engine in pooled["engine"].unique():
            _median_band(ax, pooled[pooled["engine"] == engine], metric, arm_style(engine))
        _bayes_value(ax, OMEGA_VARIANCE_INDEX_METRICS[metric])
        _context_axis(ax, ts)
        ax.set_ylabel(style.axis_label(metric))
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    style.panel_letters(axes[0][:len(metrics)])
    return style.save_figure(fig, run_dir, "coherence_variance_index")


def _omega_coherence(cell: pd.DataFrame, run_dir: str) -> list[str]:
    """Omega-1: Part A headlines vs context size, over every anchor (``coherence.svg``) and per pool (supplement).

    The pooled figure is the headline (user, 2026-10-22); ``coherence_by_pool.svg`` keeps the pre-registered split,
    since a repeated measurement at a site already in D (observed anchor) and an extrapolation (unobserved anchor)
    test the update in different regimes.

    Args:
        cell: ``coherence_cell.csv`` rows.
        run_dir: Output directory.

    Returns:
        Written paths.
    """
    cell = _shown(_nominal(cell))
    metrics = [m for m in OMEGA_COHERENCE_METRICS if m in cell]
    ts = sorted(cell["context_t"].unique())
    written: list[str] = []
    pooled = cell[cell["pool"] == POOLED_ANCHORS]
    if not pooled.empty:
        ncols = _COHERENCE_COLUMNS
        fig, axes = style.figure("double", nrows=-(-len(metrics) // ncols), ncols=ncols, squeeze=False,
                                 panel_aspect=_OMEGA_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
        for ax, metric in zip(axes.ravel(), metrics):
            for engine in pooled["engine"].unique():
                _median_band(ax, pooled[pooled["engine"] == engine], metric, arm_style(engine))
            _bayes_value(ax, OMEGA_COHERENCE_METRICS[metric])
            _context_axis(ax, ts)
            ax.set_ylabel(style.axis_label(metric))
        for ax in axes.ravel()[len(metrics):]:
            ax.set_axis_off()
        axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
        style.panel_letters(axes.ravel()[:len(metrics)])
        written += style.save_figure(fig, run_dir, "coherence")
        written += _omega_variance_index(pooled, ts, run_dir)
    pools = [p for p in ("unobserved", "observed") if p in set(cell["pool"])]
    if pools:
        fig, axes = style.figure("wide", nrows=len(pools), ncols=len(metrics), squeeze=False,
                                 panel_aspect=_OMEGA_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
        for r, pool in enumerate(pools):
            sub = cell[cell["pool"] == pool]
            for c, metric in enumerate(metrics):
                ax = axes[r][c]
                for engine in sub["engine"].unique():
                    _median_band(ax, sub[sub["engine"] == engine], metric, arm_style(engine))
                _bayes_value(ax, OMEGA_COHERENCE_METRICS[metric])
                _context_axis(ax, ts)
                ax.set_ylabel(style.axis_label(metric))
                ax.set_title(f"{pool} anchors", fontsize=style.FONT_SIZES["legend"])
        axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
        style.panel_letters(axes.ravel())
        written += style.save_figure(fig, run_dir, "coherence_by_pool")
    return written


#: Legend of the implied-noise scatter: below the axes (axes coordinates), markers enlarged to be readable.
_IMPLIED_NOISE_LEGEND_ANCHOR: tuple[float, float] = (0.5, -0.22)
_SCATTER_LEGEND_MARKER_SCALE: float = 3.0


def _omega_implied_noise(cell: pd.DataFrame, coh: pd.DataFrame, run_dir: str) -> list[str]:
    """Omega-2: how much of the predictive variance at x' a model treats as noise, and whether that noise is real.

    (a) Latent share C(x', x') / s^2(x') vs t, over every anchor: the part of the predictive variance at x' that an
    observation there can remove (1 - share = what the model calls observation noise). (b) Implied noise
    s^2(x') - C(x', x') against the empirical variance of the repeated trials at x', one point per anchor, on log
    axes (the values span four decades); a model that has learned the site's noise sits on the diagonal. Implied noise
    <= 0 (C(x', x') >= s^2(x'), impossible for a coherent model) cannot be drawn on a log axis, so its share is in the
    legend next to the median Spearman rho over cells.

    Args:
        cell: ``coherence_cell.csv`` rows.
        coh: ``coherence.csv`` rows (one per anchor).
        run_dir: Output directory.

    Returns:
        Written paths.
    """
    cell, coh = _shown(_nominal(cell)), _shown(_nominal(coh))
    pooled = cell[cell["pool"] == POOLED_ANCHORS]
    fig, axes = style.figure("double", ncols=2, squeeze=False, panel_aspect=_OMEGA_PANEL_ASPECT,
                             layout=style.LAYOUT_ENGINE)
    ax = axes[0][0]
    for engine in pooled["engine"].unique():
        _median_band(ax, pooled[pooled["engine"] == engine], "latent_frac", arm_style(engine))
    _context_axis(ax, sorted(cell["context_t"].unique()))
    ax.set_ylabel(style.axis_label("latent_frac"))
    ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    ax = axes[0][1]
    positive = coh[(coh["trial_var"] > 0) & (coh["implied_noise"] > 0)]
    for engine in coh["engine"].unique():
        st = arm_style(engine)
        sub = coh[coh["engine"] == engine]
        rho = pooled[pooled["engine"] == engine]["sigma_recovery_rho"].median()
        neg = float((sub["implied_noise"] <= 0).mean())
        pts = positive[positive["engine"] == engine]
        ax.scatter(pts["trial_var"], pts["implied_noise"], s=style.MARKER_SIZE, color=st.color, marker=st.marker,
                   alpha=style.FAINT_ALPHA * 3, lw=0,
                   label=style.IMPLIED_NOISE_LEGEND.format(model=st.label, rho=rho, neg=100 * neg))
    vals = positive[["trial_var", "implied_noise"]].to_numpy()
    lim = [float(np.nanmin(vals)), float(np.nanmax(vals))]
    ax.plot(lim, lim, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(style.axis_label("trial_var"))
    ax.set_ylabel(style.axis_label("implied_noise"))
    ax.legend(loc="upper center", bbox_to_anchor=_IMPLIED_NOISE_LEGEND_ANCHOR, fontsize=style.FONT_SIZES["legend"] - 1,
              frameon=False, markerscale=_SCATTER_LEGEND_MARKER_SCALE)
    style.panel_letters(axes.ravel())
    return style.save_figure(fig, run_dir, "implied_noise")


def _omega_headtohead(h2h: pd.DataFrame, run_dir: str) -> list[str]:
    """Omega-3: site-matched contrasts with each reference vs t (unobserved anchors, trial source).

    Every other engine is drawn against each reference, so the GP-vs-GP spread sits next to TabPFN's.
    """
    sub = _shown(_nominal(h2h))
    sub = sub[(sub["reference"] != "none") & (sub["source"] == "trial") & (sub["pool"] == "unobserved")]
    refs = [r for r in ("gp_bayes", "gp_mll_refit", "gp_fixed") if r in set(sub["reference"])]
    metrics = [m for m in OMEGA_CONTRAST_METRICS if m in sub]
    if not refs or not metrics:
        return []
    ts = sorted(sub["context_t"].unique())
    keys = ["subject", "emg", "level", "context_t", "draw", "engine", "reference"]
    per_cell = sub.groupby(keys)[metrics].median().reset_index()                  # anchors -> cell
    fig, axes = style.figure("wide", nrows=len(refs), ncols=len(metrics), squeeze=False,
                             panel_aspect=_OMEGA_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
    for r, ref in enumerate(refs):
        for c, metric in enumerate(metrics):
            ax = axes[r][c]
            for engine in per_cell["engine"].unique():
                _median_band(ax, per_cell[(per_cell["engine"] == engine) & (per_cell["reference"] == ref)], metric,
                             arm_style(engine))
            _bayes_value(ax, OMEGA_CONTRAST_METRICS[metric])
            _context_axis(ax, ts)
            ax.set_ylabel(style.axis_label(metric))
            ax.set_title(f"vs {arm_style(ref).label}", fontsize=style.FONT_SIZES["legend"])
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    style.panel_letters(axes.ravel())
    return style.save_figure(fig, run_dir, "headtohead")


def _omega_surprise(surprise: pd.DataFrame, panel_ts: Sequence[int], run_dir: str,
                    engines: Sequence[str] | None = None, name: str = "surprise_response",
                    reference: str | None = None) -> list[str]:
    """Omega-4 (and C1): update at the anchor vs injected surprise, one panel per context size.

    Both axes are in units of the ladder reference's predictive SD at the anchor (``bayes_update.shared.reference``).

    Args:
        surprise: ``surprise.csv`` rows.
        panel_ts: Context sizes drawn, one panel each.
        run_dir: Output directory.
        engines: Engines drawn (``None`` = every engine of the table).
        name: Figure name.
        reference: The run's ladder reference, named in the figure title (``None`` = no title).

    Returns:
        Written paths.
    """
    sub = _shown(_nominal(surprise))
    sub = sub[sub["pool"] == "unobserved"]
    if engines is not None:
        sub = sub[sub["engine"].isin(list(engines))]
    ts = [t for t in panel_ts if t in set(sub["context_t"])]
    if not ts:
        return []
    cell_keys = ["subject", "emg", "level", "context_t", "draw", "engine"]
    per_cell = sub.groupby([*cell_keys, "c"])["update_anchor_sd"].median().reset_index()   # anchors -> cell
    # least-squares slope through the origin of the update on the surprise, per anchor, then per cell
    slope = (sub.assign(cu=sub["c"] * sub["update_anchor_sd"], cc=sub["c"] ** 2)
             .groupby([*cell_keys, "anchor"])[["cu", "cc"]].sum())
    slope = (slope["cu"] / slope["cc"]).rename("surprise_slope").reset_index()
    slope = slope.groupby(cell_keys)["surprise_slope"].median().reset_index()
    fig, axes = style.figure("wide", ncols=len(ts) + 1, squeeze=False, panel_aspect=_OMEGA_PANEL_ASPECT,
                             layout=style.LAYOUT_ENGINE)
    for ax, t in zip(axes[0], ts):
        at = per_cell[per_cell["context_t"] == t]
        for engine in at["engine"].unique():
            _median_band(ax, at[at["engine"] == engine], "update_anchor_sd", arm_style(engine), x="c")
        ax.axhline(0.0, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":", zorder=0)
        ax.set_xlabel(style.axis_label("ladder_c"))
        ax.set_ylabel(style.axis_label("update_anchor_sd"))
        ax.set_title(f"$t$ = {int(t)}", fontsize=style.FONT_SIZES["legend"])
    ax = axes[0][-1]
    for engine in slope["engine"].unique():
        _median_band(ax, slope[slope["engine"] == engine], "surprise_slope", arm_style(engine))
    _bayes_value(ax, 1.0)
    _context_axis(ax, sorted(slope["context_t"].unique()))
    ax.set_ylabel(style.axis_label("surprise_slope"))
    ax.set_title("slope vs context size", fontsize=style.FONT_SIZES["legend"])
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    style.panel_letters(axes.ravel())
    if reference is not None:
        fig.suptitle(style.SURPRISE_REFERENCE_TITLE.format(reference=arm_style(reference).label),
                     x=0.01, ha="left", fontsize=style.FONT_SIZES["title"])
    return style.save_figure(fig, run_dir, name)


def _omega_exemplar(run_dir: str, npz: str = "exemplar.npz", key: str = "C", label: str = "implied_cov",
                    name: str = "implied_cov_exemplar") -> list[str]:
    """Maps on the electrode grid at one common anchor, one row per engine, one column per t.

    Omega-5 draws the implied covariance C(x', .) (``exemplar.npz``); Hypothesis C1 draws the change of the mean after
    the ladder observation (``c1_exemplar.npz``, key ``dmu``).

    Args:
        run_dir: Run directory holding ``npz``.
        npz: Exemplar file.
        key: Array of maps, shape [t, engine, M].
        label: Colour-bar axis key.
        name: Figure name.

    Returns:
        Written paths.
    """
    path = os.path.join(run_dir, npz)
    if not os.path.exists(path):
        return []
    ex = np.load(path)
    C, ts, engines, xy = ex[key], ex["context_ts"], [str(e) for e in ex["engines"]], ex["ch2xy"].astype(int)
    shape = tuple(xy.max(axis=0) + 1)
    fig, axes = style.figure("full", nrows=len(engines), ncols=len(ts), squeeze=False, panel_aspect=_OMEGA_MAP_ASPECT,
                             layout=style.LAYOUT_ENGINE)
    for r, engine in enumerate(engines):
        vmax = float(np.abs(C[:, r]).max()) or 1.0
        for c, t in enumerate(ts):
            ax = axes[r][c]
            grid = np.full(shape, np.nan)
            grid[xy[:, 0], xy[:, 1]] = C[c, r]
            im = ax.imshow(grid.T, origin="lower", cmap=style.DIVERGING_CMAP, vmin=-vmax, vmax=vmax)
            obs = ex["observed"][c]
            ax.scatter(xy[obs, 0], xy[obs, 1], s=style.MARKER_SIZE, color=style.NEUTRAL_GREY, marker=".", lw=0)
            a = int(ex["anchor"])
            ax.scatter([xy[a, 0]], [xy[a, 1]], s=style.MARKER_SIZE * 4, facecolors="none",
                       edgecolors=style.NEUTRAL_GREY, lw=style.LINE_WIDTH)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"$t$ = {int(t)}", fontsize=style.FONT_SIZES["legend"])
            if c == 0:
                ax.set_ylabel(arm_style(engine).label)
        fig.colorbar(im, ax=list(axes[r]), label=style.axis_label(label))
    return style.save_figure(fig, run_dir, name)


# ---------------------------------------------------------------------------
# Hypothesis C1 (2026-10-19): is TabPFN's update GP-like? Descriptive, against GP-MLL frozen
# ---------------------------------------------------------------------------
#: C1 readouts with the value a perfect GP scores against the reference; ``None`` = the reference's own value (its
#: curve is drawn instead of a constant line). Source of each: the contrast rows or the per-engine descriptors.
C1_READOUTS: dict[str, tuple[float | None, str, str]] = {
    "transfer_slope": (1.0, "contrast", "trial"),
    "footprint_r": (1.0, "contrast", "trial"),
    "gain": (None, "engine", "trial"),
    "far_field_share": (None, "engine", "trial"),
    "saturation": (1.0, "engine", "ladder"),
}
#: Panels per row of the C1 readouts figure (five readouts + the legend slot on a 2 x 3 grid).
_C1_READOUT_COLUMNS: int = 3


def _c1_readouts(h2h: pd.DataFrame, reference: str, engines: Sequence[str], run_dir: str) -> list[str]:
    """C1: every GP-likeness readout vs context size, one panel each, against the perfect-GP value.

    Unobserved anchors at the nominal level; per anchor -> cell median -> median and IQR over cells.

    Args:
        h2h: ``headtohead.csv`` rows.
        reference: The C1 reference (GP-MLL frozen).
        engines: Engines drawn.
        run_dir: Output directory.

    Returns:
        Written paths.
    """
    engines = [e for e in engines if e not in FIGURE_EXCLUDED_ENGINES]
    sub = _nominal(h2h)
    sub = sub[(sub["pool"] == "unobserved") & sub["engine"].isin(list(engines))]
    panels = [m for m in C1_READOUTS if m in sub]
    if not panels or reference not in set(sub["engine"]) | set(sub["reference"]):
        return []
    ts = sorted(sub["context_t"].unique())
    keys = ["subject", "emg", "level", "context_t", "draw", "engine"]
    # two rows, so axis labels never run into the neighbouring panel; the spare slot holds the legend, which
    # must list the reference too (panel (a) skips it: a contrast against itself is not drawn)
    ncols = _C1_READOUT_COLUMNS
    nrows = -(-(len(panels) + 1) // ncols)
    fig, axes = style.figure("wide", nrows=nrows, ncols=ncols, squeeze=False, panel_aspect=_OMEGA_PANEL_ASPECT,
                             layout=style.LAYOUT_ENGINE)
    flat = list(axes.ravel())
    for ax, metric in zip(flat, panels):
        perfect, kind, source = C1_READOUTS[metric]
        rows = sub[(sub["source"] == source)
                   & ((sub["reference"] == reference) if kind == "contrast" else (sub["reference"] == "none"))]
        per_cell = rows.groupby(keys)[metric].median().reset_index()
        for engine in engines:
            if kind == "contrast" and engine == reference:
                continue
            _median_band(ax, per_cell[per_cell["engine"] == engine], metric, arm_style(engine))
        if perfect is not None:
            _bayes_value(ax, perfect)
        _context_axis(ax, ts)
        ax.set_ylabel(style.axis_label(metric))
    handles: dict[str, Any] = {}
    for ax in flat[:len(panels)]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    for ax in flat[len(panels):]:
        ax.set_axis_off()
    flat[len(panels)].legend(list(handles.values()), list(handles), loc="center", frameon=False,
                             fontsize=style.FONT_SIZES["legend"])
    style.panel_letters(flat[:len(panels)])
    return style.save_figure(fig, run_dir, "c1_gp_like_readouts")


def _c1_distance(profile: pd.DataFrame, engines: Sequence[str], panel_ts: Sequence[int], run_dir: str) -> list[str]:
    """C1: mean |transfer| vs distance from the anchor (unobserved sites, unobserved anchors), one panel per t.

    Args:
        profile: ``transfer_profile.csv`` rows.
        engines: Engines drawn.
        panel_ts: Context sizes drawn.
        run_dir: Output directory.

    Returns:
        Written paths.
    """
    engines = [e for e in engines if e not in FIGURE_EXCLUDED_ENGINES]
    sub = _nominal(profile)
    sub = sub[(sub["pool"] == "unobserved") & (sub["site_pool"] == "unobserved") & sub["engine"].isin(list(engines))]
    ts = [t for t in panel_ts if t in set(sub["context_t"])]
    if not ts:
        return []
    sub = sub.assign(bin_mid=0.5 * (sub["bin_lo"] + sub["bin_hi"]))
    keys = ["subject", "emg", "level", "context_t", "draw", "engine", "bin_mid"]
    per_cell = sub.groupby(keys)["abs_transfer"].median().reset_index()
    fig, axes = style.figure("wide" if len(ts) > 2 else "double", ncols=len(ts), squeeze=False,
                             panel_aspect=_OMEGA_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
    for ax, t in zip(axes[0], ts):
        at = per_cell[per_cell["context_t"] == t]
        for engine in engines:
            _median_band(ax, at[at["engine"] == engine], "abs_transfer", arm_style(engine), x="bin_mid")
        ax.set_xlabel(style.axis_label("distance_pitch"))
        ax.set_ylabel(style.axis_label("abs_transfer"))
        ax.set_title(f"$t$ = {int(t)}", fontsize=style.FONT_SIZES["legend"])
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    style.panel_letters(axes.ravel())
    return style.save_figure(fig, run_dir, "c1_update_vs_distance")


# ---------------------------------------------------------------------------
# M8 predictive link (exploratory, 2026-10-21): per-channel scatter + random-intercept slope
# ---------------------------------------------------------------------------
#: Panels per row of a predictive-link figure (one panel per predictor x reference pair).
_LINK_COLUMNS: int = 5
#: Panel height / width of the predictive-link scatters.
_LINK_PANEL_ASPECT: float = 0.8
#: Scatter marker area, in multiples of ``style.MARKER_SIZE``.
_LINK_MARKER_SCALE: float = 4.0
#: Panel letters run (a) ... (z); a larger grid is an analysis figure and goes unlettered.
_MAX_PANEL_LETTERS: int = 26


def _link_title(row: pd.Series) -> str:
    """Panel title naming what the predictor is read against (engine alone, contrast, or matched difference)."""
    model, ref = arm_style(str(row["model"])).label, str(row["reference"])
    if ref == "none":
        key = "none"
    else:
        key = "contrast" if row["kind"] == "contrast" else "difference"
    ref_label = arm_style(ref).label if ref != "none" else ""
    return style.LINK_REFERENCE_TITLES[key].format(model=model, reference=ref_label)


def _predictive_link_figure(links: pd.DataFrame, points: pd.DataFrame, outcome: str, run_dir: str) -> list[str]:
    """One outcome: a scatter per (predictor, reference) pair, one colour per subject, with the fitted slope.

    Args:
        links: ``predictive_link.csv`` rows.
        points: ``predictive_link_channels.csv`` rows (one per channel x predictor x reference, outcome columns).
        outcome: Outcome drawn.
        run_dir: Output directory.

    Returns:
        Written paths.
    """
    rows = links[links["outcome"] == outcome].reset_index(drop=True)
    if rows.empty or outcome not in points:
        return []
    subjects = sorted(points["subject"].unique())
    n = len(rows)
    ncols = min(_LINK_COLUMNS, n)
    nrows = -(-n // ncols)
    fig, axes = style.figure("full" if ncols > 2 else "double", nrows=nrows, ncols=ncols, squeeze=False,
                             panel_aspect=_LINK_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
    flat = list(axes.ravel())
    for ax, (_, row) in zip(flat, rows.iterrows()):
        pts = points[(points["predictor"] == row["predictor"]) & (points["reference"] == row["reference"])]
        pts = pts.dropna(subset=["value", outcome])
        for subject in subjects:
            st = style.subject_style(subject, subjects)
            sp = pts[pts["subject"] == subject]
            ax.scatter(sp["value"], sp[outcome], s=style.MARKER_SIZE * _LINK_MARKER_SCALE, color=st.color,
                       marker=st.marker, label=st.label, lw=0, zorder=st.zorder)
        xs = np.array([pts["value"].min(), pts["value"].max()])                    # [2]
        ax.plot(xs, row["intercept"] + row["slope"] * xs, color=style.OUTLINE_COLOR, lw=style.LINE_WIDTH)
        ax.text(0.02, 0.98, style.LINK_SLOPE_NOTE.format(slope=row["slope"], lo=row["ci_low"], hi=row["ci_high"]),
                transform=ax.transAxes, va="top", ha="left", fontsize=style.FONT_SIZES["annotation"])
        ax.set_xlabel(style.axis_label(str(row["predictor"])))
        ax.set_ylabel(style.axis_label(outcome))
        ax.set_title(_link_title(row), fontsize=style.FONT_SIZES["legend"])
    for ax in flat[n:]:
        ax.set_axis_off()
    handles = [plt.Line2D([], [], ls="", marker=style.subject_style(s, subjects).marker,
                          color=style.subject_style(s, subjects).color, label=style.subject_style(s, subjects).label)
               for s in subjects]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False,
               fontsize=style.FONT_SIZES["legend"])
    first = rows.iloc[0]
    fig.suptitle(style.LINK_TITLE.format(outcome=style.axis_label(outcome).split("\n")[0],
                                         model=arm_style(str(first["model"])).label, pool=first["pool"],
                                         level=float(first["level"])),
                 x=0.01, ha="left", fontsize=style.FONT_SIZES["title"])
    if n <= _MAX_PANEL_LETTERS:
        style.panel_letters(flat[:n])
    return style.save_figure(fig, run_dir, f"predictive_link_{outcome}")


def render_predictive_link(run_dir: str) -> list[str]:
    """M8: one figure per outcome of ``predictive_link.csv``, from the run's files alone.

    Args:
        run_dir: ``bayes_update`` run directory.

    Returns:
        Written paths (none when the link was not run).
    """
    links_path = os.path.join(run_dir, "predictive_link.csv")
    points_path = os.path.join(run_dir, "predictive_link_channels.csv")
    if not (_has_rows(links_path) and _has_rows(points_path)):
        return []
    links, points = pd.read_csv(links_path), pd.read_csv(points_path)
    if links.empty:
        return []
    written: list[str] = []
    for outcome in links["outcome"].unique():
        written += _predictive_link_figure(links, points, str(outcome), run_dir)
    return written


def render_bayes_update(run_dir: str) -> list[str]:
    """Every OMEGA figure (Omega-1 ... Omega-5) and the Hypothesis C1 figures, from the run's files alone.

    Args:
        run_dir: ``bayes_update`` run directory.

    Returns:
        Written paths.
    """
    import yaml  # noqa: PLC0415

    written: list[str] = []
    paths = {k: os.path.join(run_dir, f"{k}.csv") for k in ("coherence", "coherence_cell", "headtohead", "surprise")}
    if _has_rows(paths["coherence_cell"]):
        cell = pd.read_csv(paths["coherence_cell"])
        anchors = pd.read_csv(paths["coherence"]) if _has_rows(paths["coherence"]) else None
        if anchors is not None and {"n_influenced", "drift_abs_anchor", "pool"} <= set(anchors):
            derived = _anchor_readouts(anchors)
            cell = cell.drop(columns=[c for c in ("update_share", "drift_abs_anchor") if c in cell])
            cell = cell.merge(derived, on=[k for k in (*_OMEGA_CELL_KEYS, "pool") if k in cell and k in derived],
                              how="left")
        written += _omega_coherence(cell, run_dir)
        if anchors is not None:
            written += _omega_implied_noise(cell, anchors, run_dir)
    if _has_rows(paths["headtohead"]):
        written += _omega_headtohead(pd.read_csv(paths["headtohead"]), run_dir)
    cfg_path = os.path.join(run_dir, "config.yaml")
    cfg_block: dict[str, Any] = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            cfg_block = yaml.safe_load(fh).get("bayes_update") or {}
    panel_ts: Sequence[int] = cfg_block.get("panel_ts") or ()
    ladder_ref = (cfg_block.get("shared") or {}).get("reference")        # the surprise unit (user, 2026-10-21)
    if _has_rows(paths["surprise"]):
        written += _omega_surprise(pd.read_csv(paths["surprise"]), panel_ts, run_dir, reference=ladder_ref)
    written += _omega_exemplar(run_dir)
    # Hypothesis C1: the GP-like figures (only when the run has the C1 reference)
    c1 = cfg_block.get("c1") or {}
    ref, c1_engines = c1.get("reference"), list(c1.get("engines") or ())
    if ref and c1_engines:
        if _has_rows(paths["headtohead"]):
            written += _c1_readouts(pd.read_csv(paths["headtohead"]), ref, c1_engines, run_dir)
        prof = os.path.join(run_dir, "transfer_profile.csv")
        if _has_rows(prof):
            written += _c1_distance(pd.read_csv(prof), c1_engines, panel_ts, run_dir)
        if _has_rows(paths["surprise"]):
            written += _omega_surprise(pd.read_csv(paths["surprise"]), panel_ts, run_dir, engines=c1_engines,
                                       name="c1_surprise_response", reference=ladder_ref)
        written += _omega_exemplar(run_dir, "c1_exemplar.npz", "dmu", "dmu_exemplar", "c1_update_exemplar")
    written += render_predictive_link(run_dir)            # M8, only when the link was run
    return written


def render_placement(run_dir: str) -> list[str]:
    """Placement context trajectory (formulation C).

    The formulation-B strip plot (``placement_strip``: one dot per channel, x = placement p, four
    floor/ceiling pair definitions side by side) was **dropped on 2026-09-30**. Three of its four columns
    existed only to justify the choice of the fourth, one of them (split-half self / shuffled self) rested on
    the split-half machinery retired everywhere else, and the per-channel scatter it showed is carried by the
    trajectory panel at every context size. ``placement.csv`` still records all four pairs with their
    bootstrap CIs and gaps, so the pair comparison remains auditable as a table.

    Args:
        run_dir: Placement run directory.

    Returns:
        Written paths.
    """
    written: list[str] = []
    path_b = os.path.join(run_dir, "placement.csv")
    path_c = os.path.join(run_dir, "placement_context.csv")
    if not _has_rows(path_c):
        return written
    ctx = pd.read_csv(path_c)
    if ctx.empty:
        return written
    metrics = sorted(pd.read_csv(path_b)["metric"].unique()) if os.path.exists(path_b)         else sorted(ctx["metric"].unique())
    ctx = ctx[~ctx["is_full"].astype(bool)]
    levels = sorted(ctx["level"].unique())
    snr_labels = _level_snr_labels(ctx)
    cmap = plt.get_cmap(style.LEVEL_CMAP)
    fig, axes = style.figure("wide", ncols=len(metrics), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        sub = ctx[ctx["metric"] == metric]
        for i, level in enumerate(levels):
            # Sequential colour, dark = nominal, light = most degraded. Grey shades collided with the
            # reference bands (changed 2026-09-27); the bands are chromatic since 2026-09-30 and sit in a
            # hue region viridis never reaches.
            color = cmap(0.1 + 0.8 * (i / max(len(levels) - 1, 1)))
            agg = sub[sub["level"] == level].groupby("context_t")["p"].median()
            ax.plot(agg.index, agg.values, color=color, marker="o",
                    label=snr_labels.get(float(level), f"{level:g}"))
        _reference_bands(ax, "h", label_it=False, lines_only=True)
        ax.set_xscale("log")
        # Ticks at the measured context sizes only; the default log locator crowds them together.
        ticks = sorted(sub["context_t"].unique())
        ax.set_xticks(ticks, [str(int(t)) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(style.axis_label("context_t"))
        ax.set_ylabel(style.axis_label("placement"))
        ax.set_title(metric.upper())
    axes[0][0].legend(loc="best", title=PLACEMENT_LEVEL_LEGEND_TITLE,
                      fontsize=style.FONT_SIZES["legend"],
                      title_fontsize=style.FONT_SIZES["legend"])
    written += style.save_figure(fig, run_dir, "placement_context")
    return written


def render_cka(run_dir: str) -> list[str]:
    """CKA (b): placement of the feature-token mean vs layer, and its layer x context surface.

    CKA (a) and its figure (``cka_vs_layer_<readout>``) were removed on 2026-10-21 (BETA §7): the ``representation``
    analysis replaces them (:func:`render_representation`).

    Args:
        run_dir: CKA run directory.

    Returns:
        Written paths.
    """
    path_b = os.path.join(run_dir, "cka_placement.csv")
    if not _has_rows(path_b):
        return []
    return _cka_placement_figures(pd.read_csv(path_b), run_dir)


#: Style key of each placement statistic's axis label. The CSV column name is NOT the key: the ratio's label is
#: registered as ``placement`` (it predates the column), and ``p`` alone was the permutation p-value of the retired
#: CKA (a) table, so looking the column up directly silently falls back to a title-cased "P".
PLACEMENT_AXIS_KEYS: dict[str, str] = {"p": "placement", "p_rank": "p_rank"}


def _placement_vs_layer(
    pl: pd.DataFrame,
    column: str,
    run_dir: str,
    name: str,
    *,
    dropped: list[int],
    rank_reading: bool,
) -> list[str]:
    """One placement-vs-layer figure: median over channels per layer, one trace per context size.

    Args:
        pl: Rows of ``cka_placement.csv`` already filtered to what this statistic can use.
        column: Statistic to plot (``p_rank`` or the frozen ratio ``p``).
        run_dir: Run directory.
        name: Output basename.
        dropped: Layers excluded for a degenerate floor/ceiling gap; named on the figure when non-empty.
        rank_reading: ``True`` draws the rank's between-references guide at 0.5 and clamps the axis to the
            statistic's own bounds; ``False`` draws the ratio's floor / ceiling reference bands.

    Returns:
        Written paths.
    """
    contexts = sorted(pl["context_t"].unique())
    base = style.model_color("tabpfn_v2_5")
    fig, ax = style.figure("onehalf")
    if rank_reading:
        # The rank's 0 and 1 mean "below / above every reference map", not "at the prior / noise median", so the
        # ratio's reference bands would misread it. The one meaningful landmark is 0.5 = between the two clouds.
        mid, label = style.RANK_PLACEMENT_TICKS[1][0], style.RANK_PLACEMENT_MIDLINE_LABEL
        ax.axhline(mid, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2,
                   ls=style.REFERENCE_CEILING_LINESTYLE, label=label, zorder=0)
    else:
        _reference_bands(ax, "h", label_it=True)
    for i, t in enumerate(contexts):
        # One trace per context size (2026-09-30): placement is conditioned on the evidence the embedding saw.
        color = base if len(contexts) == 1 else style.derive_shade(base, i, len(contexts))
        per_channel = (pl[pl["context_t"] == t]
                       .groupby(["subject", "emg", "layer"])[column].mean().reset_index())
        agg = per_channel.groupby("layer")[column]
        med = agg.median()
        ax.plot(med.index, med.values, color=color, marker="o", label=f"$t$ = {int(t)}")
        ax.fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values,
                        color=color, alpha=style.BAND_ALPHA, lw=0)
    if dropped:
        ax.text(0.98, 0.02, f"layers omitted (no prior/noise gap): {', '.join(str(d) for d in dropped)}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=style.FONT_SIZES["annotation"], alpha=0.8)
    ax.set_xlabel(style.axis_label("layer"))
    style.set_integer_xaxis(ax)
    ax.set_ylabel(style.axis_label(PLACEMENT_AXIS_KEYS[column]))
    if rank_reading:
        # The statistic cannot leave [0, 1], so the axis says so instead of auto-scaling to the data's span
        # and leaving a reader to wonder whether values outside it were clipped away.
        lo, hi = style.RANK_PLACEMENT_TICKS[0][0], style.RANK_PLACEMENT_TICKS[-1][0]
        ax.set_ylim(lo, hi)
    ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    return style.save_figure(fig, run_dir, name)


def _cka_placement_figures(pl: pd.DataFrame, run_dir: str) -> list[str]:
    """CKA (b): placement of the feature-token mean vs layer (one trace per context size), and its surface.

    The **reported** statistic is the rank placement ``p_rank`` (user decision 2026-10-12): it needs no
    denominator and is bounded in [0, 1] by construction, so every layer is plotted -- there is nothing to drop
    and nothing to clip. The ratio ``p`` stays on disk and is replotted beside it as the frozen reference, still
    under its own (different) axis label and still with the degenerate-gap layers dropped, so the two readings
    are never mixed on one axis.

    Args:
        pl: Rows of ``cka_placement.csv``.
        run_dir: Run directory.

    Returns:
        Written paths.
    """
    if pl.empty:
        return []
    written: list[str] = []
    # Layers whose floor/ceiling gap is degenerate are DROPPED from the RATIO panel, not clipped: p is
    # (d - floor) / (ceiling - floor), so a ~0 (or negative) denominator makes the ratio explode and inverts its
    # sign (decision 2026-09-27). The excluded layers are named on that figure instead. The rank has no
    # denominator, so this filter does not apply to it.
    usable = pl[pl["gap_ok"].astype(bool)]
    dropped = sorted(set(pl["layer"]) - set(usable["layer"]))
    if "p_rank" in pl.columns:
        ranks = pl.dropna(subset=["p_rank"])
        if not ranks.empty:
            written += _placement_vs_layer(ranks, "p_rank", run_dir, "cka_placement",
                                           dropped=[], rank_reading=True)
            written += _cka_placement_surface(ranks, [], run_dir, "p_rank",
                                              "cka_placement_surface", rank_reading=True)
    if not usable.empty:
        written += _placement_vs_layer(usable, "p", run_dir, "cka_placement_ratio",
                                       dropped=dropped, rank_reading=False)
        written += _cka_placement_surface(usable, dropped, run_dir, "p",
                                          "cka_placement_ratio_surface", rank_reading=False)
    return written


def _cka_placement_surface(
    usable: pd.DataFrame,
    dropped: list[int],
    run_dir: str,
    column: str,
    name: str,
    *,
    rank_reading: bool,
) -> list[str]:
    """Layer x context-size heatmap of the CKA placement of the feature-token mean (2026-09-30).

    The flat view of the layer / evidence / placement volume: a 3D surface of the same numbers reads worse in print
    (occlusion, one fixed viewing angle, no way to read a value off it), so the third axis is colour. The colormap is
    the package's sequential map (:data:`style.SEQUENTIAL_CMAP`, viridis) since 2026-10-07 (user): the earlier
    two-stop ramp between the crimson / pink reference tokens had too little lightness range to read values off.
    The scale is fixed to [0, 1] either way, but the two statistics' endpoints mean different things and are
    labelled accordingly: for the frozen ratio 0 is the prior floor and 1 the noise ceiling, and values outside
    the scale saturate (shown by the colorbar's extension arrows) rather than redefining it; for the rank the
    endpoints are "below / above every reference map" (:data:`style.RANK_PLACEMENT_TICKS`) and nothing can fall
    outside, so no extension arrows are drawn.

    Args:
        usable: Rows of ``cka_placement.csv`` this statistic can use (a usable floor/ceiling gap for the ratio).
        dropped: Layers excluded for a degenerate gap, named on the figure.
        run_dir: Run directory.
        column: Statistic to map (``p_rank`` or the frozen ratio ``p``).
        name: Output basename.
        rank_reading: ``True`` labels the colorbar with the rank's endpoints and drops the saturation arrows.

    Returns:
        Written paths.
    """
    grid = usable.pivot_table(index="context_t", columns="layer", values=column, aggfunc="median")
    if grid.empty or grid.shape[0] < 2:
        return []   # a single context size is the line figure, not a surface
    fig, ax = style.figure("onehalf")
    # Context sizes are rows of EQUAL height (the ladder is not evenly spaced, so a numeric axis would let the
    # widest gap dominate the picture); cells without a usable gap at that (layer, t) stay blank.
    rows = np.arange(len(grid.index))                                        # [T]
    ticks = style.RANK_PLACEMENT_TICKS if rank_reading else style.RATIO_PLACEMENT_TICKS
    im = ax.pcolormesh(grid.columns.to_numpy(), rows, grid.to_numpy(),
                       cmap=style.SEQUENTIAL_CMAP, vmin=ticks[0][0], vmax=ticks[-1][0], shading="nearest")
    ax.set_xlabel(style.axis_label("layer"))
    style.set_integer_xaxis(ax)
    ax.set_ylabel(style.axis_label("context_t"))
    ax.set_yticks(rows, [str(int(t)) for t in grid.index])
    cb = fig.colorbar(im, ax=ax, label=style.axis_label(PLACEMENT_AXIS_KEYS[column]),
                      extend="neither" if rank_reading else "both")
    cb.set_ticks([v for v, _ in ticks], labels=[lab for _, lab in ticks])
    notes = [f"layers omitted: {', '.join(str(d) for d in dropped)}"] if dropped else []
    if grid.isna().to_numpy().any():
        notes.append("blank: no usable prior/noise gap")
    if notes:
        ax.text(0.98, 1.02, "; ".join(notes), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=style.FONT_SIZES["annotation"], alpha=0.8)
    return style.save_figure(fig, run_dir, name)


# ---------------------------------------------------------------------------
# representation (OBJECTIVE BETA, 2026-10-21): β2 alignment vs block, β1a site matching, lens, β1b covariances
# ---------------------------------------------------------------------------
#: The primary β2 panel grid (D3c / D3e): rows = location / spread, columns = TabPFN's own belief / GP-MLL refit's.
#: The two columns sit side by side and are never averaged or differenced (G6).
REPRESENTATION_PRIMARY_GRID: tuple[tuple[str, str], ...] = (("mean_own", "mean_mll"), ("sd_own", "sd_mll"))

#: Panel height / width of the representation figures (wide grids of short panels).
_REPRESENTATION_PANEL_ASPECT: float = 0.7

#: Columns of the secondary β2 grid.
_REPRESENTATION_SECONDARY_COLUMNS: int = 3

#: Panel height / width of the one-column site-match figure.
_SITE_MATCH_PANEL_ASPECT: float = 0.55

#: Statistics of the β1a site-match figure (rows), with their axis-label keys.
SITE_MATCH_STATS: dict[str, str] = {"w2_over_sd": "w2_over_sd", "location": "w2_location", "spread": "w2_spread",
                                    "shape": "w2_shape"}

#: Statistics of the β1b posterior-covariance figure (columns), with the value of a perfect match.
POSTERIOR_COV_STATS: dict[str, float] = {"nbs": 1.0, "cka": 1.0, "bures_rel": 0.0, "trace_ratio": 1.0}


def _figures_block(run_dir: str) -> dict[str, Any]:
    """The ``representation.figures`` block of the run's resolved config (empty when absent)."""
    import yaml  # noqa: PLC0415

    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return ((yaml.safe_load(fh) or {}).get("representation") or {}).get("figures") or {}


def _figure_level(df: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    """Rows at ``representation.figures.level`` of the run's config (null = the nominal, lowest level)."""
    level = _figures_block(run_dir).get("level")
    if level is None:
        return _nominal(df)
    return df[np.isclose(df["level"], float(level))]


def _drop_excluded_ts(df: pd.DataFrame, run_dir: str) -> pd.DataFrame:
    """Rows without the context sizes of ``representation.figures.exclude_ts`` (site match and lens only)."""
    drop = [int(t) for t in (_figures_block(run_dir).get("exclude_ts") or [])]
    return df[~df["context_t"].isin(drop)]


def _t_color(i: int, n: int) -> str:
    """Shade of the TabPFN anchor for the i-th of n context sizes (one trace per t, as in every Hyp C figure)."""
    base = style.model_color("tabpfn_v2_5")
    return base if n == 1 else style.derive_shade(base, i, n)


#: The β2 figure without geometry regression: (value rows, floor rows, y-axis key, figure-name suffix). It sits
#: beside the headline (deconfounded) figure so the share that geometry explains is visible (user, 2026-10-22).
_RAW_ALIGNMENT: tuple[str, str, str, str] = ("cka", "surrogate_q_cka", "cka_debiased", "_raw")


def _alignment_panel(ax: Any, sub: pd.DataFrame, readouts: Sequence[str], ts: Sequence[int], *, value: str,
                     floor: str, legend: bool) -> None:
    """One target: a statistic vs block, one trace per (readout, t), its surrogate floor as a band.

    Args:
        ax: Target axes.
        sub: Summary rows of one target.
        readouts: Readouts drawn (the first solid with its interval, the rest thin).
        ts: Context sizes, one trace each.
        value: ``value`` rows drawn as lines.
        floor: ``value`` rows drawn as the chance band (min to max over t of the median, first readout).
        legend: Attach legend labels.
    """
    floors = sub[(sub["value"] == floor) & (sub["readout"] == readouts[0])]
    if not floors.empty:
        band = floors.groupby("layer")["median"]
        ax.fill_between(band.min().index, band.min().values, band.max().values, color=style.NEUTRAL_GREY,
                        alpha=style.BAND_ALPHA, lw=0, label=style.SURROGATE_BAND_LABEL if legend else None, zorder=0)
    vals = sub[sub["value"] == value]
    for readout in readouts:
        for i, t in enumerate(ts):
            part = vals[(vals["readout"] == readout) & (vals["context_t"] == t)].sort_values("layer")
            if part.empty:
                continue
            color = _t_color(i, len(ts))
            label = f"$t$ = {int(t)}" if (legend and readout == readouts[0]) else None
            ax.plot(part["layer"], part["median"], color=color, ls=style.READOUT_LINESTYLES.get(readout, "-"),
                    lw=style.LINE_WIDTH if readout == readouts[0] else style.LINE_WIDTH / 2, label=label)
            if readout == readouts[0]:
                ax.fill_between(part["layer"], part["ci_low"], part["ci_high"], color=color, alpha=style.BAND_ALPHA,
                                lw=0)
    ax.axhline(0.0, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, zorder=0)
    style.set_integer_xaxis(ax)


def _representation_grid(summary: pd.DataFrame, grid: Sequence[Sequence[str | None]], readouts: Sequence[str],
                         value: str, floor: str, ylabel: str, run_dir: str, name: str) -> list[str]:
    """A grid of alignment panels (one target each), shared y, blocks on x."""
    nrows, ncols = len(grid), max(len(r) for r in grid)
    ts = sorted(summary["context_t"].unique())
    fig, axes = style.figure("wide", nrows=nrows, ncols=ncols, squeeze=False, sharex=True, sharey=True,
                             panel_aspect=_REPRESENTATION_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
    first = True
    for r, row in enumerate(grid):
        for c in range(ncols):
            ax = axes[r][c]
            target = row[c] if c < len(row) else None
            if target is None:
                ax.set_visible(False)
                continue
            _alignment_panel(ax, summary[summary["target"] == target], readouts, ts, value=value, floor=floor,
                             legend=first)
            first = False
            ax.set_title(style.BELIEF_TARGET_LABELS.get(target, target), fontsize=style.FONT_SIZES["base"])
            if r == nrows - 1:
                ax.set_xlabel(style.axis_label("layer"))
    fig.supylabel(style.axis_label(ylabel), fontsize=style.FONT_SIZES["base"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    for readout in readouts:
        handles.append(plt.Line2D([], [], color=style.NEUTRAL_GREY, ls=style.READOUT_LINESTYLES.get(readout, "-")))
        labels.append(style.READOUT_LABELS.get(readout, readout))
    fig.legend(handles, labels, loc="outside upper center", ncol=min(len(labels), 4),
               fontsize=style.FONT_SIZES["legend"], frameon=False)
    style.panel_letters([ax for row in axes for ax in row if ax.get_visible()])
    return style.save_figure(fig, run_dir, name)


def _representation_vs_layer(summary: pd.DataFrame, statistic: str, headline_readout: str,
                             run_dir: str) -> list[str]:
    """β2: the primary 2 x 2 grid (location / spread x own / GP-MLL refit) and the secondary targets.

    Each is drawn twice: the headline statistic (geometry regressed out) and raw debiased CKA (``*_raw``), whose
    chance band is the raw surrogate quantile.

    Args:
        summary: ``representation_summary.csv`` rows at the figure level and site set.
        statistic: Headline statistic (``headline.json``).
        headline_readout: Readout drawn solid with its interval.
        run_dir: Output directory.

    Returns:
        Written paths.
    """
    present = set(summary["target"])
    readouts = [headline_readout] + [r for r in ("feature_mean",) if r in set(summary["readout"])
                                     and r != headline_readout]
    primary = [[t if t in present else None for t in row] for row in REPRESENTATION_PRIMARY_GRID]
    rest = [t for t in style.BELIEF_TARGET_LABELS if t in present
            and t not in {x for row in REPRESENTATION_PRIMARY_GRID for x in row}]
    n = _REPRESENTATION_SECONDARY_COLUMNS
    secondary = [rest[i:i + n] for i in range(0, len(rest), n)]
    raw_value, raw_floor, raw_label, raw_suffix = _RAW_ALIGNMENT
    variants = [("headline_value", "headline_floor", f"headline_{statistic}", "")]
    if raw_value in set(summary["value"]):
        variants.append((raw_value, raw_floor, raw_label, raw_suffix))
    written: list[str] = []
    for value, floor, ylabel, suffix in variants:
        if any(t is not None for row in primary for t in row):
            written += _representation_grid(summary, primary, readouts, value, floor, ylabel, run_dir,
                                            f"representation_vs_layer{suffix}")
        if secondary:
            written += _representation_grid(summary, secondary, readouts, value, floor, ylabel, run_dir,
                                            f"representation_vs_layer_secondary{suffix}")
    return written


def _site_match_figure(sm: pd.DataFrame, run_dir: str) -> list[str]:
    """β1a: per-site W2 to each reference and its location / spread / shape split vs t, every site pooled.

    One column: the references share each panel, told apart by colour and line style (user, 2026-10-22). The
    observed / unobserved split stays in ``site_match_summary.csv``.

    Args:
        sm: ``site_match_summary.csv`` rows at the figure level, excluded context sizes removed.
        run_dir: Output directory.

    Returns:
        Written paths.
    """
    sm = sm[sm["site_pool"] == SITE_POOL_ALL] if SITE_POOL_ALL in set(sm["site_pool"]) else sm
    refs = list(dict.fromkeys(sm["reference"]))
    stats = [s for s in SITE_MATCH_STATS if s in set(sm["value"])]
    if not refs or not stats:
        return []
    fig, axes = style.figure("single", nrows=len(stats), ncols=1, squeeze=False, sharex=True,
                             panel_aspect=_SITE_MATCH_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
    ts = sorted(sm["context_t"].unique())
    for r, stat in enumerate(stats):
        ax = axes[r][0]
        for ref in refs:
            st = arm_style(ref)
            part = sm[(sm["reference"] == ref) & (sm["value"] == stat)].sort_values("context_t")
            ax.plot(part["context_t"], part["median"], color=st.color, ls=st.linestyle, marker=st.marker,
                    label=f"vs {st.label}" if r == 0 else None)
            ax.fill_between(part["context_t"], part["ci_low"], part["ci_high"], color=st.color,
                            alpha=style.BAND_ALPHA, lw=0)
        ax.set_xticks(list(ts), [str(int(t)) for t in ts])
        ax.set_ylabel(style.axis_label(SITE_MATCH_STATS[stat]))
    axes[-1][0].set_xlabel(style.axis_label("context_t"))
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    style.panel_letters([ax for row in axes for ax in row])
    return style.save_figure(fig, run_dir, "site_match")


def _lens_figure(lens: pd.DataFrame, run_dir: str) -> list[str]:
    """Lens predictive of every block against TabPFN's final one and each reference (W2 / reference SD), per t."""
    part = lens[(lens["value"] == "w2_over_sd") & (lens["site_pool"] == "unobserved")]
    comps = list(dict.fromkeys(part["comparison"]))
    if not comps:
        return []
    ts = sorted(part["context_t"].unique())
    fig, axes = style.figure("wide", ncols=len(comps), squeeze=False, sharey=True,
                             panel_aspect=_REPRESENTATION_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
    for c, comp in enumerate(comps):
        ax = axes[0][c]
        for i, t in enumerate(ts):
            sub = part[(part["comparison"] == comp) & (part["context_t"] == t)].sort_values("layer")
            color = _t_color(i, len(ts))
            ax.plot(sub["layer"], sub["median"], color=color, marker="o", label=f"$t$ = {int(t)}" if c == 0 else None)
            ax.fill_between(sub["layer"], sub["ci_low"], sub["ci_high"], color=color, alpha=style.BAND_ALPHA, lw=0)
        title = "vs final TabPFN-2.5" if comp == "tabpfn_v2_5" else f"vs {arm_style(comp).label}"
        ax.set_title(title, fontsize=style.FONT_SIZES["base"])
        ax.set_xlabel(style.axis_label("layer"))
        style.set_integer_xaxis(ax)
    axes[0][0].set_ylabel(style.axis_label("lens_w2_over_sd"))
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    style.panel_letters(list(axes[0]))
    return style.save_figure(fig, run_dir, "lens_distance")


def _posterior_cov_figure(pc: pd.DataFrame, run_dir: str) -> list[str]:
    """β1b: NBS, CKA, relative Bures distance and trace ratio of C_PFN vs each reference's covariance, vs t."""
    stats = [s for s in POSTERIOR_COV_STATS if s in set(pc["value"])]
    if not stats:
        return []
    ts = sorted(pc["context_t"].unique())
    fig, axes = style.figure("wide", ncols=len(stats), squeeze=False, panel_aspect=_REPRESENTATION_PANEL_ASPECT,
                             layout=style.LAYOUT_ENGINE)
    for c, stat in enumerate(stats):
        ax = axes[0][c]
        for ref in dict.fromkeys(pc["reference"]):
            st = arm_style(ref)
            sub = pc[(pc["reference"] == ref) & (pc["value"] == stat)].sort_values("context_t")
            ax.plot(sub["context_t"], sub["median"], color=st.color, ls=st.linestyle, marker=st.marker,
                    label=st.label if c == 0 else None)
            ax.fill_between(sub["context_t"], sub["ci_low"], sub["ci_high"], color=st.color, alpha=style.BAND_ALPHA,
                            lw=0)
        _bayes_value(ax, POSTERIOR_COV_STATS[stat])
        _context_axis(ax, ts)
        ax.set_ylabel(style.axis_label(stat if stat != "cka" else "cka_debiased"))
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    style.panel_letters(list(axes[0]))
    return style.save_figure(fig, run_dir, "posterior_cov")


def render_representation(run_dir: str) -> list[str]:
    """Every BETA figure, from the run's summary CSVs alone (``--replot``).

    ``representation_vs_layer.svg`` (β2 primary: location / spread x own / GP-MLL refit, label token solid, feature
    mean dotted, surrogate floor as a grey band), ``representation_vs_layer_secondary.svg`` (covariance, quantile
    embeddings, K_GT, K_X), ``site_match.svg`` (β1a), ``lens_distance.svg`` and ``posterior_cov.svg`` (β1b). The
    level shown is ``representation.figures.level`` (null = nominal). Captions follow G6 / G7: decodability wording,
    no "GP-like" from a gap between two scores, no bare "Bayesian".

    Args:
        run_dir: Representation run directory.

    Returns:
        Written paths.
    """
    import json  # noqa: PLC0415

    written: list[str] = []
    path = os.path.join(run_dir, "representation_summary.csv")
    if _has_rows(path):
        summary = _figure_level(pd.read_csv(path), run_dir)
        summary = summary[summary["site_set"] == "all"]
        statistic, readout = "deconfounded_vs_surrogate", "label"
        head = os.path.join(run_dir, "headline.json")
        if os.path.exists(head):
            with open(head, encoding="utf-8") as fh:
                doc = json.load(fh)
            statistic, readout = doc.get("statistic", statistic), doc.get("readout", readout)
        if not summary.empty:
            written += _representation_vs_layer(summary, statistic, readout, run_dir)
    for name, fn in (("site_match_summary.csv", _site_match_figure), ("lens_summary.csv", _lens_figure),
                     ("posterior_cov_summary.csv", _posterior_cov_figure)):
        p = os.path.join(run_dir, name)
        if _has_rows(p):
            df = _figure_level(pd.read_csv(p), run_dir)
            if fn is not _posterior_cov_figure:
                df = _drop_excluded_ts(df, run_dir)
            if not df.empty:
                written += fn(df, run_dir)
    return written
