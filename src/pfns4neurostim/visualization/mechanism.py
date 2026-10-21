"""Hyp C mechanism figures: placement (MMD / sliced W2), CKA, Procrustes.

The legacy ID/OOD dashboard that used to live under this name (its own hardcoded palette)
is in :mod:`pfns4neurostim.legacy_code.visualization_id_ood`; everything here takes geometry,
colours, labels and axis strings from :mod:`visualization.style` (CLAUDE.md §7).
Each ``render_*`` function reads only the CSVs of a run directory, so ``--replot`` rebuilds
every figure without re-running an analysis.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style

__all__ = [
    "arm_style",
    "render_bayes_update",
    "render_predictive_link",
    "render_placement",
    "render_cka",
    "render_procrustes",
]

#: Legend title over the stress-level series of the placement trajectory, so the lines are not an
#: unlabelled ladder of colours. The series were labelled by the raw K2 level (``1 / 2 / 4``) until
#: 2026-10-12; they are now the **absolute achieved SNR** of the level, the same quantity and the same
#: numbers as the x-axis of every Hyp B stress figure, so a reader can place a mechanism panel against a
#: degradation curve (user decision 2026-10-12).
PLACEMENT_LEVEL_LEGEND_TITLE: str = "Achieved SNR (dB)"

#: Half-width of the floor / ceiling reference bands, in placement units.
REFERENCE_BAND_HALF_WIDTH: float = 0.05


#: CKA (a) target kernels that get a panel, in display order. ``K_X`` (electrode geometry) and
#: ``K_GT_linear`` were dropped from the figure on 2026-09-27: ``K_X`` is an explicit *control* -- the model's
#: input IS the coordinates, so agreement with geometry is trivial and it read highest, which invited the
#: wrong conclusion -- and ``K_GT_linear`` is a bandwidth *sensitivity* check on ``K_GT``. Both stay in
#: ``cka.csv`` and in the config's ``targets``, so the control and the sensitivity check remain auditable;
#: they are simply not headline panels. The targets of interest are the response and the GP posterior.
CKA_PLOTTED_TARGETS: tuple[str, ...] = ("K_GT", "K_GP")


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
    return style.model_style(_ENGINE_MODEL.get(engine, engine))


# ---------------------------------------------------------------------------
# bayes_update (OBJECTIVE OMEGA, 2026-10-15): Omega-1 ... Omega-5
# ---------------------------------------------------------------------------
#: Part A headline metrics of Omega-1, one column each, with the value every coherent Bayesian model scores.
OMEGA_COHERENCE_METRICS: dict[str, float] = {"drift_abs": 0.0, "total_var_ratio": 1.0, "cov_asymmetry": 0.0,
                                            "latent_frac": float("nan")}
#: Part B headline contrasts of Omega-3, with the value of a perfect match to the reference.
OMEGA_CONTRAST_METRICS: dict[str, float] = {"transfer_slope": 1.0, "footprint_r": 1.0, "gap_change": 0.0,
                                           "delta_crps": 0.0}
#: Panel height / width of the Omega line grids, and of the exemplar maps.
_OMEGA_PANEL_ASPECT: float = 0.8
_OMEGA_MAP_ASPECT: float = 0.7


def _nominal(df: pd.DataFrame) -> pd.DataFrame:
    """Rows at the nominal (lowest) stress level -- the headline operating point."""
    return df[np.isclose(df["level"], df["level"].min())]


def _median_band(ax: Any, df: pd.DataFrame, col: str, st: style.ModelStyle) -> None:
    """Median over cells vs context size, with an IQR band, in one engine's style."""
    g = df.dropna(subset=[col]).groupby("context_t")[col]
    if g.ngroups == 0:
        return
    med = g.median()
    ax.plot(med.index, med.values, color=st.color, ls=st.linestyle, marker=st.marker, label=st.label,
            zorder=st.zorder)
    ax.fill_between(med.index, g.quantile(0.25).values, g.quantile(0.75).values, color=st.color,
                    alpha=style.BAND_ALPHA, lw=0)


def _bayes_value(ax: Any, value: float) -> None:
    """The value every coherent Bayesian model scores, as a thin neutral line."""
    if np.isfinite(value):
        ax.axhline(value, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":", zorder=0)


def _context_axis(ax: Any, ts: Sequence[int]) -> None:
    """Context-size axis with ticks at the measured sizes only."""
    ax.set_xticks(list(ts), [str(int(t)) for t in ts])
    ax.set_xlabel(style.axis_label("context_t"))


def _omega_coherence(cell: pd.DataFrame, run_dir: str) -> list[str]:
    """Omega-1: Part A headlines vs context size; rows = anchor pool (never pooled)."""
    cell = _nominal(cell)
    pools = [p for p in ("unobserved", "observed") if p in set(cell["pool"])]
    metrics = [m for m in OMEGA_COHERENCE_METRICS if m in cell]
    ts = sorted(cell["context_t"].unique())
    fig, axes = style.figure("wide", nrows=len(pools), ncols=len(metrics), squeeze=False,
                             panel_aspect=_OMEGA_PANEL_ASPECT)
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
    return style.save_figure(fig, run_dir, "coherence")


def _omega_implied_noise(cell: pd.DataFrame, coh: pd.DataFrame, run_dir: str) -> list[str]:
    """Omega-2: latent share of the predictive variance vs t; implied noise vs the trials' own variance."""
    cell, coh = _nominal(cell), _nominal(coh)
    fig, axes = style.figure("double", ncols=2, squeeze=False, panel_aspect=_OMEGA_PANEL_ASPECT)
    ax = axes[0][0]
    for engine in cell["engine"].unique():
        _median_band(ax, cell[(cell["engine"] == engine) & (cell["pool"] == "observed")], "latent_frac",
                     arm_style(engine))
    _context_axis(ax, sorted(cell["context_t"].unique()))
    ax.set_ylabel(style.axis_label("latent_frac"))
    ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    ax = axes[0][1]
    for engine in coh["engine"].unique():
        st = arm_style(engine)
        sub = coh[coh["engine"] == engine]
        rho = cell[cell["engine"] == engine]["sigma_recovery_rho"].median()
        ax.scatter(sub["trial_var"], sub["implied_noise"], s=style.MARKER_SIZE, color=st.color, marker=st.marker,
                   alpha=style.FAINT_ALPHA * 3, label=f"{st.label} (median rho {rho:.2f})", lw=0)
    lim = [0.0, float(np.nanmax(coh[["trial_var", "implied_noise"]].to_numpy()))]
    ax.plot(lim, lim, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":")
    ax.set_xlabel(style.axis_label("trial_var"))
    ax.set_ylabel(style.axis_label("implied_noise"))
    ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    style.panel_letters(axes.ravel())
    return style.save_figure(fig, run_dir, "implied_noise")


def _omega_headtohead(h2h: pd.DataFrame, run_dir: str) -> list[str]:
    """Omega-3: site-matched contrasts with each reference vs t (unobserved anchors, trial source).

    Every other engine is drawn against each reference, so the GP-vs-GP spread sits next to TabPFN's.
    """
    sub = _nominal(h2h)
    sub = sub[(sub["reference"] != "none") & (sub["source"] == "trial") & (sub["pool"] == "unobserved")]
    refs = [r for r in ("gp_bayes", "gp_mll_refit", "gp_fixed") if r in set(sub["reference"])]
    metrics = [m for m in OMEGA_CONTRAST_METRICS if m in sub]
    if not refs or not metrics:
        return []
    ts = sorted(sub["context_t"].unique())
    keys = ["subject", "emg", "level", "context_t", "draw", "engine", "reference"]
    per_cell = sub.groupby(keys)[metrics].median().reset_index()                  # anchors -> cell
    fig, axes = style.figure("wide", nrows=len(refs), ncols=len(metrics), squeeze=False,
                             panel_aspect=_OMEGA_PANEL_ASPECT)
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
                    engines: Sequence[str] | None = None, name: str = "surprise_response") -> list[str]:
    """Omega-4 (and C1): update at the anchor vs injected surprise (GP-Bayes SD units), one panel per context size.

    Args:
        surprise: ``surprise.csv`` rows.
        panel_ts: Context sizes drawn, one panel each.
        run_dir: Output directory.
        engines: Engines drawn (``None`` = every engine of the table).
        name: Figure name.

    Returns:
        Written paths.
    """
    sub = _nominal(surprise)
    sub = sub[sub["pool"] == "unobserved"]
    if engines is not None:
        sub = sub[sub["engine"].isin(list(engines))]
    ts = [t for t in panel_ts if t in set(sub["context_t"])]
    if not ts:
        return []
    fig, axes = style.figure("double", ncols=len(ts), squeeze=False, panel_aspect=_OMEGA_PANEL_ASPECT)
    for ax, t in zip(axes[0], ts):
        at = sub[sub["context_t"] == t]
        for engine in at["engine"].unique():
            st = arm_style(engine)
            g = at[at["engine"] == engine].groupby("c")["update_anchor_sd"]
            ax.plot(g.median().index, g.median().values, color=st.color, ls=st.linestyle, marker=st.marker,
                    label=st.label, zorder=st.zorder)
            ax.fill_between(g.median().index, g.quantile(0.25).values, g.quantile(0.75).values, color=st.color,
                            alpha=style.BAND_ALPHA, lw=0)
        ax.axhline(0.0, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":", zorder=0)
        ax.set_xlabel(style.axis_label("ladder_c"))
        ax.set_ylabel(style.axis_label("update_anchor_sd"))
        ax.set_title(f"$t$ = {int(t)}", fontsize=style.FONT_SIZES["legend"])
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    style.panel_letters(axes.ravel())
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
    fig, axes = style.figure("full", nrows=len(engines), ncols=len(ts), squeeze=False, panel_aspect=_OMEGA_MAP_ASPECT)
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
    fig, axes = style.figure("wide", nrows=nrows, ncols=ncols, squeeze=False, panel_aspect=_OMEGA_PANEL_ASPECT)
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
    sub = _nominal(profile)
    sub = sub[(sub["pool"] == "unobserved") & (sub["site_pool"] == "unobserved") & sub["engine"].isin(list(engines))]
    ts = [t for t in panel_ts if t in set(sub["context_t"])]
    if not ts:
        return []
    sub = sub.assign(bin_mid=0.5 * (sub["bin_lo"] + sub["bin_hi"]))
    keys = ["subject", "emg", "level", "context_t", "draw", "engine", "bin_mid"]
    per_cell = sub.groupby(keys)["abs_transfer"].median().reset_index()
    fig, axes = style.figure("double", ncols=len(ts), squeeze=False, panel_aspect=_OMEGA_PANEL_ASPECT)
    for ax, t in zip(axes[0], ts):
        at = per_cell[per_cell["context_t"] == t]
        for engine in engines:
            st = arm_style(engine)
            g = at[at["engine"] == engine].groupby("bin_mid")["abs_transfer"]
            if g.ngroups == 0:
                continue
            ax.plot(g.median().index, g.median().values, color=st.color, ls=st.linestyle, marker=st.marker,
                    label=st.label, zorder=st.zorder)
            ax.fill_between(g.median().index, g.quantile(0.25).values, g.quantile(0.75).values, color=st.color,
                            alpha=style.BAND_ALPHA, lw=0)
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
        written += _omega_coherence(cell, run_dir)
        if _has_rows(paths["coherence"]):
            written += _omega_implied_noise(cell, pd.read_csv(paths["coherence"]), run_dir)
    if _has_rows(paths["headtohead"]):
        written += _omega_headtohead(pd.read_csv(paths["headtohead"]), run_dir)
    cfg_path = os.path.join(run_dir, "config.yaml")
    cfg_block: dict[str, Any] = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            cfg_block = yaml.safe_load(fh).get("bayes_update") or {}
    panel_ts: Sequence[int] = cfg_block.get("panel_ts") or ()
    if _has_rows(paths["surprise"]):
        written += _omega_surprise(pd.read_csv(paths["surprise"]), panel_ts, run_dir)
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
                                       name="c1_surprise_response")
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


#: Height / width of a CKA (a) panel: one row per context size stacks four or more rows, so panels are kept flat.
_CKA_PANEL_ASPECT: float = 0.45


def _cka_a_figure(df: pd.DataFrame, readout: str, targets: list[str], run_dir: str) -> list[str]:
    """CKA (a) of one feature readout: rows = context sizes, columns = targets, colour = stress level."""
    levels = sorted(df["level"].unique())
    snr_labels = _level_snr_labels(df)
    # ONE ROW PER CONTEXT SIZE (2026-09-30): the figure's subject is what the network does with its evidence.
    contexts = sorted(df["context_t"].unique())
    fig, axes = style.figure("wide", nrows=len(contexts), ncols=len(targets), panel_aspect=_CKA_PANEL_ASPECT,
                             sharey=True, sharex=True, squeeze=False)
    base = style.model_color("tabpfn_v2_5")
    for r, t_ctx in enumerate(contexts):
        for c, target in enumerate(targets):
            ax = axes[r, c]
            sub = df[(df["target"] == target) & (df["context_t"] == t_ctx)]
            for i, level in enumerate(levels):
                color = base if i == 0 else style.derive_shade(base, i - 1, max(len(levels) - 1, 1))
                # Average the draws WITHIN a channel first, then spread over channels: the band is then
                # between-channel heterogeneity (a finding) and not draw noise (an artefact).
                per_channel = (sub[sub["level"] == level]
                               .groupby(["subject", "emg", "layer"])["cka_minus_null"].mean().reset_index())
                agg = per_channel.groupby("layer")["cka_minus_null"]
                med, q1, q3 = agg.median(), agg.quantile(0.25), agg.quantile(0.75)
                ax.plot(med.index, med.values, color=color, ls=style.READOUT_LINESTYLES.get(readout, "-"),
                        label=snr_labels.get(float(level), f"{level:g}"))
                ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=style.BAND_ALPHA, lw=0)
            ax.axhline(0.0, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2)
            if r == 0:
                ax.set_title(target.replace("_", " "))
            style.set_integer_xaxis(ax)
            if r == len(contexts) - 1:
                ax.set_xlabel(style.axis_label("layer"))
            if c == 0:
                ax.set_ylabel(f"$t$ = {int(t_ctx)}")
    fig.supylabel(style.axis_label("cka_minus_null"), fontsize=style.FONT_SIZES["base"])
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"], title=style.READOUT_LABELS.get(readout, readout),
                      title_fontsize=style.FONT_SIZES["legend"])
    return style.save_figure(fig, run_dir, f"cka_vs_layer_{readout}")


def render_cka(run_dir: str) -> list[str]:
    """CKA (a) minus null vs layer per feature readout and target, and CKA (b) placement vs layer.

    Only feature-token readouts exist (the label token was dropped on 2026-10-07, see
    :mod:`pfns4neurostim.analysis.embeddings`), so there is no readout-comparison figure: ``feature_tokens`` is the
    shadow run's readout and its verdict is ``shadow_criterion.json``.

    Args:
        run_dir: CKA run directory.

    Returns:
        Written paths.
    """
    written: list[str] = []
    path_a = os.path.join(run_dir, "cka.csv")
    if _has_rows(path_a):
        df = pd.read_csv(path_a)
        targets = [t for t in CKA_PLOTTED_TARGETS if t in set(df["target"])]
        for readout in [r for r in style.READOUT_LINESTYLES if r in set(df["readout"])]:
            written += _cka_a_figure(df[df["readout"] == readout], readout, targets, run_dir)
    path_b = os.path.join(run_dir, "cka_placement.csv")
    if _has_rows(path_b):
        written += _cka_placement_figures(pd.read_csv(path_b), run_dir)
    return written


#: Style key of each placement statistic's axis label. The CSV column name is NOT the key: the ratio's label is
#: registered as ``placement`` (it predates the column), and ``p`` alone is the permutation p-value in ``cka.csv``,
#: so looking the column up directly silently falls back to a title-cased "P".
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


# ---------------------------------------------------------------------------
# procrustes (M14 revived, 2026-10-14): the spectrum-weighting complement of CKA
# ---------------------------------------------------------------------------
#: Linestyle of the shuffled-site null reference in every Procrustes figure. The null is drawn in
#: :data:`style.NEUTRAL_GREY` -- the token for a non-learning baseline, which is exactly what a row-permutation
#: reference is -- and never in a model anchor or in the prior/noise reference colours (those mean something else
#: in this project's figures).
PROCRUSTES_NULL_LINESTYLE: tuple[int, tuple[int, int]] = (0, (3, 2))

#: Aggregation of a Procrustes panel: the context draws of a channel are averaged FIRST and the spread is taken
#: over channels, so the band is between-channel heterogeneity (a finding) and not draw noise (an artefact).
PROCRUSTES_CHANNEL_KEYS: tuple[str, ...] = ("subject", "emg")


def _procrustes_context_layers(run_dir: str, available: Sequence[int]) -> list[int]:
    """Layers to draw in the disparity-vs-context figure, from the run's own resolved config.

    Args:
        run_dir: Procrustes run directory (holds ``config.yaml``).
        available: Layers present in ``procrustes.csv``.

    Returns:
        The configured subset, or every available layer when the config does not pin one. Honours ``--replot``:
        the resolved config written by the run is part of the run directory, so nothing is recomputed.
    """
    path = os.path.join(run_dir, "config.yaml")
    wanted: Sequence[int] | None = None
    if os.path.exists(path):
        import yaml  # noqa: PLC0415 - only needed when a config is present

        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        wanted = ((doc.get("procrustes") or {}).get("figures") or {}).get("context_layers")
    if not wanted:
        return sorted(int(v) for v in available)
    return sorted(int(v) for v in wanted if int(v) in set(int(a) for a in available))


def _procrustes_median(df: pd.DataFrame, column: str, by: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Median and interquartile band of ``column`` over channels, after averaging each channel's draws.

    Args:
        df: Rows of ``procrustes.csv`` (already filtered to one series).
        column: Value column.
        by: Grouping axis (``'layer'`` or ``'context_t'``).

    Returns:
        ``(median, q1, q3)``, each indexed by ``by``.
    """
    per_channel = df.groupby([*PROCRUSTES_CHANNEL_KEYS, by])[column].mean().reset_index()
    agg = per_channel.groupby(by)[column]
    return agg.median(), agg.quantile(0.25), agg.quantile(0.75)


def _procrustes_vs_context(df: pd.DataFrame, run_dir: str) -> list[str]:
    """Disparity along the context ladder, per layer, with the shuffled-site null drawn as its reference.

    The two axes of this figure answer different questions and must not be conflated: the **ladder** asks whether
    the geometric drift depends on how much evidence the network has, while the **null** asks whether a disparity
    number means anything at all at 96 rows in 192 dimensions. The null is a property of the clouds' marginal
    geometry, not of the context size.
    """
    layers = _procrustes_context_layers(run_dir, sorted(df["layer"].unique()))
    if not layers:
        return []
    base = style.model_color("tabpfn_v2_5")
    fig, ax = style.figure("onehalf")
    for i, layer in enumerate(layers):
        sub = df[df["layer"] == layer]
        color = base if len(layers) == 1 else style.derive_shade(base, i, len(layers))
        med, q1, q3 = _procrustes_median(sub, "disparity", "context_t")
        ax.plot(med.index, med.values, color=color, marker="o", label=rf"$\ell$ = {int(layer)}")
        ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=style.BAND_ALPHA, lw=0)
        null_med, _, _ = _procrustes_median(sub, "null_mean", "context_t")
        ax.plot(null_med.index, null_med.values, color=style.NEUTRAL_GREY, ls=PROCRUSTES_NULL_LINESTYLE,
                lw=style.LINE_WIDTH, label="Shuffled-site null" if i == 0 else None)
    ax.set_xlabel(style.axis_label("context_t"))
    ax.set_ylabel(style.axis_label("procrustes_disparity"))
    ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"], ncol=2)
    return style.save_figure(fig, run_dir, "procrustes_vs_context")


def _procrustes_vs_layer(df: pd.DataFrame, run_dir: str) -> list[str]:
    """The headline statistic (disparity minus the shuffled-site null) over layers, one trace per context size.

    The baseline context size is dropped: it is each cloud compared with itself, so its disparity is 0 by
    construction and the statistic degenerates to minus the null.
    """
    moved = df[~df["is_baseline"].astype(bool)]
    if moved.empty:
        return []
    contexts = sorted(moved["context_t"].unique())
    base = style.model_color("tabpfn_v2_5")
    fig, ax = style.figure("onehalf")
    for i, t in enumerate(contexts):
        color = base if len(contexts) == 1 else style.derive_shade(base, i, len(contexts))
        med, q1, q3 = _procrustes_median(moved[moved["context_t"] == t], "disparity_minus_null", "layer")
        ax.plot(med.index, med.values, color=color, marker="o", label=f"$t$ = {int(t)}")
        ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=style.BAND_ALPHA, lw=0)
    ax.axhline(0.0, color=style.NEUTRAL_GREY, ls=PROCRUSTES_NULL_LINESTYLE, lw=style.LINE_WIDTH,
               label="Shuffled-site null")
    ax.set_xlabel(style.axis_label("layer"))
    style.set_integer_xaxis(ax)
    ax.set_ylabel(style.axis_label("disparity_minus_null"))
    ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"], ncol=2)
    baseline = int(df["baseline_t"].iloc[0])
    ax.set_title(rf"vs the $t$ = {baseline} cloud", fontsize=style.FONT_SIZES["base"])
    return style.save_figure(fig, run_dir, "procrustes_vs_layer")


def _procrustes_vs_cka(df: pd.DataFrame, run_dir: str) -> list[str]:
    """Procrustes and CKA over layers **on the same pairs of clouds** -- the comparison this analysis exists for.

    Top: the disparity with its shuffled-site null. Bottom: debiased linear CKA of the same two clouds. The two
    statistics share an invariance group (translation, isotropic scale, orthogonal maps) but weight the
    cross-covariance spectrum differently -- CKA by a Frobenius norm, so the leading directions dominate, the
    disparity by a nuclear norm, so every direction counts equally (see
    :mod:`pfns4neurostim.analysis.procrustes`). A layer where the two curves disagree is a layer whose change
    sits in the weakly-populated directions CKA discounts. Neither panel speaks to the model's prediction:
    Procrustes declares a global rotation irrelevant and the prediction head is not rotation-equivariant.
    """
    moved = df[~df["is_baseline"].astype(bool)]
    if moved.empty or "cka" not in moved.columns or moved["cka"].isna().all():
        return []
    contexts = sorted(moved["context_t"].unique())
    base = style.model_color("tabpfn_v2_5")
    fig, axes = style.figure("onehalf", nrows=2, sharex=True, squeeze=False)
    top, bottom = axes[0][0], axes[1][0]
    for i, t in enumerate(contexts):
        color = base if len(contexts) == 1 else style.derive_shade(base, i, len(contexts))
        sub = moved[moved["context_t"] == t]
        for ax, column in ((top, "disparity"), (bottom, "cka")):
            med, q1, q3 = _procrustes_median(sub, column, "layer")
            ax.plot(med.index, med.values, color=color, marker="o", label=f"$t$ = {int(t)}")
            ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=style.BAND_ALPHA, lw=0)
        null_med, _, _ = _procrustes_median(sub, "null_mean", "layer")
        top.plot(null_med.index, null_med.values, color=style.NEUTRAL_GREY, ls=PROCRUSTES_NULL_LINESTYLE,
                 lw=style.LINE_WIDTH, label="Shuffled-site null" if i == 0 else None)
    top.set_ylabel(style.axis_label("procrustes_disparity"))
    bottom.set_ylabel(style.axis_label("cka_pairwise"))
    bottom.set_xlabel(style.axis_label("layer"))
    style.set_integer_xaxis(bottom)
    top.legend(loc="best", fontsize=style.FONT_SIZES["legend"], ncol=2)
    style.panel_letters([top, bottom])
    return style.save_figure(fig, run_dir, "procrustes_vs_cka")


def render_procrustes(run_dir: str) -> list[str]:
    """Figures of the Procrustes complement: disparity vs context, vs layer, and against CKA on the same pairs.

    Args:
        run_dir: Procrustes run directory (``output/mechanism/procrustes/cka/<dataset>/<tag>/``).

    Returns:
        Written paths.
    """
    path = os.path.join(run_dir, "procrustes.csv")
    if not _has_rows(path):
        return []
    df = pd.read_csv(path)
    written = _procrustes_vs_context(df, run_dir)
    written += _procrustes_vs_layer(df, run_dir)
    return written + _procrustes_vs_cka(df, run_dir)


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
