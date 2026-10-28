"""Figures for the NFM progress presentation (user request 2026-10-24), written to ``output/NFM-presentation/``.

Each figure is a trimmed variant of an existing run figure, rebuilt from that run's files alone (no experiment is
re-run). Style comes from :mod:`pfns4neurostim.visualization.style`; the presentation-only departures are TabICL's
colour and label (purple, ``TabICL-2``: the vendored checkpoint is ``tabicl-regressor-v2-20260212``) and the K2 noise
axes below, applied here and nowhere else, so the paper figures keep the registered palette and wording.

    A  ``A_benchmark_nhp``              exploration score and R^2 of hyp-a-nhp's trajectories
    B  ``B_benchmark_{5d_rat,spinal}``  the same for 5d_rat and spinal
    C  ``C_pfn_models_nhp``             exploration score and R^2 of the PFN benchmark (with GP-MLL)
    D  ``D_stress_{k2_channel,k5_outliers,k6_failure}_nhp``  exploration score and R^2 vs the knob (draws, in vivo);
       K2 against its noise multiplier (:data:`X_AXIS_OVERRIDES`)
    E  ``E_update_exemplar``            C1 exemplar maps, TabPFN-2.5 and GP-fixed, one colour scale shared by both rows
    F  ``F_update_vs_distance``         C1 transfer vs distance without GP-MLL frozen
    G  ``G_surprise_response``          Omega-4 without GP-MLL frozen, y-axis without the unit parenthesis
    H  ``H_coherence``                  drift, total-variance ratio and covariance asymmetry (nhp-l1-refine), no GP-MLL
    I  ``I_ground_truth_map``           ground-truth response map of the exemplar NHP channel
    J  ``J_representation_vs_layer``    t = 60: zero-parameter vs fitted lens (mean-map r); label-token CKA with three mean maps

Usage:
    python scripts/nfm_presentation_figures.py
    python scripts/nfm_presentation_figures.py --only A C --formats svg
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Any, Sequence

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml

from pfns4neurostim.analysis.bayes_update import POOLED_ANCHORS
from pfns4neurostim.evaluation.results import canonical_demo_column
from pfns4neurostim.visualization import mechanism as MV
from pfns4neurostim.visualization import stress as SV
from pfns4neurostim.visualization import style as S
from pfns4neurostim.visualization import traces as T

#: Presentation-only identity of TabICL (user 2026-10-24: purple, named by its version). ColorBrewer's dark purple,
#: >= 39 dE from every other PFN colour (TabPFN-3.5 magenta 53, TabFM blue 39, TabPFN-2.5 43).
TABICL_PRESENTATION_COLOR: str = "#6A3D9A"
TABICL_PRESENTATION_LABEL: str = "TabICL-2"

#: Trace panels kept in A-C and metric panels kept in D (exploration score, then R^2).
TRACE_PANELS: tuple[tuple[str, str], ...] = (("exploration_per_step", "exploration_score"), ("r2_per_step", "r2"))
STRESS_METRICS: tuple[str, ...] = ("exploration_score", "r2")

#: Benchmark runs of A / B / C: (figure name, run dir relative to the output root, dataset).
BENCHMARK_RUNS: tuple[tuple[str, str, str], ...] = (
    ("A_benchmark_nhp", "benchmark/nhp/hyp-a-nhp", "nhp"),
    ("B_benchmark_5d_rat", "benchmark/5d_rat/hyp-a-5d_rat", "5d_rat"),
    ("B_benchmark_spinal", "benchmark/spinal/hyp-a-spinal", "spinal"),
    ("C_pfn_models_nhp", "benchmark/nhp/hyp0-pfn-bench-nhp", "nhp"),
)
#: Stress runs of D (draws, in vivo): (figure name, run dir, knob, dataset).
STRESS_RUNS: tuple[tuple[str, str, str, str], ...] = (
    ("D_stress_k2_channel_nhp", "stress/k2_channel/nhp/stress-k2-channel-draws-nhp", "k2_channel", "nhp"),
    ("D_stress_k5_outliers_nhp", "stress/k5_outliers/nhp/stress-k5-nhp", "k5_outliers", "nhp"),
    ("D_stress_k6_failure_nhp", "stress/k6_failure/nhp/stress-k6-failure-nhp", "k6_failure", "nhp"),
)
#: Knobs whose default regime is the first level of the axis (no stress = 0), so the default marker says nothing.
NO_NOMINAL_MARKER: frozenset[str] = frozenset({"k5_outliers", "k6_failure"})

#: Knob x-axes that replace the run figure's, as ``(column, label, scale)``. K2 is drawn against its noise multiplier
#: alpha (the level), chosen by the user 2026-10-24 over achieved SNR, noise SD / signal SD and noise share: K2
#: multiplies every channel's trial residuals by alpha, so it is exact and channel-free, with 1 = the recorded data.
X_AXIS_OVERRIDES: dict[str, tuple[str, str, str]] = {
    "k2_channel": ("level", "Trial noise (× recorded level)", "log"),
}

#: Hyp C runs: the pre-registered grid (E-G: t 10-80, exemplar) and the level-1 refinement (H, user's choice).
OMEGA_RUN: str = "mechanism/bayes_update/nhp/nhp"
OMEGA_REFINE_RUN: str = "mechanism/bayes_update/nhp/nhp-l1-refine"
#: BETA run of J: the main representation grid (official for M2 beta2).
#: Inputs of J (user 2026-10-24): one context size, t = 60, which is on no pre-registered ladder, so both inputs were
#: computed for this figure -- the BETA cells at t = 60 with GP-fixed's mean map added as a target, and the tuned lens
#: retrained on t in {10, 25, 50, 60, 80} (``scripts/tuned_lens_nhp.py``, held-out-subject folds). Exploratory.
REPRESENTATION_T60_RUN: str = "mechanism/representation/nhp/nhp-t60-presentation"
TUNED_LENS_T60_DIR: str = "diagnostics/tuned_lens_t60"
J_CONTEXT_T: int = 60
#: Panel (a): lens variants as (variant in the tuned-lens summary, legend label, linestyle).
J_LENS_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("zero", "Zero-parameter lens", "--"),
    ("fitted", "Fitted lens (held-out subjects)", "-"),
)
#: Panel (b): label-token CKA targets as (target, engine whose style it takes, legend label). The ground-truth map and the
#: smooth-field chance lines are left out of the presentation figure (user 2026-10-26); the run still computes them.
J_CKA_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("mean_own", "tabpfn_v2_5", "TabPFN-2.5 mean map"),
    ("mean_mll", "gp_mll_refit", "GP-MLL refit mean map"),
    ("mean_fixed", "gp_fixed", "GP-fixed mean map"),
)
#: y-label of panel (b): the library's, without the word "deconfounded" (user 2026-10-26).
J_CKA_YLABEL: str = "Debiased CKA\n(geometry removed)"
#: Top of each J legend, in axes fraction below the panel (under the x-axis label).
J_LEGEND_DROP: float = -0.2

#: Engines removed per Hyp C figure (user 2026-10-24); the T 1.0 control is already left out of every Hyp C figure.
EXEMPLAR_DROP: tuple[str, ...] = ("gp_mll_refit", "gp_mll_frozen")
DISTANCE_DROP: tuple[str, ...] = ("gp_mll_frozen",)
SURPRISE_DROP: tuple[str, ...] = ("gp_mll_frozen",)
COHERENCE_DROP: tuple[str, ...] = ("gp_mll_refit",)
COHERENCE_METRICS: tuple[str, ...] = ("drift_abs", "total_var_ratio", "cov_asymmetry")
#: Fixed y-limits of the H panels (user 2026-10-26), so the three readouts are read on comparable ranges.
COHERENCE_YLIM: dict[str, tuple[float, float]] = {"drift_abs": (0.0, 1.0), "total_var_ratio": (0.5, 1.5),
                                                  "cov_asymmetry": (0.0, 1.0)}

#: Note on the GP-fixed curve of the asymmetry panel: an exact GP scores 0; what remains is quadrature error.
ASYMMETRY_NOTE: str = "GP-fixed: exact value 0;\nresidual = RQMC quadrature floor"

#: Panel (d) of H (user 2026-10-25): the exchangeability test, ``scripts/exchangeability_drift.py`` (row-order drift of
#: the predictive, in predictive SD; 0 for an exchangeable model). Read from the CSV that script writes.
EXCHANGEABILITY_CSV: str = os.path.join("diagnostics", "exchangeability", "exchangeability.csv")
EXCHANGEABILITY_VALUE: str = "drift_perm"
EXCHANGEABILITY_YLABEL: str = "Row-order drift\n(predictive SD)"
EXCHANGEABILITY_NOTE: str = "dotted: TabPFN-2.5 drift in (a)"


def _apply_presentation_style() -> None:
    """Give TabICL its presentation colour and label (in-process only; style.py is untouched)."""
    S.MODEL_STYLES["tabicl"] = dataclasses.replace(
        S.MODEL_STYLES["tabicl"], color=TABICL_PRESENTATION_COLOR, label=TABICL_PRESENTATION_LABEL)


def _save(fig: Any, out_dir: str, name: str, formats: Sequence[str]) -> list[str]:
    """Save through :func:`style.save_figure` in the requested formats."""
    return S.save_figure(fig, out_dir, name, formats=tuple(formats))


def _width(token: str, n_panels: int, n_reference: int) -> float:
    """Width that keeps each panel the size it has in the reference figure (same aspect, same physical size).

    Args:
        token: Width token of the reference figure.
        n_panels: Panels in the new figure.
        n_reference: Panels in the reference figure.

    Returns:
        Width in inches.
    """
    return S.FIG_WIDTHS[token] * n_panels / n_reference


# ---------------------------------------------------------------------------
# A / B / C: benchmark trajectories
# ---------------------------------------------------------------------------
def _trace_axis(ax: Any, frame: pd.DataFrame, field: str, key: str) -> None:
    """Draw one trace panel (mean / 95% CI or median / IQR over channels), as in ``traces.plot_trace_panels``.

    The y-label is the metric alone (user 2026-10-24): the aggregation goes in the caption, not on the axis.
    """
    for _, _, rows, st in T._series(frame):
        band = T._channel_band(rows, field)
        if band is None:
            continue
        centre, lower, upper, _ = band
        x = int(rows["n_init"].iloc[0]) + np.arange(centre.shape[0])
        ax.plot(x, centre, color=st.color, linestyle=st.linestyle, linewidth=S.LINE_WIDTH, label=st.label,
                zorder=st.zorder)
        ax.fill_between(x, lower, upper, color=st.color, alpha=S.BAND_ALPHA, linewidth=0)
    ax.set_xlabel(S.axis_label("budget"))
    ax.set_ylabel(S.axis_label(key))
    if key == "exploration_score":
        ax.set_ylim(0.0, 1.0)
    if key == "r2":
        ax.set_ylim(bottom=S.R2_AXIS_FLOOR, top=1.0)
        ax.text(0.98, 0.02, f"axis clipped at {S.R2_AXIS_FLOOR:g}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=S.FONT_SIZES["annotation"], alpha=0.7)


def benchmark_figure(run_dir: str, dataset: str, out_dir: str, name: str, formats: Sequence[str]) -> list[str]:
    """A / B / C: exploration score and R^2 along the BO run, same panel size as the reference.

    Args:
        run_dir: Benchmark run directory (``trajectories.pkl``).
        dataset: Dataset name, for the title.
        out_dir: Destination directory.
        name: Output basename.
        formats: File formats.

    Returns:
        Paths written.
    """
    frame = T.load_trace_frame(run_dir)
    if frame is None or frame.empty:
        raise FileNotFoundError(f"No trajectories in {run_dir}.")
    fig, axes = S.figure(_width("double", len(TRACE_PANELS), 3), nrows=1, ncols=len(TRACE_PANELS),
                         layout=S.LAYOUT_ENGINE)
    axes = np.atleast_1d(axes)
    for ax, (field, key) in zip(axes, TRACE_PANELS):
        _trace_axis(ax, frame, field, key)
    fig.suptitle(T._title(frame, dataset, "BO trajectories"), x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    S.panel_letters(axes, x=0.0)
    T._legend_below(fig, axes, len(T._series(frame)))
    return _save(fig, out_dir, name, formats)


# ---------------------------------------------------------------------------
# D: stress degradation curves
# ---------------------------------------------------------------------------
def stress_figure(run_dir: str, knob: str, dataset: str, out_dir: str, name: str, formats: Sequence[str]) -> list[str]:
    """D: exploration score and R^2 vs the knob, as ``degradation_draws.svg`` minus the regret panel.

    Per user 2026-10-24: no per-channel traces, no channels-per-level counts (every level keeps all channels), and no
    default-regime marker for knobs whose default is the axis origin (:data:`NO_NOMINAL_MARKER`). More stress always
    reads to the right.

    Args:
        run_dir: Stress run directory (``tidy.csv``).
        knob: Knob name.
        dataset: Dataset name.
        out_dir: Destination directory.
        name: Output basename.
        formats: File formats.

    Returns:
        Paths written.
    """
    df = SV._shared_ladder(pd.read_csv(os.path.join(run_dir, "tidy.csv"), low_memory=False))
    df = df.assign(demo=canonical_demo_column(df["demo"]))
    df, _ = SV.clip_low_snr(df, SV.DEFAULT_MIN_SNR_DB)
    x_col, x_label, x_scale = X_AXIS_OVERRIDES.get(knob) or (SV._knob_axis(df, knob), None, "linear")
    x_label = x_label or S.knob_x_label(knob, x_col)
    nominal_x = None if knob in NO_NOMINAL_MARKER else SV._nominal_x(df, knob, x_col)
    demo = str(df["demo"].iloc[0])
    models = [m for m in S.MODEL_ORDER if m in set(df["model"])]
    metrics = [m for m in STRESS_METRICS if m in df.columns]
    fig, axes = S.figure(_width("wide", len(metrics), 3), nrows=1, ncols=len(metrics), sharex=True,
                         layout=S.LAYOUT_ENGINE)
    axes = np.atleast_1d(axes)
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        heavy = SV._is_heavy_tailed(metric)
        for model in models:
            sub = df[df["model"] == model]
            x, centre, lower, upper = SV._model_band(sub, x_col, metric)
            ax.plot(x, centre, **S.plot_kwargs(model))
            ax.fill_between(x, lower, upper, color=S.model_color(model), alpha=S.BAND_ALPHA, linewidth=0)
        if nominal_x is not None:
            S.mark_nominal(ax, nominal_x, label=S.NOMINAL_MARKER_LABEL if idx == 0 else None)
        ax.set_ylabel(S.axis_label(f"{metric}_median" if heavy else metric))
        if metric == "r2":
            ax.set_ylim(bottom=S.R2_AXIS_FLOOR, top=1.0)
            ax.text(0.98, 0.02, f"axis clipped at {S.R2_AXIS_FLOOR:g}", transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=S.FONT_SIZES["annotation"], alpha=0.7)
    ax0 = axes[0]
    ax0.set_xscale(x_scale)
    if x_col == "level" and x_scale == "log":
        # One tick per level, in plain numbers: the noise multiplier is the knob itself.
        levels = sorted(df["level"].astype(float).unique())
        ax0.xaxis.set_major_locator(mticker.FixedLocator(levels))
        ax0.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax0.xaxis.set_minor_locator(mticker.NullLocator())
    elif x_scale == "log":
        ax0.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax0.xaxis.set_minor_formatter(mticker.NullFormatter())
    if x_col in SV.X_DESCENDS_WITH_STRESS:
        ax0.invert_xaxis()
    for ax in axes:
        ax.set_xlabel(x_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False,
               fontsize=S.FONT_SIZES["legend"])
    fig.suptitle(f"{S.DATASET_LABELS.get(dataset, dataset)} - {S.KNOB_LABELS.get(knob, knob)} - "
                 f"{S.DEMO_LABELS.get(demo, demo)}", x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    S.panel_letters(axes, x=0.0, y=1.02)
    return _save(fig, out_dir, name, formats)


# ---------------------------------------------------------------------------
# E-H: Hypothesis C
# ---------------------------------------------------------------------------
def _bayes_block(run_dir: str) -> dict[str, Any]:
    """The run's resolved ``bayes_update`` block."""
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh).get("bayes_update") or {}


def exemplar_figure(run_dir: str, out_dir: str, name: str, formats: Sequence[str]) -> list[str]:
    """E: change of mean after the exemplar observation, one row per engine, ONE colour scale for every row.

    Args:
        run_dir: OMEGA run directory (``c1_exemplar.npz``).
        out_dir: Destination directory.
        name: Output basename.
        formats: File formats.

    Returns:
        Paths written.
    """
    ex = np.load(os.path.join(run_dir, "c1_exemplar.npz"))
    engines_all = [str(e) for e in ex["engines"]]
    keep = [i for i, e in enumerate(engines_all) if e not in EXEMPLAR_DROP]
    dmu = ex["dmu"][:, keep]                                                       # [t, engine, M]
    engines = [engines_all[i] for i in keep]
    ts, xy = ex["context_ts"], ex["ch2xy"].astype(int)                             # [t], [M, 2]
    shape = tuple(xy.max(axis=0) + 1)
    vmax = float(np.abs(dmu).max()) or 1.0                                         # shared across rows
    a = int(ex["anchor"])
    fig, axes = S.figure("full", nrows=len(engines), ncols=len(ts), squeeze=False, panel_aspect=MV._OMEGA_MAP_ASPECT,
                         layout=S.LAYOUT_ENGINE)
    im = None
    for r, engine in enumerate(engines):
        for c, t in enumerate(ts):
            ax = axes[r][c]
            grid = np.full(shape, np.nan)
            grid[xy[:, 0], xy[:, 1]] = dmu[c, r]
            im = ax.imshow(grid.T, origin="lower", cmap=S.DIVERGING_CMAP, vmin=-vmax, vmax=vmax)
            obs = ex["observed"][c]
            ax.scatter(xy[obs, 0], xy[obs, 1], s=S.MARKER_SIZE, color=S.NEUTRAL_GREY, marker=".", lw=0)
            ax.scatter([xy[a, 0]], [xy[a, 1]], s=S.MARKER_SIZE * 4, facecolors="none", edgecolors=S.NEUTRAL_GREY,
                       lw=S.LINE_WIDTH)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"$t$ = {int(t)}", fontsize=S.FONT_SIZES["legend"])
            if c == 0:
                ax.set_ylabel(MV.arm_style(engine).label)
    fig.colorbar(im, ax=list(axes.ravel()), label=S.axis_label("dmu_exemplar"), shrink=0.6)
    ladder = MV.arm_style(str((_bayes_block(run_dir).get("shared") or {}).get("reference"))).label
    fig.suptitle(f"Channel {ex['channel']}: change of mean after one observation at the circled site, "
                 f"$y' = \\mu + {float(ex['exemplar_c']):g}\\,\\sigma$ of {ladder}; dots = context sites; "
                 f"one colour scale for every row", x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    return _save(fig, out_dir, name, formats)


def distance_figure(run_dir: str, out_dir: str, name: str, formats: Sequence[str]) -> list[str]:
    """F: C1's update-vs-distance figure without GP-MLL frozen (``visualization.mechanism._c1_distance``)."""
    cfg = _bayes_block(run_dir)
    engines = [e for e in cfg["c1"]["engines"] if e not in DISTANCE_DROP]
    profile = pd.read_csv(os.path.join(run_dir, "transfer_profile.csv"))
    return _renamed(lambda d: MV._c1_distance(profile, engines, cfg["panel_ts"], d), "c1_update_vs_distance",
                    out_dir, name, formats)


def surprise_figure(run_dir: str, out_dir: str, name: str, formats: Sequence[str]) -> list[str]:
    """G: Omega-4 without GP-MLL frozen; the y-axis reads "Update at x'" (the unit stays in the title)."""
    cfg = _bayes_block(run_dir)
    surprise = pd.read_csv(os.path.join(run_dir, "surprise.csv"))
    engines = [e for e in surprise["engine"].unique() if e not in SURPRISE_DROP]
    old = S.AXIS_LABELS["update_anchor_sd"]
    S.AXIS_LABELS["update_anchor_sd"] = "Update at $x'$"
    try:
        return _renamed(lambda d: MV._omega_surprise(surprise, cfg["panel_ts"], d, engines=engines, name="surprise",
                                                     reference=(cfg.get("shared") or {}).get("reference")),
                        "surprise", out_dir, name, formats)
    finally:
        S.AXIS_LABELS["update_anchor_sd"] = old


def coherence_figure(run_dir: str, out_dir: str, name: str, formats: Sequence[str],
                     exchangeability_csv: str | None = None) -> list[str]:
    """H: pooled-anchor drift, total-variance ratio and covariance asymmetry vs t, without GP-MLL refit.

    With ``exchangeability_csv`` a fourth panel (d) is added: the row-order drift of the predictive
    (``scripts/exchangeability_drift.py``) on a log axis, with TabPFN-2.5's drift of panel (a) as a dotted reference line.

    Args:
        run_dir: OMEGA run directory (``coherence_cell.csv``).
        out_dir: Destination directory.
        name: Output basename.
        formats: File formats.
        exchangeability_csv: Output of the exchangeability script; ``None`` draws the three OMEGA panels only.
    Returns:
        Paths written.
    """
    cell = MV._shown(MV._nominal(pd.read_csv(os.path.join(run_dir, "coherence_cell.csv"))))
    pooled = cell[(cell["pool"] == POOLED_ANCHORS) & ~cell["engine"].isin(COHERENCE_DROP)]
    ts = sorted(pooled["context_t"].unique())
    n_panels = len(COHERENCE_METRICS) + (exchangeability_csv is not None)
    fig, axes = S.figure(_width("wide", n_panels, len(COHERENCE_METRICS)), nrows=1, ncols=n_panels, squeeze=False,
                         panel_aspect=MV._OMEGA_PANEL_ASPECT, layout=S.LAYOUT_ENGINE)
    for ax, metric in zip(axes[0], COHERENCE_METRICS):
        for engine in pooled["engine"].unique():
            MV._median_band(ax, pooled[pooled["engine"] == engine], metric, MV.arm_style(engine))
        MV._bayes_value(ax, MV.OMEGA_COHERENCE_METRICS[metric])
        MV._context_axis(ax, ts)
        ax.set_ylabel(S.axis_label(metric))
        ax.set_ylim(*COHERENCE_YLIM[metric])
        if metric == "cov_asymmetry":
            ax.text(0.98, 0.98, ASYMMETRY_NOTE, transform=ax.transAxes, ha="right", va="top",
                    fontsize=S.FONT_SIZES["annotation"], alpha=0.8)
    axes[0][0].legend(loc="best", fontsize=S.FONT_SIZES["legend"] - 1)
    if exchangeability_csv is not None:
        ex = pd.read_csv(exchangeability_csv)
        ex = ex[np.isclose(ex["level"], ex["level"].min()) & ~ex["engine"].isin(COHERENCE_DROP)
                & ex["context_t"].isin(ts)]
        ax = axes[0][-1]
        for engine in ex["engine"].unique():
            MV._median_band(ax, ex[ex["engine"] == engine], EXCHANGEABILITY_VALUE, MV.arm_style(engine))
        reference = float(pooled[pooled["engine"] == "tabpfn_v2_5"]["drift_abs"].median())
        ax.axhline(reference, color=S.NEUTRAL_GREY, lw=S.LINE_WIDTH / 2, ls=":", zorder=0)
        ax.text(0.98, 0.98, EXCHANGEABILITY_NOTE, transform=ax.transAxes, ha="right", va="top",
                fontsize=S.FONT_SIZES["annotation"], alpha=0.8)
        ax.set_yscale("log")
        MV._context_axis(ax, ts)
        ax.set_ylabel(EXCHANGEABILITY_YLABEL)
    S.panel_letters(axes[0])
    return _save(fig, out_dir, name, formats)


def ground_truth_figure(run_dir: str, out_dir: str, name: str, formats: Sequence[str]) -> list[str]:
    """I: the ground-truth response map of one NHP channel on the electrode array (the exemplar channel of E).

    The ground truth is the mean over every valid trial at each site (``gt_mode: full_mean``), on the z-scored response
    scale of the preprocessing contract; corners without an electrode stay blank.

    Args:
        run_dir: OMEGA run directory, whose config gives the dataset block and the exemplar channel.
        out_dir: Destination directory.
        name: Output basename.
        formats: File formats.

    Returns:
        Paths written.
    """
    from pfns4neurostim.data.channels import iter_channels  # noqa: PLC0415

    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    ds, ex = cfg["dataset"], cfg["bayes_update"]["exemplar"]
    # A null exemplar subject / EMG means "the first one", as in the OMEGA runner's exemplar pick.
    subjects = ds["subjects"] if ex["subject"] is None else [int(ex["subject"])]
    emgs = None if ex["emg"] is None else [int(ex["emg"])]
    ch = next(iter(iter_channels(ds["name"], subjects, emgs, data_root=ds["data_root"],
                                 normalization=ds["normalization"])))
    xy = np.asarray(ch.ch2xy).astype(int)                                          # [M, 2]
    grid = np.full(tuple(xy.max(axis=0) + 1), np.nan)
    grid[xy[:, 0], xy[:, 1]] = ch.y_gt
    n_trials = int(np.median(np.sum(~np.isnan(ch.Y_trials), axis=1)))
    best = int(np.argmax(ch.y_gt))
    fig, ax = S.figure("single", panel_aspect=grid.shape[1] / grid.shape[0], layout=S.LAYOUT_ENGINE)
    im = ax.imshow(grid.T, origin="lower", cmap=S.SEQUENTIAL_CMAP)
    ax.scatter([xy[best, 0]], [xy[best, 1]], s=S.MARKER_SIZE * 6, facecolors="none", edgecolors=S.NEUTRAL_GREY,
               lw=S.LINE_WIDTH)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Electrode column")
    ax.set_ylabel("Electrode row")
    fig.colorbar(im, ax=ax, label="Mean evoked response (z-scored)")
    fig.suptitle(f"{S.DATASET_LABELS.get(ds['name'], ds['name'])} - subject {ch.subject}, EMG {ch.emg}: "
                 f"ground-truth map\n{len(xy)} electrodes, median {n_trials} trials per site; circle = best site",
                 x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    return _save(fig, out_dir, name, formats)


def representation_figure(run_dir: str, lens_dir: str, out_dir: str, name: str, formats: Sequence[str]) -> list[str]:
    """J at t = :data:`J_CONTEXT_T`: (a) lens fidelity of the mean map, (b) label-token CKA with three mean maps.

    (a) Pearson r between each layer's lens-decoded mean map and TabPFN's final mean map, for the zero-parameter lens
    and the fitted (tuned) lens, median over channels with the channel-bootstrap interval, held-out subjects.
    (b) Geometry-deconfounded debiased CKA of the label token with TabPFN-2.5's, GP-MLL refit's and GP-fixed's mean maps
    (no ground-truth line and no smooth-field chance lines in the presentation version, user 2026-10-26). Layers 0-1 are
    left out of (a), as in every lens figure.

    Args:
        run_dir: BETA representation run at t = 60 (``representation_summary.csv``, ``headline.json``).
        lens_dir: Tuned-lens output directory (``summary.csv``).
        out_dir: Destination directory.
        name: Output basename.
        formats: File formats.

    Returns:
        Paths written.
    """
    import json  # noqa: PLC0415

    with open(os.path.join(run_dir, "headline.json"), encoding="utf-8") as fh:
        head = json.load(fh)
    summary = MV._figure_level(pd.read_csv(os.path.join(run_dir, "representation_summary.csv")), run_dir)
    summary = summary[(summary["site_set"] == "all") & (summary["readout"] == head["readout"])
                      & (summary["context_t"] == J_CONTEXT_T)]
    tuned = os.path.join(lens_dir, "summary.csv")
    if os.path.exists(tuned):
        lens = pd.read_csv(tuned)
    else:
        # The tuned lens is still training: draw the zero-parameter lens from the run itself (same definition:
        # unobserved sites against TabPFN's final map) and say so in the figure.
        lens = pd.read_csv(os.path.join(run_dir, "lens_summary.csv"))
        lens = lens[(lens["comparison"] == "tabpfn_v2_5") & (lens["site_pool"] == "unobserved")].assign(variant="zero")
    lens = lens[(lens["context_t"] == J_CONTEXT_T) & (lens["value"] == "lens_r_final")
                & ~lens["layer"].isin(MV.LENS_EXCLUDED_BLOCKS)]
    if summary.empty or lens.empty:
        raise ValueError(f"J needs t = {J_CONTEXT_T} rows in {run_dir} and {lens_dir}.")
    fig, axes = S.figure(_width("wide", 2, 3), nrows=1, ncols=2, squeeze=False, sharex=True,
                         panel_aspect=MV._REPRESENTATION_PANEL_ASPECT, layout=S.LAYOUT_ENGINE)
    ax_lens, ax_cka = axes[0]
    tab = S.model_color("tabpfn_v2_5")
    for variant, label, ls in J_LENS_VARIANTS:
        sub = lens[lens["variant"] == variant].sort_values("layer")
        if sub.empty:
            ax_lens.plot([], [], color=tab, ls=ls, lw=S.LINE_WIDTH, label=f"{label}: pending")
            continue
        ax_lens.plot(sub["layer"], sub["median"], color=tab, ls=ls, lw=S.LINE_WIDTH, label=label)
        ax_lens.fill_between(sub["layer"], sub["ci_low"], sub["ci_high"], color=tab, alpha=S.BAND_ALPHA, lw=0)
    MV._bayes_value(ax_lens, MV.LENS_FIDELITY_METRICS["lens_r_final"])
    ax_lens.set_title("Lens-decoded vs final TabPFN-2.5 mean map", fontsize=S.FONT_SIZES["base"])
    ax_lens.set_ylabel(S.axis_label("lens_r_final"))
    ax_lens.legend(loc="upper center", bbox_to_anchor=(0.5, J_LEGEND_DROP), ncol=1, fontsize=S.FONT_SIZES["legend"],
                   frameon=False)
    for target, engine, label in J_CKA_TARGETS:
        rows = summary[summary["target"] == target]
        if rows.empty:
            raise ValueError(f"J: target {target!r} has no t = {J_CONTEXT_T} rows in {run_dir}.")
        st = MV.arm_style(engine)
        color, ls = st.color, st.linestyle
        val = rows[rows["value"] == "headline_value"].sort_values("layer")
        ax_cka.plot(val["layer"], val["median"], color=color, ls=ls, lw=S.LINE_WIDTH, label=label)
        ax_cka.fill_between(val["layer"], val["ci_low"], val["ci_high"], color=color, alpha=S.BAND_ALPHA, lw=0)
    ax_cka.axhline(0.0, color=S.NEUTRAL_GREY, lw=S.LINE_WIDTH / 2, zorder=0)
    ax_cka.set_title("Label token vs mean maps", fontsize=S.FONT_SIZES["base"])
    ax_cka.set_ylabel(J_CKA_YLABEL)
    ax_cka.legend(loc="upper center", bbox_to_anchor=(0.5, J_LEGEND_DROP), ncol=1, fontsize=S.FONT_SIZES["legend"],
                  frameon=False)
    for ax in (ax_lens, ax_cka):
        ax.set_xlabel(S.axis_label("layer"))
        S.set_integer_xaxis(ax)
    fig.suptitle(f"{S.DATASET_LABELS.get('nhp', 'nhp')} - $t$ = {J_CONTEXT_T} context sites, "
                 f"{int(lens['n_channels'].max())} channels (median, 95% channel-bootstrap interval)",
                 x=0.01, ha="left", fontsize=S.FONT_SIZES["title"])
    S.panel_letters([ax_lens, ax_cka])
    return _save(fig, out_dir, name, formats)


def _renamed(draw: Any, written_name: str, out_dir: str, name: str, formats: Sequence[str]) -> list[str]:
    """Run a library figure function whose output name is fixed, then give its files the presentation name.

    The library saves SVG only; for every other requested format the figure is drawn again with
    :func:`style.save_figure` temporarily asked for that format.

    Args:
        draw: Callable taking the destination directory and returning the written paths.
        written_name: Basename the library function writes.
        out_dir: Destination directory.
        name: Presentation basename.
        formats: File formats.

    Returns:
        Paths written.
    """
    original = S.save_figure

    def save(fig: Any, d: str, n: str, **kw: Any) -> list[str]:
        return original(fig, d, name if n == written_name else n, formats=tuple(formats))

    S.save_figure = save
    try:
        return draw(out_dir)
    finally:
        S.save_figure = original


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Build the requested presentation figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output-root", default="output", help="Root holding the run directories.")
    parser.add_argument("--out-dir", default=os.path.join("output", "NFM-presentation"))
    parser.add_argument("--formats", nargs="+", default=["svg"])
    parser.add_argument("--only", nargs="+", default=list("ABCDEFGHIJ"), help="Figure letters to build.")
    args = parser.parse_args(argv)
    _apply_presentation_style()
    root, out, fmt = args.output_root, args.out_dir, args.formats
    written: list[str] = []
    for name, run, dataset in BENCHMARK_RUNS:
        if name[0] in args.only:
            written += benchmark_figure(os.path.join(root, run), dataset, out, name, fmt)
    if "D" in args.only:
        for name, run, knob, dataset in STRESS_RUNS:
            written += stress_figure(os.path.join(root, run), knob, dataset, out, name, fmt)
    omega, refine = os.path.join(root, OMEGA_RUN), os.path.join(root, OMEGA_REFINE_RUN)
    if "E" in args.only:
        written += exemplar_figure(omega, out, "E_update_exemplar", fmt)
    if "F" in args.only:
        written += distance_figure(omega, out, "F_update_vs_distance", fmt)
    if "G" in args.only:
        written += surprise_figure(omega, out, "G_surprise_response", fmt)
    if "H" in args.only:
        written += coherence_figure(refine, out, "H_coherence", fmt)
    if "Hd" in args.only:
        written += coherence_figure(refine, out, "H_coherence_exchangeability", fmt,
                                    exchangeability_csv=os.path.join(root, EXCHANGEABILITY_CSV))
    if "I" in args.only:
        written += ground_truth_figure(omega, out, "I_ground_truth_map", fmt)
    if "J" in args.only:
        written += representation_figure(os.path.join(root, REPRESENTATION_T60_RUN),
                                         os.path.join(root, TUNED_LENS_T60_DIR), out, "J_representation_vs_layer", fmt)
    for path in written:
        print(f"[nfm] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
