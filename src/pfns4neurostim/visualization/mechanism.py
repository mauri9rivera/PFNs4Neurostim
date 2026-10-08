"""Hyp C mechanism figures: update rule (M10), placement (MMD / sliced W2), CKA.

The legacy ID/OOD dashboard that used to live under this name (its own hardcoded palette)
is in :mod:`pfns4neurostim.legacy_code.visualization_id_ood`; everything here takes geometry,
colours, labels and axis strings from :mod:`visualization.style` (CLAUDE.md §7).
Each ``render_*`` function reads only the CSVs of a run directory, so ``--replot`` rebuilds
every figure without re-running an analysis.
"""
from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style

__all__ = [
    "arm_style",
    "render_update_rule",
    "render_placement",
    "render_cka",
]

#: Legend title over the stress-level series of the placement trajectory, so the lines are not an
#: unlabelled ladder of colours (the levels are the K2 residual-amplification factor alpha).
PLACEMENT_LEVEL_LEGEND_TITLE: str = "K2 noise level (alpha)"

#: Half-width of the floor / ceiling reference bands, in placement units.
REFERENCE_BAND_HALF_WIDTH: float = 0.05


#: CKA (a) target kernels that get a panel, in display order. ``K_X`` (electrode geometry) and
#: ``K_GT_linear`` were dropped from the figure on 2026-09-27: ``K_X`` is an explicit *control* -- the model's
#: input IS the coordinates, so agreement with geometry is trivial and it read highest, which invited the
#: wrong conclusion -- and ``K_GT_linear`` is a bandwidth *sensitivity* check on ``K_GT``. Both stay in
#: ``cka.csv`` and in the config's ``targets``, so the control and the sensitivity check remain auditable;
#: they are simply not headline panels. The targets of interest are the response and the GP posterior.
CKA_PLOTTED_TARGETS: tuple[str, ...] = ("K_GT", "K_GP")


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


#: Update-rule engine arms and how they map onto the model palette.
_GP_ARMS: tuple[str, ...] = ("frozen", "refit")
#: Geometry of the F1 heatmap grid: square panels, a small gap between them, and a colorbar close to the grid.
_F1_PANEL_ASPECT: float = 1.0
_F1_WSPACE: float = 0.08
_F1_HSPACE: float = 0.12
_F1_COLORBAR_PAD: float = 0.02


def arm_style(engine: str) -> style.ModelStyle:
    """Style of an analysis arm (``tabpfn_v2_5``, ``gp_mll_frozen``, ``gp_mll_refit``, ...).

    Args:
        engine: Arm name.

    Returns:
        The model style; GP-MLL arms are shades of the GP-MLL anchor.
    """
    for arm in _GP_ARMS:
        if engine == f"gp_mll_{arm}":
            return style.series_style("gp_mll", arm, list(_GP_ARMS))
    if engine == "gp_fixed_frozen":
        return style.model_style("gp_naive")   # the GP-fixed arm of the paper (fixed kernel, no fitting)
    return style.model_style(engine)


def _iqr(values: pd.Series) -> tuple[float, float, float]:
    """``(median, q25, q75)`` of a series."""
    return float(values.median()), float(values.quantile(0.25)), float(values.quantile(0.75))


# ---------------------------------------------------------------------------
# Update rule (task #9, roadmap M10 F1-F5)
# ---------------------------------------------------------------------------
def _f1_exemplar(run_dir: str) -> list[str]:
    """F1: grid heatmaps of the PFN update, the GP update and their difference, per context size.

    One COLUMN PER CONTEXT SIZE (2026-09-30) for one channel, so the reader sees how the update sharpens as
    evidence accumulates; rows are the PFN update, the GP update and their difference. The operating point
    is chosen by ``update_rule.exemplar`` in the config and stored in the ``.npz``, so the subtitle states
    exactly which channel, stress level, context sizes, anchor stratum and surprise magnitude produced the
    panel, and which rule picked the channel. Each panel marks its own anchor: anchors are drawn per
    (context, draw) cell upstream, so they need not coincide across columns.
    """
    path = os.path.join(run_dir, "update_rule_exemplar.npz")
    if not os.path.exists(path):
        return []
    ex = np.load(path)
    shape = tuple(int(v) for v in ex["grid_shape"])
    ch2xy = ex["ch2xy"]
    if len(shape) != 2:
        return []   # 5D condition sets have no map view
    g_model, g_gp = np.atleast_2d(ex["g_model"]), np.atleast_2d(ex["g_gp"])     # [T, N] each
    if not {"context_ts", "context_t"} & set(ex.files):
        raise RuntimeError(
            f"{path} carries no context size (neither 'context_ts' nor 'context_t'), so the exemplar "
            "columns cannot be labelled. It predates the 2026-09-30 exemplar contract: re-run the "
            "analysis (python -m pfns4neurostim mechanism --config "
            "configs/experiment/mechanism_update_rule_nhp.yaml) rather than replotting it."
        )
    context_ts = np.atleast_1d(ex["context_ts"] if "context_ts" in ex.files else ex["context_t"])
    anchors = np.atleast_1d(ex["anchors"] if "anchors" in ex.files else ex["anchor"])
    n_ctx = g_model.shape[0]
    reference = str(ex["reference_gp"]) if "reference_gp" in ex.files else "mll"   # older files predate the key
    row_titles = (arm_style(str(ex["engine"])).label,
                  style.model_label("gp_naive" if reference == "fixed" else "gp_mll"), "Difference")
    fig, axes = style.figure(
        "full", nrows=3, ncols=n_ctx, squeeze=False, panel_aspect=_F1_PANEL_ASPECT,
        gridspec_kw={"wspace": _F1_WSPACE, "hspace": _F1_HSPACE},
    )
    im = None
    for c in range(n_ctx):
        # Each column is normalized by its own peak: the update SHRINKS with context, and a shared scale
        # would render the late columns as empty maps rather than as a sharper, weaker bump.
        maps = []
        for vec in (g_model[c], g_gp[c]):
            grid = np.full(shape, np.nan)
            grid[ch2xy[:, 0], ch2xy[:, 1]] = vec / max(float(np.max(np.abs(vec))), 1e-12)
            maps.append(grid)
        maps.append(maps[0] - maps[1])
        a = int(anchors[min(c, len(anchors) - 1)])
        for r, (grid, row_title) in enumerate(zip(maps, row_titles)):
            ax = axes[r, c]
            im = ax.imshow(grid.T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
            ax.plot(ch2xy[a, 0], ch2xy[a, 1], marker="*", color="k", ms=style.MARKER_SIZE * 2)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"$t$ = {int(context_ts[min(c, len(context_ts) - 1)])}")
            if c == 0:
                ax.set_ylabel(row_title)
    fig.colorbar(im, ax=axes, shrink=0.6, pad=_F1_COLORBAR_PAD, label=style.axis_label("update_normalized"))
    fig.suptitle(_exemplar_provenance(ex), x=0.01, ha="left", fontsize=style.FONT_SIZES["annotation"])
    return style.save_figure(fig, run_dir, "update_exemplar")


def _contexts(ex: Any) -> str:
    """Comma-separated context sizes stored in an exemplar ``.npz``."""
    key = "context_ts" if "context_ts" in getattr(ex, "files", []) else "context_t"
    if key not in getattr(ex, "files", []):
        return "?"
    return ", ".join(str(int(v)) for v in np.atleast_1d(ex[key]))


def _anchors(ex: Any) -> str:
    """Anchor site index of each context column of an exemplar ``.npz`` (``'?'`` when none is stored)."""
    key = "anchors" if "anchors" in getattr(ex, "files", []) else "anchor"
    if key not in getattr(ex, "files", []):
        return "?"
    values = [int(v) for v in np.atleast_1d(ex[key])]
    return str(values[0]) if len(set(values)) == 1 else ", ".join(str(v) for v in values)


def _exemplar_provenance(ex: Any) -> str:
    """One-line description of the exemplar's operating point, for the figure subtitle.

    Args:
        ex: The loaded ``update_rule_exemplar.npz``.

    Returns:
        A caption string; degrades gracefully for an ``.npz`` written before the provenance fields existed.
    """
    def field(name: str, fmt: str = "{}") -> str:
        if name not in getattr(ex, "files", []):
            return "?"
        try:
            return fmt.format(ex[name].item())
        except (ValueError, AttributeError, TypeError):
            return str(ex[name])

    return (
        f"{field('label')} | {field('knob')} level {field('level', '{:g}')} "
        f"({field('achieved_snr_db', '{:.1f}')} dB) | context t = {_contexts(ex)}, "
        f"draw {field('draw')} | anchor {_anchors(ex)} ({field('anchor_stratum')}), "
        f"surprise c = {field('surprise_c', '{:g}')} | channel picked by {field('select_rule')}"
    )


def _f2_shape_vs_context(cell: pd.DataFrame, run_dir: str, gates: list[dict[str, Any]]) -> list[str]:
    """F2: rho_shape vs context size with the anchor-permutation null and the seed floor."""
    fig, ax = style.figure("onehalf")
    reference = str(cell["reference_gp"].iloc[0]) if "reference_gp" in cell and len(cell) else "mll"
    for engine, sub in cell.groupby("engine"):
        if engine == f"gp_{reference}_frozen":
            continue   # identical to the reference by construction
        st = arm_style(str(engine))
        agg = sub.groupby("context_t")["rho_shape_median"].apply(_iqr)
        ts = np.array(agg.index, dtype=float)
        med = np.array([v[0] for v in agg])
        ax.plot(ts, med, color=st.color, ls=st.linestyle, marker=st.marker, label=st.label)
        ax.fill_between(ts, [v[1] for v in agg], [v[2] for v in agg], color=st.color, alpha=style.BAND_ALPHA)
        null = sub.groupby("context_t")["rho_shape_null_median"].apply(_iqr)
        ax.fill_between(ts, [v[1] for v in null], [v[2] for v in null], color=style.NEUTRAL_GREY,
                        alpha=style.BAND_ALPHA, label="Anchor-permutation null")
        sds = [gt["seed_sd_rho_shape_median"] for gt in gates
               if gt.get("gate") == "seed_floor" and gt.get("seed_sd_rho_shape_median") is not None]
        if sds and engine == "tabpfn_v2_5":
            sd = float(np.mean(sds))
            ax.fill_between(ts, med - sd, med + sd, facecolor="none", edgecolor=st.color,
                            hatch="///", lw=0, label="Seed floor (±1 SD)")
    ax.set_xlabel(style.axis_label("context_t"))
    ax.set_ylabel(style.axis_label("rho_shape"))
    ax.legend(loc="best")
    return style.save_figure(fig, run_dir, "shape_vs_context")


def _f3_surprise(surprise: pd.DataFrame, run_dir: str) -> list[str]:
    """F3: update at the anchor vs surprise, one panel per stress level (H10.2)."""
    levels = sorted(surprise["level"].unique())
    fig, axes = style.figure("wide", ncols=len(levels), sharey=True)
    axes = np.atleast_1d(axes)
    df = surprise.assign(
        d_norm=surprise["d_anchor"] / surprise["base_sd_anchor"],
        d_gp_norm=surprise["d_gp_anchor"] / surprise["base_sd_anchor"],
    )
    for ax, level in zip(axes, levels):
        sub = df[df["level"] == level]
        for engine, e in sub.groupby("engine"):
            st = arm_style(str(engine))
            agg = e.groupby("c")["d_norm"].median()
            ax.plot(agg.index, agg.values, color=st.color, ls=st.linestyle, marker=st.marker, label=st.label)
        ref = sub.groupby("c")["d_gp_norm"].median()
        ax.plot(ref.index, ref.values, color=style.NEUTRAL_GREY, ls="--", label="GP update (linear)")
        knob = str(sub["knob"].iloc[0]) if "knob" in sub and len(sub) else ""
        ax.set_title(f"{style.KNOB_LABELS.get(knob, knob)} level {level:g}")
        ax.set_xlabel(style.axis_label("surprise_c"))
    axes[0].set_ylabel(style.axis_label("update_at_anchor"))
    axes[0].legend(loc="best")
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "surprise_response")


def _f5_kernel_properties(cell: pd.DataFrame, run_dir: str) -> list[str]:
    """F5: implied-kernel properties per engine (symmetry, PSD, stationarity)."""
    metrics = [m for m in ("sym_err", "psd_neg_mass", "ell_cv_anchor") if m in cell]
    fig, axes = style.figure("wide", ncols=len(metrics))
    axes = np.atleast_1d(axes)
    engines = sorted(cell["engine"].unique())
    for ax, metric in zip(axes, metrics):
        for i, engine in enumerate(engines):
            vals = cell.loc[cell["engine"] == engine, metric].dropna().to_numpy()
            st = arm_style(str(engine))
            jitter = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else np.zeros(len(vals))
            ax.scatter(i + jitter, vals, s=style.MARKER_SIZE ** 2, color=st.color, marker=st.marker, linewidths=0)
            if len(vals):
                ax.hlines(np.median(vals), i - 0.25, i + 0.25, color="k", lw=style.LINE_WIDTH)
        ax.set_xticks(range(len(engines)))
        ax.set_xticklabels([arm_style(str(e)).label for e in engines], rotation=30, ha="right")
        ax.set_ylabel(style.axis_label(metric))
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "kernel_properties")


def render_update_rule(run_dir: str) -> list[str]:
    """Every M10 figure from ``update_rule*.csv`` (+ ``gates.json`` for the seed floor).

    F4 (``lengthscale_vs_snr``, the implicit-kernel lengthscale against achieved SNR) was dropped on 2026-10-07
    (user); ``ell_hat`` stays in ``update_rule.csv`` / ``update_rule_cell.csv`` and in the F7 supplement.

    Args:
        run_dir: Update-rule run directory.

    Returns:
        Written paths.
    """
    paths = {k: os.path.join(run_dir, f"update_rule{k}.csv") for k in ("_cell", "_surprise")}
    if not all(os.path.exists(p) for p in paths.values()):
        return []
    cell, surprise = (pd.read_csv(paths[k]) for k in ("_cell", "_surprise"))
    gates_path = os.path.join(run_dir, "gates.json")
    gates = json.load(open(gates_path, encoding="utf-8")) if os.path.exists(gates_path) else []
    written = _f1_exemplar(run_dir)
    written += _f2_shape_vs_context(cell, run_dir, gates)
    written += _f3_surprise(surprise, run_dir)
    written += _f5_kernel_properties(cell, run_dir)
    layers_path = os.path.join(run_dir, "update_rule_layers.csv")
    if os.path.exists(layers_path):
        written += _f6_layer_alignment(pd.read_csv(layers_path), run_dir)
    spec = _profile_spec(run_dir)
    profile_path = os.path.join(run_dir, "update_rule_profile.csv")
    if spec is not None and os.path.exists(profile_path):
        written += _f7_update_vs_distance(pd.read_csv(profile_path), cell, gates, spec, run_dir)
        written += _f7_controls(run_dir, cell, gates, spec)
    return written


def _f6_layer_alignment(layers: pd.DataFrame, run_dir: str) -> list[str]:
    """F6 (secondary): layer alignment and probe R^2 vs layer, ONE TRACE PER CONTEXT SIZE.

    Widened to two full panels on 2026-09-27 and split by context size on 2026-09-30: the layer profile is a
    property of the surrogate *given its evidence*. Contexts are shades of the TabPFN anchor, dark = smallest; the
    band is the IQR over channels, i.e. between-channel spread, not estimator noise. The representation is the
    feature-token mean (the label-token readout was dropped on 2026-10-07).

    Args:
        layers: ``update_rule_layers.csv``.
        run_dir: Run directory.

    Returns:
        Written paths.
    """
    fig, axes = style.figure("wide", ncols=2)
    st = arm_style("tabpfn_v2_5")
    contexts = sorted(layers["context_t"].unique())
    for ax, col in zip(axes, ("layer_corr", "layer_probe_r2")):
        for i, t in enumerate(contexts):
            color = st.color if len(contexts) == 1 else style.derive_shade(st.color, i, len(contexts))
            agg = layers[layers["context_t"] == t].groupby("layer")[col]
            med = agg.median()
            ax.plot(med.index, med.values, color=color, marker=st.marker, label=f"$t$ = {int(t)}")
            ax.fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values,
                            color=color, alpha=style.BAND_ALPHA, lw=0)
        ax.set_xlabel(style.axis_label("layer"))
        style.set_integer_xaxis(ax)
        ax.set_ylabel(style.axis_label(col))
    peaks = layers[layers["is_peak"].astype(bool)]["layer"]
    if len(peaks):
        axes[0].plot(peaks.value_counts().index, np.zeros(peaks.nunique()), "|", color=style.NEUTRAL_GREY,
                     ms=style.MARKER_SIZE * 3)
    axes[0].legend(loc="best", title=style.axis_label("context_t"),
                   fontsize=style.FONT_SIZES["legend"], title_fontsize=style.FONT_SIZES["legend"])
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_alignment")


# ---------------------------------------------------------------------------
# F7 update_vs_distance (B2 Step 11, 2026-10-06)
# ---------------------------------------------------------------------------
#: Engines F7 draws, in legend order; GP-fixed and GP-MLL-frozen are the two reference GPs.
_F7_ENGINES: tuple[str, ...] = ("tabpfn_v2_5", "gp_fixed_frozen", "gp_mll_frozen")
#: Opacity of the GP that is drawn but is not the stated comparator, so the headline GP reads first.
_F7_SECONDARY_ALPHA: float = 0.55
#: Height / width of an F7 panel (three panels side by side at the ``full`` analysis width).
_F7_PANEL_ASPECT: float = 0.8
#: Horizontal jitter of the per-channel points of the far-field panel, in units of the ladder step.
_F7_JITTER: float = 0.06
#: Lower limit of F7 panel (a): transfers below 1e-4 of the injected surprise are indistinguishable from no update.
_F7_TRANSFER_FLOOR: float = 1e-4
#: Linear range of the off-map panel's symlog axis: transfers below 1e-4 of the injected surprise read as ~0.
_F7_SYMLOG_THRESHOLD: float = 1e-4


def _profile_spec(run_dir: str) -> dict[str, Any] | None:
    """The ``update_rule.profile`` block of a run's resolved ``config.yaml`` (``None`` if absent or disabled)."""
    import yaml  # noqa: PLC0415

    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    spec = (doc.get("update_rule") or {}).get("profile")
    return spec if spec and spec.get("enabled") else None


def _f7_alpha(engine: str, headline: str) -> float:
    """Full opacity for TabPFN and the headline GP, reduced for the other GP."""
    return 1.0 if engine in ("tabpfn_v2_5", headline) else _F7_SECONDARY_ALPHA


def _per_channel(df: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame:
    """Median of ``col`` within each channel (over draws, levels already filtered), keeping ``keys``."""
    return df.groupby(["subject", "emg", *keys])[col].median().reset_index()


def _f7_update_vs_distance(
    profile: pd.DataFrame,
    cell: pd.DataFrame,
    gates: list[dict[str, Any]],
    spec: dict[str, Any],
    run_dir: str,
) -> list[str]:
    """F7: update magnitude against topographic distance, TabPFN vs GP-fixed vs GP-MLL.

    Layout of the 2026-10-02 proposal preview, restored as the default on 2026-10-07 (user):

    * one panel per context size in ``spec['panel_ts']`` (default: the ladder's first and middle sizes, t = 10 and
      50 on NHP): the radial transfer profile ``median |T|`` (update divided by the injected surprise) against the
      distance from the anchor in electrode pitches, unobserved sites only, log y, the anchor (d = 0) as its own
      large marker and the far field (d >= ``far_pitch``) shaded; the radial panels share their y-axis;
    * a last panel with the cumulative share of update energy within radius d for EVERY context size of the ladder,
      shaded dark (smallest t) to light, against the uniform-field reference (energy proportional to site count).

    TabPFN and both reference GPs are drawn (user, 2026-10-07): GP-fixed, the stated comparator (``spec['headline_gp']``,
    whose difference to TabPFN the supplement prints), and the converged GP-MLL (frozen at its fit on the context).
    The far-field share against context size is in the supplement (:func:`_f7_controls`). The unit of analysis is the
    channel: median over anchors and draws within a channel, then the median (and IQR band) across channels.

    Args:
        profile: ``update_rule_profile.csv``.
        cell: ``update_rule_cell.csv`` (unused here; kept so both F7 figures share one call signature).
        gates: ``gates.json`` (unused here, see ``cell``).
        spec: The run's resolved ``update_rule.profile`` block.
        run_dir: Run directory.

    Returns:
        Written paths.
    """
    level = float(spec["level"])
    headline = str(spec["headline_gp"])
    far = float(spec["far_pitch"])
    prof = profile[(profile["level"] == level) & (~profile["observed"].astype(bool))]
    engines = [e for e in _F7_ENGINES if e in set(prof["engine"])]
    panel_ts = [int(t) for t in spec["panel_ts"] if t in set(prof["context_t"])]
    if prof.empty or not engines or not panel_ts:
        return []
    fig, axes = style.figure("full", ncols=len(panel_ts) + 1, panel_aspect=_F7_PANEL_ASPECT)
    axes = np.atleast_1d(axes)

    # Radial profile, one panel per context size.
    lowest = np.inf
    for ax, t in zip(axes, panel_ts):
        for engine in engines:
            st = arm_style(engine)
            sub = prof[(prof["engine"] == engine) & (prof["context_t"] == t)]
            pc = sub.groupby(["subject", "emg", "bin"]).agg(d=("distance_mean", "median"),
                                                            v=("transfer_abs_median", "median")).reset_index()
            agg = pc.groupby("bin").agg(d=("d", "median"), med=("v", "median"),
                                        q1=("v", lambda s: s.quantile(0.25)), q3=("v", lambda s: s.quantile(0.75)))
            ring = agg[agg["d"] > 0]
            positive = ring["q1"][ring["q1"] > 0]
            if len(positive):
                lowest = min(lowest, float(positive.min()))
            ax.plot(ring["d"], ring["med"], color=st.color, ls=st.linestyle, marker=st.marker, label=st.label)
            ax.fill_between(ring["d"], ring["q1"], ring["q3"], color=st.color, alpha=style.BAND_ALPHA, lw=0)
            at = agg[agg["d"] == 0]
            if len(at):
                ax.plot([0.0], at["med"].to_numpy()[:1], marker=st.marker, color=st.color,
                        ms=style.MARKER_SIZE * 2.5, ls="none")
        ax.axvspan(far, prof["distance_mean"].max() * 1.05, color=style.NEUTRAL_GREY, alpha=style.BAND_ALPHA, lw=0)
        ax.set_yscale("log")
        ax.set_title(f"$t$ = {t}")
        ax.set_xlabel(style.axis_label("distance_pitch"))
        if ax is not axes[0]:
            ax.sharey(axes[0])
    # The shared axis stops half a decade below the lowest band, and never below the floor: a converged GP's far field
    # decays to ~1e-13 of the surprise, and below the floor every engine reads as "no update".
    axes[0].set_ylim(bottom=max(_F7_TRANSFER_FLOOR, lowest / np.sqrt(10.0)) if np.isfinite(lowest) else _F7_TRANSFER_FLOOR)
    axes[0].set_ylabel(style.axis_label("transfer_abs"))
    axes[0].legend(loc="lower left", fontsize=style.FONT_SIZES["legend"])

    # Cumulative energy within radius d, every context size of the ladder (dark = smallest t).
    ax = axes[-1]
    ladder = sorted(prof["context_t"].unique())
    for engine in engines:
        st = arm_style(engine)
        for i, t in enumerate(ladder):
            e = prof[(prof["engine"] == engine) & (prof["context_t"] == t)]
            if e.empty:
                continue
            color = st.color if len(ladder) == 1 else style.derive_shade(st.color, i, len(ladder))
            pc = e.groupby(["subject", "emg", "bin"]).agg(d=("distance_mean", "median"),
                                                          v=("cum_energy_share", "median")).reset_index()
            agg = pc.groupby("bin").agg(d=("d", "median"), med=("v", "median"))
            ax.plot(agg["d"], agg["med"], color=color, ls=st.linestyle, drawstyle="steps-post",
                    lw=style.LINE_WIDTH * (1.5 if i == 0 else 1.0),
                    label=st.label if i == 0 else None)
    uni = (prof[prof["context_t"] == ladder[0]].groupby("bin")
           .agg(d=("distance_mean", "median"), f=("cum_site_share", "median")))
    ax.plot(uni["d"], uni["f"], color=style.OUTLINE_COLOR, ls=":", lw=style.LINE_WIDTH, drawstyle="steps-post",
            label="Uniform field")
    ax.set_ylim(0.0, 1.03)
    ax.set_title(f"$t$ = {', '.join(str(int(t)) for t in ladder)} (dark to light)")
    ax.set_xlabel(style.axis_label("distance_pitch"))
    ax.set_ylabel(style.axis_label("cum_energy_share"))
    ax.legend(loc="lower right", fontsize=style.FONT_SIZES["legend"])
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "update_vs_distance")


def _f7_far_field_panel(ax: Any, cell: pd.DataFrame, gates: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    """Far-field energy share against context size, all engines (F7 supplement panel).

    Per-channel points, the median per engine, TabPFN's seed floor (+-1 SD over inference seeds) hatched, and
    TabPFN's median paired difference against the headline GP at the smallest context size.

    Args:
        ax: Target axes.
        cell: ``update_rule_cell.csv``.
        gates: ``gates.json`` (seed floor of the far-field share).
        spec: The run's resolved ``update_rule.profile`` block.
    """
    level = float(spec["level"])
    headline = str(spec["headline_gp"])
    far = float(spec["far_pitch"])
    cl = cell[(cell["level"] == level)] if "far_field_share" in cell else cell.iloc[0:0]
    engines = [e for e in _F7_ENGINES if e in set(cl["engine"])]
    ladder = sorted(cl["context_t"].unique()) if len(cl) else []
    step = float(np.min(np.diff(ladder))) if len(ladder) > 1 else 1.0
    med_by: dict[str, pd.Series] = {}
    for k, engine in enumerate(engines):
        st = arm_style(engine)
        pc = _per_channel(cl[cl["engine"] == engine], ["context_t"], "far_field_share")
        if pc.empty:
            continue
        offset = (k - (len(engines) - 1) / 2) * _F7_JITTER * step
        alpha = _f7_alpha(engine, headline)
        ax.scatter(pc["context_t"] + offset, pc["far_field_share"], s=style.MARKER_SIZE ** 2, color=st.color,
                   marker=st.marker, alpha=style.FAINT_ALPHA * 2 * alpha, linewidths=0)
        med = pc.groupby("context_t")["far_field_share"].median()
        med_by[engine] = med
        ax.plot(med.index, med.values, color=st.color, ls=st.linestyle, marker=st.marker, alpha=alpha,
                label=st.label)
    sds = [g.get("seed_sd_far_field_share") for g in gates
           if g.get("gate") == "seed_floor" and g.get("seed_sd_far_field_share") is not None]
    if sds and "tabpfn_v2_5" in med_by:
        sd = float(np.mean(sds))
        m = med_by["tabpfn_v2_5"]
        ax.fill_between(m.index, m.values - sd, m.values + sd, facecolor="none",
                        edgecolor=arm_style("tabpfn_v2_5").color, hatch="///", lw=0, label="Seed floor (±1 SD)")
    if "tabpfn_v2_5" in med_by and headline in med_by:
        diff = _per_channel(cl, ["context_t", "engine"], "far_field_share").pivot_table(
            index=["subject", "emg", "context_t"], columns="engine", values="far_field_share").dropna(
            subset=["tabpfn_v2_5", headline])
        if len(diff):
            t0 = int(min(diff.index.get_level_values("context_t")))
            d0 = (diff["tabpfn_v2_5"] - diff[headline]).xs(t0, level="context_t").median()
            ax.set_title(f"TabPFN − {arm_style(headline).label} at $t$ = {t0}: {d0:+.2f}",
                         fontsize=style.FONT_SIZES["annotation"])
    ax.set_xticks(ladder, [str(int(t)) for t in ladder])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(style.axis_label("context_t"))
    ax.set_ylabel(style.axis_label("far_field_share") + f" ($d \\geq {far:g}$)")
    ax.legend(loc="upper right", fontsize=style.FONT_SIZES["legend"] - 1)


def _radial(ax: Any, rows: pd.DataFrame, engine: str, *, marker_face: str = "full", label: str | None = None) -> None:
    """One radial ``median |T|`` trace of the F7 supplement (median over channels / cells per bin)."""
    st = style.TRUE_GP_STYLE if engine == "truth" else arm_style(engine)
    agg = rows.groupby("bin").agg(d=("distance_mean", "median"), v=("transfer_abs_median", "median"))
    ax.plot(agg["d"], agg["v"], color=st.color, ls=st.linestyle, marker=st.marker, fillstyle=marker_face,
            label=label or st.label)


def _f7_controls(run_dir: str, cell: pd.DataFrame, gates: list[dict[str, Any]], spec: dict[str, Any]) -> list[str]:
    """F7 supplement ``update_vs_distance_controls``: the five controls of B2 Step 11, plus the far-field share.

    (1) off-map anchor: median ``|T|`` on the grid for an observation placed off the map, beside each engine's
    on-grid far-field ``|T|`` (a GP's is ~0; a level shift is not); TabPFN's input-space separation of the
    off-map point is printed, since a quantile transform can fold it onto the grid edge. (2) offset
    decomposition: ``ell_hat`` against the offset share of the energy, per cell -- a rising cloud means a
    uniform shift inflates ``ell_hat``. (3) the profile on the known-GP synthetic channel, with the true GP's
    (analytic) profile. (4) shuffled coordinates: profiles in the true geometry go flat. (5) far-field / anchor
    transfer against the surprise magnitude (a level shift grows linearly while the anchor saturates). (6) the
    far-field energy share against context size for every engine, with TabPFN's seed floor (the main figure's third
    panel until the preview layout was restored on 2026-10-07).

    Args:
        run_dir: Run directory.
        cell: ``update_rule_cell.csv``.
        gates: ``gates.json`` (seed floor of the far-field share).
        spec: The run's ``update_rule.profile`` block.

    Returns:
        Written paths.
    """
    level = float(spec["level"])
    paths = {k: os.path.join(run_dir, f"update_rule_{k}.csv") for k in ("offmap", "profile_controls", "profile_c")}
    have = {k: _has_rows(v) for k, v in paths.items()}
    if not any(have.values()):
        return []
    fig, axes = style.figure("full", nrows=2, ncols=3, panel_aspect=_F7_PANEL_ASPECT)
    axes = axes.ravel()
    # (1) off-map anchor
    ax = axes[0]
    if have["offmap"]:
        off = pd.read_csv(paths["offmap"])
        cl = cell[cell["level"] == level] if "transfer_far_median" in cell else cell.iloc[0:0]
        engines = [e for e in _F7_ENGINES if e in set(off["engine"])]
        for i, engine in enumerate(engines):
            st = arm_style(engine)
            v = _per_channel(off[off["engine"] == engine], [], "offmap_transfer_median")["offmap_transfer_median"]
            ax.scatter(np.full(len(v), i - 0.12), v, s=style.MARKER_SIZE ** 2, color=st.color, marker=st.marker,
                       linewidths=0, label="Off-map anchor" if i == 0 else None)
            f = _per_channel(cl[cl["engine"] == engine], [], "transfer_far_median")["transfer_far_median"]
            ax.scatter(np.full(len(f), i + 0.12), f, s=style.MARKER_SIZE ** 2, facecolors="none",
                       edgecolors=st.color, marker=st.marker, label="On-grid anchor, far field" if i == 0 else None)
        ax.set_xticks(range(len(engines)), [arm_style(e).label for e in engines])
        # symlog, not log: a GP's off-map transfer is ~0 (that is the point of the control) and must stay visible.
        ax.set_yscale("symlog", linthresh=_F7_SYMLOG_THRESHOLD)
        ax.set_ylabel(style.axis_label("offmap_transfer"))
        sep = off[off["engine"] == "tabpfn_v2_5"]["offmap_separation"]
        if len(sep):
            ax.set_title(f"TabPFN input-space separation {sep.median():.2f}", fontsize=style.FONT_SIZES["annotation"])
        ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    # (2) offset decomposition vs ell_hat
    ax = axes[1]
    if {"offset_share", "ell_hat_median"} <= set(cell.columns):
        cl = cell[cell["level"] == level]
        for engine in [e for e in _F7_ENGINES if e in set(cl["engine"])]:
            st = arm_style(engine)
            e = cl[cl["engine"] == engine]
            ax.scatter(e["offset_share"], e["ell_hat_median"], s=style.MARKER_SIZE ** 2, color=st.color,
                       marker=st.marker, alpha=0.6, linewidths=0, label=st.label)
        ax.set_xlabel(style.axis_label("offset_share"))
        ax.set_ylabel(style.axis_label("ell_hat"))
        ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    ctrl = pd.read_csv(paths["profile_controls"]) if have["profile_controls"] else pd.DataFrame()
    if not ctrl.empty:
        ctrl = ctrl[~ctrl["observed"].astype(bool)]
    # (3) known-GP channel
    ax = axes[2]
    known = ctrl[ctrl["control"] == "known_gp"] if not ctrl.empty else ctrl
    for engine in ["truth", *[e for e in _F7_ENGINES if not known.empty and e in set(known["engine"])]]:
        if not known.empty and engine in set(known["engine"]):
            _radial(ax, known[known["engine"] == engine], engine)
    ax.set_yscale("log")
    ax.set_title("Known-GP channel", fontsize=style.FONT_SIZES["annotation"])
    ax.set_xlabel(style.axis_label("distance_pitch"))
    ax.set_ylabel(style.axis_label("transfer_abs"))
    if not known.empty:
        ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    # (4) shuffled coordinates
    ax = axes[3]
    sh = ctrl[ctrl["control"].isin(["original", "shuffled"])] if not ctrl.empty else ctrl
    for engine in [e for e in _F7_ENGINES if not sh.empty and e in set(sh["engine"])]:
        for control, face in (("original", "full"), ("shuffled", "none")):
            part = sh[(sh["engine"] == engine) & (sh["control"] == control)]
            if len(part):
                _radial(ax, part, engine, marker_face=face, label=f"{arm_style(engine).label}, {control}")
    ax.set_yscale("log")
    ax.set_title("Shuffled coordinates", fontsize=style.FONT_SIZES["annotation"])
    ax.set_xlabel(style.axis_label("distance_pitch"))
    if not sh.empty:
        ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 2)
    # (5) far/peak ratio vs surprise magnitude
    ax = axes[4]
    if have["profile_c"]:
        pcf = pd.read_csv(paths["profile_c"])
        pcf = pcf[pcf["level"] == level]
        for engine in [e for e in _F7_ENGINES if e in set(pcf["engine"])]:
            st = arm_style(engine)
            agg = _per_channel(pcf[pcf["engine"] == engine], ["surprise_abs"], "far_peak_ratio").groupby(
                "surprise_abs")["far_peak_ratio"]
            med = agg.median()
            ax.plot(med.index, med.values, color=st.color, ls=st.linestyle, marker=st.marker, label=st.label)
            ax.fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values, color=st.color,
                            alpha=style.BAND_ALPHA, lw=0)
        ax.set_xlabel(style.axis_label("surprise_abs"))
        ax.set_ylabel(style.axis_label("far_peak_ratio"))
        ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    # (6) far-field share vs context size
    _f7_far_field_panel(axes[5], cell, gates, spec)
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "update_vs_distance_controls")


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
            ax.plot(agg.index, agg.values, color=color, marker="o", label=f"{level:g}")
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
                        label=f"level {level:g}")
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


def _cka_placement_figures(pl: pd.DataFrame, run_dir: str) -> list[str]:
    """CKA (b): placement of the feature-token mean vs layer (one trace per context size), and its surface."""
    # Layers whose floor/ceiling gap is degenerate are DROPPED, not clipped: placement is
    # (d - floor) / (ceiling - floor), so a ~0 (or negative) denominator makes the ratio explode and
    # inverts its sign (decision 2026-09-27). The excluded layers are named on the figure instead.
    usable = pl[pl["gap_ok"].astype(bool)]
    dropped = sorted(set(pl["layer"]) - set(usable["layer"]))
    if usable.empty:
        return []
    contexts = sorted(usable["context_t"].unique())
    base = style.model_color("tabpfn_v2_5")
    fig, ax = style.figure("onehalf")
    _reference_bands(ax, "h", label_it=True)
    for i, t in enumerate(contexts):
        # One trace per context size (2026-09-30): placement is conditioned on the evidence the embedding saw.
        color = base if len(contexts) == 1 else style.derive_shade(base, i, len(contexts))
        per_channel = (usable[usable["context_t"] == t]
                       .groupby(["subject", "emg", "layer"])["p"].mean().reset_index())
        agg = per_channel.groupby("layer")["p"]
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
    ax.set_ylabel(style.axis_label("placement"))
    ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    written = style.save_figure(fig, run_dir, "cka_placement")
    return written + _cka_placement_surface(usable, dropped, run_dir)


def _cka_placement_surface(usable: pd.DataFrame, dropped: list[int], run_dir: str) -> list[str]:
    """Layer x context-size heatmap of the CKA placement p of the feature-token mean (2026-09-30).

    The flat view of the layer / evidence / placement volume: a 3D surface of the same numbers reads worse in print
    (occlusion, one fixed viewing angle, no way to read a value off it), so the third axis is colour. The colormap is
    the package's sequential map (:data:`style.SEQUENTIAL_CMAP`, viridis) since 2026-10-07 (user): the earlier
    two-stop ramp between the crimson / pink reference tokens had too little lightness range to read values off.
    The scale is fixed to [0, 1] -- 0 = prior floor, 1 = noise ceiling, both named on the colorbar -- and values
    outside it saturate, shown by the colorbar's extension arrows, rather than redefining the scale.

    Args:
        usable: Rows of ``cka_placement.csv`` with a usable floor/ceiling gap.
        dropped: Layers excluded for a degenerate gap, named on the figure.
        run_dir: Run directory.

    Returns:
        Written paths.
    """
    grid = usable.pivot_table(index="context_t", columns="layer", values="p", aggfunc="median")
    if grid.empty or grid.shape[0] < 2:
        return []   # a single context size is the line figure, not a surface
    fig, ax = style.figure("onehalf")
    # Context sizes are rows of EQUAL height (the ladder is not evenly spaced, so a numeric axis would let the
    # widest gap dominate the picture); cells without a usable gap at that (layer, t) stay blank.
    rows = np.arange(len(grid.index))                                        # [T]
    im = ax.pcolormesh(grid.columns.to_numpy(), rows, grid.to_numpy(),
                       cmap=style.SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, shading="nearest")
    ax.set_xlabel(style.axis_label("layer"))
    style.set_integer_xaxis(ax)
    ax.set_ylabel(style.axis_label("context_t"))
    ax.set_yticks(rows, [str(int(t)) for t in grid.index])
    cb = fig.colorbar(im, ax=ax, label=style.axis_label("placement"), extend="both")
    cb.set_ticks([0.0, 0.5, 1.0], labels=["0 (prior floor)", "0.5", "1 (noise ceiling)"])
    notes = [f"layers omitted: {', '.join(str(d) for d in dropped)}"] if dropped else []
    if grid.isna().to_numpy().any():
        notes.append("blank: no usable prior/noise gap")
    if notes:
        ax.text(0.98, 1.02, "; ".join(notes), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=style.FONT_SIZES["annotation"], alpha=0.8)
    return style.save_figure(fig, run_dir, "cka_placement_surface")
