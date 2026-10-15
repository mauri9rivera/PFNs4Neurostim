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
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style

__all__ = [
    "arm_style",
    "render_update_rule",
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
    snr_labels = _level_snr_labels(surprise)
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
        ax.set_title(f"{style.KNOB_LABELS.get(knob, knob)}, {snr_labels.get(float(level), f'{level:g}')}")
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


def render_layer_dynamics(run_dir: str) -> list[str]:
    """F6-0/F6-1 preview: per-block alignment, common-mode share and the rebuilt ridge probe.

    First-pass figure for :mod:`pfns4neurostim.experiments.mechanism_layers` -- a working preview
    of A2/A3 while the Phase-0 gate is open, not the redesigned, captioned F6 of F6-4 (G6: no
    layer-wise claim is drawn from this before that item lands).

    Args:
        run_dir: Layer-dynamics run directory.

    Returns:
        Written paths.
    """
    align_path = os.path.join(run_dir, "layer_dynamics.csv")
    probe_path = os.path.join(run_dir, "layer_probe.csv")
    written: list[str] = []
    if os.path.exists(align_path) and _has_rows(align_path):
        align = pd.read_csv(align_path)
        written += _layer_dynamics_alignment(align, run_dir)
        written += _layer_dynamics_geometry(align, run_dir)
        written += _layer_dynamics_robust(align, run_dir)
        written += _layer_dynamics_direction(align, run_dir)
    if os.path.exists(probe_path) and _has_rows(probe_path):
        written += _layer_dynamics_probe(pd.read_csv(probe_path), run_dir)
    for name, fn in (("layer_precision", _layer_dynamics_precision),
                     ("layer_causal", _layer_dynamics_causal)):
        path = os.path.join(run_dir, f"{name}.csv")
        if os.path.exists(path) and _has_rows(path):
            written += fn(pd.read_csv(path), run_dir)
    return written


#: Head stages appended to the layer axis after block 17, in order, behind a visual break. They are
#: deterministic functions of the block-17 label token, so they are positions on the depth axis
#: rather than further blocks; H2 is the first state not under a parameter-free LayerNorm.
HEAD_STAGES: tuple[str, ...] = ("H1", "H2", "H3")

#: Gap (in x units) between block 17 and the first head stage, where the axis break is drawn.
HEAD_BREAK_GAP: float = 1.6


def _stage_positions(stages: Sequence[str]) -> dict[str, float]:
    """x position of every stage: blocks at their own index, head stages past a gap."""
    pos = {s: float(s[len("block"):]) for s in stages if s.startswith("block")}
    last = max(pos.values(), default=-1.0)
    for i, name in enumerate(h for h in HEAD_STAGES if h in stages):
        pos[name] = last + HEAD_BREAK_GAP + i
    return pos


def _layer_dynamics_alignment(align: pd.DataFrame, run_dir: str) -> list[str]:
    """F6 two-curve figure: representation alignment per context size, plus the output-space curve.

    Panel (a) carries both curves on one y-axis, because both are the same statistic -- a
    correlation across sites against the reference GP's update -- read off two different things:

    * **representation geometry** (solid, one shade per context size): the size of the change in the
      feature-token mean. Defined at every block and, past the break, at the three head stages.
    * **output space** (dashed, the largest context only): the change in the prediction the network
      *would* make if it stopped at that block, read through the checkpoint's own decoder with zero
      fitted parameters. Drawn for one context only because two curves per context size is
      unreadable (user decision 2026-10-12); at block 17 it equals the real prediction.

    Panel (b) is the share of each stage's change carried by a site-independent shift.

    Args:
        align: ``layer_dynamics.csv``.
        run_dir: Run directory.

    Returns:
        Written paths.
    """
    fig, axes = style.figure("wide", ncols=2)
    st = arm_style("tabpfn_v2_5")
    contexts = sorted(align["context_t"].unique())
    stages = [s for s in align["stage"].unique()] if "stage" in align.columns else []
    pos = _stage_positions(stages) if stages else {}

    def _x(frame: pd.DataFrame) -> pd.Series:
        return frame["stage"].map(pos) if pos else frame["layer"].astype(float)

    for ax, col in zip(axes, ("pearson", "common_mode_share")):
        for i, t in enumerate(contexts):
            color = st.color if len(contexts) == 1 else style.derive_shade(st.color, i, len(contexts))
            sub = align[align["context_t"] == t].copy()
            sub["_x"] = _x(sub)
            agg = sub.groupby("_x")[col]
            med = agg.median()
            ax.plot(med.index, med.values, color=color, marker=st.marker, label=f"$t$ = {int(t)}")
            ax.fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values,
                            color=color, alpha=style.BAND_ALPHA, lw=0)
        ax.set_xlabel(style.axis_label("layer"))
        ax.set_ylabel("Pearson $r$ with $|\\Delta_{GP}|$" if col == "pearson" else "Common-mode share")

    # The output-space curve: largest context only, dashed, same axis as the representation curve.
    if "lens_pearson" in align.columns and contexts:
        t_max = max(contexts)
        sub = align[(align["context_t"] == t_max) & align["lens_pearson"].notna()].copy()
        if len(sub):
            sub["_x"] = _x(sub)
            med = sub.groupby("_x")["lens_pearson"].median()
            axes[0].plot(med.index, med.values, color=st.color, ls="--", lw=1.4, marker="s",
                         ms=style.MARKER_SIZE, mfc="none",
                         label=f"output space, $t$ = {int(t_max)}")

    # Mark the break and name the head stages.
    head_pos = [pos[h] for h in HEAD_STAGES if h in pos]
    for ax in axes:
        if head_pos:
            ax.axvline(min(head_pos) - HEAD_BREAK_GAP / 2.0, color=style.NEUTRAL_GREY, lw=0.8, ls=":")
            block_ticks = [v for k, v in pos.items() if k.startswith("block") and int(k[5:]) % 4 == 0]
            ax.set_xticks(block_ticks + head_pos)
            ax.set_xticklabels([f"{int(v)}" for v in block_ticks] + [h for h in HEAD_STAGES if h in pos])
        else:
            style.set_integer_xaxis(ax)
    axes[0].legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_dynamics_alignment")


def _layer_dynamics_probe(probe: pd.DataFrame, run_dir: str) -> list[str]:
    """Rebuilt ridge-probe R^2 vs block: penalty sensitivity, and random vs spatial fold schemes.

    The spatial arm holds out contiguous grid patches instead of random sites. Because the
    representation change is spatially smooth, a random split lets a weakly penalised probe
    interpolate between neighbours; the gap between the two schemes is the size of that effect.
    """
    fig, axes = style.figure("wide", ncols=2)
    st = arm_style("tabpfn_v2_5")
    has_scheme = "fold_scheme" in probe.columns
    rnd = probe[probe["fold_scheme"] == "random"] if has_scheme else probe
    lambdas = sorted(rnd["lambda_rel"].unique())
    for i, lam in enumerate(lambdas):
        color = st.color if len(lambdas) == 1 else style.derive_shade(st.color, i, len(lambdas))
        agg = rnd[rnd["lambda_rel"] == lam].groupby("layer")["ridge_r2"]
        med = agg.median()
        axes[0].plot(med.index, med.values, color=color, marker=st.marker,
                     label=f"$\\lambda_{{rel}}$ = {lam:g}")
        axes[0].fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values,
                             color=color, alpha=style.BAND_ALPHA, lw=0)
    axes[0].legend(loc="best", fontsize=style.FONT_SIZES["legend"], title="random folds",
                   title_fontsize=style.FONT_SIZES["legend"])

    if has_scheme and (probe["fold_scheme"] == "spatial").any():
        lam_s = float(probe[probe["fold_scheme"] == "spatial"]["lambda_rel"].iloc[0])
        for scheme, ls, label in (("random", "-", "random sites"), ("spatial", "--", "spatial patches")):
            sub = probe[(probe["fold_scheme"] == scheme) & np.isclose(probe["lambda_rel"], lam_s)]
            if not len(sub):
                continue
            agg = sub.groupby("layer")["ridge_r2"]
            med = agg.median()
            axes[1].plot(med.index, med.values, color=st.color, ls=ls, marker=st.marker,
                         mfc="none" if scheme == "spatial" else st.color, label=label)
            axes[1].fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values,
                                 color=st.color, alpha=style.BAND_ALPHA, lw=0)
        axes[1].legend(loc="best", fontsize=style.FONT_SIZES["legend"],
                       title=f"held-out groups ($\\lambda_{{rel}}$ = {lam_s:g})",
                       title_fontsize=style.FONT_SIZES["legend"])
    for ax in axes:
        ax.set_xlabel(style.axis_label("layer"))
        style.set_integer_xaxis(ax)
        ax.set_ylabel("ridge $R^2$ (rebuilt probe)")
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_dynamics_probe")


def _layer_dynamics_geometry(align: pd.DataFrame, run_dir: str) -> list[str]:
    """The geometry test: does the conditioned kernel explain anything the prior kernel does not?

    ``|Delta_GP|`` is proportional, across sites, to the *conditioned* kernel slice, so the headline
    alignment cannot by itself separate GP-like updating from similarity smearing. Panel (a) shows
    the plain correlations against the conditioned and the prior kernel together; panel (b) shows the
    partial correlation with the conditioned kernel once the prior one is partialled out, which is
    the quantity that carries the Bayesian content.
    """
    cols = ("pearson", "pearson_prior_kernel", "partial_r_gp_given_prior")
    if not all(c in align.columns for c in cols):
        return []
    blocks = align[align["stage"].str.startswith("block")].copy() if "stage" in align.columns else align.copy()
    blocks["_x"] = blocks["layer"].astype(float)
    fig, axes = style.figure("wide", ncols=2)
    st = arm_style("tabpfn_v2_5")
    gp_st = arm_style("gp_mll_frozen")
    for col, color, ls, label in (
        ("pearson", st.color, "-", "conditioned kernel ($|\\Delta_{GP}|$)"),
        ("pearson_prior_kernel", gp_st.color, "--", "prior kernel $k(x, x^*)$"),
        ("pearson_gp_fixed", style.NEUTRAL_GREY, ":", "fixed-kernel GP"),
    ):
        if col not in blocks.columns or blocks[col].isna().all():
            continue
        med = blocks.groupby("_x")[col].median()
        axes[0].plot(med.index, med.values, color=color, ls=ls, marker=st.marker, label=label)
    axes[0].set_ylabel("Pearson $r$ with $\\|\\Delta Z_\\ell\\|$")
    axes[0].legend(loc="best", fontsize=style.FONT_SIZES["legend"])

    agg = blocks.groupby("_x")["partial_r_gp_given_prior"]
    med = agg.median()
    axes[1].plot(med.index, med.values, color=st.color, marker=st.marker)
    axes[1].fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values,
                         color=st.color, alpha=style.BAND_ALPHA, lw=0)
    axes[1].axhline(0.0, color=style.NEUTRAL_GREY, lw=0.8)
    axes[1].set_ylabel("partial $r$, conditioned | prior")
    for ax in axes:
        ax.set_xlabel(style.axis_label("layer"))
        style.set_integer_xaxis(ax)
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_dynamics_geometry")


def _layer_dynamics_robust(align: pd.DataFrame, run_dir: str) -> list[str]:
    """C0.b: does the early peak survive excluding the injected site and the observed sites?

    Panel (a) puts the four alignment variants on one axis, so the reader can see which exclusion
    moves the curve and by how much. Panel (b) is the pre-registered test itself: the margin by
    which the second block exceeds the average over blocks 3-17, per context size, against the
    frozen 0.10 threshold -- the statistic the rule names is the one that drops *both* the injected
    site and the observed sites, and it is drawn last so it reads on top.

    Args:
        align: ``layer_dynamics.csv``.
        run_dir: Run directory.

    Returns:
        Written paths.
    """
    variants = [
        ("pearson", "all sites", "-"),
        ("pearson_unobserved", "observed dropped", "--"),
        ("pearson_off_anchor", "injected site dropped", ":"),
        ("pearson_unobserved_off_anchor", "both dropped (the named rule)", "-."),
    ]
    present = [(c, lab, ls) for c, lab, ls in variants if c in align.columns and align[c].notna().any()]
    if not present:
        return []
    blocks = align[align["stage"].str.startswith("block")].copy()
    blocks["L"] = blocks["layer"].astype(float)
    fig, axes = style.figure("wide", ncols=2)
    st = arm_style("tabpfn_v2_5")
    for i, (col, label, ls) in enumerate(present):
        color = st.color if len(present) == 1 else style.derive_shade(st.color, i, len(present))
        med = blocks.groupby("L")[col].median()
        axes[0].plot(med.index, med.values, color=color, ls=ls, marker=st.marker, label=label)
    axes[0].axhline(0.0, color=style.NEUTRAL_GREY, lw=0.8)
    axes[0].set_ylabel("Pearson $r$ with $|\\Delta_{GP}|$")
    axes[0].legend(loc="best", fontsize=style.FONT_SIZES["legend"])

    contexts = sorted(blocks["context_t"].unique())
    width = 0.8 / max(len(present), 1)
    for i, (col, label, _ls) in enumerate(present):
        color = st.color if len(present) == 1 else style.derive_shade(st.color, i, len(present))
        margins = []
        for t in contexts:
            g = blocks[blocks["context_t"] == t]
            margins.append(g[g["L"] == 1][col].median() - g[g["L"].between(3, 17)][col].median())
        axes[1].bar(np.arange(len(contexts)) + i * width, margins, width=width, color=color, label=label)
    axes[1].axhline(0.10, color=style.NEUTRAL_GREY, ls="--", lw=1.0)
    axes[1].set_xticks(np.arange(len(contexts)) + 0.4 - width / 2)
    axes[1].set_xticklabels([str(int(t)) for t in contexts])
    axes[1].set_xlabel(style.axis_label("context_t"))
    axes[1].set_ylabel("block 1 $-$ blocks 3–17")
    axes[0].set_xlabel(style.axis_label("layer"))
    style.set_integer_xaxis(axes[0])
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_dynamics_robust")


def _layer_dynamics_direction(align: pd.DataFrame, run_dir: str) -> list[str]:
    """C5 and C6b: the direction-aware readouts, and similarity between consecutive blocks.

    Panel (a) normalises each readout by its own maximum over blocks, because they live on
    different scales and the pre-registered question is *relative*: whether the mid-depth dip is
    shallower for a direction-aware statistic than for the size-only one. Panel (b) is the
    consecutive-block similarity whose minima were predicted to mark stage boundaries.

    Args:
        align: ``layer_dynamics.csv``.
        run_dir: Run directory.

    Returns:
        Written paths.
    """
    cols = [("pearson", "size only"), ("vector_cka", "GP-shaped component"),
            ("signed_pc_r", "signed leading direction")]
    present = [(c, lab) for c, lab in cols if c in align.columns and align[c].notna().any()]
    if len(present) < 2:
        return []
    blocks = align[align["stage"].str.startswith("block")].copy()
    blocks["L"] = blocks["layer"].astype(float)
    fig, axes = style.figure("wide", ncols=2)
    st = arm_style("tabpfn_v2_5")
    for i, (col, label) in enumerate(present):
        color = st.color if len(present) == 1 else style.derive_shade(st.color, i, len(present))
        med = blocks.groupby("L")[col].median()
        peak = med.max()
        if peak <= 0:
            continue
        axes[0].plot(med.index, med.values / peak, color=color, marker=st.marker,
                     ls="-" if i == 0 else "--", label=label)
    axes[0].axhline(0.75, color=style.NEUTRAL_GREY, ls=":", lw=1.0)
    axes[0].set_ylabel("share of each readout's own peak")
    axes[0].set_xlabel(style.axis_label("layer"))
    axes[0].legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    style.set_integer_xaxis(axes[0])

    if "cka_next_block" in blocks.columns and blocks["cka_next_block"].notna().any():
        agg = blocks.groupby("L")["cka_next_block"]
        med = agg.median().dropna()
        axes[1].plot(med.index, med.values, color=st.color, marker=st.marker)
        axes[1].fill_between(med.index, agg.quantile(0.25).reindex(med.index).values,
                             agg.quantile(0.75).reindex(med.index).values,
                             color=st.color, alpha=style.BAND_ALPHA, lw=0)
        lows = med.nsmallest(3)
        axes[1].plot(lows.index, lows.values, "v", color=style.NEUTRAL_GREY,
                     ms=style.MARKER_SIZE * 2, label="three lowest")
        axes[1].legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    axes[1].set_ylabel("similarity, block $\\ell$ vs $\\ell+1$")
    axes[1].set_xlabel(style.axis_label("layer"))
    style.set_integer_xaxis(axes[1])
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_dynamics_direction")


def _layer_dynamics_precision(prec: pd.DataFrame, run_dir: str) -> list[str]:
    """C0.a: the three-arm precision comparison.

    Panel (a) is the agreement between arms: the cosine between the half- and full-precision
    representation changes isolates precision, the cosine between the two full-precision devices
    isolates hardware, and the frozen 0.9 threshold is drawn. Panel (b) is what that does to the
    reported statistic -- the arms' alignment side by side, where only the first block separates.

    Args:
        prec: ``layer_precision.csv``.
        run_dir: Run directory.

    Returns:
        Written paths.
    """
    if not {"cos_precision", "cos_device"} <= set(prec.columns):
        return []
    fig, axes = style.figure("wide", ncols=2)
    st = arm_style("tabpfn_v2_5")
    gp = arm_style("gp_mll_frozen")
    for col, color, label in (("cos_precision", st.color, "half vs full precision (same GPU)"),
                              ("cos_device", gp.color, "GPU vs CPU (both full precision)")):
        med = prec.groupby("layer")[col].median()
        axes[0].plot(med.index, med.values, color=color, marker=st.marker, label=label)
    axes[0].axhline(0.9, color=style.NEUTRAL_GREY, ls="--", lw=1.0)
    axes[0].set_ylabel("median cosine between arms")
    axes[0].legend(loc="best", fontsize=style.FONT_SIZES["legend"])

    arms = [("align_cuda_fp16", "half precision, GPU", "--"),
            ("align_cuda_fp32", "full precision, GPU", "-"),
            ("align_cpu_fp32", "full precision, CPU", ":")]
    for i, (col, label, ls) in enumerate(a for a in arms if a[0] in prec.columns):
        color = style.derive_shade(st.color, i, 3)
        med = prec.groupby("layer")[col].median()
        axes[1].plot(med.index, med.values, color=color, ls=ls, marker=st.marker, label=label)
    axes[1].axhline(0.0, color=style.NEUTRAL_GREY, lw=0.8)
    axes[1].set_ylabel("Pearson $r$ with $|\\Delta_{GP}|$")
    axes[1].legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    for ax in axes:
        ax.set_xlabel(style.axis_label("layer"))
        style.set_integer_xaxis(ax)
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_dynamics_precision")


def _layer_dynamics_causal(causal: pd.DataFrame, run_dir: str) -> list[str]:
    """C6a and C2: the two causal interventions.

    Panel (a) is block influence -- how much the output update changes when one block is skipped,
    with both frozen thresholds drawn (0.20 for the first block, 0.065 for the middle ones). Panel
    (b) is the routing hand-off: the share of the update still carried by the injected row when its
    state is swapped at that block, one line per context size. A share near 1 means the sites being
    predicted are still reading the value from that row rather than holding it themselves.

    Args:
        causal: ``layer_causal.csv``.
        run_dir: Run directory.

    Returns:
        Written paths.
    """
    if "intervention" not in causal.columns:
        return []
    fig, axes = style.figure("wide", ncols=2)
    st = arm_style("tabpfn_v2_5")

    skip = causal[causal["intervention"] == "skip"].copy()
    if len(skip):
        skip["d_rho"] = (skip["rho_shape_plain"] - skip["rho_shape"]).abs()
        med = skip.groupby("block")["d_rho"].median()
        axes[0].bar(med.index.astype(int), med.values, color=st.color, width=0.7)
        axes[0].axhline(0.20, color=style.NEUTRAL_GREY, ls="--", lw=1.0)
        axes[0].axhline(0.065, color=style.NEUTRAL_GREY, ls=":", lw=1.0)
        axes[0].set_ylabel("change in output alignment when skipped")

    swap = causal[causal["intervention"] == "swap"]
    if len(swap) and "injected_row_share" in swap.columns:
        contexts = sorted(swap["context_t"].unique())
        for i, t in enumerate(contexts):
            color = st.color if len(contexts) == 1 else style.derive_shade(st.color, i, len(contexts))
            med = swap[swap["context_t"] == t].groupby("block")["injected_row_share"].median()
            axes[1].plot(med.index, med.values, color=color, marker=st.marker, label=f"$t$ = {int(t)}")
        axes[1].set_ylabel("share still carried by the injected row")
        axes[1].legend(loc="best", fontsize=style.FONT_SIZES["legend"])
    for ax in axes:
        ax.set_xlabel(style.axis_label("layer"))
        style.set_integer_xaxis(ax)
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_dynamics_causal")


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
