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

import numpy as np
import pandas as pd

from . import style

__all__ = [
    "arm_style",
    "render_update_rule",
    "render_placement",
    "render_cka",
]

#: Update-rule engine arms and how they map onto the model palette.
_GP_ARMS: tuple[str, ...] = ("frozen", "refit")


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
    return style.model_style(engine)


def _iqr(values: pd.Series) -> tuple[float, float, float]:
    """``(median, q25, q75)`` of a series."""
    return float(values.median()), float(values.quantile(0.25)), float(values.quantile(0.75))


# ---------------------------------------------------------------------------
# Update rule (task #9, roadmap M10 F1-F5)
# ---------------------------------------------------------------------------
def _f1_exemplar(run_dir: str) -> list[str]:
    """F1: grid heatmaps of the PFN update, the GP update and their difference (2D grids)."""
    path = os.path.join(run_dir, "update_rule_exemplar.npz")
    if not os.path.exists(path):
        return []
    ex = np.load(path)
    shape = tuple(int(v) for v in ex["grid_shape"])
    ch2xy = ex["ch2xy"]
    if len(shape) != 2:
        return []   # 5D condition sets have no map view
    maps = []
    for vec in (ex["g_model"], ex["g_gp"]):
        grid = np.full(shape, np.nan)
        grid[ch2xy[:, 0], ch2xy[:, 1]] = vec / np.max(np.abs(vec))
        maps.append(grid)
    maps.append(maps[0] - maps[1])
    fig, axes = style.figure("double", ncols=3)
    titles = (arm_style(str(ex["engine"])).label, style.model_label("gp_mll"), "Difference")
    a = int(ex["anchor"])
    for ax, grid, title in zip(axes, maps, titles):
        im = ax.imshow(grid.T, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.plot(ch2xy[a, 0], ch2xy[a, 1], marker="*", color="k", ms=style.MARKER_SIZE * 2)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label=style.axis_label("update_normalized"))
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "update_exemplar")


def _f2_shape_vs_context(cell: pd.DataFrame, run_dir: str, gates: list[dict[str, Any]]) -> list[str]:
    """F2: rho_shape vs context size with the anchor-permutation null and the seed floor."""
    fig, ax = style.figure("single")
    for engine, sub in cell.groupby("engine"):
        if engine == "gp_mll_frozen":
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
    fig, axes = style.figure("double", ncols=len(levels), sharey=True)
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


def _f4_lengthscale(anchor: pd.DataFrame, run_dir: str) -> list[str]:
    """F4: effective lengthscale vs achieved SNR — PFN, refit GP, frozen GP (H10.3)."""
    fig, ax = style.figure("single")
    for engine, sub in anchor.groupby("engine"):
        st = arm_style(str(engine))
        agg = sub.groupby("achieved_snr_db")["ell_hat"].median().sort_index()
        ax.plot(agg.index, agg.values, color=st.color, ls=st.linestyle, marker=st.marker, label=st.label)
    ax.invert_xaxis()   # more stress reads rightward, as in every K2 figure
    ax.set_xlabel(style.axis_label("achieved_snr_db"))
    ax.set_ylabel(style.axis_label("ell_hat"))
    ax.legend(loc="best")
    return style.save_figure(fig, run_dir, "lengthscale_vs_snr")


def _f5_kernel_properties(cell: pd.DataFrame, run_dir: str) -> list[str]:
    """F5: implied-kernel properties per engine (symmetry, PSD, stationarity)."""
    metrics = [m for m in ("sym_err", "psd_neg_mass", "ell_cv_anchor") if m in cell]
    fig, axes = style.figure("double", ncols=len(metrics))
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

    Args:
        run_dir: Update-rule run directory.

    Returns:
        Written paths.
    """
    paths = {k: os.path.join(run_dir, f"update_rule{k}.csv") for k in ("", "_cell", "_surprise")}
    if not all(os.path.exists(p) for p in paths.values()):
        return []
    anchor, cell, surprise = (pd.read_csv(paths[k]) for k in ("", "_cell", "_surprise"))
    gates_path = os.path.join(run_dir, "gates.json")
    gates = json.load(open(gates_path, encoding="utf-8")) if os.path.exists(gates_path) else []
    written = _f1_exemplar(run_dir)
    written += _f2_shape_vs_context(cell, run_dir, gates)
    written += _f3_surprise(surprise, run_dir)
    written += _f4_lengthscale(anchor, run_dir)
    written += _f5_kernel_properties(cell, run_dir)
    layers_path = os.path.join(run_dir, "update_rule_layers.csv")
    if os.path.exists(layers_path):
        written += _f6_layer_alignment(pd.read_csv(layers_path), run_dir)
    return written


def _f6_layer_alignment(layers: pd.DataFrame, run_dir: str) -> list[str]:
    """F6 (secondary): layer alignment curve and the distribution of ``layer_peak``."""
    fig, axes = style.figure("onehalf", ncols=2)
    st = arm_style("tabpfn_v2_5")
    for ax, col in zip(axes, ("layer_corr", "layer_probe_r2")):
        agg = layers.groupby("layer")[col]
        med = agg.median()
        ax.plot(med.index, med.values, color=st.color, marker=st.marker, label=st.label)
        ax.fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values,
                        color=st.color, alpha=style.BAND_ALPHA, lw=0)
        ax.set_xlabel(style.axis_label("layer"))
        ax.set_ylabel(style.axis_label(col))
    peaks = layers[layers["is_peak"].astype(bool)]["layer"]
    if len(peaks):
        axes[0].plot(peaks.value_counts().index, np.zeros(peaks.nunique()), "|", color=style.NEUTRAL_GREY,
                     ms=style.MARKER_SIZE * 3)
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_alignment")


#: Floor/ceiling pairs of the placement analysis (task #6 Step 3), in display order.
PLACEMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("f1c1", "Floor 1 / Ceiling 1\n(held-out prior / noise)"),
    ("f2c2", "Floor 2 / Ceiling 2\n(split-half / shuffled self)"),
    ("f1c2", "Floor 1 / Ceiling 2"),
    ("f2c1", "Floor 2 / Ceiling 1"),
)


def render_placement(run_dir: str) -> list[str]:
    """Placement strip plot (formulation B) and context trajectory (formulation C).

    Args:
        run_dir: Placement run directory.

    Returns:
        Written paths.
    """
    path_b = os.path.join(run_dir, "placement.csv")
    if not os.path.exists(path_b):
        return []
    df = pd.read_csv(path_b)
    metrics = sorted(df["metric"].unique())
    pairs = [(k, lbl) for k, lbl in PLACEMENT_PAIRS if f"p_{k}" in df]
    fig, axes = style.figure("double", nrows=len(metrics), ncols=len(pairs),
                             squeeze=False, sharex=True)
    rng = np.random.default_rng(0)
    for r, metric in enumerate(metrics):
        sub = df[df["metric"] == metric]
        for c, (key, label) in enumerate(pairs):
            ax = axes[r, c]
            ax.axvspan(-0.05, 0.05, color=style.ANCHOR_PFN, alpha=style.BAND_ALPHA, lw=0)
            ax.axvspan(0.95, 1.05, color=style.ANCHOR_GP, alpha=style.BAND_ALPHA, lw=0)
            vals = sub[f"p_{key}"].to_numpy(dtype=float)
            jit = rng.uniform(-0.3, 0.3, size=len(vals))
            ax.scatter(vals, jit, s=style.MARKER_SIZE ** 2, color=style.NEUTRAL_GREY, linewidths=0)
            lo, hi = sub[f"p_{key}_lo"].to_numpy(dtype=float), sub[f"p_{key}_hi"].to_numpy(dtype=float)
            ax.hlines(jit, lo, hi, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2)
            if np.isfinite(vals).any():
                ax.axvline(np.nanmedian(vals), color="k", lw=style.LINE_WIDTH)
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label)
            if c == 0:
                valid = bool(sub["metric_valid"].all()) if "metric_valid" in sub else True
                ax.set_ylabel(metric.upper() + ("" if valid else " (failed M0)"))
            if r == len(metrics) - 1:
                ax.set_xlabel(style.axis_label("placement"))
    written = style.save_figure(fig, run_dir, "placement_strip")

    path_c = os.path.join(run_dir, "placement_context.csv")
    if os.path.exists(path_c) and os.path.getsize(path_c) > 1:
        ctx = pd.read_csv(path_c)
        if not ctx.empty:
            ctx = ctx[~ctx["is_full"].astype(bool)]
            levels = sorted(ctx["level"].unique())
            fig, axes = style.figure("onehalf", ncols=len(metrics), squeeze=False)
            for ax, metric in zip(axes[0], metrics):
                sub = ctx[ctx["metric"] == metric]
                for i, level in enumerate(levels):
                    color = style.derive_shade(style.NEUTRAL_GREY, i, len(levels))
                    agg = sub[sub["level"] == level].groupby("context_t")["p"].median()
                    ax.plot(agg.index, agg.values, color=color, marker="o", label=f"level {level:g}")
                ax.axhline(0.0, color=style.ANCHOR_PFN, lw=style.LINE_WIDTH / 2)
                ax.axhline(1.0, color=style.ANCHOR_GP, lw=style.LINE_WIDTH / 2)
                ax.set_xscale("log")
                ax.set_xlabel(style.axis_label("context_t"))
                ax.set_ylabel(style.axis_label("placement"))
                ax.set_title(metric.upper())
            axes[0][0].legend(loc="best")
            written += style.save_figure(fig, run_dir, "placement_context")
    return written


def render_cka(run_dir: str) -> list[str]:
    """CKA (a) minus null vs layer per target, and CKA (b) placement vs layer.

    Args:
        run_dir: CKA run directory.

    Returns:
        Written paths.
    """
    written: list[str] = []
    path_a = os.path.join(run_dir, "cka.csv")
    if os.path.exists(path_a) and os.path.getsize(path_a) > 1:
        df = pd.read_csv(path_a)
        targets = [t for t in ("K_GT", "K_GP", "K_GT_linear", "K_X") if t in set(df["target"])]
        levels = sorted(df["level"].unique())
        fig, axes = style.figure("double", ncols=len(targets),
                                 sharey=True, squeeze=False)
        base = style.model_color("tabpfn_v2_5")
        for ax, target in zip(axes[0], targets):
            sub = df[df["target"] == target]
            for i, level in enumerate(levels):
                color = base if i == 0 else style.derive_shade(base, i - 1, max(len(levels) - 1, 1))
                per_channel = (sub[sub["level"] == level]
                               .groupby(["subject", "emg", "layer"])["cka_minus_null"].mean().reset_index())
                agg = per_channel.groupby("layer")["cka_minus_null"]
                med, q1, q3 = agg.median(), agg.quantile(0.25), agg.quantile(0.75)
                ax.plot(med.index, med.values, color=color, label=f"level {level:g}")
                ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=style.BAND_ALPHA, lw=0)
            ax.axhline(0.0, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2)
            ax.set_title(target.replace("_", " "))
            ax.set_xlabel(style.axis_label("layer"))
        axes[0][0].set_ylabel(style.axis_label("cka_minus_null"))
        axes[0][0].legend(loc="best")
        style.panel_letters(axes[0])
        written += style.save_figure(fig, run_dir, "cka_vs_layer")
    path_b = os.path.join(run_dir, "cka_placement.csv")
    if os.path.exists(path_b) and os.path.getsize(path_b) > 1:
        pl = pd.read_csv(path_b)
        if not pl.empty:
            fig, ax = style.figure("single")
            agg = pl.groupby("layer")["p"]
            med = agg.median()
            ax.axhspan(-0.05, 0.05, color=style.ANCHOR_PFN, alpha=style.BAND_ALPHA, lw=0, label="Prior floor")
            ax.axhspan(0.95, 1.05, color=style.ANCHOR_GP, alpha=style.BAND_ALPHA, lw=0, label="Noise ceiling")
            ax.plot(med.index, med.values, color=style.model_color("tabpfn_v2_5"), marker="o",
                    label=style.model_label("tabpfn_v2_5"))
            ax.fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values,
                            color=style.model_color("tabpfn_v2_5"), alpha=style.BAND_ALPHA, lw=0)
            ax.set_xlabel(style.axis_label("layer"))
            ax.set_ylabel(style.axis_label("placement"))
            ax.legend(loc="best")
            written += style.save_figure(fig, run_dir, "cka_placement")
    return written
