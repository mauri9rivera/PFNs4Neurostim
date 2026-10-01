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
from matplotlib.colors import LinearSegmentedColormap
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

#: Width of the achieved-SNR bins in F4. 3 dB is one halving of SNR and is coarse enough that every bin
#: holds several channels at the NHP grid (54 raw SNR values over a ~30 dB span).
SNR_BIN_WIDTH_DB: float = 3.0

#: CKA (a) target kernels that get a panel, in display order. ``K_X`` (electrode geometry) and
#: ``K_GT_linear`` were dropped from the figure on 2026-09-27: ``K_X`` is an explicit *control* -- the model's
#: input IS the coordinates, so agreement with geometry is trivial and it read highest, which invited the
#: wrong conclusion -- and ``K_GT_linear`` is a bandwidth *sensitivity* check on ``K_GT``. Both stay in
#: ``cka.csv`` and in the config's ``targets``, so the control and the sensitivity check remain auditable;
#: they are simply not headline panels. The targets of interest are the response and the GP posterior.
CKA_PLOTTED_TARGETS: tuple[str, ...] = ("K_GT", "K_GP")


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
    row_titles = (arm_style(str(ex["engine"])).label, style.model_label("gp_mll"), "Difference")
    fig, axes = style.figure("full", nrows=3, ncols=n_ctx, squeeze=False)
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
    fig.colorbar(im, ax=axes, shrink=0.6, label=style.axis_label("update_normalized"))
    fig.suptitle(_exemplar_provenance(ex), x=0.01, ha="left", fontsize=style.FONT_SIZES["annotation"])
    return style.save_figure(fig, run_dir, "update_exemplar")


def _contexts(ex: Any) -> str:
    """Comma-separated context sizes stored in an exemplar ``.npz``."""
    key = "context_ts" if "context_ts" in getattr(ex, "files", []) else "context_t"
    if key not in getattr(ex, "files", []):
        return "?"
    return ", ".join(str(int(v)) for v in np.atleast_1d(ex[key]))


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
        f"draw {field('draw')} | anchor {field('anchor')} ({field('anchor_stratum')}), "
        f"surprise c = {field('surprise_c', '{:g}')} | channel picked by {field('select_rule')}"
    )


def _f2_shape_vs_context(cell: pd.DataFrame, run_dir: str, gates: list[dict[str, Any]]) -> list[str]:
    """F2: rho_shape vs context size with the anchor-permutation null and the seed floor."""
    fig, ax = style.figure("onehalf")
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


def _f4_lengthscale(anchor: pd.DataFrame, run_dir: str, *, bin_width_db: float = SNR_BIN_WIDTH_DB) -> list[str]:
    """F4: effective lengthscale vs achieved SNR — PFN, refit GP, frozen GP (H10.3).

    ``achieved_snr_db`` is a *per-channel* quantity, so grouping on its raw value gave one group per
    (channel, level) -- on the NHP run, 54 groups of 3 rows each -- and the "trend" was a line through 54
    medians of 3, which read as noise (diagnosed 2026-09-27). Rows are therefore binned into fixed-width dB
    bins and each bin shows the median with an interquartile band, so the band states the spread instead of
    hiding it. Binning is a display choice; ``update_rule_cell.csv`` is unchanged.

    Args:
        anchor: Per-cell frame with ``engine``, ``achieved_snr_db`` and ``ell_hat``.
        run_dir: Destination directory.
        bin_width_db: Width of each achieved-SNR bin, in dB.

    Returns:
        Written paths.
    """
    fig, ax = style.figure("onehalf")
    snr = anchor["achieved_snr_db"].to_numpy(dtype=float)
    finite = snr[np.isfinite(snr)]
    if finite.size == 0:
        return []
    edges = np.arange(np.floor(finite.min() / bin_width_db) * bin_width_db,
                      np.ceil(finite.max() / bin_width_db) * bin_width_db + bin_width_db,
                      bin_width_db)
    binned = anchor.assign(_bin=pd.cut(anchor["achieved_snr_db"], bins=edges))
    for engine, sub in binned.groupby("engine"):
        st = arm_style(str(engine))
        grp = sub.dropna(subset=["_bin"]).groupby("_bin", observed=True)["ell_hat"]
        centre = np.array([iv.mid for iv in grp.median().index])
        med, q1, q3 = grp.median().to_numpy(), grp.quantile(0.25).to_numpy(), grp.quantile(0.75).to_numpy()
        order = np.argsort(centre)
        ax.plot(centre[order], med[order], color=st.color, ls=st.linestyle, marker=st.marker, label=st.label)
        ax.fill_between(centre[order], q1[order], q3[order], color=st.color,
                        alpha=style.BAND_ALPHA, lw=0)
    ax.invert_xaxis()   # more stress reads rightward, as in every K2 figure
    ax.set_xlabel(f"{style.axis_label('achieved_snr_db')} ({bin_width_db:g} dB bins)")
    ax.set_ylabel(style.axis_label("ell_hat"))
    ax.legend(loc="best")
    return style.save_figure(fig, run_dir, "lengthscale_vs_snr")


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
    """F6 (secondary): layer alignment and probe R^2 vs layer, ONE TRACE PER CONTEXT SIZE.

    Widened to two full panels on 2026-09-27 (at ``onehalf`` the right panel's title overran the left
    panel's axes) and split by context size on 2026-09-30: the layer profile is a property of the surrogate
    *given its evidence*, so a single trace conflated "where the update becomes GP-like" with "how much the
    model was told". Contexts are shades of the TabPFN anchor, dark = smallest; the band is the IQR over
    channels at that context, i.e. between-channel spread, not estimator noise.

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
        ax.set_ylabel(style.axis_label(col))
    peaks = layers[layers["is_peak"].astype(bool)]["layer"]
    if len(peaks):
        axes[0].plot(peaks.value_counts().index, np.zeros(peaks.nunique()), "|", color=style.NEUTRAL_GREY,
                     ms=style.MARKER_SIZE * 3)
    axes[0].legend(loc="best", title=style.axis_label("context_t"),
                   fontsize=style.FONT_SIZES["legend"], title_fontsize=style.FONT_SIZES["legend"])
    style.panel_letters(axes)
    return style.save_figure(fig, run_dir, "layer_alignment")


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
    if not os.path.exists(path_c) or os.path.getsize(path_c) <= 1:
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
        targets = [t for t in CKA_PLOTTED_TARGETS if t in set(df["target"])]
        levels = sorted(df["level"].unique())
        # ONE ROW PER CONTEXT SIZE (2026-09-30). Until now every context size was pooled into a single
        # band, so a figure whose whole subject is what the network does with its evidence hid that axis
        # inside its own error band -- and the band looked like noise when much of it was the t dependence.
        contexts = sorted(df["context_t"].unique())
        fig, axes = style.figure("full", nrows=len(contexts), ncols=len(targets),
                                 sharey=True, sharex=True, squeeze=False)
        base = style.model_color("tabpfn_v2_5")
        for r, t_ctx in enumerate(contexts):
            for c, target in enumerate(targets):
                ax = axes[r, c]
                sub = df[(df["target"] == target) & (df["context_t"] == t_ctx)]
                for i, level in enumerate(levels):
                    color = base if i == 0 else style.derive_shade(base, i - 1, max(len(levels) - 1, 1))
                    # Average the draws WITHIN a channel first, then spread over channels: the band is
                    # then between-channel heterogeneity (a finding) and not draw noise (an artefact).
                    per_channel = (sub[sub["level"] == level]
                                   .groupby(["subject", "emg", "layer"])["cka_minus_null"].mean().reset_index())
                    agg = per_channel.groupby("layer")["cka_minus_null"]
                    med, q1, q3 = agg.median(), agg.quantile(0.25), agg.quantile(0.75)
                    ax.plot(med.index, med.values, color=color, label=f"level {level:g}")
                    ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=style.BAND_ALPHA, lw=0)
                ax.axhline(0.0, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2)
                if r == 0:
                    ax.set_title(target.replace("_", " "))
                if r == len(contexts) - 1:
                    ax.set_xlabel(style.axis_label("layer"))
                if c == 0:
                    ax.set_ylabel(f"$t$ = {int(t_ctx)}" + "\n" + style.axis_label("cka_minus_null"))
        axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"])
        written += style.save_figure(fig, run_dir, "cka_vs_layer")
    path_b = os.path.join(run_dir, "cka_placement.csv")
    if os.path.exists(path_b) and os.path.getsize(path_b) > 1:
        pl = pd.read_csv(path_b)
        if not pl.empty:
            # Layers whose floor/ceiling gap is degenerate are DROPPED, not clipped: placement is
            # (d - floor) / (ceiling - floor), so a ~0 (or negative) denominator makes the ratio explode and
            # inverts its sign. Clipping to [0, 1] would present a broken denominator as a tidy number
            # (decision 2026-09-27). The excluded layers are named on the figure instead.
            usable = pl[pl["gap_ok"].astype(bool)] if "gap_ok" in pl.columns else pl
            dropped = sorted(set(pl["layer"]) - set(usable["layer"]))
            if usable.empty:
                return written
            contexts = sorted(usable["context_t"].unique())
            base = style.model_color("tabpfn_v2_5")
            fig, ax = style.figure("onehalf")
            _reference_bands(ax, "h", label_it=True)
            for i, t in enumerate(contexts):
                # One trace per context size (2026-09-30): placement is conditioned on the evidence the
                # embedding saw, so a single curve could not say whether a layer moves toward the prior bag
                # or simply had more observations.
                color = base if len(contexts) == 1 else style.derive_shade(base, i, len(contexts))
                per_channel = (usable[usable["context_t"] == t]
                               .groupby(["subject", "emg", "layer"])["p"].mean().reset_index())
                agg = per_channel.groupby("layer")["p"]
                med = agg.median()
                ax.plot(med.index, med.values, color=color, marker="o", label=f"$t$ = {int(t)}")
                ax.fill_between(med.index, agg.quantile(0.25).values, agg.quantile(0.75).values,
                                color=color, alpha=style.BAND_ALPHA, lw=0)
            if dropped:
                ax.text(0.98, 0.02,
                        f"layers omitted (no prior/noise gap): {', '.join(str(d) for d in dropped)}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=style.FONT_SIZES["annotation"], alpha=0.8)
            ax.set_xlabel(style.axis_label("layer"))
            ax.set_ylabel(style.axis_label("placement"))
            ax.legend(loc="best", fontsize=style.FONT_SIZES["legend"])
            written += style.save_figure(fig, run_dir, "cka_placement")
            written += _cka_placement_surface(usable, dropped, run_dir)
    return written


def _cka_placement_surface(usable: pd.DataFrame, dropped: list[int], run_dir: str) -> list[str]:
    """Layer x context-size heatmap of the CKA placement p (2026-09-30).

    The flat view of the layer / evidence / placement volume: a 3D surface of the same numbers reads worse
    in print (occlusion, one fixed viewing angle, no way to read a value off it), so the third axis is
    colour. The colormap is anchored at the two references -- ``p`` = 0 is the prior floor and ``p`` = 1 the
    noise ceiling -- so the reference colours mean the same thing here as the bands do in the line figures,
    and values outside [0, 1] saturate rather than redefining the scale.

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
    # A two-stop ramp between the two reference tokens: placement is a sequential 0 -> 1 quantity, so the
    # colormap runs floor colour -> ceiling colour and the ends mean exactly what the bands mean elsewhere.
    cmap = LinearSegmentedColormap.from_list(
        "placement_floor_ceiling", [style.REFERENCE_FLOOR_COLOR, style.REFERENCE_CEILING_COLOR]
    )
    fig, ax = style.figure("onehalf")
    im = ax.pcolormesh(grid.columns.to_numpy(), grid.index.to_numpy(), grid.to_numpy(),
                       cmap=cmap, vmin=0.0, vmax=1.0, shading="nearest")
    ax.set_xlabel(style.axis_label("layer"))
    ax.set_ylabel(style.axis_label("context_t"))
    ax.set_yticks(grid.index.to_numpy(), [str(int(t)) for t in grid.index])
    cb = fig.colorbar(im, ax=ax, label=style.axis_label("placement"))
    cb.set_ticks([0.0, 0.5, 1.0], labels=["0 (prior floor)", "0.5", "1 (noise ceiling)"])
    if dropped:
        ax.text(0.98, 1.02, f"layers omitted: {', '.join(str(d) for d in dropped)}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=style.FONT_SIZES["annotation"], alpha=0.8)
    return style.save_figure(fig, run_dir, "cka_placement_surface")

