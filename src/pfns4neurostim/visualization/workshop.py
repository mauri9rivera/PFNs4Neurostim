"""The four items of the AAAI-27 NeuroAI workshop short paper, at AAAI geometry (research_design §0 figure plan).

Figures are named by title, never by number (user, 2026-11-04):

* ``multi_pfn_benchmark_table``   -- *Multi-PFN benchmark table* (SVG / PDF rendering + AAAI-27 LaTeX ``.txt`` + ``.csv``)
* ``nhp_stress_regimes``          -- *NHP stress regimes (in vivo)*
* ``is_the_update_gp_like``       -- *Is the update GP-like?*
* ``is_the_update_coherent``      -- *Is the update coherent?*

Everything is read from run directories named in a YAML config (``configs/workshop/*.yaml``), so the figures rebuild
unchanged when OMEGA Phase 1, the BETA re-run and the GP-MLL recomputes land. Two rules keep stale numbers out:

* **G2 (version gate):** a model whose recorded ``model_version`` differs from the registry's current one is never
  shown as a number or curve; its table cells read :data:`style.PENDING_RECOMPUTE_LABEL`. Today every GP-MLL row on
  disk predates the restart-rule fix, so every GP-MLL cell is pending.
* **Stand-in data is stamped:** a mechanism panel drawn from fewer channels than the analysis set, or at a context size
  other than the requested one, carries a DRAFT stamp naming what it shows.

Widths are the AAAI column (``single``, 3.27 in <= 3.3 in) and text width (``double``, 6.93 in <= 7.0 in), unconfirmed
against ``AuthorKit27`` (not in the repository).
"""
from __future__ import annotations

import json
import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

from ..models.registry import MODEL_REGISTRY
from . import mechanism as M
from . import stress as ST
from . import style as S

__all__ = [
    "current_models",
    "benchmark_table",
    "render_benchmark_table",
    "render_stress_regimes",
    "render_gp_like",
    "render_coherence",
    "render_beta_supplement",
    "render_all",
]

#: Output basenames, one per workshop item (the title in snake case).
FIGURE_NAMES: dict[str, str] = {
    "benchmark": "multi_pfn_benchmark_table",
    "stress": "nhp_stress_regimes",
    "gp_like": "is_the_update_gp_like",
    "coherence": "is_the_update_coherent",
    "beta_supplement": "what_the_representation_encodes_supplement",
}
#: Columns that identify one mechanism row across run directories (host and wall time are provenance, not identity).
_PROVENANCE_COLUMNS: tuple[str, ...] = ("host", "seconds")
#: Layout of the rendered table, in inches: first-column width, row height, header block height.
_TABLE_LABEL_WIDTH: float = 1.25
_TABLE_ROW_HEIGHT: float = 0.19
_TABLE_HEADER_HEIGHT: float = 0.42


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def current_models(tidy: pd.DataFrame) -> dict[str, bool]:
    """Whether every row of each model carries the registry's current ``model_version`` (guardrail G2).

    Args:
        tidy: Tidy rows with ``model`` and ``model_version``.

    Returns:
        Model -> True when its rows are current (a model absent from the registry counts as current).
    """
    out: dict[str, bool] = {}
    for model, rows in tidy.groupby("model"):
        spec = MODEL_REGISTRY.get(str(model))
        out[str(model)] = spec is None or set(rows["model_version"].astype(str)) == {spec.version}
    return out


def _stamp(ax: Any, text: str) -> None:
    """Write a DRAFT stamp across the top of a panel."""
    ax.text(0.5, 0.98, f"DRAFT: {text}", transform=ax.transAxes, ha="center", va="top", color=S.DRAFT_STAMP_COLOR,
            fontsize=S.FONT_SIZES["annotation"], fontweight="bold", zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0})


def _placeholder(ax: Any, text: str) -> None:
    """An empty panel that says what it is waiting for."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=S.DRAFT_STAMP_COLOR,
            fontsize=S.FONT_SIZES["annotation"], wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _read_runs(run_dirs: Sequence[str], name: str) -> pd.DataFrame:
    """Concatenate one CSV over several run directories, dropping rows computed twice (same identity columns).

    Args:
        run_dirs: Run directories (e.g. OMEGA Phase 1's grid, refinement and exemplar runs).
        name: CSV basename.

    Returns:
        The pooled rows (empty when no run has the file).
    """
    frames = [pd.read_csv(os.path.join(d, name)) for d in run_dirs if os.path.exists(os.path.join(d, name))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=[c for c in df.columns if c not in _PROVENANCE_COLUMNS])


def _n_channels(df: pd.DataFrame) -> int:
    """Distinct (subject, emg) channels of a frame."""
    return int(df[["subject", "emg"]].drop_duplicates().shape[0]) if not df.empty else 0


def _shared_legend(fig: Any, axes: Sequence[Any], extra: Sequence[Line2D] = (), ncol: int | None = None) -> None:
    """One legend below the figure, deduplicated over ``axes``, plus ``extra`` proxy handles."""
    handles: dict[str, Any] = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    for h in extra:
        handles.setdefault(h.get_label(), h)
    if handles:
        fig.legend(list(handles.values()), list(handles), loc="outside lower center",
                   ncol=ncol or min(len(handles), 4), frameon=False, fontsize=S.FONT_SIZES["legend"])


def _pending_proxy(model: str) -> Line2D:
    """Legend proxy of a model whose curve is withheld under G2."""
    st = S.model_style(model)
    return Line2D([], [], color=st.color, ls=st.linestyle, alpha=S.FAINT_ALPHA * 2,
                  label=f"{st.label}: {S.PENDING_RECOMPUTE_LABEL}")


# ---------------------------------------------------------------------------
# Multi-PFN benchmark table
# ---------------------------------------------------------------------------
def benchmark_table(
    tidy_by_dataset: dict[str, pd.DataFrame],
    models: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Final R^2 and exploration score per model and dataset, reduced as ``comparison_table.csv`` reduces them.

    Repetitions are reduced within a channel first (median for heavy-tailed R^2, mean otherwise); then R^2 is the
    median over channels with its interquartile range, a bounded metric the mean over channels with a 95% CI.

    Args:
        tidy_by_dataset: Dataset -> tidy rows of every model (several runs already pooled).
        models: Rows of the table, in order.
        metrics: Metric columns, in order.

    Returns:
        One row per (dataset, model, metric): ``centre``, ``lower``, ``upper``, ``n_channels``, ``n_reps``,
        ``current`` (G2 gate), ``acq_type``.

    Raises:
        ValueError: If a model appears with two different summaries in one dataset (two runs disagree on the
            same cells), or with several acquisitions.
    """
    records: list[dict[str, Any]] = []
    for dataset, tidy in tidy_by_dataset.items():
        gate = current_models(tidy)
        for model in models:
            rows = tidy[tidy["model"] == model]
            if rows.empty:
                continue
            acqs = sorted(set(rows["acq_type"].astype(str)))
            if len(acqs) != 1:
                raise ValueError(f"benchmark_table: {model} on {dataset} has several acquisitions {acqs}.")
            for metric in metrics:
                heavy = metric in ST.HEAVY_TAILED_METRICS
                per_channel = rows.groupby(["subject", "emg"])[metric].agg("median" if heavy else "mean")  # [C]
                vals = per_channel.to_numpy(dtype=float)
                if not np.isfinite(vals).all():
                    raise ValueError(f"benchmark_table: non-finite {metric} for {model} on {dataset}.")
                if heavy:
                    centre, lower, upper = np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75)
                else:
                    half = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                    centre, lower, upper = vals.mean(), vals.mean() - half, vals.mean() + half
                records.append({"dataset": dataset, "model": model, "metric": metric, "centre": float(centre),
                                "lower": float(lower), "upper": float(upper), "n_channels": int(len(vals)),
                                "n_reps": int(rows["rep"].nunique()), "current": gate[model],
                                "acq_type": acqs[0]})
    return pd.DataFrame.from_records(records)


def _load_benchmark(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Pool the tidy rows of every benchmark run per dataset; a model taken from two runs must agree row for row."""
    out: dict[str, pd.DataFrame] = {}
    for dataset in cfg["datasets"]:
        frames = [pd.read_csv(os.path.join(d, "tidy.csv"), low_memory=False) for d in cfg["runs"][dataset]]
        tidy = pd.concat(frames, ignore_index=True)
        keys = ["model", "subject", "emg", "rep", "acq_type"]
        dup = tidy[tidy.duplicated(subset=keys, keep=False)]
        for key, grp in dup.groupby(keys):
            if grp[list(cfg["metrics"])].nunique().max() > 1:
                raise ValueError(f"Benchmark runs of {dataset} disagree on the same cell {key}.")
        out[dataset] = tidy.drop_duplicates(subset=keys)
    return out


def _cell_text(row: pd.Series | None, metric: str, latex: bool) -> str:
    """One table cell: ``0.81 (0.70-0.84)`` for R^2, ``0.88 +- 0.05`` for a bounded metric, or pending."""
    if row is None:
        return "--" if latex else "–"
    if not row["current"]:
        return r"\textit{pending}" if latex else S.PENDING_RECOMPUTE_LABEL
    def num(v: float) -> str:
        return f"{v:.2f}".replace("-", "$-$" if latex else "−")

    if metric in ST.HEAVY_TAILED_METRICS:
        return f"{num(row['centre'])} [{num(row['lower'])}, {num(row['upper'])}]"
    pm = r"$\pm$" if latex else "±"
    return f"{num(row['centre'])} {pm} {row['upper'] - row['centre']:.2f}"


def _latex_text(text: str) -> str:
    """Escape the LaTeX specials of plain caption text (``%`` and ``_``; math stays in ``$...$``)."""
    return text.replace("%", r"\%").replace("_", r"\_")


def _write_latex(table: pd.DataFrame, models: Sequence[str], metrics: Sequence[str], datasets: Sequence[str],
                 titles: dict[str, str], path: str) -> None:
    """Write the table as LaTeX source for the AAAI-27 kit, ready to paste into the paper.

    Follows the kit's table rules (``AnonymousSubmission2027.tex``, *Tables* and *Table Captions*): 9 pt type
    (``\\small`` in the 10 pt class; 10 pt is the default and 9 pt the permitted minimum), no ``\\resizebox`` or other
    whole-table scaling, ``table*`` across both columns when the table is wider than one, the caption *under* the table
    in the kit's own caption style, and booktabs rules (recommended by the kit). The caption names every model
    adaptation in the table (guardrail G3, :data:`style.ADAPTATION_NOTES`).

    Args:
        table: Output of :func:`benchmark_table`.
        models: Rows, in order.
        metrics: Metric columns, in order.
        datasets: Dataset column groups, in order.
        titles: Dataset -> column-group title.
        path: Destination file (``.txt``: LaTeX source, the paper needs ``\\usepackage{booktabs}``).
    """
    ncol = len(datasets) * len(metrics)
    head = " & ".join(rf"\multicolumn{{{len(metrics)}}}{{c}}{{{titles[d]}}}" for d in datasets)
    rules = " ".join(rf"\cmidrule(lr){{{2 + i * len(metrics)}-{1 + (i + 1) * len(metrics)}}}"
                     for i in range(len(datasets)))
    sub = " & ".join(S.TABLE_METRIC_LABELS[m] for _ in datasets for m in metrics)
    lines = [r"% AAAI-27 kit table: 9 pt (\small), no resizing, caption under the table; needs \usepackage{booktabs}.",
             r"\begin{table*}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{l{'c' * ncol}}}", r"\toprule",
             f"Model & {head} \\\\", rules, f" & {sub} \\\\", r"\midrule"]
    for model in models:
        label = S.model_label(model)
        cells = []
        for d in datasets:
            for m in metrics:
                hit = table[(table["dataset"] == d) & (table["model"] == model) & (table["metric"] == m)]
                cells.append(_cell_text(hit.iloc[0] if len(hit) else None, m, latex=True))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    summaries = "; ".join(f"{S.TABLE_METRIC_LABELS[m]}: {S.TABLE_METRIC_SUMMARIES[m]}" for m in metrics)
    notes = "; ".join(f"{S.model_label(m)}: {S.ADAPTATION_NOTES[m]}" for m in models if m in S.ADAPTATION_NOTES)
    notes = f" Adaptations: {_latex_text(notes)}." if notes else ""
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Multi-PFN benchmark table. {_latex_text(summaries)}.{notes}}}",
              r"\label{tab:multi-pfn-benchmark}", r"\end{table*}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def render_benchmark_table(cfg: dict[str, Any], out_dir: str, formats: Sequence[str]) -> list[str]:
    """*Multi-PFN benchmark table*: rendered at double-column width, plus ``.txt`` (AAAI-27 LaTeX source) and ``.csv``.

    Args:
        cfg: The ``benchmark`` block of the workshop config (``dataset_titles`` names the column groups).
        out_dir: Destination directory.
        formats: Figure formats.

    Returns:
        Written paths.
    """
    models, metrics, datasets = list(cfg["models"]), list(cfg["metrics"]), list(cfg["datasets"])
    titles = {d: str(cfg["dataset_titles"][d]) for d in datasets}
    table = benchmark_table(_load_benchmark(cfg), models, metrics)
    name = FIGURE_NAMES["benchmark"]
    os.makedirs(out_dir, exist_ok=True)
    csv_path, txt_path = os.path.join(out_dir, f"{name}.csv"), os.path.join(out_dir, f"{name}.txt")
    table.to_csv(csv_path, index=False)
    _write_latex(table, models, metrics, datasets, titles, txt_path)

    ncol = len(datasets) * len(metrics)
    width = S.FIG_WIDTHS["double"]
    height = _TABLE_HEADER_HEIGHT + _TABLE_ROW_HEIGHT * (len(models) + 2.5)         # rows + rules + the metric line
    S.apply_style()
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)                                                       # inches, top-down
    col_w = (width - _TABLE_LABEL_WIDTH) / ncol
    fs = S.FONT_SIZES["annotation"]
    rule = {"color": "black", "lw": S.LINE_WIDTH}
    y = _TABLE_ROW_HEIGHT * 0.5
    ax.plot([0, width], [y, y], **rule)
    y += _TABLE_ROW_HEIGHT * 0.75
    for i, d in enumerate(datasets):
        x0 = _TABLE_LABEL_WIDTH + i * len(metrics) * col_w
        ax.text(x0 + len(metrics) * col_w / 2, y, titles[d], ha="center", va="center",
                fontsize=fs, fontweight="bold")
        ax.plot([x0 + 0.05, x0 + len(metrics) * col_w - 0.05], [y + 0.11, y + 0.11], color="black", lw=0.5)
    ax.text(0.02, y + _TABLE_ROW_HEIGHT * 0.5, "Model", ha="left", va="center", fontsize=fs, fontweight="bold")
    y += _TABLE_ROW_HEIGHT * 1.0
    for j in range(ncol):
        metric = metrics[j % len(metrics)]
        ax.text(_TABLE_LABEL_WIDTH + (j + 0.5) * col_w, y, S.TABLE_METRIC_LABELS[metric], ha="center", va="center",
                fontsize=fs)
    y += _TABLE_ROW_HEIGHT * 0.6
    ax.plot([0, width], [y, y], color="black", lw=0.5)
    for model in models:
        y += _TABLE_ROW_HEIGHT
        ax.text(0.02, y, S.model_label(model), ha="left", va="center", fontsize=fs)
        for i, d in enumerate(datasets):
            for k, m in enumerate(metrics):
                hit = table[(table["dataset"] == d) & (table["model"] == model) & (table["metric"] == m)]
                row = hit.iloc[0] if len(hit) else None
                ax.text(_TABLE_LABEL_WIDTH + (i * len(metrics) + k + 0.5) * col_w, y, _cell_text(row, m, False),
                        ha="center", va="center", fontsize=fs,
                        style="italic" if row is not None and not row["current"] else "normal")
    y += _TABLE_ROW_HEIGHT * 0.6
    ax.plot([0, width], [y, y], **rule)
    summaries = "; ".join(f"{S.TABLE_METRIC_LABELS[m]}: {S.TABLE_METRIC_SUMMARIES[m]}" for m in metrics)
    y += _TABLE_ROW_HEIGHT
    ax.text(0.02, y, summaries + ".", ha="left", va="center", fontsize=fs - 0.5)
    return [csv_path, txt_path, *S.save_figure(fig, out_dir, name, formats=formats)]


# ---------------------------------------------------------------------------
# NHP stress regimes (in vivo)
# ---------------------------------------------------------------------------
def render_stress_regimes(cfg: dict[str, Any], out_dir: str, formats: Sequence[str]) -> list[str]:
    """*NHP stress regimes (in vivo)*: one row per metric, one column per knob, degradation vs the knob's x-axis.

    Rows share their y-axis and columns their x-axis; the knob title sits on the top row only and the y label on the
    first column only. No breakdown point is marked unless ``cfg['breakdown']`` names one: the breakdown metric is
    left open until the recomputed results are read (user, 2026-11-05).

    Args:
        cfg: The ``stress`` block of the workshop config (``knob_titles`` names the columns, ``default_marker_knobs``
            the knobs whose panels mark the default regime, ``annotate_clip`` the "axis clipped" note).
        out_dir: Destination directory.
        formats: Figure formats.

    Returns:
        Written paths.
    """
    knobs, metrics = dict(cfg["knobs"]), list(cfg["metrics"])
    fig, axes = S.figure(cfg["width"], nrows=len(metrics), ncols=len(knobs), squeeze=False, sharex="col",
                         sharey="row", panel_aspect=float(cfg["panel_aspect"]), layout=S.LAYOUT_ENGINE)
    pending: list[str] = []
    for c, (knob, run_dir) in enumerate(knobs.items()):
        raw = pd.read_csv(os.path.join(run_dir, "tidy.csv"), low_memory=False)
        gate = current_models(raw)
        shown = [m for m in cfg["models"] if gate.get(m, False)]
        pending += [m for m in cfg["models"] if m in gate and not gate[m] and m not in pending]
        df, _ = ST.prepare_sweep_frame(raw[raw["model"].isin(shown)], cfg.get("min_snr_db", ST.DEFAULT_MIN_SNR_DB))
        breakdowns = None
        if cfg.get("breakdown"):
            table, _ = ST.build_robustness_table(df, out_dir, knob=knob, margin=float(cfg["breakdown"]["margin"]),
                                                 metric=str(cfg["breakdown"]["metric"]),
                                                 reference=str(cfg["breakdown"]["reference"]))
            breakdowns = {str(m): float(x) for m, x in zip(table["model"], table["breakdown_x"]) if np.isfinite(x)}
        marked = knob in cfg["default_marker_knobs"]
        for r, metric in enumerate(metrics):
            ax = axes[r][c]
            ST.degradation_panel(ax, df, knob=knob, metric=metric, models=shown, breakdowns=breakdowns,
                                 show_channels=bool(cfg["show_channels"]), label_nominal=marked and r == 0,
                                 mark_default=marked, annotate_clip=bool(cfg["annotate_clip"]))
            # one-line labels; the summary (median / IQR, mean / 95% CI) goes in the caption
            ax.set_ylabel(S.axis_label(metric) if c == 0 else "")
            if r < len(metrics) - 1:
                ax.set_xlabel("")                                                     # the column's x label sits below
        axes[0][c].set_title(str(cfg["knob_titles"][knob]), fontsize=S.FONT_SIZES["legend"])
    for idx, ax in enumerate(axes.ravel()):                                           # reading order, row-major
        S.panel_letters([ax], start=idx, x=-0.30 if idx % len(knobs) == 0 else -0.08, y=1.06)
    _shared_legend(fig, axes.ravel(), extra=[_pending_proxy(m) for m in pending], ncol=3)
    return S.save_figure(fig, out_dir, FIGURE_NAMES["stress"], formats=formats)


# ---------------------------------------------------------------------------
# Is the update GP-like?
# ---------------------------------------------------------------------------
def _nearest(available: Sequence[int], wanted: int) -> int:
    """The available context size closest to ``wanted``."""
    return int(min(available, key=lambda t: abs(int(t) - wanted)))


def _exemplar(run_dirs: Sequence[str], t: int) -> tuple[Any, int] | None:
    """The C1 exemplar holding context size ``t`` (else the closest one), and the size it holds."""
    found = []
    for d in run_dirs:
        path = os.path.join(d, "c1_exemplar.npz")
        if os.path.exists(path):
            ex = np.load(path)
            found.append((ex, _nearest(list(ex["context_ts"]), t)))
    if not found:
        return None
    return min(found, key=lambda e: abs(e[1] - t))


def _map_legend_handles(labels: dict[str, str]) -> list[Line2D]:
    """Legend proxies of the exemplar-map markers drawn by :func:`mechanism.exemplar_map`.

    Args:
        labels: ``observed`` (context-site dots) and ``anchor`` (the ringed anchor) legend texts.

    Returns:
        Two handles, in that order, sized like the scatter markers.
    """
    return [Line2D([], [], ls="", marker=".", color=S.NEUTRAL_GREY, markersize=np.sqrt(S.MARKER_SIZE),
                   label=labels["observed"]),
            Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=S.NEUTRAL_GREY,
                   markeredgewidth=S.LINE_WIDTH, markersize=np.sqrt(S.MARKER_SIZE * 4), label=labels["anchor"])]


def render_gp_like(cfg: dict[str, Any], out_dir: str, formats: Sequence[str]) -> list[str]:
    """*Is the update GP-like?*: (a) exemplar change maps at one anchor across the top row; (b) surprise response and
    (c) update vs distance at the exemplar's context size below. A 2 x 2 layout: the footprint-correlation panel was
    dropped (user, 2026-11-07; backlogged in the roadmap), since it adds no view the other panels do not give.

    Args:
        cfg: The ``gp_like`` block of the workshop config. The shared change-of-mean colour bar sits in a column
            ``middle_width_ratio`` wide between the first two maps (user, 2026-11-07), at ``colorbar_box``
            (``[x0, y0, width, height]`` in that column's axes fraction); the map-marker entries (``map_legend``
            words them) join the figure's shared legend, ``legend_ncol`` entries per row. ``map_engines`` are the maps
            of (a); ``engines`` the lines of (b) and (c) (user, 2026-11-07: GP-MLL refit joins (b) and (c), not the
            maps).
        out_dir: Destination directory.
        formats: Figure formats.

    Returns:
        Written paths.
    """
    runs, engines, t = list(cfg["runs"]), list(cfg["engines"]), int(cfg["t"])
    map_engines = list(cfg["map_engines"])  # panel (a)'s maps; (b) and (c) draw every engine of cfg["engines"]
    expected = int(cfg["expected_channels"])
    S.apply_style()
    width = S.FIG_WIDTHS[cfg["width"]]
    fig = plt.figure(figsize=(width, width * float(cfg["height_ratio"])), layout=S.LAYOUT_ENGINE)
    top, bottom = fig.subfigures(2, 1, height_ratios=list(cfg["row_heights"]))
    # the colour bar sits in a narrow column between the first two maps
    ratios = [1.0, float(cfg["middle_width_ratio"]), *([1.0] * (len(map_engines) - 1))]
    row_axes = np.atleast_1d(top.subplots(1, len(map_engines) + 1, width_ratios=ratios))
    middle_ax, map_axes = row_axes[1], np.array([row_axes[0], *row_axes[2:]])
    middle_ax.set_axis_off()

    hit = _exemplar(runs, t)
    if hit is None:
        for ax in map_axes:
            _placeholder(ax, f"awaiting the C1 exemplar at t = {t}")
    else:
        ex, t_ex = hit
        names = [str(e) for e in ex["engines"]]
        ti = list(ex["context_ts"]).index(t_ex)
        xy, a = ex["ch2xy"].astype(int), int(ex["anchor"])
        dmu = {e: ex["dmu"][ti, names.index(e)] for e in map_engines if e in names}                  # [M] each
        vmax = max(float(np.abs(v).max()) for v in dmu.values()) or 1.0          # one scale: magnitudes compare
        im = None
        for ax, engine in zip(map_axes, map_engines):
            if engine not in dmu:
                _placeholder(ax, f"{M.arm_style(engine).label} not in the exemplar")
                continue
            im = M.exemplar_map(ax, dmu[engine], xy, ex["observed"][ti], a, vmax)
            ax.set_title(f"{M.arm_style(engine).label}, $t$ = {t_ex}", fontsize=S.FONT_SIZES["legend"])
        if im is not None:
            fig.colorbar(im, cax=middle_ax.inset_axes(tuple(float(v) for v in cfg["colorbar_box"])),
                         label=S.axis_label("dmu_exemplar" + str(cfg["label_suffix"])))
        if t_ex != t:
            _stamp(map_axes[0], f"t = {t_ex} stand-in for t = {t}")
    S.panel_letters(map_axes[:1], x=float(cfg["map_letter_x"]))

    ax_b, ax_c = bottom.subplots(1, 2)
    sur, prof = (_read_runs(runs, f) for f in ("surprise.csv", "transfer_profile.csv"))
    # (x, y) axis-label keys of each panel; label_suffix picks a style variant (e.g. "_compact" at one column)
    panels = ((ax_b, sur, M.surprise_panel, ("ladder_c", "update_anchor_sd")),
              (ax_c, prof, M.distance_panel, ("distance_pitch", "abs_transfer")))
    for ax, frame, draw, (xkey, ykey) in panels:
        if frame.empty:
            _placeholder(ax, "awaiting OMEGA Phase 1")
            continue
        t_avail = _nearest(sorted(frame["context_t"].unique()), t)
        draw(ax, frame, t_avail, engines)
        ax.set_xlabel(S.axis_label(xkey + str(cfg["label_suffix"])))
        ax.set_ylabel(S.axis_label(ykey + str(cfg["label_suffix"])))
        n = _n_channels(frame[frame["context_t"] == t_avail])
        if t_avail != t or n < expected:
            _stamp(ax, f"{n} channel(s), t = {t_avail}")
    S.panel_letters([ax_b, ax_c], start=1, x=float(cfg["readout_letter_xy"][0]), y=float(cfg["readout_letter_xy"][1]))
    markers = _map_legend_handles(dict(cfg["map_legend"]))
    _shared_legend(fig, [ax_b, ax_c], extra=markers, ncol=int(cfg["legend_ncol"]))
    return S.save_figure(fig, out_dir, FIGURE_NAMES["gp_like"], formats=formats)


# ---------------------------------------------------------------------------
# Is the update coherent?
# ---------------------------------------------------------------------------
def render_coherence(cfg: dict[str, Any], out_dir: str, formats: Sequence[str]) -> list[str]:
    """*Is the update coherent?*: Part A coherence readouts vs context size, one row of panels per figure variant.

    Each variant of ``cfg["variants"]`` is one figure, written as the item's name plus the variant's ``suffix``:
    the moment readouts (mean drift, total-variance ratio, implied-covariance asymmetry; second moments under the run's
    tail measure, central, G8 convention C) and the CDF readouts that G8 makes the headline (Kolmogorov distance and
    grid total variation of the martingale gap, identical under every tail convention). Each panel is the median over
    channels with its channel-bootstrap interval per engine, against the value every coherent Bayesian model scores
    (drawn by :func:`mechanism.coherence_panel`, as in the legacy ``coherence.svg``). Pooled anchors at the nominal
    level. A variant's optional ``ylim`` (metric -> ``[low, high]``) fixes that panel's y range; the block's optional
    ``xticks`` replaces the one-tick-per-context-size default, whose labels collide once t spans 30-160.

    Args:
        cfg: The ``coherence`` block of the workshop config.
        out_dir: Destination directory.
        formats: Figure formats.

    Returns:
        Written paths.
    """
    engines = [str(e) for e in cfg["engines"]]
    cell = _read_runs(list(cfg["runs"]), "coherence_cell.csv")
    if not cell.empty:
        cell = cell[(cell["pool"] == M.POOLED_ANCHORS) & np.isclose(cell["level"], cell["level"].min())]
    written: list[str] = []
    for variant in cfg["variants"]:
        metrics = list(variant["metrics"])
        fig, axes = S.figure(variant["width"], nrows=1, ncols=len(metrics), squeeze=False,
                             panel_aspect=float(variant["panel_aspect"]), layout=S.LAYOUT_ENGINE)
        for ax, metric in zip(axes[0], metrics):
            if cell.empty or metric not in cell or cell[metric].isna().all():
                _placeholder(ax, "awaiting OMEGA Phase 1")
                ax.set_ylabel(S.axis_label(metric))
                continue
            M.coherence_panel(ax, cell, metric, engines=engines, yscale=str(variant["yscale"]))
            if metric in (variant.get("ylim") or {}):
                ax.set_ylim(*map(float, variant["ylim"][metric]))
            if cfg.get("xticks") is not None:
                ax.set_xticks([float(t) for t in cfg["xticks"]], [str(t) for t in cfg["xticks"]])
            n = _n_channels(cell)
            if n < int(cfg["expected_channels"]) or cell["context_t"].nunique() < 2:
                _stamp(ax, f"{n} channel(s), t = {sorted(cell['context_t'].unique())}")
        S.panel_letters(axes[0], x=float(variant["letter_x"]))
        _shared_legend(fig, axes[0], ncol=len(engines))
        written += S.save_figure(fig, out_dir, FIGURE_NAMES["coherence"] + str(variant["suffix"]), formats=formats)
    return written


# ---------------------------------------------------------------------------
# Supplement: what the representation encodes (BETA)
# ---------------------------------------------------------------------------
def _read_nominal(path: str) -> pd.DataFrame:
    """A BETA summary CSV at its nominal (lowest) stress level; empty when the file is missing or empty."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "level" not in df:
        return df
    return df[np.isclose(df["level"], df["level"].min())]


def _beta2_panel(ax: Any, run: str, cfg: dict[str, Any]) -> None:
    """β2: label-token alignment with one belief target vs block, one trace per t, surrogate floor as a grey band."""
    summary = _read_nominal(os.path.join(run, "representation_summary.csv"))
    if summary.empty:
        _placeholder(ax, "awaiting BETA β2")
        return
    sub = summary[(summary["site_set"] == cfg["site_set"]) & (summary["target"] == cfg["target"])]
    ts = sorted(sub["context_t"].unique())
    M._alignment_panel(ax, sub, [str(cfg["readout"])], ts, value=str(cfg["value"]), floor=str(cfg["floor"]),
                       legend=True)
    with open(os.path.join(run, "headline.json"), encoding="utf-8") as fh:
        statistic = json.load(fh)["statistic"]
    ax.set_title(S.BELIEF_TARGET_LABELS.get(cfg["target"], cfg["target"]), fontsize=S.FONT_SIZES["base"])
    ax.set_xlabel(S.axis_label("layer"))
    ax.set_ylabel(S.axis_label(f"headline_{statistic}"))
    ax.set_ylim(*map(float, cfg["ylim"]))
    ax.legend(loc=str(cfg["legend_loc"]), fontsize=S.FONT_SIZES["legend"], ncol=int(cfg["legend_ncol"]))


def _beta1a_panel(ax: Any, run: str, cfg: dict[str, Any]) -> None:
    """β1a: per-site W2 between TabPFN-2.5 and each GP vs t, the GP-fixed vs GP-MLL refit yardstick in grey."""
    sm = _read_nominal(os.path.join(run, "site_match_summary.csv"))
    if sm.empty:
        _placeholder(ax, "awaiting BETA β1a")
        return
    sm = sm[(sm["site_pool"] == cfg["site_pool"]) & (sm["value"] == cfg["value"])]
    own, yard = sm[sm["model"] == M.BELIEF_CODES["own"]], sm[sm["model"] != M.BELIEF_CODES["own"]]
    M.reference_summary_panel(ax, own, str(cfg["value"]), list(dict.fromkeys(own["reference"])),
                              label_fmt=f"{M.arm_style(M.BELIEF_CODES['own']).label} vs {{label}}")
    for (model, ref), part in yard.groupby(["model", "reference"], sort=False):
        part = part.sort_values("context_t")
        ax.plot(part["context_t"], part["median"], color=S.NEUTRAL_GREY, ls=":", marker="o",
                label=f"{M.arm_style(model).label} vs {M.arm_style(ref).label}")
        ax.fill_between(part["context_t"], part["ci_low"], part["ci_high"], color=S.NEUTRAL_GREY,
                        alpha=S.BAND_ALPHA, lw=0)
    ax.set_xscale(str(cfg["xscale"]))
    ax.set_yscale(str(cfg["yscale"]))
    ts = sorted(sm["context_t"].unique())
    ax.set_xticks([float(t) for t in ts], [str(int(t)) for t in ts])
    # W2 spans less than a decade, so a log axis would show no labelled tick: label the configured ones instead
    ax.set_yticks([float(v) for v in cfg["yticks"]], [f"{float(v):g}" for v in cfg["yticks"]])
    ax.minorticks_off()
    ax.set_xlabel(S.axis_label("context_t"))
    ax.set_ylabel(S.axis_label(M.SITE_MATCH_STATS[str(cfg["value"])]))
    ax.legend(loc="best", fontsize=S.FONT_SIZES["legend"])


def _beta1b_panel(ax: Any, run: str, cfg: dict[str, Any]) -> None:
    """β1b: alignment of TabPFN-2.5's implied covariance with each reference covariance vs t."""
    pc = _read_nominal(os.path.join(run, "posterior_cov_summary.csv"))
    if pc.empty:
        _placeholder(ax, "awaiting BETA β1b")
        return
    pc = pc[pc["engine"] == cfg["engine"]]
    M.reference_summary_panel(ax, pc, str(cfg["value"]), [str(r) for r in cfg["references"]],
                              label_fmt="vs {label}")
    ts = sorted(pc["context_t"].unique())
    ax.set_xticks([float(t) for t in ts], [str(int(t)) for t in ts])
    ax.axhline(0.0, color=S.NEUTRAL_GREY, lw=S.LINE_WIDTH / 2, zorder=0)
    ax.set_xlabel(S.axis_label("context_t"))
    ax.set_ylabel(S.axis_label(str(cfg["ylabel"])))
    ax.legend(loc="best", fontsize=S.FONT_SIZES["legend"])


def _lens_panel(ax: Any, cfg: dict[str, Any]) -> None:
    """Zero-parameter vs tuned lens: one fidelity metric vs block at one t, excluded blocks shaded."""
    summary = _read_nominal(os.path.join(str(cfg["dir"]), "summary.csv"))
    if summary.empty:
        _placeholder(ax, "awaiting the tuned lens")
        return
    part = summary[(summary["context_t"] == int(cfg["t"])) & (summary["value"] == cfg["metric"])]
    blocks = list(M.LENS_EXCLUDED_BLOCKS)
    ax.axvspan(min(blocks) - 0.5, max(blocks) + 0.5, color=S.NEUTRAL_GREY, alpha=S.FAINT_ALPHA, lw=0)
    for variant, label in cfg["variants"].items():
        s = part[part["variant"] == variant].sort_values("layer")
        if cfg["yscale"] == "log":
            # the zero-parameter lens IS the final readout at the last block (KL exactly 0): no point on a log axis
            s = s[s["median"] > 0.0]
        control = variant in cfg["controls"]
        color = S.NEUTRAL_GREY if control else S.model_color("tabpfn_v2_5")
        ls = "--" if control else S.model_style("tabpfn_v2_5").linestyle
        ax.fill_between(s["layer"], s["ci_low"], s["ci_high"], color=color, alpha=S.BAND_ALPHA, lw=0)
        ax.plot(s["layer"], s["median"], color=color, ls=ls, marker=S.model_style("tabpfn_v2_5").marker,
                markersize=S.MARKER_SIZE, markerfacecolor="none" if control else color, label=str(label))
    ax.set_yscale(str(cfg["yscale"]))
    ax.set_title(f"$t$ = {int(cfg['t'])}", fontsize=S.FONT_SIZES["base"])
    ax.set_xlabel(S.axis_label("layer"))
    ax.set_ylabel(S.axis_label(str(cfg["metric"])))
    S.set_integer_xaxis(ax)
    ax.legend(loc="best", fontsize=S.FONT_SIZES["legend"])


def render_beta_supplement(cfg: dict[str, Any], out_dir: str, formats: Sequence[str]) -> list[str]:
    """Supplement *What the representation encodes* (BETA): a 2 x 2 template from the amended BETA run.

    (a) β2: label-token alignment with a belief target vs block (headline statistic of ``headline.json``,
    geometry removed, read against the smooth-field surrogate; G6: decodability, never "GP-like" from a gap).
    (b) β1a: per-site W2 between TabPFN-2.5 and each GP vs t (log-log, so a power-law decrease is a line), with the
    GP-fixed vs GP-MLL refit yardstick. (c) β1b: CKA of TabPFN's implied covariance with each reference covariance
    vs t. (d) zero-parameter vs tuned lens fidelity vs block at one t. Every panel reads an existing summary CSV;
    nothing is recomputed. A missing input leaves a placeholder.

    Args:
        cfg: The ``beta_supplement`` block of the workshop config.
        out_dir: Destination directory.
        formats: Figure formats.

    Returns:
        Written paths.
    """
    run = str(cfg["run"])
    fig, axes = S.figure(cfg["width"], nrows=2, ncols=2, squeeze=False, panel_aspect=float(cfg["panel_aspect"]),
                         layout=S.LAYOUT_ENGINE)
    _beta2_panel(axes[0][0], run, cfg["beta2"])
    _beta1a_panel(axes[0][1], run, cfg["beta1a"])
    _beta1b_panel(axes[1][0], run, cfg["beta1b"])
    _lens_panel(axes[1][1], cfg["lens"])
    S.panel_letters([ax for row in axes for ax in row], x=float(cfg["letter_x"]))
    return S.save_figure(fig, out_dir, FIGURE_NAMES["beta_supplement"], formats=formats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
RENDERERS = {"benchmark": render_benchmark_table, "stress": render_stress_regimes, "gp_like": render_gp_like,
             "coherence": render_coherence, "beta_supplement": render_beta_supplement}


def render_all(config_path: str, only: Sequence[str] | None = None) -> list[str]:
    """Render every workshop item named in the config (or ``only`` those).

    Args:
        config_path: Workshop YAML (``configs/workshop/*.yaml``).
        only: Item keys of :data:`FIGURE_NAMES` to render (``None`` = every item the config has).

    Returns:
        Written paths.

    Raises:
        KeyError: If ``only`` names an unknown item.
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    items = list(only) if only else [k for k in RENDERERS if k in cfg]
    unknown = [k for k in items if k not in RENDERERS]
    if unknown:
        raise KeyError(f"Unknown workshop item(s) {unknown}; expected {list(RENDERERS)}.")
    written: list[str] = []
    for key in items:
        written += RENDERERS[key](cfg[key], cfg["out_dir"], tuple(cfg["formats"]))
    return written
