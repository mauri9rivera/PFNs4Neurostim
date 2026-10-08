"""Export the headline summary tables (NHP, 5d_rat, spinal) as vector and raster images.

Reads the pooled tidy table written by ``scripts/aggregate_runs.py`` (task plan A8), summarises the final
recommended-site simple regret and exploration score (mean over channels, 95% CI half-width) and the final
R^2 (median over channels, interquartile range -- heavy-tailed, as in ``comparison_table.csv``), bolds the
best centre per metric within each table, and writes ``summary_tables.{pdf,svg,png}``.

    python scripts/aggregate_runs.py
    python scripts/export_summary_tables.py
    python scripts/export_summary_tables.py --pooled output/aggregated/pooled.csv --out-dir output/summary_tables
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from pfns4neurostim.evaluation.pooling import channel_summary
from pfns4neurostim.visualization import style as S

#: Metric column, display header, whether a larger value is better, and whether it is heavy-tailed.
METRICS: tuple[tuple[str, str, bool, bool], ...] = (
    ("recommended_regret", "Simple regret", False, False),
    ("exploration_score", "Exploration score", True, False),
    ("r2", "$R^2$", True, True),
)
FORMATS: tuple[str, ...] = ("pdf", "svg", "png")
BOLD_WEIGHT: str = "bold"
HEADER_FACE: str = "#EEEEEE"
DEFAULT_POOLED: str = "output/aggregated/pooled.csv"
ROW_HEIGHT_IN: float = 0.30   # figure height per table row (inches)
TABLE_PAD_IN: float = 0.6     # extra height for the caption


@dataclass(frozen=True)
class RowSpec:
    """One table row: display name and the pooled rows it summarises.

    Attributes:
        label: Row label.
        run_dir: ``run_dir`` value in the pooled table (relative to ``output/``).
        model: Model key.
        acq_label: Acquisition label.
    """

    label: str
    run_dir: str
    model: str
    acq_label: str


NHP_ACQ = "benchmark/nhp/hyp0-acq-core-nhp"
NHP_PFN = "benchmark/nhp/hyp0-pfn-bench-nhp"
RAT_A = "benchmark/5d_rat/hyp-a-5d_rat"
RAT_PFN = "benchmark/5d_rat/hyp0-pfn-bench-5d_rat"
SPI_A = "benchmark/spinal/hyp-a-spinal"
SPI_PFN = "benchmark/spinal/hyp0-pfn-bench-spinal"

TABLES: dict[str, tuple[RowSpec, ...]] = {
    "NHP (96 sites, budget 96)": (
        RowSpec("TabPFN-2.5 (TS)", NHP_ACQ, "tabpfn_v2_5", "ts_marginal"),
        RowSpec("GP-fixed (TS)", NHP_ACQ, "gp_naive", "ts_marginal"),
        RowSpec("GP-MLL (TS)", NHP_ACQ, "gp_mll", "ts_marginal"),
        RowSpec("TabICL (TS)", NHP_PFN, "tabicl", "ts_marginal"),
        RowSpec("TabPFN-2.5 (EI)", NHP_ACQ, "tabpfn_v2_5", "ei"),
        RowSpec("GP-MLL (EI)", NHP_ACQ, "gp_mll", "ei"),
    ),
    "5d_rat (budget 100)": (
        RowSpec("TabPFN-2.5 (TS)", RAT_A, "tabpfn_v2_5", "ts_marginal"),
        RowSpec("GP-fixed (TS)", RAT_A, "gp_naive", "ts_marginal"),
        RowSpec("GP-MLL (TS)", RAT_A, "gp_mll", "ts_marginal"),
        RowSpec("TabICL (TS)", RAT_PFN, "tabicl", "ts_marginal"),
    ),
    "Spinal (64 sites, budget 64)": (
        RowSpec("TabPFN-2.5 (TS)", SPI_A, "tabpfn_v2_5", "ts_marginal"),
        RowSpec("GP-fixed (TS)", SPI_A, "gp_naive", "ts_marginal"),
        RowSpec("GP-MLL (TS)", SPI_A, "gp_mll", "ts_marginal"),
        RowSpec("TabICL (TS)", SPI_PFN, "tabicl", "ts_marginal"),
    ),
}


def load_table(pooled: pd.DataFrame, rows: tuple[RowSpec, ...]) -> pd.DataFrame:
    """Summarise the requested rows of the pooled table.

    Args:
        pooled: Pooled tidy table.
        rows: Row specifications.

    Returns:
        Frame indexed by row label with ``<metric>`` and its band columns, ``n_channels`` and ``n_reps``.

    Raises:
        KeyError: If a requested (run, model, acquisition) has no pooled rows.
    """
    records: list[pd.Series] = []
    for spec in rows:
        sel = pooled[(pooled["run_dir"] == spec.run_dir) & (pooled["model"] == spec.model)
                     & (pooled["acq_label"] == spec.acq_label)]
        if sel.empty:
            raise KeyError(f"{spec.label}: no pooled rows for ({spec.run_dir}, {spec.model}, {spec.acq_label}).")
        parts = [channel_summary(sel, ["model"], col, robust).iloc[0] for col, _, _, robust in METRICS]
        rec = pd.concat(parts)
        rec = rec[~rec.index.duplicated()]
        rec["label"] = spec.label
        records.append(rec)
    return pd.DataFrame(records).set_index("label")


def _cell(row: pd.Series, col: str, robust: bool) -> str:
    """Format one metric cell: ``mean ± ci`` or ``median [q25, q75]``."""
    if robust:
        return f"{row[col]:.3f} [{row[f'{col}_q25']:.2f}, {row[f'{col}_q75']:.2f}]"
    return f"{row[col]:.3f} ± {row[f'{col}_ci']:.3f}"


def draw_table(ax, title: str, data: pd.DataFrame) -> None:
    """Draw one table on ``ax`` with the best centre per metric in bold.

    Args:
        ax: Matplotlib axes.
        title: Table title.
        data: Output of :func:`load_table`.
    """
    ax.axis("off")
    channels = sorted(set(int(n) for n in data["n_channels"]))
    reps = sorted(set(int(n) for n in data["n_reps"]))
    ax.set_title(f"{title} — {'/'.join(map(str, channels))} channels × {'/'.join(map(str, reps))} reps",
                 loc="left", fontsize=S.FONT_SIZES["title"], fontweight=BOLD_WEIGHT)
    best = {col: (data[col].idxmax() if higher else data[col].idxmin()) for col, _, higher, _ in METRICS}
    cell_text = [[_cell(r, col, robust) for col, _, _, robust in METRICS] for _, r in data.iterrows()]
    tab = ax.table(
        cellText=cell_text,
        rowLabels=list(data.index),
        colLabels=[header for _, header, _, _ in METRICS],
        cellLoc="center",
        loc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(S.FONT_SIZES["base"])
    tab.scale(1.0, 1.35)
    for (row, col), cell in tab.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor(HEADER_FACE)
            cell.set_text_props(fontweight=BOLD_WEIGHT)
        elif col >= 0 and row >= 1:
            metric = METRICS[col][0]
            if data.index[row - 1] == best[metric]:
                cell.set_text_props(fontweight=BOLD_WEIGHT)


def main() -> None:
    """Build and save the summary-tables figure."""
    parser = argparse.ArgumentParser(description="Export the headline summary tables as an image.")
    parser.add_argument("--pooled", default=DEFAULT_POOLED, help="Pooled table from scripts/aggregate_runs.py.")
    parser.add_argument("--out-dir", default="output/summary_tables")
    args = parser.parse_args()

    pooled = pd.read_csv(args.pooled, low_memory=False)
    loaded = {title: load_table(pooled, rows) for title, rows in TABLES.items()}
    n_rows = sum(len(d) + 2 for d in loaded.values())            # table rows + header + title per table
    height = ROW_HEIGHT_IN * n_rows + TABLE_PAD_IN
    width = S.FIG_WIDTHS["double"]
    aspect = (height - S.DECORATION_HEIGHT) / (width * len(loaded))  # style.figure sizes height by panel aspect
    fig, axes = S.figure("double", nrows=len(loaded), ncols=1, panel_aspect=aspect)
    for ax, (title, data) in zip(axes, loaded.items()):
        draw_table(ax, title, data)
    fig.text(
        0.01, 0.005,
        "Final value at the end of the run, repetitions averaged within a channel. Simple regret and exploration "
        "score: mean ± 95% CI over channels; $R^2$: median [IQR] over channels (heavy-tailed). Bold = best centre "
        "per metric (no significance test). TS = ts_marginal; EI = expected improvement.",
        fontsize=S.FONT_SIZES["annotation"], va="bottom", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for path in S.save_figure(fig, args.out_dir, "summary_tables", formats=FORMATS):
        print(path)


if __name__ == "__main__":
    main()
