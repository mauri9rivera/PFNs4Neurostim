"""Export the two headline summary tables (NHP and 5d_rat) as vector and raster images.

Reads the ``comparison_table.csv`` of the benchmark runs (final simple regret, exploration score and R^2,
mean over channels with a 95% CI half-width), bolds the best mean per metric within each table, and
writes ``summary_tables.{pdf,svg,png}``.

    python scripts/export_summary_tables.py
    python scripts/export_summary_tables.py --out-dir output/summary_tables
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from pfns4neurostim.visualization import style as S

#: Metric column, display header, and whether a larger value is better.
METRICS: tuple[tuple[str, str, bool], ...] = (
    ("simple_regret_final", "Simple regret", False),
    ("exploration_score_final", "Exploration score", True),
    ("r2_final", "$R^2$", True),
)
FORMATS: tuple[str, ...] = ("pdf", "svg", "png")
BOLD_WEIGHT: str = "bold"
HEADER_FACE: str = "#EEEEEE"


@dataclass(frozen=True)
class RowSpec:
    """One table row: display name and where its numbers come from.

    Attributes:
        label: Row label.
        table_csv: ``comparison_table.csv`` holding the row.
        model: Model key in that table.
        acq_label: Acquisition label in that table.
    """

    label: str
    table_csv: str
    model: str
    acq_label: str


NHP_ACQ = "output/benchmark/nhp/hyp0-acq-core-nhp/comparison_table.csv"
NHP_PFN = "output/mila/benchmark/nhp/hyp0-pfn-bench-nhp/comparison_table.csv"
RAT_D1 = "output/mila/benchmark/5d_rat/hyp-a-5d_rat/comparison_table.csv"
RAT_PFN = "output/mila/benchmark/5d_rat/hyp0-pfn-bench-5d_rat/comparison_table.csv"

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
        RowSpec("TabPFN-2.5 (TS)", RAT_D1, "tabpfn_v2_5", "ts_marginal"),
        RowSpec("GP-fixed (TS)", RAT_D1, "gp_naive", "ts_marginal"),
        RowSpec("GP-MLL (TS)", RAT_D1, "gp_mll", "ts_marginal"),
        RowSpec("TabICL (TS)", RAT_PFN, "tabicl", "ts_marginal"),
    ),
}


def load_table(rows: tuple[RowSpec, ...]) -> pd.DataFrame:
    """Collect the metric means and CIs of the requested rows.

    Args:
        rows: Row specifications.

    Returns:
        Frame indexed by row label with ``<metric>`` and ``<metric>_ci`` columns.

    Raises:
        KeyError: If a requested (model, acquisition) is absent from its comparison table.
    """
    records: list[dict[str, float | str]] = []
    cache: dict[str, pd.DataFrame] = {}
    for spec in rows:
        table = cache.setdefault(spec.table_csv, pd.read_csv(spec.table_csv).set_index(["model", "acq_label"]))
        if (spec.model, spec.acq_label) not in table.index:
            raise KeyError(f"{spec.label}: ({spec.model}, {spec.acq_label}) not in {spec.table_csv}")
        found = table.loc[(spec.model, spec.acq_label)]
        rec: dict[str, float | str] = {"label": spec.label}
        for col, _, _ in METRICS:
            rec[col] = float(found[col])
            rec[f"{col}_ci"] = float(found[f"{col}_ci"])
        records.append(rec)
    return pd.DataFrame.from_records(records).set_index("label")


def draw_table(ax, title: str, data: pd.DataFrame) -> None:
    """Draw one table on ``ax`` with the best mean per metric in bold.

    Args:
        ax: Matplotlib axes.
        title: Table title.
        data: Output of :func:`load_table`.
    """
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=S.FONT_SIZES["title"], fontweight=BOLD_WEIGHT)
    best = {
        col: (data[col].idxmax() if higher else data[col].idxmin()) for col, _, higher in METRICS
    }
    cell_text = [
        [f"{r[col]:.3f} ± {r[f'{col}_ci']:.3f}" for col, _, _ in METRICS] for _, r in data.iterrows()
    ]
    tab = ax.table(
        cellText=cell_text,
        rowLabels=list(data.index),
        colLabels=[header for _, header, _ in METRICS],
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
    parser = argparse.ArgumentParser(description="Export the NHP and 5d_rat summary tables as an image.")
    parser.add_argument("--out-dir", default="output/summary_tables")
    args = parser.parse_args()

    loaded = {title: load_table(rows) for title, rows in TABLES.items()}
    n_rows = sum(len(d) + 2 for d in loaded.values())
    fig, axes = S.figure("double", nrows=len(loaded), ncols=1, height=0.30 * n_rows + 0.6)
    for ax, (title, data) in zip(axes, loaded.items()):
        draw_table(ax, title, data)
    fig.text(
        0.01, 0.005,
        "Final value at the end of the run; mean ± 95% CI over 18 channels (10 reps averaged within a channel). "
        "Bold = best mean per metric (no significance test). TS = ts_marginal; EI = expected improvement.",
        fontsize=S.FONT_SIZES["annotation"], va="bottom", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for path in S.save_figure(fig, args.out_dir, "summary_tables", formats=FORMATS):
        print(path)


if __name__ == "__main__":
    main()
