"""EXPLORATORY: OMEGA Part A with the empirical-Bayes band at two GP-MLL restart budgets (task plan row Omega12 (b)).

Draws the pooled-anchor Part A readouts (``drift_abs``, ``total_var_ratio``, ``cov_asymmetry``, ``latent_frac``) vs
context size for TabPFN-2.5 and GP-MLL refit from a ``bayes_update`` run (5 restarts; the refinement grid
``nhp-l1-refine`` by default), beside GP-MLL refit from the converged-yardstick arm
(``configs/experiment/mechanism_bayes_update_nhp_gp20.yaml``, 20 restarts, tag ``nhp-gp20``). Each arm's restart
count is read from its run's ``config.yaml``, never assumed.

The summary is the Hyp C figures' own (``visualization/mechanism.py::_channel_band``): median over channels per t,
with the 95 % within-subject channel bootstrap of the median; GP-MLL arms are drawn as the median line alone
(``style.HYPC_LINE_ONLY_ENGINES``). Runs on whatever cells the two runs hold: a partial gp20 assembly is fine, a
missing one is skipped with a note. ``--matched`` keeps only the (subject, emg, t, draw) cells every arm has, so the
arms are compared on the same cells while the gp20 grid is still partial.

Not pre-registered and not read by any other script. Writes only under ``--out`` (default
``output/diagnostics/omega_gp20/``): the figure and the band table behind it.

Usage::

    python scripts/omega_gp20_band.py
    python scripts/omega_gp20_band.py --base-run output/mechanism/bayes_update/nhp/nhp --matched
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Sequence

import pandas as pd
import yaml

from pfns4neurostim.analysis.bayes_update import POOLED_ANCHORS
from pfns4neurostim.visualization import style
from pfns4neurostim.visualization.mechanism import (
    _COHERENCE_COLUMNS,
    _OMEGA_PANEL_ASPECT,
    OMEGA_COHERENCE_METRICS,
    _bayes_value,
    _channel_band,
    _context_axis,
    _median_band,
    arm_style,
)

#: Part A readouts drawn, one panel each (the three identities + latent_frac).
METRICS: tuple[str, ...] = ("drift_abs", "total_var_ratio", "cov_asymmetry", "latent_frac")
#: Engines read from the 5-restart run, and from the gp20 run.
BASE_ENGINES: tuple[str, ...] = ("tabpfn_v2_5", "gp_mll_refit")
GP20_ENGINES: tuple[str, ...] = ("gp_mll_refit",)
#: GP engines whose arm is labelled by its restart budget.
RESTART_ENGINES: tuple[str, ...] = ("gp_mll_refit",)
#: Keys of one cell (one channel, context size and layout); ``--matched`` joins the arms on them.
CELL_KEYS: tuple[str, ...] = ("subject", "emg", "context_t", "draw")
#: Legend label of a GP arm, and the figure title (the figure is exploratory and says so where it is read).
ARM_LABEL: str = "{label} ({n} restarts)"
TITLE: str = ("Exploratory: empirical-Bayes band vs GP-MLL restart budget; level {level:g}, every anchor; "
              "median over channels, {ci:.0%} channel bootstrap{matched}")
MATCHED_NOTE: str = "; matched cells"
FIGURE_NAME: str = "omega_gp20_band"

DEFAULT_BASE_RUN: str = os.path.join("output", "mechanism", "bayes_update", "nhp", "nhp-l1-refine")
DEFAULT_GP20_RUN: str = os.path.join("output", "mechanism", "bayes_update", "nhp", "nhp-gp20")
DEFAULT_OUT: str = os.path.join("output", "diagnostics", "omega_gp20")


def n_restarts(run_dir: str) -> int:
    """GP-MLL restart budget of a ``bayes_update`` run, from its resolved ``config.yaml``.

    Args:
        run_dir: Run directory.

    Returns:
        ``bayes_update.gp_params.n_restarts``.

    Raises:
        FileNotFoundError: If the run has no ``config.yaml``.
        KeyError: If the config holds no restart budget.
    """
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return int(cfg["bayes_update"]["gp_params"]["n_restarts"])


def load_arms(run_dir: str, engines: Sequence[str]) -> pd.DataFrame:
    """Pooled-anchor Part A rows of ``engines`` from one run, tagged with an ``arm`` column.

    Args:
        run_dir: ``bayes_update`` run directory holding ``coherence_cell.csv`` and ``config.yaml``.
        engines: Engines to keep.

    Returns:
        Rows with ``pool == POOLED_ANCHORS``; ``arm`` is ``<engine>`` or, for :data:`RESTART_ENGINES`,
        ``<engine>@<n_restarts>``; ``n_restarts`` holds the run's budget on those rows (null on the others).
    """
    cell = pd.read_csv(os.path.join(run_dir, "coherence_cell.csv"))
    cell = cell[(cell["pool"] == POOLED_ANCHORS) & cell["engine"].isin(engines)].copy()
    n = n_restarts(run_dir)
    is_gp = cell["engine"].isin(RESTART_ENGINES)
    cell["n_restarts"] = pd.Series(n, index=cell.index, dtype="Int64").where(is_gp)
    cell["arm"] = [f"{e}@{n}" if e in RESTART_ENGINES else e for e in cell["engine"]]
    return cell


def arm_styles(df: pd.DataFrame) -> dict[str, style.ModelStyle]:
    """Style of every arm: TabPFN as in the Hyp C figures, each GP-MLL refit budget a shade of GP-MLL.

    The smallest budget keeps the GP-MLL refit style of the Hyp C figures; larger ones are lighter shades
    (``style.derive_shade``) with another GP-family linestyle, so the arms differ in hue lightness and in dash.

    Args:
        df: Rows with ``arm``, ``engine`` and ``n_restarts``.

    Returns:
        ``arm -> ModelStyle``, labels naming the restart budget.
    """
    out: dict[str, style.ModelStyle] = {}
    for engine, sub in df.groupby("engine"):
        base = arm_style(str(engine))
        if engine not in RESTART_ENGINES:
            out[str(engine)] = base
            continue
        budgets = sorted(int(n) for n in sub["n_restarts"].dropna().unique())
        dashes = [ls for ls in style.SERIES_LINESTYLES if ls not in ("-", base.linestyle)]
        for i, n in enumerate(budgets):
            st = base if i == 0 else dataclasses.replace(
                base, color=style.derive_shade(base.color, i, len(budgets)), linestyle=dashes[(i - 1) % len(dashes)])
            out[f"{engine}@{n}"] = dataclasses.replace(st, label=ARM_LABEL.format(label=base.label, n=n))
    return out


def matched_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the cells (:data:`CELL_KEYS`) that every arm holds.

    Args:
        df: Rows of every arm.

    Returns:
        The subset of ``df`` on the common cells.
    """
    keys = list(CELL_KEYS)
    sets = [set(map(tuple, g[keys].itertuples(index=False))) for _, g in df.groupby("arm")]
    common = set.intersection(*sets) if sets else set()
    return df[[tuple(r) in common for r in df[keys].itertuples(index=False)]]


def band_table(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """The plotted summary per arm, metric and t, with its cell and channel counts.

    Args:
        df: Rows of every arm.
        metrics: Readouts.

    Returns:
        Columns ``arm, n_restarts, metric, context_t, median, ci_low, ci_high, n_cells, n_channels``.
    """
    rows: list[pd.DataFrame] = []
    for arm, sub in df.groupby("arm"):
        for metric in metrics:
            band = _channel_band(sub, metric, "context_t")
            if band.empty:
                continue
            valid = sub.dropna(subset=[metric]).assign(channel=lambda d: list(zip(d["subject"], d["emg"])))
            counts = valid.groupby("context_t").agg(n_cells=("draw", "size"), n_channels=("channel", "nunique"))
            rows.append(band.merge(counts.reset_index(), on="context_t", how="left").assign(
                arm=arm, n_restarts=sub["n_restarts"].iloc[0], metric=metric))
    cols = ["arm", "n_restarts", "metric", "context_t", "median", "ci_low", "ci_high", "n_cells", "n_channels"]
    return pd.concat(rows, ignore_index=True)[cols] if rows else pd.DataFrame(columns=cols)


def draw(df: pd.DataFrame, styles: dict[str, style.ModelStyle], level: float, matched: bool, out: str,
         formats: Sequence[str]) -> list[str]:
    """The four Part A panels vs t, every arm side by side.

    Args:
        df: Rows of every arm (one level).
        styles: ``arm -> ModelStyle``.
        level: Stress level shown (title).
        matched: Whether the rows were restricted to matched cells (title).
        out: Output directory.
        formats: File formats (``style.save_figure``; SVG by default, PNG / PDF opt-in).

    Returns:
        Written paths.
    """
    metrics = [m for m in METRICS if m in df]
    ts = sorted(df["context_t"].unique())
    fig, axes = style.figure("double", nrows=-(-len(metrics) // _COHERENCE_COLUMNS), ncols=_COHERENCE_COLUMNS,
                             squeeze=False, panel_aspect=_OMEGA_PANEL_ASPECT, layout=style.LAYOUT_ENGINE)
    for ax, metric in zip(axes.ravel(), metrics):
        for arm in styles:
            # engine column kept: _median_band draws HYPC_LINE_ONLY_ENGINES as the median line alone
            _median_band(ax, df[df["arm"] == arm], metric, styles[arm])
        _bayes_value(ax, OMEGA_COHERENCE_METRICS[metric])
        _context_axis(ax, ts)
        ax.set_ylabel(style.axis_label(metric))
    for ax in axes.ravel()[len(metrics):]:
        ax.set_axis_off()
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    style.panel_letters(axes.ravel()[:len(metrics)])
    fig.suptitle(TITLE.format(level=level, ci=style.HYPC_BAND_CI, matched=MATCHED_NOTE if matched else ""),
                 fontsize=style.FONT_SIZES["legend"])
    return style.save_figure(fig, out, FIGURE_NAME, formats=tuple(formats))


def main(argv: Sequence[str] | None = None) -> int:
    """Load both runs, summarise, draw and write the band table.

    Args:
        argv: Command-line arguments (``sys.argv[1:]`` when ``None``).

    Returns:
        Process exit code.

    Raises:
        FileNotFoundError: If the base run has no ``coherence_cell.csv``.
        ValueError: If no arm has rows at the requested level.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-run", default=DEFAULT_BASE_RUN, help="bayes_update run with TabPFN + GP-MLL refit.")
    ap.add_argument("--gp20-run", default=DEFAULT_GP20_RUN, help="bayes_update run of the gp20 arm (optional).")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output directory.")
    ap.add_argument("--level", type=float, default=None, help="Stress level (default: the lowest in the base run).")
    ap.add_argument("--matched", action="store_true", help="Keep only the cells every arm holds.")
    ap.add_argument("--formats", nargs="+", default=["svg"], help="Figure formats (svg default; png / pdf opt-in).")
    args = ap.parse_args(argv)

    base_csv = os.path.join(args.base_run, "coherence_cell.csv")
    if not os.path.exists(base_csv):
        raise FileNotFoundError(f"{base_csv} not found: assemble the base run first (--only-cached).")
    parts = [load_arms(args.base_run, BASE_ENGINES)]
    if os.path.exists(os.path.join(args.gp20_run, "coherence_cell.csv")):
        parts.append(load_arms(args.gp20_run, GP20_ENGINES))
    else:
        print(f"[omega_gp20_band] no coherence_cell.csv in {args.gp20_run}: drawing the base run only.")
    df = pd.concat(parts, ignore_index=True)
    level = float(df["level"].min()) if args.level is None else float(args.level)
    df = df[(df["level"] - level).abs() <= max(abs(level), 1.0) * 1e-9]
    if df.empty:
        raise ValueError(f"No pooled Part A rows at level {level:g}.")
    clash = df.groupby("arm")["n_restarts"].nunique()
    if (clash > 1).any():
        raise ValueError(f"Arms with mixed restart budgets: {list(clash[clash > 1].index)}.")
    if args.matched:
        df = matched_cells(df)
    styles = {a: s for a, s in arm_styles(df).items() if a in set(df["arm"])}
    os.makedirs(args.out, exist_ok=True)
    table = band_table(df, [m for m in METRICS if m in df])
    table_path = os.path.join(args.out, f"{FIGURE_NAME}.csv")
    table.to_csv(table_path, index=False)
    for path in [*draw(df, styles, level, args.matched, args.out, args.formats), table_path]:
        print(f"[omega_gp20_band] wrote {path}")
    counts = df.groupby(["arm", "context_t"]).size().unstack(fill_value=0)
    print(f"[omega_gp20_band] cells per arm and t:\n{counts.to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
