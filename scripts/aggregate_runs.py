"""Pool every assembled run's ``tidy.csv`` into one provenance-stamped table (task plan A8).

Walks the given roots for run directories holding ``tidy.csv`` + ``config.yaml`` (archives, ``_stale_cohort/``
and ``shards/`` are skipped), stamps each row with ``run_dir`` / ``family`` / ``tag`` / ``online_y_scaler``
and writes ``<out>``. ``scripts/export_summary_tables.py`` and cross-dataset analyses read that file.

    python scripts/aggregate_runs.py
    python scripts/aggregate_runs.py --roots output/benchmark --out output/aggregated/benchmark.csv
"""
from __future__ import annotations

import argparse
import os

from pfns4neurostim.evaluation.pooling import discover_tidy_runs, pool_runs

DEFAULT_OUTPUT_ROOT: str = "output"
DEFAULT_ROOTS: tuple[str, ...] = ("output/benchmark", "output/stress")
DEFAULT_OUT: str = "output/aggregated/pooled.csv"


def main() -> None:
    """Discover, pool and write the run tables."""
    parser = argparse.ArgumentParser(description="Pool the tidy.csv of every assembled run.")
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS), help="Directories to search.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root the run_dir column is relative to.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Pooled CSV path.")
    args = parser.parse_args()

    run_dirs = discover_tidy_runs(args.roots)
    pooled = pool_runs(run_dirs, args.output_root)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pooled.to_csv(args.out, index=False)
    counts = pooled.groupby("run_dir", sort=True).size()
    for run_dir, n in counts.items():
        print(f"{n:>8}  {run_dir}")
    print(f"{len(pooled)} rows from {len(run_dirs)} runs -> {args.out}")


if __name__ == "__main__":
    main()
