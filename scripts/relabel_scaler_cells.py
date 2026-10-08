"""One-off fix: stamp ``online_y_scaler`` into cached rows that were written before the row field existed.

The cell identity carries ``online_y_scaler`` whenever it is active (2026-09-27), so a cell keyed ``minmax`` was
computed under ``minmax``. Some cells were cached before the tidy row got its own ``online_y_scaler`` field; their
rows read back the ``TidyRow`` default ``'none'`` and so mislabel every run that serves them (2026-10-08: 270 local
NHP cells). This script copies the identity's value into the row. The cache key is unchanged (identity untouched);
each rewritten JSON is first copied to ``<backup>/<dataset>/<experiment>/``.

    python scripts/relabel_scaler_cells.py             # dry run
    python scripts/relabel_scaler_cells.py --apply
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import shutil


def relabel(store: str, backup: str, *, apply: bool) -> collections.Counter:
    """Copy the identity's ``online_y_scaler`` into every cached row that lacks or contradicts it.

    Args:
        store: Cell-cache root (``output/cells``).
        backup: Directory the original JSON files are copied to before rewriting.
        apply: Rewrite the cells; otherwise only count them.

    Returns:
        Count of affected cells per ``(dataset, experiment, scaler)``.
    """
    counts: collections.Counter = collections.Counter()
    for path in glob.glob(os.path.join(store, "*", "*", "*.json")):
        with open(path, encoding="utf-8") as fh:
            cell = json.load(fh)
        scaler = cell.get("identity", {}).get("online_y_scaler")
        row = cell.get("payload", {}).get("row")
        if scaler is None or row is None or row.get("online_y_scaler") == scaler:
            continue
        dataset, experiment = os.path.normpath(path).split(os.sep)[-3:-1]
        counts[(dataset, experiment, scaler)] += 1
        if not apply:
            continue
        dest = os.path.join(backup, dataset, experiment)
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(path, dest)
        cell["payload"]["row"] = {**row, "online_y_scaler": scaler}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cell, fh)
        os.replace(tmp, path)
    return counts


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", default=os.path.join("output", "cells"))
    parser.add_argument("--backup", default=os.path.join(
        "output", "archive", f"{dt.datetime.now():%Y%m%d-%H%M%S}-scaler-relabel"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    counts = relabel(args.store, args.backup, apply=args.apply)
    for (dataset, experiment, scaler), n in sorted(counts.items()):
        print(f"{n:>7}  {dataset}/{experiment}  -> {scaler}")
    verb = f"relabelled (originals in {args.backup})" if args.apply else "would relabel (dry run)"
    print(f"[relabel] {sum(counts.values())} cell(s) {verb}")


if __name__ == "__main__":
    main()
