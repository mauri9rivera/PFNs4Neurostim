"""One-off migration: relabel cached stress cells from a renamed knob to its successor.

The 2026-09-23 stress restructure renamed ``k2_snr`` to ``k2_channel``. The transform is
identical (residual amplification by alpha, i.e. -20*log10(alpha) dB relative to each
channel's floor), and so is every metric when no electrode fails, so the cells already
computed under the old name are valid results of the new knob. Only the name inside the
cell identity (and the tidy row) differs, which changes the cache key. This script writes
a relabelled copy of each matching cell under its new key; the originals are left in place.

K5 and K6-dropout cells are **not** relabelled: their knobs changed definition (slot-fraction
heavy-tail contamination; in-run electrode failure), so those results must be recomputed.

    python scripts/relabel_knob_cells.py                              # dry run, k2_snr -> k2_channel
    python scripts/relabel_knob_cells.py --apply
    python scripts/relabel_knob_cells.py --apply --store output/cells --old k2_snr --new k2_channel
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

from pfns4neurostim.evaluation.cache import cell_key, cell_path


def relabel(store: str, old: str, new: str, *, apply: bool) -> tuple[int, int]:
    """Copy every cell whose identity names knob ``old`` to a cell naming ``new``.

    Args:
        store: Cell-cache root (``output/cells``).
        old: Knob name to replace.
        new: Knob name to write.
        apply: Write the copies; otherwise only count them.

    Returns:
        ``(n_matching, n_written)``; a copy whose target already exists is skipped.
    """
    n_match = n_written = 0
    for path in glob.glob(os.path.join(store, "*", "*", "*.json")):
        with open(path, encoding="utf-8") as fh:
            cell = json.load(fh)
        identity = cell.get("identity", {})
        if identity.get("knob") != old:
            continue
        n_match += 1
        identity = {**identity, "knob": new}
        payload = cell["payload"]
        payload = {**payload, "row": {**payload["row"], "knob": new}}
        dataset, experiment = identity["dataset"], identity["experiment"]
        target = cell_path(store, dataset, experiment, cell_key(identity))
        if os.path.exists(target) or not apply:
            continue
        shutil.copyfile(path[:-5] + ".pkl", target[:-5] + ".pkl")   # trajectory first, as in save_cell
        with open(target, "w", encoding="utf-8") as fh:
            json.dump({"identity": identity, "payload": payload}, fh)
        n_written += 1
    return n_match, n_written


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", default=os.path.join("output", "cells"))
    parser.add_argument("--old", default="k2_snr")
    parser.add_argument("--new", default="k2_channel")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    n_match, n_written = relabel(args.store, args.old, args.new, apply=args.apply)
    verb = "wrote" if args.apply else "would write (dry run)"
    print(f"[relabel] {n_match} cell(s) named {args.old!r}; {verb} {n_written if args.apply else n_match} as {args.new!r}")


if __name__ == "__main__":
    main()
