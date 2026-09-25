"""Find results the current code can never read again, verify it, and archive (or delete) them.

A cached cell is served only when a run requests exactly its identity, and a run directory is only a view
assembled from cells. So "dead" has a precise meaning here: nothing the current code and configs can
request will ever read it. The rules, each checked against the code rather than assumed:

Cells (``output/cells/{dataset}/{experiment}/{key}.json`` + ``.pkl``):
    C1  knob not in ``KNOB_REGISTRY`` (``k6_dropout``). A renamed knob (``k2_snr`` -> ``k2_channel``) is dead
        only when its relabelled twin exists in the cache; a cell without a twin is KEPT and reported.
    C2  knob parameters the knob's constructor no longer accepts (old-design K5 ``source=invalid|heavy_tail``).
    C3  ``model_version`` differs from the model's current registry version (e.g. TabFM before 2026-09-25).

Run directories (``output/benchmark/*/*``, ``output/stress/*/*/*``):
    R1  ``stress/<knob>/`` for a knob that no longer exists (``k2_snr``, ``k6_dropout``), provided every
        C1 cell of that knob was dead (a ``k2_snr`` cell without a twin keeps its folder too).
    R2  a run whose knob parameters are unrequestable (C2 applied to its ``config.yaml``).
    R3  a pre-2026-09-23 per-shard run (``*-{cpu,gpu}-shard<i>of<n>``) or a retired run family
        (``hyp0-pfns4bo``, merged into ``hyp0-pfn-bench``), provided every one of its tidy rows is present in
        a kept run of the same dataset. Otherwise it is KEPT and reported.

Nothing is touched without ``--apply``. ``--apply`` MOVES everything into ``<output>/archive/<stamp>-pruned/``
under its original relative path (reversible); ``--delete`` removes it instead. Every run writes a manifest.

    python scripts/prune_results.py                      # dry run: what would go, and why
    python scripts/prune_results.py --apply              # move into output/archive/<stamp>-pruned/
    python scripts/prune_results.py --apply --delete     # remove permanently
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import inspect
import json
import os
import re
import shutil
from typing import Any

import pandas as pd
import yaml

from pfns4neurostim.data.stress import KNOB_REGISTRY
from pfns4neurostim.evaluation.cache import cell_key, cell_path
from pfns4neurostim.models.registry import MODEL_REGISTRY

#: Knobs renamed without a change of definition: old name -> new name (see scripts/relabel_knob_cells.py).
RENAMED_KNOBS: dict[str, str] = {"k2_snr": "k2_channel"}
#: Run families retired by a merge: family -> the family whose runs now hold its rows.
RETIRED_FAMILIES: dict[str, str] = {"hyp0-pfns4bo": "hyp0-pfn-bench"}
SHARD_DIR = re.compile(r"-(cpu|gpu)-shard\d+of\d+$")
#: Tidy columns that identify one repetition across runs (present in both benchmark and stress tidies).
ROW_KEY: tuple[str, ...] = ("dataset", "subject", "emg", "model", "acq_label", "knob", "level", "rep")


def _accepted_params(knob: str) -> set[str] | None:
    """Keyword parameters a knob's constructor accepts beyond ``levels`` (None for an unknown knob)."""
    cls = KNOB_REGISTRY.get(knob)
    if cls is None:
        return None
    return {p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "levels")}


def _params_requestable(knob: str | None, params: dict[str, Any] | None) -> bool:
    """Whether a knob with these parameters can still be built from a config."""
    if knob in (None, "nominal"):
        return True
    accepted = _accepted_params(knob)
    return accepted is not None and set(params or {}) <= accepted


def classify_cells(cells_root: str) -> tuple[list[tuple[str, str]], collections.Counter]:
    """Return the dead cells ``[(json_path, rule)]`` and a count of the kept-but-flagged ones.

    Args:
        cells_root: The cell-cache root (``output/cells``).

    Returns:
        ``(dead, kept_flags)``; ``kept_flags`` counts cells a rule would target but that are kept for safety.
    """
    dead: list[tuple[str, str]] = []
    kept: collections.Counter = collections.Counter()
    for path in glob.glob(os.path.join(cells_root, "*", "*", "*.json")):
        with open(path, encoding="utf-8") as fh:
            identity = json.load(fh)["identity"]
        knob, model = identity.get("knob"), identity.get("model")
        if knob in RENAMED_KNOBS:
            twin = {**identity, "knob": RENAMED_KNOBS[knob]}
            target = cell_path(cells_root, twin["dataset"], twin["experiment"], cell_key(twin))
            if os.path.exists(target):
                dead.append((path, f"C1 renamed knob {knob} (twin present)"))
            else:
                kept[f"{knob}: no {RENAMED_KNOBS[knob]} twin"] += 1
            continue
        if knob is not None and knob not in KNOB_REGISTRY:
            dead.append((path, f"C1 unregistered knob {knob}"))
        elif not _params_requestable(knob, identity.get("knob_params")):
            dead.append((path, f"C2 {knob} params {sorted(identity.get('knob_params') or {})} not accepted"))
        elif model in MODEL_REGISTRY and identity.get("model_version") != MODEL_REGISTRY[model].version:
            dead.append((path, f"C3 stale {model} version"))
    return dead, kept


def _run_dirs(output: str) -> list[str]:
    """Every run directory (one holding a ``config.yaml``) under benchmark/ and stress/."""
    found = glob.glob(os.path.join(output, "benchmark", "*", "*", "config.yaml"))
    found += glob.glob(os.path.join(output, "stress", "*", "*", "*", "config.yaml"))
    return sorted(os.path.dirname(p) for p in found)


def _config(run: str) -> dict[str, Any]:
    with open(os.path.join(run, "config.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _row_keys(run: str) -> pd.DataFrame | None:
    path = os.path.join(run, "tidy.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = [c for c in ROW_KEY if c in df.columns]
    return df[cols].astype(str).drop_duplicates()


def classify_runs(output: str, dead_knob_ok: dict[str, bool]) -> tuple[list[tuple[str, str]], list[str]]:
    """Return the dead run directories ``[(dir, rule)]`` and the notes on runs kept for safety.

    Args:
        output: The output root.
        dead_knob_ok: For each removed/renamed knob, whether all of its cells were dead.

    Returns:
        ``(dead, kept_notes)``.
    """
    dead: list[tuple[str, str]] = []
    notes: list[str] = []
    runs = _run_dirs(output)
    cfg = {r: _config(r) for r in runs}

    def is_retired(r: str) -> bool:
        return bool(SHARD_DIR.search(os.path.basename(r))) or cfg[r].get("family") in RETIRED_FAMILIES

    for knob_dir in glob.glob(os.path.join(output, "stress", "*")):
        knob = os.path.basename(knob_dir)
        if knob not in KNOB_REGISTRY and os.path.isdir(knob_dir):
            if dead_knob_ok.get(knob, True):
                dead.append((knob_dir, f"R1 knob {knob} no longer exists"))
            else:
                notes.append(f"kept {knob_dir}: some {knob} cells have no twin")
    dead_prefixes = tuple(d + os.sep for d, _ in dead)

    kept_runs = [r for r in runs if not r.startswith(dead_prefixes) and not is_retired(r)
                 and _params_requestable(cfg[r].get("knob", {}).get("type"), cfg[r].get("knob", {}).get("params"))]
    for r in runs:
        if r.startswith(dead_prefixes):
            continue
        knob = cfg[r].get("knob", {})
        if not _params_requestable(knob.get("type"), knob.get("params")):
            dead.append((r, f"R2 {knob.get('type')} params {sorted(knob.get('params') or {})} not accepted"))
            continue
        if not is_retired(r):
            continue
        rows = _row_keys(r)
        dataset = os.path.basename(os.path.dirname(r))
        pool = [k for k in (_row_keys(x) for x in kept_runs if os.path.basename(os.path.dirname(x)) == dataset)
                if k is not None]
        if rows is None or rows.empty:
            dead.append((r, "R3 retired run without tidy rows"))
            continue
        union = pd.concat([p[[c for c in rows.columns if c in p.columns]] for p in pool
                           if set(rows.columns) <= set(p.columns)], ignore_index=True).drop_duplicates() if pool else None
        missing = len(rows) if union is None else len(rows.merge(union, how="left", indicator=True).query("_merge == 'left_only'"))
        if missing == 0:
            dead.append((r, f"R3 retired run; all {len(rows)} rows present in kept runs"))
        else:
            notes.append(f"kept {r}: {missing}/{len(rows)} rows not found in any kept run")
    return dead, notes


def _move(path: str, output: str, archive: str | None) -> None:
    """Archive (move, preserving the relative path) or delete one file or directory."""
    if archive is None:
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        return
    dest = os.path.join(archive, os.path.relpath(path, output))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(path, dest)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="output")
    parser.add_argument("--apply", action="store_true", help="Act (default: dry run).")
    parser.add_argument("--delete", action="store_true", help="With --apply: delete instead of archiving.")
    args = parser.parse_args()

    cells_root = os.path.join(args.output, "cells")
    dead_cells, kept_cells = classify_cells(cells_root)
    removed_knobs = {k for k in list(RENAMED_KNOBS) + [p.split()[-1] for _, p in dead_cells if p.startswith("C1 unreg")]}
    dead_knob_ok = {k: not any(f.startswith(f"{k}:") for f in kept_cells) for k in removed_knobs}
    dead_runs, notes = classify_runs(args.output, dead_knob_ok)

    by_rule = collections.Counter(rule.split(" (")[0] for _, rule in dead_cells)
    print(f"[prune] cells: {len(dead_cells)} dead")
    for rule, n in sorted(by_rule.items()):
        print(f"    {n:6d}  {rule}")
    for flag, n in sorted(kept_cells.items()):
        print(f"    KEPT {n}: {flag}")
    print(f"[prune] run directories: {len(dead_runs)} dead")
    for path, rule in dead_runs:
        print(f"    {os.path.relpath(path, args.output)}  <- {rule}")
    for note in notes:
        print(f"    {note}")

    if not args.apply:
        print("[prune] dry run: nothing touched (add --apply to archive, --apply --delete to remove)")
        return
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = None if args.delete else os.path.join(args.output, "archive", f"{stamp}-pruned")
    manifest_dir = archive or os.path.join(args.output, "archive")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest = os.path.join(manifest_dir, f"prune-manifest-{stamp}.tsv")
    with open(manifest, "w", encoding="utf-8") as fh:
        for path, rule in dead_runs:
            if os.path.exists(path):
                _move(path, args.output, archive)
                fh.write(f"run\t{os.path.relpath(path, args.output)}\t{rule}\n")
        for path, rule in dead_cells:
            for part in (path, path[:-5] + ".pkl"):
                if os.path.exists(part):
                    _move(part, args.output, archive)
            fh.write(f"cell\t{os.path.relpath(path, args.output)}\t{rule}\n")
    verb = "deleted" if args.delete else f"moved to {archive}"
    print(f"[prune] {len(dead_runs)} run dirs and {len(dead_cells)} cells {verb}; manifest {manifest}")


if __name__ == "__main__":
    main()
