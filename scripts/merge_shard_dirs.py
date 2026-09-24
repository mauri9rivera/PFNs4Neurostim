"""One-off migration: fold legacy per-shard run directories into their experiment's single run directory.

Before 2026-09-23 every shard/lane wrote its own ``<family>-<tag>-shard<i>of<n>`` directory, later assembled into
``<family>-<tag>``. This script turns each legacy shard directory into a provenance record under
``<merged>/shards/`` (machine, GPU, models, row count, read from the shard's own ``config.yaml`` and ``tidy.csv``) and moves
the shard directory to ``output/archive/<date>-shard-dirs/`` (moved, never deleted). Cells are untouched.

    python scripts/merge_shard_dirs.py                 # dry run: prints what would happen
    python scripts/merge_shard_dirs.py --apply
    python scripts/merge_shard_dirs.py --apply --output-root output --archive-name 2026-09-23-shard-dirs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import pandas as pd
import yaml

from pfns4neurostim.evaluation.shards import SHARD_DIR

_SHARD_NAME = re.compile(r"-shard(?P<i>\d+)of(?P<n>\d+)$")


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def plan(output_root: str) -> list[tuple[str, str, str]]:
    """Find ``(shard_dir, merged_dir, shard_spec)`` triples under ``output_root/{benchmark,stress}``.

    Args:
        output_root: The ``output`` directory.

    Returns:
        One triple per legacy shard directory that has a sibling merged directory of the same family.
    """
    found: list[tuple[str, str, str]] = []
    for top in ("benchmark", "stress"):
        for parent, dirs, _ in os.walk(os.path.join(output_root, top)):
            shard_dirs = [d for d in dirs if _SHARD_NAME.search(d)]
            merged = {}
            for d in dirs:
                cfg = os.path.join(parent, d, "config.yaml")
                if not _SHARD_NAME.search(d) and os.path.exists(cfg):
                    merged.setdefault(str(_load_yaml(cfg).get("family")), os.path.join(parent, d))
            for d in shard_dirs:
                cfg = os.path.join(parent, d, "config.yaml")
                family = str(_load_yaml(cfg).get("family")) if os.path.exists(cfg) else ""
                target = merged.get(family) or next(
                    (path for fam, path in merged.items() if d.startswith(f"{fam}-")), None
                )  # a shard that died before writing config.yaml is matched by its family prefix
                match = _SHARD_NAME.search(d)
                if target and match:
                    found.append((os.path.join(parent, d), target, f"{match['i']}of{match['n']}"))
            dirs[:] = [d for d in dirs if not _SHARD_NAME.search(d)]
    return found


def convert(shard_dir: str, merged_dir: str, spec: str) -> dict:
    """Build the provenance record of one legacy shard directory.

    Args:
        shard_dir: Legacy shard run directory.
        merged_dir: The experiment's merged run directory.
        spec: ``'{i}of{n}'``.

    Returns:
        A record in the format of :func:`pfns4neurostim.evaluation.shards.write_shard_record`.
    """
    cfg_path = os.path.join(shard_dir, "config.yaml")
    cfg = _load_yaml(cfg_path) if os.path.exists(cfg_path) else {}
    tidy_path = os.path.join(shard_dir, "tidy.csv")
    df = pd.read_csv(tidy_path) if os.path.exists(tidy_path) else pd.DataFrame()
    host = dict(cfg.get("host") or {})
    host.setdefault("cpu", "unknown")
    return {
        "shard": spec,
        "models": sorted(df["model"].unique()) if "model" in df else list(cfg.get("models", [])),
        "device": cfg.get("device"),
        "model_devices": {m: p["device"] for m, p in (cfg.get("model_params") or {}).items() if p.get("device")},
        "host": host,
        "cells_computed": int(len(df)),
        "cells_cached": 0,
        "cells_failed": 0,
        "wall_time_s": None,
        "finished_utc": pd.Timestamp(os.path.getmtime(tidy_path if df.size else cfg_path), unit="s").strftime("%Y-%m-%dT%H:%M:%SZ")
        if os.path.exists(cfg_path) else "",
        "status": "ok",
        "migrated_from": os.path.basename(shard_dir),
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--archive-name", default="2026-09-23-shard-dirs")
    parser.add_argument("--apply", action="store_true", help="Write records and move the shard directories.")
    args = parser.parse_args()
    triples = plan(args.output_root)
    print(f"{len(triples)} legacy shard directories")
    for shard_dir, merged_dir, spec in triples:
        rel = os.path.relpath(shard_dir, args.output_root)
        print(f"  {rel}  ->  {os.path.relpath(merged_dir, args.output_root)}/{SHARD_DIR}/ ({spec})")
        if not args.apply:
            continue
        record = convert(shard_dir, merged_dir, spec)
        out_dir = os.path.join(merged_dir, SHARD_DIR)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"migrated-{os.path.basename(shard_dir)}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
        dest = os.path.join(args.output_root, "archive", args.archive_name, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(shard_dir, dest)
    if not args.apply:
        print("dry run: nothing changed (use --apply)")


if __name__ == "__main__":
    main()
