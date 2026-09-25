"""Rebuild every figure and table under ``output/`` from the results already on disk (no compute).

Each run directory records its resolved config (``config.yaml``: ``experiment``, ``family``, dataset, knob,
``gt_mode``). This script matches it to the experiment YAML with the same family and dataset and calls the
runner's ``--replot`` on that exact directory (``--run-dir``), so runs with overridden tags (``nhp-sh``)
replot in place. Mechanism analyses (``output/mechanism/<analysis>/<dataset>``) and the GT-sensitivity run
are replotted through their own runners. Use it after any change to ``visualization/``.

    python scripts/replot_all.py            # print the commands
    python scripts/replot_all.py --run      # run them
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import yaml

EXPERIMENT_DIR = os.path.join("configs", "experiment")


def _yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _config_index() -> dict[tuple[str, str], str]:
    """Map ``(family, dataset)`` to the experiment YAML that produces it."""
    index: dict[tuple[str, str], str] = {}
    for path in sorted(glob.glob(os.path.join(EXPERIMENT_DIR, "*.yaml"))):
        raw = _yaml(path)
        family, defaults = raw.get("family"), raw.get("defaults") or {}
        dataset = defaults.get("dataset") if isinstance(defaults.get("dataset"), str) else None
        if family and dataset:
            index.setdefault((family, dataset), path)
    return index


def commands(output: str) -> list[list[str]]:
    """Return one replot command per run directory.

    Args:
        output: The output root.

    Returns:
        Argument vectors for ``python -m pfns4neurostim``.

    Raises:
        SystemExit: If a run directory has no matching experiment YAML.
    """
    index = _config_index()
    cmds: list[list[str]] = []
    runs = glob.glob(os.path.join(output, "benchmark", "*", "*", "tidy.csv"))
    runs += glob.glob(os.path.join(output, "stress", "*", "*", "*", "tidy.csv"))
    missing = []
    for tidy in sorted(runs):
        run = os.path.dirname(tidy)
        cfg = _yaml(os.path.join(run, "config.yaml"))
        key = (cfg.get("family"), (cfg.get("dataset") or {}).get("name"))
        if key not in index:
            missing.append(f"{run} (family={key[0]}, dataset={key[1]})")
            continue
        cmds.append([cfg["experiment"], "--config", index[key], "--replot", "--run-dir", run])
    for path in sorted(glob.glob(os.path.join(EXPERIMENT_DIR, "mechanism_*.yaml"))):
        raw = _yaml(path)
        analysis = path.split("mechanism_")[1].rsplit("_", 1)[0]
        dataset = (raw.get("defaults") or {}).get("dataset")
        if glob.glob(os.path.join(output, "mechanism", analysis, str(dataset), "*", "*.csv")):
            cmds.append(["mechanism", "--config", path, "--replot"])
    for path in sorted(glob.glob(os.path.join(EXPERIMENT_DIR, "gt_sensitivity_*.yaml"))):
        dataset = (_yaml(path).get("defaults") or {}).get("dataset")
        for run in glob.glob(os.path.join(output, "gt_sensitivity", str(dataset), "*")):
            if os.path.exists(os.path.join(run, "gt_sensitivity.csv")):
                cmds.append(["gt_sensitivity", "--config", path, "--replot", "--run-dir", run])
    if missing:
        raise SystemExit("No experiment YAML for:\n  " + "\n  ".join(missing))
    return cmds


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", default="output")
    parser.add_argument("--run", action="store_true", help="Run the commands (default: print them).")
    args = parser.parse_args()
    failed = []
    for cmd in commands(args.output):
        line = [sys.executable, "-m", "pfns4neurostim", *cmd]
        print(" ".join(line[1:]), flush=True)
        if args.run and subprocess.run(line).returncode != 0:
            failed.append(" ".join(cmd))
    if failed:
        raise SystemExit("Replot failed for:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
