"""Rebuild every figure and table under ``output/`` from the results already on disk (no compute).

Each run directory records its resolved config (``config.yaml``: ``experiment``, ``family``, dataset, knob,
``gt_mode``). This script matches it to the experiment YAML with the same family and dataset and calls the
runner's ``--replot`` on that exact directory (``--run-dir``), so runs with overridden tags
replot in place. Mechanism analyses (``output/mechanism/<analysis>/<dataset>``) are replotted
through their own runner. Use it after any change to ``visualization/``.

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


def _config_index() -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """Map each run to the experiment YAML that produces it, two ways.

    Returns:
        ``(by_family, by_run_name)``: ``(family, dataset) -> yaml`` and ``(dataset, "<family>-<tag>") -> yaml``. The second
        is the fallback for a run whose recorded family was later renamed while its directory stayed the same: the
        2026-10-05 rename (``8ef8955``) turned family ``stress-k2-channel`` + tag ``draws-nhp`` into family
        ``stress-k2-channel-draws`` + tag ``nhp``, which writes the SAME ``stress-k2-channel-draws-nhp`` directory.
    """
    by_family: dict[tuple[str, str], str] = {}
    by_run_name: dict[tuple[str, str], str] = {}
    for path in sorted(glob.glob(os.path.join(EXPERIMENT_DIR, "*.yaml"))):
        raw = _yaml(path)
        family, defaults = raw.get("family"), raw.get("defaults") or {}
        dataset = defaults.get("dataset") if isinstance(defaults.get("dataset"), str) else None
        if family and dataset:
            by_family.setdefault((family, dataset), path)
            if raw.get("tag"):
                by_run_name.setdefault((dataset, f"{family}-{raw['tag']}"), path)
    return by_family, by_run_name


def commands(output: str) -> list[list[str]]:
    """Return one replot command per run directory.

    Args:
        output: The output root.

    Returns:
        Argument vectors for ``python -m pfns4neurostim``.

    Run directories that no current experiment YAML can produce (a retired family such as ``stress-k2-channel-demo1``)
    are skipped and listed on stderr instead of aborting every other replot.
    """
    index, by_run_name = _config_index()
    cmds: list[list[str]] = []
    runs = glob.glob(os.path.join(output, "benchmark", "*", "*", "tidy.csv"))
    runs += glob.glob(os.path.join(output, "stress", "*", "*", "*", "tidy.csv"))
    missing = []
    for tidy in sorted(runs):
        run = os.path.dirname(tidy)
        cfg = _yaml(os.path.join(run, "config.yaml"))
        dataset = (cfg.get("dataset") or {}).get("name")
        yaml_path = index.get((cfg.get("family"), dataset)) or by_run_name.get((dataset, os.path.basename(run)))
        if yaml_path is None:
            missing.append(f"{run} (family={cfg.get('family')}, dataset={dataset})")
            continue
        cmds.append([cfg["experiment"], "--config", yaml_path, "--replot", "--run-dir", run])
    for path in sorted(glob.glob(os.path.join(EXPERIMENT_DIR, "mechanism_*.yaml"))):
        raw = _yaml(path)
        analysis = path.split("mechanism_")[1].rsplit("_", 1)[0]
        dataset = (raw.get("defaults") or {}).get("dataset")
        if glob.glob(os.path.join(output, "mechanism", analysis, str(dataset), "*", "*.csv")):
            cmds.append(["mechanism", "--config", path, "--replot"])
    if missing:
        print("[replot_all] skipped, no current experiment YAML produces:\n  " + "\n  ".join(missing), file=sys.stderr)
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
