"""Retire the results of a superseded data cohort: purge its cells, quarantine its runs.

When a dataset's raw files are replaced, every number computed from the old files is
stale. Two stores hold such numbers and they need opposite treatment:

* **The cell cache** (``output/cells/<dataset>/<experiment>/*.json|pkl``) is a pure
  computation cache. Stale cells can never be *returned* — since 2026-09-25 the cohort
  stamp is part of a cell's identity, and a cell whose identity differs is not a hit —
  but they still occupy disk and clutter every cache scan, so they are **deleted**.
* **Run directories** (``output/<kind>/<dataset>/<run tag>/``) hold deliverables. The
  figures stay in place as a proxy until new experiments overwrite them, so only the
  *data* artefacts that ``--replot`` and the aggregation would silently re-read
  (``tidy.csv``, ``trajectories.pkl``, ``shards/``, the summary tables, the resolved
  config) are **moved** into ``<run>/<quarantine>/``, next to a ``STALE_COHORT.md``
  saying why. Nothing is deleted there, and moving the files back undoes it.

Dry run by default; ``--apply`` performs it. Idempotent: a second run finds nothing.

    python scripts/retire_dataset_cohort.py --dataset 5d_rat --cohort noOutliers-2026-09-25
    python scripts/retire_dataset_cohort.py --dataset 5d_rat --cohort noOutliers-2026-09-25 --apply

Related: ``scripts/prune_results.py`` retires results the *code* can no longer request (a removed
knob, changed knob parameters, a superseded model version). This script retires results a changed
*dataset* invalidates. The two are complementary and could become one tool with a shared manifest;
until then, run this one after replacing a cohort and that one after a code change.

On the cluster, run it on the LOGIN node against the same paths (``output/`` is a
symlink into ``$SCRATCH``):

    module load anaconda/3 && conda activate pfns4neurostim
    python scripts/retire_dataset_cohort.py --dataset 5d_rat --cohort noOutliers-2026-09-25 --apply
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

#: Extensions of derived DATA in a run directory. Named as a rule rather than as a list
#: of filenames so a runner that starts writing a new table does not silently leave stale
#: numbers behind. Figures (``.svg``, ``.png``, ``.pdf``) and notes (``.md``) are not here:
#: they stay in place as the proxy.
STALE_SUFFIXES: frozenset[str] = frozenset({".csv", ".pkl", ".yaml", ".yml", ".json", ".npz"})

#: Sub-directories of a run that hold derived data.
STALE_DIRS: frozenset[str] = frozenset({"shards"})

#: Output trees that hold run directories, as ``<root>/<kind>/<dataset>/<tag>/``.
RUN_KINDS: tuple[str, ...] = ("benchmark", "stress", "gt_sensitivity", "mechanism", "scaling")

_MARKER = "STALE_COHORT.md"


def cohort_datasets(dataset: str) -> tuple[str, ...]:
    """Return every dataset name whose results derive from one cohort's raw files.

    Demo 1 twins are fitted to the recorded channels and are stored under their own
    dataset name, so retiring a cohort must retire them too.

    Args:
        dataset: The recorded dataset name.

    Returns:
        The recorded name and its Demo 1 twin name.
    """
    return (dataset, f"synthetic_{dataset}")


def purge_cells(store: str, dataset: str, cohort: str, *, apply: bool) -> tuple[int, int]:
    """Delete cached cells of ``dataset`` that were not computed on ``cohort``.

    Args:
        store: Cell-cache root (``output/cells``).
        dataset: Dataset whose cells to examine.
        cohort: The cohort to KEEP; every other cell of this dataset goes.
        apply: Delete; otherwise only count.

    Returns:
        ``(n_examined, n_removed)``.
    """
    n_examined = n_removed = 0
    for path in glob.glob(os.path.join(store, dataset, "*", "*.json")):
        n_examined += 1
        with open(path, encoding="utf-8") as fh:
            identity = json.load(fh).get("identity", {})
        if identity.get("data_cohort") == cohort:
            continue
        n_removed += 1
        if apply:
            os.remove(path)
            trajectory = path[:-5] + ".pkl"
            if os.path.exists(trajectory):
                os.remove(trajectory)
    return n_examined, n_removed


def quarantine_runs(
    output_root: str, dataset: str, cohort: str, quarantine: str, *, apply: bool
) -> list[tuple[str, int]]:
    """Move each run's stale data artefacts aside, leaving its figures in place.

    Args:
        output_root: Output root (``output``).
        dataset: Dataset whose runs to quarantine.
        cohort: Cohort that supersedes them, named in the marker file.
        quarantine: Sub-directory name to move the artefacts into.
        apply: Move the files and write the marker; otherwise only report.

    Returns:
        ``(run_dir, n_artefacts)`` per affected run, in path order.
    """
    affected: list[tuple[str, int]] = []
    for kind in RUN_KINDS:
        pattern = os.path.join(output_root, kind, "*", dataset, "*")
        flat = os.path.join(output_root, kind, dataset, "*")
        for run in sorted(set(glob.glob(pattern)) | set(glob.glob(flat))):
            if not os.path.isdir(run) or os.path.basename(run) == quarantine:
                continue
            present = sorted(
                name
                for name in os.listdir(run)
                if (name in STALE_DIRS and os.path.isdir(os.path.join(run, name)))
                or (
                    os.path.isfile(os.path.join(run, name))
                    and os.path.splitext(name)[1].lower() in STALE_SUFFIXES
                )
            )
            if not present:
                continue
            affected.append((run, len(present)))
            if not apply:
                continue
            target = os.path.join(run, quarantine)
            os.makedirs(target, exist_ok=True)
            for name in present:
                shutil.move(os.path.join(run, name), os.path.join(target, name))
            with open(os.path.join(run, _MARKER), "w", encoding="utf-8") as fh:
                fh.write(_marker_text(dataset, cohort, quarantine, present))
    return affected


def _marker_text(dataset: str, cohort: str, quarantine: str, moved: list[str]) -> str:
    """Compose the marker left in a quarantined run directory.

    Args:
        dataset: Dataset name.
        cohort: The cohort that supersedes this run.
        quarantine: Sub-directory the artefacts were moved into.
        moved: Artefact names that were moved.

    Returns:
        Markdown text.
    """
    return (
        f"# Stale cohort\n\n"
        f"This run was computed on a superseded cohort of `{dataset}`. The current cohort is "
        f"`{cohort}`; see `src/pfns4neurostim/data/loaders/` for what changed.\n\n"
        f"The figures in this directory are kept as a **proxy** until new `{dataset}` "
        f"experiments overwrite them. **Do not cite their numbers.**\n\n"
        f"The data artefacts this run was built from were moved into `{quarantine}/` so that "
        f"`--replot` and the aggregation cannot silently re-read them:\n\n"
        + "".join(f"- `{name}`\n" for name in moved)
        + f"\nMoving them back out of `{quarantine}/` undoes this.\n"
    )


def main() -> None:
    """Parse arguments and run the purge and the quarantine."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset whose cohort was replaced, e.g. 5d_rat.")
    parser.add_argument("--cohort", required=True, help="The CURRENT cohort stamp; cells of any other go.")
    parser.add_argument("--output-root", default="output", help="Output root (default: output).")
    parser.add_argument("--store", default=None, help="Cell-cache root (default: <output-root>/cells).")
    parser.add_argument("--quarantine", default="_stale_cohort", help="Sub-directory for stale run artefacts.")
    parser.add_argument("--apply", action="store_true", help="Perform it; otherwise report only.")
    args = parser.parse_args()

    store = args.store or os.path.join(args.output_root, "cells")
    datasets = cohort_datasets(args.dataset)
    examined = removed = 0
    for name in datasets:
        seen, gone = purge_cells(store, name, args.cohort, apply=args.apply)
        examined, removed = examined + seen, removed + gone
    verb = "removed" if args.apply else "would remove"
    print(
        f"cells ({', '.join(datasets)}): examined {examined}, {verb} {removed} "
        f"not on cohort {args.cohort!r}"
    )

    runs = [
        entry
        for name in datasets
        for entry in quarantine_runs(
            args.output_root, name, args.cohort, args.quarantine, apply=args.apply
        )
    ]
    verb = "quarantined" if args.apply else "would quarantine"
    print(f"runs: {verb} {len(runs)} run directories (figures left in place)")
    for run, count in runs:
        print(f"  {run}  ({count} artefacts)")
    if not args.apply:
        print("\nDry run. Re-run with --apply to perform it.")


if __name__ == "__main__":
    main()
