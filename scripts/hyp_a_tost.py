"""Hyp A equivalence: paired TOST of TabPFN-2.5 against each baseline, per dataset and pooled (task plan A6 / A8).

Reads the pooled tidy table (``scripts/aggregate_runs.py``), keeps the Hyp A runs (``family: hyp-a``, the canonical
``benchmark/<ds>/hyp-a-<ds>`` dirs) and each model under its headline acquisition, and tests the co-primary regret outcomes
of research_design.md (Hyp A): recommended-site regret, best-queried simple regret, and cumulative regret as the **mean
per-query regret** (``cumulative_regret / budget``, so all three share the range-normalized units of the pre-registered
margin, read from the runs' ``equivalence_margin``). The unit is a channel (repetitions averaged within it); a
subject-level table (channels averaged within a subject) is the sensitivity check for channel non-independence. Holm
correction over metrics x comparisons within each dataset. Equivalence is the Hyp A claim only against GP-MLL; against
GP-fixed and Random the CI of the difference is the readout (superiority when it lies below 0).

    python scripts/hyp_a_tost.py
    python scripts/hyp_a_tost.py --pooled output/aggregated/pooled.csv --out output/aggregated/hyp_a_tost.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from pfns4neurostim.evaluation.results import read_config
from pfns4neurostim.evaluation.stats import tost_table

DEFAULT_POOLED: str = "output/aggregated/pooled.csv"
DEFAULT_OUT: str = "output/aggregated/hyp_a_tost.csv"
OUTPUT_ROOT: str = "output"
DATASETS: tuple[str, ...] = ("nhp", "5d_rat", "spinal")
POOLED_LABEL: str = "pooled"
#: Each model under its headline acquisition (random search serves the `random` acquisition only).
ARMS: tuple[tuple[str, str], ...] = (("tabpfn_v2_5", "ts_marginal"), ("gp_mll", "ts_marginal"),
                                     ("gp_naive", "ts_marginal"), ("random", "random"))
COMPARISONS: list[tuple[str, str]] = [("tabpfn_v2_5", "gp_mll"), ("tabpfn_v2_5", "gp_naive"), ("tabpfn_v2_5", "random")]
METRICS: list[str] = ["recommended_regret", "best_queried_regret", "mean_query_regret"]
UNITS: dict[str, tuple[str, ...]] = {"channel": ("dataset", "subject", "emg"), "subject": ("dataset", "subject")}


def hyp_a_frame(pooled: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Select the Hyp A rows of the canonical runs and derive the mean per-query regret.

    Args:
        pooled: Pooled tidy table.

    Returns:
        ``(frame, margin)``; ``frame`` holds every dataset's rows plus a copy labelled :data:`POOLED_LABEL`.

    Raises:
        ValueError: If a dataset's run is missing or the runs disagree on the margin.
    """
    run_dirs = {ds: f"benchmark/{ds}/hyp-a-{ds}" for ds in DATASETS}
    missing = [d for d in run_dirs.values() if d not in set(pooled["run_dir"])]
    if missing:
        raise ValueError(f"Hyp A runs missing from the pooled table: {missing}")
    margins = {float(read_config(os.path.join(OUTPUT_ROOT, d))["equivalence_margin"]) for d in run_dirs.values()}
    if len(margins) != 1:
        raise ValueError(f"Hyp A runs disagree on equivalence_margin: {sorted(margins)}")
    keep = pooled["run_dir"].isin(run_dirs.values())
    arm = pd.Series(False, index=pooled.index)
    for model, acq in ARMS:
        arm |= (pooled["model"] == model) & (pooled["acq_label"] == acq)
    frame = pooled[keep & arm].copy()
    frame["mean_query_regret"] = frame["cumulative_regret"] / frame["budget"]
    both = pd.concat([frame, frame.assign(dataset=POOLED_LABEL, subject=frame["dataset"] + "/" + frame["subject"].astype(str))])
    return both, margins.pop()


def main() -> None:
    """Run the channel- and subject-level TOST tables and write them."""
    parser = argparse.ArgumentParser(description="Hyp A paired TOST per dataset and pooled.")
    parser.add_argument("--pooled", default=DEFAULT_POOLED)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    pooled = pd.read_csv(args.pooled, low_memory=False)
    frame, margin = hyp_a_frame(pooled)
    tables = [tost_table(frame, COMPARISONS, METRICS, margin, by=["dataset"], unit=unit).assign(level=level)
              for level, unit in UNITS.items()]
    out = pd.concat(tables, ignore_index=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    cols = ["level", "dataset", "model_b", "metric", "n", "mean_diff", "ci_lo", "ci_hi", "tost_p_holm", "equivalent_holm"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(f"margin = {margin}  (differences are TabPFN-2.5 minus model_b; 90% CI)")
        print(out[cols].round(4).to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
