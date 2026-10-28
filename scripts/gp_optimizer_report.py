"""Summaries of ``gp_optimizer_sweep.py`` (exploratory; user 2026-10-27).

Answers, from the sweep's CSVs: (1) which arms satisfy the pre-declared convergence criterion, per arm and per context size;
(2) whether an Adam arm reproduces the L-BFGS fit *from the same initialisation* (optimiser convergence, separate from the
choice of basin); (3) how often the production recipe's selected fit changes with ``restart_seed`` and how far the production
fit sits from the best-known one; (4) the refit nonlinearity per arm and context size, the optimiser-injected part of drift.

    python scripts/gp_optimizer_report.py --tag main
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

#: Tolerance (nats) under which an Adam arm counts as reproducing L-BFGS from the same initialisation.
SAME_INIT_NLL_TOL: float = 0.01
#: Differences between two production fits (seeds) counted as "the selected fit changed", in nats / predictive SDs.
SEED_NLL_TOL: float = 0.01
SEED_MEAN_TOL: float = 0.01


def load(out: str, tag: str) -> pd.DataFrame:
    """Concatenate the shard CSVs of one tag.

    Args:
        out: Sweep output directory.
        tag: Sweep tag.

    Returns:
        All rows.

    Raises:
        FileNotFoundError: If no shard exists.
    """
    files = sorted(glob.glob(os.path.join(out, f"sweep_{tag}_shard*.csv")))
    if not files:
        raise FileNotFoundError(f"no sweep_{tag}_shard*.csv under {out}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main() -> None:
    """Print the four summaries and write ``report_<tag>_*.csv`` next to the sweep."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default="output/diagnostics/gp_optimizer")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    d = load(args.out, args.tag)
    fit = d[d["dataset"] != "nonlinearity"].copy()
    ctx_keys = ["subject", "emg", "context_t", "draw", "dataset"]
    n_ctx = fit[ctx_keys].drop_duplicates().shape[0]
    print(f"{fit[['subject', 'emg', 'context_t', 'draw']].drop_duplicates().shape[0]} contexts, {n_ctx} datasets, "
          f"{fit['arm'].nunique()} arms")

    arms = fit.groupby("arm").agg(converged=("converged", "mean"), gap_med=("nll_gap_nats", "median"),
                                  gap_p90=("nll_gap_nats", lambda s: s.quantile(0.9)),
                                  dmu_med=("dmu_median", "median"), degenerate=("degenerate", "mean"),
                                  seconds=("seconds", "median"), steps=("steps", "median"))
    print("\n(1) converged fraction per arm (gap <= tol AND median |dmu|/sd <= tol vs the best-known fit)")
    print(arms.round(4).sort_values("converged", ascending=False).to_string())
    by_t = fit.pivot_table(index="arm", columns="context_t", values="converged", aggfunc="mean")
    print("\n    converged fraction by context size")
    print(by_t.round(3).to_string())
    arms.to_csv(os.path.join(args.out, f"report_{args.tag}_arms.csv"))

    ref = fit[fit["arm"] == "lbfgs1_init0"][ctx_keys + ["mll_final", "n_train", "noise"]].rename(
        columns={"mll_final": "mll_ref", "noise": "noise_ref"})
    adam = fit[fit["arm"].str.startswith("adam_")].merge(ref, on=ctx_keys + ["n_train"])
    adam["same_init"] = (((adam["mll_final"] - adam["mll_ref"]).abs() * adam["n_train"] <= SAME_INIT_NLL_TOL)
                         & ((adam["noise"] / adam["noise_ref"] - 1.0).abs() <= 0.05)).astype(float)
    if len(adam):
        same = adam.groupby("arm")["same_init"].mean()
        print("\n(2) fraction of datasets where the Adam arm reproduces L-BFGS from the SAME init (NLML within "
              f"{SAME_INIT_NLL_TOL} nats and noise within 5 %)")
        print(same.round(3).to_string())
        same.to_csv(os.path.join(args.out, f"report_{args.tag}_same_init.csv"))

    prod = fit[fit["arm"].str.startswith("lbfgs") & fit["arm"].str.contains("_s")]
    p0 = prod[prod["arm"].str.endswith("_s0")]
    print("\n(3) production recipe (restart_seed 0) against the best-known fit")
    print(f"    NLML gap > {SEED_NLL_TOL} nats on {float((p0['nll_gap_nats'] > SEED_NLL_TOL).mean()):.3f} of datasets; "
          f"median |dmu|/sd {p0['dmu_median'].median():.4f}, p90 {p0['dmu_median'].quantile(0.9):.4f}; "
          f"best-known fit degenerate on {float(fit[fit['arm'] == 'best']['degenerate'].mean()):.3f}")
    piv = prod.pivot_table(index=ctx_keys, columns="arm", values="mll_final")
    seeds = [c for c in piv.columns]
    spread = (piv.max(axis=1) - piv.min(axis=1)) * fit.groupby(ctx_keys)["n_train"].first().reindex(piv.index)
    print(f"    restart seeds disagree by > {SEED_NLL_TOL} nats on {float((spread > SEED_NLL_TOL).mean()):.3f} of datasets "
          f"({', '.join(seeds)})")

    nl = d[d["dataset"] == "nonlinearity"]
    print("\n(4) refit nonlinearity, median over channels of median_x |mu(+2)+mu(-2)-2mu(0)|/sd (0 = affine refit)")
    tab = nl.groupby(["arm", "context_t"])["nonlin_median"].median().unstack()
    print(tab.round(3).to_string())
    print("\n    90th percentile over channels of nonlin_median")
    print(nl.groupby(["arm", "context_t"])["nonlin_median"].quantile(0.9).unstack().round(3).to_string())
    tab.to_csv(os.path.join(args.out, f"report_{args.tag}_nonlinearity.csv"))


if __name__ == "__main__":
    main()
