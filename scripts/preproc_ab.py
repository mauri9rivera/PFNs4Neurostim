"""Task 1 Step 2: measure which (X, y) scaling GP-MLL should see.

For every normalization arm x channel it records (a) median-able BO regret from the
package runner and (b) the GP hyperparameters fitted on one fixed random design, which
explain *why* an arm wins. Writes ``preproc_ab.csv`` and prints the per-arm summary.

    python scripts/preproc_ab.py --dataset nhp --subjects 0 --emgs 0 1 2 --n-reps 5 --device cpu
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from pfns4neurostim.data.channels import ChannelData, iter_channels
from pfns4neurostim.data.preprocessing import NORMALIZATIONS
from pfns4neurostim.evaluation.bo_runner import run_channel_bo
from pfns4neurostim.models.gp.surrogates import GPSurrogate
from pfns4neurostim.seeding import seed_for


def fitted_hyperparameters(
    channel: ChannelData, n_design: int, seed: int, device: str
) -> dict[str, float]:
    """Fit a GP on a fixed random design of noisy trials and read its hyperparameters.

    Args:
        channel: Preprocessed channel.
        n_design: Number of random sites observed.
        seed: Design seed (identical across arms, so arms see the same sites).
        device: Torch device string.

    Returns:
        Mean ARD lengthscale, outputscale and observation noise.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(channel.n_sites, size=min(n_design, channel.n_sites), replace=False)  # [n]
    first_valid = np.argmax(np.isfinite(channel.Y_trials[idx]), axis=1)                    # [n]
    y = channel.Y_trials[idx, first_valid]                                                 # [n]
    gp = GPSurrogate(device=device)
    gp.fit(channel.X_pool[idx], y)
    kern = gp._model.covar_module
    return {
        "lengthscale": float(kern.base_kernel.lengthscale.detach().mean()),
        "outputscale": float(kern.outputscale.detach()),
        "noise": float(gp._likelihood.noise.detach().mean()),
    }


def main() -> None:
    """Run every arm on the selected channels and summarize."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="nhp")
    ap.add_argument("--subjects", type=int, nargs="+", default=[0])
    ap.add_argument("--emgs", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--n-design", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="output/preproc_ab")
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    for arm in NORMALIZATIONS:
        for ch in iter_channels(args.dataset, args.subjects, args.emgs, normalization=arm):
            try:
                hp = fitted_hyperparameters(ch, args.n_design, args.seed, args.device)
            except Exception as exc:  # noqa: BLE001 - a numerically unstable arm is a result
                hp = {"lengthscale": np.nan, "outputscale": np.nan, "noise": np.nan}
                print(f"[preproc_ab] {arm} {ch.label} hyperparameter fit FAILED: {exc}", flush=True)
            for rep in range(args.n_reps):
                # Same seed for every arm: only the preprocessing differs.
                seed = seed_for(ch.label, "ei", "gp_mll", rep, base_seed=args.seed)
                try:
                    res = run_channel_bo(
                        "gp_mll", ch, acq_fn="ei", budget=args.budget, n_init=args.n_init,
                        seed=seed, device=args.device,
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append({"normalization": arm, "channel": ch.label, "rep": rep,
                                 "error": f"{type(exc).__name__}", **hp})
                    print(f"[preproc_ab] {arm} {ch.label} rep{rep} FAILED: {type(exc).__name__}", flush=True)
                    continue
                rows.append({
                    "normalization": arm, "channel": ch.label, "rep": rep,
                    "recommended_regret": res.row["recommended_regret"],
                    "best_queried_regret": res.row["best_queried_regret"],
                    "r2": res.row["r2"], "error": "", **hp,
                })
            print(f"[preproc_ab] {arm:<18} {ch.label} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "preproc_ab.csv"), index=False)
    cols = ["recommended_regret", "best_queried_regret", "r2", "lengthscale", "outputscale", "noise"]
    print("failed cells per arm:")
    print(df.assign(failed=df["error"] != "").groupby("normalization")["failed"].agg(["sum", "count"]).to_string())
    print(df.groupby("normalization")[cols].median().round(4).to_string())
    print(df.groupby(["normalization", "channel"])["recommended_regret"].median().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
