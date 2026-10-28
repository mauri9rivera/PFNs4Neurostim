"""Refit nonlinearity of GP-MLL restart recipes, with the restart selected by the marginal likelihood (exploratory; user 2026-10-27).

``gp_convergence_traces.py`` showed that ``GPSurrogate`` ranks its restarts by an evaluation-mode loss, not the marginal likelihood.
This script asks what the fix would buy for the optimiser-injected part of the OMEGA drift: for the base context D and the three refits
D + (x', y'), y' = mu(x') + z sd(x'), z in {-2, 0, +2}, it measures the *refit nonlinearity*
``median_x |mu(+2) + mu(-2) - 2 mu(0)| / sd_ref(x)`` (0 for any model whose refit mean is affine in y', e.g. GP-fixed) and the
disagreement between restart seeds, for

    k1_default      one L-BFGS start from gpytorch's initialisation (deterministic: no restart seed)
    k5_nll_s0..s2   5 restarts, selected by the train-mode NLML (the proposed fix), seeds 0 / 1 / 2
    k20_nll         20 restarts, selected by the train-mode NLML (the reference)
    prod_s0         the production recipe: 5 restarts selected by the evaluation-mode loss, seed 0

    python scripts/gp_refit_nonlinearity.py --shard 0/4
    python scripts/gp_refit_nonlinearity.py --report
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import _channels, _validate_bayes_update, load_mechanism_config
from pfns4neurostim.models.gp.surrogates import GPSurrogate
from pfns4neurostim.seeding import rng_for, set_seed

#: Standardised surprises of the refit datasets.
REFIT_Z: tuple[float, ...] = (-2.0, 0.0, 2.0)
#: ``restart_seed`` of the reference fit (disjoint from the production seeds).
REFERENCE_SEED: int = 100


class MarginalLikelihoodGP(GPSurrogate):
    """``GPSurrogate`` whose restarts are ranked by the train-mode marginal likelihood (prototype of the fix)."""

    def _fit_candidate(self, train_x: torch.Tensor, train_y: torch.Tensor, init: dict[str, float] | None,
                       floor: float) -> tuple[float, Any, Any, dict[str, float]]:
        """Fit one start; return the train-mode NLML per datum in place of the evaluation-mode loss."""
        _, model, likelihood, diag = super()._fit_candidate(train_x, train_y, init, floor)
        model.train()
        likelihood.train()
        with torch.no_grad():
            nll = -float(ExactMarginalLogLikelihood(likelihood, model)(model(train_x), train_y))
        model.eval()
        likelihood.eval()
        diag["gp_mll_final"] = nll
        return nll, model, likelihood, diag


def arms(base: dict[str, Any]) -> dict[str, tuple[type, dict[str, Any]]]:
    """The recipes compared, ``name -> (class, GPSurrogate kwargs)``."""
    out: dict[str, tuple[type, dict[str, Any]]] = {"k1_default": (MarginalLikelihoodGP, {**base, "n_restarts": 1})}
    for s in (0, 1, 2):
        out[f"k5_nll_s{s}"] = (MarginalLikelihoodGP, {**base, "n_restarts": 5, "restart_seed": s})
    out["k20_nll"] = (MarginalLikelihoodGP, {**base, "n_restarts": 20, "n_opt_steps": 100, "restart_seed": REFERENCE_SEED})
    out["prod_s0"] = (GPSurrogate, {**base, "restart_seed": 0})
    return out


def run(args: argparse.Namespace) -> None:
    """Fit every arm on every dataset of the shard and append ``refit_nl_<tag>_shard*.csv``."""
    torch.set_num_threads(1)
    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    shard = tuple(int(v) for v in args.shard.split("/"))
    out_csv = os.path.join(args.out, f"refit_nl_{args.tag}_shard{shard[0]}of{shard[1]}.csv")
    keys = ["subject", "emg", "context_t", "draw"]
    done = set()
    if os.path.exists(out_csv):
        done = {tuple(r) for r in pd.read_csv(out_csv)[keys].itertuples(index=False, name=None)}
    recipes = arms(dict(p["gp_params"]))
    knob = build_knob(p["stress"]["knob"], [args.level])
    tasks = [(ch, int(t), int(d)) for ch in _channels(cfg) for t in args.ts for d in args.draws]
    tasks = [task for i, task in enumerate(tasks) if i % shard[1] == shard[0]]
    t0 = time.time()
    for n, (ch, t, draw) in enumerate(tasks):
        if (ch.subject, ch.emg, t, draw) in done:
            continue
        st = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
        ctx = context_for(st, t, draw, base_seed=cfg.seed, level=float(args.level))
        rng = rng_for(ch.label, "bayes_update", args.level, t, draw, base_seed=cfg.seed)
        anchors, _, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]), int(p["anchors"]["n_observed"]), rng, ctx.sites)
        a, Q = int(anchors[0]), st.X_pool
        cls, kw = recipes["k20_nll"]
        ref = cls(**kw)
        ref.fit(ctx.X, ctx.y)
        m_a, s_a = (float(v[a]) for v in ref.predict_frozen(Q))
        _, sd_ref = ref.predict_frozen(Q)                                                 # [M]
        datasets = {z: (np.vstack([ctx.X, Q[a]]), np.append(ctx.y, m_a + z * s_a)) for z in REFIT_Z}
        means: dict[str, dict[float, np.ndarray]] = {}
        for name, (cls, kw) in recipes.items():
            means[name] = {}
            for z, (Xd, yd) in datasets.items():
                gp = cls(**kw)
                gp.fit(Xd, yd)
                means[name][z] = gp.predict_frozen(Q)[0]
        rows = []
        for name in recipes:
            nl = np.abs(means[name][2.0] + means[name][-2.0] - 2.0 * means[name][0.0]) / sd_ref   # [M]
            d_ref = {z: np.abs(means[name][z] - means["k20_nll"][z]) / sd_ref for z in REFIT_Z}
            rows.append({"subject": ch.subject, "emg": ch.emg, "context_t": t, "draw": draw, "arm": name,
                         "nonlin_median": float(np.median(nl)), "nonlin_max": float(nl.max()),
                         "dmu_vs_ref_median": float(np.median(np.concatenate([d_ref[z] for z in REFIT_Z])))})
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        print(f"[refit_nl] {n + 1}/{len(tasks)} ({time.time() - t0:.0f}s) {ch.label} t={t} d={draw}", flush=True)


def report(args: argparse.Namespace) -> None:
    """Print the nonlinearity per arm and context size, and the seed disagreement of the k = 5 recipes."""
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.out, f"refit_nl_{args.tag}_shard*.csv")))])
    pd.set_option("display.width", 220)
    print(f"{d[['subject', 'emg', 'context_t', 'draw']].drop_duplicates().shape[0]} contexts")
    print("median over contexts of nonlin_median")
    print(d.pivot_table(index="arm", columns="context_t", values="nonlin_median", aggfunc="median").round(3).to_string())
    print("90th percentile over contexts")
    print(d.pivot_table(index="arm", columns="context_t", values="nonlin_median", aggfunc=lambda s: s.quantile(0.9)).round(3).to_string())
    print("median |mu - mu_k20| / sd over refits")
    print(d.pivot_table(index="arm", columns="context_t", values="dmu_vs_ref_median", aggfunc="median").round(3).to_string())
    k5 = d[d["arm"].str.startswith("k5_nll")].pivot_table(index=["subject", "emg", "context_t", "draw"], columns="arm", values="nonlin_median")
    spread = (k5.max(axis=1) - k5.min(axis=1)).groupby(level="context_t").median()
    print("seed spread of nonlin_median across the three k = 5 seeds (median over contexts)")
    print(spread.round(3).to_string())


def main() -> None:
    """Parse the CLI."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--ts", type=int, nargs="+", default=[30, 40, 50, 80])
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default="output/diagnostics/gp_optimizer")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report(args) if args.report else run(args)


if __name__ == "__main__":
    main()
