"""Optimiser sweep of the GP-MLL yardstick on OMEGA contexts (exploratory; user 2026-10-27, task plan Omega12).

Question: can a fixed-budget Adam fit stand in for the multi-start L-BFGS fit of GP-MLL refit (P0.10), and would that remove
the optimiser-driven instability of the empirical-Bayes band? The P0.10 audit declared Adam-100 (lr 0.1, float32)
unconverged because its *final gradient* was 1.6e-2 against 6e-6 for L-BFGS (0 of 90 BO cells). The 2026-10-01 note in
``GPSurrogate._convergence_flag`` already records that a gradient test ranks the arms backwards (a 250-step Adam fit had the
smaller gradient and sat 0.35 nats worse), so this sweep pre-declares a criterion on the *fit* and on what the experiment
reads, the prediction:

    converged  <=>  NLML gap to the best-known fit <= ``--nll-tol`` nats  AND  median_x |mu_arm - mu_best| / sd_best <= ``--mean-tol``

The best-known fit is L-BFGS with many restarts and a long budget on the same data. Every arm sees the same datasets:
a context D (the base fit of an OMEGA cell) and three refits D + {(x', y')} with y' = mu(x') + z sd(x'), z in {-2, 0, 2},
which is the condition the 768 refits per cell actually run in. Also recorded per context: the three production fits that differ
only in ``restart_seed`` (how often the selected fit changes with the seed), and the *refit nonlinearity*
``median_x |mu(z=+2) + mu(z=-2) - 2 mu(z=0)| / sd_best(x)`` of each arm (0 for any model whose refit mean is affine in y',
e.g. GP-fixed, so it is the part of the drift the optimiser can inject).

    python scripts/gp_optimizer_sweep.py --shard 0/4
    python scripts/gp_optimizer_sweep.py --ts 40 --revisit-fraction 0.25 --tag revisit
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import (
    _channels, _n_revisits, _validate_bayes_update, load_mechanism_config,
)
from pfns4neurostim.models.gp.surrogates import GPSurrogate
from pfns4neurostim.seeding import rng_for, set_seed

#: Standardised surprises of the three refit datasets (y' = mu + z sd at the anchor).
REFIT_Z: tuple[float, ...] = (-2.0, 0.0, 2.0)
#: Adam grid (user 2026-10-27): steps x learning rate.
ADAM_STEPS: tuple[int, ...] = (100, 200, 400)
ADAM_LRS: tuple[float, ...] = (0.01, 0.05, 0.1)
#: ``restart_seed`` of the production-recipe repeats and of the best-known fit (disjoint from the production seed 0).
PRODUCTION_SEEDS: tuple[int, ...] = (0, 1, 2)
BEST_SEED: int = 100


def arm_specs(base: dict[str, Any], best_restarts: int, best_steps: int) -> dict[str, dict[str, Any]]:
    """The fit recipes under comparison, each a ``GPSurrogate`` kwargs dict derived from the config's ``gp_params``.

    Args:
        base: The config's ``bayes_update.gp_params`` (production L-BFGS recipe).
        best_restarts: Restarts of the best-known reference fit.
        best_steps: L-BFGS ``max_iter`` of the best-known reference fit.

    Returns:
        ``{arm name: kwargs}``; the reference is ``best``.
    """
    specs: dict[str, dict[str, Any]] = {"best": {**base, "optimizer": "lbfgs", "n_restarts": best_restarts,
                                                 "n_opt_steps": best_steps, "restart_seed": BEST_SEED}}
    for s in PRODUCTION_SEEDS:
        specs[f"lbfgs{base['n_restarts']}_s{s}"] = {**base, "restart_seed": s}
    specs["lbfgs1_init0"] = {**base, "n_restarts": 1}
    for steps in ADAM_STEPS:
        for lr in ADAM_LRS:
            specs[f"adam_{steps}_{lr:g}"] = {**base, "optimizer": "adam", "n_opt_steps": steps, "lr": lr,
                                             "n_restarts": 1}
    return specs


class WarmGP(GPSurrogate):
    """``GPSurrogate`` whose single start is a given hyperparameter vector instead of gpytorch's initialisation.

    The candidate "warm start" of the yardstick: every refit D + {(x', y')} starts from the fit on D, so the refits stay
    in the base fit's basin (continuation in y') and cost one L-BFGS run instead of ``n_restarts``.

    Args:
        warm_init: ``{'lengthscale', 'outputscale', 'noise'}`` of the start (scalar lengthscale, as the multi-start inits).
        **kwargs: ``GPSurrogate`` arguments (``n_restarts`` is 1).
    """

    def __init__(self, warm_init: dict[str, float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._warm_init = warm_init

    def _fit_candidate(self, train_x: torch.Tensor, train_y: torch.Tensor, init: dict[str, float] | None,
                       floor: float) -> tuple[float, Any, Any, dict[str, float]]:
        """Replace the default start by the warm one."""
        return super()._fit_candidate(train_x, train_y, self._warm_init if init is None else init, floor)


def warm_init_of(hp: dict[str, Any]) -> dict[str, float]:
    """Start vector of a warm fit from a fit's hyperparameters (geometric-mean lengthscale over the ARD dimensions)."""
    return {"lengthscale": float(np.exp(np.mean(np.log(np.asarray(hp["lengthscale"], dtype=np.float64))))),
            "outputscale": float(hp["outputscale"]), "noise": float(hp["noise"])}


def fit_arm(kwargs: dict[str, Any], X: np.ndarray, y: np.ndarray, Q: np.ndarray,
            warm_init: dict[str, float] | None = None) -> dict[str, Any]:
    """Fit one arm and read its predictive and diagnostics.

    Args:
        kwargs: ``GPSurrogate`` keyword arguments.
        X: Training inputs, shape [N, D].
        y: Training targets, shape [N].
        Q: Readout sites, shape [M, D].
        warm_init: A start vector; the arm is then a :class:`WarmGP`.

    Returns:
        ``mean``/``sd`` over ``Q`` ([M] each), the fit diagnostics and the fitted hyperparameters ``hp``.
    """
    gp = GPSurrogate(**kwargs) if warm_init is None else WarmGP(warm_init, **kwargs)
    t0 = time.time()
    gp.fit(X, y)
    seconds = time.time() - t0
    mean, sd = gp.predict_frozen(Q)                                                  # [M], [M]
    return {"mean": mean, "sd": sd, "diag": gp.fit_diagnostics(), "seconds": seconds, "hp": gp.hyperparameters()}


def main() -> None:
    """Run the sweep and write ``<out>/sweep[_<tag>].csv`` (rows per dataset and arm) plus ``nonlinearity`` rows."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--ts", type=int, nargs="+", default=[30, 40, 50, 80])
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--channels-per-subject", type=int, default=None, help="First k channels per subject (default all).")
    ap.add_argument("--revisit-fraction", type=float, default=0.0, help="Fraction of context rows that revisit a site.")
    ap.add_argument("--best-restarts", type=int, default=20)
    ap.add_argument("--best-steps", type=int, default=100)
    ap.add_argument("--nll-tol", type=float, default=0.01, help="Converged: NLML gap to the best-known fit (nats).")
    ap.add_argument("--mean-tol", type=float, default=0.01, help="Converged: median |mean difference| / best sd.")
    ap.add_argument("--warm", action="store_true", help="Add the warm-start arms (refits start from the base fit).")
    ap.add_argument("--arms", nargs="+", default=None, help="Keep only these fit arms (best is always kept).")
    ap.add_argument("--shard", default="0/1", help="i/n: keep every n-th context.")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default="output/diagnostics/gp_optimizer")
    args = ap.parse_args()

    torch.set_num_threads(1)
    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    shard = tuple(int(v) for v in args.shard.split("/"))
    out_csv = os.path.join(args.out, f"sweep_{args.tag}_shard{shard[0]}of{shard[1]}.csv")
    key_cols = ["subject", "emg", "context_t", "draw"]
    done: set[tuple[Any, ...]] = set()
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        done = {tuple(r) for r in old[key_cols].itertuples(index=False, name=None)}

    channels = _channels(cfg)
    if args.channels_per_subject is not None:
        from pfns4neurostim.experiments.mechanism import _first_per_subject  # noqa: PLC0415

        keep = _first_per_subject(channels, int(args.channels_per_subject))
        channels = [c for c in channels if c.label in keep]
    specs = arm_specs(dict(p["gp_params"]), args.best_restarts, args.best_steps)
    prod_s0 = f"lbfgs{p['gp_params']['n_restarts']}_s0"
    if args.arms is not None:
        specs = {k: v for k, v in specs.items() if k in {*args.arms, "best", prod_s0, "lbfgs1_init0"}}
    knob = build_knob(p["stress"]["knob"], [args.level])
    revisit_p = {"revisits": {"fraction": float(args.revisit_fraction), "row_order": "sorted"}}
    tasks = [(ch, int(t), int(d)) for ch in channels for t in args.ts for d in args.draws]
    tasks = [task for i, task in enumerate(tasks) if i % shard[1] == shard[0]]
    print(f"[gp_sweep] {len(tasks)} contexts, {len(specs)} arms, {len(REFIT_Z)} refits + base ({len(done)} done)", flush=True)

    t_start = time.time()
    for i, (ch, t, draw) in enumerate(tasks):
        if (ch.subject, ch.emg, t, draw) in done:
            continue
        st = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
        ctx = context_for(st, t, draw, base_seed=cfg.seed, level=float(args.level),
                          n_revisits=_n_revisits(revisit_p, t))
        rng = rng_for(ch.label, "bayes_update", args.level, t, draw, base_seed=cfg.seed)
        anchors, _, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]), int(p["anchors"]["n_observed"]),
                                                 rng, ctx.sites)
        a = int(anchors[0])                                                           # first unobserved anchor
        Q = st.X_pool                                                                 # [M, D]
        ref = fit_arm(specs[f"lbfgs{p['gp_params']['n_restarts']}_s0"], ctx.X, ctx.y, Q)
        m_a, s_a = float(ref["mean"][a]), float(ref["sd"][a])
        datasets = {"base": (ctx.X, ctx.y)}
        for z in REFIT_Z:
            datasets[f"z{z:+g}"] = (np.vstack([ctx.X, Q[a]]), np.append(ctx.y, m_a + z * s_a))
        fits: dict[str, dict[str, dict[str, Any]]] = {}
        rows: list[dict[str, Any]] = []
        warm_from: dict[str, dict[str, float]] = {}
        if args.warm:
            # warm arms: the refits start from the base fit of the production recipe (seed 0) / of the best-known fit
            for name, src in (("warm_prod", f"lbfgs{p['gp_params']['n_restarts']}_s0"), ("warm_best", "best")):
                warm_from[name] = warm_init_of(fit_arm(specs[src], ctx.X, ctx.y, Q)["hp"])
        for ds, (Xd, yd) in datasets.items():
            fits[ds] = {arm: fit_arm(kw, Xd, yd, Q) for arm, kw in specs.items()}
            if args.warm and ds != "base":
                for name, init in warm_from.items():
                    fits[ds][name] = fit_arm({**specs["lbfgs1_init0"]}, Xd, yd, Q, warm_init=init)
            best = fits[ds]["best"]
            for arm, f in fits[ds].items():
                d = f["diag"]
                gap = float(d["gp_mll_final"] - best["diag"]["gp_mll_final"]) * float(len(yd))   # NLML is per-point
                dmu = np.abs(f["mean"] - best["mean"]) / best["sd"]
                rows.append({
                    "subject": ch.subject, "emg": ch.emg, "context_t": t, "draw": draw, "dataset": ds, "arm": arm,
                    "n_train": len(yd), "mll_final": d["gp_mll_final"], "nll_gap_nats": gap,
                    "dmu_median": float(np.median(dmu)), "dmu_max": float(dmu.max()),
                    "sd_ratio_median": float(np.median(f["sd"] / best["sd"])),
                    "grad_max_free": d.get("gp_grad_max_free"), "steps": d["gp_n_opt_steps"],
                    "evals": d["gp_n_obj_evals"], "degenerate": d["gp_fit_degenerate"], "noise": d["gp_noise"],
                    "lengthscale_min": d["gp_lengthscale_min"], "lengthscale_max": d["gp_lengthscale_max"],
                    "outputscale": d["gp_outputscale"], "seconds": f["seconds"],
                    "converged": float(gap <= args.nll_tol and float(np.median(dmu)) <= args.mean_tol)})
        sd_best = fits["base"]["best"]["sd"]
        for arm in fits["z+0"]:
            nl = np.abs(fits["z+2"][arm]["mean"] + fits["z-2"][arm]["mean"] - 2.0 * fits["z+0"][arm]["mean"]) / sd_best
            rows.append({"subject": ch.subject, "emg": ch.emg, "context_t": t, "draw": draw, "dataset": "nonlinearity",
                         "arm": arm, "n_train": len(ctx.y) + 1, "nonlin_median": float(np.median(nl)),
                         "nonlin_max": float(nl.max())})
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        best_rows = [r for r in rows if r["arm"] == "best" and r["dataset"] == "base"]
        print(f"[gp_sweep] {i + 1}/{len(tasks)} ({time.time() - t_start:.0f}s) {ch.label} t={t} d={draw} "
              f"best noise {best_rows[0]['noise']:.2e}", flush=True)
    print(f"[gp_sweep] done ({time.time() - t_start:.0f}s) -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
