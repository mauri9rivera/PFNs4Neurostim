"""Exchangeability test of the predictive (exploratory; panel (d) of the NFM coherence figure, user 2026-10-25).

A Bayesian model conditions on a *set*: ``p(y | x, D)`` must not depend on the order in which the context rows are
listed (de Finetti / exchangeability). TabPFN reads the rows as a sequence, so this is a property to measure, not one
that holds by construction (``PFNRefitEngine`` docstring: permuting D moved the predictive SD by up to 0.016 and, at a
revisited site, the mean by up to 0.09). For every context, P random row orders of the same D are fitted and the
predictive at every site is read; the readout is the disagreement between two orders, in the model's own predictive SD
(the units of panel (a)'s drift):

    drift_perm(x) = E_{p<q} |mu_p(x) - mu_q(x)| / ((s_p(x) + s_q(x)) / 2)     [per site; median over sites]

0 for any exchangeable model. GP-fixed is exactly exchangeable (a closed-form conditional of a Gaussian), so its value is
the floating-point floor; GP-MLL refit is exchangeable in exact arithmetic, but L-BFGS on a non-convex marginal likelihood
can land in another optimum when the rounding of a permuted kernel matrix differs, so its value is measured, not assumed.

Contexts are the OMEGA cells' own (``context_for`` after ``knob.apply`` at the level, same seed and draw), so panel (d)
sits on the layouts of panels (a)-(c). Writes ``exchangeability.csv`` (one row per cell and engine, resumable) in the
schema the coherence figure reads (``subject``, ``emg``, ``draw``, ``engine``, ``context_t``, ``level``).

    python scripts/exchangeability_drift.py --engines tabpfn_v2_5 gp_fixed --device cuda
    python scripts/exchangeability_drift.py --engines gp_mll_refit --channels-per-subject 1 --n-orders 4 --shard 0/4
"""
from __future__ import annotations

import argparse
import itertools
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import (
    _channels, _first_per_subject, _make_engine, _validate_bayes_update, load_mechanism_config,
)
from pfns4neurostim.seeding import rng_for, set_seed

#: Columns that identify one (cell, engine) row.
KEY_COLUMNS: tuple[str, ...] = ("subject", "emg", "draw", "engine", "context_t", "level")


def order_disagreement(mean: np.ndarray, var: np.ndarray, in_context: np.ndarray) -> dict[str, float]:
    """Disagreement between row orders of one context.

    Args:
        mean: Predictive mean at every site under each row order, shape [P, M].
        var: Predictive variance, shape [P, M].
        in_context: Whether each site is a context site, shape [M].

    Returns:
        ``drift_perm`` (median over sites of the pair-averaged |mean difference| / mean SD), the same over the context
        sites and over the unobserved sites, ``max_drift_perm`` (worst site) and ``sd_log_perm`` (median over sites of
        the pair-averaged |log SD ratio|).

    Raises:
        ValueError: With fewer than two orders.
        RuntimeError: If a readout is non-finite.
    """
    P = mean.shape[0]
    if P < 2:
        raise ValueError(f"order_disagreement: need at least 2 row orders, got {P}.")
    sd = np.sqrt(var)                                                                # [P, M]
    pairs = list(itertools.combinations(range(P), 2))
    d = np.mean([np.abs(mean[p] - mean[q]) / (0.5 * (sd[p] + sd[q])) for p, q in pairs], axis=0)        # [M]
    ls = np.mean([np.abs(np.log(sd[p] / sd[q])) for p, q in pairs], axis=0)          # [M]
    if not (np.isfinite(d).all() and np.isfinite(ls).all()):
        raise RuntimeError("order_disagreement: non-finite readout.")
    ctx = np.asarray(in_context, dtype=bool)
    return {"drift_perm": float(np.median(d)), "drift_perm_context": float(np.median(d[ctx])),
            "drift_perm_unobserved": float(np.median(d[~ctx])) if (~ctx).any() else float("nan"),
            "max_drift_perm": float(d.max()), "sd_log_perm": float(np.median(ls))}


def predictives_over_orders(engine_name: str, p: dict[str, Any], device: str, draw: int, X: np.ndarray,
                            y: np.ndarray, Q: np.ndarray, n_orders: int, rng: np.random.Generator,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Predictive mean and variance at ``Q`` for ``n_orders`` random row orders of the context ``(X, y)``.

    Args:
        engine_name: ``tabpfn_v2_5`` (the OMEGA public-API refit engine), ``gp_fixed`` or ``gp_mll_refit``.
        p: Resolved ``bayes_update`` block (engine settings).
        device: Torch device of TabPFN.
        draw: Context draw (TabPFN's inference seed is ``inference_seed + draw``, as in the OMEGA cells).
        X: Context inputs, shape [t, D].
        y: Context responses, shape [t].
        Q: Sites to predict, shape [M, D].
        n_orders: Row orders.
        rng: Generator of the row orders.

    Returns:
        ``(mean, var)``, each shape [n_orders, M].
    """
    means, variances = [], []
    if engine_name == "tabpfn_v2_5":
        # one engine: every base() call re-fits on a fresh random order of the same rows, with the same inference seed
        tp = p["tabpfn"]
        eng = B.PFNRefitEngine(device=device, random_state=int(tp["inference_seed"]) + int(draw),
                               softmax_temperature=float(tp["softmax_temperature"]), name=engine_name,
                               shuffle="all", n_permutations=1, rng=rng)
        eng.fit(X, y)
        for _ in range(n_orders):
            pred = eng.base(Q)
            means.append(pred.mean[0])
            variances.append(pred.var[0])
    else:
        for _ in range(n_orders):
            o = rng.permutation(len(y))                                              # [t]
            eng = _make_engine(engine_name, p, device, draw)
            eng.fit(X[o], y[o])
            pred = eng.base(Q)
            means.append(pred.mean[0])
            variances.append(pred.var[0])
    return np.stack(means), np.stack(variances)                                      # [P, M] x 2


def main() -> None:
    """Run the exchangeability test over the OMEGA refinement grid's contexts and write ``exchangeability.csv``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml",
                    help="OMEGA config (dataset, seeds and engine settings).")
    ap.add_argument("--engines", nargs="+", default=["tabpfn_v2_5", "gp_fixed"],
                    choices=["tabpfn_v2_5", "gp_fixed", "gp_mll_refit"], help="Engines to test.")
    ap.add_argument("--device", default="cuda", help="Torch device of TabPFN.")
    ap.add_argument("--level", type=float, default=1.0, help="Stress level (K2 noise multiplier) of the contexts.")
    ap.add_argument("--ts", type=int, nargs="+", default=[30, 40, 50, 60, 80], help="Context sizes.")
    ap.add_argument("--draws", type=int, nargs="+", default=list(range(10)), help="Context draws (layouts).")
    ap.add_argument("--n-orders", type=int, default=8, help="Row orders per context (pairs = n (n - 1) / 2).")
    ap.add_argument("--channels-per-subject", type=int, default=None, help="First k channels of each subject.")
    ap.add_argument("--shard", default=None, help="i/n: keep every n-th task starting at i (parallel lanes).")
    ap.add_argument("--out", default="output/diagnostics/exchangeability", help="Output directory.")
    args = ap.parse_args()

    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    shard = (0, 1) if args.shard is None else tuple(int(v) for v in args.shard.split("/"))
    out_csv = os.path.join(args.out, "exchangeability.csv" if args.shard is None
                           else f"exchangeability_shard{shard[0]}of{shard[1]}.csv")
    done: set[tuple[Any, ...]] = set()
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        done = {tuple(r) for r in old[list(KEY_COLUMNS)].itertuples(index=False, name=None)}

    channels = _channels(cfg)
    if args.channels_per_subject is not None:
        keep = _first_per_subject(channels, int(args.channels_per_subject))
        channels = [c for c in channels if c.label in keep]
    knob = build_knob(p["stress"]["knob"], [args.level])
    tasks = [(ch, int(t), int(d), e) for ch in channels for t in args.ts for d in args.draws for e in args.engines]
    tasks = [task for i, task in enumerate(tasks) if i % shard[1] == shard[0]]
    print(f"[exchangeability] {len(tasks)} tasks ({len(done)} rows already in {out_csv})", flush=True)

    stressed: dict[str, Any] = {}
    t0 = time.time()
    for i, (ch, t, draw, engine) in enumerate(tasks):
        key = (ch.subject, ch.emg, draw, engine, t, float(args.level))
        if key in done:
            continue
        if ch.label not in stressed:
            stressed[ch.label] = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level,
                                                                      base_seed=cfg.seed))
        st = stressed[ch.label]
        ctx = context_for(st, t, draw, base_seed=cfg.seed, level=float(args.level))
        in_context = np.isin(np.arange(st.n_sites), ctx.sites)
        rng = rng_for(ch.label, "exchangeability", engine, t, draw, base_seed=cfg.seed)
        mean, var = predictives_over_orders(engine, p, args.device, draw, ctx.X, ctx.y, st.X_pool, args.n_orders, rng)
        row = {"dataset": ch.dataset, "subject": ch.subject, "emg": ch.emg, "draw": draw, "engine": engine,
               "context_t": t, "level": float(args.level), "n_orders": args.n_orders,
               **order_disagreement(mean, var, in_context)}
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        if (i + 1) % 20 == 0:
            print(f"[exchangeability] {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s) {ch.label} t={t} d={draw} "
                  f"{engine}: drift_perm {row['drift_perm']:.4g}", flush=True)
    print(f"[exchangeability] done ({time.time() - t0:.0f}s) -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
