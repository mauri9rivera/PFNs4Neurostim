"""Bounded versus unbounded coherence readouts under the three tail conventions (exploratory; user 2026-10-28, advisor-framed).

A readout that is a function of the *CDF* (a Kolmogorov distance, a grid total variation) can change between the measures A / B / C by at
most the mass of the outer buckets (1e-6 .. 7e-5): the three measures share every interior bucket and the same CDF. A readout that is a
function of a *second moment* (total-variance ratio, implied covariance, latent fraction) can be decided by the convention. This script
shows both on the same nodes and probes: per anchor, K deterministic strata per measure, then

    ks        median over sites of ``max_y |F_mix(y) - F_base(y)|`` (``F_mix`` = node-weighted mixture of the refit CDFs)
    tv_grid   median over sites of the grid total variation ``0.5 sum |dD|`` of ``D = F_mix - F_base`` (a lower bound on TV)
    drift_abs, total_var_ratio, cov_asymmetry (+ moment-matched)   the unbounded readouts of ``coherence_anchor``

    python scripts/rqmc_bounded_readouts.py --subjects 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import (
    _channels, _first_per_subject, _make_engine, _validate_bayes_update, load_mechanism_config,
)
from pfns4neurostim.seeding import rng_for, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rqmc_measure_study import MeasureCriterionPredictive, measure_nodes  # noqa: E402

K_STRATA: int = 64
CONVERT = {"A": lambda x: x, "B": lambda x: MeasureCriterionPredictive(x.parts, "linear"),
           "C": lambda x: MeasureCriterionPredictive(x.parts, "central")}


def main() -> None:
    """Run the comparison and write ``bounded_readouts_<subjects>.csv``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--t", type=int, default=40)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default="output/diagnostics/rqmc")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    eps, u_tail, pts = float(p["eps"]), float(p["y_grid"]["u_tail"]), int(p["y_grid"]["points"])
    knob = build_knob(p["stress"]["knob"], [args.level])
    out_csv = os.path.join(args.out, f"bounded_readouts_common_{'-'.join(map(str, args.subjects))}.csv")
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for ch in [c for c in _channels(cfg) if c.label in _first_per_subject(_channels(cfg), 1) and c.subject in args.subjects]:
        st = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
        Q = st.X_pool
        ctx = context_for(st, args.t, args.draw, base_seed=cfg.seed, level=float(args.level))
        rng = rng_for(ch.label, "bayes_update", args.level, args.t, args.draw, base_seed=cfg.seed)
        anchors, pools, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]), int(p["anchors"]["n_observed"]), rng, ctx.sites)
        tv_var = np.nanvar(st.Y_trials[anchors], axis=1, ddof=1)
        eng = _make_engine("tabpfn_v2_5", p, "cpu", args.draw)
        eng.fit(ctx.X, ctx.y)
        base = eng.base(Q)
        for m, conv in CONVERT.items():
            base_m = conv(base)
            arows, Cs, Ccs, ks, tvs = [], [], [], [], []
            for a in anchors:
                y, w = measure_nodes(base, int(a), m, K_STRATA)
                probes = eng.probe(np.repeat(Q[[a]], len(y), axis=0), y, Q)
                grid = B.site_grid([base, probes], u_tail, pts)                       # [M, G]
                row, C, Cc = B.coherence_anchor(base_m, conv(probes), y, w, int(a), grid, eps)
                common = B.site_grid([base], u_tail, pts)                     # [M, G] the SAME grid for every measure
                D = np.einsum("k,kmg->mg", w, probes.cdf(common)) - base.cdf(common)[0]   # [M, G]
                arows.append(row), Cs.append(C), Ccs.append(Cc)
                ks.append(np.median(np.abs(D).max(-1)))
                tvs.append(np.median(0.5 * np.abs(np.diff(D, axis=-1)).sum(-1)))
            every = [B.POOLED_ANCHORS] * len(anchors)
            cell = B.coherence_cell(arows, np.asarray(Cs), anchors, every, tv_var, C_c=np.asarray(Ccs), pool_names=(B.POOLED_ANCHORS,))[0]
            rows.append({"subject": ch.subject, "emg": ch.emg, "context_t": args.t, "measure": m, "K": K_STRATA,
                         "ks": float(np.median(ks)), "tv_grid": float(np.median(tvs)),
                         **{k: cell.get(k) for k in ("drift_abs", "drift_abs_c", "total_var_ratio", "total_var_ratio_c",
                                                     "cov_asymmetry", "cov_asymmetry_c", "latent_frac", "drift_w1")}})
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"[bounded] {ch.label} {m} ({time.time() - t0:.0f}s) ks {rows[-1]['ks']:.5f} tv {rows[-1]['tv_grid']:.5f} "
                  f"tvr_c {rows[-1]['total_var_ratio_c']}", flush=True)


if __name__ == "__main__":
    main()
