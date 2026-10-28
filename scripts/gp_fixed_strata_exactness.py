"""How exact are deterministic Gaussian strata on GP-fixed, raw and moment-matched? (amendment 2026-10-28, gate G1 evidence)

GP-fixed's refit is affine in y', so its implied covariance equals the latent posterior covariance times the node set's variance ratio
``v / s^2``: exact for the moment-matched readouts under any node set, and for the raw ones up to the within-stratum variance the
nodes drop. This reports, per number of strata K and for the old RQMC 8 x 8, the worst case over cells (first NHP channel per subject,
t = 40 and 80) of

    cov_err      max |C - Sigma| / max |Sigma|        (raw)         and    cov_err_c      (moment-matched)
    drift, tvr   max |drift|, max |total-variance ratio - 1|        (raw)  and the moment-matched versions
    asym         the pooled covariance asymmetry                      (raw)  and ``asym_c``
    mvr          the node set's variance / s^2 (median over anchors)

    python scripts/gp_fixed_strata_exactness.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import (
    _channels, _first_per_subject, _make_engine, _validate_bayes_update, load_mechanism_config,
)
from pfns4neurostim.seeding import rng_for, set_seed

SCHEMES: tuple[tuple[str, dict], ...] = (
    ("strata_K16", {"method": "strata", "K": 16}), ("strata_K32", {"method": "strata", "K": 32}),
    ("strata_K64", {"method": "strata", "K": 64}), ("strata_K128", {"method": "strata", "K": 128}),
    ("rqmc_8x8", {"method": "rqmc", "K": 8, "n_scrambles": 8}),
)


def main() -> None:
    """Run and write ``gp_fixed_strata_exactness.csv``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp_amended.yaml")
    ap.add_argument("--ts", type=int, nargs="+", default=[10, 25, 40, 80])
    ap.add_argument("--out", default="output/diagnostics/rqmc")
    args = ap.parse_args()

    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    knob = build_knob(p["stress"]["knob"], [1.0])
    rows = []
    channels = [c for c in _channels(cfg) if c.label in _first_per_subject(_channels(cfg), 1)]
    for ch in channels:
        st = knob.apply(ch, 1.0, rng_for(ch.label, "bayes_update", 1.0, base_seed=cfg.seed))
        for t in args.ts:
            ctx = context_for(st, t, 0, base_seed=cfg.seed, level=1.0)
            rng = rng_for(ch.label, "bayes_update", 1.0, t, 0, base_seed=cfg.seed)
            anchors, pools, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]), int(p["anchors"]["n_observed"]), rng, ctx.sites)
            eng = _make_engine("gp_fixed", p, "cpu")
            eng.fit(ctx.X, ctx.y)
            Sig = eng.posterior_covariance(st.X_pool)[anchors]
            scale = float(np.abs(Sig).max())
            tv = np.nanvar(st.Y_trials[anchors], axis=1, ddof=1)
            for name, spec in SCHEMES:
                kw = {"u_min": 1e-12, "rng": rng_for(ch.label, "exactness", name, t, base_seed=cfg.seed), **{k: v for k, v in spec.items() if k != "K"}}
                r, C, Cc = B.coherence(eng, st.X_pool, anchors, spec["K"], float(p["eps"]), u_tail=float(p["y_grid"]["u_tail"]),
                                       grid_points=int(p["y_grid"]["points"]), **kw)
                cell = B.coherence_cell(r, C, anchors, list(pools), tv, C_c=Cc)
                asym = max(c.get("cov_asymmetry") or 0.0 for c in cell)
                asym_c = max((c.get("cov_asymmetry_c") or 0.0) for c in cell)
                rows.append({"subject": ch.subject, "t": t, "scheme": name,
                             "cov_err": float(np.abs(C - Sig).max() / scale), "cov_err_c": float(np.abs(Cc - Sig).max() / scale),
                             "drift": max(x["max_abs_drift"] for x in r), "drift_c": max(x["max_abs_drift_c"] for x in r),
                             "tvr": max(x["max_abs_total_var_dev"] for x in r), "tvr_c": max(x["max_abs_total_var_dev_c"] for x in r),
                             "asym": asym, "asym_c": asym_c, "mvr": float(np.median([x["measure_var_ratio"] for x in r]))})
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "gp_fixed_strata_exactness.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.groupby("scheme").max(numeric_only=True).drop(columns=["subject", "t"]).round(6).to_string())
    print(df.groupby("scheme")["mvr"].min().round(4).to_string())


if __name__ == "__main__":
    main()
