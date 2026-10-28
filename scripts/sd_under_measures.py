"""Predictive standard deviation of TabPFN under the three tail conventions, per channel (exploratory; user 2026-10-28).

For every NHP channel (level 1, draw 0, context size ``--t``) the standard deviation at each pool site under
    A  bell-shaped tails   (``criterion.variance``, the library's own),
    B  flat edges          (outer buckets uniform on their finite width: what ``icdf`` / ``cdf`` / sampling use),
    C  drop the edges      (outer buckets removed, interior renormalised).
Reported per channel: the median over pool sites of each SD, the median ratios B / A and C / A, and the share of sites where the
outer buckets carry more than half of A's variance.

    python scripts/sd_under_measures.py --t 40
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import _channels, _make_engine, _validate_bayes_update, load_mechanism_config
from pfns4neurostim.seeding import rng_for, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rqmc_measure_study import MeasureCriterionPredictive  # noqa: E402

#: Share of A's variance in the outer buckets above which a site counts as tail-dominated.
TAIL_DOMINATED: float = 0.5


def main() -> None:
    """Write ``sd_under_measures_t<t>.csv`` and print it."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--t", type=int, default=40)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="output/diagnostics/rqmc")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    knob = build_knob(p["stress"]["knob"], [args.level])
    rows = []
    for ch in _channels(cfg):
        st = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
        ctx = context_for(st, args.t, args.draw, base_seed=cfg.seed, level=float(args.level))
        eng = _make_engine("tabpfn_v2_5", p, "cpu", args.draw)
        eng.fit(ctx.X, ctx.y)
        base = eng.base(st.X_pool)
        sd_a = np.sqrt(base.var[0])                                                  # [M]
        sd_b = np.sqrt(MeasureCriterionPredictive(base.parts, "linear").var[0])      # [M]
        sd_c = np.sqrt(MeasureCriterionPredictive(base.parts, "central").var[0])     # [M]
        share_outer = 1.0 - (sd_c ** 2) / (sd_a ** 2)                                # approx: interior variance over A's
        rows.append({"channel": ch.label, "sd_A": np.median(sd_a), "sd_B": np.median(sd_b), "sd_C": np.median(sd_c),
                     "B_over_A": np.median(sd_b / sd_a), "C_over_A": np.median(sd_c / sd_a),
                     "p10_B_over_A": np.quantile(sd_b / sd_a, 0.1),
                     "tail_dominated": float(np.mean(share_outer > TAIL_DOMINATED))})
        print(f"[sd] {ch.label}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"sd_under_measures_t{args.t}.csv"), index=False)
    print(df.round(3).to_string(index=False))
    print(df[["sd_A", "sd_B", "sd_C", "B_over_A", "C_over_A", "tail_dominated"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
