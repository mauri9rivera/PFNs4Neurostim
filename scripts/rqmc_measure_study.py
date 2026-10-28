"""Which predictive measure, and how many deterministic nodes? (exploratory; user 2026-10-27, advisor-framed)

``rqmc_diagnostic.py`` showed that TabPFN's stated variance is dominated by a half-normal tail that no quantile-based node reaches
(``FullSupportBarDistribution.mean`` / ``variance`` use half-normal outer buckets, ``icdf`` / ``cdf`` use uniform ones), so the readouts
that divide by the stated variance inherit a convention. This study crosses three **measures** with deterministic node counts ``K``:

    A  stated        interior strata over the bars + the two official half-normal tails (two nodes each, matching the tail's mean and
                     variance); the stated moments are the criterion's own (``mean`` / ``variance``)
    B  linear        the same with the outer buckets uniform on their finite width (two nodes each, matching mean and variance): the
                     density ``icdf`` / ``cdf`` (hence Thompson sampling and ``bar_quantiles``) use; stated moments are the linear ones
    C  truncated     interior strata only, outer buckets dropped and the interior renormalised; stated moments likewise

Nodes are the **exact** conditional means of equal-mass strata (uniform inside every bucket), so the node set reproduces the measure's
mean exactly and its variance up to the within-stratum part, which shrinks with ``K``. There is no RQMC noise. GP-fixed (affine refit) is
the exactness control with Gaussian strata. Output: ``measure_study.csv`` next to the other quadrature diagnostics.

    python scripts/rqmc_measure_study.py --engines tabpfn_v2_5 --subjects 0
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
from scipy.special import ndtr, ndtri

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import (
    _channels, _first_per_subject, _make_engine, _validate_bayes_update, load_mechanism_config,
)
from pfns4neurostim.seeding import rng_for, set_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEASURES: tuple[str, ...] = ("A", "B", "C")
KS: tuple[int, ...] = (16, 32, 64, 128)


class MeasureCriterionPredictive(B.CriterionPredictive):
    """``CriterionPredictive`` whose stated moments follow a tail convention.

    Args:
        parts: As :class:`~pfns4neurostim.analysis.bayes_update.CriterionPredictive`.
        mode: ``'linear'`` (outer buckets uniform on their width) or ``'central'`` (outer buckets dropped, interior renormalised).
    """

    def __init__(self, parts: Any, mode: str) -> None:
        super().__init__(parts)
        means, varis = [], []
        for crit, logits in self.parts:
            p = torch.softmax(logits.double(), -1)                                  # [M, bars]
            if mode == "central":
                p = p.clone()
                p[:, 0] = 0.0
                p[:, -1] = 0.0
                p = p / p.sum(-1, keepdim=True)
            b = crit.borders.double().to(p.device)                                  # [bars + 1]
            lo, hi = b[:-1], b[1:]
            m1 = (p * (lo + hi) / 2).sum(-1)                                        # [M]
            m2 = (p * (lo ** 2 + lo * hi + hi ** 2) / 3).sum(-1)                    # [M]
            means.append(m1.cpu().numpy())
            varis.append((m2 - m1 ** 2).clamp_min(0).cpu().numpy())
        self.mean = np.stack(means).astype(np.float64)                              # [P, M]
        self.var = np.stack(varis).astype(np.float64)                               # [P, M]


def strata_means(crit: Any, probs: torch.Tensor, lo_u: float, hi_u: float, K: int) -> np.ndarray:
    """Exact conditional means of ``K`` equal-mass strata of the bar distribution between two cumulative levels.

    Uniform inside every bucket (zero-mass and zero-width buckets contribute nothing).

    Args:
        crit: ``FullSupportBarDistribution`` (only its borders are read).
        probs: Bucket probabilities at one site, shape [bars], float64.
        lo_u: Lower cumulative level of the mass to stratify.
        hi_u: Upper cumulative level.
        K: Number of strata.

    Returns:
        Conditional means, shape [K].
    """
    b = crit.borders.double().to(probs.device)                                      # [bars + 1]
    c = torch.cumsum(probs, -1)                                                     # [bars]
    c0 = c - probs                                                                  # [bars]
    edges = lo_u + (hi_u - lo_u) * torch.arange(K + 1, dtype=torch.float64, device=probs.device) / K     # [K + 1]
    a = torch.maximum(c0[:, None], edges[None, :-1])                                # [bars, K]
    e = torch.minimum(c[:, None], edges[None, 1:])                                  # [bars, K]
    m = (e - a).clamp_min(0.0)                                                      # [bars, K] mass of bucket b in stratum k
    frac = torch.where(probs[:, None] > 0, ((a + e) / 2 - c0[:, None]) / probs[:, None].clamp_min(1e-300), torch.zeros_like(m))
    y = b[:-1, None] + (b[1:] - b[:-1])[:, None] * frac.clamp(0.0, 1.0)             # [bars, K]
    return ((m * y).sum(0) / m.sum(0).clamp_min(1e-300)).cpu().numpy()              # [K]


def measure_nodes(base: B.CriterionPredictive, anchor: int, measure: str, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic nodes and weights of one measure at one anchor.

    Args:
        base: Predictive given D (one part).
        anchor: Anchor site.
        measure: ``'A'``, ``'B'`` or ``'C'``.
        K: Interior strata (``A`` and ``B`` add four tail nodes).

    Returns:
        ``(y [n], w [n])``, weights summing to 1.

    Raises:
        ValueError: On an unknown measure.
    """
    crit, logits = base.parts[0]
    probs = torch.softmax(logits[anchor].double(), -1)                              # [bars]
    p0, pL = float(probs[0]), float(probs[-1])
    if measure not in MEASURES:
        raise ValueError(f"unknown measure {measure!r}")
    inner = strata_means(crit, probs, p0, 1.0 - pL, K)                              # [K], the interior mass
    w_in = np.full(K, (1.0 - p0 - pL) / K)
    if measure == "C":
        return inner, np.full(K, 1.0 / K)
    b = crit.borders.double().cpu()
    if measure == "A":
        left = crit.halfnormal_with_p_weight_before(crit.bucket_widths[0])
        right = crit.halfnormal_with_p_weight_before(crit.bucket_widths[-1])
        ml, sl = float(b[1] - left.mean.double()), float(left.variance.double().sqrt())
        mr, sr = float(b[-2] + right.mean.double()), float(right.variance.double().sqrt())
    else:                                                                           # B: uniform on the finite outer buckets
        wl, wr = float(b[1] - b[0]), float(b[-1] - b[-2])
        ml, sl = float(b[0] + wl / 2), wl / np.sqrt(12.0)
        mr, sr = float(b[-2] + wr / 2), wr / np.sqrt(12.0)
    y = np.concatenate([[ml - sl, ml + sl], inner, [mr - sr, mr + sr]])
    w = np.concatenate([[p0 / 2, p0 / 2], w_in, [pL / 2, pL / 2]])
    return y, w


def gauss_nodes(mu: float, sd: float, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact conditional means of ``K`` equal-mass Gaussian strata (the exactness control)."""
    z = ndtri(np.arange(K + 1) / K)                                                 # [K + 1], +-inf at the ends
    pdf = np.exp(-0.5 * np.where(np.isfinite(z), z, 0.0) ** 2) / np.sqrt(2 * np.pi) * np.isfinite(z)
    return mu + sd * K * (pdf[:-1] - pdf[1:]), np.full(K, 1.0 / K)


def readouts(engine: Any, base: B.Predictive, Q: np.ndarray, anchors: np.ndarray, trial_var: np.ndarray, eps: float,
             u_tail: float, grid_points: int, node_fn: Any, convert: Any) -> dict[str, float]:
    """Pooled Part A readouts for one node rule, raw and moment-matched.

    Args:
        engine: Fitted engine.
        base: Its predictive given D.
        Q: Readout sites, shape [M, D].
        anchors: Anchor sites, shape [A].
        trial_var: Trial variance at each anchor, shape [A].
        eps: Influenced-site threshold.
        u_tail: Tail probability left off the W1 grid.
        grid_points: W1 grid points per site.
        node_fn: ``node_fn(base, anchor) -> (y, w)``.
        convert: Applied to the base and probe predictives (the measure's stated moments).

    Returns:
        Pooled medians, the median measure variance ratio and the node count.
    """
    base_m = convert(base)
    rows, Cs, Ccs = [], [], []
    for a in anchors:
        y, w = node_fn(base, int(a))
        probes = convert(engine.probe(np.repeat(Q[[a]], len(y), axis=0), y, Q))
        row, C, Cc = B.coherence_scheme(engine, base_m, Q, int(a), np.zeros((1, 1)), w, eps, u_tail=u_tail,
                                        grid_points=grid_points, extra=[(y, probes)])
        rows.append(row), Cs.append(C), Ccs.append(Cc)
    every = [B.POOLED_ANCHORS] * len(anchors)
    cell = B.coherence_cell(rows, np.asarray(Cs), anchors, every, trial_var, C_c=np.asarray(Ccs), pool_names=(B.POOLED_ANCHORS,))[0]
    out = {k: cell.get(k) for k in ("cov_asymmetry", "cov_asymmetry_c", "drift_abs", "drift_abs_c", "total_var_ratio",
                                    "total_var_ratio_c")}
    out["measure_var_ratio"] = float(np.median([r["measure_var_ratio"] for r in rows]))
    out["n_nodes"] = float(len(y))
    return out


def main() -> None:
    """Run the study and write ``measure_study.csv`` (resumable per cell and scheme)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--engines", nargs="+", default=["tabpfn_v2_5"], choices=["gp_fixed", "tabpfn_v2_5"])
    ap.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--ts", type=int, nargs="+", default=[40])
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--measures", nargs="+", default=list(MEASURES))
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default="output/diagnostics/rqmc")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    tag = "-".join(str(s) for s in args.subjects) + "_" + "-".join(args.engines)
    out_csv = os.path.join(args.out, f"measure_study_{tag}.csv")
    keys = ["engine", "subject", "emg", "context_t", "draw", "measure", "K"]
    done = set()
    if os.path.exists(out_csv):
        done = {tuple(r) for r in pd.read_csv(out_csv)[keys].itertuples(index=False, name=None)}
    channels = [c for c in _channels(cfg) if c.label in _first_per_subject(_channels(cfg), 1) and c.subject in args.subjects]
    knob = build_knob(p["stress"]["knob"], [args.level])
    yg = p["y_grid"]
    t0 = time.time()
    for ch in channels:
        st = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
        for t in args.ts:
            ctx = context_for(st, t, args.draw, base_seed=cfg.seed, level=float(args.level))
            rng = rng_for(ch.label, "bayes_update", args.level, t, args.draw, base_seed=cfg.seed)
            anchors, pools, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]), int(p["anchors"]["n_observed"]), rng, ctx.sites)
            tv = np.nanvar(st.Y_trials[anchors], axis=1, ddof=1)
            for name in args.engines:
                eng = _make_engine(name, p, "cpu", args.draw)
                eng.fit(ctx.X, ctx.y)
                base = eng.base(st.X_pool)
                jobs: list[tuple[str, int, Any, Any]] = []
                for K in args.ks:
                    if name == "gp_fixed":
                        jobs.append(("gauss", K, lambda b, a, K=K: gauss_nodes(float(b.mean[0][a]), float(np.sqrt(b.var[0][a])), K), lambda x: x))
                        continue
                    for m in args.measures:
                        conv = {"A": (lambda x: x), "B": (lambda x: MeasureCriterionPredictive(x.parts, "linear")),
                                "C": (lambda x: MeasureCriterionPredictive(x.parts, "central"))}[m]
                        jobs.append((m, K, lambda b, a, m=m, K=K: measure_nodes(b, a, m, K), conv))
                for m, K, node_fn, conv in jobs:
                    if (name, ch.subject, ch.emg, t, args.draw, m, K) in done:
                        continue
                    ts = time.time()
                    res = readouts(eng, base, st.X_pool, anchors, tv, float(p["eps"]), float(yg["u_tail"]), int(yg["points"]), node_fn, conv)
                    row = {"engine": name, "subject": ch.subject, "emg": ch.emg, "context_t": t, "draw": args.draw, "measure": m,
                           "K": K, "seconds": time.time() - ts, **res}
                    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                    print(f"[measure] {name} {ch.label} t={t} {m} K={K}: asym {res['cov_asymmetry']:.3f} / {res['cov_asymmetry_c']} "
                          f"mvr {res['measure_var_ratio']:.2f} ({time.time() - ts:.0f}s; total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
