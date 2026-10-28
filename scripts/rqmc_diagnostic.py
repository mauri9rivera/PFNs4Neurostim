"""Quadrature diagnostic of the OMEGA coherence readouts (exploratory; NFM follow-up, user 2026-10-27).

Part A integrates over the model's own predictive of ``y'`` at an anchor. The production scheme is RQMC with 8 scrambles
of 8 Sobol points, pushed through the model's quantile function. Two things make that suspect:

* a *raw* readout uses the quadrature measure's own variance ``v`` where theory uses the stated predictive variance
  ``s^2(x')``; the moment-matched ``*_c`` readouts rescale by ``rho = s^2 / v`` (``coherence_anchor``). For GP-fixed
  (affine refit) the ``*_c`` readouts are exact, so the raw floor (asymmetry ~ 0.09) is pure quadrature noise;
* TabPFN's predictive has heavy tails (outer bar buckets), so 64 nodes sample only a fraction of its stated variance
  (``measure_var_ratio`` median 0.30 on ``nhp-l1-refine``), which can bias the readouts the raw way.

For a few cells this script computes every readout under several schemes at (roughly) the same probe cost and reports raw
and moment-matched versions side by side, so the quadrature's contribution can be separated from the model's:

    rqmc_8x8      production: 8 scrambles x 8 Sobol points (64 probes per anchor)
    rqmc_8x32     4x the probes (256 per anchor)
    rqmc_1x64     one scramble of 64 points (64 probes; the node set is no longer split in eight)
    gh20          Gauss-Hermite, 20 nodes mapped through the quantile function (20 probes; exact for a Gaussian predictive)
    strat_64      64 equal-mass strata, node = conditional mean of the stratum (64 probes; the tail strata carry the tail)
    strat_tail_64 62 interior strata + the two official half-normal outer-bucket means as tail nodes (TabPFN only)

    python scripts/rqmc_diagnostic.py --engines gp_fixed --channels-per-subject 1 --ts 40 80
    python scripts/rqmc_diagnostic.py --engines tabpfn_v2_5 --device cuda --channels-per-subject 1 --ts 40 80
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import (
    _channels, _first_per_subject, _make_engine, _validate_bayes_update, load_mechanism_config,
)
from pfns4neurostim.seeding import rng_for, set_seed

SCHEMES: tuple[str, ...] = ("rqmc_8x8", "rqmc_8x32", "rqmc_1x64", "gh20", "strat_64", "strat_tail_64")
#: Sub-levels averaged inside each stratum of ``strat_64`` (the conditional mean is a mean of the quantile function).
STRATUM_SUBLEVELS: int = 32


def node_sets(scheme: str, base: B.Predictive, anchor: int, rng: np.random.Generator, u_min: float,
              ) -> tuple[list[np.ndarray], np.ndarray]:
    """The y' node sets (one per scramble) and weights of a scheme at one anchor.

    Args:
        scheme: One of :data:`SCHEMES`.
        base: Predictive given D.
        anchor: Anchor site.
        rng: Generator of the scrambles.
        u_min: Level clip.

    Returns:
        ``(ys, w)``: ``ys`` a list of ``[K]`` arrays, ``w`` shape ``[K]``.

    Raises:
        ValueError: On an unknown scheme.
    """
    if scheme in ("rqmc_8x8", "rqmc_8x32", "rqmc_1x64"):
        S, K = {"rqmc_8x8": (8, 8), "rqmc_8x32": (8, 32), "rqmc_1x64": (1, 64)}[scheme]
        u, w = B.quadrature_levels("rqmc", K, u_min=u_min, n_scrambles=S, rng=rng)                  # [S, K], [K]
        return [B.quadrature_values(base, anchor, us) for us in u], w
    if scheme == "gh20":
        u, w = B.quadrature_levels("gauss_hermite", 20, u_min=u_min)                                # [1, 20], [20]
        return [B.quadrature_values(base, anchor, u[0])], w
    if scheme == "strat_64":
        K = 64
        fine = (np.arange(K * STRATUM_SUBLEVELS) + 0.5) / (K * STRATUM_SUBLEVELS)                   # [K L]
        y = base.ppf(fine, np.array([anchor]))[0, 0]                                                # [K L]
        return [y.reshape(K, STRATUM_SUBLEVELS).mean(axis=1)], np.full(K, 1.0 / K)                  # [K]
    if scheme == "strat_tail_64":
        return tail_aware_nodes(base, anchor, 64)
    raise ValueError(f"node_sets: unknown scheme {scheme!r}; known {SCHEMES}.")


def tail_aware_nodes(base: Any, anchor: int, K: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Mass-stratified nodes whose two tail nodes are the official half-normal outer-bucket means (TabPFN only).

    ``FullSupportBarDistribution.mean`` / ``variance`` treat the two outer buckets as half-normal tails of infinite
    support, while ``icdf`` / ``cdf`` (and ``bar_quantiles``) treat them as uniform on their finite bucket. The outer
    buckets hold ~1e-6 .. 1e-4 of the mass but are 60-260 predictive SDs wide, so the stated variance carries a tail
    component no quantile-based node ever visits. Here the ``K - 2`` interior strata (equal mass, node = mean of the
    quantile function over the stratum) are completed by one node per outer bucket at the official half-normal bucket
    mean, weighted by the bucket's mass: the weighted node mean equals ``criterion.mean`` by construction.

    Args:
        base: A ``CriterionPredictive`` given D (one part).
        anchor: Anchor site.
        K: Total nodes (``K - 2`` interior + 2 tail).

    Returns:
        ``([y [K]], w [K])``.
    """
    import torch  # noqa: PLC0415

    crit, logits = base.parts[0]
    with torch.no_grad():
        probs = torch.softmax(logits[anchor].double(), -1)                                           # [bars]
        b = crit.borders.double().to(probs.device)                                                   # [bars + 1]
        left = crit.halfnormal_with_p_weight_before(crit.bucket_widths[0])
        right = crit.halfnormal_with_p_weight_before(crit.bucket_widths[-1])
        y_left = float((-left.mean + b[1]).cpu())
        y_right = float((right.mean + b[-2]).cpu())
        p0, pL = float(probs[0].cpu()), float(probs[-1].cpu())
    n_in = int(K) - 2
    mass = 1.0 - p0 - pL
    fine = p0 + mass * (np.arange(n_in * STRATUM_SUBLEVELS) + 0.5) / (n_in * STRATUM_SUBLEVELS)      # [n_in L]
    y = base.ppf(fine, np.array([anchor]))[0, 0].reshape(n_in, STRATUM_SUBLEVELS).mean(axis=1)       # [n_in]
    return [np.concatenate([[y_left], y, [y_right]])], np.concatenate([[p0], np.full(n_in, mass / n_in), [pL]])


def scheme_readouts(engine: Any, base: B.Predictive, Q: np.ndarray, anchors: np.ndarray, pools: list[str],
                    trial_var: np.ndarray, scheme: str, eps: float, rng: np.random.Generator, u_min: float,
                    u_tail: float, grid_points: int) -> dict[str, float]:
    """Pooled Part A readouts of one scheme on one fitted engine, raw and moment-matched.

    Args:
        engine: Fitted engine.
        base: Its predictive given D.
        Q: Readout sites, shape [M, D].
        anchors: Anchor sites, shape [A].
        pools: Pool of each anchor.
        trial_var: Trial variance at each anchor, shape [A].
        scheme: One of :data:`SCHEMES`.
        eps: Influenced-site threshold.
        rng: Scramble generator.
        u_min: Level clip.
        u_tail: Tail probability left off the W1 grid.
        grid_points: W1 grid points per site.

    Returns:
        Pooled medians and the quadrature measure's diagnostics.
    """
    rows, Cs, Ccs = [], [], []
    for a in anchors:
        ys, w = node_sets(scheme, base, int(a), rng, u_min)
        extra = [(yq, engine.probe(np.repeat(Q[[a]], len(yq), axis=0), yq, Q)) for yq in ys]
        row, C, Cc = B.coherence_scheme(engine, base, Q, int(a), np.zeros((len(ys), 1)), w, eps, u_tail=u_tail,
                                        grid_points=grid_points, extra=extra)
        rows.append(row), Cs.append(C), Ccs.append(Cc)
    every = [B.POOLED_ANCHORS] * len(anchors)                                      # one pool holding every anchor
    cell = B.coherence_cell(rows, np.asarray(Cs), anchors, every, trial_var, C_c=np.asarray(Ccs),
                            pool_names=(B.POOLED_ANCHORS,))[0]
    keep = ("cov_asymmetry", "cov_asymmetry_c", "drift_abs", "drift_abs_c", "total_var_ratio", "total_var_ratio_c")
    out = {k: cell.get(k) for k in keep}
    out["measure_var_ratio"] = float(np.median([r["measure_var_ratio"] for r in rows]))
    out["y_quad_max_z"] = float(np.median([r["y_quad_max_z"] for r in rows]))
    return out


def main() -> None:
    """Run the scheme comparison and write ``rqmc_diagnostic.csv`` (resumable per cell and scheme)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--engines", nargs="+", default=["gp_fixed"], choices=["gp_fixed", "gp_mll_refit", "tabpfn_v2_5"])
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=list(SCHEMES))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--ts", type=int, nargs="+", default=[40, 80])
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--channels-per-subject", type=int, default=1)
    ap.add_argument("--out", default="output/diagnostics/rqmc")
    args = ap.parse_args()

    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "rqmc_diagnostic.csv")
    done: set[tuple[Any, ...]] = set()
    key_cols = ["engine", "subject", "emg", "context_t", "draw", "scheme"]
    if os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        done = {tuple(r) for r in old[key_cols].itertuples(index=False, name=None)}

    channels = _channels(cfg)
    keep = _first_per_subject(channels, int(args.channels_per_subject))
    channels = [c for c in channels if c.label in keep]
    knob = build_knob(p["stress"]["knob"], [args.level])
    qd, yg = p["quadrature"], p["y_grid"]
    t0 = time.time()
    for ch in channels:
        st = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
        for t in args.ts:
            for draw in args.draws:
                ctx = context_for(st, t, draw, base_seed=cfg.seed, level=float(args.level))
                rng = rng_for(ch.label, "bayes_update", args.level, t, draw, base_seed=cfg.seed)
                anchors, pools, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]),
                                                             int(p["anchors"]["n_observed"]), rng, ctx.sites)
                tv = np.nanvar(st.Y_trials[anchors], axis=1, ddof=1)
                for name in args.engines:
                    todo = [s for s in args.schemes if (name, ch.subject, ch.emg, t, draw, s) not in done]
                    if not todo:
                        continue
                    eng = _make_engine(name, p, args.device, draw)
                    eng.fit(ctx.X, ctx.y)
                    base = eng.base(st.X_pool)
                    for scheme in todo:
                        ts = time.time()
                        qrng = rng_for(ch.label, "rqmc_diagnostic", scheme, t, draw, base_seed=cfg.seed)
                        res = scheme_readouts(eng, base, st.X_pool, anchors, list(pools), tv, scheme, float(p["eps"]),
                                              qrng, float(qd["u_min"]), float(yg["u_tail"]), int(yg["points"]))
                        row = {"engine": name, "subject": ch.subject, "emg": ch.emg, "context_t": t, "draw": draw,
                               "scheme": scheme, "seconds": time.time() - ts, **res}
                        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
                        print(f"[rqmc] {name} {ch.label} t={t} d={draw} {scheme}: asym {res['cov_asymmetry']:.3f} "
                              f"asym_c {res['cov_asymmetry_c']} mvr {res['measure_var_ratio']:.2f} "
                              f"({time.time() - ts:.0f}s)", flush=True)
    print(f"[rqmc] done ({time.time() - t0:.0f}s) -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
