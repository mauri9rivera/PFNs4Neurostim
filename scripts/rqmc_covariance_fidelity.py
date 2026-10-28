"""How well do RQMC and the measures A / B / C reconstruct an implied covariance? (exploratory; user 2026-10-28)

Part 1, ground truth. GP-fixed's refit is affine in y', so its implied covariance ``C(x', .) = Cov_{y'}(y', mu_new(.))`` equals the latent
posterior covariance ``Sigma(x', .)`` times the node set's variance ratio ``v / s^2`` -- whatever the node set. What a node set can
therefore break is the *scale* of each anchor's row, not its shape. To give GP-fixed the tail content of TabPFN, every scheme's nodes are
built on TabPFN's predictive at the same anchor, standardised with the measure's own mean and SD, and re-mapped to GP-fixed's anchor
predictive (``y' = mu_GP + s_GP z``); the exactness control is Gaussian strata. Per scheme, raw and moment-matched ``C`` ([anchors, sites])
are compared with ``Sigma`` by entrywise Pearson, linear CKA (sites as items, anchors as features), the median per-anchor Pearson, the
spread of the per-anchor scale ``C_a / Sigma_a`` and the asymmetry of the anchor-anchor block.

Part 2, TabPFN (no ground truth). The implied covariance of TabPFN's refit under RQMC 8x8 (two independent scramble sets), A, B and C
(K = 32 strata), compared across schemes with the same similarity measures: how much of the *structure* depends on the tail convention.

    python scripts/rqmc_covariance_fidelity.py --part 1
    python scripts/rqmc_covariance_fidelity.py --part 2 --subjects 0
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from typing import Any, Callable

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
from rqmc_diagnostic import node_sets  # noqa: E402
from rqmc_measure_study import MeasureCriterionPredictive, gauss_nodes, measure_nodes  # noqa: E402

#: Strata of the deterministic schemes.
K_STRATA: int = 32
#: Part 1 schemes (shape source: TabPFN's predictive); ``gauss`` is the exactness control.
PART1_SCHEMES: tuple[str, ...] = ("gauss", "rqmc_8x8", "rqmc_8x32", "A", "B", "C")
#: Part 2 schemes.
PART2_SCHEMES: tuple[str, ...] = ("rqmc_8x8", "rqmc_8x8_b", "A", "B", "C")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two arrays flattened (nan if either is constant)."""
    a, b = np.ravel(a), np.ravel(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA of two representations of the same items.

    Args:
        X: Features of the items, shape [N, P].
        Y: Features of the items, shape [N, Q].

    Returns:
        ``||X'Y||_F^2 / (||X'X||_F ||Y'Y||_F)`` on column-centred inputs.
    """
    X, Y = X - X.mean(0), Y - Y.mean(0)
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(np.linalg.norm(X.T @ Y) ** 2 / den) if den > 0 else float("nan")


def similarity(C: np.ndarray, T: np.ndarray, anchors: np.ndarray) -> dict[str, float]:
    """Similarity of an implied covariance matrix to a reference.

    Args:
        C: Implied covariance rows at the anchors, shape [A, M].
        T: Reference rows, shape [A, M].
        anchors: Anchor site indices, shape [A].

    Returns:
        ``pearson`` (entrywise), ``cka`` (sites as items), ``row_pearson`` (median over anchors), ``scale_cv`` (spread of the
        per-anchor scale ``C_a . T_a / T_a . T_a``), ``frob_err`` (relative error after the best global scalar) and ``asym``
        (median relative asymmetry of the anchor-anchor block of ``C``).
    """
    scale = (C * T).sum(1) / (T * T).sum(1)                                         # [A] per-anchor scale
    alpha = float((C * T).sum() / (T * T).sum())
    block = C[:, anchors]                                                           # [A, A]
    iu = np.triu_indices(len(anchors), 1)
    asym = np.abs(block[iu] - block.T[iu]) / (np.abs(block[iu]) + np.abs(block.T[iu]) + 1e-300)
    return {"pearson": pearson(C, T), "cka": linear_cka(C.T, T.T),
            "row_pearson": float(np.nanmedian([pearson(C[i], T[i]) for i in range(len(C))])),
            "scale_cv": float(scale.std() / abs(scale.mean())) if scale.mean() != 0 else float("nan"),
            "frob_err": float(np.linalg.norm(C - alpha * T) / np.linalg.norm(T)), "asym": float(np.median(asym))}


def implied(engine: Any, base: B.Predictive, Q: np.ndarray, anchors: np.ndarray, eps: float, u_tail: float, grid_points: int,
            nodes_of: Callable[[int], tuple[list[np.ndarray], np.ndarray]], convert: Callable[[Any], Any] = lambda x: x,
            ) -> tuple[np.ndarray, np.ndarray]:
    """Raw and moment-matched implied covariance rows of an engine under one node rule.

    Args:
        engine: Fitted engine.
        base: Its predictive given D.
        Q: Readout sites, shape [M, D].
        anchors: Anchor sites, shape [A].
        eps: Influenced-site threshold.
        u_tail: Tail probability left off the W1 grid.
        grid_points: W1 grid points per site.
        nodes_of: ``nodes_of(anchor) -> (list of y' node arrays, weights)``; several arrays are scramble replicates.
        convert: Applied to the base and probe predictives (the measure's stated moments).

    Returns:
        ``(C [A, M], C_c [A, M])``.
    """
    base_m = convert(base)
    Cs, Ccs = [], []
    for a in anchors:
        ys, w = nodes_of(int(a))
        extra = [(y, convert(engine.probe(np.repeat(Q[[a]], len(y), axis=0), y, Q))) for y in ys]
        _, C, Cc = B.coherence_scheme(engine, base_m, Q, int(a), np.zeros((len(ys), 1)), w, eps, u_tail=u_tail,
                                      grid_points=grid_points, extra=extra)
        Cs.append(C), Ccs.append(Cc)
    return np.asarray(Cs), np.asarray(Ccs)


def tabpfn_nodes(scheme: str, tb: B.CriterionPredictive, a: int, rng: np.random.Generator, u_min: float,
                 ) -> tuple[list[np.ndarray], np.ndarray, float, float]:
    """TabPFN-built nodes of a scheme at an anchor, with the stated mean and SD of the scheme's own measure.

    Returns:
        ``(node arrays, weights, mean, sd)``.
    """
    if scheme.startswith("rqmc"):
        name = "rqmc_8x8" if scheme.startswith("rqmc_8x8") else scheme
        ys, w = node_sets(name, tb, a, rng, u_min)
        return ys, w, float(tb.mean[0][a]), float(np.sqrt(tb.var[0][a]))
    conv = {"A": tb, "B": MeasureCriterionPredictive(tb.parts, "linear"), "C": MeasureCriterionPredictive(tb.parts, "central")}[scheme]
    y, w = measure_nodes(tb, a, scheme, K_STRATA)
    return [y], w, float(conv.mean[0][a]), float(np.sqrt(conv.var[0][a]))


def main() -> None:
    """Run the requested part and write ``covariance_fidelity_part<N>.csv``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--part", type=int, choices=(1, 2), default=1)
    ap.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--ts", type=int, nargs="+", default=[40, 80])
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="output/diagnostics/rqmc")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f"covariance_fidelity_part{args.part}_{'-'.join(map(str, args.subjects))}.csv")
    channels = [c for c in _channels(cfg) if c.label in _first_per_subject(_channels(cfg), 1) and c.subject in args.subjects]
    knob = build_knob(p["stress"]["knob"], [args.level])
    eps, u_tail, gp_pts, u_min = float(p["eps"]), float(p["y_grid"]["u_tail"]), int(p["y_grid"]["points"]), float(p["quadrature"]["u_min"])
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for ch in channels:
        st = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
        Q = st.X_pool
        for t in args.ts:
            ctx = context_for(st, t, args.draw, base_seed=cfg.seed, level=float(args.level))
            rng = rng_for(ch.label, "bayes_update", args.level, t, args.draw, base_seed=cfg.seed)
            anchors, _, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]), int(p["anchors"]["n_observed"]), rng, ctx.sites)
            tp = _make_engine("tabpfn_v2_5", p, "cpu", args.draw)
            tp.fit(ctx.X, ctx.y)
            tb = tp.base(Q)
            qrng = rng_for(ch.label, "cov_fidelity", t, base_seed=cfg.seed)
            if args.part == 1:
                gp = _make_engine("gp_fixed", p, "cpu", args.draw)
                gp.fit(ctx.X, ctx.y)
                gb = gp.base(Q)
                T = gp.posterior_covariance(Q)[anchors]                             # [A, M] latent posterior covariance rows
                for scheme in PART1_SCHEMES:
                    cache: dict[int, Any] = {}

                    def nodes_of(a: int, scheme: str = scheme) -> tuple[list[np.ndarray], np.ndarray]:
                        mu_g, s_g = float(gb.mean[0][a]), float(np.sqrt(gb.var[0][a]))
                        if scheme == "gauss":
                            y, w = gauss_nodes(mu_g, s_g, K_STRATA)
                            return [y], w
                        ys, w, mu_t, s_t = tabpfn_nodes(scheme, tb, a, qrng, u_min)
                        return [mu_g + s_g * (y - mu_t) / s_t for y in ys], w

                    C, Cc = implied(gp, gb, Q, anchors, eps, u_tail, gp_pts, nodes_of)
                    for kind, M in (("raw", C), ("moment_matched", Cc)):
                        rows.append({"engine": "gp_fixed", "subject": ch.subject, "emg": ch.emg, "context_t": t, "scheme": scheme,
                                     "kind": kind, **similarity(M, T, anchors)})
                    print(f"[fidelity1] {ch.label} t={t} {scheme} ({time.time() - t0:.0f}s)", flush=True)
            else:
                mats: dict[tuple[str, str], np.ndarray] = {}
                for scheme in PART2_SCHEMES:
                    conv = {"A": lambda x: x, "B": lambda x: MeasureCriterionPredictive(x.parts, "linear"),
                            "C": lambda x: MeasureCriterionPredictive(x.parts, "central")}.get(scheme, lambda x: x)

                    def nodes_of(a: int, scheme: str = scheme) -> tuple[list[np.ndarray], np.ndarray]:
                        ys, w, _, _ = tabpfn_nodes(scheme, tb, a, qrng, u_min)
                        return ys, w

                    C, Cc = implied(tp, tb, Q, anchors, eps, u_tail, gp_pts, nodes_of, conv)
                    mats[(scheme, "raw")], mats[(scheme, "moment_matched")] = C, Cc
                    print(f"[fidelity2] {ch.label} t={t} {scheme} ({time.time() - t0:.0f}s)", flush=True)
                for kind in ("raw", "moment_matched"):
                    for s1, s2 in itertools.combinations(PART2_SCHEMES, 2):
                        rows.append({"engine": "tabpfn_v2_5", "subject": ch.subject, "emg": ch.emg, "context_t": t, "scheme": f"{s1} vs {s2}",
                                     "kind": kind, **similarity(mats[(s1, kind)], mats[(s2, kind)], anchors)})
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.groupby(["kind", "scheme"])[["pearson", "cka", "row_pearson", "scale_cv", "frob_err", "asym"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
