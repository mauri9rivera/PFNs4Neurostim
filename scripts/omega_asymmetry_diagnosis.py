"""Exploratory: where does TabPFN-2.5's implied-covariance asymmetry come from? (user request 2026-10-25)

Reads the cached OMEGA grid cells of ``bayes_update`` (no model is run): each cell holds every Part A engine's raw
implied-covariance rows ``C(x'_i, .)`` at its 12 anchors (8 unobserved, 4 observed) and, per anchor, the predictive
variance ``s^2(x'_i) = implied_noise + C(x'_i, x'_i)``. On the anchor block ``C_A`` [12, 12] it computes, per cell and
engine:

* ``asym``          ||C_A - C_A'||_F / ||C_A||_F (OMEGA's raw ``cov_asymmetry`` over every anchor);
* ``asym_off``      the antisymmetric share of the off-diagonal pairs only (in [0, 1]);
* ``asym_off_scaled`` the same after the best positive per-anchor rescaling ``diag(d) C_A``. ``C_ij = s^2_i b_ij`` with
                    ``b_ij`` the slope of the mean at x_j per unit y' at x_i; if ``asym_scaled`` << ``asym`` the update
                    slopes come from a symmetric kernel and the asymmetry is a per-site mismatch between how variable
                    y' is at x_i and how strongly it moves the other sites (a calibration of the marginal variance
                    against the update), not an inconsistent update pattern;
* ``asym_<block>``  the antisymmetric share inside new-new, revisit-revisit and new-revisit anchor pairs;
* ``global_share``  share of the off-diagonal norm carried by each row's far-field level ``g_i`` (median of
                    ``C(x'_i, x)`` over sites >= 5 pitches away): a site-independent shift of the whole map per unit y';
                    ``asym_off_local`` is the off-diagonal asymmetry of ``C_A - g 1'``;
* ``rel_se_latent``  median RQMC relative SE of ``C(x', x')`` (the quadrature-noise level of each engine);
* ``sign_flip``     share of pairs whose C_ij and C_ji have opposite signs while both exceed 5 % of the block's
                    largest off-diagonal entry (an asymmetry no rescaling can remove);
* per-pair ``|C_ij - C_ji| / (|C_ij| + |C_ji|)`` by grid distance (pitches).

GP-fixed is the exact-Bayes floor of the same pipeline (its asymmetry is pure quadrature noise, RQMC 8 x 8); GP-MLL
refit is the empirical-Bayes yardstick (5 restarts). Outputs ``output/diagnostics/omega_asymmetry/``.

Usage:
    python scripts/omega_asymmetry_diagnosis.py
    python scripts/omega_asymmetry_diagnosis.py --levels 1.0 --min-t 30
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pfns4neurostim.analysis.bayes_update import site_distances
from pfns4neurostim.data.channels import iter_channels

#: Engines read from the cells, in report order.
ENGINES: tuple[str, ...] = ("tabpfn_v2_5", "gp_mll_refit", "gp_fixed")
#: A pair counts as a sign flip only if both entries exceed this share of the block's largest off-diagonal |C|.
SIGN_FLIP_MIN_SHARE: float = 0.05
#: Sites at least this far (grid pitches) from an anchor define its row's far-field (global) level.
GLOBAL_MIN_PITCH: float = 5.0
#: Distance bins (grid pitches) of the per-pair antisymmetry profile.
DISTANCE_BINS: tuple[float, ...] = (0.0, 1.5, 3.0, 5.0, 8.0, 14.0)


def asymmetry(C: np.ndarray) -> float:
    """``||C - C'||_F / ||C||_F`` of a square block."""
    return float(np.linalg.norm(C - C.T) / np.linalg.norm(C))


def off_asymmetry(C: np.ndarray) -> float:
    """Antisymmetric share of the off-diagonal pairs: ``||off(C - C')||_F / (sqrt(2) ||off(C)||_F)``, in [0, 1].

    The diagonal is left out so that a rescaling cannot hide the asymmetry behind one large diagonal entry.
    """
    off = ~np.eye(C.shape[0], dtype=bool)
    return float(np.linalg.norm((C - C.T)[off]) / (np.sqrt(2.0) * np.linalg.norm(C[off])))


def scaled_asymmetry(C: np.ndarray) -> tuple[float, np.ndarray]:
    """Smallest :func:`off_asymmetry` of ``diag(d) C`` over positive ``d`` (log-parametrised, geometric mean 1).

    Args:
        C: Square block, shape [A, A].

    Returns:
        ``(asymmetry, d)``.
    """
    def f(logd: np.ndarray) -> float:
        logd = logd - logd.mean()
        return off_asymmetry(np.exp(logd)[:, None] * C)

    res = minimize(f, np.zeros(C.shape[0]), method="L-BFGS-B")
    logd = res.x - res.x.mean()
    return float(res.fun), np.exp(logd)


def block_share(C: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float | None:
    """Antisymmetric share of the off-diagonal pairs (i in rows, j in cols, i != j): ||C_ij - C_ji|| / ||(C_ij, C_ji)||."""
    i, j = np.meshgrid(rows, cols, indexing="ij")
    keep = i != j
    a, b = C[i[keep], j[keep]], C[j[keep], i[keep]]
    den = np.sqrt((a ** 2).sum() + (b ** 2).sum())
    return None if den == 0 else float(np.sqrt(((a - b) ** 2).sum()) / den)


def load_cells(cell_dir: str, levels: set[float], min_t: int) -> list[dict[str, Any]]:
    """Grid cells of ``bayes_update`` holding every engine of :data:`ENGINES`, one per (channel, level, t, draw)."""
    out: dict[tuple, dict[str, Any]] = {}
    for path in glob.glob(os.path.join(cell_dir, "*.json")):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        ident, pay = doc.get("identity", {}), doc.get("payload", {})
        if ident.get("kind") != "cell" or "C" not in pay or float(ident.get("level", -1)) not in levels:
            continue
        if int(ident["t"]) < min_t or not all(e in pay["C"] for e in ENGINES):
            continue
        key = (ident["label"], float(ident["level"]), int(ident["t"]), int(ident["draw"]))
        out[key] = {"identity": ident, "payload": pay}
    return list(out.values())


def main() -> None:
    """Compute the per-cell decomposition and write the tables."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", default=os.path.join("output", "cells", "nhp", "mechanism_bayes_update"))
    ap.add_argument("--out", default=os.path.join("output", "diagnostics", "omega_asymmetry"))
    ap.add_argument("--levels", type=float, nargs="+", default=[1.0])
    ap.add_argument("--min-t", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cells = load_cells(args.cells, set(args.levels), args.min_t)
    print(f"{len(cells)} cells")
    coords = {ch.label: np.asarray(ch.ch2xy) for ch in iter_channels("nhp", [0, 1, 2, 3], None, data_root="./data",
                                                                       normalization="pfn")}
    rows, pairs = [], []
    for cell in cells:
        ident, pay = cell["identity"], cell["payload"]
        anchors = np.asarray(pay["anchors"], dtype=int)                              # [A]
        coh = pd.DataFrame(pay["coherence"])
        dist = site_distances(coords[ident["label"]], anchors)[:, anchors]           # [A, A] pitches
        for eng in ENGINES:
            Crow = np.asarray(pay["C"][eng], dtype=np.float64)                       # [A, M]
            C = Crow[:, anchors]                                                     # [A, A]
            e = coh[coh["engine"] == eng].set_index("anchor").loc[anchors]
            pool = e["pool"].to_numpy()
            new, rev = np.flatnonzero(pool == "unobserved"), np.flatnonzero(pool == "observed")
            s2 = e["implied_noise"].to_numpy() + np.diag(C)                          # [A] predictive variance
            sc, d = scaled_asymmetry(C)
            # global component: each row's far-field level (median over sites >= GLOBAL_MIN_PITCH from its anchor)
            far = site_distances(coords[ident["label"]], anchors) >= GLOBAL_MIN_PITCH       # [A, M]
            g = np.array([np.median(Crow[i, far[i]]) for i in range(len(anchors))])          # [A]
            R = C - g[:, None]                                                                # [A, A] local part
            off = ~np.eye(len(anchors), dtype=bool)
            noise = (e["latent_frac_se"] / e["latent_frac"].abs()).median() if "latent_frac_se" in e else np.nan
            big = SIGN_FLIP_MIN_SHARE * np.abs(C[off]).max()
            both = off & (np.abs(C) > big) & (np.abs(C.T) > big)
            flips = both & (np.sign(C) != np.sign(C.T))
            rows.append({
                "label": ident["label"], "subject": ident["subject"], "emg": ident["emg"], "level": ident["level"],
                "context_t": ident["t"], "draw": ident["draw"], "engine": eng,
                "asym": asymmetry(C), "asym_off": off_asymmetry(C), "asym_off_scaled": sc,
                "global_share": float(np.linalg.norm(np.broadcast_to(g[:, None], C.shape)[off]) / np.linalg.norm(C[off])),
                "asym_off_local": off_asymmetry(R),
                "global_cv": float(np.std(g * 1.0) / (np.abs(g).mean() + 1e-300)),
                "rel_se_latent": float(noise), "y_quad_max_z": float(e["y_quad_max_z"].median()),
                "asym_new_new": block_share(C, new, new), "asym_rev_rev": block_share(C, rev, rev),
                "asym_new_rev": block_share(C, new, rev),
                "sign_flip": float(flips.sum() / max(both.sum(), 1)),
                "log_d_rev_minus_new": float(np.log(d[rev]).mean() - np.log(d[new]).mean()) if rev.size and new.size
                else np.nan,
                "corr_logd_logs2": float(np.corrcoef(np.log(d), np.log(s2))[0, 1]),
            })
            iu = np.triu_indices(len(anchors), 1)
            a, b = C[iu], C.T[iu]
            pairs.append(pd.DataFrame({"engine": eng, "context_t": ident["t"], "dist": dist[iu],
                                       "anti": np.abs(a - b), "mag": np.abs(a) + np.abs(b),
                                       "kind": np.where(np.isin(iu[0], rev) & np.isin(iu[1], rev), "rev-rev",
                                                        np.where(np.isin(iu[0], new) & np.isin(iu[1], new),
                                                                 "new-new", "new-rev"))}))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "asymmetry_cells.csv"), index=False)
    cols = ["asym", "asym_off", "asym_off_scaled", "global_share", "asym_off_local", "global_cv", "rel_se_latent",
            "y_quad_max_z", "asym_new_new", "asym_rev_rev", "asym_new_rev", "sign_flip",
            "log_d_rev_minus_new", "corr_logd_logs2"]
    # draws, then channels: median within channel, then median over channels
    per_ch = df.groupby(["engine", "context_t", "label"])[cols].median().reset_index()
    summ = per_ch.groupby(["engine", "context_t"])[cols].median()
    summ.to_csv(os.path.join(args.out, "asymmetry_summary.csv"))
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string())
    pr = pd.concat(pairs, ignore_index=True)
    pr["dist_bin"] = pd.cut(pr["dist"], DISTANCE_BINS)
    prof = pr.groupby(["engine", "kind", "dist_bin"], observed=True).apply(
        lambda g: pd.Series({"anti_share": g["anti"].sum() / g["mag"].sum(), "n": len(g),
                             "mean_mag": g["mag"].mean() / 2}))
    prof.to_csv(os.path.join(args.out, "asymmetry_by_distance.csv"))
    print(prof.round(3).to_string())


if __name__ == "__main__":
    main()
