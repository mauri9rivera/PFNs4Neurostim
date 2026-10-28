"""Exploratory probes of TabPFN-2.5's implied-covariance asymmetry (user request 2026-10-25).

Three questions, each on OMEGA's own cells (same context, same 12 anchors, same RQMC 8 x 8 quadrature stream as
``bayes_update``, so the ``rep1`` arm reproduces the cached rows):

* **Quadrature noise.** ``rep2`` re-estimates every row with an independent quadrature stream. Under pure noise the
  off-diagonal asymmetry of one estimate is ~ the replicate disagreement ``||off(C1 - C2)|| / (sqrt 2 ||off C||)``,
  and the noise-corrected asymmetry is ``sqrt(max(2 a_avg^2 - a_single^2, 0))`` (the two-replicate mean halves the
  noise variance).
* **Duplicate rows.** ``fp_off`` refits with TabPFN's fingerprint feature off (``FINGERPRINT_FEATURE=False``): a
  revisit then enters as an exact duplicate of its earlier row instead of a row with a salted hash.
* **Out of distribution.** ``synthetic`` channels are draws of an RBF GP on a 10 x 10 grid (in the family TabPFN is
  pretrained on) with one trial per context site; same anchors rule, same quadrature.

Each (cell, arm) is saved under ``output/diagnostics/omega_asymmetry/probes/``; ``--lane i/n`` splits the work across
processes and ``--report`` summarises whatever exists.

Usage:
    python scripts/omega_asymmetry_probes.py --lane 0/4
    python scripts/omega_asymmetry_probes.py --report
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Any

import numpy as np
import pandas as pd

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.channels import ChannelData, iter_channels
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.data.synthetic_neurostim import gp_channel
from pfns4neurostim.seeding import rng_for

OUT = os.path.join("output", "diagnostics", "omega_asymmetry", "probes")
SEED = 42
LEVEL = 1.0
CONTEXT_TS = (30, 80)
N_UNOBSERVED, N_OBSERVED = 8, 4
QUAD = {"method": "rqmc", "K": 8, "u_min": 1.0e-12, "central_mass": 1.0, "n_scrambles": 8}
EPS, U_TAIL, GRID_POINTS = 0.01, 1.0e-4, 512
#: Synthetic GP channels: (lengthscale, noise SD) per seed; outputscale 1, 8 trials per site.
SYNTHETIC = {0: (0.2, 0.3), 1: (0.3, 0.3), 2: (0.2, 0.5), 3: (0.3, 0.5)}
SYNTH_GRID, SYNTH_TRIALS = 10, 8
#: Arms per data source.
ARMS_REAL = ("rep1", "rep2", "fp_off")
ARMS_SYNTH = ("rep1", "rep2")


def real_channels() -> list[ChannelData]:
    """The first channel of every NHP subject, at stress level 1 exactly as OMEGA builds it."""
    knob = build_knob("k2_channel", [LEVEL])
    first: dict[int, ChannelData] = {}
    for ch in iter_channels("nhp", [0, 1, 2, 3], None, data_root="./data", normalization="pfn"):
        first.setdefault(ch.subject, ch)
    return [knob.apply(ch, LEVEL, rng_for(ch.label, "bayes_update", LEVEL, base_seed=SEED)) for ch in first.values()]


def synthetic_channel(seed: int) -> ChannelData:
    """A GP draw with the hyperparameters of :data:`SYNTHETIC`."""
    ell, noise = SYNTHETIC[seed]
    ch = gp_channel(SYNTH_GRID, ell, noise, SYNTH_TRIALS, np.random.default_rng(1000 + seed))
    return dataclasses.replace(ch, emg=seed)


def engine(arm: str) -> Any:
    """TabPFN-2.5 public-API refit engine (T 0.9, inference seed 0), fingerprint off for ``fp_off``."""
    eng = B.PFNRefitEngine(device="cuda", random_state=0, softmax_temperature=0.9)
    if arm == "fp_off":
        from tabpfn import TabPFNRegressor  # noqa: PLC0415

        eng.reg = TabPFNRegressor(n_estimators=1, softmax_temperature=0.9, random_state=0,
                                  ignore_pretraining_limits=True, device="cuda",
                                  inference_config={"FINGERPRINT_FEATURE": False})
    return eng


def run_cell(st: ChannelData, t: int, arm: str, path: str) -> None:
    """Part A rows at OMEGA's 12 anchors of one (channel, t, draw 0) for one arm; saved as npz."""
    ctx = context_for(st, t, 0, base_seed=SEED, level=LEVEL)
    rng = rng_for(st.label, "bayes_update", LEVEL, t, 0, base_seed=SEED)
    anchors, pools, _ = B.select_anchors_uniform(st, N_UNOBSERVED, N_OBSERVED, rng, ctx.sites)
    eng = engine(arm)
    eng.fit(ctx.X, ctx.y)
    Q = st.X_pool
    base = eng.base(Q)
    Cs, Ccs = [], []
    for a in anchors:
        key = ("bayes_update_quadrature",) if arm != "rep2" else ("omega_asymmetry_rep2",)
        u, w = B.quadrature_levels(QUAD["method"], QUAD["K"], u_min=QUAD["u_min"], central_mass=QUAD["central_mass"],
                                   n_scrambles=QUAD["n_scrambles"],
                                   rng=rng_for(st.label, *key, LEVEL, t, 0, int(a), base_seed=SEED))
        _, C, Cc = B.coherence_scheme(eng, base, Q, int(a), u, w, EPS, u_tail=U_TAIL, grid_points=GRID_POINTS)
        Cs.append(C)
        Ccs.append(Cc)
    np.savez(path, anchors=anchors, pools=np.array(pools), C=np.array(Cs), C_c=np.array(Ccs),
             s2=base.var[0][anchors], ch2xy=np.asarray(st.ch2xy), observed=np.isin(np.arange(st.n_sites), ctx.sites))


def tasks() -> list[tuple[str, ChannelData | int, int, str]]:
    """Every (source, channel, t, arm), real first."""
    out: list[tuple[str, Any, int, str]] = []
    for ch in real_channels():
        out += [("real", ch, t, arm) for t in CONTEXT_TS for arm in ARMS_REAL]
    for seed in SYNTHETIC:
        out += [("synthetic", seed, t, arm) for t in CONTEXT_TS for arm in ARMS_SYNTH]
    return out


def task_path(source: str, ch: Any, t: int, arm: str) -> str:
    """Output file of one task."""
    label = ch.label if source == "real" else f"synthetic-{ch}"
    return os.path.join(OUT, f"{source}__{label}__t{t}__{arm}.npz")


def off_asym(C: np.ndarray) -> float:
    """Antisymmetric share of the off-diagonal pairs (see ``omega_asymmetry_diagnosis.off_asymmetry``)."""
    off = ~np.eye(C.shape[0], dtype=bool)
    return float(np.linalg.norm((C - C.T)[off]) / (np.sqrt(2.0) * np.linalg.norm(C[off])))


def block_asym(C: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> float:
    """Antisymmetric share of the off-diagonal pairs between two anchor sets."""
    i, j = np.meshgrid(rows, cols, indexing="ij")
    keep = i != j
    a, b = C[i[keep], j[keep]], C[j[keep], i[keep]]
    return float(np.sqrt(((a - b) ** 2).sum()) / np.sqrt((a ** 2).sum() + (b ** 2).sum()))


def report() -> None:
    """Per-cell readouts and medians by (source, t, arm)."""
    recs = []
    for name in sorted(os.listdir(OUT)):
        if not name.endswith(".npz"):
            continue
        source, label, t, arm = name[:-4].split("__")
        z = np.load(os.path.join(OUT, name))
        anchors = z["anchors"]
        for kind in ("C", "C_c"):
            C = z[kind][:, anchors]                                                   # [A, A]
            new, rev = np.flatnonzero(z["pools"] == "unobserved"), np.flatnonzero(z["pools"] == "observed")
            recs.append({"source": source, "label": label, "t": int(t[1:]), "arm": arm, "matrix": kind,
                         "asym_off": off_asym(C), "new_new": block_asym(C, new, new),
                         "rev_rev": block_asym(C, rev, rev), "new_rev": block_asym(C, new, rev)})
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "..", "probe_cells.csv"), index=False)
    # noise: replicate disagreement and the noise-corrected asymmetry, per cell (moment-matched rows)
    noise = []
    for (source, label, t), g in df[df["matrix"] == "C_c"].groupby(["source", "label", "t"]):
        p1, p2 = (os.path.join(OUT, f"{source}__{label}__t{t}__{a}.npz") for a in ("rep1", "rep2"))
        if not (os.path.exists(p1) and os.path.exists(p2)):
            continue
        z1, z2 = np.load(p1), np.load(p2)
        a = z1["anchors"]
        C1, C2 = z1["C_c"][:, a], z2["C_c"][:, a]
        off = ~np.eye(len(a), dtype=bool)
        avg = 0.5 * (C1 + C2)
        single = 0.5 * (off_asym(C1) + off_asym(C2))
        disagree = float(np.linalg.norm((C1 - C2)[off]) / (np.sqrt(2.0) * np.linalg.norm(avg[off])))
        noise.append({"source": source, "label": label, "t": t, "asym_single": single, "asym_avg2": off_asym(avg),
                      "replicate_disagreement": disagree,
                      "asym_noise_corrected": float(np.sqrt(max(2 * off_asym(avg) ** 2 - single ** 2, 0.0)))})
    nz = pd.DataFrame(noise)
    nz.to_csv(os.path.join(OUT, "..", "probe_noise.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["source", "t", "arm", "matrix"])[["asym_off", "new_new", "rev_rev", "new_rev"]].median()
          .round(3).to_string())
    if not nz.empty:
        print(nz.groupby(["source", "t"])[["asym_single", "asym_avg2", "replicate_disagreement",
                                           "asym_noise_corrected"]].median().round(3).to_string())


def main() -> None:
    """Run one lane of the probes, or report."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lane", default="0/1", help="i/n: this process runs tasks i, i+n, ...")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.report:
        report()
        return
    i, n = (int(v) for v in args.lane.split("/"))
    for k, (source, ch, t, arm) in enumerate(tasks()):
        path = task_path(source, ch, t, arm)
        if k % n != i or os.path.exists(path):
            continue
        st = ch if source == "real" else synthetic_channel(ch)
        print(f"[probe] {os.path.basename(path)}", flush=True)
        run_cell(st, t, arm, path)


if __name__ == "__main__":
    main()
