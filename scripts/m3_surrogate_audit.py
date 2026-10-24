"""Exploratory M3 audit (user, 2026-10-23): does a smooth random field land where neurostim channels land in CKA (b)?

CKA (b) places a channel's feature-token-mean embedding between an on-grid prior bank (p_rank -> 0) and a noise bank
(p_rank -> 1). The feature tokens are dominated by electrode geometry (CKA 0.60-0.84 to K_X at every block), so the
placement could reflect only that the channel is a *smooth* map on the grid. This script places smooth-field surrogates
(independent unit-variance RBF-GP fields that know nothing about the data, ``analysis.cka.smooth_field_surrogate``) through
the identical pipeline -- same cached banks, same context sites, same embedding readout, same rank placement -- and
compares them with the channels' rows in ``cka_placement.csv``. If the surrogates land where the channels land, the
placement measures smoothness, not neurostim content.

Self-contained: it reuses the runner's helpers read-only and writes only under ``--out``. Resumable: one CSV per
(grid, t, draw) is written as soon as it is done and skipped on a re-run.

    python scripts/m3_surrogate_audit.py --config configs/experiment/mechanism_cka_nhp.yaml
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

from pfns4neurostim.analysis import placement as P
from pfns4neurostim.analysis.cka import cka_debiased, linear_gram, smooth_field_surrogate
from pfns4neurostim.analysis.context import context_sites, grid_id
from pfns4neurostim.analysis.embeddings import readout_embeddings
from pfns4neurostim.experiments.mechanism import (
    CKA_DEFAULTS, CKA_PLACEMENT_READOUT, RunOptions, _Cells, _channels, _check_keys, _grid_banks,
    load_mechanism_config,
)
from pfns4neurostim.models.pfn.tabpfn import FrozenTabPFN
from pfns4neurostim.seeding import rng_for, set_seed
from pfns4neurostim.visualization import style

#: Seed-stream name of the surrogate fields (distinct from every pipeline stream).
SURROGATE_STREAM: str = "m3_surrogate_audit"


def _cell(cfg: Any, pl: dict[str, Any], cells: _Cells, ref: Any, t_ctx: int, draw: int, engine: Any,
          lengthscales: list[float], n_fields: int, jitter: float, layers: Any) -> pd.DataFrame:
    """Rank placement of every surrogate field at one (grid, t, draw), as in ``run_cka``'s ``b`` cell.

    Args:
        cfg: Resolved mechanism config.
        pl: Resolved ``cka.placement`` block.
        cells: Cell namespace (serves the cached prior / noise banks).
        ref: Any channel on the grid.
        t_ctx: Context size.
        draw: Context draw.
        engine: Frozen TabPFN engine.
        lengthscales: Surrogate RBF lengthscales (normalised coordinates).
        n_fields: Surrogate fields per lengthscale.
        jitter: Cholesky jitter of the surrogate sampler.
        layers: ``cka.layers`` (``None`` = every block).

    Returns:
        One row per (lengthscale, field, layer).
    """
    q = float(pl["bank_quantile"])
    bank_keys = {key: pl[key] for key in ("n_dense", "holdout_frac", "rmse_threshold", "prior_type")}
    pair = _grid_banks(cells, ref, int(pl["n_prior"]) + int(pl["n_prior_holdout"]), int(pl["n_noise"]), bank_keys)
    if pair is None:
        raise RuntimeError("prior / noise bank unavailable.")
    prior, noise = pair
    n_hold = min(int(pl["n_prior_holdout"]), len(prior.maps) // 2)
    sites = context_sites(ref.X_pool, t_ctx, draw, base_seed=cfg.seed)                    # [t]
    layer_ids: list[tuple[int, ...]] = []

    def grams(y_map: np.ndarray) -> list[np.ndarray]:
        engine.fit(ref.X_pool[sites], y_map[sites])
        emb = readout_embeddings(engine, ref.X_pool, readouts=(CKA_PLACEMENT_READOUT,), layers=layers)
        emb = emb[CKA_PLACEMENT_READOUT]
        if not layer_ids:
            layer_ids.append(emb.layers)
        return [linear_gram(a) for a in emb.arrays]

    bank = [grams(m) for m in prior.maps[n_hold:]]
    hold = [grams(m) for m in prior.maps[:n_hold]]
    nz = [grams(m) for m in noise.maps]

    def dist(Gs: list[np.ndarray], layer: int) -> float:
        return float(np.quantile([1.0 - cka_debiased(Gs[layer], B[layer]) for B in bank], q))

    n_lay = len(bank[0])
    floor_refs = [np.array([dist(h, s) for h in hold]) for s in range(n_lay)]
    ceil_refs = [np.array([dist(z, s) for z in nz]) for s in range(n_lay)]
    gid = grid_id(ref.X_pool)
    rows = []
    for ell in lengthscales:
        F = smooth_field_surrogate(ref.X_pool, float(ell), int(n_fields),
                                   rng_for(gid, SURROGATE_STREAM, float(ell), int(t_ctx), int(draw), base_seed=cfg.seed),
                                   jitter=float(jitter))                                   # [N, n_fields]
        for k in range(F.shape[1]):
            f = (F[:, k] - F[:, k].mean()) / F[:, k].std()                                 # z-scored like a channel
            G = grams(f)
            for s in range(n_lay):
                d = dist(G, s)
                rows.append({"grid_id": gid, "context_t": int(t_ctx), "draw": int(draw), "lengthscale": float(ell),
                             "field": k, "layer": int(layer_ids[0][s]), "d": d,
                             **P.rank_placement(d, floor_refs[s], ceil_refs[s])})
    return pd.DataFrame(rows)


def _figure(sur: pd.DataFrame, chan: pd.DataFrame, out: str) -> list[str]:
    """p_rank vs layer per context size: the channels' median beside each surrogate lengthscale's median."""
    ts = sorted(set(sur["context_t"]) & set(chan["context_t"]))
    fig, axes = style.figure("double", nrows=1, ncols=len(ts), squeeze=False, layout=style.LAYOUT_ENGINE)
    ells = sorted(sur["lengthscale"].unique())
    for ax, t in zip(axes[0], ts):
        c = chan[chan["context_t"] == t].groupby("layer")["p_rank"].median()
        ax.plot(c.index, c.values, color=style.ANCHOR_PFN, lw=style.LINE_WIDTH, label="Neurostim channels")
        for i, ell in enumerate(ells):
            s = sur[(sur["context_t"] == t) & (sur["lengthscale"] == ell)].groupby("layer")["p_rank"].median()
            ax.plot(s.index, s.values, color=style.derive_shade(style.NEUTRAL_GREY, i, len(ells)), ls="--",
                    lw=style.LINE_WIDTH, label=f"Smooth field, $\\ell$ = {ell:g}")
        ax.set_title(f"t = {t}", fontsize=style.FONT_SIZES["legend"])
        ax.set_xlabel(style.axis_label("layer"))
        ax.set_ylim(0.0, 1.0)
    axes[0][0].set_ylabel(style.axis_label("p_rank"))
    axes[0][0].legend(loc="best", fontsize=style.FONT_SIZES["legend"] - 1)
    return style.save_figure(fig, out, "m3_surrogate_audit")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="The CKA (b) mechanism config.")
    ap.add_argument("--lengthscales", type=float, nargs="+", default=[0.10, 0.15, 0.30],
                    help="Surrogate RBF lengthscales (the M2 surrogate's, BETA D3d).")
    ap.add_argument("--n-fields", type=int, default=10, help="Surrogate fields per lengthscale and cell.")
    ap.add_argument("--jitter", type=float, default=1.0e-8, help="Cholesky jitter of the surrogate sampler.")
    ap.add_argument("--channel-placement", default="output/mechanism/cka/nhp/nhp/cka_placement.csv",
                    help="The channels' placement rows to compare with.")
    ap.add_argument("--out", default="output/diagnostics/m3_surrogate_audit", help="Output directory.")
    args = ap.parse_args(argv)

    cfg = load_mechanism_config(args.config)
    p = _check_keys(cfg.params, CKA_DEFAULTS, "cka")
    pl = _check_keys(p["placement"], CKA_DEFAULTS["placement"], "cka.placement")
    set_seed(cfg.seed)
    cells = _Cells(cfg, RunOptions())
    engine = FrozenTabPFN(device=cfg.device, random_state=int(p["inference_seed"]))
    os.makedirs(os.path.join(args.out, "cells"), exist_ok=True)
    grids: dict[bytes, Any] = {}
    for ch in _channels(cfg):
        grids.setdefault(ch.X_pool.tobytes(), ch)
    parts = []
    for ref in grids.values():
        for t_ctx, draw in itertools.product([t for t in cfg.context_sizes if t < ref.n_sites], range(int(pl["n_draws"]))):
            path = os.path.join(args.out, "cells", f"{grid_id(ref.X_pool)}_t{t_ctx}_d{draw}.csv")
            if not os.path.exists(path):
                _cell(cfg, pl, cells, ref, int(t_ctx), int(draw), engine, args.lengthscales, args.n_fields,
                      args.jitter, p["layers"]).to_csv(path, index=False)
                print(f"[m3-audit] grid {grid_id(ref.X_pool)} t={t_ctx} draw={draw} done", flush=True)
            parts.append(pd.read_csv(path))
    sur = pd.concat(parts, ignore_index=True)
    sur.to_csv(os.path.join(args.out, "surrogate_placement.csv"), index=False)
    chan = pd.read_csv(args.channel_placement).dropna(subset=["p_rank"])
    summary = pd.concat([
        chan.groupby(["context_t", "layer"])["p_rank"].median().rename("channels"),
        sur.groupby(["context_t", "layer", "lengthscale"])["p_rank"].median().unstack("lengthscale")
           .add_prefix("surrogate_l"),
    ], axis=1).reset_index()
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    for path in _figure(sur, chan, args.out):
        print(f"[m3-audit] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
