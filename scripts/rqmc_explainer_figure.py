"""Diagrams of the OMEGA quadrature on one real TabPFN predictive (exploratory; user 2026-10-27).

Explains, on a single cell (default ``nhp-s1-e0``, t = 40, level 1, draw 0, first unobserved anchor), what the quadrature
integrates over and why TabPFN's heavy-tailed bar distribution makes the raw readouts scheme-dependent:

    (a) the electrode grid: context sites, the anchor x' (where the imagined observation y' is made), the influenced sites
    (b) the predictive of y' at the anchor, as the bar distribution (and the Gaussian with the same mean and SD)
    (c) its CDF and the quantile map u -> y': the Sobol levels of every scramble become the nodes y'_k
    (d) the two outer buckets: ``mean`` / ``variance`` treat them as half-normal tails, ``icdf`` as uniform on a finite bucket
    (e) variance accounting: the stated variance split into central buckets and tails, against the variance of the nodes
    (f) what each node does: the refit's change of the posterior mean at the most influenced site, against y'

and, from ``rqmc_diagnostic.csv``, the pooled covariance asymmetry of every scheme, raw and moment-matched, per channel.

    python scripts/rqmc_explainer_figure.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import _channels, _make_engine, _validate_bayes_update, load_mechanism_config
from pfns4neurostim.seeding import rng_for, set_seed
from pfns4neurostim.visualization import style

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rqmc_diagnostic import SCHEMES, node_sets  # noqa: E402

#: Tail mass left off each side of the window of panels (b) and (c).
WINDOW_MASS: float = 1e-3
#: Schemes whose node sets are drawn / whose variance is accounted.
SHORT: dict[str, str] = {"rqmc_8x8": "RQMC 8x8", "rqmc_8x32": "RQMC 8x32", "rqmc_1x64": "RQMC 1x64", "gh20": "GH 20", "strat_64": "Strata 64", "strat_tail_64": "Strata+tail"}
SCHEME_LABELS: dict[str, str] = {"rqmc_8x8": "RQMC 8x8\n(production)", "rqmc_8x32": "RQMC 8x32", "rqmc_1x64": "RQMC 1x64",
                                 "gh20": "Gauss-\nHermite 20", "strat_64": "Strata 64", "strat_tail_64": "Strata 64\n+ tail nodes"}


def variance_parts(crit: Any, probs: torch.Tensor) -> dict[str, dict[str, float]]:
    """Second central moment of one bar distribution, split into the central buckets and the two outer buckets.

    Two conventions: ``stated`` (what ``criterion.mean`` / ``variance`` do: half-normal outer buckets) and ``linear``
    (what ``icdf`` / ``cdf`` imply: the outer buckets uniform on their finite width).

    Args:
        crit: ``FullSupportBarDistribution``.
        probs: Bucket probabilities at one site, shape [bars].

    Returns:
        ``{'stated' | 'linear': {'mean', 'central', 'left', 'right', 'total'}}``.
    """
    b = crit.borders.double()                                                       # [bars + 1]
    w = b[1:] - b[:-1]                                                              # [bars]
    p = probs.double()                                                              # [bars]
    left = crit.halfnormal_with_p_weight_before(crit.bucket_widths[0])
    right = crit.halfnormal_with_p_weight_before(crit.bucket_widths[-1])
    out: dict[str, dict[str, float]] = {}
    for conv in ("stated", "linear"):
        m = b[:-1] + w / 2                                                          # [bars] bucket means, uniform
        sq = w ** 2 / 12.0                                                          # [bars] within-bucket variance
        if conv == "stated":
            m = m.clone()
            m[0], m[-1] = -left.mean.double() + b[1], right.mean.double() + b[-2]
            sq = sq.clone()
            sq[0], sq[-1] = left.variance.double(), right.variance.double()
        mu = (p * m).sum()
        contrib = p * (sq + (m - mu) ** 2)                                          # [bars]
        out[conv] = {"mean": float(mu), "central": float(contrib[1:-1].sum()), "left": float(contrib[0]),
                     "right": float(contrib[-1]), "total": float(contrib.sum())}
    return out


def main() -> None:
    """Build the cell, compute every panel and write the two figures."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--emg", type=int, default=0)
    ap.add_argument("--t", type=int, default=40)
    ap.add_argument("--draw", type=int, default=0)
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--diag", default="output/diagnostics/rqmc")
    ap.add_argument("--out", default="output/diagnostics/rqmc")
    args = ap.parse_args()

    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    ch = next(c for c in _channels(cfg) if c.subject == args.subject and c.emg == args.emg)
    st = build_knob(p["stress"]["knob"], [args.level]).apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
    ctx = context_for(st, args.t, args.draw, base_seed=cfg.seed, level=float(args.level))
    rng = rng_for(ch.label, "bayes_update", args.level, args.t, args.draw, base_seed=cfg.seed)
    anchors, pools, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]), int(p["anchors"]["n_observed"]), rng, ctx.sites)
    a = int(anchors[0])
    Q = st.X_pool
    eng = _make_engine("tabpfn_v2_5", p, "cpu", args.draw)
    eng.fit(ctx.X, ctx.y)
    base = eng.base(Q)
    crit, log_probs = base.parts[0]
    mu, s2 = float(base.mean[0][a]), float(base.var[0][a])
    s = np.sqrt(s2)
    probs = torch.softmax(log_probs[a].double(), -1)                                # [bars]
    parts = variance_parts(crit, probs)
    assert abs(parts["stated"]["total"] - s2) < 1e-6 * max(s2, 1.0), (parts["stated"]["total"], s2)
    u_min = float(p["quadrature"]["u_min"])
    q_lo, q_hi = base.ppf(np.array([WINDOW_MASS, 1.0 - WINDOW_MASS]), np.array([a]))[0, 0]
    pad = 0.15 * (q_hi - q_lo)
    win = (float((q_lo - pad - mu) / s), float((q_hi + pad - mu) / s))                 # window in SDs

    qrng = rng_for(ch.label, "rqmc_explainer", base_seed=cfg.seed)
    nodes = {name: node_sets(name, base, a, qrng, u_min) for name in SCHEMES}
    z = lambda y: (np.asarray(y) - mu) / s                                          # noqa: E731
    # probes: the 64 production nodes and the 64 tail-aware nodes, to see what each node does to the posterior mean
    y_prod = np.concatenate(nodes["rqmc_8x8"][0])                                    # [64]
    y_tail = nodes["strat_tail_64"][0][0]                                            # [64]
    y_all = np.concatenate([y_prod, y_tail])
    probes = eng.probe(np.repeat(Q[[a]], len(y_all), axis=0), y_all, Q)
    mus = probes.mean                                                               # [128, M]
    w_all = np.full(len(y_prod), 1.0 / len(y_prod))
    mubar = w_all @ mus[: len(y_prod)]
    move = w_all @ (mus[: len(y_prod)] - mubar) ** 2                                # [M]
    infl = move >= float(p["eps"]) * base.var[0]
    xstar = int(np.argmax(np.where(np.arange(len(move)) == a, -np.inf, move)))

    # ---------------- figure 1 ----------------
    fig, axes = style.figure("double", nrows=3, ncols=2, panel_aspect=0.62, squeeze=False, layout=style.LAYOUT_ENGINE)
    ax = axes[0][0]
    xy = np.asarray(st.ch2xy, dtype=float)
    infl_sd = np.sqrt(move / base.var[0])
    sc_a = ax.scatter(xy[:, 0], xy[:, 1], s=16, c=infl_sd, cmap=style.SEQUENTIAL_CMAP, lw=0, label="Site, shaded by posterior-mean move")
    fig.colorbar(sc_a, ax=ax, shrink=0.6).set_label("Move (SDs)")
    ax.scatter(xy[ctx.sites, 0], xy[ctx.sites, 1], s=34, facecolors="none", edgecolors=style.NEUTRAL_GREY, lw=0.8, label=f"Context (t = {args.t})")
    ax.scatter(xy[a, 0], xy[a, 1], s=90, marker="*", c=style.ANCHOR_GP, lw=0, label="Anchor $x'$", zorder=5)
    ax.scatter(xy[xstar, 0], xy[xstar, 1], s=50, marker="D", facecolors="none", edgecolors=style.ANCHOR_GP, lw=1.0, label="Readout site $x^*$ (f)", zorder=5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=style.FONT_SIZES["legend"] - 2)
    ax.set_title(ch.label, fontsize=style.FONT_SIZES["legend"])

    b = crit.borders.double().numpy()
    pr = probs.numpy()
    zi2 = z(b[1:-1])                                                                # interior borders in SDs
    w_int = np.diff(b[1:-1])
    ok = (zi2[:-1] > win[0]) & (zi2[1:] < win[1]) & (w_int > 0)
    ax = axes[0][1]
    ax.fill_between(zi2[:-1][ok], 0, (pr[1:-1][ok] / w_int[ok]) * s, step="post", color=style.ANCHOR_PFN, alpha=0.5, lw=0, label="TabPFN bars")
    zz = np.linspace(win[0], win[1], 400)
    ax.plot(zz, stats.norm.pdf(zz), color=style.NEUTRAL_GREY, ls="--", lw=style.LINE_WIDTH, label="Gaussian, same mean and SD")
    ax.set_xlim(*win)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    ax.set_xlabel(style.axis_label("z_sd"))
    ax.set_ylabel(style.axis_label("density_sd"))
    wl, wr = (b[1] - b[0]) / s, (b[-1] - b[-2]) / s
    ax.text(0.98, 0.55, f"outer buckets (off scale):\nmass {float(pr[0]):.0e} / {float(pr[-1]):.0e}\nwidth {wl:.0f} / {wr:.0f} SD", transform=ax.transAxes, ha="right", va="top", fontsize=style.FONT_SIZES["legend"] - 2)
    ax.legend(loc="upper right", fontsize=style.FONT_SIZES["legend"] - 2)

    ax = axes[1][0]
    cum = np.concatenate([[0.0], np.cumsum(pr)])                                    # CDF at the borders
    ax.plot(z(b), cum, color=style.ANCHOR_PFN, lw=style.LINE_WIDTH, label="CDF (linear inside buckets)")
    colours = [style.derive_shade(style.ANCHOR_PFN, i, 8) for i in range(8)]
    for i, ys in enumerate(nodes["rqmc_8x8"][0]):
        ax.scatter(z(ys), np.full(len(ys), -0.06), s=6, color=colours[i], lw=0, clip_on=False)
    us0 = np.interp(z(nodes["rqmc_8x8"][0][0]), z(b), cum)
    ax.vlines(z(nodes["rqmc_8x8"][0][0]), -0.05, us0, colors=colours[0], lw=0.6)
    ax.hlines(us0, win[0], z(nodes["rqmc_8x8"][0][0]), colors=colours[0], lw=0.6)
    ax.scatter(z(np.concatenate(nodes["rqmc_8x32"][0])), np.full(256, -0.14), s=2, color=style.NEUTRAL_GREY, lw=0, clip_on=False)
    ax.text(win[1], -0.06, "8x8 nodes ", ha="right", va="center", fontsize=style.FONT_SIZES["legend"] - 3)
    ax.text(win[1], -0.14, "8x32 nodes ", ha="right", va="center", fontsize=style.FONT_SIZES["legend"] - 3)
    ax.set_xlim(*win)
    ax.set_ylim(-0.2, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(style.axis_label("z_sd"))
    ax.set_ylabel(style.axis_label("cdf_u"))

    ax = axes[1][1]
    side = crit.halfnormal_with_p_weight_before(crit.bucket_widths[-1])
    scale_r = float(side.scale) / s                                                 # SDs
    width_r = wr
    x = np.geomspace(1e-2, 6 * max(width_r, scale_r), 300)
    pL = float(pr[-1])
    ax.plot(x, np.where(x <= width_r, pL / width_r, np.nan), color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH, label="quantile / CDF: uniform on the bucket")
    ax.plot(x, pL * stats.halfnorm.pdf(x, scale=scale_r), color=style.ANCHOR_PFN, lw=style.LINE_WIDTH, label="mean / variance: half-normal")
    ax.axvline(width_r, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":")
    ax.set_ylim(1e-9, None)
    tail_node = (nodes["strat_tail_64"][0][0][-1] - b[-2]) / s
    ax.scatter([tail_node], [pL * stats.halfnorm.pdf(tail_node, scale=scale_r)], s=30, marker="v", color=style.ANCHOR_PFN, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Beyond last border (SDs)")
    ax.set_ylabel(style.axis_label("density_sd"))
    ax.legend(loc="lower left", fontsize=style.FONT_SIZES["legend"] - 2)

    ax = axes[2][0]
    comp = {"Central buckets": style.ANCHOR_PFN, "Left tail": style.derive_shade(style.ANCHOR_PFN, 0, 3), "Right tail": style.derive_shade(style.ANCHOR_PFN, 2, 3)}
    xs = np.arange(2 + len(SCHEMES))
    for xpos, conv in ((0, "stated"), (1, "linear")):
        bottom = 0.0
        for key, colour in zip(("central", "left", "right"), comp.values()):
            val = parts[conv][key] / s2
            ax.bar(xpos, val, bottom=bottom, color=colour, width=0.7, label=list(comp)[("central", "left", "right").index(key)] if xpos == 0 else None)
            bottom += val
    for i, name in enumerate(SCHEMES):
        ys, wk = nodes[name]
        v = float(np.mean([wk @ (yy - wk @ yy) ** 2 for yy in ys]))
        ax.bar(2 + i, v / s2, color=style.NEUTRAL_GREY, width=0.7)
    ax.axhline(1.0, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Stated", "Quantile"] + [SHORT[n] for n in SCHEMES], rotation=45, ha="right", fontsize=style.FONT_SIZES["tick"] - 1)
    ax.set_ylabel(style.axis_label("var_stated_ratio"))
    ax.legend(loc="upper right", fontsize=style.FONT_SIZES["legend"] - 2)

    ax = axes[2][1]
    d_mu = (mus[:, xstar] - base.mean[0][xstar]) / np.sqrt(base.var[0][xstar])      # [128]
    n1 = len(y_prod)
    ax.scatter(z(y_prod), d_mu[:n1], s=8, color=style.NEUTRAL_GREY, lw=0, label="RQMC 8x8 nodes")
    ax.scatter(z(y_tail[1:-1]), d_mu[n1 + 1:-1], s=8, color=style.ANCHOR_PFN, lw=0, label="Strata 64 + tail nodes")
    ax.scatter(z(y_tail[[0, -1]]), d_mu[[n1, -1]], s=36, marker="v", color=style.ANCHOR_GP, lw=0, label="Tail nodes", zorder=5)
    ax.set_xscale("symlog", linthresh=2.0)
    ax.axhline(0.0, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":")
    ax.set_xlabel(style.axis_label("z_sd"))
    ax.set_ylabel("Change of the posterior mean at $x^*$\n(SDs at $x^*$)")
    ax.legend(loc="upper left", fontsize=style.FONT_SIZES["legend"] - 2)
    style.panel_letters(list(axes.ravel()))
    paths = style.save_figure(fig, args.out, "rqmc_explainer", formats=("svg", "png"))

    # ---------------- figure 2: readouts per scheme ----------------
    frames = [pd.read_csv(os.path.join(d, "rqmc_diagnostic.csv")) for d in (args.diag, os.path.join(args.out + "_tail"))
              if os.path.exists(os.path.join(d, "rqmc_diagnostic.csv"))]
    if frames:
        d = pd.concat(frames, ignore_index=True)
        d = d[d["engine"] == "tabpfn_v2_5"]
        subjects = sorted(d["subject"].unique())
        fig2, axes2 = style.figure("double", nrows=1, ncols=2, panel_aspect=0.8, squeeze=False, sharey=True, layout=style.LAYOUT_ENGINE)
        for ax, col, title in ((axes2[0][0], "cov_asymmetry", "Raw"), (axes2[0][1], "cov_asymmetry_c", "Moment-matched")):
            for sj in subjects:
                sty = style.subject_style(sj, subjects)
                g = d[(d["subject"] == sj) & (d["context_t"] == args.t)].set_index("scheme").reindex(SCHEMES)
                ax.plot(np.arange(len(SCHEMES)), g[col], color=sty.color, marker=sty.marker, lw=style.LINE_WIDTH, label=sty.label)
            ax.set_xticks(np.arange(len(SCHEMES)))
            ax.set_xticklabels([SCHEME_LABELS[n] for n in SCHEMES], rotation=60, ha="right", fontsize=style.FONT_SIZES["tick"] - 1)
            ax.set_title(title, fontsize=style.FONT_SIZES["legend"])
        axes2[0][0].set_ylabel(style.axis_label("cov_asymmetry"))
        axes2[0][1].legend(loc="upper right", fontsize=style.FONT_SIZES["legend"] - 2)
        style.panel_letters(list(axes2[0]))
        paths += style.save_figure(fig2, args.out, "rqmc_scheme_readouts", formats=("svg", "png"))
    print("\n".join(paths))
    print(f"stated variance parts / s2: { {k: round(v / s2, 3) for k, v in parts['stated'].items() if k != 'mean'} }")
    print(f"linear variance parts / s2: { {k: round(v / s2, 3) for k, v in parts['linear'].items() if k != 'mean'} }")
    print(f"outer buckets: p0 {float(pr[0]):.2e}, pL {pL:.2e}, widths {wl:.1f} / {wr:.1f} SD, half-normal scale R {scale_r:.1f} SD")


if __name__ == "__main__":
    main()
