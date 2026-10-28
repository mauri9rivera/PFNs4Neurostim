"""Official TabPFN bar-distribution EI / PI / UCB against the Gaussian versions of the BO loop (exploratory; user 2026-10-27).

``acquisition/registry.py`` scores every surrogate through its two moments (``marginals``): Gaussian EI (Jones et al., 1998),
Gaussian PI and ``mean + kappa * sd``. TabPFN's own ``FullSupportBarDistribution`` ships closed-form versions that integrate the
bar distribution itself:

    ei(logits, best_f)    sum over bars of p_b * E[(y - best_f)+ | bar b], uniform inside a bar, half-normal outer bars
    pi(logits, best_f)    sum over bars of p_b * P(y > best_f | bar b), same convention
    ucb(logits, best_f, rest_prob)    the 1 - rest_prob quantile (``icdf``), NOT ``mean + kappa * sd``; kappa maps to
                                      rest_prob = Phi(-kappa), the mapping ``TabPFNSurrogate.predict_ucb`` already uses

On OMEGA-style contexts (one fitted TabPFN-2.5 per channel and context size, candidates = the unobserved sites) this reports, per
acquisition, how much the two versions disagree: the Spearman correlation of the scores over the candidates, whether the argmax is
the same site, and the ground-truth gap between the two argmax picks. It is a feasibility and size-of-effect check, not a BO
re-run.

    python scripts/official_acq_check.py --ts 10 25 50 80
"""
from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.models.pfn.bar_acquisition import outer_bucket_share
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import _channels, _make_engine, _validate_bayes_update, load_mechanism_config
from pfns4neurostim.seeding import rng_for, set_seed

#: UCB exploration coefficients compared (the registry default is 2.0).
KAPPAS: tuple[float, ...] = (1.0, 2.0)
#: UCB coefficients of the outer-bucket check (the Omega14 grid).
OUTER_KAPPAS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)


def gaussian_scores(mean: np.ndarray, sd: np.ndarray, incumbent: float, kappas: tuple[float, ...]) -> dict[str, np.ndarray]:
    """The BO loop's own EI / PI / UCB from two moments (``acquisition/registry.py``).

    Args:
        mean: Predictive mean per candidate, shape [N].
        sd: Predictive SD per candidate, shape [N].
        incumbent: Best observed value, in the model's y space.
        kappas: UCB coefficients.

    Returns:
        ``{'ei', 'pi', 'ucb_k<kappa>'}``, each shape [N].
    """
    gap = mean - incumbent                                                          # [N]
    safe = np.maximum(sd, 1e-12)                                                    # [N]
    z = gap / safe                                                                  # [N]
    out = {"ei": gap * stats.norm.cdf(z) + safe * stats.norm.pdf(z), "pi": stats.norm.cdf(z)}
    for k in kappas:
        out[f"ucb_k{k:g}"] = mean + k * sd
    return out


def merge_zero_width_buckets(crit: Any, log_probs: torch.Tensor) -> tuple[Any, torch.Tensor]:
    """Merge every zero-width bucket into the next positive-width one (the repair of branch ``pfn-internal-engines``).

    The TabPFN-2.5 checkpoint's 5001 borders contain a duplicate, so one bucket has width 0. ``FullSupportBarDistribution.ei``
    divides every bucket's improvement integral by its width: 0/0 there, and the sum over bars is NaN at *every* candidate
    (measured here with tabpfn 6.3.2; the extbo branch measured it with 9.0.0). A zero-width bucket is a point mass at its
    border, so its probability is added to the next bucket and the duplicate border dropped; no other bucket changes.

    Args:
        crit: ``FullSupportBarDistribution`` of the fit.
        log_probs: Log-probabilities over its bars, shape [N, bars].

    Returns:
        ``(repaired criterion, log-probabilities over its merged bars [N, bars'])``.
    """
    from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution  # noqa: PLC0415

    borders = crit.borders.detach()                                                # [bars + 1]
    positive = (borders[1:] - borders[:-1]) > 0                                    # [bars]
    keep = torch.nonzero(positive).flatten()                                       # [bars']
    target = torch.searchsorted(keep, torch.arange(len(positive), device=keep.device))   # [bars]
    merged = torch.zeros(log_probs.shape[0], len(keep), dtype=log_probs.dtype, device=log_probs.device)   # [N, bars']
    merged.index_add_(1, target.to(log_probs.device), torch.softmax(log_probs, -1))
    return FullSupportBarDistribution(torch.cat([borders[keep], borders[-1:]])), merged.log()


def official_scores(crit: Any, log_probs: torch.Tensor, incumbent: float, kappas: tuple[float, ...]) -> dict[str, np.ndarray]:
    """The same acquisitions from the criterion's closed forms over the bars.

    Args:
        crit: ``FullSupportBarDistribution`` of the fit.
        log_probs: Log-probabilities over the bars at the candidates, shape [N, bars].
        incumbent: Best observed value, in the model's y space.
        kappas: UCB coefficients (mapped to ``rest_prob = Phi(-kappa)``).

    Returns:
        ``{'ei', 'pi', 'ucb_k<kappa>'}``, each shape [N].
    """
    crit, lg = merge_zero_width_buckets(crit, log_probs.to(crit.borders.device))   # repaired criterion, [N, bars']
    with torch.no_grad():
        out = {"ei": crit.ei(lg, incumbent).cpu().numpy(), "pi": crit.pi(lg, incumbent).cpu().numpy()}
        for k in kappas:
            out[f"ucb_k{k:g}"] = crit.ucb(lg, incumbent, rest_prob=0.5 * math.erfc(k / math.sqrt(2.0)), maximize=True).cpu().numpy()
    return out


def main() -> None:
    """Score every context with both versions and write ``official_acq_check.csv``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--ts", type=int, nargs="+", default=[10, 25, 50, 80])
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="output/diagnostics/official_acq")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "official_acq_check.csv")
    knob = build_knob(p["stress"]["knob"], [args.level])
    rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for ch in _channels(cfg):
        st = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
        for t in args.ts:
            for draw in args.draws:
                ctx = context_for(st, t, draw, base_seed=cfg.seed, level=float(args.level))
                eng = _make_engine("tabpfn_v2_5", p, "cpu", draw)
                eng.fit(ctx.X, ctx.y)
                base = eng.base(st.X_pool)
                crit, log_probs = base.parts[0]                                      # criterion, [M, bars]
                mean, sd = base.mean[0], np.sqrt(base.var[0])                        # [M], [M]
                cand = np.setdiff1d(np.arange(len(mean)), np.asarray(ctx.sites, dtype=int))   # unobserved sites
                inc = float(np.max(ctx.y))
                g = gaussian_scores(mean[cand], sd[cand], inc, KAPPAS)
                o = official_scores(crit, log_probs[cand], inc, KAPPAS)
                for k in OUTER_KAPPAS:
                    outer_rows.append({"subject": ch.subject, "emg": ch.emg, "context_t": t, "kappa": k,
                                       "outer_share": outer_bucket_share(crit, log_probs[cand], k)})
                gt = np.asarray(st.y_gt)[cand]
                for name in g:
                    a_g, a_o = int(np.argmax(g[name])), int(np.argmax(o[name]))
                    rank_of_official = int((g[name] > g[name][a_o]).sum()) + 1       # 1 = the Gaussian argmax itself
                    rows.append({"subject": ch.subject, "emg": ch.emg, "context_t": t, "draw": draw, "acq": name,
                                 "n_candidates": len(cand), "spearman": float(stats.spearmanr(g[name], o[name])[0]),
                                 "same_argmax": float(a_g == a_o), "rank_of_official_pick": rank_of_official,
                                 "delta_gt": float((gt[a_o] - gt[a_g]) / st.gt_range),
                                 "score_ratio_median": float(np.median(o[name] / np.where(g[name] == 0, np.nan, g[name])))})
                print(f"[official_acq] {ch.label} t={t} ({time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    summary = df.groupby(["acq", "context_t"]).agg(spearman=("spearman", "median"), same_argmax=("same_argmax", "mean"),
                                                   rank_p90=("rank_of_official_pick", lambda s: s.quantile(0.9)),
                                                   delta_gt_mean=("delta_gt", "mean"), delta_gt_abs=("delta_gt", lambda s: s.abs().mean()))
    summary.to_csv(os.path.join(args.out, "official_acq_summary.csv"))
    outer = pd.DataFrame(outer_rows)
    outer.to_csv(os.path.join(args.out, "ucb_outer_share.csv"), index=False)
    print("share of candidates whose UCB quantile lies in an outer bucket (mean over contexts)")
    print(outer.pivot_table(index="kappa", columns="context_t", values="outer_share", aggfunc="mean").round(3).to_string())
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
