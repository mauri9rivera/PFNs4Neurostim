"""Convergence traces of the GP-MLL optimisers on OMEGA contexts (exploratory; user 2026-10-27).

``gp_optimizer_sweep.py`` reports only the END of each fit. This script records the objective at EVERY evaluation, so the
way each optimiser gets to its answer, and what it costs in wall time, can be drawn:

* ``best``   L-BFGS, 20 restarts x 100 iterations (``restart_seed`` 100): the *reference fit*. Each start is traced; the fit
             is the start with the lowest marginal likelihood among those that gave a usable model.
* ``prod``   the production recipe, L-BFGS 5 restarts x 50 iterations, ``restart_seed`` 0 / 1 / 2.
* ``adam``   Adam, one start from gpytorch's initialisation, ``--adam-steps`` steps, learning rate 0.01 / 0.05 / 0.1
             (the sweep's grid stops at 400 steps; this runs on to show whether Adam would arrive at all).

Datasets: the context D of an OMEGA cell (``base``) and D + (x', y') with y' = mu(x') + 2 sd(x') (``z+2``), the condition of the
768 refits per cell. Every start of a multi-start arm is a row of ``starts_*.csv`` (final NLML, gap to the reference fit, how far its
prediction is from the reference one), every objective evaluation a row of ``traces_*.csv``. ``--figures-only`` draws
``gp_convergence.svg`` and ``gp_convergence_tolerances.csv`` from those files and from the sweep's ``sweep_main_shard*.csv``.

    python scripts/gp_convergence_traces.py --shard 0/4
    python scripts/gp_convergence_traces.py --figures-only
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from pfns4neurostim.analysis import bayes_update as B
from pfns4neurostim.analysis.context import context_for
from pfns4neurostim.data.stress import build_knob
from pfns4neurostim.experiments.mechanism import _channels, _validate_bayes_update, load_mechanism_config
from pfns4neurostim.models.gp.surrogates import GPSurrogate
from pfns4neurostim.seeding import rng_for, set_seed
from pfns4neurostim.visualization import style

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gp_optimizer_sweep import BEST_SEED, PRODUCTION_SEEDS, arm_specs  # noqa: E402

#: Learning rates of the Adam traces (the sweep's grid).
ADAM_LRS: tuple[float, ...] = (0.01, 0.05, 0.1)
#: Standardised surprise of the refit dataset (y' = mu + z sd at the anchor).
REFIT_Z: float = 2.0
#: Tolerances (nats on the NLML, SDs on the prediction) of the convergence criterion: the pre-declared one first.
TOLERANCES: tuple[tuple[float, float], ...] = ((0.01, 0.01), (0.1, 0.05), (1.0, 0.1), (1.0, 0.25), (3.0, 0.25))
#: NLML-gap tolerances (nats) drawn in the restart ladder.
LADDER_TOLS: tuple[float, ...] = (0.01, 0.1, 1.0)
#: Time grid (s) of the wall-clock panel.
TIME_GRID: tuple[float, float, int] = (0.01, 100.0, 80)


class _Recorder:
    """Collects ``(seconds, total NLML)`` per objective evaluation of the candidate being fitted."""

    active: list[tuple[float, float]] | None = None
    t0: float = 0.0


def _install_recorder() -> None:
    """Wrap ``ExactMarginalLogLikelihood.forward`` so every training-mode evaluation lands in :class:`_Recorder`."""
    original = ExactMarginalLogLikelihood.forward

    def forward(self: ExactMarginalLogLikelihood, function_dist: Any, target: torch.Tensor, *params: Any) -> torch.Tensor:
        out = original(self, function_dist, target, *params)
        if _Recorder.active is not None and self.model.training:
            _Recorder.active.append((time.perf_counter() - _Recorder.t0, -float(out.detach()) * int(target.shape[-1])))
        return out

    ExactMarginalLogLikelihood.forward = forward  # type: ignore[method-assign]


class TracedGP(GPSurrogate):
    """``GPSurrogate`` that keeps the objective trace and the prediction of every start.

    Args:
        Q: Readout sites at which each start's predictive mean and SD are stored, shape [M, D].
        **kwargs: ``GPSurrogate`` arguments.
    """

    def __init__(self, Q: np.ndarray, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._Q = Q
        self.starts: list[dict[str, Any]] = []

    def _fit_candidate(self, train_x: torch.Tensor, train_y: torch.Tensor, init: dict[str, float] | None,
                       floor: float) -> tuple[float, Any, Any, dict[str, float]]:
        """Fit one start and record its trace, final likelihood and prediction (also when the start fails)."""
        _Recorder.active, _Recorder.t0 = [], time.perf_counter()
        row: dict[str, Any] = {"start": len(self.starts), "feasible": 0.0, "eval_nll_total": np.nan,
                               "train_nll_total": np.nan}
        try:
            result = super()._fit_candidate(train_x, train_y, init, floor)
        finally:
            row["trace"], _Recorder.active = _Recorder.active, None
            row["seconds"] = time.perf_counter() - _Recorder.t0
            self.starts.append(row)
        diag = result[3]
        n = int(train_x.shape[0])
        mean, sd = self.predict_frozen(self._Q)                                    # [M], [M]
        # ``result[0]`` is the *evaluation-mode* loss (the criterion that picks the best restart): gpytorch's
        # ``model(train_x)`` in eval mode is the posterior at the training inputs, so it is a training-set predictive
        # likelihood, not the marginal likelihood the optimiser minimised. The train-mode value at the same parameters is
        # the actual NLML; both are recorded.
        model, likelihood = result[1], result[2]
        model.train()
        likelihood.train()
        with torch.no_grad():
            train_nll = -float(ExactMarginalLogLikelihood(likelihood, model)(model(train_x), train_y)) * n
        model.eval()
        likelihood.eval()
        row.update(feasible=1.0, eval_nll_total=float(result[0]) * n, train_nll_total=train_nll, mean=mean, sd=sd,
                   n_evals=diag["gp_n_obj_evals"], noise=diag["gp_noise"], lengthscale=diag["gp_lengthscale"],
                   outputscale=diag["gp_outputscale"])
        return result


def _fit(kwargs: dict[str, Any], Q: np.ndarray, X: np.ndarray, y: np.ndarray) -> TracedGP:
    """Fit one traced arm.

    Args:
        kwargs: ``GPSurrogate`` arguments of the arm.
        Q: Readout sites, shape [M, D].
        X: Training inputs, shape [N, D].
        y: Training targets, shape [N].

    Returns:
        The fitted surrogate with its ``starts``.
    """
    gp = TracedGP(Q, **kwargs)
    gp.fit(X, y)
    return gp


def _rows(gp: TracedGP, meta: dict[str, Any], arm: str, ref: tuple[float, np.ndarray, np.ndarray] | None,
          n_train: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Trace rows and start rows of one arm.

    Args:
        gp: The fitted arm.
        meta: Context and dataset identifiers.
        arm: Arm name.
        ref: ``(NLML, mean, sd)`` of the reference fit, or None when this arm is the reference.
        n_train: Training rows of the dataset.

    Returns:
        ``(trace_rows, start_rows)``.
    """
    feasible = [s for s in gp.starts if s["feasible"] > 0.0]
    by_eval = min(feasible, key=lambda s: s["eval_nll_total"]) if feasible else None     # the production rule
    by_train = min(feasible, key=lambda s: s["train_nll_total"]) if feasible else None   # the marginal likelihood
    if ref is None:
        ref = (by_train["train_nll_total"], by_train["mean"], by_train["sd"])
    nll_ref, mean_ref, sd_ref = ref
    traces, starts = [], []
    for s in gp.starts:
        traces += [{**meta, "arm": arm, "start": s["start"], "i": i, "seconds": sec, "nll_total": nll}
                   for i, (sec, nll) in enumerate(s["trace"])]
        row = {**meta, "arm": arm, "start": s["start"], "n_train": n_train, "feasible": s["feasible"],
               "eval_nll_total": s["eval_nll_total"], "train_nll_total": s["train_nll_total"], "seconds": s["seconds"],
               "selected_eval": float(s is by_eval), "selected_train": float(s is by_train)}
        if s["feasible"] > 0.0:
            dmu = np.abs(s["mean"] - mean_ref) / sd_ref
            row.update(nll_gap_nats=s["train_nll_total"] - nll_ref, dmu_median=float(np.median(dmu)),
                       dmu_max=float(dmu.max()), n_evals=s["n_evals"], noise=s["noise"], lengthscale=s["lengthscale"],
                       outputscale=s["outputscale"])
        starts.append(row)
    return traces, starts


def run(args: argparse.Namespace) -> None:
    """Trace every arm on every context of the shard and append ``traces_*.csv`` / ``starts_*.csv``."""
    torch.set_num_threads(1)
    _install_recorder()
    cfg = load_mechanism_config(args.config, ["device=cpu"])
    p = _validate_bayes_update(cfg.params, cfg.context_sizes)
    set_seed(cfg.seed)
    os.makedirs(args.out, exist_ok=True)
    shard = tuple(int(v) for v in args.shard.split("/"))
    stem = f"{args.tag}_shard{shard[0]}of{shard[1]}"
    out_tr, out_st = (os.path.join(args.out, f"{k}_{stem}.csv") for k in ("traces", "starts"))
    key_cols = ["subject", "emg", "context_t", "draw"]
    done: set[tuple[Any, ...]] = set()
    if os.path.exists(out_st):
        done = {tuple(r) for r in pd.read_csv(out_st)[key_cols].itertuples(index=False, name=None)}

    base_kw = dict(p["gp_params"])
    specs = arm_specs(base_kw, args.best_restarts, args.best_steps)
    prod = {f"prod_s{s}": specs[f"lbfgs{base_kw['n_restarts']}_s{s}"] for s in PRODUCTION_SEEDS}
    adam = {f"adam_lr{lr:g}": {**base_kw, "optimizer": "adam", "n_opt_steps": args.adam_steps, "lr": lr,
                                "n_restarts": 1} for lr in ADAM_LRS}
    knob = build_knob(p["stress"]["knob"], [args.level])
    tasks = [(ch, int(t), int(d)) for ch in _channels(cfg) for t in args.ts for d in args.draws]
    tasks = [task for i, task in enumerate(tasks) if i % shard[1] == shard[0]]
    print(f"[traces] {len(tasks)} contexts ({len(done)} done), adam {args.adam_steps} steps", flush=True)
    t_start = time.time()
    for n, (ch, t, draw) in enumerate(tasks):
        if (ch.subject, ch.emg, t, draw) in done:
            continue
        st = knob.apply(ch, args.level, rng_for(ch.label, "bayes_update", args.level, base_seed=cfg.seed))
        ctx = context_for(st, t, draw, base_seed=cfg.seed, level=float(args.level))
        rng = rng_for(ch.label, "bayes_update", args.level, t, draw, base_seed=cfg.seed)
        anchors, _, _ = B.select_anchors_uniform(st, int(p["anchors"]["n_unobserved"]), int(p["anchors"]["n_observed"]),
                                                 rng, ctx.sites)
        a, Q = int(anchors[0]), st.X_pool
        ident = {"subject": ch.subject, "emg": ch.emg, "context_t": t, "draw": draw}
        gp_prod0 = _fit(prod["prod_s0"], Q, ctx.X, ctx.y)
        sel0 = min((s for s in gp_prod0.starts if s["feasible"] > 0.0), key=lambda s: s["eval_nll_total"])
        m_a, s_a = float(sel0["mean"][a]), float(sel0["sd"][a])
        datasets = {"base": (ctx.X, ctx.y),
                    f"z{REFIT_Z:+g}": (np.vstack([ctx.X, Q[a]]), np.append(ctx.y, m_a + REFIT_Z * s_a))}
        traces: list[dict[str, Any]] = []
        starts: list[dict[str, Any]] = []
        for ds, (Xd, yd) in datasets.items():
            meta = {**ident, "dataset": ds}
            gp_best = _fit(specs["best"], Q, Xd, yd)
            tr, sr = _rows(gp_best, meta, "best", None, len(yd))
            traces += tr
            starts += sr
            chosen = min((s for s in gp_best.starts if s["feasible"] > 0.0), key=lambda s: s["train_nll_total"])
            ref = (chosen["train_nll_total"], chosen["mean"], chosen["sd"])
            arms = {**prod, **(adam if ds == "base" else {})}
            for name, kw in arms.items():
                gp = _fit(kw, Q, Xd, yd)
                tr, sr = _rows(gp, meta, name, ref, len(yd))
                traces += tr
                starts += sr
        pd.DataFrame(traces).to_csv(out_tr, mode="a", header=not os.path.exists(out_tr), index=False)
        pd.DataFrame(starts).to_csv(out_st, mode="a", header=not os.path.exists(out_st), index=False)
        print(f"[traces] {n + 1}/{len(tasks)} ({time.time() - t_start:.0f}s) {ch.label} t={t} d={draw}", flush=True)
    print(f"[traces] done ({time.time() - t_start:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------------------------------------------------
def _load(out: str, kind: str, tag: str) -> pd.DataFrame:
    """Concatenate the shard CSVs ``<kind>_<tag>_shard*.csv``.

    Raises:
        FileNotFoundError: If no shard exists.
    """
    files = sorted(glob.glob(os.path.join(out, f"{kind}_{tag}_shard*.csv")))
    if not files:
        raise FileNotFoundError(f"no {kind}_{tag}_shard*.csv under {out}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _running_gap(trace: pd.DataFrame, nll_ref: float, offset: float = 0.0, evals_offset: int = 0) -> pd.DataFrame:
    """Best-so-far NLML gap along one start's trace.

    Args:
        trace: Rows of one start (``i``, ``seconds``, ``nll_total``), in evaluation order.
        nll_ref: NLML of the reference fit.
        offset: Seconds already spent by earlier starts of the same fit.
        evals_offset: Evaluations already spent by earlier starts.

    Returns:
        Frame with ``seconds``, ``evals``, ``gap`` (running minimum, clipped at 0).
    """
    gap = np.minimum.accumulate(trace["nll_total"].to_numpy()) - nll_ref
    return pd.DataFrame({"seconds": trace["seconds"].to_numpy() + offset,
                         "evals": np.arange(1, len(trace) + 1) + evals_offset, "gap": np.maximum(gap, 0.0)})


def _fit_curve(traces: pd.DataFrame, starts: pd.DataFrame, arm: str, nll_ref: float) -> pd.DataFrame:
    """Running-best gap of a whole fit: its starts one after the other.

    Args:
        traces: Trace rows of one context, dataset and arm.
        starts: Start rows of the same.
        arm: Arm name (documentation only).
        nll_ref: NLML of the reference fit.

    Returns:
        Frame with ``seconds``, ``evals``, ``gap`` (running minimum over all starts so far).
    """
    parts, t_off, e_off, best = [], 0.0, 0, np.inf
    for k, sub in traces.groupby("start", sort=True):
        nll = np.minimum(sub["nll_total"].to_numpy(), best)
        sub = sub.assign(nll_total=nll)
        parts.append(_running_gap(sub, nll_ref, t_off, e_off))
        best = float(nll.min())
        t_off += float(starts.loc[starts["start"] == k, "seconds"].iloc[0])
        e_off += len(sub)
    return pd.concat(parts, ignore_index=True)


def _on_grid(curve: pd.DataFrame, x: str, grid: np.ndarray) -> np.ndarray:
    """Step function of ``gap`` over ``x`` on ``grid`` (the first value before the first point, the last after the last)."""
    xs, g = curve[x].to_numpy(), curve["gap"].to_numpy()
    idx = np.clip(np.searchsorted(xs, grid, side="right") - 1, 0, len(xs) - 1)
    return g[idx]


def figures(args: argparse.Namespace) -> None:
    """Draw ``gp_convergence.svg`` and ``gp_selection_rule.svg``; write the tolerance and restart-ladder tables."""
    tr = _load(args.out, "traces", args.tag)
    st = _load(args.out, "starts", args.tag)
    keys = ["subject", "emg", "context_t", "draw", "dataset"]
    feas = st[(st["arm"] == "best") & (st["feasible"] > 0)]
    ref = feas.groupby(keys)["train_nll_total"].min()                                # the reference fit's marginal NLML
    base = tr[tr["dataset"] == "base"]
    adam_arms = [f"adam_lr{lr:g}" for lr in ADAM_LRS]
    contexts = list(base[keys].drop_duplicates().itertuples(index=False, name=None))
    grid = np.geomspace(*TIME_GRID[:2], TIME_GRID[2])
    labels = {"one": "L-BFGS, 1 start", "prod": "L-BFGS, best of 5 by NLML", "best": "L-BFGS, 20 restarts"}
    curves: dict[str, list[np.ndarray]] = {k: [] for k in (*labels.values(), *adam_arms)}
    by_ctx = {k: g for k, g in base.groupby(keys[:4])}
    sts = {k: g for k, g in st[st["dataset"] == "base"].groupby(keys[:4])}
    gaps_one: dict[tuple[Any, ...], float] = {}
    for c in contexts:
        c4 = c[:4]
        t_c, s_c, r = by_ctx[c4], sts[c4], float(ref.loc[c])
        one = t_c[(t_c["arm"] == "best") & (t_c["start"] == 0)]
        curves[labels["one"]].append(_on_grid(_running_gap(one, r), "seconds", grid))
        for label, arm in ((labels["prod"], "prod_s0"), (labels["best"], "best")):
            curves[label].append(_on_grid(_fit_curve(t_c[t_c["arm"] == arm], s_c[s_c["arm"] == arm], arm, r), "seconds", grid))
        for arm in adam_arms:
            sub = t_c[(t_c["arm"] == arm) & (t_c["start"] == 0)]
            if len(sub):
                curves[arm].append(_on_grid(_running_gap(sub, r), "seconds", grid))
        gaps_one[c4] = float(_running_gap(one, r)["gap"].iloc[-1])
    med = float(np.median(list(gaps_one.values())))
    pick = min(gaps_one, key=lambda k: abs(gaps_one[k] - med))
    sub_t, r_pick = by_ctx[pick], float(ref.loc[(*pick, "base")])
    ex_starts = {k: _running_gap(g, r_pick) for k, g in sub_t[sub_t["arm"] == "best"].groupby("start")}
    ex_adam = {a: _running_gap(sub_t[(sub_t["arm"] == a) & (sub_t["start"] == 0)], r_pick) for a in adam_arms}

    # restart ladder: the first k starts of the reference recipe, picked by the marginal NLML or by the production rule
    ladder: list[dict[str, Any]] = []
    for _, g in st[(st["arm"] == "best") & (st["feasible"] > 0)].groupby(keys):
        g = g.sort_values("start")
        tn, en = g["train_nll_total"].to_numpy(), g["eval_nll_total"].to_numpy()
        r = tn.min()
        for k in range(1, len(g) + 1):
            for rule, score in (("marginal NLML", tn), ("production rule", en)):
                row = g.iloc[int(np.argmin(score[:k]))]
                ladder.append({"k": k, "rule": rule, "gap": float(row["train_nll_total"] - r),
                               "seconds": g["seconds"].iloc[:k].sum(), "dmu_median": float(row["dmu_median"])})
    lad = pd.DataFrame(ladder)
    lad_tab = lad.groupby(["rule", "k"]).agg(seconds_median=("seconds", "median"), dmu_median=("dmu_median", "median"),
                                             dmu_p90=("dmu_median", lambda s: s.quantile(0.9)),
                                             **{f"share_gap_le_{tol:g}": ("gap", lambda s, tol=tol: float((s <= tol).mean()))
                                                for tol in LADDER_TOLS})
    lad_tab.to_csv(os.path.join(args.out, "gp_convergence_restart_ladder.csv"))

    # final fits per arm (what each recipe would return) against the marginal-NLML reference
    is_adam = st["arm"].str.startswith("adam")
    multi = (st["arm"] != "best") & ~is_adam & (st["selected_eval"] > 0)
    best_ref = (st["arm"] == "best") & (st["selected_train"] > 0)
    best_one = (st["arm"] == "best") & (st["start"] == 0)
    best_prod = (st["arm"] == "best") & (st["selected_eval"] > 0)
    rows_fin = []
    for label, sel in (("ref: 20 restarts, picked by NLML", best_ref), ("20 restarts, production rule", best_prod),
                       ("L-BFGS, default start only", best_one)):
        rows_fin.append(st[(st["feasible"] > 0) & sel].assign(arm_label=label))
    rest = st[(st["feasible"] > 0) & (multi | is_adam)]
    rows_fin.append(rest.assign(arm_label=rest["arm"]))
    fin = pd.concat(rows_fin, ignore_index=True)
    tol_rows = []
    for arm, g in fin.groupby("arm_label"):
        row = {"arm": arm, "datasets": len(g), "seconds_median": g["seconds"].median(),
               "gap_median": g["nll_gap_nats"].median(), "dmu_median": g["dmu_median"].median(),
               "dmu_p90": g["dmu_median"].quantile(0.9)}
        for nats, sds in TOLERANCES:
            row[f"gap<={nats:g}&dmu<={sds:g}"] = float(((g["nll_gap_nats"] <= nats) & (g["dmu_median"] <= sds)).mean())
        tol_rows.append(row)
    tol = pd.DataFrame(tol_rows).set_index("arm")
    tol.round(4).to_csv(os.path.join(args.out, "gp_convergence_tolerances.csv"))

    fig, axes = style.figure("double", nrows=2, ncols=2, panel_aspect=0.7, squeeze=False, layout=style.LAYOUT_ENGINE)
    ax = axes[0][0]
    for k, g in ex_starts.items():
        ax.plot(g["evals"], g["gap"], color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, alpha=0.7,
                label="20 L-BFGS starts" if k == 0 else None)
    ax.plot(ex_starts[0]["evals"], ex_starts[0]["gap"], color=style.ANCHOR_GP, lw=style.LINE_WIDTH,
            label="L-BFGS, default start")
    for i, a in enumerate(adam_arms):
        ax.plot(ex_adam[a]["evals"], ex_adam[a]["gap"], color=style.derive_shade(style.NEUTRAL_GREY, i, len(adam_arms)),
                ls="--", lw=style.LINE_WIDTH, label=f"Adam, lr {a.split('lr')[1]}")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-2)
    for nats in (0.01, 1.0):
        ax.axhline(nats, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":")
    ax.set_xlabel(style.axis_label("gp_evals"))
    ax.set_ylabel(style.axis_label("nlml_gap"))
    ax.set_ylim(bottom=0)
    ax.set_title("One context (median final gap)", fontsize=style.FONT_SIZES["legend"])
    ax.legend(loc="upper right", fontsize=style.FONT_SIZES["legend"] - 1)

    ax = axes[0][1]
    styles = {labels["one"]: (style.derive_shade(style.ANCHOR_GP, 0, 3), "-"),
              labels["prod"]: (style.derive_shade(style.ANCHOR_GP, 1, 3), "-"),
              labels["best"]: (style.derive_shade(style.ANCHOR_GP, 2, 3), "-")}
    for i, a in enumerate(adam_arms):
        styles[a] = (style.derive_shade(style.NEUTRAL_GREY, i, len(adam_arms)), "--")
    for label, runs in curves.items():
        if not runs:
            continue
        arr = np.vstack(runs)                                                       # [contexts, grid]
        colour, ls = styles[label]
        ax.fill_between(grid, np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0), color=colour,
                        alpha=style.BAND_ALPHA, lw=0)
        ax.plot(grid, np.median(arr, axis=0), color=colour, ls=ls, lw=style.LINE_WIDTH,
                label=label if not label.startswith("adam") else f"Adam, lr {label.split('lr')[1]}")
    pr = st[(st["arm"] == "prod_s0") & (st["dataset"] == "base") & (st["feasible"] > 0)]
    pr_t = pr.groupby(keys[:4])["seconds"].sum().median()
    pr_g = pr[pr["selected_eval"] > 0]["nll_gap_nats"].median()
    ax.scatter([pr_t], [pr_g], marker="X", s=40, color=style.ANCHOR_GP, zorder=6, label="5 restarts, production pick (median)")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-2)
    for nats in (0.01, 1.0):
        ax.axhline(nats, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":")
    ax.set_xlabel(style.axis_label("fit_seconds"))
    ax.set_ylabel(style.axis_label("nlml_gap"))
    ax.set_ylim(bottom=0)
    ax.set_title(f"{len(contexts)} contexts (median, IQR)", fontsize=style.FONT_SIZES["legend"])
    ax.legend(loc="upper right", fontsize=style.FONT_SIZES["legend"] - 2)

    rule_ls = {"marginal NLML": "-", "production rule": "--"}
    ax = axes[1][0]
    for rule, ls in rule_ls.items():
        d = lad_tab.loc[rule]
        for i, tl in enumerate(LADDER_TOLS):
            ax.plot(d.index, d[f"share_gap_le_{tl:g}"], color=style.derive_shade(style.ANCHOR_GP, i, len(LADDER_TOLS)),
                    ls=ls, lw=style.LINE_WIDTH, label=f"{rule}, $\\leq$ {tl:g} nats" if i in (0, 2) else None)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(style.axis_label("restarts"))
    ax.set_ylabel(style.axis_label("share_within_tol"))
    ax.legend(loc="lower right", fontsize=style.FONT_SIZES["legend"] - 2)

    ax = axes[1][1]
    for rule, ls in rule_ls.items():
        d = lad_tab.loc[rule]
        ax.plot(d.index, d["dmu_median"], color=style.ANCHOR_GP, ls=ls, lw=style.LINE_WIDTH, label=f"{rule}, median")
        ax.plot(d.index, d["dmu_p90"], color=style.derive_shade(style.ANCHOR_GP, 2, 3), ls=ls, lw=style.LINE_WIDTH / 2,
                label=f"{rule}, p90")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.axhline(0.01, color=style.NEUTRAL_GREY, lw=style.LINE_WIDTH / 2, ls=":")
    ax.set_xlabel(style.axis_label("restarts"))
    ax.set_ylabel(style.axis_label("dmu_sd"))
    ax.set_ylim(bottom=0)
    ax.legend(loc="center right", fontsize=style.FONT_SIZES["legend"] - 2)
    style.panel_letters([axes[0][0], axes[0][1], axes[1][0], axes[1][1]])
    style.save_figure(fig, args.out, "gp_convergence", formats=("svg", "png"))

    # the selection rule: evaluation-mode loss against the marginal likelihood, per start
    f2 = st[(st["arm"] == "best") & (st["feasible"] > 0)].copy()
    f2["train_gap"] = f2["train_nll_total"] - f2.groupby(keys)["train_nll_total"].transform("min")
    f2["eval_gap"] = f2["eval_nll_total"] - f2.groupby(keys)["eval_nll_total"].transform("min")
    fig, axes = style.figure("double", nrows=1, ncols=2, panel_aspect=0.9, squeeze=False, layout=style.LAYOUT_ENGINE)
    ax = axes[0][0]
    sc = ax.scatter(f2["train_gap"] + 1e-3, f2["eval_gap"] + 1e-3, c=np.log10(f2["noise"]), s=3,
                    cmap=style.SEQUENTIAL_CMAP, alpha=0.5, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.colorbar(sc, ax=ax).set_label("log$_{10}$ fitted noise variance")
    ax.set_xlabel("Marginal NLML minus the best start (nats)")
    ax.set_ylabel("Evaluation-mode loss minus the best start (nats)")
    ax.set_title("Every start of every dataset", fontsize=style.FONT_SIZES["legend"])
    ax = axes[0][1]
    picked = f2[f2["selected_eval"] > 0]
    ax.hist(np.log10(picked["train_gap"] + 1e-4), bins=40, color=style.ANCHOR_GP, alpha=0.8)
    ax.set_xlabel("log$_{10}$ NLML gap of the pick (nats)")
    ax.set_ylabel("Datasets")
    ax.set_title(f"{float((picked['train_gap'] > 0.01).mean()):.0%} of picks are > 0.01 nats off",
                 fontsize=style.FONT_SIZES["legend"])
    style.panel_letters(list(axes[0]))
    style.save_figure(fig, args.out, "gp_selection_rule", formats=("svg", "png"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(lad_tab.round(4).to_string())
    print(tol.round(3).to_string())


def main() -> None:
    """Parse the CLI and trace or draw."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="configs/experiment/mechanism_bayes_update_nhp.yaml")
    ap.add_argument("--level", type=float, default=1.0)
    ap.add_argument("--ts", type=int, nargs="+", default=[30, 40, 50, 80])
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--best-restarts", type=int, default=20)
    ap.add_argument("--best-steps", type=int, default=100)
    ap.add_argument("--adam-steps", type=int, default=1000)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default="output/diagnostics/gp_optimizer")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    if args.figures_only:
        figures(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
