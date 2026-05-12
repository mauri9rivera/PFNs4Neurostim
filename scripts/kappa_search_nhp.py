"""
Kappa value search for Vanilla TabPFN BO vs GP on all NHP subjects.

Tests whether a fixed UCB exploration coefficient or the auto cosine-annealed schedule
produces better BO performance on NHP neurostimulation data.  Compares seven fixed kappa
values against the cosine baseline (kappa_schedule=0.0, default alpha=0.5) for both
TabPFN and GP surrogates.

Plot columns = NHP subjects (0, 1, 3), with EMG channels averaged within each subject.
Corrects a bug in kappa_ablation.py: recommendation values are extracted from
best_rec_indices (index 6 of the 7-tuple return, argmax of posterior mean at each step)
rather than real_values (index 2, observed ground-truth at queried sites).

Usage:
    cd PFNs4Neurostim
    python scripts/kappa_search_nhp.py
"""
from __future__ import annotations  # PEP 563 — lets Python 3.9 handle `X | Y` annotations
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.data_utils import load_data, preprocess_neural_data
from utils.bo_loops import run_finetunedbo_loop, run_gpbo_loop
from utils.gpbo_utils import compute_ucb_kappa, _auto_kappa_max, _auto_kappa_min
from tabpfn import TabPFNRegressor

# ── Config ────────────────────────────────────────────────────────────────────
KAPPAS          : list[float] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
COSINE_KAPPA    : float       = 0.0      # kappa_schedule=0.0 → auto cosine, alpha=0.5
ALL_KAPPAS      : list[float] = KAPPAS + [COSINE_KAPPA]
N_REPS          : int         = 10
BUDGET          : int         = 100
N_INIT          : int         = 5
DATASET         : str         = "nhp"
ALL_NHP_SUBJECTS: list[int]   = [0, 1, 3]
SEED            : int         = 42
TABPFN_DEVICE   : str         = "cuda" if torch.cuda.is_available() else "cpu"
GP_DEVICE       : str         = "cpu"
TABPFN_WORKERS  : int         = 1                           # serialize GPU jobs to avoid OOM
GP_WORKERS      : int         = min(4, os.cpu_count() or 4)
OUTPUT_DIR      : Path        = ROOT / "output" / "kappa_search"

# ── GPU sharing between TabPFN and GP ─────────────────────────────────────────
_tabpfn_done: threading.Event     = threading.Event()
_gp_gpu_sem:  threading.Semaphore = threading.Semaphore(1)

# ── Reproducibility ───────────────────────────────────────────────────────────
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _extract_rec_values(best_rec_indices: list[int], y_test: np.ndarray) -> np.ndarray:
    """Extract ground-truth values at each step's recommended location.

    Args:
        best_rec_indices: Per-step argmax-of-mean indices, shape (n_steps,).
        y_test: Ground-truth per-site mean responses, shape (n_locs,).

    Returns:
        Array of recommendation values, shape (n_steps,).
    """
    return y_test[np.array(best_rec_indices, dtype=int)]


def run_single(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    y_test: np.ndarray,
    kappa_schedule: float,
    rep_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one TabPFN BO repetition; return (regret, rec_score).

    Args:
        X_pool: Electrode coordinates, shape (n_locs, 2).
        y_pool: Standardized noisy responses, shape (n_locs, n_reps).
        y_test: Raw mean responses (µV), shape (n_locs,).
        kappa_schedule: 0.0 = auto cosine schedule; else fixed kappa value.
        rep_seed: Per-rep random seed.

    Returns:
        regret:    Normalized simple regret (model recommendation), shape (n_steps,).
        rec_score: Fraction of optimum recovered, shape (n_steps,).
    """
    np.random.seed(rep_seed)
    torch.manual_seed(rep_seed)

    model = TabPFNRegressor(device=TABPFN_DEVICE)

    _, _, _, _, _, _, best_rec_indices = run_finetunedbo_loop(
        X_pool=X_pool,
        y_pool=y_pool,
        x_test=X_pool,
        y_test=y_test,
        model=model,
        n_init=N_INIT,
        budget=BUDGET,
        device=TABPFN_DEVICE,
        kappa_schedule=kappa_schedule,
    )

    running_best = _extract_rec_values(best_rec_indices, y_test)   # (n_steps,)
    y_opt = max(float(np.max(y_test)), 1e-6)
    regret    = (y_opt - running_best) / y_opt
    rec_score = running_best / y_opt
    return regret, rec_score


def run_single_gp(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    y_test: np.ndarray,
    kappa_schedule: float,
    rep_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one GP BO repetition; return (regret, rec_score).

    Device selection mirrors kappa_ablation.py: GP runs on CPU while TabPFN jobs are
    active.  Once all TabPFN jobs finish (_tabpfn_done is set), each GP worker tries to
    claim the GPU semaphore (_gp_gpu_sem).

    Args:
        X_pool: Electrode coordinates, shape (n_locs, 2).
        y_pool: Standardized noisy responses, shape (n_locs, n_reps).
        y_test: Raw mean responses (µV), shape (n_locs,).
        kappa_schedule: 0.0 = auto cosine schedule; else fixed kappa value.
        rep_seed: Per-rep random seed.

    Returns:
        regret:    Normalized simple regret (model recommendation), shape (n_steps,).
        rec_score: Fraction of optimum recovered, shape (n_steps,).
    """
    torch.set_num_threads(1)
    np.random.seed(rep_seed)
    torch.manual_seed(rep_seed)

    cuda_free = TABPFN_DEVICE != GP_DEVICE and _tabpfn_done.is_set()
    on_gpu    = cuda_free and _gp_gpu_sem.acquire(blocking=False)
    device    = TABPFN_DEVICE if on_gpu else GP_DEVICE

    try:
        _, _, _, _, _, _, best_rec_indices = run_gpbo_loop(
            X_pool=X_pool,
            y_pool=y_pool,
            x_test=X_pool,
            y_test=y_test,
            n_init=N_INIT,
            budget=BUDGET,
            device=device,
            kappa_schedule=kappa_schedule,
        )
    finally:
        if on_gpu:
            _gp_gpu_sem.release()

    running_best = _extract_rec_values(best_rec_indices, y_test)   # (n_steps,)
    y_opt = max(float(np.max(y_test)), 1e-6)
    regret    = (y_opt - running_best) / y_opt
    rec_score = running_best / y_opt
    return regret, rec_score


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _tabpfn_done.clear()

    # ── Load all NHP subjects ──────────────────────────────────────────────────
    print(f"Loading NHP subjects {ALL_NHP_SUBJECTS} ...")
    channels: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        s: [] for s in ALL_NHP_SUBJECTS
    }
    for subj_idx in ALL_NHP_SUBJECTS:
        data = load_data(DATASET, subj_idx)
        n_emg: int = data["sorted_resp"].shape[1]
        valid = 0
        for emg_idx in range(n_emg):
            X_pool, Y_train, _, y_test, _ = preprocess_neural_data(
                data, emg_idx=emg_idx, normalization="pfn"
            )
            if np.max(y_test) <= 0:
                continue
            channels[subj_idx].append((X_pool, Y_train, y_test))
            valid += 1
        print(f"  Subject {subj_idx}: {valid}/{n_emg} valid EMG channels")

    # ── Results storage — indexed [kappa][subj_idx] ───────────────────────────
    # Each leaf: list of dicts {'regret': [array, ...], 'rec': [array, ...]}
    # One dict per EMG channel; N_REPS arrays per dict after collection.
    tabpfn_results: dict[float, dict[int, list[dict[str, list]]]] = {
        k: {s: [{"regret": [], "rec": []} for _ in channels[s]] for s in ALL_NHP_SUBJECTS}
        for k in ALL_KAPPAS
    }
    gp_results: dict[float, dict[int, list[dict[str, list]]]] = {
        k: {s: [{"regret": [], "rec": []} for _ in channels[s]] for s in ALL_NHP_SUBJECTS}
        for k in ALL_KAPPAS
    }

    n_steps = BUDGET - N_INIT

    # ── Build flat job list ───────────────────────────────────────────────────
    job_specs: list[tuple] = [
        (kappa, rep, subj_idx, ch_idx, X_pool, y_pool, y_test,
         SEED + rep * 10000 + subj_idx * 1000 + ch_idx)
        for kappa in ALL_KAPPAS
        for rep in range(N_REPS)
        for subj_idx in ALL_NHP_SUBJECTS
        for ch_idx, (X_pool, y_pool, y_test) in enumerate(channels[subj_idx])
    ]
    n_jobs = len(job_specs)
    print(
        f"\nDispatching {n_jobs} TabPFN jobs (device={TABPFN_DEVICE}, "
        f"workers={TABPFN_WORKERS}) and {n_jobs} GP jobs "
        f"(device={GP_DEVICE}→{TABPFN_DEVICE} after TabPFN, workers={GP_WORKERS}) ..."
    )

    pfn_done = 0
    gp_done  = 0

    with ThreadPoolExecutor(max_workers=TABPFN_WORKERS) as pfn_pool, \
         ThreadPoolExecutor(max_workers=GP_WORKERS)     as gp_pool:

        pfn_futures: dict = {
            pfn_pool.submit(run_single, X_pool, y_pool, y_test, kappa, rep_seed):
                (kappa, rep, subj_idx, ch_idx)
            for kappa, rep, subj_idx, ch_idx, X_pool, y_pool, y_test, rep_seed in job_specs
        }
        gp_futures: dict = {
            gp_pool.submit(run_single_gp, X_pool, y_pool, y_test, kappa, rep_seed):
                (kappa, rep, subj_idx, ch_idx)
            for kappa, rep, subj_idx, ch_idx, X_pool, y_pool, y_test, rep_seed in job_specs
        }

        pfn_future_set = set(pfn_futures.keys())
        all_futures    = {**pfn_futures, **gp_futures}

        for future in as_completed(all_futures):
            if future in pfn_future_set:
                kappa, rep, subj_idx, ch_idx = pfn_futures[future]
                pfn_done += 1
                try:
                    regret, rec_score = future.result()
                    tabpfn_results[kappa][subj_idx][ch_idx]["regret"].append(regret)
                    tabpfn_results[kappa][subj_idx][ch_idx]["rec"].append(rec_score)
                except Exception as exc:
                    print(
                        f"\n  [WARN TabPFN] kappa={kappa} subj={subj_idx} "
                        f"emg={ch_idx} rep={rep}: {exc}"
                    )
                if pfn_done == n_jobs:
                    _tabpfn_done.set()
                    print(f"\n  TabPFN complete — GPU handed off to remaining GP jobs.")
            else:
                kappa, rep, subj_idx, ch_idx = gp_futures[future]
                gp_done += 1
                try:
                    regret_gp, rec_gp = future.result()
                    gp_results[kappa][subj_idx][ch_idx]["regret"].append(regret_gp)
                    gp_results[kappa][subj_idx][ch_idx]["rec"].append(rec_gp)
                except Exception as exc:
                    print(
                        f"\n  [WARN GP] kappa={kappa} subj={subj_idx} "
                        f"emg={ch_idx} rep={rep}: {exc}"
                    )
            print(f"  TabPFN {pfn_done}/{n_jobs}  GP {gp_done}/{n_jobs}", end="\r")

    print()

    # ── Aggregate: average (rep, emg) pairs within each (kappa, subj_idx) ─────
    def _compute_stats(
        results: dict[float, dict[int, list[dict[str, list]]]],
        label: str = "",
    ) -> dict[float, dict[int, dict | None]]:
        stats: dict[float, dict[int, dict | None]] = {k: {} for k in ALL_KAPPAS}
        for k in ALL_KAPPAS:
            for s in ALL_NHP_SUBJECTS:
                all_regret: list[np.ndarray] = []
                all_rec:    list[np.ndarray] = []
                for ch_data in results[k][s]:
                    all_regret.extend(ch_data["regret"])
                    all_rec.extend(ch_data["rec"])
                # Filter out any arrays with unexpected length to guard against np.stack failure.
                good_regret = [r for r in all_regret if r.shape == (n_steps,)]
                good_rec    = [r for r in all_rec    if r.shape == (n_steps,)]
                n_dropped = len(all_regret) - len(good_regret)
                if n_dropped:
                    print(
                        f"[WARN{' ' + label if label else ''}] Dropped {n_dropped} "
                        f"shape-mismatched arrays for kappa={k} subj={s} "
                        f"(expected ({n_steps},))"
                    )
                if not good_regret:
                    print(f"[WARN] No successful runs for kappa={k} subj={s}")
                    stats[k][s] = None
                    continue
                arr_r = np.stack(good_regret)   # (n_samples, n_steps)
                arr_c = np.stack(good_rec)
                n = arr_r.shape[0]
                stats[k][s] = {
                    "regret_mean": arr_r.mean(axis=0),
                    "regret_sem":  arr_r.std(axis=0) / math.sqrt(n),
                    "rec_mean":    arr_c.mean(axis=0),
                    "rec_sem":     arr_c.std(axis=0) / math.sqrt(n),
                    "n_samples": n,
                }
        return stats

    tabpfn_stats = _compute_stats(tabpfn_results, label="TabPFN")
    gp_stats     = _compute_stats(gp_results,     label="GP")

    # ── Shadow validation: verify every non-None stats entry has the right shape ─
    def _shadow_validate(
        stats_dict: dict[float, dict[int, dict | None]],
        label: str,
    ) -> None:
        """Raise ValueError if any stats array has length != n_steps."""
        for k in ALL_KAPPAS:
            for s in ALL_NHP_SUBJECTS:
                st = stats_dict[k][s]
                if st is None:
                    continue
                for key in ("regret_mean", "regret_sem", "rec_mean", "rec_sem"):
                    arr = st[key]
                    if arr.shape != (n_steps,):
                        raise ValueError(
                            f"[{label}] stats[{k}][{s}]['{key}'] has shape "
                            f"{arr.shape}, expected ({n_steps},) — plot would fail."
                        )

    _shadow_validate(tabpfn_stats, "TabPFN")
    _shadow_validate(gp_stats,     "GP")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_cols = len(ALL_NHP_SUBJECTS)
    n_rows = 2
    palette = sns.color_palette("coolwarm", n_colors=len(KAPPAS))
    COSINE_COLOR = "black"
    query_steps = np.arange(N_INIT + 1, BUDGET + 1)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 8),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        "Kappa value search — TabPFN (solid) vs GP (dashed)\n"
        f"NHP subjects {ALL_NHP_SUBJECTS}, EMGs averaged per subject, {N_REPS} reps",
        fontsize=11,
    )

    kappa_legend_handles: list = []

    for col, subj_idx in enumerate(ALL_NHP_SUBJECTS):
        ax_regret = axes[0, col]
        ax_rec    = axes[1, col]

        ax_regret.set_title(f"NHP Subject {subj_idx}", fontsize=10)

        # ── Fixed kappa values ────────────────────────────────────────────────
        for i, k in enumerate(KAPPAS):
            color = palette[i]
            label = f"κ={k:.1f}"

            s_pfn = tabpfn_stats[k][subj_idx]
            s_gp  = gp_stats[k][subj_idx]

            if s_pfn is not None:
                line_pfn, = ax_regret.plot(
                    query_steps, s_pfn["regret_mean"],
                    color=color, linewidth=1.6, linestyle="-",
                    label=label,
                )
                ax_regret.fill_between(
                    query_steps,
                    s_pfn["regret_mean"] - s_pfn["regret_sem"],
                    s_pfn["regret_mean"] + s_pfn["regret_sem"],
                    color=color, alpha=0.12,
                )
                ax_rec.plot(
                    query_steps, s_pfn["rec_mean"],
                    color=color, linewidth=1.6, linestyle="-",
                )
                ax_rec.fill_between(
                    query_steps,
                    s_pfn["rec_mean"] - s_pfn["rec_sem"],
                    s_pfn["rec_mean"] + s_pfn["rec_sem"],
                    color=color, alpha=0.12,
                )
                if col == 0:
                    kappa_legend_handles.append(line_pfn)
            if s_gp is not None:
                ax_regret.plot(
                    query_steps, s_gp["regret_mean"],
                    color=color, linewidth=1.6, linestyle="--",
                )
                ax_rec.plot(
                    query_steps, s_gp["rec_mean"],
                    color=color, linewidth=1.6, linestyle="--",
                )

        # ── Cosine schedule baseline ──────────────────────────────────────────
        s_pfn_cos = tabpfn_stats[COSINE_KAPPA][subj_idx]
        s_gp_cos  = gp_stats[COSINE_KAPPA][subj_idx]

        if s_pfn_cos is not None:
            line_cos, = ax_regret.plot(
                query_steps, s_pfn_cos["regret_mean"],
                color=COSINE_COLOR, linewidth=2.5, linestyle="-",
                label="cosine (auto)",
            )
            ax_regret.fill_between(
                query_steps,
                s_pfn_cos["regret_mean"] - s_pfn_cos["regret_sem"],
                s_pfn_cos["regret_mean"] + s_pfn_cos["regret_sem"],
                color=COSINE_COLOR, alpha=0.10,
            )
            ax_rec.plot(
                query_steps, s_pfn_cos["rec_mean"],
                color=COSINE_COLOR, linewidth=2.5, linestyle="-",
            )
            ax_rec.fill_between(
                query_steps,
                s_pfn_cos["rec_mean"] - s_pfn_cos["rec_sem"],
                s_pfn_cos["rec_mean"] + s_pfn_cos["rec_sem"],
                color=COSINE_COLOR, alpha=0.10,
            )
            if col == 0:
                kappa_legend_handles.append(line_cos)
        if s_gp_cos is not None:
            ax_regret.plot(
                query_steps, s_gp_cos["regret_mean"],
                color=COSINE_COLOR, linewidth=2.5, linestyle="--",
            )
            ax_rec.plot(
                query_steps, s_gp_cos["rec_mean"],
                color=COSINE_COLOR, linewidth=2.5, linestyle="--",
            )

        ax_regret.set_ylim(bottom=0)
        ax_regret.grid(True, alpha=0.3)
        ax_rec.set_ylim(0, 1.05)
        ax_rec.set_xlabel("Query index", fontsize=9)
        ax_rec.grid(True, alpha=0.3)

        if col == 0:
            ax_regret.set_ylabel(
                "Normalized simple regret\n(model recommendation)", fontsize=9
            )
            ax_rec.set_ylabel("Fraction of optimum recovered", fontsize=9)

    # ── Kappa schedule inset (bottom-right panel) ─────────────────────────────
    d       = 2                                 # NHP electrode coordinates are 2D
    k_max   = _auto_kappa_max(d, n_steps)
    k_min   = _auto_kappa_min(d, n_steps)
    t_vals  = np.linspace(0, n_steps - 1, 200)

    ax_inset = axes[1, -1].inset_axes([0.50, 0.04, 0.46, 0.40])
    for i, k in enumerate(KAPPAS):
        ax_inset.axhline(y=k, color=palette[i], linewidth=1.2, label=f"κ={k:.1f}")
    cosine_curve = [
        compute_ucb_kappa(t, n_steps, kappa_max=k_max, kappa_min=k_min)
        for t in t_vals
    ]
    ax_inset.plot(t_vals, cosine_curve, color=COSINE_COLOR, linewidth=2.0, label="cosine")
    ax_inset.set_title("κ schedules", fontsize=7)
    ax_inset.set_xlabel("BO step t", fontsize=6)
    ax_inset.set_ylabel("κ", fontsize=6)
    ax_inset.tick_params(labelsize=5)
    ax_inset.grid(True, alpha=0.2)

    # ── Shared legend ─────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-",  label="TabPFN"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="GP"),
    ]
    fig.legend(
        handles=kappa_legend_handles + style_handles,
        loc="lower center",
        ncol=len(KAPPAS) + 3,
        fontsize=8,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    svg_path = OUTPUT_DIR / "regret_rec_per_subject.svg"
    png_path = OUTPUT_DIR / "regret_rec_per_subject.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    print(f"\nSaved:\n  {svg_path}\n  {png_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Final-step summary (mean ± SEM across reps × EMGs) ──")
    header = (
        f"{'kappa':>8}  {'subj':>4}  {'model':>6}  "
        f"{'final_regret':>14}  {'final_rec':>12}  {'n_samples':>9}"
    )
    print(header)
    print("─" * len(header))
    for k in ALL_KAPPAS:
        k_label = "cosine" if k == COSINE_KAPPA else f"{k:.1f}"
        for s in ALL_NHP_SUBJECTS:
            for label, stats_dict in [("TabPFN", tabpfn_stats), ("GP", gp_stats)]:
                st = stats_dict[k][s]
                if st is None:
                    continue
                fr    = st["regret_mean"][-1]
                fr_se = st["regret_sem"][-1]
                rec   = st["rec_mean"][-1]
                rec_se = st["rec_sem"][-1]
                print(
                    f"{k_label:>8}  {s:>4}  {label:>6}  "
                    f"{fr:>6.4f} ± {fr_se:.4f}  "
                    f"{rec:>5.4f} ± {rec_se:.4f}  "
                    f"{st['n_samples']:>9}"
                )


if __name__ == "__main__":
    main()
