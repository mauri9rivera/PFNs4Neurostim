"""
Kappa schedule alpha ablation for Vanilla TabPFN BO vs GP.

Tests whether the convergence failure of TabPFN relative to GP is an
acquisition-function-side problem (wrong alpha) or a surrogate-side problem
(TabPFN's uncertainty doesn't contract with context size).

Regret is computed from the model's own recommendation (argmax of posterior
mean over explored points), not from the cumulative maximum of observed values.
This reflects how well the model *believes* it has found the optimum at each step.

Usage:
    cd PFNs4Neurostim
    python scripts/kappa_ablation.py
"""
import math
import os
import sys
import time
import random
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
from tabpfn import TabPFNRegressor

# ── Config ────────────────────────────────────────────────────────────────────
ALPHAS      : list[float] = [0.1, 0.25, 0.5, 0.75, 1.0]
N_REPS      : int         = 10
BUDGET      : int         = 100
N_INIT      : int         = 5
SUBJECT_IDX : int         = 0
DATASET     : str         = "nhp"
SEED           : int   = 42
TABPFN_DEVICE  : str   = "cuda" if torch.cuda.is_available() else "cpu"
GP_DEVICE      : str   = "cpu"
TABPFN_WORKERS : int   = 1                          # serialize GPU jobs to avoid OOM
GP_WORKERS     : int   = min(4, os.cpu_count() or 4)  # parallel CPU workers
OUTPUT_DIR     : Path  = ROOT / "output" / "kappa_ablation"

# ── GPU sharing between TabPFN and GP ─────────────────────────────────────────
# Set by main() once all TabPFN futures have completed; GP workers poll this
# before deciding whether to claim the GPU.
_tabpfn_done: threading.Event     = threading.Event()
# Serializes GP jobs that do run on GPU (at most one at a time).
_gp_gpu_sem:  threading.Semaphore = threading.Semaphore(1)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def run_single(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    rep_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one TabPFN BO repetition with given kappa_alpha; return (regret, rec_score).

    Regret uses the model's recommended best point (argmax of posterior mean
    over explored points) as running_best, not the cumulative max of observed
    values. This is non-monotonic and reflects the model's belief about the
    optimum at each step.

    Args:
        X_pool: Electrode coordinates, shape (n_locs, 2).
        y_pool: Standardized noisy responses, shape (n_locs, n_reps).
        y_test: Raw mean responses (µV), shape (n_locs,).  Used for regret.
        alpha: Cosine-annealing amplitude parameter (0.5 = default schedule).
        rep_seed: Per-rep random seed for init randomness.

    Returns:
        regret:    Normalized simple regret at each BO step, shape (n_steps,).
        rec_score: Recommendation quality (fraction of optimum), shape (n_steps,).
    """
    np.random.seed(rep_seed)
    torch.manual_seed(rep_seed)

    model = TabPFNRegressor(device=TABPFN_DEVICE)

    _, _, rec_values, _, _, _, _ = run_finetunedbo_loop(
        X_pool=X_pool,
        y_pool=y_pool,
        x_test=X_pool,
        y_test=y_test,
        model=model,
        n_init=N_INIT,
        budget=BUDGET,
        device=TABPFN_DEVICE,
        kappa_alpha=alpha,
    )

    y_opt = float(np.max(y_test))
    if y_opt <= 0:
        y_opt = 1.0

    # running_best = model's recommended point at each step (non-monotonic)
    running_best = np.array(rec_values)           # (n_steps,)
    regret    = (y_opt - running_best) / y_opt    # (n_steps,)
    rec_score = running_best / y_opt              # (n_steps,)  == 1 - regret

    return regret, rec_score


def run_single_gp(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    rep_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one GP BO repetition with given kappa_alpha; return (regret, rec_score).

    Mirrors run_single but uses the GP surrogate. GP is constructed internally
    by run_gpbo_loop — no model argument required.

    Device selection is dynamic: GP runs on CPU while TabPFN jobs are active.
    Once all TabPFN jobs finish (_tabpfn_done is set), each GP worker tries to
    claim the GPU semaphore (_gp_gpu_sem). The first to succeed runs on GPU;
    the rest fall back to CPU, serializing GPU access across GP workers.

    Args:
        X_pool: Electrode coordinates, shape (n_locs, 2).
        y_pool: Standardized noisy responses, shape (n_locs, n_reps).
        y_test: Raw mean responses (µV), shape (n_locs,).  Used for regret.
        alpha: Cosine-annealing amplitude parameter passed to kappa schedule.
        rep_seed: Per-rep random seed for init randomness.

    Returns:
        regret:    Normalized simple regret at each BO step, shape (n_steps,).
        rec_score: Recommendation quality (fraction of optimum), shape (n_steps,).
    """
    # Cap PyTorch's intraop thread count to 1 so that GP_WORKERS concurrent workers
    # don't each spawn cpu_count threads — that would cause severe oversubscription.
    torch.set_num_threads(1)

    np.random.seed(rep_seed)
    torch.manual_seed(rep_seed)

    # Claim GPU only after TabPFN is done and CUDA is actually a different device
    cuda_free = TABPFN_DEVICE != GP_DEVICE and _tabpfn_done.is_set()
    on_gpu    = cuda_free and _gp_gpu_sem.acquire(blocking=False)
    device    = TABPFN_DEVICE if on_gpu else GP_DEVICE

    try:
        _, _, rec_values, _, _, _, _ = run_gpbo_loop(
            X_pool=X_pool,
            y_pool=y_pool,
            x_test=X_pool,
            y_test=y_test,
            n_init=N_INIT,
            budget=BUDGET,
            device=device,
            kappa_alpha=alpha,
        )
    finally:
        if on_gpu:
            _gp_gpu_sem.release()

    y_opt = float(np.max(y_test))
    if y_opt <= 0:
        y_opt = 1.0

    running_best = np.array(rec_values)
    regret    = (y_opt - running_best) / y_opt
    rec_score = running_best / y_opt

    return regret, rec_score


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _tabpfn_done.clear()   # reset in case main() is called more than once

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {DATASET} subject {SUBJECT_IDX} ...")
    data = load_data(DATASET, SUBJECT_IDX)

    n_emg: int = data["sorted_resp"].shape[1]
    print(f"  {n_emg} EMG channels, {data['sorted_resp'].shape[0]} electrode locations")

    channels: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for emg_idx in range(n_emg):
        X_pool, Y_train, _, y_test, _ = preprocess_neural_data(
            data, emg_idx=emg_idx, normalization="pfn"
        )
        if np.max(y_test) <= 0:
            print(f"  Skipping EMG {emg_idx}: non-positive optimum")
            continue
        channels.append((X_pool, Y_train, y_test))
    n_ch = len(channels)
    print(f"  Using {n_ch} valid EMG channels")

    # ── Results storage — indexed [alpha][ch_idx] ─────────────────────────────
    tabpfn_results: dict[float, list[dict[str, list]]] = {
        a: [{"regret": [], "rec": []} for _ in range(n_ch)] for a in ALPHAS
    }
    gp_results: dict[float, list[dict[str, list]]] = {
        a: [{"regret": [], "rec": []} for _ in range(n_ch)] for a in ALPHAS
    }

    n_steps   = BUDGET - N_INIT
    n_jobs    = len(ALPHAS) * N_REPS * n_ch   # jobs per model
    print(
        f"\nDispatching {n_jobs} TabPFN jobs (device={TABPFN_DEVICE}, "
        f"workers={TABPFN_WORKERS}) and {n_jobs} GP jobs "
        f"(device={GP_DEVICE}→{TABPFN_DEVICE} after TabPFN, workers={GP_WORKERS}) "
        f"in parallel ..."
    )

    # ── Build flat job list ───────────────────────────────────────────────────
    job_specs: list[tuple] = [
        (alpha, rep, ch_idx, X_pool, y_pool, y_test, SEED + rep * 1000 + ch_idx)
        for alpha in ALPHAS
        for rep in range(N_REPS)
        for ch_idx, (X_pool, y_pool, y_test) in enumerate(channels)
    ]

    # ── Run TabPFN (GPU) and GP (CPU) pools concurrently ─────────────────────
    # TABPFN_WORKERS=1 serializes GPU jobs to avoid VRAM exhaustion.
    # GP_WORKERS runs multiple CPU jobs in parallel.
    # Both pools are live at the same time, so GPU and CPU work overlaps.
    pfn_done = 0
    gp_done  = 0

    with ThreadPoolExecutor(max_workers=TABPFN_WORKERS) as pfn_pool, \
         ThreadPoolExecutor(max_workers=GP_WORKERS)     as gp_pool:

        pfn_futures: dict = {
            pfn_pool.submit(run_single, X_pool, y_pool, y_test, alpha, rep_seed):
                (alpha, rep, ch_idx)
            for alpha, rep, ch_idx, X_pool, y_pool, y_test, rep_seed in job_specs
        }
        gp_futures: dict = {
            gp_pool.submit(run_single_gp, X_pool, y_pool, y_test, alpha, rep_seed):
                (alpha, rep, ch_idx)
            for alpha, rep, ch_idx, X_pool, y_pool, y_test, rep_seed in job_specs
        }

        # Single merged loop — both counters update in real-time as jobs finish.
        # _tabpfn_done is set the moment the last TabPFN future completes, giving
        # queued GP workers the earliest possible chance to claim the GPU.
        pfn_future_set = set(pfn_futures.keys())
        all_futures    = {**pfn_futures, **gp_futures}

        for future in as_completed(all_futures):
            if future in pfn_future_set:
                alpha, rep, ch_idx = pfn_futures[future]
                pfn_done += 1
                try:
                    regret, rec_score = future.result()
                    tabpfn_results[alpha][ch_idx]["regret"].append(regret)
                    tabpfn_results[alpha][ch_idx]["rec"].append(rec_score)
                except Exception as exc:
                    print(f"\n  [WARN TabPFN] alpha={alpha} rep={rep} emg={ch_idx}: {exc}")
                if pfn_done == n_jobs:
                    _tabpfn_done.set()
                    print(f"\n  TabPFN complete — GPU handed off to remaining GP jobs.")
            else:
                alpha, rep, ch_idx = gp_futures[future]
                gp_done += 1
                try:
                    regret_gp, rec_gp = future.result()
                    gp_results[alpha][ch_idx]["regret"].append(regret_gp)
                    gp_results[alpha][ch_idx]["rec"].append(rec_gp)
                except Exception as exc:
                    print(f"\n  [WARN GP] alpha={alpha} rep={rep} emg={ch_idx}: {exc}")
            print(f"  TabPFN {pfn_done}/{n_jobs}  GP {gp_done}/{n_jobs}", end="\r")

    print()  # newline after final \r

    # ── Compute statistics per (alpha, ch_idx) ────────────────────────────────
    # tabpfn_stats[alpha] = list of length n_ch; each element is a stats dict or None
    tabpfn_stats: dict[float, list[dict | None]] = {a: [] for a in ALPHAS}
    gp_stats:     dict[float, list[dict | None]] = {a: [] for a in ALPHAS}

    for a in ALPHAS:
        for ch_idx in range(n_ch):
            for dest, src in [(tabpfn_stats, tabpfn_results), (gp_stats, gp_results)]:
                runs_r = src[a][ch_idx]["regret"]
                runs_c = src[a][ch_idx]["rec"]
                if not runs_r:
                    print(f"[WARN] No successful runs for alpha={a} ch={ch_idx}")
                    dest[a].append(None)
                    continue
                arr_r = np.stack(runs_r)   # (n_runs, n_steps)
                arr_c = np.stack(runs_c)
                n = arr_r.shape[0]
                dest[a].append({
                    "regret_mean": arr_r.mean(axis=0),
                    "regret_sem":  arr_r.std(axis=0) / math.sqrt(n),
                    "rec_mean":    arr_c.mean(axis=0),
                    "rec_sem":     arr_c.std(axis=0) / math.sqrt(n),
                    "n_runs": n,
                })

    # ── Plot: 2 rows × n_ch columns ───────────────────────────────────────────
    palette     = sns.color_palette("coolwarm", n_colors=len(ALPHAS))
    query_steps = np.arange(N_INIT + 1, BUDGET + 1)   # 1-indexed query index

    n_cols = n_ch
    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 8),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Kappa schedule alpha ablation — TabPFN (solid) vs GP (dashed)\n"
        f"NHP subject {SUBJECT_IDX}, {n_ch} EMGs, {N_REPS} reps each",
        fontsize=11,
    )

    legend_handles: list = []

    for ch_idx in range(n_ch):
        ax_regret = axes[0, ch_idx]
        ax_rec    = axes[1, ch_idx]

        ax_regret.set_title(f"EMG {ch_idx}", fontsize=10)

        for i, alpha in enumerate(ALPHAS):
            color       = palette[i]
            kappa_start = 1.0 + 2 * alpha * 4.0
            label       = f"α={alpha:.2f}  (κ₀≈{kappa_start:.1f})"

            s_pfn = tabpfn_stats[alpha][ch_idx]
            s_gp  = gp_stats[alpha][ch_idx]

            # ── Regret row ────────────────────────────────────────────────────
            if s_pfn is not None:
                line_pfn, = ax_regret.plot(
                    query_steps, s_pfn["regret_mean"],
                    color=color, linewidth=1.8, linestyle="-",
                    label=label,
                )
                ax_regret.fill_between(
                    query_steps,
                    s_pfn["regret_mean"] - s_pfn["regret_sem"],
                    s_pfn["regret_mean"] + s_pfn["regret_sem"],
                    color=color, alpha=0.15,
                )
            if s_gp is not None:
                ax_regret.plot(
                    query_steps, s_gp["regret_mean"],
                    color=color, linewidth=1.8, linestyle="--",
                )

            # ── Rec score row ─────────────────────────────────────────────────
            if s_pfn is not None:
                ax_rec.plot(
                    query_steps, s_pfn["rec_mean"],
                    color=color, linewidth=1.8, linestyle="-",
                )
                ax_rec.fill_between(
                    query_steps,
                    s_pfn["rec_mean"] - s_pfn["rec_sem"],
                    s_pfn["rec_mean"] + s_pfn["rec_sem"],
                    color=color, alpha=0.15,
                )
            if s_gp is not None:
                ax_rec.plot(
                    query_steps, s_gp["rec_mean"],
                    color=color, linewidth=1.8, linestyle="--",
                )

            # Collect legend handles once (from first column)
            if ch_idx == 0 and s_pfn is not None:
                legend_handles.append(line_pfn)

        # Formatting
        ax_regret.set_ylim(bottom=0)
        ax_regret.grid(True, alpha=0.3)
        ax_rec.set_ylim(0, 1.05)
        ax_rec.set_xlabel("Query index", fontsize=9)
        ax_rec.grid(True, alpha=0.3)

        if ch_idx == 0:
            ax_regret.set_ylabel("Normalized simple regret\n(model recommendation)", fontsize=9)
            ax_rec.set_ylabel("Fraction of optimum recovered", fontsize=9)

    # ── Kappa schedule inset (bottom-right panel) ─────────────────────────────
    ax_inset = axes[1, -1].inset_axes([0.52, 0.05, 0.44, 0.40])
    t_vals = np.linspace(0, n_steps - 1, 200)
    for i, alpha in enumerate(ALPHAS):
        kappa_vals = [
            1.0 + alpha * 4.0 * (1 + math.cos(math.pi * t / n_steps))
            for t in t_vals
        ]
        ax_inset.plot(t_vals, kappa_vals, color=palette[i], linewidth=1.2)
    ax_inset.set_title("κ schedules", fontsize=7)
    ax_inset.set_xlabel("BO step t", fontsize=6)
    ax_inset.set_ylabel("κ", fontsize=6)
    ax_inset.tick_params(labelsize=5)
    ax_inset.grid(True, alpha=0.2)

    # ── Shared legend (solid = TabPFN, dashed = GP) ───────────────────────────
    # Add proxy artists for linestyle legend entries
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-",  label="TabPFN"),
        Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="GP"),
    ]
    fig.legend(
        handles=legend_handles + style_handles,
        loc="lower center",
        ncol=len(ALPHAS) + 2,
        fontsize=8,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    svg_path = OUTPUT_DIR / "regret_rec_per_emg.svg"
    png_path = OUTPUT_DIR / "regret_rec_per_emg.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    print(f"\nSaved:\n  {svg_path}\n  {png_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Final-step summary (mean ± SEM across reps) ──")
    header = f"{'alpha':>6}  {'EMG':>4}  {'model':>6}  {'final_regret':>14}  {'final_rec':>12}  {'n_runs':>6}"
    print(header)
    print("─" * len(header))
    for alpha in ALPHAS:
        kappa_start = 1.0 + 2 * alpha * 4.0
        for ch_idx in range(n_ch):
            for label, stats_dict in [("TabPFN", tabpfn_stats), ("GP", gp_stats)]:
                s = stats_dict[alpha][ch_idx]
                if s is None:
                    continue
                fr    = s["regret_mean"][-1]
                fr_se = s["regret_sem"][-1]
                rec   = s["rec_mean"][-1]
                rec_se = s["rec_sem"][-1]
                print(
                    f"{alpha:>6.2f}  {ch_idx:>4}  {label:>6}  "
                    f"{fr:>6.4f} ± {fr_se:.4f}  "
                    f"{rec:>5.4f} ± {rec_se:.4f}  "
                    f"{s['n_runs']:>6}"
                )


if __name__ == "__main__":
    main()
