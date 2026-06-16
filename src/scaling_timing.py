"""A7 — Synthetic grid scaling timing harness.

Benchmarks GP vs TabPFN per-query wall-clock as electrode grid size n grows,
substantiating the O(n³) vs O(n²) complexity claim for the paper's scalability
argument.

Each timing trial represents one "oracle query" in a BO loop at a fixed
exploration state: fit the surrogate on n_context observed points, then
predict over n_pool remaining candidates.  GP always runs on CPU (its natural
device); TabPFN runs on the specified device.

Usage::

    python src/scaling_timing.py --save
    python src/scaling_timing.py --grid_sizes 100 250 500 1000 --n_reps 10 --device cuda --save
    python src/scaling_timing.py --n_context_frac 0.1 --gp_n_opt_steps 10 --save
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Ensure src/ is on sys.path when invoked from project root.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.regressors import GPSurrogate, TabPFNSurrogate
from tabpfn import TabPFNRegressor
from utils.visualization import plot_scaling_timing


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_synthetic_data(
    n: int,
    n_context: int,
    d: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random synthetic electrode grid for one timing trial.

    Samples ``n`` 2D coordinates uniformly from [0,1]^d and a corresponding
    response vector.  The first ``n_context`` points form the observed context;
    the remainder form the prediction pool.

    Args:
        n: Total grid size (context + pool).
        n_context: Number of observed training points.
        d: Feature dimensionality (2 for 2D electrode coordinates).
        rng: NumPy random generator for reproducible sampling.

    Returns:
        Tuple of ``(X_context [n_context, d], y_context [n_context],
        X_pool [n-n_context, d])``.
    """
    X = rng.uniform(0.0, 1.0, (n, d)).astype(np.float32)          # [n, d]
    y = rng.standard_normal(n_context).astype(np.float32)          # [n_context]
    X_context = X[:n_context]                                       # [n_context, d]
    # Predict over the full grid — mirrors run_bo_loop where X_pool is always
    # the complete candidate set, not just the unobserved remainder.
    X_pool = X                                                       # [n, d]
    return X_context, y, X_pool


def _time_query(
    surrogate: GPSurrogate | TabPFNSurrogate,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_pool: np.ndarray,
    acq_fn: str = 'ts',
    ts_temperature: float = 1.0,
    kappa: float = 2.0,
) -> float:
    """Time one BO-step cycle matching run_bo_loop's exact per-step sequence.

    Replicates ``run_bo_loop``'s per-step work:
      1. ``fit`` on observed context
      2. acquisition evaluation: ``predict_ts`` (TS) or ``predict_ucb`` (UCB)
      3. ``predict`` for the pool-mean exploitation recommendation

    For GP, step 2 is the dominant cost: ``predict_ts`` calls
    ``f_posterior.rsample()`` — a joint draw from the n_pool-dimensional MVN
    requiring an O(n_pool³) Cholesky of the posterior covariance.  This is the
    operation that exhibits cubic scaling with grid size.  Step 3 (marginal
    predictions) is O(n_pool × n_context) and cheap.

    For TabPFN, ``predict_ts`` / ``predict_ucb`` cache the bar-distribution
    logits, so step 3 hits the cache and costs essentially nothing extra.
    TabPFN's total cost is one transformer forward pass (step 2), matching
    what ``run_bo_loop`` measures in practice.

    Args:
        surrogate: A SurrogateModel instance (GPSurrogate or TabPFNSurrogate).
        X_context: Observed feature matrix, shape [n_context, d].  # [n_context, d]
        y_context: Observed response vector, shape [n_context].     # [n_context]
        X_pool: Prediction candidate matrix, shape [n_pool, d].    # [n_pool, d]
        acq_fn: Acquisition function to benchmark (``'ts'`` or ``'ucb'``).
        ts_temperature: Temperature for TabPFN bar-distribution TS sampling.
        kappa: UCB exploration coefficient (used when ``acq_fn='ucb'``).

    Returns:
        Elapsed wall-clock time in seconds.
    """
    t0 = time.perf_counter()
    surrogate.fit(X_context, y_context)
    if acq_fn == 'ts':
        # Joint posterior sample — O(n_pool³) Cholesky for GP (rsample);
        # one transformer forward pass for TabPFN (logits cached).
        surrogate.predict_ts(X_pool, temperature=ts_temperature)
    else:  # ucb
        # For GP: predict_ucb calls predict once (no cache); predict below is 2nd eval.
        # For TabPFN: predict_ucb caches logits; predict below hits cache (free).
        surrogate.predict_ucb(X_pool, kappa=kappa, t=0, n_steps=1)
    # pool_mean for exploitation recommendation — mirrors run_bo_loop exactly.
    surrogate.predict(X_pool)
    return time.perf_counter() - t0


def _time_query_with_timeout(
    surrogate: GPSurrogate | TabPFNSurrogate,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_pool: np.ndarray,
    acq_fn: str,
    ts_temperature: float,
    kappa: float,
    timeout: float,
) -> float:
    """Run ``_time_query`` in a thread; return NaN if it exceeds ``timeout`` seconds.

    Uses a ``ThreadPoolExecutor`` so BLAS/CUDA calls release the GIL and the
    wall-clock timeout is respected.  A timed-out GP thread continues running
    in the background but is abandoned — acceptable for a benchmark harness.

    Args:
        surrogate: SurrogateModel instance.
        X_context: Context feature matrix, shape [n_context, d].
        y_context: Context response vector, shape [n_context].
        X_pool: Full candidate grid, shape [n, d].
        acq_fn: Acquisition function (``'ts'`` or ``'ucb'``).
        ts_temperature: TS temperature for TabPFN.
        kappa: UCB kappa for GP.
        timeout: Per-rep wall-clock limit in seconds.  ``float('inf')`` disables.

    Returns:
        Elapsed time in seconds, or ``float('nan')`` on timeout.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _time_query, surrogate, X_context, y_context, X_pool,
            acq_fn, ts_temperature, kappa,
        )
        try:
            # Windows' threading.Condition.wait() cannot convert float('inf')
            # to a C timeout value, so pass None to mean "no timeout".
            return future.result(timeout=None if np.isinf(timeout) else timeout)
        except concurrent.futures.TimeoutError:
            return float('nan')


def run_scaling_benchmark(
    grid_sizes: list[int],
    n_context_frac: float = 0.1,
    n_reps: int = 15,
    d: int = 2,
    tabpfn_device: str = 'cpu',
    gp_n_opt_steps: int = 50,
    acq_fn: str = 'ts',
    ts_temperature: float = 1.0,
    kappa: float = 2.0,
    timeout_per_rep: float = float('inf'),
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark GP and TabPFN per-query wall-clock at each grid size.

    For each ``n`` in ``grid_sizes``, constructs a random D-dimensional
    synthetic grid and times one BO-step cycle (``fit + acquire + predict``)
    for both surrogates across ``n_reps`` independent trials.  The timed
    sequence mirrors ``run_bo_loop``'s per-step work exactly so the figure
    reflects true BO cost.

    GP always runs on CPU; TabPFN runs on ``tabpfn_device``.  A single
    ``TabPFNRegressor`` instance is reused across all trials so model weights
    are loaded only once.

    Args:
        grid_sizes: Total grid sizes to benchmark (e.g. ``[100, 250, 500, 1000, 2000]``).
        n_context_frac: Fraction of n used as training context.  Default 0.1
            simulates ≈10% of the grid explored at the query moment.
        n_reps: Timing repetitions per (model, n).
        d: Feature dimensionality (2 for 2D electrode coordinates).
        tabpfn_device: Inference device for TabPFN (``'cpu'`` or ``'cuda'``).
        gp_n_opt_steps: GP marginal-likelihood optimisation steps per fit call.
        acq_fn: Acquisition function to benchmark (``'ts'`` or ``'ucb'``).
            ``'ts'`` (default) is the primary acquisition used in the paper.
        ts_temperature: Temperature for TabPFN bar-distribution TS sampling.
        kappa: UCB exploration coefficient used when ``acq_fn='ucb'``.
        timeout_per_rep: Per-rep wall-clock limit in seconds.  If a single
            trial exceeds this, the remaining reps at that ``n`` are skipped
            and recorded as ``NaN``.  ``float('inf')`` (default) disables.
            Useful for large n where GP becomes infeasible.
        seed: Master random seed.

    Returns:
        DataFrame with columns ``['model', 'n', 'rep', 'time_s']``.

    Raises:
        ValueError: If ``acq_fn`` is not ``'ts'`` or ``'ucb'``.
    """
    if acq_fn not in ('ts', 'ucb'):
        raise ValueError(f"acq_fn must be 'ts' or 'ucb', got {acq_fn!r}.")
    set_seed(seed)
    rng = np.random.default_rng(seed)

    tabpfn_base = TabPFNRegressor(device=tabpfn_device)

    records: list[dict] = []
    for n in grid_sizes:
        n_context = max(5, int(round(n * n_context_frac)))
        n_context = min(n_context, n)  # cap at n — pool is always the full grid
        print(
            f"  n={n:5d}  n_context={n_context:4d}  n_pool={n:5d}",
            flush=True,
        )

        gp_timed_out = False
        pfn_timed_out = False
        for rep in range(n_reps):
            X_ctx, y_ctx, X_pool = _make_synthetic_data(n, n_context, d, rng)

            # --- GP (always CPU) ---
            if gp_timed_out:
                t_gp = float('nan')
            else:
                gp = GPSurrogate(device='cpu', n_opt_steps=gp_n_opt_steps)
                t_gp = _time_query_with_timeout(
                    gp, X_ctx, y_ctx, X_pool,
                    acq_fn=acq_fn, ts_temperature=ts_temperature, kappa=kappa,
                    timeout=timeout_per_rep,
                )
                del gp
                if np.isnan(t_gp):
                    gp_timed_out = True
                    print(f"    GP:     TIMEOUT at rep {rep} (>{timeout_per_rep:.0f}s) — skipping remaining reps", flush=True)
            records.append({'model': 'GP', 'n': n, 'rep': rep, 'time_s': t_gp})

            # --- TabPFN ---
            if pfn_timed_out:
                t_pfn = float('nan')
            else:
                pfn = TabPFNSurrogate(tabpfn_base)
                t_pfn = _time_query_with_timeout(
                    pfn, X_ctx, y_ctx, X_pool,
                    acq_fn=acq_fn, ts_temperature=ts_temperature, kappa=kappa,
                    timeout=timeout_per_rep,
                )
                if np.isnan(t_pfn):
                    pfn_timed_out = True
                    print(f"    TabPFN: TIMEOUT at rep {rep} (>{timeout_per_rep:.0f}s) — skipping remaining reps", flush=True)
            records.append({'model': 'TabPFN', 'n': n, 'rep': rep, 'time_s': t_pfn})

        gp_t = [r['time_s'] for r in records if r['model'] == 'GP' and r['n'] == n and not np.isnan(r['time_s'])]
        pfn_t = [r['time_s'] for r in records if r['model'] == 'TabPFN' and r['n'] == n and not np.isnan(r['time_s'])]
        if gp_t:
            print(
                f"    GP:     median={np.median(gp_t):.3f}s  "
                f"Q25={np.percentile(gp_t, 25):.3f}  Q75={np.percentile(gp_t, 75):.3f}"
                f"  (n_valid={len(gp_t)})",
                flush=True,
            )
        if pfn_t:
            print(
                f"    TabPFN: median={np.median(pfn_t):.3f}s  "
                f"Q25={np.percentile(pfn_t, 25):.3f}  Q75={np.percentile(pfn_t, 75):.3f}"
                f"  (n_valid={len(pfn_t)})",
                flush=True,
            )

    return pd.DataFrame(records)


def main() -> None:
    """CLI entry point for the A7 scaling timing harness."""
    parser = argparse.ArgumentParser(
        description='A7: Synthetic grid scaling timing benchmark (GP vs TabPFN).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--grid_sizes', nargs='+', type=int,
        default=[100, 250, 500, 1000, 2000],
        metavar='N',
        help='Grid sizes (pool sizes) to benchmark.',
    )
    parser.add_argument(
        '--n_context_frac', type=float, default=0.1,
        help='Fraction of n used as training context (10%% ≈ realistic explored budget).',
    )
    parser.add_argument(
        '--n_reps', type=int, default=15,
        help='Timing repetitions per (model, n).',
    )
    parser.add_argument(
        '--d', type=int, default=2,
        help='Feature dimensionality (2 for 2D electrode coordinates).',
    )
    parser.add_argument(
        '--device', type=str, default='cpu', choices=['cpu', 'cuda'],
        help='Device for TabPFN inference.  GP always runs on CPU.',
    )
    parser.add_argument(
        '--gp_n_opt_steps', type=int, default=50,
        help='GP marginal-likelihood optimisation steps per fit call.',
    )
    parser.add_argument(
        '--acq_fn', type=str, default='ts', choices=['ts', 'ucb'],
        help="Acquisition function to benchmark.  'ts' (default) is the paper's primary; "
             "'ucb' times the UCB path for comparison.",
    )
    parser.add_argument(
        '--ts_temperature', type=float, default=1.0,
        help='Temperature for TabPFN bar-distribution TS sampling (only used when acq_fn=ts).',
    )
    parser.add_argument(
        '--kappa', type=float, default=2.0,
        help='UCB exploration coefficient (only used when acq_fn=ucb).',
    )
    parser.add_argument(
        '--timeout_per_rep', type=float, default=float('inf'),
        metavar='SEC',
        help='Per-rep wall-clock limit in seconds.  If a trial exceeds this, '
             'remaining reps at that n are skipped and recorded as NaN. '
             'Default: no limit.',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument(
        '--output_dir', type=str, default='./output/scaling',
        help='Directory for CSV and SVG outputs.',
    )
    parser.add_argument(
        '--anchor_n', type=int, default=100,
        help='Grid size to annotate as the paper real-data scale (NHP 10×10 = 100).',
    )
    parser.add_argument(
        '--anchor_label', type=str, default='Paper scale (NHP 10×10)',
        help='Label text for the anchor annotation.',
    )
    args = parser.parse_args()

    print(
        f"A7 scaling benchmark: n={args.grid_sizes}  "
        f"n_context_frac={args.n_context_frac}  n_reps={args.n_reps}  "
        f"device={args.device}  gp_n_opt_steps={args.gp_n_opt_steps}  "
        f"acq_fn={args.acq_fn}",
        flush=True,
    )

    df = run_scaling_benchmark(
        grid_sizes=args.grid_sizes,
        n_context_frac=args.n_context_frac,
        n_reps=args.n_reps,
        d=args.d,
        tabpfn_device=args.device,
        gp_n_opt_steps=args.gp_n_opt_steps,
        acq_fn=args.acq_fn,
        ts_temperature=args.ts_temperature,
        kappa=args.kappa,
        timeout_per_rep=args.timeout_per_rep,
        seed=args.seed,
    )

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, 'scaling_timing.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved -> {csv_path}")

    plot_scaling_timing(
        df,
        save=args.save,
        output_dir=args.output_dir,
        anchor_n=args.anchor_n,
        anchor_label=args.anchor_label,
    )


if __name__ == '__main__':
    main()
