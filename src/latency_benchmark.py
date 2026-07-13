"""§19 — Online / real-time latency feasibility benchmark.

Grounds the paper's "real-time" motivation concretely, with no live animals.
Measures per-query end-to-end latency — the actual closed-loop cost of one BO
step: fit the surrogate on the observed context, evaluate the acquisition
function over the candidate pool, and read out the exploitation recommendation
— for TabPFN vs GP at the *real* neurostim grid scales (2D electrode grids and
the 5D multi-parameter search space), then reports it against typical
biological inter-stimulus intervals (ISIs).

A per-query latency comfortably below the ISI means the method is feasible for
closed-loop stimulation at that scale.  The timed sequence mirrors
``bo_loops.run_bo_loop``'s per-step work exactly (see
``scaling_timing._time_query``), so the numbers reflect true BO cost.

Usage::

    python src/latency_benchmark.py --datasets nhp rat spinal 5d_rat --n_reps 20 --device cpu --save
    python src/latency_benchmark.py --datasets nhp --context_frac 0.1 0.25 0.5 --save
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from models.regressors import GPSurrogate, TabPFNSurrogate
from scaling_timing import _time_query
from tabpfn import TabPFNRegressor
from utils.data_utils import load_data, preprocess_neural_data, ALL_SUBJECTS
from utils.visualization import plot_latency_vs_isi


# Reference inter-stimulus intervals (seconds).  Cortical/spinal microstimulation
# closed-loop cycles typically run at 0.5–2 s per query; values are drawn as
# horizontal reference lines and are overridable via --isi.
_DEFAULT_ISI_REFS: Dict[str, float] = {
    'Fast ISI (0.5 s)': 0.5,
    'Typical ISI (1 s)': 1.0,
    'Slow ISI (2 s)': 2.0,
}


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across torch, numpy, and random.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _real_pool(dataset_type: str, subject_idx: int, emg_idx: int = 0) -> np.ndarray:
    """Return the real candidate-coordinate pool for a (dataset, subject, EMG).

    Args:
        dataset_type: ``'rat'``, ``'nhp'``, ``'spinal'``, or ``'5d_rat'``.
        subject_idx: Subject index.
        emg_idx: EMG channel index (only affects which sites are valid).

    Returns:
        Coordinate pool ``X_pool``, shape [N, D], for one representative channel.
    """
    data = load_data(dataset_type, subject_idx)
    X_pool, _y_pool, _X_test, _y_test, _scaler = preprocess_neural_data(
        data, emg_idx, 'pfn'
    )
    return X_pool.astype(np.float32)  # [N, D]


def run_latency_benchmark(
    datasets: List[str],
    context_fracs: Optional[List[float]] = None,
    n_reps: int = 20,
    device: str = 'cpu',
    acq_fn: str = 'ucb',
    ts_temperature: float = 1.0,
    kappa: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure per-query latency for TabPFN and GP at real neurostim scales.

    For each dataset, uses the first subject in ``ALL_SUBJECTS`` as the
    representative grid.  For each requested context fraction, times one full
    BO-step cycle (fit + acquire + predict) across ``n_reps`` trials for both
    surrogates.  The dataset "scale" label encodes the grid size and
    dimensionality so the figure conveys both.

    Args:
        datasets: Datasets to benchmark, e.g. ``['nhp', 'rat', 'spinal', '5d_rat']``.
        context_fracs: Fractions of the pool used as observed context.
            Defaults to ``[0.1, 0.25, 0.5]`` (early / mid / late BO states).
        n_reps: Timing repetitions per (dataset, context_frac, model).
        device: Inference device for TabPFN.  GP always runs on CPU.
        acq_fn: Acquisition function to time (``'ucb'`` or ``'ts'``).
        ts_temperature: Temperature for TabPFN Thompson Sampling.
        kappa: UCB exploration coefficient used when ``acq_fn='ucb'``.
        seed: Master random seed.

    Returns:
        DataFrame with columns
        ``['dataset', 'scale', 'context_frac', 'model', 'rep', 'latency_s']``.

    Raises:
        ValueError: If ``acq_fn`` is not ``'ts'`` or ``'ucb'``.
    """
    if acq_fn not in ('ts', 'ucb'):
        raise ValueError(f"acq_fn must be 'ts' or 'ucb', got {acq_fn!r}.")
    if context_fracs is None:
        context_fracs = [0.1, 0.25, 0.5]
    set_seed(seed)
    rng = np.random.default_rng(seed)

    tabpfn_base = TabPFNRegressor(device=device, n_estimators=1,
                                  ignore_pretraining_limits=True)

    records: List[dict] = []
    for dataset_type in datasets:
        subj = ALL_SUBJECTS[dataset_type][0]
        X_pool = _real_pool(dataset_type, subj)           # [N, D]
        n, d = X_pool.shape
        scale_label = f'{dataset_type}\n(N={n}, D={d})'
        print(f"\n[{dataset_type}] subject={subj}  N={n}  D={d}")

        for cf in context_fracs:
            n_ctx = max(3, int(round(n * cf)))
            n_ctx = min(n_ctx, n)
            for rep in range(n_reps):
                # Fresh random context + response draw each rep.
                ctx_idx = rng.choice(n, size=n_ctx, replace=False)
                X_ctx = X_pool[ctx_idx]                    # [n_ctx, D]
                y_ctx = rng.standard_normal(n_ctx).astype(np.float32)  # [n_ctx]

                gp = GPSurrogate(device='cpu')
                t_gp = _time_query(gp, X_ctx, y_ctx, X_pool,
                                   acq_fn=acq_fn, ts_temperature=ts_temperature, kappa=kappa)
                del gp
                records.append({'dataset': dataset_type, 'scale': scale_label,
                                'context_frac': cf, 'model': 'GP', 'rep': rep,
                                'latency_s': t_gp})

                pfn = TabPFNSurrogate(tabpfn_base)
                t_pfn = _time_query(pfn, X_ctx, y_ctx, X_pool,
                                    acq_fn=acq_fn, ts_temperature=ts_temperature, kappa=kappa)
                records.append({'dataset': dataset_type, 'scale': scale_label,
                                'context_frac': cf, 'model': 'TabPFN', 'rep': rep,
                                'latency_s': t_pfn})

            gp_med = np.median([r['latency_s'] for r in records
                                if r['scale'] == scale_label and r['context_frac'] == cf and r['model'] == 'GP'])
            pfn_med = np.median([r['latency_s'] for r in records
                                 if r['scale'] == scale_label and r['context_frac'] == cf and r['model'] == 'TabPFN'])
            print(f"    context_frac={cf:.2f}  GP median={gp_med:.4f}s  TabPFN median={pfn_med:.4f}s")

    return pd.DataFrame(records)


def main() -> None:
    """CLI entry point for the §19 latency feasibility benchmark."""
    parser = argparse.ArgumentParser(
        description='§19 per-query latency benchmark (TabPFN vs GP) at real neurostim scales.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['nhp', 'rat', 'spinal', '5d_rat'],
                        choices=['rat', 'nhp', 'spinal', '5d_rat'])
    parser.add_argument('--context_fracs', type=float, nargs='+', default=[0.1, 0.25, 0.5],
                        metavar='F', help='Context fractions (BO exploration states) to time.')
    parser.add_argument('--n_reps', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device for TabPFN.  GP always runs on CPU.')
    parser.add_argument('--acq_fn', type=str, default='ucb', choices=['ucb', 'ts'])
    parser.add_argument('--ts_temperature', type=float, default=1.0)
    parser.add_argument('--kappa', type=float, default=2.0)
    parser.add_argument('--isi', type=float, nargs='+', default=None, metavar='SEC',
                        help='Override reference inter-stimulus intervals (seconds).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, default='./output/latency')
    args = parser.parse_args()

    df = run_latency_benchmark(
        datasets=args.datasets,
        context_fracs=args.context_fracs,
        n_reps=args.n_reps,
        device=args.device,
        acq_fn=args.acq_fn,
        ts_temperature=args.ts_temperature,
        kappa=args.kappa,
        seed=args.seed,
    )

    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, 'latency_benchmark.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved -> {csv_path}")

    isi_refs = (
        {f'{v} s': v for v in args.isi} if args.isi is not None else _DEFAULT_ISI_REFS
    )
    plot_latency_vs_isi(df, isi_refs=isi_refs, save=args.save, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
