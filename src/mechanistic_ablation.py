"""§18 — Spatial-autocorrelation ablation (mechanistic evidence for Hyp A).

Directly probes *why* vanilla TabPFN works on low-D spatial neurostim by
destroying the one property Hypothesis A depends on: smooth spatial
autocorrelation between electrode coordinate and EMG response.  At each shuffle
fraction ``f`` a random ``f`` of sites have their coordinate↔response pairing
scrambled (see :func:`utils.data_utils.shuffle_response_pairing`), and both
TabPFN and GP re-run BO on the corrupted map.

Prediction: TabPFN's advantage collapses monotonically as ``f→1``, converging
to a GP-with-the-wrong-kernel — a causal dose-response, not a correlational
claim.  This is the direct neurostim probe that replaces the Ye-et-al.-by-
analogy argument (companion probe: layer-wise CKA, produced by
``id_ood_analysis.py --analyses cka --prior_source gp_bag``).

Usage::

    python src/mechanistic_ablation.py --dataset nhp --shuffle_fracs 0.0 0.25 0.5 0.75 1.0 --n_reps 20 --save
    python src/mechanistic_ablation.py --config configs/nhp_vanilla_benchmark.yaml --shuffle_fracs 0.0 0.5 1.0
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

# Ensure src/ is importable when run from the project root.
import sys
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from evaluation import evaluate_optimization
from models.regressors import GPSurrogate, TabPFNSurrogate
from tabpfn import TabPFNRegressor
from utils.data_utils import (
    load_data, ALL_SUBJECTS, HELD_OUT_SUBJECTS,
    generate_experiment_tag, create_run_dir, write_run_config,
)
from utils.visualization import plot_shuffle_ablation


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


def _build_experiments(
    dataset_type: str,
    subjects: List[int],
) -> List[tuple]:
    """Build the list of (subject_idx, emg_idx) pairs to evaluate.

    Args:
        dataset_type: ``'rat'``, ``'nhp'``, ``'spinal'``, or ``'5d_rat'``.
        subjects: Subject indices to include.

    Returns:
        List of ``(subject_idx, emg_idx)`` tuples across all EMG channels.
    """
    experiments: List[tuple] = []
    for subj_idx in subjects:
        data = load_data(dataset_type, subj_idx)
        n_emgs = data['sorted_respMean'].shape[1]
        for emg_idx in range(n_emgs):
            experiments.append((subj_idx, emg_idx))
    return experiments


def _final_regret_per_rep(result: dict) -> np.ndarray:
    """Normalised final simple regret per rep from an ``evaluate_optimization`` result.

    Args:
        result: Return dict of :func:`evaluation.evaluate_optimization`.

    Returns:
        1-D array of per-rep normalised final regret, shape [n_reps].
    """
    y_test = np.asarray(result['y_test'], dtype=float)
    y_range = float(y_test.max() - y_test.min())
    if y_range < 1e-8:
        return np.array([])
    optimal = float(y_test.max())
    running_best = np.maximum.accumulate(np.array(result['values']), axis=1)  # [n_reps, budget]
    return (optimal - running_best[:, -1]) / y_range  # [n_reps]


def run_shuffle_ablation(
    dataset_type: str,
    shuffle_fracs: List[float],
    device: str = 'cpu',
    budget: int = 100,
    n_reps: int = 20,
    kappa_schedule: float = 0.0,
    acq_fn: str = 'ucb',
    ts_temperature: float = 1.0,
    subjects_mode: str = 'held_out',
    seed: int = 42,
    save: bool = False,
) -> pd.DataFrame:
    """Run the spatial-autocorrelation dose-response for TabPFN and GP.

    For each fraction in ``shuffle_fracs`` and each (subject, EMG) pair, runs BO
    with both surrogates on the map whose coordinate↔response pairing has been
    scrambled at that fraction, and records per-rep final regret.

    Args:
        dataset_type: ``'rat'``, ``'nhp'``, ``'spinal'``, or ``'5d_rat'``.
        shuffle_fracs: Shuffle fractions to sweep (each in [0, 1]).
        device: PyTorch device string.
        budget: Total BO query budget.
        n_reps: Repetitions per (subject, EMG, fraction).
        kappa_schedule: UCB coefficient (``0.0`` → auto cosine-annealed).
        acq_fn: Acquisition function (``'ucb'`` or ``'ts'``).
        ts_temperature: Temperature for TabPFN Thompson Sampling.
        subjects_mode: ``'held_out'`` (default) or ``'all'``.
        seed: Master random seed.
        save: If True, persist the DataFrame and dose-response plot.

    Returns:
        Long-form DataFrame with columns
        ``['shuffle_frac', 'Model', 'Regret', 'Subject', 'EMG']``.
    """
    set_seed(seed)

    subjects = (
        list(ALL_SUBJECTS[dataset_type]) if subjects_mode == 'all'
        else list(HELD_OUT_SUBJECTS[dataset_type])
    )
    experiments = _build_experiments(dataset_type, subjects)
    print(f"[INFO] {len(experiments)} experiments across subjects {subjects}")

    tag_config: Dict[str, Any] = {
        'dataset': dataset_type,
        'shuffle_fracs': sorted(shuffle_fracs),
        'budget': budget,
        'n_reps': n_reps,
        'kappa_schedule': kappa_schedule,
        'acq_fn': acq_fn,
        'subjects_mode': subjects_mode,
        'seed': seed,
    }
    save_tag = generate_experiment_tag(dataset_type, 'shuffle-ablation', tag_config)
    run_dir = create_run_dir(save_tag, tag=save_tag) if save else None
    if run_dir is not None:
        write_run_config(run_dir, {
            'run_type': 'mechanistic_shuffle_ablation',
            'experiment_tag': save_tag,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            **tag_config,
            'device': device,
        })
        print(f"[INFO] Run directory: {run_dir}")

    rows: List[dict] = []
    tabpfn_base = TabPFNRegressor(device=device, n_estimators=1,
                                  ignore_pretraining_limits=True)

    for frac in shuffle_fracs:
        print(f"\n{'=' * 50}\n  shuffle_frac = {frac}\n{'=' * 50}")
        for subj_idx, emg_idx in experiments:
            for model_name, surrogate_factory, normalization in (
                ('TabPFN', lambda: TabPFNSurrogate(model=tabpfn_base), 'pfn'),
                ('GP', lambda: GPSurrogate(device=device), 'gp'),
            ):
                res = evaluate_optimization(
                    surrogate=surrogate_factory(),
                    dataset_type=dataset_type,
                    subject_idx=subj_idx,
                    emg_idx=emg_idx,
                    device=device,
                    budget=budget,
                    n_reps=n_reps,
                    kappa_schedule=kappa_schedule,
                    normalization=normalization,
                    acq_fn=acq_fn,
                    ts_temperature=ts_temperature,
                    shuffle_frac=frac,
                )
                for reg in _final_regret_per_rep(res):
                    rows.append({
                        'shuffle_frac': float(frac),
                        'Model': model_name,
                        'Regret': float(reg),
                        'Subject': subj_idx,
                        'EMG': emg_idx,
                    })
            print(f"    subject={subj_idx}, emg={emg_idx} done")
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    plot_shuffle_ablation(df, dataset=dataset_type, split_type=subjects_mode,
                          save=save, output_dir=run_dir)

    if save and run_dir is not None:
        results_dir = os.path.join(run_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        pkl_path = os.path.join(results_dir, f'{save_tag}_shuffle_ablation.pkl')
        df.to_pickle(pkl_path)
        print(f"Saved shuffle-ablation DataFrame -> {pkl_path}")

    return df


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML experiment config file (mapping).

    Args:
        path: Filesystem path to a ``.yaml`` config file.

    Returns:
        Dict of key-value pairs from the YAML document.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the YAML document is not a mapping.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}: {path}")
    return cfg


def main() -> None:
    """CLI entry point for the §18 spatial-autocorrelation ablation."""
    parser = argparse.ArgumentParser(
        description='§18 spatial-autocorrelation ablation (mechanistic Hyp A evidence).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML config; CLI flags override its values.')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['rat', 'nhp', 'spinal', '5d_rat'])
    parser.add_argument('--shuffle_fracs', type=float, nargs='+', default=None,
                        metavar='F', help='Shuffle fractions to sweep (default: 0.0 0.25 0.5 0.75 1.0).')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'])
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--n_reps', type=int, default=None)
    parser.add_argument('--kappa_schedule', type=float, default=None)
    parser.add_argument('--acq_fn', type=str, default=None, choices=['ucb', 'ts'])
    parser.add_argument('--ts_temperature', type=float, default=None)
    parser.add_argument('--subjects', type=str, default=None, choices=['held_out', 'all'],
                        dest='subjects_mode')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()

    if args.config is not None:
        cfg = _load_yaml_config(args.config)
        for key, value in cfg.items():
            if key == 'save':
                if not args.save:
                    args.save = value
            elif getattr(args, key, None) is None:
                setattr(args, key, value)
        print(f"[config] Loaded YAML defaults from {args.config}")

    _defaults = {
        'dataset': 'nhp',
        'shuffle_fracs': [0.0, 0.25, 0.5, 0.75, 1.0],
        'device': 'cuda',
        'budget': 100,
        'n_reps': 20,
        'kappa_schedule': 0.0,
        'acq_fn': 'ts',
        'ts_temperature': 1.0,
        'subjects_mode': 'held_out',
        'seed': 42,
    }
    for key, default in _defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)

    run_shuffle_ablation(
        dataset_type=args.dataset,
        shuffle_fracs=args.shuffle_fracs,
        device=args.device,
        budget=args.budget,
        n_reps=args.n_reps,
        kappa_schedule=args.kappa_schedule,
        acq_fn=args.acq_fn,
        ts_temperature=args.ts_temperature,
        subjects_mode=args.subjects_mode,
        seed=args.seed,
        save=args.save,
    )


if __name__ == '__main__':
    main()
