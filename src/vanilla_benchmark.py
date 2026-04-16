"""Vanilla TabPFN v2 vs GP benchmark (Hypothesis A).

CLI entry point that evaluates pretrained TabPFNRegressor (v2) against
ExactGP (RBF kernel) on neurostimulation datasets.  No finetuning is
performed -- this replaces the deleted ``src/main.py`` which incorrectly
used PFNs4BO ``TransformerBOMethod`` (PFN v1).

Three evaluation modes:
  - ``fit``                -- R² comparison on held-out subjects (A1)
  - ``optimization``       -- Cumulative regret + timing via BO loop (A2, A3)
  - ``optimization_budget``-- Budget sweep over multiple query budgets (A4)

Usage::

    python src/vanilla_benchmark.py --dataset nhp --mode optimization --save
    python src/vanilla_benchmark.py --config configs/nhp_vanilla_benchmark.yaml
    python src/vanilla_benchmark.py --config configs/nhp_vanilla_benchmark.yaml --budget 50
"""
import argparse
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tabpfn import TabPFNRegressor

from evaluation import gp_baseline, finetuned_fit, evaluate_optimization
from models.regressors import GPSurrogate, TabPFNSurrogate
from utils.data_utils import (
    load_data,
    HELD_OUT_SUBJECTS,
    generate_experiment_tag,
    save_results,
    create_run_dir,
    write_run_config,
)
from utils.visualization import (
    r2_per_muscle,
    r2_by_subject,
    show_emg_map,
    regret_with_timing,
    regret_by_subject,
    regret_by_emg,
    budget_sweep_plot,
    visualize_representation,
)


_VALID_MODES: frozenset = frozenset({'fit', 'optimization', 'optimization_budget'})

# GP kappa values are fixed to match legacy GP BO behaviour
_GP_KAPPA_0: float = 5.0
_GP_KAPPA_MIN: float = 1.0


# ============================================
#           Helpers
# ============================================


def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML experiment config file.

    Duplicated from finetuning._load_yaml_config pending extraction to
    utils/config_utils.py (Sprint Step 6).

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
        raise ValueError(
            f"Config file must be a YAML mapping, got {type(cfg).__name__}: {path}"
        )
    return cfg


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


def _build_vanilla_tabpfn(
    device: str,
    n_estimators: int = 1,
) -> TabPFNRegressor:
    """Construct a vanilla (pretrained, unfinetuned) TabPFNRegressor.

    Args:
        device: PyTorch device string, ``'cpu'`` or ``'cuda'``.
        n_estimators: Number of ensemble members.  Use 1 for BO loops
            (acquisition speed), 8 for fit evaluation (accuracy).

    Returns:
        Configured TabPFNRegressor with ``ignore_pretraining_limits=True``.
    """
    return TabPFNRegressor(
        device=device,
        n_estimators=n_estimators,
        ignore_pretraining_limits=True,
    )


def _build_experiments(
    dataset_type: str,
    held_out_subj_idx: Optional[int] = None,
) -> List[tuple]:
    """Build the list of (subject_idx, emg_idx) pairs to evaluate.

    Uses ``HELD_OUT_SUBJECTS[dataset_type]`` x all valid EMG indices for
    each held-out subject.  When *held_out_subj_idx* is given, only that
    subject is evaluated (used by SLURM job-array Family 7).

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        held_out_subj_idx: If provided, restrict evaluation to this single
            held-out subject index instead of all ``HELD_OUT_SUBJECTS``.

    Returns:
        List of (subject_idx, emg_idx) tuples.
    """
    subjects = (
        [held_out_subj_idx]
        if held_out_subj_idx is not None
        else HELD_OUT_SUBJECTS[dataset_type]
    )
    experiments: List[tuple] = []
    for subj_idx in subjects:
        data = load_data(dataset_type, subj_idx)
        n_emgs = data['sorted_respMean'].shape[1]
        for emg_idx in range(n_emgs):
            experiments.append((subj_idx, emg_idx))
    return experiments


# ============================================
#           Mode-specific inner loops
# ============================================


def _vanilla_fit(
    dataset_type: str,
    experiments: List[tuple],
    device: str,
    budget: int,
    n_reps: int,
    run_dir: str,
    exp_tag: str,
    save: bool,
    save_tag: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """Run fit evaluation for vanilla TabPFN and GP on held-out experiments.

    For each (subject, emg) pair:
      - TabPFN: calls ``finetuned_fit()`` with a freshly constructed vanilla
        TabPFNRegressor (n_estimators=8 for accuracy).
      - GP: calls ``gp_baseline(mode='fit')``.

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        experiments: List of (subject_idx, emg_idx) from ``_build_experiments()``.
        device: PyTorch device string.
        budget: Number of training points sampled per repetition.
        n_reps: Number of repetitions per (subject, emg) pair.
        run_dir: Output directory from ``create_run_dir()``.
        exp_tag: Experiment tag suffix for plot filenames.
        save: If True, persist results as pkl + CSV.
        save_tag: Full experiment tag from ``generate_experiment_tag()``.

    Returns:
        ``{'GP': list[dict], 'TabPFN': list[dict]}``.
    """
    results_tabpfn: List[dict] = []
    results_gp: List[dict] = []

    for subj_idx, emg_idx in experiments:
        print(f"  Fit: subject={subj_idx}, emg={emg_idx}")

        # Fresh model per experiment to avoid context bleed
        tabpfn_model = _build_vanilla_tabpfn(device, n_estimators=8)
        res_tabpfn = finetuned_fit(
            dataset_type, subj_idx, emg_idx,
            model=tabpfn_model,
            device=device, budget=budget, n_reps=n_reps,
        )
        res_tabpfn['model_type'] = 'vanilla_tabpfn'

        if np.isnan(res_tabpfn['y_pred']).any():
            raise RuntimeError(
                f"NaN in TabPFN fit predictions: subject={subj_idx}, emg={emg_idx}"
            )

        res_gp = gp_baseline(
            dataset_type, subj_idx, emg_idx, mode='fit',
            device=device, budget=budget, n_reps=n_reps,
        )

        results_tabpfn.append(res_tabpfn)
        results_gp.append(res_gp)
        print(f"    TabPFN R2={np.mean(res_tabpfn['r2']):.3f}  |  "
              f"GP R2={np.mean(res_gp['r2']):.3f}")

    results_dict = {'GP': results_gp, 'TabPFN': results_tabpfn}

    # --- Plots ---
    tag = f'_vanilla_{exp_tag}'
    r2_per_muscle(results_dict, mode=tag, save=True, output_dir=run_dir)
    r2_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)

    n_maps = min(6, len(experiments))
    for idx in random.sample(range(len(experiments)), n_maps):
        show_emg_map(results_tabpfn, idx, 'TabPFN',
                     mode=f'_vanilla_{exp_tag}', save=True, output_dir=run_dir)
        show_emg_map(results_gp, idx, 'GP',
                     mode=f'_vanilla_{exp_tag}_gp', save=True, output_dir=run_dir)

    all_r2 = [np.mean(r['r2']) for r in results_tabpfn]
    print(f"\nFit done. {len(results_tabpfn)} experiments.")
    print(f"Vanilla TabPFN mean R2: {np.mean(all_r2):.3f} +/- {np.std(all_r2):.3f}")

    if save:
        save_results(
            results_dict, 'fit',
            output_dir=os.path.join(run_dir, 'results'),
            tag=save_tag,
            metadata=metadata,
        )
    return results_dict


def _vanilla_optimization(
    dataset_type: str,
    experiments: List[tuple],
    device: str,
    budget: int,
    n_reps: int,
    kappa_0: float,
    kappa_min: float,
    run_dir: str,
    exp_tag: str,
    save: bool,
    save_tag: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, list]:
    """Run BO optimization evaluation for vanilla TabPFN and GP.

    Uses ``evaluate_optimization()`` (Step 2 unified pipeline) for both
    models, validating the surrogate-agnostic BO loop end-to-end.

    GP uses fixed kappa values ``_GP_KAPPA_0`` / ``_GP_KAPPA_MIN`` and
    ``normalization='gp'``, regardless of the TabPFN kappa settings.

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        experiments: List of (subject_idx, emg_idx).
        device: PyTorch device string.
        budget: Total BO query budget.
        n_reps: Repetitions per experiment.
        kappa_0: Initial UCB coefficient for TabPFN surrogate.
        kappa_min: Final UCB coefficient for TabPFN surrogate.
        run_dir: Output directory.
        exp_tag: Tag suffix for plot filenames.
        save: If True, persist results.
        save_tag: Full experiment tag.

    Returns:
        ``{'GP': list[dict], 'TabPFN': list[dict]}``.
    """
    results_tabpfn: List[dict] = []
    results_gp: List[dict] = []

    for subj_idx, emg_idx in experiments:
        print(f"  Optimization: subject={subj_idx}, emg={emg_idx}")

        # TabPFN: n_estimators=1 for BO speed
        tabpfn_base = _build_vanilla_tabpfn(device, n_estimators=1)
        tabpfn_surrogate = TabPFNSurrogate(
            model=tabpfn_base, kappa_0=kappa_0, kappa_min=kappa_min,
        )
        res_tabpfn = evaluate_optimization(
            surrogate=tabpfn_surrogate,
            dataset_type=dataset_type,
            subject_idx=subj_idx,
            emg_idx=emg_idx,
            device=device,
            budget=budget,
            n_reps=n_reps,
            kappa_0=kappa_0,
            kappa_min=kappa_min,
            normalization='pfn',
        )
        res_tabpfn['model_type'] = 'vanilla_tabpfn'

        # GP: fixed kappa values appropriate for GP scale
        gp_surrogate = GPSurrogate(device=device)
        res_gp = evaluate_optimization(
            surrogate=gp_surrogate,
            dataset_type=dataset_type,
            subject_idx=subj_idx,
            emg_idx=emg_idx,
            device=device,
            budget=budget,
            n_reps=n_reps,
            kappa_0=_GP_KAPPA_0,
            kappa_min=_GP_KAPPA_MIN,
            normalization='gp',
        )
        res_gp['model_type'] = 'gp'

        results_tabpfn.append(res_tabpfn)
        results_gp.append(res_gp)
        print(f"    TabPFN R2={np.mean(res_tabpfn['r2']):.3f}  |  "
              f"GP R2={np.mean(res_gp['r2']):.3f}")

    results_dict = {'GP': results_gp, 'TabPFN': results_tabpfn}

    # --- Plots ---
    tag = f'_vanilla_{exp_tag}_opt'
    r2_per_muscle(results_dict, mode=tag, save=True,
                  output_dir=run_dir, eval_type='optimization')
    regret_with_timing(results_dict, split_type=exp_tag, save=True,
                       output_dir=run_dir)
    regret_by_subject(results_dict, split_type=exp_tag, save=True,
                      output_dir=run_dir)
    regret_by_emg(results_dict, split_type=exp_tag, save=True,
                  output_dir=run_dir)

    n_maps = min(6, len(experiments))
    for idx in random.sample(range(len(experiments)), n_maps):
        show_emg_map(results_tabpfn, idx, 'TabPFN',
                     mode=f'_vanilla_{exp_tag}_opt', save=True,
                     output_dir=run_dir, eval_type='optimization')
        show_emg_map(results_gp, idx, 'GP',
                     mode=f'_vanilla_{exp_tag}_opt_gp', save=True,
                     output_dir=run_dir, eval_type='optimization')
    visualize_representation(results_dict, mode=f'_vanilla_{exp_tag}',
                             save=True, output_dir=run_dir)

    all_r2 = [np.mean(r['r2']) for r in results_tabpfn]
    print(f"\nOptimization done. {len(results_tabpfn)} experiments.")
    print(f"Vanilla TabPFN mean R2: {np.mean(all_r2):.3f} +/- {np.std(all_r2):.3f}")

    if save:
        save_results(
            results_dict, 'optimization',
            output_dir=os.path.join(run_dir, 'results'),
            tag=save_tag,
            metadata=metadata,
        )
    return results_dict


def _vanilla_optimization_budget(
    dataset_type: str,
    experiments: List[tuple],
    device: str,
    budgets: List[int],
    n_reps: int,
    kappa_0: float,
    kappa_min: float,
    run_dir: str,
    exp_tag: str,
    save: bool,
    save_tag: str,
) -> pd.DataFrame:
    """Budget sweep: run optimization evaluation across multiple query budgets.

    For each budget in *budgets*, evaluates both vanilla TabPFN and GP using
    ``evaluate_optimization()``.  Results are collected into a long-form
    DataFrame suitable for ``budget_sweep_plot()``.

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        experiments: List of (subject_idx, emg_idx).
        device: PyTorch device string.
        budgets: List of integer query budgets to evaluate.
        n_reps: Repetitions per (subject, emg, budget) combination.
        kappa_0: Initial UCB coefficient for TabPFN.
        kappa_min: Final UCB coefficient for TabPFN.
        run_dir: Output directory.
        exp_tag: Tag suffix.
        save: If True, persist DataFrame as pkl.
        save_tag: Full experiment tag.

    Returns:
        Long-form DataFrame with columns: Budget, Model, Regret, R2, ID.
    """
    plot_data: List[dict] = []

    for b in budgets:
        print(f"\n  Budget = {b}")
        for subj_idx, emg_idx in experiments:
            print(f"    subject={subj_idx}, emg={emg_idx}")

            # --- TabPFN ---
            tabpfn_base = _build_vanilla_tabpfn(device, n_estimators=1)
            tabpfn_surrogate = TabPFNSurrogate(
                model=tabpfn_base, kappa_0=kappa_0, kappa_min=kappa_min,
            )
            res_tabpfn = evaluate_optimization(
                surrogate=tabpfn_surrogate,
                dataset_type=dataset_type,
                subject_idx=subj_idx,
                emg_idx=emg_idx,
                device=device,
                budget=b,
                n_reps=n_reps,
                kappa_0=kappa_0,
                kappa_min=kappa_min,
                normalization='pfn',
            )
            _collect_budget_rows(
                plot_data, res_tabpfn, 'TabPFN', b, subj_idx, emg_idx,
            )

            # --- GP ---
            gp_surrogate = GPSurrogate(device=device)
            res_gp = evaluate_optimization(
                surrogate=gp_surrogate,
                dataset_type=dataset_type,
                subject_idx=subj_idx,
                emg_idx=emg_idx,
                device=device,
                budget=b,
                n_reps=n_reps,
                kappa_0=_GP_KAPPA_0,
                kappa_min=_GP_KAPPA_MIN,
                normalization='gp',
            )
            _collect_budget_rows(
                plot_data, res_gp, 'GP', b, subj_idx, emg_idx,
            )

    df = pd.DataFrame(plot_data)

    # --- Plot ---
    budget_sweep_plot(df, eval_type='optimization', dataset=dataset_type,
                      split_type=exp_tag, save=True, output_dir=run_dir)

    if save:
        results_dir = os.path.join(run_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        pkl_path = os.path.join(results_dir, f'{save_tag}_optimization_budget.pkl')
        df.to_pickle(pkl_path)
        print(f"Saved budget DataFrame -> {pkl_path}")

    return df


def _collect_budget_rows(
    plot_data: List[dict],
    result: dict,
    model_name: str,
    budget: int,
    subj_idx: int,
    emg_idx: int,
) -> None:
    """Extract per-rep regret and R2 from a result dict into plot_data rows.

    Args:
        plot_data: Accumulator list to append rows to.
        result: Return dict from ``evaluate_optimization()``.
        model_name: Display name, e.g. ``'TabPFN'`` or ``'GP'``.
        budget: Query budget for this evaluation.
        subj_idx: Subject index.
        emg_idx: EMG channel index.
    """
    optimal = result['y_test'].max()
    raw = np.array(result['values'])                   # [n_reps, budget]
    best_so_far = np.maximum.accumulate(raw, axis=1)   # [n_reps, budget]
    final_regret = optimal - best_so_far[:, -1]        # [n_reps]

    for regret_val, r2_val in zip(final_regret, result['r2']):
        plot_data.append({
            'Budget': budget,
            'Model': model_name,
            'Regret': float(regret_val),
            'R2': float(r2_val),
            'ID': f'{subj_idx}_{emg_idx}',
        })


# ============================================
#           Orchestrator
# ============================================


def run_vanilla_benchmark(
    dataset_type: str,
    mode: List[str],
    device: str = 'cpu',
    budget: int = 100,
    n_reps: int = 30,
    budgets: Optional[List[int]] = None,
    kappa_0: float = 2.5,
    kappa_min: float = 0.5,
    held_out_subj_idx: Optional[int] = None,
    save: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Orchestrate vanilla TabPFN vs GP benchmark for one or more modes.

    Sets random seeds, builds the experiment list (held-out subjects x EMGs),
    creates a run directory with ``config.json``, then dispatches to the
    mode-specific inner functions.

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        mode: List of evaluation modes (see ``_VALID_MODES``).
        device: PyTorch device string.
        budget: Training points (fit) or BO queries (optimization).
        n_reps: Repetitions per experiment.
        budgets: Budget list for ``'optimization_budget'`` mode.
            Defaults to ``[10, 30, 50, 100, 200]``.
        kappa_0: Initial UCB coefficient for TabPFN BO.
        kappa_min: Final UCB coefficient for TabPFN BO.
        held_out_subj_idx: If given, restrict evaluation to this single held-out
            subject (used by SLURM job-array Family 7 where one job = one subject).
            When ``None``, all ``HELD_OUT_SUBJECTS`` are evaluated.
        save: If True, persist results (pkl + CSV).
        seed: Master random seed.

    Returns:
        Dict keyed by mode name.  Values are either
        ``dict[str, list[dict]]`` (fit/optimization) or ``pd.DataFrame``
        (optimization_budget).

    Raises:
        ValueError: If mode contains invalid values or dataset_type is unknown.
    """
    # --- Validate ---
    if dataset_type not in ('rat', 'nhp'):
        raise ValueError(f"Unknown dataset_type={dataset_type!r}. Use 'rat' or 'nhp'.")
    invalid = set(mode) - _VALID_MODES
    if invalid:
        raise ValueError(
            f"Invalid mode(s): {', '.join(sorted(invalid))}. "
            f"Valid: {', '.join(sorted(_VALID_MODES))}"
        )
    if budgets is None:
        budgets = [10, 30, 50, 100, 200]

    # --- Seeds ---
    set_seed(seed)

    # --- Build experiment list ---
    experiments = _build_experiments(dataset_type, held_out_subj_idx=held_out_subj_idx)
    n_held_out = 1 if held_out_subj_idx is not None else len(HELD_OUT_SUBJECTS[dataset_type])
    print(f"[INFO] {len(experiments)} experiments ({n_held_out} held-out subject(s))")

    # --- Experiment tag and run directory ---
    tag_config: Dict[str, Any] = {
        'dataset_type': dataset_type,
        'mode': sorted(mode),
        'budget': budget,
        'n_reps': n_reps,
        'kappa_0': kappa_0,
        'kappa_min': kappa_min,
        'seed': seed,
    }
    if held_out_subj_idx is not None:
        tag_config['held_out_subj_idx'] = held_out_subj_idx
    save_tag = generate_experiment_tag(dataset_type, 'vanilla-benchmark', tag_config)
    exp_tag = 'vanilla_benchmark'

    run_dir = create_run_dir(save_tag)
    run_config = {
        'run_type': 'vanilla_benchmark',
        'experiment_tag': save_tag,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'dataset_type': dataset_type,
        'mode': mode,
        'device': device,
        'budget': budget,
        'n_reps': n_reps,
        'budgets': budgets,
        'kappa_0': kappa_0,
        'kappa_min': kappa_min,
        'seed': seed,
        'held_out_subj': held_out_subj_idx,
        'held_out_subjects': (
            [held_out_subj_idx] if held_out_subj_idx is not None
            else HELD_OUT_SUBJECTS[dataset_type]
        ),
        'n_experiments': len(experiments),
    }
    write_run_config(run_dir, run_config)
    print(f"[INFO] Run directory: {run_dir}")

    # --- Build metadata for pkl provenance ---
    _metadata: Dict[str, Any] = {
        'family': 'vanilla-benchmark',
        'dataset': dataset_type,
        'tag': save_tag,
        'date': datetime.now().isoformat(timespec='seconds'),
        'run_type': 'vanilla_benchmark',
        'held_out_subj': held_out_subj_idx,
    }

    # --- Dispatch ---
    all_results: Dict[str, Any] = {}

    for m in mode:
        print(f"\n{'=' * 60}")
        print(f"Running mode: {m}")
        print('=' * 60)

        if m == 'fit':
            all_results['fit'] = _vanilla_fit(
                dataset_type, experiments, device, budget, n_reps,
                run_dir, exp_tag, save, save_tag,
                metadata=_metadata,
            )

        elif m == 'optimization':
            all_results['optimization'] = _vanilla_optimization(
                dataset_type, experiments, device, budget, n_reps,
                kappa_0, kappa_min,
                run_dir, exp_tag, save, save_tag,
                metadata=_metadata,
            )

        elif m == 'optimization_budget':
            all_results['optimization_budget'] = _vanilla_optimization_budget(
                dataset_type, experiments, device, budgets, n_reps,
                kappa_0, kappa_min,
                run_dir, exp_tag, save, save_tag,
            )

    return all_results


# ============================================
#           CLI Entry Point
# ============================================


def run_benchmark() -> None:
    """CLI entry point for vanilla TabPFN vs GP benchmarking.

    Parses arguments, applies YAML config defaults (CLI overrides YAML),
    applies hardcoded defaults for remaining None values, then calls
    ``run_vanilla_benchmark()``.
    """
    parser = argparse.ArgumentParser(
        description='Benchmark vanilla TabPFN v2 vs GP on neurostimulation data.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None, metavar='PATH',
                        help='Path to a YAML config file.  All keys are used as defaults;\n'
                             'any CLI flag that is explicitly provided overrides the YAML value.')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['rat', 'nhp'],
                        help='Dataset type (default: nhp)')
    parser.add_argument('--mode', type=lambda s: s.split(','), default=None,
                        metavar='MODE[,MODE,...]',
                        help='Comma-separated modes: fit, optimization, optimization_budget\n'
                             '(default: fit)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu or cuda (default: cuda)')
    parser.add_argument('--budget', type=int, default=None,
                        help='Training points (fit) or BO queries (optimization)\n'
                             '(default: 100)')
    parser.add_argument('--n_reps', type=int, default=None,
                        help='Repetitions per experiment (default: 30)')
    parser.add_argument('--budgets', type=int, nargs='+', default=None,
                        help='Budget sweep values for optimization_budget\n'
                             '(default: 10 30 50 100 200)')
    parser.add_argument('--kappa_0', type=float, default=None,
                        help='Initial UCB coefficient for TabPFN BO (default: 2.5)')
    parser.add_argument('--kappa_min', type=float, default=None,
                        help='Final UCB coefficient for TabPFN BO (default: 0.5)')
    parser.add_argument('--held_out_subj', type=int, default=None,
                        help='Restrict evaluation to a single held-out subject index.\n'
                             'When omitted, all HELD_OUT_SUBJECTS are evaluated.\n'
                             'Used by SLURM Family 7 job arrays (one job per subject).')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Persist results to output/runs/<tag>/results/')
    parser.add_argument('--seed', type=int, default=None,
                        help='Master random seed (default: 42)')

    args = parser.parse_args()

    # --- YAML config loading ---
    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        _bool_flags = {'save'}
        for key, value in yaml_cfg.items():
            if key in _bool_flags:
                if not getattr(args, key, False):
                    setattr(args, key, value)
            elif getattr(args, key, None) is None:
                setattr(args, key, value)
        print(f"[config] Loaded YAML defaults from {args.config}")

    # --- Apply hardcoded defaults for any remaining None values ---
    _defaults = {
        'dataset': 'nhp',
        'mode': ['fit'],
        'device': 'cuda',
        'budget': 100,
        'n_reps': 30,
        'budgets': [10, 30, 50, 100, 200],
        'kappa_0': 2.5,
        'kappa_min': 0.5,
        'seed': 42,
    }
    for key, default in _defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)

    # --- Validate modes ---
    invalid = set(args.mode) - _VALID_MODES
    if invalid:
        parser.error(
            f"Invalid mode(s): {', '.join(sorted(invalid))}. "
            f"Valid: {', '.join(sorted(_VALID_MODES))}"
        )

    # --- Run ---
    run_vanilla_benchmark(
        dataset_type=args.dataset,
        mode=args.mode,
        device=args.device,
        budget=args.budget,
        n_reps=args.n_reps,
        budgets=args.budgets,
        kappa_0=args.kappa_0,
        kappa_min=args.kappa_min,
        held_out_subj_idx=args.held_out_subj,
        save=args.save,
        seed=args.seed,
    )


if __name__ == '__main__':
    run_benchmark()
