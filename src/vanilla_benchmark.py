"""Vanilla TabPFN v2 vs GP benchmark (Hypothesis A).

CLI entry point that evaluates pretrained TabPFNRegressor (v2) against
ExactGP (RBF kernel) on neurostimulation datasets.  No finetuning is
performed -- this replaces the deleted ``src/main.py`` which incorrectly
used PFNs4BO ``TransformerBOMethod`` (PFN v1).

Three evaluation modes:
  - ``optimization``       -- Cumulative regret + timing via BO loop (A2, A3);
                              R² of the surrogate's final prediction is
                              reported as a secondary metric.
  - ``optimization_budget``-- Budget sweep over multiple query budgets (A4)
  - ``kappa_search``       -- Fixed-kappa hyperparameter search

Usage::

    python src/vanilla_benchmark.py --dataset nhp --mode optimization --save
    python src/vanilla_benchmark.py --config configs/nhp_vanilla_benchmark.yaml
    python src/vanilla_benchmark.py --config configs/nhp_vanilla_benchmark.yaml --budget 50
"""
import argparse
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tabpfn import TabPFNRegressor

from evaluation import evaluate_optimization, _unpack_budget_trajectory
from models.regressors import GPSurrogate, TabPFNSurrogate
from utils.data_utils import (
    load_data,
    HELD_OUT_SUBJECTS,
    ALL_SUBJECTS,
    generate_experiment_tag,
    save_results,
    create_run_dir,
    write_run_config,
    load_subject_result,
    save_subject_result,
)
from utils.visualization import (
    r2_by_subject,
    regret_with_timing,
    regret_by_subject,
    regret_by_emg,
    exploration_by_subject,
    exploration_by_emg,
    budget_sweep_plot,
    regret_curve,
    visualize_representation,
    show_emg_map,
    kappa_regret_curves,
    kappa_auc_bar,
    plot_gfs_profile,
    gfs_by_subject,
)


_VALID_MODES: frozenset = frozenset({'optimization', 'optimization_budget', 'kappa_search'})

# Default kappa values for kappa hyperparameter search
_DEFAULT_KAPPA_VALUES: List[float] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0]


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
    subjects_mode: str = 'held_out',
) -> List[tuple]:
    """Build the list of (subject_idx, emg_idx) pairs to evaluate.

    Subject selection priority (highest to lowest):
    1. ``held_out_subj_idx`` — single subject (SLURM job-array mode)
    2. ``subjects_mode='all'`` — all valid subjects (LOO cross-validation)
    3. ``subjects_mode='held_out'`` (default) — ``HELD_OUT_SUBJECTS`` only

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        held_out_subj_idx: If provided, restrict evaluation to this single
            subject index (overrides ``subjects_mode``).
        subjects_mode: ``'held_out'`` uses ``HELD_OUT_SUBJECTS[dataset_type]``;
            ``'all'`` uses ``ALL_SUBJECTS[dataset_type]`` for LOO cross-validation.

    Returns:
        List of (subject_idx, emg_idx) tuples.
    """
    if held_out_subj_idx is not None:
        subjects = [held_out_subj_idx]
    elif subjects_mode == 'all':
        subjects = ALL_SUBJECTS[dataset_type]
    else:
        subjects = HELD_OUT_SUBJECTS[dataset_type]
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


def _vanilla_optimization(
    dataset_type: str,
    experiments: List[tuple],
    device: str,
    budget: int,
    n_reps: int,
    kappa_schedule: float,
    run_dir: str,
    exp_tag: str,
    save: bool,
    save_tag: str,
    metadata: Optional[Dict[str, Any]] = None,
    acq_fn: str = 'ucb',
    ts_temperature: float = 1.0,
) -> Dict[str, list]:
    """Run BO optimization evaluation for vanilla TabPFN and GP.

    Uses ``evaluate_optimization()`` (unified pipeline) for both models with
    the same ``kappa_schedule`` and acquisition function.

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        experiments: List of (subject_idx, emg_idx).
        device: PyTorch device string.
        budget: Total BO query budget.
        n_reps: Repetitions per experiment.
        kappa_schedule: UCB coefficient for both TabPFN and GP.  ``0.0`` = auto
            cosine-annealed schedule; any other value = fixed kappa throughout.
        run_dir: Output directory.
        exp_tag: Tag suffix for plot filenames.
        save: If True, persist results.
        save_tag: Full experiment tag.
        metadata: Optional provenance dict written into saved pkl files.
        acq_fn: Acquisition function — ``'ucb'`` or ``'ts'`` (Thompson Sampling).
        ts_temperature: Temperature for TabPFN bar-distribution TS sampling.

    Returns:
        ``{'GP': list[dict], 'TabPFN': list[dict]}``.
    """
    results_tabpfn: List[dict] = []
    results_gp: List[dict] = []

    _vanilla_cache_params = {
        'model': 'vanilla_tabpfn',
        'budget': budget,
        'n_reps': n_reps,
        'kappa_schedule': kappa_schedule,
        'acq_fn': acq_fn,
        'ts_temperature': ts_temperature if acq_fn == 'ts' else None,
        'normalization': 'pfn',
    }
    _gp_cache_params = {
        'model': 'gp',
        'budget': budget,
        'n_reps': n_reps,
        'kappa_schedule': kappa_schedule,
        'acq_fn': acq_fn,
        'ts_temperature': ts_temperature if acq_fn == 'ts' else None,
        'normalization': 'gp',
    }

    for subj_idx, emg_idx in experiments:
        print(f"  Optimization: subject={subj_idx}, emg={emg_idx}")

        res_tabpfn = load_subject_result(
            dataset_type, subj_idx, emg_idx, 'vanilla_tabpfn', _vanilla_cache_params
        )
        if res_tabpfn is not None:
            print(f"    [CACHE HIT] vanilla_tabpfn subject={subj_idx}, emg={emg_idx}")
        else:
            tabpfn_base = _build_vanilla_tabpfn(device, n_estimators=1)
            tabpfn_surrogate = TabPFNSurrogate(model=tabpfn_base)
            res_tabpfn = evaluate_optimization(
                surrogate=tabpfn_surrogate,
                dataset_type=dataset_type,
                subject_idx=subj_idx,
                emg_idx=emg_idx,
                device=device,
                budget=budget,
                n_reps=n_reps,
                kappa_schedule=kappa_schedule,
                normalization='pfn',
                acq_fn=acq_fn,
                ts_temperature=ts_temperature,
            )
            res_tabpfn['model_type'] = 'vanilla_tabpfn'
            save_subject_result(
                res_tabpfn, dataset_type, subj_idx, emg_idx,
                'vanilla_tabpfn', _vanilla_cache_params
            )

        res_gp = load_subject_result(
            dataset_type, subj_idx, emg_idx, 'gp', _gp_cache_params
        )
        if res_gp is not None:
            print(f"    [CACHE HIT] GP subject={subj_idx}, emg={emg_idx}")
        else:
            gp_surrogate = GPSurrogate(device=device)
            res_gp = evaluate_optimization(
                surrogate=gp_surrogate,
                dataset_type=dataset_type,
                subject_idx=subj_idx,
                emg_idx=emg_idx,
                device=device,
                budget=budget,
                n_reps=n_reps,
                kappa_schedule=kappa_schedule,
                normalization='gp',
                acq_fn=acq_fn,
                ts_temperature=ts_temperature,
            )
            res_gp['model_type'] = 'gp'
            save_subject_result(
                res_gp, dataset_type, subj_idx, emg_idx, 'gp', _gp_cache_params
            )

        results_tabpfn.append(res_tabpfn)
        results_gp.append(res_gp)
        print(f"    TabPFN R2={np.mean(res_tabpfn['r2']):.3f}  |  "
              f"GP R2={np.mean(res_gp['r2']):.3f}")

    results_dict = {'GP': results_gp, 'TabPFN': results_tabpfn}

    # --- Plots ---
    r2_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
    regret_with_timing(results_dict, split_type=exp_tag, save=True,
                       output_dir=run_dir)
    regret_by_subject(results_dict, split_type=exp_tag, save=True,
                      output_dir=run_dir)
    regret_by_emg(results_dict, split_type=exp_tag, save=True,
                  output_dir=run_dir)
    exploration_by_subject(results_dict, split_type=exp_tag, save=True,
                           output_dir=run_dir)
    exploration_by_emg(results_dict, split_type=exp_tag, save=True,
                       output_dir=run_dir)
    visualize_representation(results_dict, mode=f'_vanilla_{exp_tag}',
                             save=True, output_dir=run_dir)
    for _res in results_gp:
        show_emg_map(_res, model_type='GP', mode=f'_vanilla_{exp_tag}',
                     save=True, output_dir=run_dir)
    for _res in results_tabpfn:
        show_emg_map(_res, model_type='TabPFN', mode=f'_vanilla_{exp_tag}',
                     save=True, output_dir=run_dir)

    _all_results_flat = results_gp + results_tabpfn
    if any(r.get('gfs') is not None for r in _all_results_flat):
        plot_gfs_profile(_all_results_flat, dataset=dataset_type,
                         save=True, output_dir=run_dir)
        gfs_by_subject(_all_results_flat, save=True, output_dir=run_dir)

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
    kappa_schedule: float,
    run_dir: str,
    exp_tag: str,
    save: bool,
    save_tag: str,
    acq_fn: str = 'ucb',
    ts_temperature: float = 1.0,
) -> pd.DataFrame:
    """Budget sweep: infer sub-budget performance from a single max-budget run.

    Runs ``evaluate_optimization`` once per (model, subject, emg) at
    ``budget=max(budgets)`` with ``capture_all_snapshots=True`` and
    ``snapshot_iters_override=budgets``.  Sub-budget regret is read from the
    stored trajectory (``result['values'][:, :b]``) and per-rep R² from
    ``result['r2_by_snapshot'][b]``.  This eliminates the N_budgets×n_reps
    redundant BO runs of the old outer-loop approach.

    Note: Because UCB and TS are myopic (each step is independently optimal
    given the current posterior), early-step choices under the max-budget
    trajectory are equivalent to re-running with the shorter budget.  This
    single-trajectory inference is exact for both acquisition functions.

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        experiments: List of (subject_idx, emg_idx).
        device: PyTorch device string.
        budgets: Sub-budget values to evaluate (also used as ``snapshot_iters``).
        n_reps: Repetitions per (subject, emg) pair at the max budget.
        kappa_schedule: UCB coefficient for both TabPFN and GP.  ``0.0`` = auto
            cosine-annealed schedule; any other value = fixed kappa throughout.
        run_dir: Output directory.
        exp_tag: Tag suffix.
        save: If True, persist DataFrame as pkl.
        save_tag: Full experiment tag.
        acq_fn: Acquisition function — ``'ucb'`` or ``'ts'`` (Thompson Sampling).
        ts_temperature: Temperature for TabPFN bar-distribution TS sampling.

    Returns:
        Long-form DataFrame with columns: Budget, Model, Regret, R2, ID.
    """
    plot_data: List[dict] = []
    max_b = max(budgets)

    for subj_idx, emg_idx in experiments:
        print(f"\n  Budget sweep: subject={subj_idx}, emg={emg_idx} (max_budget={max_b})")

        # --- TabPFN: single run at max_budget with snapshots at all sub-budgets ---
        tabpfn_base = _build_vanilla_tabpfn(device, n_estimators=1)
        tabpfn_surrogate = TabPFNSurrogate(model=tabpfn_base)
        res_tabpfn = evaluate_optimization(
            surrogate=tabpfn_surrogate,
            dataset_type=dataset_type,
            subject_idx=subj_idx,
            emg_idx=emg_idx,
            device=device,
            budget=max_b,
            n_reps=n_reps,
            kappa_schedule=kappa_schedule,
            normalization='pfn',
            acq_fn=acq_fn,
            ts_temperature=ts_temperature,
            capture_all_snapshots=True,
            snapshot_iters_override=budgets,
        )
        _unpack_budget_trajectory(plot_data, res_tabpfn, 'TabPFN', budgets, subj_idx, emg_idx)

        # --- GP: single run at max_budget with snapshots at all sub-budgets ---
        gp_surrogate = GPSurrogate(device=device)
        res_gp = evaluate_optimization(
            surrogate=gp_surrogate,
            dataset_type=dataset_type,
            subject_idx=subj_idx,
            emg_idx=emg_idx,
            device=device,
            budget=max_b,
            n_reps=n_reps,
            kappa_schedule=kappa_schedule,
            normalization='gp',
            acq_fn=acq_fn,
            ts_temperature=ts_temperature,
            capture_all_snapshots=True,
            snapshot_iters_override=budgets,
        )
        _unpack_budget_trajectory(plot_data, res_gp, 'GP', budgets, subj_idx, emg_idx)

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

def _vanilla_kappa_search(
    dataset_type: str,
    experiments: List[tuple],
    device: str,
    budget: int,
    n_reps: int,
    kappa_values: List[float],
    run_dir: str,
    exp_tag: str,
    save: bool,
    save_tag: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Kappa hyperparameter search for vanilla TabPFN BO.

    Evaluates TabPFN with each value in *kappa_values* as a fixed UCB
    coefficient, then runs GP with the auto schedule as a reference.
    Produces an aggregated regret-curve plot (one line per kappa) and an
    AUC bar chart — both saved as SVG.

    The search is intended to run over ALL subjects for a dataset (i.e.,
    pass experiments built from ``ALL_SUBJECTS``).  The results help the
    practitioner select a single fixed kappa appropriate for clinical use.

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        experiments: List of (subject_idx, emg_idx) from ``_build_experiments``
            with ``subjects_mode='all'``.
        device: PyTorch device string.
        budget: Total BO query budget per experiment.
        n_reps: Repetitions per (subject, emg) pair.
        kappa_values: Fixed kappa values to evaluate (must be non-zero).
        run_dir: Output directory from ``create_run_dir()``.
        exp_tag: Experiment tag suffix for filenames.
        save: If True, persist raw results and AUC CSV.
        save_tag: Full experiment tag from ``generate_experiment_tag()``.
        metadata: Optional provenance dict written into saved pkl files.

    Returns:
        Dict with keys:
          - ``'kappa_results'``: ``{kappa: [result_dicts]}`` for each kappa.
          - ``'gp_results'``: ``[result_dicts]`` for the GP reference.
          - ``'auc_df'``: ``pd.DataFrame`` with columns
            ``['kappa', 'model', 'mean_auc', 'std_auc']``.
    """
    kappa_results: Dict[float, List[dict]] = {}
    gp_results: List[dict] = []

    # --- GP reference (run once, auto kappa schedule) ---
    print('\n  [kappa_search] Running GP reference (auto kappa schedule)...')
    for subj_idx, emg_idx in experiments:
        gp_surrogate = GPSurrogate(device=device)
        res_gp = evaluate_optimization(
            surrogate=gp_surrogate,
            dataset_type=dataset_type,
            subject_idx=subj_idx,
            emg_idx=emg_idx,
            device=device,
            budget=budget,
            n_reps=n_reps,
            kappa_schedule=0.0,
            normalization='gp',
        )
        res_gp['model_type'] = 'gp'
        gp_results.append(res_gp)
        print(f"    GP  subject={subj_idx}, emg={emg_idx}  "
              f"R2={np.mean(res_gp['r2']):.3f}")

    # --- TabPFN with each fixed kappa ---
    for kappa in kappa_values:
        if kappa == 0.0:
            raise ValueError(
                f"kappa_values must not contain 0.0 (reserved for auto schedule). "
                f"Got kappa_values={kappa_values}."
            )
        print(f'\n  [kappa_search] TabPFN kappa={kappa:.3g}')
        kappa_results[kappa] = []
        for subj_idx, emg_idx in experiments:
            tabpfn_base = _build_vanilla_tabpfn(device, n_estimators=1)
            tabpfn_surrogate = TabPFNSurrogate(model=tabpfn_base)
            res = evaluate_optimization(
                surrogate=tabpfn_surrogate,
                dataset_type=dataset_type,
                subject_idx=subj_idx,
                emg_idx=emg_idx,
                device=device,
                budget=budget,
                n_reps=n_reps,
                kappa_schedule=kappa,
                normalization='pfn',
            )
            res['model_type'] = 'vanilla_tabpfn'
            kappa_results[kappa].append(res)
            print(f"    κ={kappa:.3g}  subject={subj_idx}, emg={emg_idx}  "
                  f"R2={np.mean(res['r2']):.3f}")
            
    def _compute_auc(results_list: List[dict]) -> tuple:
        """Compute mean and std AUC (area under normalized regret curve) across experiments.

        Each experiment's regret curve is normalized by the response range before
        averaging, so channels with different absolute magnitudes contribute equally.
        AUC is defined as the mean regret over all BO steps (budget-agnostic).

        Args:
            results_list: List of result dicts from ``evaluate_optimization()``,
                each containing ``'values'`` ([n_reps, budget]) and ``'y_test'``.

        Returns:
            Tuple ``(mean_auc, std_auc)`` across experiments.
        """
        auc_values: List[float] = []
        for res in results_list:
            if 'values' not in res or 'y_test' not in res:
                continue
            y_range = float(res['y_test'].max() - res['y_test'].min())
            if y_range < 1e-8:
                continue
            optimal = float(res['y_test'].max())
            running_best = np.maximum.accumulate(
                np.array(res['values']), axis=1
            )                                           # [n_reps, budget]
            regret_norm = (optimal - running_best) / y_range  # [n_reps, budget]
            # Mean over steps, then over reps → single AUC per experiment
            auc_values.append(float(np.mean(regret_norm)))
        if not auc_values:
            return 0.0, 0.0
        return float(np.mean(auc_values)), float(np.std(auc_values))

    # --- Compute AUC ---
    auc_rows: List[dict] = []
    for kappa, res_list in sorted(kappa_results.items()):
        mean_auc, std_auc = _compute_auc(res_list)
        auc_rows.append({'kappa': kappa, 'model': 'TabPFN',
                         'mean_auc': mean_auc, 'std_auc': std_auc})
    gp_mean_auc, gp_std_auc = _compute_auc(gp_results)
    auc_rows.append({'kappa': 0.0, 'model': 'GP',
                     'mean_auc': gp_mean_auc, 'std_auc': gp_std_auc})
    auc_df = pd.DataFrame(auc_rows)

    tabpfn_auc_df = auc_df[auc_df['model'] == 'TabPFN'].reset_index(drop=True)
    print(f'\n[kappa_search] AUC summary (dataset={dataset_type}):')
    print(tabpfn_auc_df[['kappa', 'mean_auc', 'std_auc']].to_string(index=False))
    print(f'  GP reference AUC: {gp_mean_auc:.4f} ± {gp_std_auc:.4f}')

    # --- Plots ---
    kappa_regret_curves(
        kappa_results={k: v for k, v in kappa_results.items()},
        gp_results=gp_results,
        dataset=dataset_type,
        split_type=exp_tag,
        save=True,
        output_dir=run_dir,
    )
    kappa_auc_bar(
        auc_df=tabpfn_auc_df,
        gp_auc=gp_mean_auc,
        dataset=dataset_type,
        split_type=exp_tag,
        save=True,
        output_dir=run_dir,
    )

    # --- Persistence ---
    if save:
        results_dir = os.path.join(run_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        auc_csv = os.path.join(results_dir, f'{save_tag}_kappa_search_auc.csv')
        auc_df.to_csv(auc_csv, index=False)
        print(f'Saved AUC CSV -> {auc_csv}')

        import pickle
        raw_pkl = os.path.join(results_dir, f'{save_tag}_kappa_search_raw.pkl')
        with open(raw_pkl, 'wb') as f:
            pickle.dump({
                'kappa_results': kappa_results,
                'gp_results': gp_results,
                '_metadata': metadata or {},
            }, f)
        print(f'Saved raw results pkl -> {raw_pkl}')

    return {
        'kappa_results': kappa_results,
        'gp_results': gp_results,
        'auc_df': auc_df,
    }


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
    kappa_schedule: float = 0.0,
    kappa_values: Optional[List[float]] = None,
    held_out_subj_idx: Optional[int] = None,
    subjects_mode: str = 'held_out',
    save: bool = False,
    seed: int = 42,
    acq_fn: str = 'ucb',
    ts_temperature: float = 1.0,
) -> Dict[str, Any]:
    """Orchestrate vanilla TabPFN vs GP benchmark for one or more modes.

    Sets random seeds, builds the experiment list, creates a run directory
    with ``config.json``, then dispatches to mode-specific inner functions.

    Subject selection (highest-priority first):
    - ``held_out_subj_idx`` given → single subject (SLURM job-array)
    - ``subjects_mode='all'`` → ``ALL_SUBJECTS`` (LOO cross-validation)
    - ``subjects_mode='held_out'`` (default) → ``HELD_OUT_SUBJECTS``

    Note: ``'kappa_search'`` mode always overrides subject selection to
    ``ALL_SUBJECTS`` regardless of ``subjects_mode`` or ``held_out_subj_idx``.

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        mode: List of evaluation modes (see ``_VALID_MODES``).
        device: PyTorch device string.
        budget: BO query budget.
        n_reps: Repetitions per experiment.
        budgets: Budget list for ``'optimization_budget'`` mode.
            Defaults to ``[10, 30, 50, 100, 200]``.
        kappa_schedule: UCB coefficient for TabPFN BO.
            ``0.0`` (default) = auto cosine-annealed schedule from GP-UCB theory.
            Any other value = fixed kappa throughout the BO loop.
        kappa_values: Fixed kappa values for ``'kappa_search'`` mode.
            Defaults to ``[0.5, 1.0, 2.0, 3.0, 5.0]``.  Must not contain 0.0.
        held_out_subj_idx: If given, restrict evaluation to this single subject
            (used by SLURM job-array; overrides ``subjects_mode``).
        subjects_mode: ``'held_out'`` (default) or ``'all'`` for LOO.
        save: If True, persist results (pkl + CSV).
        seed: Master random seed.

    Returns:
        Dict keyed by mode name.  Values are either
        ``dict[str, list[dict]]`` (optimization),
        ``pd.DataFrame`` (optimization_budget), or
        ``dict`` (kappa_search).

    Raises:
        ValueError: If mode contains invalid values or dataset_type is unknown.
    """
    # --- Validate ---
    if dataset_type not in ('rat', 'nhp', 'spinal'):
        raise ValueError(
            f"Unknown dataset_type={dataset_type!r}. Use 'rat', 'nhp', or 'spinal'."
        )
    invalid = set(mode) - _VALID_MODES
    if invalid:
        raise ValueError(
            f"Invalid mode(s): {', '.join(sorted(invalid))}. "
            f"Valid: {', '.join(sorted(_VALID_MODES))}"
        )
    if budgets is None:
        budgets = [10, 30, 50, 100, 200]
    if kappa_values is None:
        kappa_values = list(_DEFAULT_KAPPA_VALUES)

    # --- Seeds ---
    set_seed(seed)

    # --- Build experiment list ---
    # kappa_search always runs over ALL subjects for a dataset-wide estimate.
    _kappa_search_mode = 'kappa_search' in mode
    if _kappa_search_mode:
        kappa_search_experiments = _build_experiments(
            dataset_type, held_out_subj_idx=None, subjects_mode='all'
        )
        _ks_subjects = list(ALL_SUBJECTS[dataset_type])
        print(f"[INFO] kappa_search: {len(kappa_search_experiments)} experiments "
              f"across ALL subjects {_ks_subjects}")

    experiments = _build_experiments(
        dataset_type,
        held_out_subj_idx=held_out_subj_idx,
        subjects_mode=subjects_mode,
    )
    if held_out_subj_idx is not None:
        _eval_subjects = [held_out_subj_idx]
    elif subjects_mode == 'all':
        _eval_subjects = list(ALL_SUBJECTS[dataset_type])
    else:
        _eval_subjects = list(HELD_OUT_SUBJECTS[dataset_type])
    print(f"[INFO] {len(experiments)} experiments ({len(_eval_subjects)} subject(s): {_eval_subjects})")

    # --- Experiment tag and run directory ---
    tag_config: Dict[str, Any] = {
        'dataset_type': dataset_type,
        'mode': sorted(mode),
        'budget': budget,
        'n_reps': n_reps,
        'kappa_schedule': kappa_schedule,
        'seed': seed,
    }
    if held_out_subj_idx is not None:
        tag_config['held_out_subj_idx'] = held_out_subj_idx
    save_tag = generate_experiment_tag(dataset_type, 'vanilla-benchmark', tag_config)
    exp_tag = 'vanilla_benchmark'

    run_dir = create_run_dir(save_tag, tag=save_tag)
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
        'kappa_schedule': kappa_schedule,
        'kappa_values': kappa_values,
        'seed': seed,
        'held_out_subj': held_out_subj_idx,
        'subjects_mode': subjects_mode,
        'eval_subjects': _eval_subjects,
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
    _t0 = time.time()

    for m in mode:
        print(f"\n{'=' * 60}")
        print(f"Running mode: {m}")
        print('=' * 60)

        if m == 'optimization':
            all_results['optimization'] = _vanilla_optimization(
                dataset_type, experiments, device, budget, n_reps,
                kappa_schedule,
                run_dir, exp_tag, save, save_tag,
                metadata=_metadata,
                acq_fn=acq_fn,
                ts_temperature=ts_temperature,
            )

        elif m == 'optimization_budget':
            all_results['optimization_budget'] = _vanilla_optimization_budget(
                dataset_type, experiments, device, budgets, n_reps,
                kappa_schedule,
                run_dir, exp_tag, save, save_tag,
                acq_fn=acq_fn,
                ts_temperature=ts_temperature,
            )

        elif m == 'kappa_search':
            all_results['kappa_search'] = _vanilla_kappa_search(
                dataset_type, kappa_search_experiments, device, budget, n_reps,
                kappa_values,
                run_dir, exp_tag, save, save_tag,
                metadata=_metadata,
            )

    # --- Update config with total wall time and re-write ---
    run_config['total_wall_time_s'] = round(time.time() - _t0, 2)
    write_run_config(run_dir, run_config)

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
                        choices=['rat', 'nhp', 'spinal'],
                        help='Dataset type (default: nhp)')
    parser.add_argument('--mode', type=lambda s: s.split(','), default=None,
                        metavar='MODE[,MODE,...]',
                        help='Comma-separated modes: optimization, optimization_budget,\n'
                             'kappa_search (default: optimization)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu or cuda (default: cuda)')
    parser.add_argument('--budget', type=int, default=None,
                        help='BO query budget (default: 100)')
    parser.add_argument('--n_reps', type=int, default=None,
                        help='Repetitions per experiment (default: 30)')
    parser.add_argument('--budgets', type=int, nargs='+', default=None,
                        help='Budget sweep values for optimization_budget\n'
                             '(default: 10 30 50 100 200)')
    parser.add_argument('--kappa_schedule', type=float, default=None,
                        help='UCB coefficient for TabPFN BO.\n'
                             '  0.0 (default) = auto cosine-annealed schedule (GP-UCB theory)\n'
                             '  any other value = fixed kappa throughout the BO loop')
    parser.add_argument('--acq_fn', type=str, default=None,
                        choices=['ucb', 'ts'],
                        help="Acquisition function: 'ucb' (default, UCB with optional kappa schedule)\n"
                             "  or 'ts' (Thompson Sampling from predictive posterior).")
    parser.add_argument('--ts_temperature', type=float, default=None,
                        help='Temperature for TabPFN bar-distribution sampling in TS mode.\n'
                             '  1.0 (default) = exact predictive distribution;\n'
                             '  <1.0 = sharper / greedier; >1.0 = more uniform / exploratory.')
    parser.add_argument('--kappa_values', type=float, nargs='+', default=None,
                        help='Fixed kappa values for kappa_search mode\n'
                             '(default: 0.5 1.0 2.0 3.0 5.0)')
    parser.add_argument('--subjects', type=str, default=None,
                        choices=['held_out', 'all'],
                        dest='subjects_mode',
                        help='Subject selection mode:\n'
                             '  held_out (default) — evaluate HELD_OUT_SUBJECTS only\n'
                             '  all — evaluate ALL_SUBJECTS (LOO cross-validation)')
    parser.add_argument('--held_out_subj', type=int, default=None,
                        help='Restrict evaluation to a single subject index.\n'
                             'Overrides --subjects. Used by SLURM job arrays\n'
                             '(one job per subject).')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Persist results to output/runs/<tag>/results/')
    parser.add_argument('--cluster-diag', action='store_true', default=False,
                        dest='cluster_diag',
                        help='Print HPC efficiency summary at job end (GPU memory, walltime, '
                             'utilisation, grade, warnings). Also activated by CLUSTER_DIAG=1 '
                             'env var. Designed for SLURM .out log visibility.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Master random seed (default: 42)')

    args = parser.parse_args()

    # --- YAML config loading ---
    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        _bool_flags = {'save', 'cluster_diag'}
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
        'mode': ['optimization'],
        'device': 'cuda',
        'budget': 100,
        'n_reps': 30,
        'budgets': [10, 30, 50, 100, 200],
        'kappa_schedule': 0.0,
        'kappa_values': list(_DEFAULT_KAPPA_VALUES),
        'subjects_mode': 'held_out',
        'seed': 42,
        'acq_fn': 'ucb',
        'ts_temperature': 1.0,
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

    # Activate cluster diagnostics via env var even when flag not passed
    import os as _os
    if not args.cluster_diag and _os.environ.get('CLUSTER_DIAG', '0') == '1':
        args.cluster_diag = True

    # --- Run ---
    _experiments = _build_experiments(
        args.dataset,
        held_out_subj_idx=args.held_out_subj,
        subjects_mode=args.subjects_mode,
    )
    from utils.cluster_diagnostics import ClusterDiagnostics as _CD
    with _CD(tag=f"{args.dataset}-vanilla-benchmark", device=args.device,
             n_planned=len(_experiments) * len(args.mode),
             enabled=args.cluster_diag) as _diag:
        run_vanilla_benchmark(
            dataset_type=args.dataset,
            mode=args.mode,
            device=args.device,
            budget=args.budget,
            n_reps=args.n_reps,
            budgets=args.budgets,
            kappa_schedule=args.kappa_schedule,
            kappa_values=args.kappa_values,
            held_out_subj_idx=args.held_out_subj,
            subjects_mode=args.subjects_mode,
            save=args.save,
            seed=args.seed,
            acq_fn=args.acq_fn,
            ts_temperature=args.ts_temperature,
        )
        _diag.record_experiment(n_completed=len(_experiments) * len(args.mode))


if __name__ == '__main__':
    run_benchmark()
