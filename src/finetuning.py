"""
Fine-tuning orchestration for TabPFN on neurostimulation data.

Two-phase workflow:
  1. Backprop finetuning: finetune_tabpfn() trains a GradientMonitoredRegressor,
     adapting pretrained weights to neurostimulation data via gradient updates.
     Diagnostics (gradient/weight metrics, CKA) are always collected.
  2. Extraction for evaluation: extract_inference_model() deep-copies the
     internal TabPFNRegressor with finetuned weights. This standalone model
     uses in-context learning (.fit() stores context, no gradients) and
     supports the full predict API including output_type="quantiles".

Usage:
    python finetuning.py --dataset rat --device cuda --epochs 30
    python finetuning.py --dataset nhp --device cuda --epochs 30
    python finetuning.py --dataset nhp --mode optimization --budget 100 --n_reps 20
"""
import argparse
import os
import random
from datetime import datetime

import numpy as np

from models.regressors import _make_finetuned_regressor, extract_inference_model
from evaluation import (
    gp_baseline, finetuned_fit, finetuned_optimization,
    finetuned_fit_budget, finetuned_optimization_budget,
    finetuned_percentage, load_sweep_results,
)
from utils.data_utils import (
    build_finetuning_dataset, load_data,
    HELD_OUT_SUBJECTS, TRAIN_SUBJECTS, ALL_SUBJECTS,
    generate_experiment_tag, save_results,
    create_run_dir, write_run_config,
)
from utils.visualization import (
    r2_per_muscle, r2_by_subject, show_emg_map,
    regret_with_timing, regret_by_subject, regret_by_emg,
    budget_sweep_plot, augmentation_sweep_plot,
    visualize_representation,
    plot_gradient_metrics, plot_weight_metrics, plot_cka_similarity,
)


def finetune_tabpfn(dataset_type, device='cuda', epochs=1, lr=1e-5,
                    n_augmentations=25, subject_indices=None,
                    held_out_emg_idx=None, seed=42,
                    silence_diagnostics=True, output_dir=None):
    """
    Fine-tune a TabPFNRegressor on augmented neurostimulation data.

    Args:
        dataset_type: 'rat' or 'nhp'
        device: 'cpu' or 'cuda'
        epochs: number of fine-tuning epochs
        lr: learning rate
        n_augmentations: augmentations per subject-EMG pair
        subject_indices: list of subject indices to train on (None = all training subjects)
        held_out_emg_idx: EMG index to exclude from training data
        seed: random seed for dataset building
        silence_diagnostics: if True (default), skip gradient/CKA monitoring for faster
            finetuning and lower memory. If False, use GradientMonitoredRegressor.
        output_dir: when set, saves diagnostic plots to {output_dir}/diagnostics/

    Returns:
        (ft_model_raw, ft_model) tuple:
          - ft_model_raw: the finetuned regressor (with _diagnostics_ when diagnostics enabled)
          - ft_model: extracted TabPFNRegressor for in-context learning
    """
    print(f"Building augmented dataset for '{dataset_type}' ...")
    X_train, y_train = build_finetuning_dataset(
        dataset_type,
        subject_indices=subject_indices,
        held_out_emg_idx=held_out_emg_idx,
        n_augmentations=n_augmentations,
        seed=seed,
    )
    print(f"  Dataset size: {X_train.shape[0]} rows, {X_train.shape[1]} features")

    print(f"Initializing finetuned regressor (epochs={epochs}, lr={lr}) ...")

    ft_model_raw = _make_finetuned_regressor(
        silence_diagnostics=silence_diagnostics,
        device=device,
        epochs=epochs,
        learning_rate=lr,
        n_estimators_finetune=8,
        n_estimators_validation=8,
        n_estimators_final_inference=8,
    )

    print("Fine-tuning ...")
    ft_model_raw.fit(X_train, y_train)

    # Always save diagnostic plots when diagnostics are available
    if hasattr(ft_model_raw, '_diagnostics_') and ft_model_raw._diagnostics_:
        diag_dir = os.path.join(output_dir, 'diagnostics') if output_dir else None
        plot_gradient_metrics(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
        plot_weight_metrics(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
        plot_cka_similarity(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)

    print("Fine-tuning complete.")

    ft_model = extract_inference_model(ft_model_raw)
    return ft_model_raw, ft_model


# ============================================
#       High-Level Experiment Runner
# ============================================

_VALID_MODES = {'fit', 'optimization', 'fit_budget', 'optimization_budget'}


def run_experiment(
    dataset_type,
    split_type='inter_subject',
    mode=None,
    device='cuda',
    budget=100,
    n_reps=30,
    epochs=50,
    lr=1e-5,
    n_augmentations=25,
    held_out_emg_idx=None,
    held_out_subj_idx=None,
    budgets=None,
    save=False,
    silence_diagnostics=True,
):
    """
    Unified entry point for transfer learning evaluation.

    Args:
        dataset_type: 'rat' or 'nhp'
        split_type: 'inter_subject' — train on TRAIN_SUBJECTS, test on HELD_OUT_SUBJECTS;
                    'intra_emg'     — train on ALL_SUBJECTS (excluding held_out_emg_idx),
                                      test on that EMG across ALL_SUBJECTS.
        mode: str or list of str — any combination of 'fit', 'optimization',
              'fit_budget', 'optimization_budget'. The model is finetuned once
              and all requested modes are evaluated sequentially.
        device: 'cpu' or 'cuda'
        budget: number of training points (fit) or BO queries (optimization)
        n_reps: number of repetitions per experiment
        epochs: fine-tuning epochs
        lr: fine-tuning learning rate
        n_augmentations: augmentations per subject-EMG pair
        held_out_emg_idx: required when split_type='intra_emg'; the EMG index
            held out from training and used as the test set.
        held_out_subj_idx: optional int. When set, overrides the default subject
            split: trains on all subjects except this one and tests on it alone.
        budgets: list of budgets for 'fit_budget' / 'optimization_budget' modes.
        save: if True, persist results to output/results/ (pkl + CSV summary).
        silence_diagnostics: if True (default), skip gradient/CKA monitoring.

    Returns:
        dict keyed by mode name, each value being the result of that mode
        ('fit'/'optimization' → {'TabPFN': [...], 'GP': [...]},
         budget modes → DataFrame).
    """
    if mode is None:
        mode = ['fit']
    if isinstance(mode, str):
        mode = [mode]
    invalid = set(mode) - _VALID_MODES
    if invalid:
        raise ValueError(f"Unknown mode(s): {invalid}. Valid: {_VALID_MODES}")

    if budgets is None:
        budgets = [10, 50, 100, 150, 200]

    # --- Resolve train / test sets ---
    if split_type == 'inter_subject':
        if held_out_subj_idx is not None:
            train_subject_indices = [s for s in ALL_SUBJECTS[dataset_type] if s != held_out_subj_idx]
            test_subjects = [held_out_subj_idx]
        else:
            train_subject_indices = TRAIN_SUBJECTS[dataset_type]
            test_subjects = HELD_OUT_SUBJECTS[dataset_type]
        test_emg_indices = None
        ft_held_out_emg = None
    elif split_type == 'intra_emg':
        if held_out_emg_idx is None:
            raise ValueError("held_out_emg_idx must be set when split_type='intra_emg'")
        if held_out_subj_idx is not None:
            train_subject_indices = [s for s in ALL_SUBJECTS[dataset_type] if s != held_out_subj_idx]
            test_subjects = [held_out_subj_idx]
        else:
            train_subject_indices = ALL_SUBJECTS[dataset_type]
            test_subjects = ALL_SUBJECTS[dataset_type]
        test_emg_indices = [held_out_emg_idx]
        ft_held_out_emg = held_out_emg_idx
    else:
        raise ValueError(f"Unknown split_type={split_type!r}. Use 'inter_subject' or 'intra_emg'.")

    # --- Build experiment tag for unique filenames ---
    tag_parts = [split_type]
    if held_out_subj_idx is not None:
        tag_parts.append(f'subj{held_out_subj_idx}')
    if held_out_emg_idx is not None:
        tag_parts.append(f'emg{held_out_emg_idx}')
    exp_tag = '_'.join(tag_parts)

    _save_tag = generate_experiment_tag(
        dataset_type, split_type, epochs, lr, n_augmentations,
        held_out_subj_idx=held_out_subj_idx,
        held_out_emg_idx=held_out_emg_idx,
    )

    # --- Build test experiment list ---
    experiments = []
    for subj_idx in test_subjects:
        data = load_data(dataset_type, subj_idx)
        n_emgs = data['sorted_respMean'].shape[1]
        emgs = test_emg_indices if test_emg_indices is not None else range(n_emgs)
        for emg_idx in emgs:
            if emg_idx < n_emgs:
                experiments.append((subj_idx, emg_idx))

    # --- Always create per-run output directory so plots land in runs/<tag>/ ---
    run_dir = create_run_dir(_save_tag)
    write_run_config(run_dir, {
        'run_type': 'run_experiment',
        'experiment_tag': _save_tag,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'dataset_type': dataset_type,
        'split_type': split_type,
        'mode': mode,
        'device': device,
        'budget': budget,
        'n_reps': n_reps,
        'epochs': epochs,
        'lr': lr,
        'n_augmentations': n_augmentations,
        'held_out_emg_idx': held_out_emg_idx,
        'held_out_subj_idx': held_out_subj_idx,
        'budgets': budgets,
        'train_subjects': train_subject_indices,
        'test_subjects': test_subjects,
        'test_emg_indices': test_emg_indices,
        'n_experiments': len(experiments),
    })
    print(f"[INFO] Run directory: {run_dir}")

    # --- Fine-tune on the correct split (once, shared across all modes) ---
    print("=" * 60)
    print(f"Fine-tuning TabPFN  [{dataset_type} | {split_type} | modes={mode}]")
    print("=" * 60)

    ft_model_raw, ft_model = finetune_tabpfn(
        dataset_type,
        device=device,
        epochs=epochs,
        lr=lr,
        n_augmentations=n_augmentations,
        subject_indices=train_subject_indices,
        held_out_emg_idx=ft_held_out_emg,
        seed=42,
        silence_diagnostics=silence_diagnostics,
        output_dir=run_dir,
    )

    # --- Run each requested mode ---
    all_results = {}

    for m in mode:
        print(f"\n{'=' * 60}")
        print(f"Running mode: {m}")
        print('=' * 60)

        if m == 'fit':
            results_ft, results_gp = [], []
            for subj_idx, emg_idx in experiments:
                print(f"  Fit: subject={subj_idx}, emg={emg_idx}")
                res_ft = finetuned_fit(dataset_type, subj_idx, emg_idx, ft_model,
                                       device=device, budget=budget, n_reps=n_reps)
                res_gp = gp_baseline(dataset_type, subj_idx, emg_idx, mode='fit',
                                      device=device, budget=budget, n_reps=n_reps)
                results_ft.append(res_ft)
                results_gp.append(res_gp)
                print(f"    TabPFN R2={np.mean(res_ft['r2']):.3f}  |  GP R2={np.mean(res_gp['r2']):.3f}")

            results_dict = {'GP': results_gp, 'TabPFN': results_ft}
            tag = f'_{exp_tag}_finetuned_vs_gp'
            r2_per_muscle(results_dict, mode=tag, save=True, output_dir=run_dir)
            r2_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            n_maps = min(6, len(experiments))
            for idx in random.sample(range(len(experiments)), n_maps):
                show_emg_map(results_ft, idx, 'TabPFN', mode=f'_{exp_tag}_finetuned', save=True, output_dir=run_dir)
                show_emg_map(results_gp, idx, 'GP', mode=f'_{exp_tag}_baseline', save=True, output_dir=run_dir)

            all_r2 = [np.mean(r['r2']) for r in results_ft]
            print(f"\nDone. {len(results_ft)} experiments.")
            print(f"Finetuned TabPFN mean R²: {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f}")

            if save:
                save_results(results_dict, 'fit',
                             output_dir=os.path.join(run_dir, 'results'),
                             tag=_save_tag)
            all_results['fit'] = results_dict

        elif m == 'optimization':
            results_ft, results_gp = [], []
            for subj_idx, emg_idx in experiments:
                print(f"  Optimization: subject={subj_idx}, emg={emg_idx}")
                res_ft = finetuned_optimization(dataset_type, subj_idx, emg_idx, ft_model,
                                                 device=device, budget=budget, n_reps=n_reps)
                res_gp = gp_baseline(dataset_type, subj_idx, emg_idx, mode='optimization',
                                      device=device, budget=budget, n_reps=n_reps)
                results_ft.append(res_ft)
                results_gp.append(res_gp)
                print(f"    TabPFN R2={np.mean(res_ft['r2']):.3f}  |  GP R2={np.mean(res_gp['r2']):.3f}")

            results_dict = {'GP': results_gp, 'TabPFN': results_ft}
            tag = f'_{exp_tag}_opt_finetuned_vs_gp'
            r2_per_muscle(results_dict, mode=tag, save=True, output_dir=run_dir, eval_type='optimization')
            regret_with_timing(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            regret_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            regret_by_emg(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            n_maps = min(6, len(experiments))
            for idx in random.sample(range(len(experiments)), n_maps):
                show_emg_map(results_ft, idx, 'TabPFN', mode=f'_{exp_tag}_opt_finetuned',
                             save=True, output_dir=run_dir, eval_type='optimization')
                show_emg_map(results_gp, idx, 'GP', mode=f'_{exp_tag}_opt_baseline',
                             save=True, output_dir=run_dir, eval_type='optimization')
            visualize_representation(results_dict, mode=f'_{exp_tag}',
                                     save=True, output_dir=run_dir)

            all_r2 = [np.mean(r['r2']) for r in results_ft]
            print(f"\nDone. {len(results_ft)} experiments.")
            print(f"Finetuned TabPFN mean R²: {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f}")

            if save:
                save_results(results_dict, 'optimization',
                             output_dir=os.path.join(run_dir, 'results'),
                             tag=_save_tag)
            all_results['optimization'] = results_dict

        elif m == 'fit_budget':
            df = finetuned_fit_budget(
                dataset_type, ft_model,
                device=device,
                budgets=budgets,
                test_subjects=test_subjects,
                test_emg_indices=test_emg_indices,
                split_type=exp_tag,
                output_dir=run_dir,
            )
            if save:
                results_dir = os.path.join(run_dir, 'results')
                os.makedirs(results_dir, exist_ok=True)
                pkl_path = os.path.join(results_dir, f'{_save_tag}_fit_budget.pkl')
                df.to_pickle(pkl_path)
                print(f"Saved budget DataFrame -> {pkl_path}")
            all_results['fit_budget'] = df

        elif m == 'optimization_budget':
            df = finetuned_optimization_budget(
                dataset_type, ft_model,
                device=device,
                budgets=budgets,
                test_subjects=test_subjects,
                test_emg_indices=test_emg_indices,
                split_type=exp_tag,
                output_dir=run_dir,
            )
            if save:
                results_dir = os.path.join(run_dir, 'results')
                os.makedirs(results_dir, exist_ok=True)
                pkl_path = os.path.join(results_dir, f'{_save_tag}_optimization_budget.pkl')
                df.to_pickle(pkl_path)
                print(f"Saved budget DataFrame -> {pkl_path}")
            all_results['optimization_budget'] = df

    return all_results


# ============================================
#       CLI Entry Point
# ============================================

def run_finetuning():
    parser = argparse.ArgumentParser(
        description='Fine-tune TabPFN on neurostimulation data and run evaluation.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default='nhp', choices=['rat', 'nhp', 'spinal'],
                        help='Dataset type (default: nhp)')
    parser.add_argument('--split', type=str, default='inter_subject',
                        choices=['inter_subject', 'intra_emg'],
                        help='Train/test split strategy:\n'
                             '  inter_subject — train on TRAIN_SUBJECTS, test on HELD_OUT_SUBJECTS\n'
                             '  intra_emg     — train on ALL_SUBJECTS excl. held_out_emg, test on that EMG\n'
                             '(default: inter_subject)')
    parser.add_argument('--mode', type=lambda s: s.split(','), default=['fit'],
                        metavar='MODE[,MODE,...]',
                        help='Comma-separated evaluation modes. Valid values: '
                             'fit, optimization, fit_budget, optimization_budget, '
                             'aug_sweep_fit, aug_sweep_optimization. '
                             '(default: fit, example: --mode aug_sweep_optimization)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for training: cpu or cuda (default: cuda)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of fine-tuning epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-6)')
    parser.add_argument('--n_augmentations', type=int, default=25,
                        help='Augmentations per subject-EMG pair (default: 25)')
    parser.add_argument('--budget', type=int, default=100,
                        help='Training points (fit) or BO queries (optimization) (default: 100)')
    parser.add_argument('--n_reps', type=int, default=30,
                        help='Repetitions per experiment (default: 30)')
    parser.add_argument('--held_out_emg', type=int, default=None,
                        help='EMG index to hold out; required when --split intra_emg')
    parser.add_argument('--held_out_subj', type=int, default=None,
                        help='Subject index to hold out as the sole test subject; '
                             'overrides the default HELD_OUT_SUBJECTS split when set')
    parser.add_argument('--budgets', type=int, nargs='+', default=[10, 30, 50, 100],
                        help='Budget sweep values for *_budget modes (default: 10 30 50 100)')
    parser.add_argument('--aug_counts', type=float, nargs='+', default=None,
                        help='Augmentation counts to sweep for aug_sweep_* modes '
                             '(default: 1 2 5 7 10 25). Values in (0,1) are fractions '
                             'of the 1-aug reference dataset. Vanilla TabPFN (0 augs) is '
                             'always included as baseline.')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Persist results to output/results/ (pkl + CSV summary)')
    parser.add_argument('--diagnostics', action='store_true', default=False,
                        help='Enable gradient/CKA monitoring via GradientMonitoredRegressor '
                             '(slower finetuning, higher memory). Off by default.')

    args = parser.parse_args()

    _CLI_MODES = _VALID_MODES | {'aug_sweep_fit', 'aug_sweep_optimization'}
    invalid = set(args.mode) - _CLI_MODES
    if invalid:
        parser.error(f"Invalid mode(s): {', '.join(sorted(invalid))}. "
                     f"Valid: {', '.join(sorted(_CLI_MODES))}")

    silence_diagnostics = not args.diagnostics

    exp_modes = [m for m in args.mode if m in _VALID_MODES]

    if exp_modes:
        run_experiment(
            dataset_type=args.dataset,
            split_type=args.split,
            mode=exp_modes,
            device=args.device,
            budget=args.budget,
            n_reps=args.n_reps,
            epochs=args.epochs,
            lr=args.lr,
            n_augmentations=args.n_augmentations,
            held_out_emg_idx=args.held_out_emg,
            held_out_subj_idx=args.held_out_subj,
            budgets=args.budgets,
            save=args.save,
            silence_diagnostics=silence_diagnostics,
        )

    if 'aug_sweep_fit' in args.mode:
        finetuned_percentage(
            dataset_type=args.dataset,
            split_type=args.split,
            mode='fit',
            device=args.device,
            budget=args.budget,
            n_reps=args.n_reps,
            epochs=args.epochs,
            lr=args.lr,
            n_augmentations=args.aug_counts,
            held_out_emg_idx=args.held_out_emg,
            held_out_subj_idx=args.held_out_subj,
            save=args.save,
            silence_diagnostics=silence_diagnostics,
        )

    if 'aug_sweep_optimization' in args.mode:
        finetuned_percentage(
            dataset_type=args.dataset,
            split_type=args.split,
            mode='optimization',
            device=args.device,
            budget=args.budget,
            n_reps=args.n_reps,
            epochs=args.epochs,
            lr=args.lr,
            n_augmentations=args.aug_counts,
            held_out_emg_idx=args.held_out_emg,
            held_out_subj_idx=args.held_out_subj,
            save=args.save,
            silence_diagnostics=silence_diagnostics,
        )


if __name__ == '__main__':
    run_finetuning()
