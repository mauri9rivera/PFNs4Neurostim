"""
Evaluation functions for finetuned TabPFN and GP baselines.

- gp_baseline(): GP fit/optimization evaluation
- finetuned_fit(): evaluate fit quality with finetuned TabPFN
- finetuned_optimization(): evaluate optimization with finetuned TabPFN
- finetuned_fit_budget(): budget sweep for fit evaluation
- finetuned_optimization_budget(): budget sweep for optimization evaluation
- finetuned_percentage(): augmentation ablation study
- load_sweep_results(): load and merge sweep DataFrames from disk
"""
import copy
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor

from models.gaussians import ExactGP
from models.regressors import _make_finetuned_regressor, extract_inference_model
from utils.bo_loops import run_gpbo_loop, run_finetunedbo_loop, _snapshot_iters
from utils.data_utils import (
    build_finetuning_dataset, load_data, preprocess_neural_data,
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


def gp_baseline(dataset, subject_idx, emg_idx, mode='fit',
                device='cpu', budget=150, n_reps=30):
    """
    GP baseline that mirrors finetuned_fit / finetuned_optimization.

    Takes the same inputs as finetuned_fit and finetuned_optimization
    (minus finetuned_model) and builds a GP to perform the task instead.

    Args:
        dataset: 'rat' or 'nhp'
        subject_idx: subject index
        emg_idx: EMG channel index
        mode: 'fit' or 'optimization'
        device: 'cpu' or 'cuda'
        budget: number of training points (fit) or queries (optimization)
        n_reps: number of repetitions

    Returns:
        dict with results matching the format of finetuned_fit (mode='fit')
        or finetuned_optimization (mode='optimization').
    """
    data = load_data(dataset, subject_idx)

    X_train_full, y_train_full, X_test, y_test, scaler_y = preprocess_neural_data(
        data, emg_idx, 'gp'
    )

    if mode == 'fit':
        n_stims = y_train_full.shape[1]
        y_train_full_flat = y_train_full.flatten()

        r2_scores = []
        y_preds_all = []
        total_time = 0

        for i in range(n_reps):

            indices = np.random.choice(len(y_train_full_flat), budget, replace=False)
            X_train = np.repeat(X_train_full, n_stims, axis=0)[indices]
            y_train = y_train_full_flat[indices]

            # Convert to Tensors
            train_x = torch.tensor(X_train, dtype=torch.float32, device=device)
            train_y = torch.tensor(y_train, dtype=torch.float32, device=device)

            # Initialize Model & Likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = ExactGP(train_x, train_y, likelihood).to(device)

            start = time.time()

            # Training Loop (Optimize Hyperparameters)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for _ in range(50):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            # Prediction
            model.eval()
            likelihood.eval()
            test_x_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            with torch.no_grad():
                posterior = likelihood(model(test_x_tensor))
                y_pred = posterior.mean.cpu().numpy()

            total_time += (time.time() - start)

            og_shape = y_pred.shape
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(og_shape)
            r2 = r2_score(y_test, y_pred)
            r2_scores.append(np.clip(r2, 0.0, 1.0))
            y_preds_all.append(y_pred)

        y_pred_mean = np.mean(np.array(y_preds_all), axis=0)

        return {
            'model_type': 'gp',
            'r2': r2_scores,
            'times': total_time / n_reps,
            'y_test': y_test,
            'y_pred': y_pred_mean,
            'dataset': dataset,
            'subject': subject_idx,
            'emg': emg_idx
        }

    elif mode == 'optimization':
        mean_times = []
        values_all = []
        r2_scores = []
        y_preds_all = []

        n_init = max(3, int(0.05 * budget))
        snapshot_rep = np.random.randint(n_reps)
        snap_iters = _snapshot_iters(budget, n_init)
        collected_snapshots = None

        for i in range(n_reps):

            traj, observed_values, real_values, times, y_pred, snap = run_gpbo_loop(
                X_train_full, y_train_full, X_test, y_test,
                n_init=n_init, budget=budget, device=device,
                snapshot_iters=snap_iters if i == snapshot_rep else None,
            )

            if snap is not None:
                collected_snapshots = snap

            mean_times.append(times)
            values_all.append(real_values)

            # Compute R2 from the final model predictions
            og_shape = y_pred.shape
            y_pred_unscaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(og_shape)
            r2 = r2_score(y_test, y_pred_unscaled)
            r2_scores.append(np.clip(r2, 0.0, 1.0))
            y_preds_all.append(y_pred_unscaled)

        mean_times = np.mean(np.array(mean_times), axis=0)
        y_pred_mean = np.mean(np.array(y_preds_all), axis=0)

        # Inverse-transform snapshot predictions and compute R2
        snapshot_results = None
        if collected_snapshots is not None:
            snapshot_results = {}
            for it, s_pred in collected_snapshots.items():
                s_pred_unscaled = scaler_y.inverse_transform(s_pred.reshape(-1, 1)).ravel()
                s_r2 = float(np.clip(r2_score(y_test, s_pred_unscaled), 0.0, 1.0))
                snapshot_results[it] = {'y_pred': s_pred_unscaled, 'r2': s_r2}

        return {
            'model_type': 'gp',
            'times': mean_times,
            'values': values_all,
            'y_test': y_test,
            'r2': r2_scores,
            'y_pred': y_pred_mean,
            'dataset': dataset,
            'subject': subject_idx,
            'emg': emg_idx,
            'snapshots': snapshot_results,
        }


def finetuned_fit(dataset, subject_idx, emg_idx, model,
                  device='cpu', budget=150, n_reps=30):
    """
    Evaluate fit quality using a finetuned TabPFN model.

    The .fit() call stores in-context learning examples that the transformer
    uses to make predictions (no gradient updates).

    Args:
        model: A TabPFNRegressor from extract_inference_model()
    """
    data = load_data(dataset, subject_idx)

    # Use 'pfn' normalization (MinMax for X, Standard for y)
    X_train_full, y_train_full, X_test, y_test, scaler_y = preprocess_neural_data(
        data, emg_idx, 'pfn'
    )

    n_stims = y_train_full.shape[1]
    y_train_full = y_train_full.flatten()

    r2_scores = []
    y_preds_all = []
    total_time = 0

    for i in range(n_reps):

        indices = np.random.choice(len(y_train_full), budget, replace=False)
        X_train = np.repeat(X_train_full, n_stims, axis=0)[indices]
        y_train = y_train_full[indices]

        start = time.time()

        # .fit() provides in-context examples for prediction
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        total_time += (time.time() - start)

        y_pred = np.asarray(y_pred)
        og_shape = y_pred.shape
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(og_shape)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(np.clip(r2, 0.0, 1.0))
        y_preds_all.append(y_pred)

    y_pred_mean = np.mean(np.array(y_preds_all), axis=0)

    return {
        'model_type': 'finetuned_tabpfn',
        'r2': r2_scores,
        'times': total_time / n_reps,
        'y_test': y_test,
        'y_pred': y_pred_mean,
        'dataset': dataset,
        'subject': subject_idx,
        'emg': emg_idx
    }


def finetuned_optimization(dataset, subject_idx, emg_idx, model,
                            device='cpu', budget=100, n_reps=20):
    """
    Evaluate optimization performance using a finetuned TabPFN model.

    Args:
        model: A TabPFNRegressor from extract_inference_model()
    """
    data = load_data(dataset, subject_idx)

    # Use 'pfn' normalization
    X_train_full, y_train_full, X_test, y_test, scaler_y = preprocess_neural_data(
        data, emg_idx, 'pfn'
    )

    mean_times = []
    values_all = []
    r2_scores = []
    y_preds_all = []

    # Use a single-estimator copy for the BO loop: ensemble averaging
    # (n_estimators=8) means 8 forward passes per step, making each step
    # ~8× slower than necessary. One pass suffices for acquisition.
    bo_model = copy.deepcopy(model)
    bo_model.n_estimators = 1

    n_init = max(3, int(0.05 * budget))
    snapshot_rep = np.random.randint(n_reps)
    snap_iters = _snapshot_iters(budget, n_init)
    collected_snapshots = None

    for i in range(n_reps):

        traj, observed_values, real_values, times, y_pred, snap = run_finetunedbo_loop(
            X_train_full, y_train_full, X_test, y_test, bo_model,
            n_init=n_init, budget=budget, device=device,
            snapshot_iters=snap_iters if i == snapshot_rep else None,
        )

        if snap is not None:
            collected_snapshots = snap

        mean_times.append(times)
        values_all.append(real_values)

        # Compute R2 from the final model predictions
        og_shape = y_pred.shape
        y_pred_unscaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(og_shape)
        r2 = r2_score(y_test, y_pred_unscaled)
        r2_scores.append(np.clip(r2, 0.0, 1.0))
        y_preds_all.append(y_pred_unscaled)

    mean_times = np.mean(np.array(mean_times), axis=0)
    y_pred_mean = np.mean(np.array(y_preds_all), axis=0)

    # Inverse-transform snapshot predictions and compute R2
    snapshot_results = None
    if collected_snapshots is not None:
        snapshot_results = {}
        for it, s_pred in collected_snapshots.items():
            s_pred_unscaled = scaler_y.inverse_transform(s_pred.reshape(-1, 1)).ravel()
            s_r2 = float(np.clip(r2_score(y_test, s_pred_unscaled), 0.0, 1.0))
            snapshot_results[it] = {'y_pred': s_pred_unscaled, 'r2': s_r2}

    return {
        'model_type': 'finetuned_tabpfn',
        'times': mean_times,
        'values': values_all,
        'y_test': y_test,
        'r2': r2_scores,
        'y_pred': y_pred_mean,
        'dataset': dataset,
        'subject': subject_idx,
        'emg': emg_idx,
        'snapshots': snapshot_results,
    }


def finetuned_fit_budget(dataset_type, model, device='cpu',
                          budgets=[10, 50, 100, 150, 200],
                          test_subjects=None, test_emg_indices=None,
                          split_type='', visualize=True, output_dir=None):
    """
    Run fit evaluation for varying budget levels on held-out subjects.

    Args:
        dataset_type: 'rat' or 'nhp'
        model: A TabPFNRegressor from extract_inference_model()
        device: 'cpu' or 'cuda'
        budgets: List of training budgets to test
        test_subjects: list of subject indices to test on. If None, uses
            HELD_OUT_SUBJECTS[dataset_type] (inter-subject, existing behavior).
        test_emg_indices: list of EMG indices to test on per subject. If None,
            iterates over all EMGs for each subject (existing behavior).
        visualize: if True, produce and save the budget-sweep line plot.
    """
    if test_subjects is None:
        test_subjects = HELD_OUT_SUBJECTS[dataset_type]
    plot_data = []

    print(f"Starting Finetuned TabPFN Budget Sweep: {budgets}")

    for b in budgets:
        print(f"  > Running budget: {b}...")

        for subj_idx in test_subjects:
            data = load_data(dataset_type, subj_idx)
            n_emgs = data['sorted_respMean'].shape[1]

            emg_range = test_emg_indices if test_emg_indices is not None else range(n_emgs)
            for emg_idx in emg_range:
                if emg_idx >= n_emgs:
                    continue
                res_ft = finetuned_fit(
                    dataset_type, subj_idx, emg_idx,
                    model,
                    device=device,
                    budget=b,
                    n_reps=15
                )
                for score in res_ft['r2']:
                    plot_data.append({
                        'Budget': b,
                        'Model': 'TabPFN',
                        'R2': np.clip(score, 0.0, 1.0),
                        'ID': f"{res_ft['subject']}_{res_ft['emg']}"
                    })

                res_gp = gp_baseline(dataset_type, subj_idx, emg_idx, mode='fit',
                                     device=device, budget=b, n_reps=15)
                for score in res_gp['r2']:
                    plot_data.append({
                        'Budget': b,
                        'Model': 'GP',
                        'R2': np.clip(score, 0.0, 1.0),
                        'ID': f"{res_gp['subject']}_{res_gp['emg']}"
                    })

    df = pd.DataFrame(plot_data)
    if visualize:
        budget_sweep_plot(df, eval_type='fit', dataset=dataset_type,
                          split_type=split_type, save=True, output_dir=output_dir)

    return df


def finetuned_optimization_budget(dataset_type, model, regret_metric='abs',
                                   device='cpu', budgets=[10, 50, 100, 150, 200],
                                   test_subjects=None, test_emg_indices=None,
                                   split_type='', output_dir=None):
    """
    Run optimization evaluation for varying budgets on held-out subjects.

    Args:
        dataset_type: 'rat' or 'nhp'
        model: A TabPFNRegressor from extract_inference_model()
        regret_metric: 'abs' (Final Simple Regret) or 'cum' (Mean Simple Regret)
        device: 'cpu' or 'cuda'
        budgets: List of budgets to sweep
        test_subjects: list of subject indices to test on. If None, uses
            HELD_OUT_SUBJECTS[dataset_type] (inter-subject, existing behavior).
        test_emg_indices: list of EMG indices to test on per subject. If None,
            iterates over all EMGs for each subject (existing behavior).
    """
    if test_subjects is None:
        test_subjects = HELD_OUT_SUBJECTS[dataset_type]
    plot_data = []

    print(f"Starting Finetuned TabPFN Optimization Sweep ({regret_metric}): {budgets}")

    for b in budgets:
        print(f"  > Running budget: {b}...")
        budget_results_ft = []
        budget_results_gp = []

        for subj_idx in test_subjects:
            data = load_data(dataset_type, subj_idx)
            n_emgs = data['sorted_respMean'].shape[1]

            emg_range = test_emg_indices if test_emg_indices is not None else range(n_emgs)
            for emg_idx in emg_range:
                if emg_idx >= n_emgs:
                    continue

                res_ft = finetuned_optimization(
                    dataset_type, subj_idx, emg_idx,
                    model,
                    device=device,
                    budget=b,
                    n_reps=20
                )
                budget_results_ft.append(res_ft)
                optimal_ft = res_ft['y_test'].max()
                raw_ft = np.array(res_ft['values'])
                best_ft = np.maximum.accumulate(raw_ft, axis=1)
                regret_ft = optimal_ft - best_ft
                scores_ft = regret_ft[:, -1] if regret_metric == 'abs' \
                    else np.mean(regret_ft, axis=1)
                for score, r2 in zip(scores_ft, res_ft['r2']):
                    plot_data.append({
                        'Budget': b,
                        'Model': 'TabPFN',
                        'Regret': score,
                        'R2': float(np.clip(r2, 0.0, 1.0)),
                        'ID': f"{res_ft['subject']}_{res_ft['emg']}"
                    })

                res_gp = gp_baseline(dataset_type, subj_idx, emg_idx,
                                     mode='optimization',
                                     device=device, budget=b, n_reps=20)
                budget_results_gp.append(res_gp)
                optimal_gp = res_gp['y_test'].max()
                raw_gp = np.array(res_gp['values'])
                best_gp = np.maximum.accumulate(raw_gp, axis=1)
                regret_gp = optimal_gp - best_gp
                scores_gp = regret_gp[:, -1] if regret_metric == 'abs' \
                    else np.mean(regret_gp, axis=1)
                for score, r2 in zip(scores_gp, res_gp['r2']):
                    plot_data.append({
                        'Budget': b,
                        'Model': 'GP',
                        'Regret': score,
                        'R2': float(np.clip(r2, 0.0, 1.0)),
                        'ID': f"{res_gp['subject']}_{res_gp['emg']}"
                    })

        if output_dir and budget_results_ft:
            visualize_representation(
                {'GP': budget_results_gp, 'TabPFN': budget_results_ft},
                mode=f'_{split_type}_budget{b}', save=True, output_dir=output_dir)

    df = pd.DataFrame(plot_data)
    budget_sweep_plot(df, eval_type='optimization', dataset=dataset_type,
                      split_type=split_type, save=True, output_dir=output_dir)

    return df


def finetuned_percentage(
    dataset_type,
    split_type='inter_subject',
    mode='optimization',
    device='cpu',
    budget=100,
    n_reps=20,
    epochs=50,
    lr=1e-5,
    n_augmentations=None,
    held_out_emg_idx=None,
    held_out_subj_idx=None,
    save=False,
    print_diagnostics=False,
):
    """
    Ablation study: evaluate BO performance (R² + regret) across augmentation counts.

    Compares vanilla TabPFN (n_aug=0, no finetuning) against finetuned TabPFN
    for each value in n_augmentations, using the same train/test split logic as
    run_experiment().

    Args:
        dataset_type: 'rat' or 'nhp'
        split_type: 'inter_subject' or 'intra_emg'
        mode: 'fit' or 'optimization'
        device: 'cpu' or 'cuda'
        budget: training points (fit) or BO queries (optimization)
        n_reps: repetitions per experiment
        epochs: finetuning epochs
        lr: finetuning learning rate
        n_augmentations: list of int aug counts to sweep; default [1, 2, 5, 7, 10, 25]
        held_out_emg_idx: required when split_type='intra_emg'
        held_out_subj_idx: optional override for the test subject
        save: if True, save plot and DataFrame to disk
        print_diagnostics: if True, print gradient diagnostics to stdout

    Returns:
        DataFrame with columns: n_aug, R2, (Regret), ID
    """
    if n_augmentations is None:
        n_augmentations = [1, 2, 5, 7, 10, 25]

    for v in n_augmentations:
        if v < 0:
            raise ValueError(f"n_augmentations values must be >= 0, got {v}")
        if v >= 1 and abs(v - round(v)) > 1e-9:
            raise ValueError(
                f"Values >= 1 must be whole numbers (got {v}). "
                "Use a value in (0,1) for fractional dataset size."
            )

    # --- Resolve train / test sets (mirrors run_experiment logic) ---
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

    # Build experiment tag for filenames
    tag_parts = [split_type]
    if held_out_subj_idx is not None:
        tag_parts.append(f'subj{held_out_subj_idx}')
    if held_out_emg_idx is not None:
        tag_parts.append(f'emg{held_out_emg_idx}')
    exp_tag = '_'.join(tag_parts)

    aug_sweep_tag = (
        f'{dataset_type}_{split_type}_aug_sweep_ep{epochs}_lr{lr:.2e}'
        + (f'_subj{held_out_subj_idx}' if held_out_subj_idx is not None else '')
        + (f'_emg{held_out_emg_idx}' if held_out_emg_idx is not None else '')
    )

    # Build list of (subj_idx, emg_idx) experiment pairs
    experiments = []
    for subj_idx in test_subjects:
        data = load_data(dataset_type, subj_idx)
        n_emgs = data['sorted_respMean'].shape[1]
        emgs = test_emg_indices if test_emg_indices is not None else range(n_emgs)
        for emg_idx in emgs:
            if emg_idx < n_emgs:
                experiments.append((subj_idx, emg_idx))

    # --- Create per-run output directory and write config ---
    if save:
        run_dir = create_run_dir(aug_sweep_tag)
        write_run_config(run_dir, {
            'run_type': 'finetuned_percentage',
            'experiment_tag': aug_sweep_tag,
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
            'train_subjects': train_subject_indices,
            'test_subjects': test_subjects,
            'test_emg_indices': test_emg_indices,
            'n_experiments': len(experiments),
        })
    else:
        run_dir = None

    plot_data = []

    def _accumulate(res):
        """Append per-rep rows from a result dict into plot_data."""
        if mode == 'optimization':
            optimal = res['y_test'].max()
            raw = np.array(res['values'])
            best = np.maximum.accumulate(raw, axis=1)
            regrets = optimal - best[:, -1]
            for r2, reg in zip(res['r2'], regrets):
                plot_data.append({
                    'n_aug':  res['n_aug'],
                    'R2':     float(np.clip(r2, 0.0, 1.0)),
                    'Regret': float(reg),
                    'ID':     f"{res['subject']}_{res['emg']}",
                })
        else:
            for r2 in res['r2']:
                plot_data.append({
                    'n_aug': res['n_aug'],
                    'R2':    float(np.clip(r2, 0.0, 1.0)),
                    'ID':    f"{res['subject']}_{res['emg']}",
                })

    # --- Phase 1: Vanilla TabPFN baseline (n_aug = 0) ---
    print("=" * 60)
    print(f"[Aug Sweep] Vanilla TabPFN baseline (n_aug=0) | {dataset_type} | {split_type} | mode={mode}")
    print("=" * 60)
    vanilla_model = TabPFNRegressor(device=device)
    vanilla_results = []
    for subj_idx, emg_idx in experiments:
        print(f"  Vanilla: subject={subj_idx}, emg={emg_idx}")
        if mode == 'optimization':
            res = finetuned_optimization(dataset_type, subj_idx, emg_idx, vanilla_model,
                                         device=device, budget=budget, n_reps=n_reps)
        else:
            res = finetuned_fit(dataset_type, subj_idx, emg_idx, vanilla_model,
                                device=device, budget=budget, n_reps=n_reps)
        res['n_aug'] = 0
        _accumulate(res)
        if mode == 'optimization':
            vanilla_results.append(res)
    if vanilla_results and run_dir:
        visualize_representation({'TabPFN': vanilla_results},
                                 mode=f'_{exp_tag}_aug0', save=True, output_dir=run_dir)

    # --- Phase 2: Finetuned TabPFN sweep ---
    fraction_values = [v for v in n_augmentations if 0 < v < 1]
    if fraction_values:
        print("[Aug Sweep] Building 1-aug reference dataset for fraction subsampling...")
        X_ref, y_ref = build_finetuning_dataset(
            dataset_type,
            subject_indices=train_subject_indices,
            held_out_emg_idx=ft_held_out_emg,
            n_augmentations=1,
            seed=42,
        )
        n_ref = len(X_ref)
        print(f"  Reference size: {n_ref} rows")
    else:
        X_ref = y_ref = n_ref = None

    for n_aug in n_augmentations:
        print("=" * 60)
        print(f"[Aug Sweep] n_aug={n_aug} | {dataset_type} | {split_type}")
        print("=" * 60)

        if 0 < n_aug < 1:
            # Fractional mode: subsample from reference dataset
            n_sample = int(np.floor(n_aug * n_ref))
            if n_sample == 0:
                print(f"  WARNING: fraction {n_aug} × {n_ref} = 0 samples. Skipping.")
                continue
            rng = np.random.RandomState(42)
            idx = rng.choice(n_ref, size=n_sample, replace=False)
            X_ft, y_ft = X_ref[idx], y_ref[idx]
            print(f"  Subsample: {n_sample}/{n_ref} rows ({int(round(n_aug * 100))}%)")
        else:
            # Integer mode: build full augmented dataset
            n_aug_int = int(round(n_aug))
            X_ft, y_ft = build_finetuning_dataset(
                dataset_type,
                subject_indices=train_subject_indices,
                held_out_emg_idx=ft_held_out_emg,
                n_augmentations=n_aug_int,
                seed=42,
            )
            print(f"  Dataset size: {X_ft.shape[0]} rows")

        ft_model_raw = _make_finetuned_regressor(
            print_diagnostics=print_diagnostics,
            device=device, epochs=epochs, learning_rate=lr,
            n_estimators_finetune=8, n_estimators_validation=8,
            n_estimators_final_inference=8,
        )
        ft_model_raw.fit(X_ft, y_ft)
        # Always save diagnostics plots when available
        if hasattr(ft_model_raw, '_diagnostics_') and ft_model_raw._diagnostics_:
            diag_dir = os.path.join(run_dir, 'diagnostics') if run_dir else None
            plot_gradient_metrics(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
            plot_weight_metrics(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
            plot_cka_similarity(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
        ft_model = extract_inference_model(ft_model_raw)

        aug_results = []
        for subj_idx, emg_idx in experiments:
            print(f"  n_aug={n_aug}: subject={subj_idx}, emg={emg_idx}")
            if mode == 'optimization':
                res = finetuned_optimization(dataset_type, subj_idx, emg_idx, ft_model,
                                             device=device, budget=budget, n_reps=n_reps)
            else:
                res = finetuned_fit(dataset_type, subj_idx, emg_idx, ft_model,
                                    device=device, budget=budget, n_reps=n_reps)
            res['n_aug'] = n_aug
            _accumulate(res)
            if mode == 'optimization':
                aug_results.append(res)
        if aug_results and run_dir:
            aug_label = int(round(n_aug)) if n_aug >= 1 else f'{int(round(n_aug * 100))}pct'
            visualize_representation({'TabPFN': aug_results},
                                     mode=f'_{exp_tag}_aug{aug_label}',
                                     save=True, output_dir=run_dir)

    # --- Phase 3: Visualize & save ---
    df = pd.DataFrame(plot_data)
    augmentation_sweep_plot(df, eval_type=mode, dataset=dataset_type,
                            split_type=exp_tag, save=save, output_dir=run_dir)

    if save:
        results_dir = os.path.join(run_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        pkl_path = os.path.join(results_dir, f'{aug_sweep_tag}.pkl')
        df.to_pickle(pkl_path)
        print(f"Saved aug sweep DataFrame -> {pkl_path}")

    return df


def load_sweep_results(tags, result_type, runs_dir='output/runs'):
    """
    Load and merge pickle DataFrames from multiple experiment tags.

    For each tag, searches for a matching run directory under runs_dir and
    loads the corresponding pkl file. Useful for aggregating family-1/2/3
    results across subjects before producing cross-subject plots.

    Args:
        tags: list of experiment tags
        result_type: 'optimization_budget' → loads {tag}_optimization_budget.pkl
                     'fit_budget'          → loads {tag}_fit_budget.pkl
                     'aug_sweep'           → loads {tag}.pkl
        runs_dir: root directory containing run subdirectories (default: 'output/runs')

    Returns:
        Merged pd.DataFrame from all matched pickle files.

    Raises:
        FileNotFoundError: if no pkl is found for a given tag.
    """
    import glob as _glob

    frames = []
    for tag in tags:
        # Each run dir is named {tag}_{timestamp}; find any matching dir
        pattern = os.path.join(runs_dir, f'{tag}_*', 'results')
        candidates = _glob.glob(pattern)
        if not candidates:
            # Also try without timestamp suffix (manual saves)
            candidates = _glob.glob(os.path.join(runs_dir, tag, 'results'))
        if not candidates:
            raise FileNotFoundError(
                f"No run directory found for tag '{tag}' under '{runs_dir}'. "
                f"Searched: {os.path.join(runs_dir, tag + '_*', 'results')}"
            )
        # Use the most recently modified results dir if multiple matches
        results_dir = sorted(candidates, key=os.path.getmtime)[-1]

        if result_type == 'aug_sweep':
            pkl_name = f'{tag}.pkl'
        else:
            pkl_name = f'{tag}_{result_type}.pkl'

        pkl_path = os.path.join(results_dir, pkl_name)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Expected pkl not found: {pkl_path}")

        df = pd.read_pickle(pkl_path)
        frames.append(df)
        print(f"Loaded: {pkl_path}  ({len(df)} rows)")

    merged = pd.concat(frames, ignore_index=True)
    print(f"Merged {len(tags)} files → {len(merged)} total rows")
    return merged
