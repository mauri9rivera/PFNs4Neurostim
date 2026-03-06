"""
Fine-tuning orchestration for TabPFN on neurostimulation data.

Two-phase workflow:
  1. Backprop finetuning: finetune_tabpfn() trains a FinetunedTabPFNRegressor,
     adapting pretrained weights to neurostimulation data via gradient updates.
  2. Extraction for evaluation: extract_inference_model() deep-copies the
     internal TabPFNRegressor with finetuned weights. This standalone model
     uses in-context learning (.fit() stores context, no gradients) and
     supports the full predict API including output_type="quantiles".

Usage:
    python finetuning.py --dataset rat --device cuda --epochs 30
    python finetuning.py --dataset nhp --device cuda --epochs 30
    python finetuning.py --evaluate --dataset rat --device cpu
"""
import argparse
import copy
import os
import time
from datetime import datetime
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

import gpytorch

from tabpfn import TabPFNRegressor
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
from pathlib import Path

from models.gaussians import ExactGP
from utils.data_utils import (
    build_finetuning_dataset, load_data, preprocess_neural_data,
    HELD_OUT_SUBJECTS, TRAIN_SUBJECTS, ALL_SUBJECTS,
    generate_experiment_tag, save_results,
    create_run_dir, write_run_config,
)
from utils.gpbo_utils import expected_improvement
from utils.visualization import (
    show_emg_map, r2_comparison, regret_curve, plot_runtime_trajectory,
    r2_by_subject, r2_by_emg,
    regret_by_subject, regret_by_emg,
    budget_sweep_plot, regret_with_timing,
    augmentation_sweep_plot,
)
import random


def finetune_tabpfn(dataset_type, device='cuda', epochs=1, lr=1e-5,
                    n_augmentations=25):
    """
    Fine-tune a TabPFNRegressor on augmented neurostimulation data.

    Args:
        dataset_type: 'rat' or 'nhp'
        device: 'cpu' or 'cuda'
        epochs: number of fine-tuning epochs
        lr: learning rate
        n_augmentations: augmentations per subject-EMG pair
        output_dir: directory to save the fine-tuned model

    Returns:
        model: The finetuned FinetunedTabPFNRegressor instance (in-memory)
    """
    print(f"Building augmented dataset for '{dataset_type}' ...")
    X_train, y_train = build_finetuning_dataset(
        dataset_type,
        n_augmentations=n_augmentations,
        seed=42
    )
    print(f"  Dataset size: {X_train.shape[0]} rows, {X_train.shape[1]} features")

    print(f"Initializing FinetunedTabPFNRegressor for fine-tuning (epochs={epochs}, lr={lr}) ...")

    model = FinetunedTabPFNRegressor(
        device=device,
        epochs=epochs,
        learning_rate=lr,
        n_estimators_finetune=8,
        n_estimators_validation=8,
        n_estimators_final_inference=8,
    )

    print("Fine-tuning ...")
    model.fit(X_train, y_train)
    print("Fine-tuning complete.")

    return model


def extract_inference_model(finetuned_regressor):
    """Extract a TabPFNRegressor with finetuned weights for in-context learning.

    After finetuning completes, the FinetunedTabPFNRegressor stores an internal
    TabPFNRegressor with finetuned weights at `finetuned_inference_regressor_`.
    This function deep-copies it to produce a standalone regressor where:
      - .fit(X, y) stores context (no gradient updates)
      - .predict() supports output_type="quantiles" and all other options
    """
    if not hasattr(finetuned_regressor, 'finetuned_inference_regressor_'):
        raise AttributeError(
            "FinetunedTabPFNRegressor has not been fit yet. "
            "Call .fit() before extracting the inference model."
        )
    inference_model = copy.deepcopy(finetuned_regressor.finetuned_inference_regressor_)
    print(f"  Extracted TabPFNRegressor with finetuned weights "
          f"(fit_mode={inference_model.fit_mode!r})")
    return inference_model


# ============================================
#       GP BO Loop (mirrors run_bo_loop, GP only)
# ============================================

def run_gpbo_loop(X_pool, y_pool, x_test, y_test,
                  n_init=5, budget=100, device='cpu'):
    """
    Performs the Active Learning loop using a GP model.
    (This mirrors run_bo_loop from main.py but only for the GP case)

    Args:
        X_pool: Feature matrix for the candidate pool (n_locs, n_features).
        y_pool: Response matrix (n_locs, n_reps) with noisy observations.
        x_test: Test feature matrix for final prediction.
        y_test: Test response vector (mean across repetitions).
        n_init: Number of random initial observations.
        budget: Total number of observations (including initial).
        device: 'cpu' or 'cuda'.

    Returns:
        - observed_indices: Indices of points chosen
        - observed_values: Observed y values (with noise)
        - real_values: True y values at observed indices
        - times: Time taken at each step
        - y_pred: Final predictions on x_test
    """
    n_locs, n_reps = y_pool.shape

    def sample_from_pool(idx):
        col_idx = np.random.randint(0, n_reps)
        return y_pool[idx, col_idx]

    # 1. Initialization (Random)
    pool_indices = np.arange(n_locs)
    observed_indices = np.random.choice(pool_indices, size=n_init, replace=False).tolist()
    observed_values = [sample_from_pool(idx) for idx in observed_indices]
    real_values = y_test[observed_indices].tolist()

    times = []

    # --- LOOP ---
    for t in range(budget - n_init):
        step_start = time.time()

        X_train = torch.tensor(X_pool[observed_indices], dtype=torch.float32, device=device)
        y_train = torch.tensor(observed_values, dtype=torch.float32, device=device)
        X_cand = torch.tensor(X_pool, dtype=torch.float32, device=device)

        # Initialize Model & Likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGP(X_train, y_train, likelihood).to(device)

        # Training Loop (Optimize Hyperparameters)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(50):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        # Select Next Point (EI)
        best_f = y_train.max()
        acq_vals = expected_improvement(model, likelihood, X_cand, best_f, device)
        next_idx = acq_vals.argmax().item()

        observed_indices.append(next_idx)
        new_val = sample_from_pool(next_idx)
        observed_values.append(new_val)
        real_values.append(y_test[next_idx])

        step_time = time.time() - step_start
        times.append(step_time)

    # Final model fit on all observed data to predict on x_test
    X_train_final = torch.tensor(X_pool[observed_indices], dtype=torch.float32, device=device)
    y_train_final = torch.tensor(observed_values, dtype=torch.float32, device=device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGP(X_train_final, y_train_final, likelihood).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(50):
        optimizer.zero_grad()
        output = model(X_train_final)
        loss = -mll(output, y_train_final)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        posterior = likelihood(model(torch.tensor(x_test, dtype=torch.float32, device=device)))
        y_pred = posterior.mean.cpu().numpy()

    return observed_indices, observed_values, real_values, times, y_pred


# ============================================
#       GP Baseline (mirrors finetuned_fit / finetuned_optimization)
# ============================================

def gp_baseline(dataset, subject_idx, emg_idx, mode='fit',
                device='cpu', budget=150, n_reps=30):
    """
    GP baseline that mirrors finetuned_fit / finetuned_optimization.

    Takes the same inputs as finetuned_fit and finetuned_optimization
    (minus finetuned_model) and builds a GP to perform the task instead.

    Identical to the 'gp' options in evaluate_fit and evaluate_optimization
    in main.py.

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

        for i in range(n_reps):

            traj, observed_values, real_values, times, y_pred = run_gpbo_loop(
                X_train_full, y_train_full, X_test, y_test,
                n_init=8, budget=budget, device=device
            )

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

        return {
            'model_type': 'gp',
            'times': mean_times,
            'values': values_all,
            'y_test': y_test,
            'r2': r2_scores,
            'y_pred': y_pred_mean,
            'dataset': dataset,
            'subject': subject_idx,
            'emg': emg_idx
        }


# ============================================
#       Finetuned BO Loop (mirrors run_bo_loop)
# ============================================

def run_finetunedbo_loop(X_pool, y_pool, x_test, y_test, model,
                          n_init=5, budget=100, device='cpu'):
    """
    Performs the Active Learning loop using a finetuned TabPFN model.

    This mirrors run_bo_loop from main.py but takes a TabPFNRegressor
    extracted via extract_inference_model(). The .fit() call stores
    in-context learning examples (no gradient updates).

    Args:
        model: A TabPFNRegressor from extract_inference_model()

    Returns:
        - observed_indices: Indices of points chosen
        - observed_values: Observed y values (with noise)
        - real_values: True y values at observed indices
        - times: Time taken at each step
        - y_pred: Final predictions on x_test
    """
    n_locs, n_reps = y_pool.shape

    def sample_from_pool(idx):
        col_idx = np.random.randint(0, n_reps)
        return y_pool[idx, col_idx]

    # 1. Initialization (Random)
    pool_indices = np.arange(n_locs)
    observed_indices = np.random.choice(pool_indices, size=n_init, replace=False).tolist()
    observed_values = [sample_from_pool(idx) for idx in observed_indices]
    real_values = y_test[observed_indices].tolist()

    times = []

    # --- LOOP ---
    for t in range(budget - n_init):
        step_start = time.time()

        X_obs_np = X_pool[observed_indices]
        y_obs_np = np.array(observed_values)

        # Fit provides in-context examples for the transformer
        model.fit(X_obs_np, y_obs_np)

        # Compute EI directly from the bar distribution (no Gaussian assumption)
        full_output = model.predict(X_pool, output_type="full")
        logits = full_output['logits']
        criterion = full_output['criterion']

        best_f = float(np.max(y_obs_np))
        ei_vals = criterion.ei(logits, best_f, maximize=True)
        next_idx = int(ei_vals.argmax().item())

        observed_indices.append(next_idx)
        new_val = sample_from_pool(next_idx)
        observed_values.append(new_val)
        real_values.append(y_test[next_idx])

        step_time = time.time() - step_start
        times.append(step_time)

    # Final prediction with all observed data as context
    X_obs_final = X_pool[observed_indices]
    y_obs_final = np.array(observed_values)
    model.fit(X_obs_final, y_obs_final)
    y_pred = model.predict(x_test)

    return observed_indices, observed_values, real_values, times, np.asarray(y_pred)


# ============================================
#       Finetuned Fit (mirrors evaluate_fit)
# ============================================

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

# ============================================
#   Finetuned Optimization (mirrors evaluate_optimization)
# ============================================

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

    for i in range(n_reps):

        traj, observed_values, real_values, times, y_pred = run_finetunedbo_loop(
            X_train_full, y_train_full, X_test, y_test, model,
            n_init=8, budget=budget, device=device
        )

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

    return {
        'model_type': 'finetuned_tabpfn',
        'times': mean_times,
        'values': values_all,
        'y_test': y_test,
        'r2': r2_scores,
        'y_pred': y_pred_mean,
        'dataset': dataset,
        'subject': subject_idx,
        'emg': emg_idx
    }


# ============================================
#       Budget Sweep Functions
# ============================================

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
                optimal_ft = res_ft['y_test'].max()
                raw_ft = np.array(res_ft['values'])
                best_ft = np.maximum.accumulate(raw_ft, axis=1)
                regret_ft = optimal_ft - best_ft
                scores_ft = regret_ft[:, -1] if regret_metric == 'abs' \
                    else np.mean(regret_ft, axis=1)
                for score in scores_ft:
                    plot_data.append({
                        'Budget': b,
                        'Model': 'TabPFN',
                        'Regret': score,
                        'ID': f"{res_ft['subject']}_{res_ft['emg']}"
                    })

                res_gp = gp_baseline(dataset_type, subj_idx, emg_idx,
                                     mode='optimization',
                                     device=device, budget=b, n_reps=20)
                optimal_gp = res_gp['y_test'].max()
                raw_gp = np.array(res_gp['values'])
                best_gp = np.maximum.accumulate(raw_gp, axis=1)
                regret_gp = optimal_gp - best_gp
                scores_gp = regret_gp[:, -1] if regret_metric == 'abs' \
                    else np.mean(regret_gp, axis=1)
                for score in scores_gp:
                    plot_data.append({
                        'Budget': b,
                        'Model': 'GP',
                        'Regret': score,
                        'ID': f"{res_gp['subject']}_{res_gp['emg']}"
                    })

    df = pd.DataFrame(plot_data)
    budget_sweep_plot(df, eval_type='optimization', dataset=dataset_type,
                      split_type=split_type, save=True, output_dir=output_dir)

    return df


# ============================================
#       Augmentation Ablation Study
# ============================================

def finetuned_percentage(
    dataset_type,
    split_type='inter_subject',
    mode='optimization',
    device='cpu',
    budget=100,
    n_reps=20,
    epochs=20,
    lr=1e-6,
    n_augmentations=None,
    held_out_emg_idx=None,
    held_out_subj_idx=None,
    save=False,
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

    Returns:
        DataFrame with columns: n_aug, R2, (Regret), ID
    """
    if n_augmentations is None:
        n_augmentations = [1, 2, 5, 7, 10, 25]

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

    # --- Phase 2: Finetuned TabPFN sweep ---
    for n_aug in n_augmentations:
        print("=" * 60)
        print(f"[Aug Sweep] Finetuning with n_aug={n_aug} | {dataset_type} | {split_type}")
        print("=" * 60)
        X_ft, y_ft = build_finetuning_dataset(
            dataset_type,
            subject_indices=train_subject_indices,
            held_out_emg_idx=ft_held_out_emg,
            n_augmentations=n_aug,
            seed=42,
        )
        print(f"  Dataset size: {X_ft.shape[0]} rows")
        ft_model_raw = FinetunedTabPFNRegressor(
            device=device, epochs=epochs, learning_rate=lr,
            n_estimators_finetune=8, n_estimators_validation=8,
            n_estimators_final_inference=8,
        )
        ft_model_raw.fit(X_ft, y_ft)
        ft_model = extract_inference_model(ft_model_raw)

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
    epochs=20,
    lr=1e-6,
    n_augmentations=25,
    held_out_emg_idx=None,
    held_out_subj_idx=None,
    budgets=None,
    save=False,
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
            Works for both split_type values.
        budgets: list of budgets for 'fit_budget' / 'optimization_budget' modes.
            Defaults to [10, 50, 100, 150, 200].
        save: if True, persist results to output/results/ (pkl + CSV summary).

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

    # --- Build test experiment list (needed for config and mode evaluation) ---
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
    else:
        run_dir = None

    # --- Fine-tune on the correct split (once, shared across all modes) ---
    print("=" * 60)
    print(f"Fine-tuning TabPFN  [{dataset_type} | {split_type} | modes={mode}]")
    print("=" * 60)
    print(f"Building augmented dataset ({split_type}) ...")
    X_ft, y_ft = build_finetuning_dataset(
        dataset_type,
        subject_indices=train_subject_indices,
        held_out_emg_idx=ft_held_out_emg,
        n_augmentations=n_augmentations,
        seed=42,
    )
    print(f"  Dataset size: {X_ft.shape[0]} rows")
    ft_model_raw = FinetunedTabPFNRegressor(
        device=device,
        epochs=epochs,
        learning_rate=lr,
        n_estimators_finetune=8,
        n_estimators_validation=8,
        n_estimators_final_inference=8,
        #TODO: try eval_metric='r2' or 'rmse'
    )
    print(f"Fine-tuning (epochs={epochs}, lr={lr}) ...")
    ft_model_raw.fit(X_ft, y_ft)
    ft_model = extract_inference_model(ft_model_raw)

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
            r2_comparison(results_dict, mode=tag, save=True, output_dir=run_dir)
            r2_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            r2_by_emg(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
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
            r2_comparison(results_dict, mode=tag, save=True, output_dir=run_dir)
            regret_curve(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            plot_runtime_trajectory(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            regret_with_timing(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            regret_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            regret_by_emg(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)

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
    parser.add_argument('--dataset', type=str, default='nhp', choices=['rat', 'nhp'],
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
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of fine-tuning epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-6,
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
    parser.add_argument('--budgets', type=int, nargs='+', default=[10, 50, 100, 150, 200],
                        help='Budget sweep values for *_budget modes (default: 10 50 100 150 200)')
    parser.add_argument('--aug_counts', type=int, nargs='+', default=None,
                        help='Augmentation counts to sweep for aug_sweep_* modes '
                             '(default: 1 2 5 7 10 25). Vanilla TabPFN (0 augs) is '
                             'always included as baseline.')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Persist results to output/results/ (pkl + CSV summary)')

    args = parser.parse_args()

    _CLI_MODES = _VALID_MODES | {'aug_sweep_fit', 'aug_sweep_optimization'}
    invalid = set(args.mode) - _CLI_MODES
    if invalid:
        parser.error(f"Invalid mode(s): {', '.join(sorted(invalid))}. "
                     f"Valid: {', '.join(sorted(_CLI_MODES))}")

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
        )


if __name__ == '__main__':
    run_finetuning()

