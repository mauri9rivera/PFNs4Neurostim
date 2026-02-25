import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import gzip
import time
import scipy.io
import seaborn as sns
import pickle
import joblib
import pandas as pd
import gpytorch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
import os

from models.gaussians import *
from utils.data_utils import *
from utils.gpbo_utils import *
from utils.visualization import *


# ============================================
#           Bayesian Optimization Loop
# ============================================

def run_bo_loop(X_pool, y_pool, x_test, y_test, model_type, pfn_model=None,
                n_init=1, budget=100, device='cpu'):
    """
    Performs the Active Learning loop for GP and PFN models.

    Args:
        pfn_model: PFN torch weights (model_type='pfn'). Ignored for 'gp'.

    Returns:
        - trajectory: Indices of points chosen
        - times: Time taken at each step
        - final_model_state: Data needed for final prediction
    """

    n_locs, n_reps = y_pool.shape

    # Helper to sample a random repetition for a specific location index
    def sample_from_pool(idx):
        col_idx = np.random.randint(0, n_reps)
        return y_pool[idx, col_idx]

    # 1. Initialization (Random)
    pool_indices = np.arange(n_locs)
    observed_indices = np.random.choice(pool_indices, size=n_init, replace=False).tolist()
    observed_values = [sample_from_pool(idx) for idx in observed_indices]
    real_values = y_test[observed_indices].tolist()

    times = []

    if model_type == 'pfn':
        pfn_wrapper = TransformerBOMethod(pfn_model, device=device, acq_function='ei')

    n_samples = np.zeros(n_locs, dtype=int)
    for idx in observed_indices:
        n_samples[idx] += 1


    # --- LOOP ---
    for t in range(budget - n_init):

        # Prepare Tensors
        X_train = torch.tensor(X_pool[observed_indices], dtype=torch.float32, device=device)
        y_train = torch.tensor(observed_values, dtype=torch.float32, device=device)

        X_cand = torch.tensor(X_pool, dtype=torch.float32, device=device)

        # --- GP Logic ---
        if model_type == 'gp':
            step_start = time.time()

            # Initialize Model & Likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = ExactGP(X_train, y_train, likelihood).to(device)

            # Training Loop (Optimize Hyperparameters)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # 50 iterations is usually sufficient for simple regression
            for _ in range(50):
                optimizer.zero_grad()
                output = model(X_train)
                loss = -mll(output, y_train)
                loss.backward()
                optimizer.step()

            # Select Next Point (EI)
            best_f = y_train.max()
            acq_vals = expected_improvement(model, likelihood, X_cand, best_f, device)

            # Get index relative to X_pool (0 to 95)
            next_idx = acq_vals.argmax().item()

        # --- PFN Logic ---
        elif model_type == 'pfn':

            step_start = time.time()
            idx_rel, acq_vals = pfn_wrapper.observe_and_suggest(
                X_obs=X_train,
                y_obs=y_train,
                X_pen=X_cand,
                return_actual_ei=True
            )

            if isinstance(idx_rel, torch.Tensor): idx_rel = idx_rel.item()

            next_idx = acq_vals.argmax().item()

        observed_indices.append(next_idx)
        new_val = sample_from_pool(next_idx)
        observed_values.append(new_val)
        real_values.append(y_test[next_idx])

        step_time = time.time() - step_start
        times.append(step_time)

    # Final model fit on all observed data to predict on full pool
    X_train_final = torch.tensor(X_pool[observed_indices], dtype=torch.float32, device=device)
    y_train_final = torch.tensor(observed_values, dtype=torch.float32, device=device)

    if model_type == 'gp':
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

    elif model_type == 'pfn':
        pfn_pred = TransformerBOMethod(pfn_model, device=device, acq_function='mean')
        _, y_pred = pfn_pred.observe_and_suggest(
            X_obs=X_train_final,
            y_obs=y_train_final,
            X_pen=torch.tensor(x_test, dtype=torch.float32, device=device),
            return_actual_ei=True
        )
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()

    return observed_indices, observed_values, real_values, times, y_pred

def evaluate_fit(dataset, subject_idx, emg_idx, model_type,
                   pfn_model_weights=None, device='cpu', budget=150, n_reps=30):
    """
    Evaluate fit quality for GP and PFN models.

    Args:
        pfn_model_weights: PFN torch weights (model_type='pfn'). Ignored for 'gp'.
    """
    data = load_data(dataset, subject_idx)

    X_train_full, y_train_full, X_test, y_test, scaler_y = preprocess_neural_data(data, emg_idx, model_type)

    n_stims = y_train_full.shape[1]
    y_train_full = y_train_full.flatten()

    r2_scores = []
    y_preds_all = []
    total_time = 0

    for i in range(n_reps):

        indices = np.random.choice(len(y_train_full), budget, replace=False)
        X_train = np.repeat(X_train_full, n_stims, axis=0)[indices]
        y_train = y_train_full[indices]


        if model_type == 'gp':

            # 1. Convert to Tensors
            train_x = torch.tensor(X_train, dtype=torch.float32, device=device)
            train_y = torch.tensor(y_train, dtype=torch.float32, device=device)

            # 2. Initialize Model & Likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = ExactGP(train_x, train_y, likelihood).to(device)

            start = time.time()

            # 3. Training Loop (Optimize Hyperparameters)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # 50 iterations is usually sufficient for simple regression
            for _ in range(50):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            # 4. Prediction
            model.eval()
            likelihood.eval()
            test_x_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
            with torch.no_grad():
                # Get the mean of the posterior
                posterior = likelihood(model(test_x_tensor))
                y_pred = posterior.mean.cpu().numpy()

        elif model_type == 'pfn':

            pfn_bo = TransformerBOMethod(
            pfn_model_weights,
            device=device,
            acq_function='mean' # This tells the PFN to output posterior mean
            )

            start = time.time()

            _, y_pred = pfn_bo.observe_and_suggest(
                X_obs=X_train,
                y_obs=y_train,
                X_pen=X_test,
                return_actual_ei=True
            )

            if isinstance(y_pred, torch.Tensor): y_pred = y_pred.numpy()

        # repetition results
        total_time += (time.time() - start)
        og_shape = y_pred.shape
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(og_shape)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(np.clip(r2, 0.0, 1.0))
        y_preds_all.append(y_pred)

    y_pred_mean = np.mean(np.array(y_preds_all), axis=0)


    return {
        'model_type': model_type,
        'r2': r2_scores,
        'times': total_time / n_reps,
        'y_test': y_test,
        'y_pred': y_pred_mean,
        'dataset': dataset,
        'subject': subject_idx,
        'emg': emg_idx
    }

def evaluate_optimization(dataset, subject_idx, emg_idx, model_type,
                   pfn_model_weights=None, device='cpu', budget=100, n_reps=20):
    """
    Evaluate optimization performance for GP and PFN models.

    Args:
        pfn_model_weights: PFN torch weights (model_type='pfn'). Ignored for 'gp'.
    """
    data = load_data(dataset, subject_idx)

    X_train_full, y_train_full, X_test, y_test, scaler_y = preprocess_neural_data(data, emg_idx, model_type)

    mean_times = []
    values_all = []
    r2_scores = []
    y_preds_all = []

    for i in range(n_reps):

        traj, observed_values, real_values, times, y_pred = run_bo_loop(
            X_train_full, y_train_full, X_test, y_test, model_type, pfn_model_weights,
            n_init=8, budget=budget, device=device)

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
        'model_type': model_type,
        'times': mean_times,
        'values': values_all, # Needed for regret curve
        'y_test': y_test,
        'r2': r2_scores,
        'y_pred': y_pred_mean,
        'dataset': dataset,
        'subject': subject_idx,
        'emg': emg_idx
    }

# ============================================
#           Other plotters
# ============================================

def fit_budget(datasets, device='cpu', budgets=[10, 50, 100, 150, 200]):
    """
    Runs the fit evaluation for varying budget levels and plots the R2 curve.

    Args:
        datasets: List of experiment tuples (dataset, subject, emg).
        device: 'cpu' or 'cuda'.
        budgets: List of integers representing the budgets to test.
    """
    plot_data = []

    print(f"Starting Budget Sweep: {budgets}")

    for b in budgets:
        print(f"  > Running budget: {b}...")

        results_gp, results_pfn = main(
            datasets,
            evaluation_type='fit',
            device=device,
            budget=b,
            n_reps=15
        )

        def _collect(results_list, model_name):
            for res in results_list:
                for score in res['r2']:
                    plot_data.append({
                        'Budget': b,
                        'Model': model_name,
                        'R2': np.clip(score, 0.0, 1.0),
                        'ID': f"{res['subject']}_{res['emg']}"
                    })

        _collect(results_gp, 'GP')
        _collect(results_pfn, 'PFN')

    # --- Visualization ---
    df = pd.DataFrame(plot_data)

    fig = plt.figure(figsize=(10, 6))

    custom_palette = {
        'GP': 'sandybrown',
        'PFN': 'royalblue'
    }

    sns.lineplot(
        data=df,
        x='Budget',
        y='R2',
        hue='Model',
        palette=custom_palette,
        marker='o',
        errorbar=('ci', 95),
        err_kws={'alpha': 0.2},
        linewidth=2
    )

    plt.title("Model Fit Quality vs. Training Budget")
    plt.ylabel("R² Score (Test Set)")
    plt.xlabel("Budget (Number of Training Points)")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Model Type')
    output_dir = os.path.join('output', 'fitness')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f'fit_budget.svg'
    )
    plt.savefig(plot_path, format="svg")
    plt.show()
    print(f"Saved plot to {plot_path}")

def optimization_budget(datasets, regret_metric='abs', device='cpu',
                        budgets=[10, 50, 100, 150, 200]):
    """
    Runs the optimization evaluation for varying budgets and plots the Regret.

    Args:
        datasets: List of experiment tuples.
        regret_metric: 'abs' (Final Simple Regret) or 'cum' (Mean Simple Regret).
        device: 'cpu' or 'cuda'.
        budgets: List of budgets to sweep.
    """
    plot_data = []

    if regret_metric == 'abs':
        y_label = "Final Simple Regret"
        title = "Optimization Performance: Final Regret vs Budget"
    elif regret_metric == 'cum':
        y_label = "Mean Simple Regret"
        title = "Optimization Cost: Mean Regret vs Budget"

    print(f"Starting Optimization Sweep ({regret_metric}): {budgets}")

    for b in budgets:
        print(f"  > Running budget: {b}...")

        results_gp, results_pfn = main(
            datasets,
            evaluation_type='optimization',
            device=device,
            budget=b,
            n_reps=20
        )

        def process_results(results_list, model_name):
            for res in results_list:
                optimal_val = res['y_test'].max()
                raw_values = np.array(res['values'])
                best_so_far = np.maximum.accumulate(raw_values, axis=1)
                simple_regret_curve = optimal_val - best_so_far

                if regret_metric == 'abs':
                    scores = simple_regret_curve[:, -1]
                elif regret_metric == 'cum':
                    scores = np.mean(simple_regret_curve, axis=1)

                for score in scores:
                    plot_data.append({
                        'Budget': b,
                        'Model': model_name,
                        'Regret': score,
                        'ID': f"{res['subject']}_{res['emg']}"
                    })

        process_results(results_gp, 'GP')
        process_results(results_pfn, 'PFN')

    # --- Visualization ---
    df = pd.DataFrame(plot_data)

    custom_palette = {
        'GP': 'sandybrown',
        'PFN': 'royalblue'
    }

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x='Budget',
        y='Regret',
        hue='Model',
        palette=custom_palette,
        marker='o',
        errorbar=('ci', 95),
        err_kws={'alpha': 0.2},
        linewidth=2
    )

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Budget (Number of Queries)")
    plt.grid(True, alpha=0.3)
    plt.legend(title='Model Type')
    output_dir = os.path.join('output', 'optimization')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f'optimization_budget.svg'
    )
    plt.savefig(plot_path, format="svg")
    plt.show()
    print(f"Saved plot to {plot_path}")

# ============================================
#          Main Runner
# ============================================

def main(datasets, evaluation_type='fit', device='cpu', **kwargs):
    """
    Run GP and PFN evaluations across all datasets.

    Args:
        datasets: list of (dataset_type, subject_idx, emg_idx) tuples
        evaluation_type: 'fit' or 'optimization'
        device: 'cpu' or 'cuda'
        **kwargs: forwarded to evaluate_fit / evaluate_optimization
                  (budget, n_reps, etc.)

    Returns:
        (results_gp, results_pfn) — both lists of result dicts.
    """
    # --- Load PFN Weights --- #
    pfn_weights_path = './libs/PFNs4BO/pfns4bo/final_models/model_hebo_morebudget_9_unused_features_3.pt.gz'
    with gzip.open(pfn_weights_path, 'rb') as f:
        pfn_weights = torch.load(f, map_location=device, weights_only=False)

    results_gp = []
    results_pfn = []

    for experiment in datasets:

        dataset_type, subject_idx, emg_idx = experiment

        if evaluation_type == 'fit':

            res_gp = evaluate_fit(dataset_type, subject_idx, emg_idx, 'gp',
                                  device=device, **kwargs)
            res_pfn = evaluate_fit(dataset_type, subject_idx, emg_idx, 'pfn',
                                   pfn_model_weights=pfn_weights, device=device, **kwargs)

        elif evaluation_type == 'optimization':

            res_gp = evaluate_optimization(dataset_type, subject_idx, emg_idx, 'gp',
                                           device=device, **kwargs)
            res_pfn = evaluate_optimization(dataset_type, subject_idx, emg_idx, 'pfn',
                                            pfn_model_weights=pfn_weights, device=device, **kwargs)

        results_gp.append(res_gp)
        results_pfn.append(res_pfn)

    return results_gp, results_pfn


if __name__ == '__main__':

    nhp_dataset = [
    ('nhp', 0, 0), ('nhp', 0, 1),  ('nhp', 0, 2), ('nhp', 0, 3),  ('nhp', 0, 4),  ('nhp', 0, 5),
    ('nhp',1, 0), ('nhp', 1, 1),  ('nhp', 1, 2), ('nhp', 1, 3),  ('nhp', 1, 4),  ('nhp', 1, 5),  ('nhp',1, 6),  ('nhp', 1, 7),
    ('nhp', 2, 0), ('nhp', 2, 1),  ('nhp', 2, 2), ('nhp', 2, 3),
    ('nhp', 3, 0), ('nhp', 3, 1),  ('nhp', 3, 2), ('nhp', 3, 3)
        ]

    rat_dataset = [
    ('rat', 0, 0), ('rat', 0, 1), ('rat', 0, 2), ('rat', 0, 3), ('rat', 0, 4), ('rat', 0, 5),
    ('rat', 1, 0), ('rat', 1, 1), ('rat', 1, 2), ('rat', 1, 3), ('rat', 1, 4), ('rat', 1, 5), ('rat', 1, 6),
    ('rat', 2, 0), ('rat', 2, 1), ('rat', 2, 2), ('rat', 2, 3), ('rat', 2, 4), ('rat', 2, 5), ('rat', 2, 6), ('rat', 2, 7),
    ('rat', 3, 0), ('rat', 3, 1), ('rat', 3, 2), ('rat', 3, 3), ('rat', 3, 4), ('rat', 3, 5),
    ('rat', 4, 0), ('rat', 4, 1), ('rat', 4, 2), ('rat', 4, 3), ('rat', 4, 4),
    ('rat', 5, 0), ('rat', 5, 1), ('rat', 5, 2), ('rat', 5, 3), ('rat', 5, 4), ('rat', 5, 5), ('rat', 5, 6), ('rat', 5, 7)
        ]

    # Uncomment this section and change field of dataset to test optimization
    #optimization_budget(rat_dataset, regret_metric='abs', budgets=[10, 30, 50, 64, 100])
    results_gp, results_pfn = main(rat_dataset, 'optimization', device='cpu', budget=32)

    results_dict = {'GP': results_gp, 'PFN': results_pfn}

    r2_comparison(results_dict, mode='_opt', save=True)
    #plot_runtime_trajectory(results_dict, save=False)
    #regret_curve(results_dict)

    # Uncomment this section and change field of dataset to test fitness
    #r2_comparison(results_dict, save=True)
    #fit_budget(rat_dataset, 'cpu')

    print(f'Runner will be implemented soon.')
