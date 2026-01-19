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

def run_bo_loop(X_pool, y_pool, model_type, pfn_model=None, n_init=8, budget=100, device='cpu'):
    """
    Performs the Active Learning loop.
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
    # We maintain a mask of available points
    pool_indices = np.arange(n_locs)
    observed_indices = np.random.choice(pool_indices, size=n_init, replace=False).tolist()
    observed_values = [sample_from_pool(idx) for idx in observed_indices]

    times = []

    pfn_wrapper = TransformerBOMethod(pfn_model, device=device, acq_function='ei')

    n_samples = np.zeros(n_locs, dtype=int)
    for idx in observed_indices:
        n_samples[idx] += 1

    kernel = 1.0 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)

    
    # --- LOOP ---
    for t in range(budget - n_init):
        
        
        # Prepare Tensors
        X_train = torch.tensor(X_pool[observed_indices], dtype=torch.float32, device=device)     
        y_train = torch.tensor(observed_values, dtype=torch.float32, device=device)

        X_cand = torch.tensor(X_pool, dtype=torch.float32, device=device)
        
        # --- GP Logic ---
        if model_type == 'gp':
            step_start = time.time()
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
            gp.fit(X_train, y_train)
                
            # Select Next Point (EI)
            best_f = y_train.max()
            acq_vals = expected_improvement2(gp, X_cand, best_f, device )#expected_improvement(model, likelihood, X_cand, best_f, device)      

            # 🔑 Penalize EI for repeated sampling
            #penalty = 1.0 / (1.0 + torch.tensor(n_samples, device=device, dtype=torch.float32))
            #acq_vals = acq_vals * penalty      

            # Get index relative to X_pool (0 to 95)
            next_idx = acq_vals.argmax().item()

        # --- PFN Logic ---
        elif model_type == 'pfn':
            
            step_start = time.time()
            # observe_and_suggest returns (relative_index, value)
            # It expects (N, D) inputs and handles the internal batching/unsqueezing
            idx_rel, acq_vals = pfn_wrapper.observe_and_suggest(
                X_obs=X_train, 
                y_obs=y_train, 
                X_pen=X_cand,
                return_actual_ei=True
            )
            
            if isinstance(idx_rel, torch.Tensor): idx_rel = idx_rel.item()

            #penalty = 1.0 / (1.0 + 0.5*torch.tensor(n_samples, device=device, dtype=torch.float32))
            #acq_vals = acq_vals * penalty
            next_idx = acq_vals.argmax().item()

            #?#next_idx = idx_rel
        
        observed_indices.append(next_idx)  
        new_val = sample_from_pool(next_idx)
        observed_values.append(new_val)  

        step_time = time.time() - step_start
        times.append(step_time)

    return observed_indices, observed_values, times

def evaluate_fit(dataset, subject_idx, emg_idx, model_type, 
                   pfn_model_weights=None, device='cpu', budget=100):
    
    data = load_data(dataset, subject_idx)

    X_train, y_train, X_test, y_test, scaler_y = preprocess_neural_data(data, emg_idx, model_type)

    #subsampling context
    n_reps = y_train.shape[1]
    y_train = y_train.flatten()
    indices = np.random.choice(len(y_train), budget, replace=False)
    X_train=np.repeat(X_train, n_reps, axis=0)[indices]
    y_train=y_train[indices]


    if model_type == 'gp':
        
        start = time.time()
        kernel = 1.0 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gp.fit(X_train, y_train)
        y_pred = gp.predict(X_test)        

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

    times = time.time() - start
    r2 = r2_score(y_test, y_pred)

    return {
        'model_type': model_type,
        'r2': r2, 
        'times': times, 
        'y_test': y_test, 
        'y_pred': y_pred,
        'dataset': dataset,
        'subject': subject_idx,
        'emg': emg_idx
    }
 
def evaluate_optimization(dataset, subject_idx, emg_idx, model_type, 
                   pfn_model_weights=None, device='cpu', budget=100):
    
    data = load_data(dataset, subject_idx)
    
     # 1. Preprocess based on model type
    norm_strategy = 'pfn' if model_type == 'pfn' else 'neural'
    X_train, y_train, X_test, y_test, scalar_y = preprocess_neural_data(
        data, emg_idx, normalization=norm_strategy
    )

    # 2. Run BO Loop
    # We pass pfn_model_weights only if needed.
    traj, values, times = run_bo_loop(
        X_train, y_train, 
        model_type=model_type, 
        pfn_model=pfn_model_weights, 
        n_init=5, 
        budget=budget, 
        device=device
    )

    # 3. Final Prediction & Metric Calculation
    y_pred_scaled = None

    if model_type == 'gp':
        # Re-construct data from trajectory for final fit
        X_final = torch.tensor(X_train[traj], dtype=torch.float32, device=device)
        y_final = torch.tensor(values, dtype=torch.float32, device=device)
        X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device)

        lik = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGP(X_final, y_final, lik).to(device) # Ensure ExactGP is imported
        
        # Train
        model.train(); lik.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        
        for _ in range(50):
            optimizer.zero_grad()
            loss = -mll(model(X_final), y_final)
            loss.backward()
            optimizer.step()
            
        model.eval(); lik.eval()
        with torch.no_grad():
            y_pred_scaled = lik(model(X_test_torch)).mean.cpu().numpy()

    elif model_type == 'pfn':
        # Prepare data for manual forward pass
        X_final = torch.tensor(X_train[traj], dtype=torch.float32, device=device)
        y_final = torch.tensor(values, dtype=torch.float32, device=device)
        X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device)
        
        # Reshape for PFN: (Seq, Batch=1, Dim)
        X_ctx = X_final.unsqueeze(1)
        y_ctx = y_final.unsqueeze(1)
        X_qry = X_test_torch.unsqueeze(1)
        
        pfn_model_weights.eval()
        with torch.no_grad():
            logits = pfn_model_weights(X_ctx, y_ctx, X_qry)
            probs = torch.softmax(logits, dim=-1)
            borders = pfn_model_weights.borders.to(device)
            bucket_centers = (borders[:-1] + borders[1:]) / 2
            y_pred_scaled = (probs * bucket_centers).sum(dim=-1).squeeze(1).cpu().numpy()

    # 4. Inverse Transform for Reporting
    # We must flatten/reshape correctly for inverse_transform
    y_test_raw = scalar_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_raw = scalar_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    values_raw = scalar_y.inverse_transform(np.array(values).reshape(-1, 1)).flatten()

    r2 = r2_score(y_test_raw, y_pred_raw)

    return {
        'model_type': model_type,
        'r2': r2, 
        'times': times, 
        'traj': traj,
        'values': values_raw, # Needed for regret curve
        'y_test': y_test_raw, 
        'y_pred': y_pred_raw,
        'dataset': dataset,
        'subject': subject_idx,
        'emg': emg_idx
    }

def evaluate_models(dataset, subject_idx, emg_idx, 
                   device='cpu', budget=100):

    data = load_data(dataset, subject_idx)

    # --- 0. Load PFN Model Once ---
    model_path = './libs/PFNs4BO/pfns4bo/final_models/model_hebo_morebudget_9_unused_features_3.pt.gz'
    with gzip.open(model_path, 'rb') as f:
        model_weights = torch.load(f, map_location=device, weights_only=False)

    # --- 1. Run GPBO ---
    X_train_gp, y_train_gp, X_test_gp, y_test_gp, scalar_y_gp = preprocess_neural_data(data, emg_idx, normalization='neural')
    gp_traj, gp_values, gp_times = run_bo_loop(X_train_gp, y_train_gp, 'gp', n_init=5, budget=budget, device=device)

    # --- Run PFN BO Loop ---
    X_train_pfn, y_train_pfn, X_test_pfn, y_test_pfn, scalar_y_pfn = preprocess_neural_data(data, emg_idx, normalization='pfn')
    pfn_traj, pfn_values, pfn_times = run_bo_loop(X_train_pfn, y_train_pfn, 'pfn', pfn_model=model_weights, n_init=5, budget=budget, device=device)

    # --- Final Evaluation (R2) ---
        # 1. GP Final Prediction
    X_train_gp = torch.tensor(X_train_gp[gp_traj], dtype=torch.float32, device=device)   
    y_train_gp = torch.tensor(gp_values, dtype=torch.float32, device=device)
    X_test_gp = torch.tensor(X_test_gp, dtype=torch.float32, device=device)

    
    lik_gp = gpytorch.likelihoods.GaussianLikelihood().to(device)
    gp_model = ExactGP(X_train_gp, y_train_gp, lik_gp).to(device)
    gp_model.train()
    lik_gp.train()
    # Fit one last time
    optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik_gp, gp_model)
    for _ in range(50):
        optimizer.zero_grad()
        loss = -mll(gp_model(X_train_gp), y_train_gp)
        loss.backward()
        optimizer.step()
        
    gp_model.eval()
    lik_gp.eval()

    with torch.no_grad():
        y_pred_gp = lik_gp(gp_model(X_test_gp)).mean.cpu().numpy()


    # 2. PFN Final Prediction (using 'mean' acquisition)
    X_train_pfn = torch.tensor(X_train_pfn[pfn_traj], dtype=torch.float32, device=device)
    y_train_pfn = torch.tensor(pfn_values, dtype=torch.float32, device=device)
    
    pfn_evaluator = TransformerBOMethod(model_weights, device=device, acq_function='mean')
    _, y_pred_pfn = pfn_evaluator.observe_and_suggest(
        X_obs=X_train_pfn, 
        y_obs=y_train_pfn, 
        X_pen=X_test_pfn, 
        return_actual_ei=True
    )
    if isinstance(y_pred_pfn, torch.Tensor): y_pred_pfn = y_pred_pfn.cpu().numpy()    


    r2_gp = r2_score(scalar_y_gp.inverse_transform(y_test_gp.reshape(-1,1)), scalar_y_gp.inverse_transform(y_pred_gp.reshape(-1,1)))
    r2_pfn = r2_score(scalar_y_pfn.inverse_transform(y_test_pfn.reshape(-1,1)), scalar_y_pfn.inverse_transform(y_pred_pfn.reshape(-1,1)))

    regret_curve(y_pred_gp, y_pred_pfn, max(y_train_gp), max(y_train_pfn))

    return {
        'r2_gp': r2_gp, 'gp_times': gp_times, 'gp_traj': gp_traj,
        'r2_pfn': r2_pfn, 'pfn_times': pfn_times, 'pfn_traj': pfn_traj,
        'y_test': y_test_pfn, 'y_pred_gp': y_pred_gp, 'y_pred_pfn': y_pred_pfn }


# ============================================
#          Main Runner
# ============================================

def main(datasets, evaluation_type='fit', device='cpu'):
    
    results = []

    # --- Load PFN Weights --- #
    model_path = './libs/PFNs4BO/pfns4bo/final_models/model_hebo_morebudget_9_unused_features_3.pt.gz'
    with gzip.open(model_path, 'rb') as f:
        # Load weights_only=False as discussed previously
        pfn_weights = torch.load(f, map_location=device, weights_only=False)
    
    results_gp = []
    results_pfn = []

    for experiment in datasets:

        dataset_type, subject_idx, emg_idx = experiment

        if evaluation_type == 'fit':

            res_gp = evaluate_fit(dataset_type, subject_idx, emg_idx, 'gp', device=device)
            res_pfn = evaluate_fit(dataset_type, subject_idx, emg_idx, 'pfn', pfn_model_weights=pfn_weights, device=device)

        elif evaluation_type == 'optimization':

            res_gp = evaluate_optimization(dataset_type, subject_idx, emg_idx, 'gp', device=device)
            res_pfn = evaluate_optimization(dataset_type, subject_idx, emg_idx, 'pfn', pfn_model_weights=pfn_weights, device=device)

        results_gp.append(res_gp)
        results_pfn.append(res_pfn)

    return results_gp, results_pfn


if __name__ == '__main__':

    datasets = [
    ('nhp', 0, 0), #('nhp', 0, 1),  ('nhp', 0, 2), ('nhp', 0, 3)
            ]
    
    results_gp, results_pfn = main(datasets, 'fit', device='cpu')

    r2_comparison(results_gp, results_pfn)

    exit(0)

    # 2. Detailed Plots per Experiment
    for i in range(len(results_gp)):
        
        # A. Regret Curve (Run in main as requested)
        # Note: You need the 'values' key from the result dict
        # We define "Optimal" as the max of the test set (ground truth)
        optimal_val = max(results_gp[i]['y_test'].max(), results_pfn[i]['y_test'].max())
        
        regret_curve(
            results_gp[i]['values'], 
            results_pfn[i]['values'], 
            optimal_val
        )

        # B. EMG Maps
        show_emg_map(results_gp[i], "Gaussian Process")
        show_emg_map(results_pfn[i], "PFN")