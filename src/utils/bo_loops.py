"""
Bayesian optimization loop implementations for GP and finetuned TabPFN.

- run_gpbo_loop(): GP-based active learning loop
- run_finetunedbo_loop(): TabPFN-based active learning loop
- _snapshot_iters(): compute log2-spaced snapshot iterations
"""
import math
import time

import numpy as np
import torch
import gpytorch

from models.gaussians import ExactGP
from utils.gpbo_utils import compute_ucb_kappa


def _snapshot_iters(budget, n_init):
    """Compute log2-spaced iteration counts (total observations, 1-indexed) for snapshots."""
    iters = set()
    i = 1
    while n_init + i <= budget:
        iters.add(n_init + i)
        i *= 2
    iters.add(budget)  # always include final
    return sorted(iters)


def run_gpbo_loop(X_pool, y_pool, x_test, y_test,
                  n_init=5, budget=100, device='cpu', snapshot_iters=None):
    """
    Performs the Active Learning loop using a GP model.

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
        - snapshots: dict or None
    """
    n_locs, n_reps = y_pool.shape

    def sample_from_pool(idx):
        return float(y_pool[idx, :].mean())

    # 1. Initialization (Random)
    pool_indices = np.arange(n_locs)
    observed_indices = np.random.choice(pool_indices, size=n_init, replace=False).tolist()
    observed_values = [sample_from_pool(idx) for idx in observed_indices]
    real_values = y_test[observed_indices].tolist()

    times = []
    snapshots = {}
    n_steps = budget - n_init

    # --- LOOP ---
    for t in range(n_steps):
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

        # Select Next Point (UCB with cosine-annealed kappa)
        kappa = compute_ucb_kappa(t, n_steps, kappa_0=5.0, kappa_min=1.0)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            posterior = likelihood(model(X_cand))
            mean = posterior.mean
            sigma = posterior.stddev
        acq_vals = mean + kappa * sigma
        # Prevent revisiting already-observed locations
        for obs_idx in observed_indices:
            acq_vals[obs_idx] = -float('inf')
        next_idx = acq_vals.argmax().item()

        observed_indices.append(next_idx)
        new_val = sample_from_pool(next_idx)
        observed_values.append(new_val)
        real_values.append(y_test[next_idx])

        step_time = time.time() - step_start
        times.append(step_time)

        # Capture snapshot if requested (refit on updated observations)
        if snapshot_iters is not None and len(observed_indices) in snapshot_iters:
            _X_snap = torch.tensor(X_pool[observed_indices], dtype=torch.float32, device=device)
            _y_snap = torch.tensor(observed_values, dtype=torch.float32, device=device)
            _lik_snap = gpytorch.likelihoods.GaussianLikelihood().to(device)
            _gp_snap = ExactGP(_X_snap, _y_snap, _lik_snap).to(device)
            _gp_snap.train(); _lik_snap.train()
            _opt_snap = torch.optim.Adam(_gp_snap.parameters(), lr=0.01)
            _mll_snap = gpytorch.mlls.ExactMarginalLogLikelihood(_lik_snap, _gp_snap)
            for _ in range(50):
                _opt_snap.zero_grad()
                _out = _gp_snap(_X_snap)
                _loss = -_mll_snap(_out, _y_snap)
                _loss.backward()
                _opt_snap.step()
            _gp_snap.eval(); _lik_snap.eval()
            with torch.no_grad():
                snap_post = _lik_snap(_gp_snap(torch.tensor(x_test, dtype=torch.float32, device=device)))
                snapshots[len(observed_indices)] = snap_post.mean.cpu().numpy()

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

    # Capture final snapshot (budget) if not already captured in loop
    if snapshot_iters is not None and budget not in snapshots:
        snapshots[budget] = y_pred.copy()

    return observed_indices, observed_values, real_values, times, y_pred, \
        (snapshots if snapshot_iters else None)


def run_finetunedbo_loop(X_pool, y_pool, x_test, y_test, model,
                          n_init=5, budget=100, device='cpu', snapshot_iters=None):
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
        - snapshots: dict or None — {iter: y_pred_array} at snapshot iterations
    """
    n_locs, n_reps = y_pool.shape

    def sample_from_pool(idx):
        return float(y_pool[idx, :].mean())

    # 1. Initialization (Random)
    pool_indices = np.arange(n_locs)
    observed_indices = np.random.choice(pool_indices, size=n_init, replace=False).tolist()
    observed_values = [sample_from_pool(idx) for idx in observed_indices]
    real_values = y_test[observed_indices].tolist()

    times = []
    snapshots = {}
    n_steps = budget - n_init

    # --- LOOP ---
    for t in range(n_steps):
        step_start = time.time()

        X_obs_np = X_pool[observed_indices]
        y_obs_np = np.array(observed_values)

        # Fit provides in-context examples for the transformer
        model.fit(X_obs_np, y_obs_np)

        # Compute UCB directly from the bar distribution (no Gaussian assumption)
        # kappa → rest_prob via: rest_prob = 0.5 * erfc(kappa / sqrt(2))
        kappa = compute_ucb_kappa(t, n_steps, kappa_0=2.5, kappa_min=0.5)
        rest_prob = 0.5 * math.erfc(kappa / math.sqrt(2))
        full_output = model.predict(X_pool, output_type="full")
        logits = full_output['logits']
        criterion = full_output['criterion']
        ucb_vals = criterion.ucb(logits, 0, rest_prob=rest_prob, maximize=True)
        ucb_vals = ucb_vals.clone()
        # Prevent revisiting already-observed locations
        for obs_idx in observed_indices:
            ucb_vals[obs_idx] = -float('inf')
        next_idx = int(ucb_vals.argmax().item())

        observed_indices.append(next_idx)
        new_val = sample_from_pool(next_idx)
        observed_values.append(new_val)
        real_values.append(y_test[next_idx])

        step_time = time.time() - step_start
        times.append(step_time)

        # Capture snapshot if requested (refit on updated observations)
        if snapshot_iters is not None and len(observed_indices) in snapshot_iters:
            model.fit(X_pool[observed_indices], np.array(observed_values))
            snap_pred = model.predict(x_test)
            snapshots[len(observed_indices)] = np.asarray(snap_pred)

    # Final prediction with all observed data as context
    X_obs_final = X_pool[observed_indices]
    y_obs_final = np.array(observed_values)
    model.fit(X_obs_final, y_obs_final)
    y_pred = model.predict(x_test)

    # Capture final snapshot (budget) if not already captured in loop
    if snapshot_iters is not None and budget not in snapshots:
        snapshots[budget] = np.asarray(y_pred).copy()

    return observed_indices, observed_values, real_values, times, np.asarray(y_pred), \
        (snapshots if snapshot_iters else None)
