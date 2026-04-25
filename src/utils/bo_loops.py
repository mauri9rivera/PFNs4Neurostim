"""
Bayesian optimization loop implementations for GP and finetuned TabPFN.

Public API (use these):
- run_bo_loop(): unified model-agnostic BO loop (canonical implementation)
- _snapshot_iters(): compute log2-spaced snapshot iterations

Deprecated (kept for backwards compatibility, will be removed):
- run_gpbo_loop(): GP-based active learning loop
- run_finetunedbo_loop(): TabPFN-based active learning loop
"""
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import gpytorch

from models.gaussians import ExactGP
from utils.gpbo_utils import compute_ucb_kappa, _auto_kappa_max, _auto_kappa_min


def _snapshot_iters(budget, n_init):
    """Compute log2-spaced iteration counts (total observations, 1-indexed) for snapshots."""
    iters = set()
    i = 1
    while n_init + i <= budget:
        iters.add(n_init + i)
        i *= 2
    iters.add(budget)  # always include final
    return sorted(iters)


def run_bo_loop(
    model: "SurrogateModel",
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_init: int = 5,
    budget: int = 100,
    kappa_schedule: float = 0.0,
    snapshot_iters: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Unified model-agnostic Bayesian optimisation loop using UCB acquisition.

    Performs a sequential active learning loop over a discrete candidate pool.
    At each step the surrogate is refitted on all observed data and the next
    query is selected by the surrogate's UCB acquisition function.

    The ``model`` argument must conform to the ``SurrogateModel`` protocol
    (defined in ``models.regressors``):
      - ``model.fit(X, y)`` — update the surrogate on observed data
      - ``model.predict(X)`` — return ``(mean, std)`` for candidate points
      - ``model.predict_ucb(X, kappa, t, n_steps)`` — return UCB values
        (falls back to ``mean + kappa * std`` if not implemented)

    Observations are drawn as the mean across all repetitions in ``y_pool``
    (``y_pool[idx].mean()``), matching the protocol used in the legacy loops.

    Kappa schedule: controlled by ``kappa_schedule``.  ``0.0`` activates cosine
    annealing from an auto-computed upper bound (``_auto_kappa_max``) down to an
    auto-computed lower bound (``_auto_kappa_min``), both derived from input
    dimensionality and number of active steps via GP-UCB theory scaling.  Any
    non-zero value fixes kappa at that constant throughout the loop.

    Args:
        model: Any object conforming to the ``SurrogateModel`` protocol.
        X_pool: Feature matrix for all candidate locations, shape [N, D].
        y_pool: Response matrix with repeated measurements, shape [N, n_reps].
            Each row corresponds to one candidate location; the observation
            returned for a queried location is the row mean.
        X_test: Test feature matrix for final R² prediction, shape [M, D].
        y_test: Test response vector (mean across repetitions), shape [M].
        n_init: Number of randomly selected initial observations before the
            optimisation loop starts.
        budget: Total number of observations (including ``n_init``).
        kappa_schedule: UCB exploration coefficient control.
            ``0.0`` (default) → cosine-annealed auto schedule:
              kappa_max = 2.5 * sqrt(d * log(n_steps)), floor 3.0;
              kappa_min = 0.2 * sqrt(d * log(n_steps)).
            Any other value → fixed kappa throughout (no annealing).
            Use fixed values for dataset-specific hyperparameter search.
        snapshot_iters: Optional list of observation counts at which to record
            a prediction snapshot on ``X_test`` (e.g. for R²-vs-budget plots).
            The final budget iteration is always included if provided.

    Returns:
        Dictionary with keys:
          - ``'observed_indices'``: list[int] — query indices into ``X_pool``
          - ``'observed_values'``: list[float] — noisy (mean-rep) observations
          - ``'real_values'``: list[float] — true test values at observed locs
          - ``'times'``: list[float] — per-step wall-clock time in seconds
          - ``'y_pred'``: np.ndarray shape [M] — final predictions on ``X_test``
          - ``'snapshots'``: dict[int, dict] | None — at each snapshot iteration:
            ``{'y_pred': np.ndarray [M], 'best_pred_val': float}`` (normalised
            space), or ``None`` if ``snapshot_iters`` is ``None``

    Raises:
        ValueError: If ``budget <= n_init``.
        RuntimeError: If NaN/Inf values appear in UCB acquisition values.
    """
    if budget <= n_init:
        raise ValueError(
            f"budget ({budget}) must be greater than n_init ({n_init})."
        )

    n_locs = X_pool.shape[0]   # [N, D]
    d = X_pool.shape[1]        # [N, D]
    n_steps = budget - n_init

    # Pre-compute auto kappa bounds once (only used when kappa_schedule == 0.0)
    if kappa_schedule == 0.0:
        _kappa_max = _auto_kappa_max(d, n_steps)
        _kappa_min = _auto_kappa_min(d, n_steps)

    def _sample(idx: int) -> float:
        """Return mean observation for pool index idx."""
        return float(y_pool[idx].mean())

    # --- Initialisation: random seed queries ---
    pool_indices = np.arange(n_locs)
    observed_indices: list[int] = np.random.choice(
        pool_indices, size=n_init, replace=False
    ).tolist()
    observed_values: list[float] = [_sample(i) for i in observed_indices]
    real_values: list[float] = [float(y_test[i]) for i in observed_indices]

    times: list[float] = []
    snapshots: dict[int, np.ndarray] = {}

    # --- BO loop ---
    for t in range(n_steps):
        step_start = time.time()

        X_obs = X_pool[observed_indices]          # [n_obs, D]
        y_obs = np.array(observed_values)         # [n_obs]

        # Refit surrogate on all observations so far
        model.fit(X_obs, y_obs)

        # Compute kappa for this step
        if kappa_schedule == 0.0:
            kappa = compute_ucb_kappa(t, n_steps, kappa_max=_kappa_max, kappa_min=_kappa_min)
        else:
            kappa = kappa_schedule

        # Compute UCB acquisition values for all candidates
        # Prefer native predict_ucb if available; fall back to mean + kappa*std
        if hasattr(model, 'predict_ucb') and callable(model.predict_ucb):
            ucb_vals = model.predict_ucb(X_pool, kappa, t, n_steps)  # [N]
            # Convert torch tensors if necessary
            if hasattr(ucb_vals, 'numpy'):
                ucb_vals = ucb_vals.numpy()
            ucb_vals = np.asarray(ucb_vals, dtype=np.float64)         # [N]
        else:
            mean, std = model.predict(X_pool)   # [N], [N]
            ucb_vals = mean + kappa * std        # [N]

        if not np.isfinite(ucb_vals).any():
            raise RuntimeError(
                f"run_bo_loop: all UCB values are non-finite at step {t}. "
                "Check surrogate fit and input data for NaN/Inf."
            )

        # Mask already-observed locations to prevent revisiting
        for obs_idx in observed_indices:
            ucb_vals[obs_idx] = -np.inf

        next_idx = int(np.argmax(ucb_vals))
        observed_indices.append(next_idx)
        observed_values.append(_sample(next_idx))
        real_values.append(float(y_test[next_idx]))

        times.append(time.time() - step_start)

        # Record snapshot if requested
        n_obs_now = len(observed_indices)
        if snapshot_iters is not None and n_obs_now in snapshot_iters:
            # Refit on updated observations for the snapshot prediction
            model.fit(X_pool[observed_indices], np.array(observed_values))
            snap_pred, _ = model.predict(X_test)   # [M] — for R² computation
            pool_mean, _ = model.predict(X_pool)   # [N] — for exploration score
            snapshots[n_obs_now] = {
                'y_pred': np.asarray(snap_pred),                          # [M]
                'best_pred_val': float(np.max(np.asarray(pool_mean))),   # scalar (normalised)
            }

    # --- Final prediction on X_test using all observed data ---
    model.fit(X_pool[observed_indices], np.array(observed_values))
    y_pred, _ = model.predict(X_test)  # [M]
    y_pred = np.asarray(y_pred)        # [M]

    # Capture final budget snapshot if not already recorded
    if snapshot_iters is not None and budget not in snapshots:
        pool_mean_final, _ = model.predict(X_pool)  # [N]
        snapshots[budget] = {
            'y_pred': y_pred.copy(),                                           # [M]
            'best_pred_val': float(np.max(np.asarray(pool_mean_final))),      # scalar (normalised)
        }

    return {
        'observed_indices': observed_indices,
        'observed_values': observed_values,
        'real_values': real_values,
        'times': times,
        'y_pred': y_pred,
        'snapshots': snapshots if snapshot_iters is not None else None,
    }


# ---------------------------------------------------------------------------
# DEPRECATED — kept for backwards compatibility; use run_bo_loop() instead
# ---------------------------------------------------------------------------


def run_gpbo_loop(X_pool, y_pool, x_test, y_test,
                  n_init=5, budget=100, device='cpu', snapshot_iters=None):
    """
    .. deprecated::
        Use ``run_bo_loop(GPSurrogate(device=device), ...)`` instead.
        This function is kept for backwards compatibility and will be removed
        in a future sprint.

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
        kappa = compute_ucb_kappa(t, n_steps, kappa_max=5.0, kappa_min=1.0)
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
    .. deprecated::
        Use ``run_bo_loop(TabPFNSurrogate(model), ...)`` instead.
        This function is kept for backwards compatibility and will be removed
        in a future sprint.

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
        kappa = compute_ucb_kappa(t, n_steps, kappa_max=2.5, kappa_min=0.5)
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
