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


def _draw_valid_rep(y_pool: np.ndarray, idx: int, fallback: float) -> float:
    """Draw one noisy trial from ``y_pool[idx]``, skipping NaN-masked invalid reps.

    Models a real neurostim query: one trial drawn uniformly at random from
    the *valid* repetitions at site ``idx``.  Invalid trials (flagged by
    ``sorted_isvalid`` and masked to NaN in ``preprocess_neural_data``) are
    excluded from the draw.  Falls back to ``fallback`` if every rep at this
    site is invalid (defensive — should be rare in practice).
    """
    site_reps = y_pool[idx] if y_pool.ndim > 1 else np.atleast_1d(y_pool[idx])
    valid_mask = ~np.isnan(site_reps)
    if not valid_mask.any():
        return float(fallback)
    valid_idx = np.flatnonzero(valid_mask)
    chosen = int(np.random.choice(valid_idx))
    return float(site_reps[chosen])


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
    acq_fn: str = 'ucb',
    ts_temperature: float = 1.0,
) -> Dict[str, Any]:
    """Unified model-agnostic Bayesian optimisation loop.

    Performs a sequential active learning loop over a discrete candidate pool.
    At each step the surrogate is refitted on all observed data and the next
    query is selected via the chosen acquisition function.

    The ``model`` argument must conform to the ``SurrogateModel`` protocol
    (defined in ``models.regressors``):
      - ``model.fit(X, y)`` — update the surrogate on observed data
      - ``model.predict(X)`` — return ``(mean, std)`` for candidate points
      - ``model.predict_ucb(X, kappa, t, n_steps)`` — UCB acquisition values
        (used when ``acq_fn='ucb'``; falls back to ``mean + kappa * std``)
      - ``model.predict_ts(X, temperature)`` — Thompson Sample values
        (used when ``acq_fn='ts'``)

    Observations model a real neurostim experiment: each query returns a
    single noisy trial drawn uniformly at random from the *valid* reps in
    ``y_pool[idx, :]`` (NaN-flagged invalid reps are skipped).  The agent
    never sees the per-site mean; that is reserved as offline ground truth in
    ``y_test`` for regret/R² evaluation.

    Revisits are allowed: re-querying a previously observed site yields a
    fresh independent trial and reduces variance at that site.

    UCB kappa schedule: controlled by ``kappa_schedule`` when ``acq_fn='ucb'``.
    ``0.0`` activates cosine annealing from ``_auto_kappa_max`` down to
    ``_auto_kappa_min``, derived from input dimensionality and number of active
    steps via GP-UCB theory scaling.  Any non-zero value fixes kappa constant.

    Args:
        model: Any object conforming to the ``SurrogateModel`` protocol.
        X_pool: Feature matrix for all candidate locations, shape [N, D].
        y_pool: Response matrix with repeated measurements, shape [N, n_reps].
            Each row corresponds to one candidate location; each query draws
            one noisy trial uniformly at random from ``y_pool[idx, :]``.
        X_test: Test feature matrix for final R² prediction, shape [M, D].
        y_test: Ground-truth test responses (per-site mean across reps),
            shape [M] — never observed by the agent; used only offline.
        n_init: Number of randomly selected initial observations before the
            optimisation loop starts.
        budget: Total number of observations (including ``n_init``).
        kappa_schedule: UCB exploration coefficient control (used when
            ``acq_fn='ucb'``).  ``0.0`` (default) → cosine-annealed auto
            schedule; any other value → fixed kappa throughout.
        snapshot_iters: Optional list of observation counts at which to record
            a prediction snapshot on ``X_test`` (e.g. for R²-vs-budget plots).
            The final budget iteration is always included if provided.
        acq_fn: Acquisition function choice.  ``'ucb'`` (default) uses
            Upper Confidence Bound acquisition; ``'ts'`` uses Thompson Sampling.
        ts_temperature: Temperature for TabPFN bar-distribution sampling when
            ``acq_fn='ts'``.  ``1.0`` = exact predictive distribution;
            ``<1.0`` = sharper / greedier; ``>1.0`` = more uniform.

    Returns:
        Dictionary with keys:
          - ``'observed_indices'``: list[int] — query indices into ``X_pool``
          - ``'observed_values'``: list[float] — noisy single-trial draws
            from ``y_pool`` (what the agent actually sees during BO)
          - ``'real_values'``: list[float] — ground-truth ``y_test`` values
            at observed locations (offline reference, not seen by the agent)
          - ``'times'``: list[float] — per-step wall-clock time in seconds
          - ``'y_pred'``: np.ndarray shape [M] — final predictions on ``X_test``
          - ``'snapshots'``: dict[int, dict] | None — at each snapshot iteration:
            ``{'y_pred': np.ndarray [M], 'best_pred_val': float}`` (normalised
            space), or ``None`` if ``snapshot_iters`` is ``None``

    Raises:
        ValueError: If ``budget <= n_init`` or ``acq_fn`` is not recognised.
        RuntimeError: If NaN/Inf values appear in acquisition values.
    """
    if budget <= n_init:
        raise ValueError(
            f"budget ({budget}) must be greater than n_init ({n_init})."
        )
    if acq_fn not in ('ucb', 'ts'):
        raise ValueError(
            f"Unknown acq_fn: {acq_fn!r}. Choose 'ucb' or 'ts'."
        )

    n_locs = X_pool.shape[0]   # [N, D]
    d = X_pool.shape[1]        # [N, D]
    n_steps = budget - n_init

    # Pre-compute auto kappa bounds once (only used when acq_fn='ucb' and kappa_schedule==0.0)
    if acq_fn == 'ucb' and kappa_schedule == 0.0:
        _kappa_max = _auto_kappa_max(d, n_steps)
        _kappa_min = _auto_kappa_min(d, n_steps)

    def _sample(idx: int) -> float:
        return _draw_valid_rep(y_pool, idx, fallback=float(y_test[idx]))

    # --- Initialisation: random seed queries ---
    pool_indices = np.arange(n_locs)
    observed_indices: list[int] = np.random.choice(
        pool_indices, size=n_init, replace=False
    ).tolist()
    observed_values: list[float] = [_sample(i) for i in observed_indices]
    real_values: list[float] = [float(y_test[i]) for i in observed_indices]

    times: list[float] = []
    snapshots: dict[int, np.ndarray] = {}
    # Pure-exploitation recommendation at each BO step (argmax of posterior mean).
    # Distinct from the UCB-selected query when kappa > 0.
    best_rec_indices: list[int] = []
    # Exploitation score: ground-truth response at the recommended location / optimal.
    # Stored at every BO step (not just snapshots) to enable dense trajectory plots.
    optimal_y = float(y_test.max())
    perf_explore: list[float] = []

    # --- BO loop ---
    for t in range(n_steps):
        step_start = time.time()

        X_obs = X_pool[observed_indices]          # [n_obs, D]
        y_obs = np.array(observed_values)         # [n_obs]

        # Refit surrogate on all observations so far
        model.fit(X_obs, y_obs)

        # --- Acquisition: select next query ---
        if acq_fn == 'ts':
            # Thompson Sampling: draw one function sample from the posterior
            acq_vals = np.asarray(
                model.predict_ts(X_pool, temperature=ts_temperature), dtype=np.float64
            )  # [N]
            pool_mean, _ = model.predict(X_pool)  # [N] — for exploitation rec.

        else:  # acq_fn == 'ucb'
            # Compute UCB kappa for this step (cosine-annealed or fixed)
            kappa = (
                compute_ucb_kappa(t, n_steps, kappa_max=_kappa_max, kappa_min=_kappa_min)
                if kappa_schedule == 0.0 else kappa_schedule
            )
            # Prefer native predict_ucb (bar-distribution UCB for TabPFN);
            # fall back to mean + kappa*std for surrogates without predict_ucb.
            if hasattr(model, 'predict_ucb') and callable(model.predict_ucb):
                acq_vals = np.asarray(
                    model.predict_ucb(X_pool, kappa, t, n_steps), dtype=np.float64
                )  # [N]
                pool_mean, _ = model.predict(X_pool)  # [N]
            else:
                pool_mean, std = model.predict(X_pool)  # [N], [N]
                acq_vals = (
                    np.asarray(pool_mean, dtype=np.float64)
                    + kappa * np.asarray(std, dtype=np.float64)
                )  # [N]

        pool_mean_arr = np.asarray(pool_mean, dtype=np.float64)        # [N]
        best_rec_idx = int(np.argmax(pool_mean_arr))
        best_rec_indices.append(best_rec_idx)                          # pure-exploit recommendation
        perf_explore.append(
            float(y_test[best_rec_idx]) / optimal_y if optimal_y > 1e-8 else 0.0
        )

        if not np.isfinite(acq_vals).any():
            raise RuntimeError(
                f"run_bo_loop: all {acq_fn.upper()} acquisition values are non-finite "
                f"at step {t}. Check surrogate fit and input data for NaN/Inf."
            )

        # Revisits are allowed: the noise model means re-querying a site
        # yields a fresh (independent) trial, providing variance reduction.
        next_idx = int(np.argmax(acq_vals))
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
        'best_rec_indices': best_rec_indices,   # list[int], length = n_steps
        'perf_explore': perf_explore,           # list[float], length = n_steps
    }


# ---------------------------------------------------------------------------
# DEPRECATED — kept for backwards compatibility; use run_bo_loop() instead
# ---------------------------------------------------------------------------


def run_gpbo_loop(X_pool, y_pool, x_test, y_test,
                  n_init=5, budget=100, device='cpu', snapshot_iters=None,
                  kappa_schedule: float = 0.0):
    """
    .. deprecated::
        Use ``run_bo_loop(GPSurrogate(device=device), ...)`` instead.
        This function is kept for backwards compatibility and will be removed
        in a future sprint.

    Performs the Active Learning loop using a GP model.

    Args:
        X_pool: Feature matrix for the candidate pool (n_locs, n_features).
        y_pool: Response matrix (n_locs, n_reps) with noisy single-trial
            observations.  Each query draws one rep uniformly at random.
        x_test: Test feature matrix for final prediction.
        y_test: Ground-truth response vector (per-site mean across reps),
            never observed by the agent.
        n_init: Number of random initial observations.
        budget: Total number of observations (including initial).
        device: 'cpu' or 'cuda'.

    Returns:
        - observed_indices: Indices of points chosen
        - observed_values: Noisy single-trial draws from y_pool
        - real_values: Ground-truth y_test values at observed indices
        - times: Time taken at each step
        - y_pred: Final predictions on x_test
        - snapshots: dict or None
    """
    n_locs, n_reps = y_pool.shape

    def sample_from_pool(idx):
        return _draw_valid_rep(y_pool, idx, fallback=float(y_test[idx]))

    # 1. Initialization (Random)
    pool_indices = np.arange(n_locs)
    observed_indices = np.random.choice(pool_indices, size=n_init, replace=False).tolist()
    observed_values = [sample_from_pool(idx) for idx in observed_indices]
    real_values = y_test[observed_indices].tolist()

    times = []
    snapshots = {}
    best_rec_indices = []
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

        # Select Next Point (UCB with cosine-annealed or fixed kappa)
        if kappa_schedule == 0.0:
            kappa = compute_ucb_kappa(t, n_steps, kappa_max=5.0, kappa_min=1.0)
        else:
            kappa = kappa_schedule
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            posterior = likelihood(model(X_cand))
            mean = posterior.mean
            sigma = posterior.stddev
        acq_vals = mean + kappa * sigma
        # Pure-exploitation recommendation = argmax of posterior mean
        best_rec_indices.append(int(mean.argmax().item()))
        # Revisits allowed (single-trial noise model — re-queries are informative).
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
                pool_post = _lik_snap(_gp_snap(torch.tensor(X_pool, dtype=torch.float32, device=device)))
                snapshots[len(observed_indices)] = {
                    'y_pred': snap_post.mean.cpu().numpy(),
                    'best_pred_val': float(pool_post.mean.max().item()),
                }

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
        with torch.no_grad():
            pool_posterior = likelihood(model(torch.tensor(X_pool, dtype=torch.float32, device=device)))
        snapshots[budget] = {
            'y_pred': y_pred.copy(),
            'best_pred_val': float(pool_posterior.mean.max().item()),
        }

    return observed_indices, observed_values, real_values, times, y_pred, \
        (snapshots if snapshot_iters else None), best_rec_indices


def run_finetunedbo_loop(X_pool, y_pool, x_test, y_test, model,
                          n_init=5, budget=100, device='cpu', snapshot_iters=None,
                          kappa_schedule: float = 0.0):
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
        kappa_schedule: UCB exploration coefficient.  ``0.0`` (default) =
            auto cosine-annealed schedule (kappa_max=2.5, kappa_min=0.5);
            any other value = fixed kappa throughout the BO loop.

    Returns:
        - observed_indices: Indices of points chosen
        - observed_values: Noisy single-trial draws from y_pool
        - real_values: Ground-truth y_test values at observed indices
        - times: Time taken at each step
        - y_pred: Final predictions on x_test
        - snapshots: dict or None — {iter: y_pred_array} at snapshot iterations
    """
    n_locs, n_reps = y_pool.shape

    def sample_from_pool(idx):
        return _draw_valid_rep(y_pool, idx, fallback=float(y_test[idx]))

    # 1. Initialization (Random)
    pool_indices = np.arange(n_locs)
    observed_indices = np.random.choice(pool_indices, size=n_init, replace=False).tolist()
    observed_values = [sample_from_pool(idx) for idx in observed_indices]
    real_values = y_test[observed_indices].tolist()

    times = []
    snapshots = {}
    best_rec_indices = []
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
        if kappa_schedule == 0.0:
            kappa = compute_ucb_kappa(t, n_steps, kappa_max=2.5, kappa_min=0.5)
        else:
            kappa = kappa_schedule
        rest_prob = 0.5 * math.erfc(kappa / math.sqrt(2))
        full_output = model.predict(X_pool, output_type="full")
        logits = full_output['logits']
        criterion = full_output['criterion']
        ucb_vals = criterion.ucb(logits, 0, rest_prob=rest_prob, maximize=True)
        ucb_vals = ucb_vals.clone()
        # Pure-exploitation recommendation = argmax of posterior mean
        pool_mean = model.predict(X_pool)                            # [N]
        best_rec_indices.append(int(np.argmax(np.asarray(pool_mean))))
        # Revisits allowed (single-trial noise model — re-queries are informative).
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
            snap_pool_mean = model.predict(X_pool)
            snapshots[len(observed_indices)] = {
                'y_pred': np.asarray(snap_pred),
                'best_pred_val': float(np.max(np.asarray(snap_pool_mean))),
            }

    # Final prediction with all observed data as context
    X_obs_final = X_pool[observed_indices]
    y_obs_final = np.array(observed_values)
    model.fit(X_obs_final, y_obs_final)
    y_pred = model.predict(x_test)

    # Capture final snapshot (budget) if not already captured in loop
    if snapshot_iters is not None and budget not in snapshots:
        pool_mean_final = model.predict(X_pool)
        snapshots[budget] = {
            'y_pred': np.asarray(y_pred).copy(),
            'best_pred_val': float(np.max(np.asarray(pool_mean_final))),
        }

    return observed_indices, observed_values, real_values, times, np.asarray(y_pred), \
        (snapshots if snapshot_iters else None), best_rec_indices
