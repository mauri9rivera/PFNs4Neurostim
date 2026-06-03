"""
Evaluation functions for finetuned TabPFN and GP baselines.

- gp_baseline(): GP optimization evaluation
- finetuned_optimization(): evaluate optimization with finetuned TabPFN
- finetuned_optimization_budget(): budget sweep for optimization evaluation
- finetuned_percentage(): augmentation ablation study
- load_sweep_results(): load and merge sweep DataFrames from disk
"""
from __future__ import annotations

import copy
import os
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor

from models.gaussians import ExactGP
from models.regressors import (
    _make_finetuned_regressor, extract_inference_model,
    GPSurrogate, TabPFNSurrogate, SurrogateModel,
)
from utils.bo_loops import run_bo_loop, _snapshot_iters
from utils.data_utils import (
    build_finetuning_dataset, load_data, preprocess_neural_data,
    HELD_OUT_SUBJECTS, TRAIN_SUBJECTS, ALL_SUBJECTS,
    generate_experiment_tag, save_results,
    create_run_dir, write_run_config,
    load_subject_result, save_subject_result,
)
from utils.surface_geometry import compute_gfs
from utils.visualization import (
    r2_by_subject,
    regret_with_timing, regret_by_subject, regret_by_emg,
    budget_sweep_plot, augmentation_sweep_plot,
    visualize_representation, show_emg_map,
    plot_gradient_metrics, plot_weight_metrics, plot_cka_similarity,
)


def _valid_site_mask(data: dict, emg_idx: int) -> np.ndarray:
    """Boolean mask over channels with at least one valid rep at ``emg_idx``.

    Mirrors the per-site filter in :func:`utils.data_utils.preprocess_neural_data`
    so callers can align side-channel arrays (``ch2xy``, …) with the filtered
    ``y_test`` / ``y_pred`` length.
    """
    if 'sorted_isvalid' not in data:
        return np.ones(data['sorted_resp'].shape[0], dtype=bool)
    return (data['sorted_isvalid'][:, emg_idx, :] != 0).any(axis=-1)


def _flatten_valid_pool(
    X_pool: np.ndarray, Y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``Y_train`` ``[N, n_reps]`` into per-trial (X, y) pairs, dropping
    NaN entries from sites whose corresponding trial was flagged invalid by
    ``sorted_isvalid`` (see ``preprocess_neural_data``).

    The flatten uses C-order, matching ``np.repeat(X_pool, n_reps, axis=0)``,
    so X-y alignment is preserved when the same valid mask filters both.

    Returns:
        ``(X_flat, y_flat)``: aligned arrays of shape ``[n_valid, D]`` and
        ``[n_valid]`` containing only finite trials.
    """
    n_reps = Y_train.shape[1]
    X_rep = np.repeat(X_pool, n_reps, axis=0)     # [N * n_reps, D]
    y_flat = Y_train.flatten()                     # [N * n_reps]
    valid = ~np.isnan(y_flat)
    return X_rep[valid], y_flat[valid]


def gp_baseline(dataset, subject_idx, emg_idx, mode='fit',
                device='cpu', budget=150, n_reps=30,
                kappa_schedule: float = 0.0,
                acq_fn: str = 'ucb', ts_temperature: float = 1.0):
    """GP baseline evaluation, unified with the run_bo_loop pipeline.

    For ``mode='optimization'``: delegates to :func:`evaluate_optimization`
    with a :class:`GPSurrogate` so the BO loop, kappa schedule (auto
    cosine-annealed GP-UCB bounds scaled by input dimensionality), and
    snapshot logic are identical to ``finetuned_optimization()`` and the
    vanilla benchmark.

    For ``mode='fit'``: legacy flat-fit evaluation — no longer invoked by
    any active experiment mode (fit axis removed 2026-04-27).  Retained
    so existing code that references it does not break.

    Args:
        dataset: Dataset identifier, e.g. ``'nhp'`` or ``'rat'``.
        subject_idx: Subject index.
        emg_idx: EMG channel index.
        mode: ``'optimization'`` (active) or ``'fit'`` (legacy).
        device: PyTorch device string (``'cpu'`` or ``'cuda'``).
        budget: Total BO query budget (optimization) or training points (fit).
        n_reps: Independent repetitions.
        kappa_schedule: UCB exploration coefficient passed to ``evaluate_optimization``.
            ``0.0`` (default) → auto cosine-annealed schedule (GP-UCB theory scaling).
            Any other value → fixed kappa throughout the BO loop.

    Returns:
        Dict with keys matching :func:`finetuned_optimization` /
        :func:`evaluate_optimization`: ``'model_type'``, ``'times'``,
        ``'values'``, ``'y_test'``, ``'r2'``, ``'y_pred'``, ``'dataset'``,
        ``'subject'``, ``'emg'``, ``'snapshots'``, ``'ch2xy'``, ``'grid_shape'``.

    Raises:
        ValueError: If ``mode`` is not ``'optimization'`` or ``'fit'``.
    """
    if mode == 'optimization':
        # Delegate entirely to the canonical unified pipeline.
        # GPSurrogate.predict_ucb() = mean + kappa*std (Gaussian posterior).
        # Auto kappa bounds: _auto_kappa_max / _auto_kappa_min scale with
        # sqrt(d * log(n_steps)), matching GP-UCB theory.
        result = evaluate_optimization(
            surrogate=GPSurrogate(device=device),
            dataset_type=dataset,
            subject_idx=subject_idx,
            emg_idx=emg_idx,
            device=device,
            budget=budget,
            n_reps=n_reps,
            kappa_schedule=kappa_schedule,
            normalization='gp',
            acq_fn=acq_fn,
            ts_temperature=ts_temperature,
        )
        result['model_type'] = 'gp'
        return result

    elif mode == 'fit':
        data = load_data(dataset, subject_idx)
        X_train_full, y_train_full, X_test, y_test, scaler_y = preprocess_neural_data(
            data, emg_idx, 'gp'
        )

        X_flat, y_flat = _flatten_valid_pool(X_train_full, y_train_full)
        if budget > len(y_flat):
            raise RuntimeError(
                f"gp_baseline(fit): budget={budget} exceeds valid trial count "
                f"{len(y_flat)} for subject={subject_idx}, emg={emg_idx}."
            )

        r2_scores = []
        y_preds_all = []
        total_time = 0

        for _ in range(n_reps):
            indices = np.random.choice(len(y_flat), budget, replace=False)
            train_x = torch.tensor(X_flat[indices], dtype=torch.float32, device=device)
            train_y = torch.tensor(y_flat[indices], dtype=torch.float32, device=device)

            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            gp_model = ExactGP(train_x, train_y, likelihood).to(device)

            start = time.time()
            gp_model.train(); likelihood.train()
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
            for _ in range(50):
                optimizer.zero_grad()
                loss = -mll(gp_model(train_x), train_y)
                loss.backward()
                optimizer.step()

            gp_model.eval(); likelihood.eval()
            with torch.no_grad():
                posterior = likelihood(gp_model(
                    torch.tensor(X_test, dtype=torch.float32, device=device)
                ))
                y_pred = posterior.mean.cpu().numpy()
            total_time += time.time() - start

            r2_scores.append(float(r2_score(y_test, y_pred)))
            y_preds_all.append(y_pred)

        return {
            'model_type': 'gp',
            'r2': r2_scores,
            'times': total_time / n_reps,
            'y_test': y_test,
            'y_pred': np.mean(np.array(y_preds_all), axis=0),
            'dataset': dataset,
            'subject': subject_idx,
            'emg': emg_idx,
            'ch2xy': data['ch2xy'][_valid_site_mask(data, emg_idx)],
            'grid_shape': data.get('grid_shape'),
        }

    else:
        raise ValueError(
            f"gp_baseline: unknown mode={mode!r}. Use 'optimization' or 'fit'."
        )


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

    # Drop NaN-masked invalid trials before flat-sampling.
    X_flat, y_flat = _flatten_valid_pool(X_train_full, y_train_full)
    if budget > len(y_flat):
        raise RuntimeError(
            f"finetuned_fit: budget={budget} exceeds valid trial count "
            f"{len(y_flat)} for subject={subject_idx}, emg={emg_idx}."
        )

    r2_scores = []
    y_preds_all = []
    total_time = 0

    for i in range(n_reps):

        indices = np.random.choice(len(y_flat), budget, replace=False)
        X_train = X_flat[indices]
        y_train = y_flat[indices]

        start = time.time()

        # .fit() provides in-context examples for prediction
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        total_time += (time.time() - start)

        y_pred = np.asarray(y_pred)
        r2 = r2_score(y_test, y_pred)            # both standardized
        r2_scores.append(float(r2))
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
        'emg': emg_idx,
        'ch2xy': data['ch2xy'][_valid_site_mask(data, emg_idx)],
        'grid_shape': data.get('grid_shape'),
    }


def finetuned_optimization(dataset, subject_idx, emg_idx, model,
                            device='cpu', budget=100, n_reps=20,
                            kappa_schedule: float = 0.0,
                            acq_fn: str = 'ucb', ts_temperature: float = 1.0):
    """Evaluate optimization performance using a finetuned TabPFN model.

    Uses ``run_bo_loop()`` via ``TabPFNSurrogate`` so that the native
    bar-distribution UCB (``criterion.ucb``), the logit cache, and the
    auto-scaled kappa bounds are identical to the vanilla benchmark path.

    Args:
        dataset: Dataset identifier, e.g. ``'nhp'`` or ``'rat'``.
        subject_idx: Index of the held-out subject.
        emg_idx: Index of the EMG channel.
        model: A ``TabPFNRegressor`` returned by ``extract_inference_model()``.
        device: PyTorch device string (unused — ``TabPFNSurrogate`` owns device).
        budget: Total BO query budget (including initial random points).
        n_reps: Number of independent BO repetitions.
        kappa_schedule: UCB exploration coefficient passed to ``run_bo_loop``.
            ``0.0`` (default) → auto cosine-annealed schedule using GP-UCB
            theory scaling (same bounds as ``evaluate_optimization``).
            Any other value → fixed kappa throughout the BO loop.

    Returns:
        Dict with keys matching ``gp_baseline`` / ``evaluate_optimization``:
        ``'model_type'``, ``'times'``, ``'values'``, ``'y_test'``, ``'r2'``,
        ``'y_pred'``, ``'dataset'``, ``'subject'``, ``'emg'``, ``'snapshots'``,
        ``'ch2xy'``, ``'grid_shape'``.
    """
    data = load_data(dataset, subject_idx)

    X_pool, y_pool, X_test, y_test, scaler_y = preprocess_neural_data(
        data, emg_idx, 'pfn'
    )
    # X_pool: [N, D], y_pool: [N, n_reps], X_test: [M, D], y_test: [M]

    # n_estimators=1 for BO speed: one forward pass per acquisition step
    # instead of 8, without sacrificing UCB quality.  Deep-copy so the
    # caller's model (n_estimators=8) is not mutated.
    bo_model = copy.deepcopy(model)
    bo_model.n_estimators = 1
    bo_surrogate = TabPFNSurrogate(model=bo_model)

    n_init = max(3, int(0.05 * budget))
    snap_iters = _snapshot_iters(budget, n_init)
    snapshot_rep = np.random.randint(n_reps)
    collected_snapshots: dict | None = None

    mean_times_all: list[list[float]] = []
    values_all: list[list[float]] = []
    r2_scores: list[float] = []
    y_preds_all: list[np.ndarray] = []
    perf_explore_all: list[list[float]] = []

    for i in range(n_reps):
        loop_result = run_bo_loop(
            model=bo_surrogate,
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            n_init=n_init,
            budget=budget,
            kappa_schedule=kappa_schedule,
            snapshot_iters=snap_iters if i == snapshot_rep else None,
            acq_fn=acq_fn,
            ts_temperature=ts_temperature,
        )

        if loop_result['snapshots'] is not None:
            collected_snapshots = loop_result['snapshots']

        mean_times_all.append(loop_result['times'])
        values_all.append(loop_result['real_values'])
        if loop_result.get('perf_explore'):
            perf_explore_all.append(loop_result['perf_explore'])

        y_pred = np.asarray(loop_result['y_pred'])  # [M]
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(float(r2))
        y_preds_all.append(y_pred)

    mean_times = np.mean(np.array(mean_times_all), axis=0)  # [budget - n_init]
    y_pred_mean = np.mean(np.array(y_preds_all), axis=0)    # [M]
    perf_explore_arr = np.array(perf_explore_all) if perf_explore_all else None  # [n_reps, n_steps]

    snapshot_results: dict | None = None
    if collected_snapshots is not None:
        snapshot_results = {}
        for it, s_data in collected_snapshots.items():
            s_pred = np.asarray(s_data['y_pred']).ravel()
            s_r2 = float(r2_score(y_test, s_pred))
            snapshot_results[it] = {
                'y_pred': s_pred,
                'r2': s_r2,
                'best_pred_val': float(s_data['best_pred_val']),
            }

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
        'perf_explore': perf_explore_arr,        # [n_reps, n_steps] or None
        'n_init': n_init,
        'ch2xy': data['ch2xy'][_valid_site_mask(data, emg_idx)],
        'grid_shape': data.get('grid_shape'),
    }


def evaluate_optimization(
    surrogate: SurrogateModel,
    dataset_type: str,
    subject_idx: int,
    emg_idx: int,
    device: str = 'cpu',
    budget: int = 100,
    n_reps: int = 20,
    kappa_schedule: float = 0.0,
    normalization: str = 'pfn',
    acq_fn: str = 'ucb',
    ts_temperature: float = 1.0,
    capture_all_snapshots: bool = False,
    snapshot_iters_override: Optional[List[int]] = None,
) -> dict:
    """Evaluate Bayesian optimisation performance using any surrogate model.

    This is the unified replacement for ``gp_baseline(mode='optimization')``
    and ``finetuned_optimization()``.  It accepts any object conforming to
    the ``SurrogateModel`` protocol (``GPSurrogate``, ``TabPFNSurrogate``, or
    any custom wrapper) and delegates to ``run_bo_loop()``.

    A single-estimator copy of the surrogate (if it is a ``TabPFNSurrogate``)
    is used for the BO loop to avoid ensemble-averaging overhead during
    acquisition; the original model is used for the final prediction.

    Args:
        surrogate: Any object conforming to the ``SurrogateModel`` protocol.
            Use ``GPSurrogate(device=device)`` for GP or
            ``TabPFNSurrogate(model)`` for TabPFN variants.
        dataset_type: Dataset identifier, e.g. ``'rat'`` or ``'nhp'``.
        subject_idx: Index of the held-out subject to evaluate.
        emg_idx: Index of the EMG channel to evaluate.
        device: PyTorch device string, ``'cpu'`` or ``'cuda'``.
        budget: Total number of BO queries (including initial random points).
        n_reps: Number of independent BO repetitions.
        kappa_schedule: UCB coefficient control passed to ``run_bo_loop``.
            ``0.0`` → auto cosine-annealed schedule (GP-UCB theory scaling).
            Any other value → fixed kappa throughout the BO loop.
        normalization: Preprocessing scheme passed to ``preprocess_neural_data``;
            use ``'pfn'`` for TabPFN surrogates and ``'gp'`` for GP surrogates.
        acq_fn: Acquisition function.  ``'ucb'`` (default) or ``'ts'`` (Thompson
            Sampling).  Passed through to ``run_bo_loop``.
        ts_temperature: Temperature for TabPFN Thompson Sampling bar-distribution
            sampling.  Only used when ``acq_fn='ts'``.
        capture_all_snapshots: When ``True``, collect prediction snapshots for
            every rep (not just one random rep).  Enables per-rep R² at each
            sub-budget for the budget-sweep inference path.  Aggregated into
            ``result['r2_by_snapshot']``.
        snapshot_iters_override: When provided, replaces the auto-computed
            log2-spaced ``snap_iters`` with this explicit list.  Used by the
            budget-sweep refactor to force snapshots at the exact budget values.

    Returns:
        Dictionary with keys matching ``gp_baseline`` / ``finetuned_optimization``:
          - ``'model_type'``: str — class name of the surrogate
          - ``'times'``: np.ndarray shape [budget - n_init] — mean per-step times
          - ``'values'``: list[list[float]] — observed real values per rep [n_reps, budget]
          - ``'y_test'``: np.ndarray — ground-truth test responses
          - ``'r2'``: list[float] — per-rep R² on ``X_test`` at final budget
          - ``'y_pred'``: np.ndarray — mean final prediction across reps
          - ``'dataset'``: str — ``dataset_type``
          - ``'subject'``: int — ``subject_idx``
          - ``'emg'``: int — ``emg_idx``
          - ``'snapshots'``: dict | None — R²+predictions at log-spaced budgets
          - ``'r2_by_snapshot'``: dict[int, list[float]] | None — per-rep R² at
            each snapshot iter; populated only when ``capture_all_snapshots=True``

    Raises:
        RuntimeError: If NaN/Inf values appear in predictions.
    """
    data = load_data(dataset_type, subject_idx)

    X_pool, y_pool, X_test, y_test, scaler_y = preprocess_neural_data(
        data, emg_idx, normalization
    )
    # X_pool: [N, D], y_pool: [N, n_stims], X_test: [M, D], y_test: [M]

    n_init = max(3, int(0.05 * budget))
    snap_iters = (
        snapshot_iters_override
        if snapshot_iters_override is not None
        else _snapshot_iters(budget, n_init)
    )
    snapshot_rep = np.random.randint(n_reps)

    mean_times_all: list[list[float]] = []
    values_all: list[list[float]] = []
    r2_scores: list[float] = []
    y_preds_all: list[np.ndarray] = []
    perf_explore_all: list[list[float]] = []
    collected_snapshots: dict | None = None
    r2_by_snapshot: dict[int, list[float]] = {}

    for i in range(n_reps):
        use_snaps = capture_all_snapshots or (i == snapshot_rep)
        loop_result = run_bo_loop(
            model=surrogate,
            X_pool=X_pool,
            y_pool=y_pool,
            X_test=X_test,
            y_test=y_test,
            n_init=n_init,
            budget=budget,
            kappa_schedule=kappa_schedule,
            snapshot_iters=snap_iters if use_snaps else None,
            acq_fn=acq_fn,
            ts_temperature=ts_temperature,
        )

        if loop_result['snapshots'] is not None:
            collected_snapshots = loop_result['snapshots']
            if capture_all_snapshots:
                for it, s_data in loop_result['snapshots'].items():
                    s_pred = np.asarray(s_data['y_pred']).ravel()  # [M]
                    r2_by_snapshot.setdefault(it, []).append(
                        float(r2_score(y_test, s_pred))
                    )

        mean_times_all.append(loop_result['times'])
        values_all.append(loop_result['real_values'])
        if loop_result.get('perf_explore'):
            perf_explore_all.append(loop_result['perf_explore'])

        y_pred = np.asarray(loop_result['y_pred'])  # [M] — standardized

        if np.isnan(y_pred).any():
            raise RuntimeError(
                f"evaluate_optimization: NaN in final predictions for "
                f"subject={subject_idx}, emg={emg_idx}, rep={i}."
            )

        r2 = r2_score(y_test, y_pred)            # both standardized
        r2_scores.append(float(r2))
        y_preds_all.append(y_pred)

    mean_times = np.mean(np.array(mean_times_all), axis=0)  # [budget - n_init]
    y_pred_mean = np.mean(np.array(y_preds_all), axis=0)    # [M]
    perf_explore_arr = np.array(perf_explore_all) if perf_explore_all else None  # [n_reps, n_steps]

    snapshot_results: dict | None = None
    if collected_snapshots is not None:
        snapshot_results = {}
        for it, snap_data in collected_snapshots.items():
            s_pred = np.asarray(snap_data['y_pred']).ravel()  # [M] — standardized
            s_r2 = float(r2_score(y_test, s_pred))
            snapshot_results[it] = {
                'y_pred': s_pred,
                'r2': s_r2,
                'best_pred_val': float(snap_data['best_pred_val']),
            }

    _valid_ch2xy = data['ch2xy'][_valid_site_mask(data, emg_idx)]
    _grid_shape = data.get('grid_shape')
    gfs_result: dict[float, float] | None = None
    if _grid_shape is not None:
        gfs_result = compute_gfs(y_pred_mean, y_test, _valid_ch2xy, _grid_shape)

    return {
        'model_type': type(surrogate).__name__,
        'times': mean_times,
        'values': values_all,
        'y_test': y_test,
        'r2': r2_scores,
        'y_pred': y_pred_mean,
        'gfs': gfs_result,                        # dict[sigma→GFS] or None (A5)
        'dataset': dataset_type,
        'subject': subject_idx,
        'emg': emg_idx,
        'snapshots': snapshot_results,
        'perf_explore': perf_explore_arr,         # [n_reps, n_steps] or None
        'r2_by_snapshot': r2_by_snapshot or None, # dict[int, list[float]] or None
        'n_init': n_init,
        'ch2xy': _valid_ch2xy,
        'grid_shape': _grid_shape,
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
                        'R2': float(score),
                        'ID': f"{res_ft['subject']}_{res_ft['emg']}"
                    })

                res_gp = gp_baseline(dataset_type, subj_idx, emg_idx, mode='fit',
                                     device=device, budget=b, n_reps=15)
                for score in res_gp['r2']:
                    plot_data.append({
                        'Budget': b,
                        'Model': 'GP',
                        'R2': float(score),
                        'ID': f"{res_gp['subject']}_{res_gp['emg']}"
                    })

    df = pd.DataFrame(plot_data)
    if visualize:
        budget_sweep_plot(df, eval_type='fit', dataset=dataset_type,
                          split_type=split_type, save=True, output_dir=output_dir)

    return df


def _unpack_budget_trajectory(
    plot_data: List[dict],
    result: dict,
    model_name: str,
    budgets: List[int],
    subj_idx: int,
    emg_idx: int,
) -> None:
    """Extract per-budget regret and R² from a single max-budget BO trajectory.

    Reads best-so-far regret at each sub-budget from the full trajectory in
    ``result['values']``, and per-rep R² from ``result['r2_by_snapshot']``
    (populated when ``capture_all_snapshots=True``).

    Valid because UCB and TS are myopic — each acquisition step is independently
    optimal given the current posterior, so early steps in a max-budget run are
    equivalent to re-running with the shorter budget.

    Args:
        plot_data: Accumulator list to append rows to.
        result: Return dict from :func:`evaluate_optimization` called with
            ``capture_all_snapshots=True`` and ``snapshot_iters_override=budgets``.
        model_name: Display name, e.g. ``'TabPFN'`` or ``'GP'``.
        budgets: Sub-budget values to extract (must match ``snapshot_iters_override``).
        subj_idx: Subject index.
        emg_idx: EMG channel index.
    """
    optimal = float(result['y_test'].max())
    raw = np.array(result['values'])                         # [n_reps, max_budget]
    best_so_far = np.maximum.accumulate(raw, axis=1)         # [n_reps, max_budget]
    r2_by_snap: dict = result.get('r2_by_snapshot') or {}
    n_reps = raw.shape[0]

    for b in budgets:
        final_best = best_so_far[:, b - 1]                  # [n_reps]
        final_regret = optimal - final_best                  # [n_reps]
        r2_at_b: list = r2_by_snap.get(b, [float('nan')] * n_reps)

        for regret_val, r2_val in zip(final_regret, r2_at_b):
            plot_data.append({
                'Budget': b,
                'Model': model_name,
                'Regret': float(regret_val),
                'R2': float(r2_val),
                'ID': f'{subj_idx}_{emg_idx}',
            })


def finetuned_optimization_budget(dataset_type, model, regret_metric='abs',
                                   device='cpu', budgets=[10, 50, 100, 150, 200],
                                   test_subjects=None, test_emg_indices=None,
                                   split_type='', output_dir=None,
                                   kappa_schedule: float = 0.0,
                                   acq_fn: str = 'ucb',
                                   ts_temperature: float = 1.0):
    """Run optimization budget sweep using a single max-budget trajectory.

    Calls :func:`evaluate_optimization` once per (subject, emg) pair at
    ``budget=max(budgets)`` with ``capture_all_snapshots=True``.  Sub-budget
    performance is inferred from the stored trajectory via
    :func:`_unpack_budget_trajectory`.  This is valid because UCB and TS are
    myopic — each step is independently optimal given the current posterior.

    Speedup vs the legacy per-budget outer loop: O(len(budgets)) → O(1) BO
    runs per (subject, emg) pair.

    Args:
        dataset_type: ``'rat'`` or ``'nhp'``.
        model: A ``TabPFNRegressor`` returned by ``extract_inference_model()``.
        regret_metric: ``'abs'`` (final simple regret, default).  ``'cum'``
            (mean simple regret across all steps) is not yet implemented in
            the trajectory-inference path.
        device: PyTorch device string.
        budgets: List of budget values to sweep.  ``max(budgets)`` is used as
            the single BO run budget.
        test_subjects: Subject indices to evaluate.  Defaults to
            ``HELD_OUT_SUBJECTS[dataset_type]``.
        test_emg_indices: EMG channel indices per subject.  Defaults to all.
        split_type: Label string passed to ``budget_sweep_plot``.
        output_dir: Output directory for the SVG plot.
        kappa_schedule: UCB kappa schedule.  ``0.0`` → auto cosine-annealed.
        acq_fn: Acquisition function (``'ucb'`` or ``'ts'``).
        ts_temperature: Temperature for TabPFN bar-distribution TS sampling.

    Returns:
        :class:`pandas.DataFrame` with columns ``Budget``, ``Model``,
        ``Regret``, ``R2``, ``ID``.
    """
    if test_subjects is None:
        test_subjects = HELD_OUT_SUBJECTS[dataset_type]

    max_b = max(budgets)
    plot_data: list[dict] = []

    print(f"Starting Finetuned TabPFN Optimization Budget Sweep ({regret_metric}): {budgets}")
    print(f"  Single max-budget run ({max_b} steps) with trajectory inference.")

    for subj_idx in test_subjects:
        data = load_data(dataset_type, subj_idx)
        n_emgs = data['sorted_respMean'].shape[1]

        emg_range = test_emg_indices if test_emg_indices is not None else range(n_emgs)
        for emg_idx in emg_range:
            if emg_idx >= n_emgs:
                continue

            print(f"  > subject={subj_idx}, emg={emg_idx} (budget={max_b})...")

            # --- Finetuned TabPFN: single run at max budget with snapshots ---
            # Deep-copy per (subj, emg) to avoid state sharing across the loop.
            # n_estimators=1 for BO acquisition speed (no ensemble averaging needed).
            tabpfn_bo_model = copy.deepcopy(model)
            tabpfn_bo_model.n_estimators = 1
            tabpfn_surrogate = TabPFNSurrogate(model=tabpfn_bo_model)

            res_ft = evaluate_optimization(
                surrogate=tabpfn_surrogate,
                dataset_type=dataset_type,
                subject_idx=subj_idx,
                emg_idx=emg_idx,
                device=device,
                budget=max_b,
                n_reps=20,
                kappa_schedule=kappa_schedule,
                normalization='pfn',
                acq_fn=acq_fn,
                ts_temperature=ts_temperature,
                capture_all_snapshots=True,
                snapshot_iters_override=budgets,
            )
            _unpack_budget_trajectory(plot_data, res_ft, 'TabPFN', budgets, subj_idx, emg_idx)

            # --- GP: single run at max budget with snapshots ---
            res_gp = evaluate_optimization(
                surrogate=GPSurrogate(device=device),
                dataset_type=dataset_type,
                subject_idx=subj_idx,
                emg_idx=emg_idx,
                device=device,
                budget=max_b,
                n_reps=20,
                kappa_schedule=kappa_schedule,
                normalization='gp',
                acq_fn=acq_fn,
                ts_temperature=ts_temperature,
                capture_all_snapshots=True,
                snapshot_iters_override=budgets,
            )
            _unpack_budget_trajectory(plot_data, res_gp, 'GP', budgets, subj_idx, emg_idx)

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
    epochs=100,
    lr=1e-4,
    aug_pct_list=None,
    held_out_emg_idx=None,
    held_out_subj_idx=None,
    subjects_mode: str = 'held_out',
    save=False,
    silence_diagnostics=True,
    kappa_schedule: float = 0.0,
    aug_transforms: tuple | None = None,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_target: str = 'decoder_dict',
    grad_clip: float | None = None,
    n_estimators_finetune: int = 8,
):
    """
    Ablation study: evaluate BO performance (R² + regret) across augmentation percentages.

    Compares vanilla TabPFN (aug_pct=0, no finetuning) against finetuned TabPFN
    for each value in aug_pct_list.  100% = 10 augmented maps per EMG
    (``aug_pct=1.0``); values scale linearly via
    ``n_aug_per_emg = max(1, round(aug_pct * 10))``.

    Subject-fold control (``subjects_mode`` / ``held_out_subj_idx``):

    - ``subjects_mode='held_out'`` (default): test on ``HELD_OUT_SUBJECTS``,
      train on ``TRAIN_SUBJECTS`` — the standard single-fold evaluation.
    - ``subjects_mode='all'``: LOO cross-validation over all subjects in
      ``ALL_SUBJECTS``.  For each fold the held-out subject is the sole test
      subject and all remaining subjects form the training set.  A separate
      finetuned model is trained per fold because the training set differs.
    - ``held_out_subj_idx`` given (any ``subjects_mode``): restricts to a
      single explicit fold with that subject as the sole test subject.

    Args:
        dataset_type: 'rat' or 'nhp'
        split_type: 'inter_subject' or 'intra_emg'
        mode: 'optimization'
        device: 'cpu' or 'cuda'
        budget: BO query budget
        n_reps: repetitions per experiment
        epochs: finetuning epochs
        lr: finetuning learning rate
        aug_pct_list: list of float percentages to sweep; default [0.1, 0.2, 0.5, 0.7, 1.0, 2.5]
        held_out_emg_idx: required when split_type='intra_emg'
        held_out_subj_idx: optional override restricting evaluation to one subject
        subjects_mode: 'held_out' (default) or 'all' (LOO cross-validation)
        save: if True, save plot and DataFrame to disk
        silence_diagnostics: if True (default), skip gradient/CKA monitoring
        aug_transforms: tuple of transform names to sample from per augmented map;
            None defaults to ('none',) — noise only, no spatial transforms
        use_lora: if True, use LoRA parameter-efficient finetuning at each sweep point.
        lora_rank: rank of LoRA low-rank matrices (default 8).
        lora_alpha: LoRA scaling factor (default 16).
        lora_target: which layers to apply LoRA to (default 'decoder_dict').

    Returns:
        DataFrame with columns: aug_pct, R2, Regret, ID, Subject
    """
    if aug_pct_list is None:
        aug_pct_list = [0.1, 0.2, 0.5, 0.7, 1.0, 2.5]

    for v in aug_pct_list:
        if v < 0:
            raise ValueError(f"aug_pct_list values must be >= 0, got {v}")

    if subjects_mode not in ('held_out', 'all'):
        raise ValueError(f"subjects_mode must be 'held_out' or 'all', got {subjects_mode!r}")

    if split_type == 'intra_emg' and held_out_emg_idx is None:
        raise ValueError("held_out_emg_idx must be set when split_type='intra_emg'")

    if split_type not in ('inter_subject', 'intra_emg'):
        raise ValueError(f"Unknown split_type={split_type!r}. Use 'inter_subject' or 'intra_emg'.")

    # --- Determine which subjects to loop over as the held-out fold ---
    # subjects_mode='all'  → LOO over every subject in ALL_SUBJECTS
    # held_out_subj_idx    → single explicit fold
    # default              → [None] sentinel (use HELD_OUT/TRAIN_SUBJECTS constants)
    if subjects_mode == 'all':
        loo_subjects: list = list(ALL_SUBJECTS[dataset_type])
    elif held_out_subj_idx is not None:
        loo_subjects = [held_out_subj_idx]
    else:
        loo_subjects = [None]

    # --- Run-level tag (shared across all folds, created once) ---
    _run_tag_parts = [split_type]
    if subjects_mode == 'all':
        _run_tag_parts.append('all_subjects')
    elif held_out_subj_idx is not None:
        _run_tag_parts.append(f'subj{held_out_subj_idx}')
    if held_out_emg_idx is not None:
        _run_tag_parts.append(f'emg{held_out_emg_idx}')
    _run_exp_tag = '_'.join(_run_tag_parts)

    _aug_family = 'lora-aug-sweep-opt' if use_lora else 'aug-sweep-optimization'
    _tag_config: dict[str, Any] = {
        'dataset': dataset_type,
        'split_type': split_type,
        'subjects_mode': subjects_mode,
        'mode': mode,
        'epochs': epochs,
        'lr': lr,
        'budget': budget,
        'n_reps': n_reps,
        'aug_pct_list': sorted(aug_pct_list),
        'kappa_schedule': kappa_schedule,
    }
    if use_lora:
        _tag_config.update({
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'lora_target': lora_target,
        })
    if subjects_mode != 'all' and held_out_subj_idx is not None:
        _tag_config['held_out_subj_idx'] = held_out_subj_idx
    if held_out_emg_idx is not None:
        _tag_config['held_out_emg_idx'] = held_out_emg_idx
    aug_sweep_tag = generate_experiment_tag(dataset_type, _aug_family, _tag_config)

    # --- Create output directory once (all folds share the same run_dir) ---
    if save:
        run_dir = create_run_dir(aug_sweep_tag, tag=aug_sweep_tag)
        _run_cfg: dict = {
            'run_type': 'finetuned_percentage',
            'experiment_tag': aug_sweep_tag,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'dataset_type': dataset_type,
            'split_type': split_type,
            'subjects_mode': subjects_mode,
            'loo_subjects': loo_subjects if subjects_mode == 'all' else None,
            'mode': mode,
            'device': device,
            'budget': budget,
            'n_reps': n_reps,
            'epochs': epochs,
            'lr': lr,
            'aug_pct_list': aug_pct_list,
            'held_out_emg_idx': held_out_emg_idx,
            'held_out_subj_idx': held_out_subj_idx,
        }
        if use_lora:
            _run_cfg.update({
                'use_lora': True,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'lora_target': lora_target,
            })
        write_run_config(run_dir, _run_cfg)
    else:
        run_dir = None

    plot_data: list = []

    def _accumulate(res):
        """Append per-rep rows from a result dict into plot_data."""
        if mode == 'optimization':
            optimal = res['y_test'].max()
            raw = np.array(res['values'])
            best = np.maximum.accumulate(raw, axis=1)
            regrets = optimal - best[:, -1]
            for r2, reg in zip(res['r2'], regrets):
                plot_data.append({
                    'n_aug':   res['aug_pct'],
                    'R2':      float(r2),
                    'Regret':  float(reg),
                    'ID':      f"{res['subject']}_{res['emg']}",
                    'Subject': res['subject'],
                })
        else:
            for r2 in res['r2']:
                plot_data.append({
                    'n_aug':   res['aug_pct'],
                    'R2':      float(r2),
                    'ID':      f"{res['subject']}_{res['emg']}",
                    'Subject': res['subject'],
                })

    # ================================================================
    # Outer LOO fold loop.
    # subjects_mode='held_out' → single iteration with _loo_subj=None.
    # subjects_mode='all'      → one iteration per subject; each fold
    #                            trains a fresh model on the remaining
    #                            subjects (train set differs per fold).
    # ================================================================
    for _loo_subj in loo_subjects:

        # --- Per-fold train / test resolution ---
        if _loo_subj is not None:
            # Both 'all' and explicit held_out_subj_idx reach here.
            train_subject_indices = [s for s in ALL_SUBJECTS[dataset_type] if s != _loo_subj]
            test_subjects = [_loo_subj]
        else:
            if split_type == 'inter_subject':
                train_subject_indices = TRAIN_SUBJECTS[dataset_type]
                test_subjects = HELD_OUT_SUBJECTS[dataset_type]
            else:  # intra_emg
                train_subject_indices = ALL_SUBJECTS[dataset_type]
                test_subjects = ALL_SUBJECTS[dataset_type]

        if split_type == 'inter_subject':
            test_emg_indices = None
            ft_held_out_emg = None
        else:
            test_emg_indices = [held_out_emg_idx]
            ft_held_out_emg = held_out_emg_idx

        # Per-fold visualization tag used in plot/emg-map filenames within run_dir
        _fold_parts = [split_type]
        if _loo_subj is not None:
            _fold_parts.append(f'subj{_loo_subj}')
        if held_out_emg_idx is not None:
            _fold_parts.append(f'emg{held_out_emg_idx}')
        exp_tag = '_'.join(_fold_parts)

        # Build (subj_idx, emg_idx) experiment pairs for this fold
        experiments = []
        for subj_idx in test_subjects:
            data = load_data(dataset_type, subj_idx)
            n_emgs = data['sorted_respMean'].shape[1]
            emgs = test_emg_indices if test_emg_indices is not None else range(n_emgs)
            for emg_idx in emgs:
                if emg_idx < n_emgs:
                    experiments.append((subj_idx, emg_idx))

        _fold_label = f' | fold=subj{_loo_subj}' if _loo_subj is not None else ''

        # --- Phase 1: Vanilla TabPFN baseline (aug_pct = 0) ---
        print("=" * 60)
        print(f"[Aug Sweep] Vanilla TabPFN baseline (aug_pct=0) | {dataset_type} | {split_type} | mode={mode}{_fold_label}")
        print("=" * 60)
        vanilla_model = TabPFNRegressor(device=device)
        _vanilla_cache_params = {
            'model': 'vanilla_tabpfn',
            'budget': budget,
            'n_reps': n_reps,
            'kappa_schedule': kappa_schedule,
            'normalization': 'pfn',
        }
        vanilla_results = []
        for subj_idx, emg_idx in experiments:
            print(f"  Vanilla: subject={subj_idx}, emg={emg_idx}")
            if mode == 'optimization':
                res = load_subject_result(
                    dataset_type, subj_idx, emg_idx, 'vanilla_tabpfn', _vanilla_cache_params
                )
                if res is not None:
                    print(f"    [CACHE HIT] vanilla_tabpfn subject={subj_idx}, emg={emg_idx}")
                else:
                    res = finetuned_optimization(dataset_type, subj_idx, emg_idx, vanilla_model,
                                                 device=device, budget=budget, n_reps=n_reps,
                                                 kappa_schedule=kappa_schedule)
                    save_subject_result(
                        res, dataset_type, subj_idx, emg_idx, 'vanilla_tabpfn', _vanilla_cache_params
                    )
            else:
                res = finetuned_fit(dataset_type, subj_idx, emg_idx, vanilla_model,
                                    device=device, budget=budget, n_reps=n_reps)
            res['aug_pct'] = 0
            _accumulate(res)
            if mode == 'optimization':
                vanilla_results.append(res)
        if vanilla_results and run_dir:
            visualize_representation({'TabPFN': vanilla_results},
                                     mode=f'_{exp_tag}_aug0pct', save=True, output_dir=run_dir)
            for _res in vanilla_results:
                show_emg_map(_res, model_type='TabPFN', mode=f'_{exp_tag}_aug0pct',
                             save=True, output_dir=run_dir)

        # --- Phase 2: Finetuned TabPFN sweep ---
        for aug_pct in aug_pct_list:
            print("=" * 60)
            print(f"[Aug Sweep] aug_pct={aug_pct} | {dataset_type} | {split_type}{_fold_label}")
            print("=" * 60)

            aug_label = f'{int(round(aug_pct * 100))}pct'

            # Cache key includes the fold subject so LOO folds never share
            # cached results (different train sets → different finetuned models).
            # 'lora_tabpfn' vs 'finetuned_tabpfn' prevents LoRA and full-FT
            # sweep results from sharing the same cache slot.
            _aug_model_type = 'lora_tabpfn' if use_lora else 'finetuned_tabpfn'
            _ft_aug_cache_params: dict[str, Any] = {
                'model': _aug_model_type,
                'aug_pct': aug_pct,
                'epochs': epochs,
                'lr': lr,
                'budget': budget,
                'n_reps': n_reps,
                'kappa_schedule': kappa_schedule,
                'normalization': 'pfn',
                'split_type': split_type,
            }
            if use_lora:
                _ft_aug_cache_params.update({
                    'lora_rank': lora_rank,
                    'lora_alpha': lora_alpha,
                    'lora_target': lora_target,
                })
            if held_out_emg_idx is not None:
                _ft_aug_cache_params['held_out_emg_idx'] = held_out_emg_idx
            if _loo_subj is not None:
                _ft_aug_cache_params['held_out_subj_idx'] = _loo_subj

            # Pre-check cache to skip expensive finetuning when all results exist.
            _cached: dict = {}
            if mode == 'optimization':
                for subj_idx, emg_idx in experiments:
                    _hit = load_subject_result(
                        dataset_type, subj_idx, emg_idx,
                        _aug_model_type, _ft_aug_cache_params,
                    )
                    if _hit is not None:
                        _cached[(subj_idx, emg_idx)] = _hit

            _needs_training = (mode != 'optimization') or (len(_cached) < len(experiments))

            if _needs_training:
                X_ft, y_ft = build_finetuning_dataset(
                    dataset_type,
                    subject_indices=train_subject_indices,
                    held_out_emg_idx=ft_held_out_emg,
                    aug_pct=aug_pct,
                    seed=42,
                    aug_transforms=aug_transforms,
                )
                print(f"  Dataset size: {X_ft.shape[0]} rows")

                ft_model_raw = _make_finetuned_regressor(
                    silence_diagnostics=silence_diagnostics,
                    use_lora=use_lora,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_target=lora_target,
                    device=device, epochs=epochs, learning_rate=lr,
                    grad_clip=grad_clip,
                    n_estimators_finetune=n_estimators_finetune,
                    n_estimators_validation=n_estimators_finetune,
                    n_estimators_final_inference=n_estimators_finetune,
                )
                ft_model_raw.fit(X_ft, y_ft)
                if hasattr(ft_model_raw, '_diagnostics_') and ft_model_raw._diagnostics_:
                    diag_dir = os.path.join(run_dir, 'diagnostics') if run_dir else None
                    plot_gradient_metrics(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
                    plot_weight_metrics(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
                    plot_cka_similarity(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
                ft_model = extract_inference_model(ft_model_raw)
            else:
                print(
                    f"  [ALL CACHED] Skipping training for aug_pct={aug_pct} "
                    f"({len(experiments)} results already cached)"
                )
                ft_model = None

            aug_results = []
            for subj_idx, emg_idx in experiments:
                print(f"  aug_pct={aug_pct}: subject={subj_idx}, emg={emg_idx}")
                if mode == 'optimization':
                    if (subj_idx, emg_idx) in _cached:
                        res = _cached[(subj_idx, emg_idx)]
                        print(f"    [CACHE HIT] finetuned_tabpfn subject={subj_idx}, emg={emg_idx}")
                    else:
                        res = finetuned_optimization(
                            dataset_type, subj_idx, emg_idx, ft_model,
                            device=device, budget=budget, n_reps=n_reps,
                            kappa_schedule=kappa_schedule,
                        )
                        save_subject_result(
                            res, dataset_type, subj_idx, emg_idx,
                            _aug_model_type, _ft_aug_cache_params,
                        )
                else:
                    res = finetuned_fit(dataset_type, subj_idx, emg_idx, ft_model,
                                        device=device, budget=budget, n_reps=n_reps)
                res['aug_pct'] = aug_pct
                _accumulate(res)
                if mode == 'optimization':
                    aug_results.append(res)
            if aug_results and run_dir:
                visualize_representation({'TabPFN': aug_results},
                                         mode=f'_{exp_tag}_aug{aug_label}',
                                         save=True, output_dir=run_dir)
                for _res in aug_results:
                    show_emg_map(_res, model_type='TabPFN', mode=f'_{exp_tag}_aug{aug_label}',
                                 save=True, output_dir=run_dir)

    # --- Phase 3: Visualize & save (aggregates all folds) ---
    df = pd.DataFrame(plot_data)
    augmentation_sweep_plot(df, dataset=dataset_type,
                            split_type=_run_exp_tag, save=save, output_dir=run_dir)

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
