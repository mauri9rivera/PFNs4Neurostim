"""
Core ID/OOD analysis: Shannon entropy, MMD, Wasserstein, Mahalanobis, CKA,
and gradient L2-norm.

Tests whether neurostim data lies within TabPFN's pre-training prior by
comparing against synthetic reference distributions (GP and/or Prior Bag).
"""
from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import procrustes as scipy_procrustes
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabpfn import TabPFNRegressor

from models.regressors import linear_cka

from utils.data_utils import load_data, ALL_SUBJECTS

from analysis.synthetic_gp import generate_synthetic_gp_bank
from analysis.synthetic_noise import generate_noise_bank
from analysis.synthetic_tabpfn_prior import generate_tabpfn_prior_bank

# Phase-aligned layer indices for TabPFN v2 (18-layer model).
# Maps the three-phase attention structure from Ye et al. (2025,
# arXiv:2502.17361) Section 5, scaled from 12-layer to 18-layer:
#   Early  (0-4):  label-token attention, attribute identity internalized
#   Middle (5-12): uniform mixing, cross-attribute information exchange
#   Deep   (13-17): selective attention on predictive attributes
#
# ID/OOD scalar metrics: phase boundaries capture WHERE representations diverge
ID_OOD_LAYERS = [4, 13, 17]
# Finetuning diagnostics: full phase coverage
DIAGNOSTIC_LAYERS = [0, 4, 9, 13, 17]
# Dense sweep for layer-wise CKA heatmap (Tier 2 experiment)
LAYERWISE_HEATMAP_LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]


def _layer_name(idx: int) -> str:
    """Convert layer index to module path for forward hooks."""
    return f'transformer_encoder.layers.{idx}'


def _normalize_for_tabpfn(X, y):
    """Normalize synthetic data to ranges TabPFN expects.

    TabPFN's internal safe_power_transformer overflows on features outside
    a moderate range.  Map X → [0, 1] via MinMaxScaler and re-standardize y
    so every synthetic dataset matches the neurostim preprocessing convention.

    Returns (X_norm, y_norm) as float32 arrays, or None if degenerate.
    """
    if X.shape[0] < 2:
        return None
    scaler_x = MinMaxScaler()
    X_norm = scaler_x.fit_transform(X)
    # Degenerate: constant feature column → NaN after MinMaxScaler
    if not np.all(np.isfinite(X_norm)):
        return None
    scaler_y = StandardScaler()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    if not np.all(np.isfinite(y_norm)):
        return None
    return X_norm.astype(np.float32), y_norm.astype(np.float32)


# ============================================================================
#  3a. Shannon Entropy of Bar-Distribution Outputs
# ============================================================================

def compute_bar_distribution_entropy(model, X_train, y_train, X_test):
    """Entropy of frozen TabPFN's predictive distribution per test sample.

    The bar distribution discretizes the output space into bins. For each
    test point, TabPFN outputs log-probabilities over these bins.
    Low entropy → confident (data looks familiar / in-distribution).
    High entropy → uncertain (data looks unfamiliar / out-of-distribution).

    Args:
        model: TabPFNRegressor (frozen, no finetuning)
        X_train: (n_train, d) context features
        y_train: (n_train,) context targets
        X_test: (n_test, d) query features

    Returns:
        (n_test,) numpy array of per-sample Shannon entropy values.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='overflow', category=RuntimeWarning)
        model.fit(X_train, y_train)
        result = model.predict(X_test, output_type='full')
    logits = result['logits']  # shape (n_test, n_bars), values are log(probs)

    # logits are log-probabilities (softmax already applied, then .log())
    # Recover probabilities via softmax (equivalent to exp since already normalized,
    # but softmax is numerically safer)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)

    # Shannon entropy: H = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.detach().cpu().numpy()


def entropy_analysis(dataset_types, device='cpu', n_context=500,
                     prior_source='tabpfn_prior', n_synthetic=500, seed=42):
    """Entropy across all datasets/subjects/EMGs + synthetic reference(s).

    For each subject-EMG pair:
      - Split channels: first n_context as context, rest as test
      - Compute per-sample entropy with frozen TabPFN

    Also computes entropy on synthetic data (same protocol) as baseline.

    Args:
        dataset_types: list of dataset names (e.g., ['rat', 'nhp'])
        device: 'cpu' or 'cuda'
        n_context: number of context points for TabPFN
        prior_source: 'gp' | 'tabpfn_prior' | 'both'
        n_synthetic: number of synthetic datasets for reference
        seed: random seed

    Returns:
        dict with keys per dataset + 'synthetic_gp' and/or 'synthetic_prior',
        each containing entropy arrays.
    """
    model = TabPFNRegressor(device=device)
    results = {}

    # --- Neurostim data ---
    rng_ctx = np.random.RandomState(seed)

    for dataset_type in dataset_types:
        subjects = ALL_SUBJECTS[dataset_type]
        ctx_results = {}   # in-context (n_ctx random samples as context)
        gt_results = {}    # ground truth (full map as context)

        for subj_idx in subjects:
            data = load_data(dataset_type, subj_idx)
            coords = data['ch2xy']                        # [n_channels, 2]
            scaler_x = MinMaxScaler()
            X = scaler_x.fit_transform(coords).astype(np.float32)  # [n_channels, 2]
            n_channels = X.shape[0]
            n_emgs = data['sorted_respMean'].shape[1]

            subj_ctx = {}
            subj_gt = {}
            for emg_idx in range(n_emgs):
                # Fit scaler on ALL reps (matching preprocess_neural_data)
                resp_all = data['sorted_resp'][:, emg_idx, :]
                scaler_y = StandardScaler()
                scaler_y.fit(resp_all.reshape(-1, 1))
                y_mean = data['sorted_respMean'][:, emg_idx]
                y = scaler_y.transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)              # [n_channels]

                # --- Ground truth: full map as context, predict on full map ---
                gt_entropy = compute_bar_distribution_entropy(
                    model, X, y, X,
                )
                subj_gt[emg_idx] = gt_entropy

                # --- In-context: n_ctx randomly sampled points as context,
                #     remaining points as test (mirrors BO initialisation) ---
                n_ctx = min(n_context, n_channels - 1)
                all_idx = rng_ctx.permutation(n_channels)
                ctx_idx = all_idx[:n_ctx]
                tst_idx = all_idx[n_ctx:]

                X_ctx = X[ctx_idx]                        # [n_ctx, 2]
                y_ctx = y[ctx_idx]                        # [n_ctx]
                X_tst = X[tst_idx]                        # [n_channels - n_ctx, 2]

                if len(X_tst) == 0:
                    continue

                ctx_entropy = compute_bar_distribution_entropy(
                    model, X_ctx, y_ctx, X_tst,
                )
                subj_ctx[emg_idx] = ctx_entropy

            ctx_results[subj_idx] = subj_ctx
            gt_results[subj_idx] = subj_gt

        results[dataset_type] = ctx_results
        results[f'{dataset_type}_gt'] = gt_results

    # --- Synthetic references ---
    rng = np.random.RandomState(seed)

    if prior_source in ('gp', 'both'):
        gp_bank = generate_synthetic_gp_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        gp_entropies = _entropy_from_bank(model, gp_bank, n_context)
        results['synthetic_gp'] = gp_entropies

    if prior_source in ('tabpfn_prior', 'both'):
        prior_bank = generate_tabpfn_prior_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        prior_entropies = _entropy_from_bank(model, prior_bank, n_context)
        results['synthetic_prior'] = prior_entropies

    # Noise baseline (always included)
    noise_bank = generate_noise_bank(
        n_datasets=n_synthetic, n_features=2,
    )
    results['noise'] = _entropy_from_bank(model, noise_bank, n_context)

    return results


def _entropy_from_bank(model, bank, n_context):
    """Compute entropy for a bank of synthetic datasets.

    Uses exactly n_context training points so that all conditions are
    evaluated under identical context size.  Datasets with fewer than
    n_context + 1 points are skipped (too small to provide full context).
    """
    all_entropies = []
    for X, y in bank:
        normed = _normalize_for_tabpfn(X, y)
        if normed is None:
            continue
        X, y = normed
        # Require exactly n_context context points + at least 1 test point
        if len(X) <= n_context:
            continue
        X_ctx, y_ctx = X[:n_context], y[:n_context]
        X_tst = X[n_context:]
        entropy = compute_bar_distribution_entropy(model, X_ctx, y_ctx, X_tst)
        all_entropies.append(entropy)
    return np.concatenate(all_entropies) if all_entropies else np.array([])


# ============================================================================
#  3b. MMD (Maximum Mean Discrepancy)
# ============================================================================

def rbf_kernel_matrix(X, Y, bandwidth):
    """RBF kernel matrix: K(x,y) = exp(-||x-y||^2 / (2*bandwidth^2))."""
    dists = cdist(X, Y, metric='sqeuclidean')
    return np.exp(-dists / (2.0 * bandwidth ** 2))


def median_bandwidth(X, Y):
    """Median heuristic: bandwidth = median of all pairwise distances."""
    dists = cdist(X, Y, metric='euclidean')
    med = np.median(dists[dists > 0])
    return max(med, 1e-8)


def compute_mmd_squared(X, Y, bandwidth):
    """Unbiased MMD^2 U-statistic (diagonal terms excluded).

    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    where expectations exclude i==j terms.
    """
    n, m = len(X), len(Y)
    Kxx = rbf_kernel_matrix(X, X, bandwidth)
    Kyy = rbf_kernel_matrix(Y, Y, bandwidth)
    Kxy = rbf_kernel_matrix(X, Y, bandwidth)

    # Exclude diagonal
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (Kxx.sum() / (n * (n - 1))
             - 2.0 * Kxy.sum() / (n * m)
             + Kyy.sum() / (m * (m - 1)))
    return mmd2


def mmd_permutation_test(X, Y, bandwidth=None, n_permutations=500):
    """MMD^2 with permutation p-value.

    Args:
        X, Y: (n, d) and (m, d) arrays
        bandwidth: RBF bandwidth (auto via median heuristic if None)
        n_permutations: number of permutations for p-value

    Returns:
        (mmd2, p_value) tuple.
    """
    if bandwidth is None:
        bandwidth = median_bandwidth(X, Y)

    mmd2_observed = compute_mmd_squared(X, Y, bandwidth)

    # Permutation test
    combined = np.vstack([X, Y])
    n = len(X)
    count = 0

    rng = np.random.RandomState(42)
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        X_perm = combined[perm[:n]]
        Y_perm = combined[perm[n:]]
        mmd2_perm = compute_mmd_squared(X_perm, Y_perm, bandwidth)
        if mmd2_perm >= mmd2_observed:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return mmd2_observed, p_value


def mmd_analysis(dataset_types, prior_source='both', n_synthetic=500,
                 n_permutations=500, seed=42):
    """MMD between neurostim data and synthetic reference(s).

    Feature vector: concatenate [X, y.reshape(-1,1)] → (n, 3).
    Computes MMD against GP bank and/or TabPFN Prior Bag bank.

    Args:
        dataset_types: list of dataset names
        prior_source: 'gp' | 'tabpfn_prior' | 'both'
        n_synthetic: number of synthetic datasets for reference
        n_permutations: permutations for p-value estimation
        seed: random seed

    Returns:
        dict with per-dataset, per-subject, per-EMG MMD results.
    """
    # Build synthetic reference feature matrix
    ref_features = {}

    if prior_source in ('gp', 'both'):
        gp_bank = generate_synthetic_gp_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        ref_features['gp'] = _bank_to_features(gp_bank)

    if prior_source in ('tabpfn_prior', 'both'):
        prior_bank = generate_tabpfn_prior_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        ref_features['prior'] = _bank_to_features(prior_bank)

    # Noise baseline (always included)
    noise_bank = generate_noise_bank(
        n_datasets=n_synthetic, n_features=2,
    )
    ref_features['noise'] = _bank_to_features(noise_bank)

    results = {}

    for dataset_type in dataset_types:
        subjects = ALL_SUBJECTS[dataset_type]
        dataset_results = {}

        for subj_idx in subjects:
            data = load_data(dataset_type, subj_idx)
            coords = data['ch2xy']
            scaler_x = MinMaxScaler()
            X = scaler_x.fit_transform(coords).astype(np.float32)
            n_emgs = data['sorted_respMean'].shape[1]

            subj_results = {}
            for emg_idx in range(n_emgs):
                # Fit scaler on ALL reps (matching preprocess_neural_data)
                resp_all = data['sorted_resp'][:, emg_idx, :]
                scaler_y = StandardScaler()
                scaler_y.fit(resp_all.reshape(-1, 1))
                y_mean = data['sorted_respMean'][:, emg_idx]
                y = scaler_y.transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)

                # Feature vector: [X, y]
                neurostim_features = np.hstack([X, y.reshape(-1, 1)])

                emg_result = {}
                for ref_name, ref_feat in ref_features.items():
                    mmd2, p_val = mmd_permutation_test(
                        neurostim_features, ref_feat,
                        n_permutations=n_permutations,
                    )
                    emg_result[f'mmd2_{ref_name}'] = mmd2
                    emg_result[f'p_{ref_name}'] = p_val

                subj_results[emg_idx] = emg_result

            dataset_results[subj_idx] = subj_results

        results[dataset_type] = dataset_results

    return results


def _bank_to_features(bank, max_samples=5000):
    """Convert a bank of (X, y) to a single feature matrix [X, y]."""
    parts = []
    for X, y in bank:
        feat = np.hstack([X, y.reshape(-1, 1)])
        parts.append(feat)

    combined = np.vstack(parts)
    # Subsample if too large for efficient MMD computation
    if len(combined) > max_samples:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(combined), max_samples, replace=False)
        combined = combined[idx]
    return combined


# ============================================================================
#  3b′. Sliced Wasserstein-2 Distance
# ============================================================================

def sliced_wasserstein(X, Y, n_projections=200, seed=42):
    """Sliced Wasserstein-2 distance between empirical distributions.

    Projects X, Y onto random 1D directions, computes exact W2 on each
    projection (sort + L2), averages over projections.

    Args:
        X: (n, d) array
        Y: (m, d) array
        n_projections: number of random 1D projections
        seed: random seed

    Returns:
        float: sliced W2 distance
    """
    d = X.shape[1]
    rng = np.random.RandomState(seed)

    # Random unit vectors on the d-sphere
    directions = rng.randn(n_projections, d)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    w2_sum = 0.0
    for theta in directions:
        # Project onto 1D
        Xp = X @ theta
        Yp = Y @ theta

        # Sort projections
        Xp_sorted = np.sort(Xp)
        Yp_sorted = np.sort(Yp)

        # Resample to equal length if n != m
        n, m = len(Xp_sorted), len(Yp_sorted)
        if n != m:
            # Interpolate the shorter array to match the longer
            target_len = max(n, m)
            Xp_sorted = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, n),
                Xp_sorted,
            )
            Yp_sorted = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, m),
                Yp_sorted,
            )

        w2_sum += np.mean((Xp_sorted - Yp_sorted) ** 2)

    return float(np.sqrt(w2_sum / n_projections))


def wasserstein_analysis(dataset_types, prior_source='both',
                         n_synthetic=500, n_projections=200, seed=42):
    """Sliced Wasserstein-2 between neurostim data and synthetic references.

    Feature vector: concatenate [X, y.reshape(-1,1)] → (n, 3).
    Mirrors mmd_analysis() structure but uses sliced W2 instead of
    kernel-based MMD.  W2 is parameter-free (no bandwidth choice).

    Args:
        dataset_types: list of dataset names
        prior_source: 'gp' | 'tabpfn_prior' | 'both'
        n_synthetic: number of synthetic datasets for reference
        n_projections: number of 1D projections for sliced W2
        seed: random seed

    Returns:
        dict with per-dataset, per-subject, per-EMG W2 results.
    """
    # Build synthetic reference feature matrices (reuse _bank_to_features)
    ref_features = {}

    if prior_source in ('gp', 'both'):
        gp_bank = generate_synthetic_gp_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        ref_features['gp'] = _bank_to_features(gp_bank)

    if prior_source in ('tabpfn_prior', 'both'):
        prior_bank = generate_tabpfn_prior_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        ref_features['prior'] = _bank_to_features(prior_bank)

    # Noise baseline (always included)
    noise_bank = generate_noise_bank(
        n_datasets=n_synthetic, n_features=2, seed=seed + 10000,
    )
    ref_features['noise'] = _bank_to_features(noise_bank)

    results = {}

    for dataset_type in dataset_types:
        subjects = ALL_SUBJECTS[dataset_type]
        dataset_results = {}

        for subj_idx in subjects:
            data = load_data(dataset_type, subj_idx)
            coords = data['ch2xy']
            scaler_x = MinMaxScaler()
            X = scaler_x.fit_transform(coords).astype(np.float32)
            n_emgs = data['sorted_respMean'].shape[1]

            subj_results = {}
            for emg_idx in range(n_emgs):
                resp_all = data['sorted_resp'][:, emg_idx, :]
                scaler_y = StandardScaler()
                scaler_y.fit(resp_all.reshape(-1, 1))
                y_mean = data['sorted_respMean'][:, emg_idx]
                y = scaler_y.transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)

                neurostim_features = np.hstack([X, y.reshape(-1, 1)])

                emg_result = {}
                for ref_name, ref_feat in ref_features.items():
                    w2 = sliced_wasserstein(
                        neurostim_features, ref_feat,
                        n_projections=n_projections, seed=seed,
                    )
                    emg_result[f'w2_{ref_name}'] = w2

                subj_results[emg_idx] = emg_result

            dataset_results[subj_idx] = subj_results

        results[dataset_type] = dataset_results

    return results


# ============================================================================
#  3c. Mahalanobis Distance in Representation Space
# ============================================================================

def extract_embeddings_frozen(model, X_train, y_train, X_test,
                              layer_name='encoder'):
    """Extract embeddings from frozen TabPFN via forward hooks.

    Hooks the feature encoder of TabPFN's PerFeatureTransformer, which
    produces shape (n_train + n_test, n_features, hidden_dim). The last
    n_test rows are test-point embeddings, flattened to (n_test, n_features * hidden_dim).

    Note: The transformer layers use cached/compressed representations with
    a fixed sequence length, so their outputs are not directly indexable per
    test point.  The encoder output is the correct hook target for per-point
    embeddings.

    Args:
        model: TabPFNRegressor (frozen, already fitted or will be fitted)
        X_train: (n_train, d) context features
        y_train: (n_train,) context targets
        X_test: (n_test, d) query features
        layer_name: which module to hook (default: 'encoder')

    Returns:
        (n_test, d_embed) embedding matrix as numpy array.
    """
    # Fit to establish context (suppress TabPFN internal power transform overflow)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='overflow', category=RuntimeWarning)
        model.fit(X_train, y_train)

    # Access the underlying PerFeatureTransformer
    inner_model = model.models_[0]

    # Navigate to hook target
    module = inner_model
    for attr in layer_name.split('.'):
        module = module[int(attr)] if attr.isdigit() else getattr(module, attr)

    # Register hook
    activations = {}
    n_test = len(X_test)

    def hook_fn(mod, inp, output):
        act = output if isinstance(output, torch.Tensor) else output[0]
        act = act.detach().cpu().float()
        # Transformer layer output: (batch=1, seq_len, n_feat_groups, hidden_dim)
        #   seq_len = n_thinking_tokens + n_train + n_test; n_feat_groups varies with dataset.
        #   Squeeze batch, mean-pool over n_feat_groups → (seq_len, hidden_dim).
        # Encoder output (legacy): (n_data, n_features, hidden_dim)
        #   Mean-pool over n_features → (n_data, hidden_dim).
        if act.ndim == 4:
            act = act[0].mean(dim=1)   # (seq_len, hidden_dim)
        elif act.ndim == 3:
            act = act.mean(dim=1)      # (n_data, hidden_dim)
        # act is now 2D: (n_rows, hidden_dim) — last n_test rows are test-point embeddings
        activations['target'] = act

    handle = module.register_forward_hook(hook_fn)

    # Trigger forward pass through TabPFN's normal predict path
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='overflow', category=RuntimeWarning)
        try:
            model.predict(X_test)
        finally:
            handle.remove()

    if 'target' not in activations:
        raise RuntimeError(f"Hook on '{layer_name}' did not capture activations")

    # Last n_test rows are test-point embeddings
    embeddings = activations['target'].numpy()
    return embeddings[-n_test:]


def compute_mahalanobis_distance(z, mu, sigma_inv):
    """Mahalanobis distance: D_M(z) = sqrt((z-mu)^T Sigma^{-1} (z-mu)).

    Args:
        z: (n, d) data points
        mu: (d,) reference mean
        sigma_inv: (d, d) inverse covariance matrix

    Returns:
        (n,) distances.
    """
    diff = z - mu
    left = diff @ sigma_inv
    dists_sq = np.sum(left * diff, axis=1)
    # Clip negative values from numerical errors
    dists_sq = np.maximum(dists_sq, 0.0)
    return np.sqrt(dists_sq)


def mahalanobis_analysis(dataset_types, device='cpu', prior_source='both',
                         n_synthetic=500, n_context=50, regularization=1e-2,
                         seed=42, layer: int = 17):
    """Full Mahalanobis pipeline.

    Phase 1: Build reference embedding distribution from synthetic data.
    Phase 2: Compute D_M for neurostim data against each reference.

    Automatically falls back to CPU if CUDA errors are encountered during
    embedding extraction (forward hooks are more sensitive to CUDA issues
    than regular inference).

    Args:
        dataset_types: list of dataset names.
        device: 'cpu' or 'cuda'.
        prior_source: 'gp' | 'tabpfn_prior' | 'both'.
        n_synthetic: number of synthetic datasets for reference.
        n_context: context size for embedding extraction.
        regularization: Tikhonov regularization for covariance inversion.
        seed: random seed.
        layer: transformer layer index for embedding extraction.
            Default 17 (last layer) — the most context-aware representation.

    Returns:
        dict with per-dataset Mahalanobis distances + reference statistics.
    """
    try:
        return _mahalanobis_analysis_inner(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context,
            regularization=regularization, seed=seed, layer=layer,
        )
    except _CudaError:
        if device == 'cpu':
            raise  # already on CPU, nothing to fall back to
        print("  [Mahalanobis] CUDA error in embedding extraction, "
              "falling back to CPU...")
        torch.cuda.empty_cache()
        return _mahalanobis_analysis_inner(
            dataset_types, device='cpu', prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context,
            regularization=regularization, seed=seed, layer=layer,
        )


def _mahalanobis_analysis_inner(
    dataset_types, device, prior_source, n_synthetic, n_context,
    regularization, seed, layer,
):
    """Core Mahalanobis implementation (called by mahalanobis_analysis)."""
    model = TabPFNRegressor(device=device)
    # Use single estimator for speed
    model.n_estimators = 1

    # Phase 1: Build reference embeddings
    ref_stats = {}

    _LAST_LAYER = _layer_name(layer)

    if prior_source in ('gp', 'both'):
        gp_bank = generate_synthetic_gp_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        gp_embeds = _embeddings_from_bank(model, gp_bank, n_context, _LAST_LAYER)
        ref_stats['gp'] = _fit_reference(gp_embeds, regularization)

    if prior_source in ('tabpfn_prior', 'both'):
        prior_bank = generate_tabpfn_prior_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        prior_embeds = _embeddings_from_bank(model, prior_bank, n_context, _LAST_LAYER)
        ref_stats['prior'] = _fit_reference(prior_embeds, regularization)

    # Noise baseline (always included)
    noise_bank = generate_noise_bank(
        n_datasets=n_synthetic, n_features=2, 
    )
    noise_embeds = _embeddings_from_bank(model, noise_bank, n_context, _LAST_LAYER)
    ref_stats['noise'] = _fit_reference(noise_embeds, regularization)

    # ------------------------------------------------------------------
    # OOD reference distances (Issue 4):
    #   • "vs GP" and "vs Prior" panels: add Noise OOD distances as contrast
    #   • "vs Noise" panel: add two structurally-different OOD types
    # ------------------------------------------------------------------
    print("  [Mahalanobis] Computing OOD reference distances...")
    rng_ood = np.random.RandomState(seed + 99)

    # Single noise OOD bank used against GP and Prior references
    ood_bank = generate_noise_bank(
        n_datasets=min(n_synthetic, 100), n_features=2
    )
    ood_embeds = _embeddings_from_bank(model, ood_bank, n_context, _LAST_LAYER)

    for ref_name in ('gp', 'prior'):
        if ref_name not in ref_stats:
            continue
        stats = ref_stats[ref_name]
        z = ood_embeds
        if stats['pca'] is not None:
            z = stats['pca'].transform(z)
        idx = rng_ood.choice(len(z), min(300, len(z)), replace=False)
        ref_stats[ref_name]['ood_distances'] = compute_mahalanobis_distance(
            z[idx], stats['mu'], stats['sigma_inv'],
        )

    # OOD distances from the correlated-noise bank (upper-bound reference)
    if 'noise' in ref_stats:
        extra_noise_bank = generate_noise_bank(
            n_datasets=min(n_synthetic, 100), n_features=2, seed=seed + 30000,
        )
        noise_stats = ref_stats['noise']
        z = _embeddings_from_bank(model, extra_noise_bank, n_context, _LAST_LAYER)
        if noise_stats['pca'] is not None:
            z = noise_stats['pca'].transform(z)
        idx = rng_ood.choice(len(z), min(300, len(z)), replace=False)
        noise_stats['ood_distances'] = compute_mahalanobis_distance(
            z[idx], noise_stats['mu'], noise_stats['sigma_inv'],
        )

    # Phase 2: Compute D_M for neurostim data
    results = {}

    for dataset_type in dataset_types:
        subjects = ALL_SUBJECTS[dataset_type]
        dataset_results = {}

        for subj_idx in subjects:
            data = load_data(dataset_type, subj_idx)
            coords = data['ch2xy']
            scaler_x = MinMaxScaler()
            X = scaler_x.fit_transform(coords).astype(np.float32)
            n_emgs = data['sorted_respMean'].shape[1]

            subj_results = {}
            for emg_idx in range(n_emgs):
                # Fit scaler on ALL reps (matching preprocess_neural_data)
                resp_all = data['sorted_resp'][:, emg_idx, :]
                scaler_y = StandardScaler()
                scaler_y.fit(resp_all.reshape(-1, 1))
                y_mean = data['sorted_respMean'][:, emg_idx]
                y = scaler_y.transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)

                n_ctx = min(n_context, len(X) - 1)
                X_ctx, y_ctx = X[:n_ctx], y[:n_ctx]
                X_tst = X[n_ctx:]

                if len(X_tst) == 0:
                    continue

                embeddings = extract_embeddings_frozen(
                    model, X_ctx, y_ctx, X_tst,
                    layer_name=_LAST_LAYER,
                )

                emg_result = {}
                for ref_name, stats in ref_stats.items():
                    # Project to same PCA space if used
                    z = embeddings
                    if stats['pca'] is not None:
                        z = stats['pca'].transform(z)

                    dists = compute_mahalanobis_distance(
                        z, stats['mu'], stats['sigma_inv'],
                    )
                    emg_result[f'distances_{ref_name}'] = dists
                    emg_result[f'mean_dist_{ref_name}'] = float(np.mean(dists))

                subj_results[emg_idx] = emg_result

            dataset_results[subj_idx] = subj_results

        results[dataset_type] = dataset_results

    # Include reference statistics for comparison plots
    results['ref_stats'] = {}
    for ref_name, stats in ref_stats.items():
        entry = {
            'self_distances': stats['self_distances'],
            'mean_self_dist': float(np.mean(stats['self_distances'])),
        }
        # Pass through OOD distance arrays added during Phase 1
        for key in ('ood_distances', 'ood1_distances', 'ood2_distances'):
            if key in stats:
                entry[key] = stats[key]
        results['ref_stats'][ref_name] = entry

    return results


class _CudaError(RuntimeError):
    """CUDA errors corrupt the device context and are unrecoverable."""
    pass


def _is_cuda_error(exc):
    """Check if an exception is a CUDA error (unrecoverable)."""
    msg = str(exc).lower()
    return 'cuda' in msg or 'cublas' in msg or 'cudnn' in msg


def _embeddings_from_bank(model, bank, n_context,
                          layer_name='transformer_encoder.layers.17'):
    """Extract embeddings for all datasets in a bank.

    Args:
        layer_name: module path to hook inside PerFeatureTransformer.
            Default is the last transformer block, which produces contextualised
            representations that discriminate in-distribution from OOD data.
            The old 'encoder' target (TorchPreprocessingPipeline) was too shallow
            to be discriminative — all datasets had CKA ≈ 0.05-0.07.
    """
    all_embeds = []
    n_failed = 0
    n_nonfinite_input = 0
    n_nonfinite_embed = 0
    n_exception = 0
    first_exception = None
    for X, y in bank:
        normed = _normalize_for_tabpfn(X, y)
        if normed is None:
            n_failed += 1
            n_nonfinite_input += 1
            continue
        X, y = normed
        n_ctx = min(n_context, len(X) - 1)
        if n_ctx < 2 or len(X) - n_ctx < 1:
            continue
        X_ctx, y_ctx = X[:n_ctx], y[:n_ctx]
        X_tst = X[n_ctx:]
        try:
            embeds = extract_embeddings_frozen(
                model, X_ctx, y_ctx, X_tst, layer_name=layer_name,
            )
            # Skip non-finite embeddings
            if not np.all(np.isfinite(embeds)):
                n_failed += 1
                n_nonfinite_embed += 1
                continue
            all_embeds.append(embeds)
        except Exception as e:
            n_failed += 1
            n_exception += 1
            if first_exception is None:
                first_exception = e
            # CUDA errors corrupt the context — all further attempts will fail
            if _is_cuda_error(e):
                break
            continue

    n_total = len(bank)
    n_ok = len(all_embeds)
    if n_failed > 0:
        print(f"  [Embeddings] {n_ok}/{n_total} succeeded, {n_failed} failed/skipped")
        print(f"    Breakdown: {n_nonfinite_input} non-finite input, "
              f"{n_nonfinite_embed} non-finite embeddings, "
              f"{n_exception} exceptions")
        if first_exception is not None:
            print(f"    First exception: {type(first_exception).__name__}: "
                  f"{first_exception}")

    if not all_embeds:
        # Surface CUDA errors as a distinct type for fallback handling
        if first_exception and _is_cuda_error(first_exception):
            raise _CudaError(
                f"CUDA error during embedding extraction: {first_exception}"
            ) from first_exception
        raise RuntimeError(
            f"All {n_total} datasets in the bank failed embedding extraction. "
            "Check synthetic data generation for inf/NaN values."
        )
    return np.vstack(all_embeds)


def _fit_reference(embeddings, regularization, max_pca_components=30):
    """Fit Gaussian reference (mu, Sigma^{-1}) from embeddings.

    Applies PCA if embedding dim >> n_samples to avoid singular covariance.
    Computes held-out self-distances for calibration.
    """
    n, d = embeddings.shape
    print(f"  [Reference] Fitting from {n} embeddings of dim {d}")

    if n < 10:
        raise RuntimeError(
            f"Only {n} valid embeddings — need at least 10 for a "
            "meaningful reference distribution."
        )

    pca = None
    if d > max_pca_components and n > max_pca_components:
        pca = PCA(n_components=max_pca_components)
        embeddings = pca.fit_transform(embeddings)
        d = embeddings.shape[1]

    mu = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)

    # np.cov returns a scalar when d==1; ensure always 2-D
    cov = np.atleast_2d(cov)

    # Regularize
    cov += regularization * np.eye(cov.shape[0])
    sigma_inv = np.linalg.inv(cov)

    # Self-distances (leave-one-out style via held-out split)
    n_held = max(1, min(100, n // 5))
    held_out = embeddings[-n_held:]
    self_dists = compute_mahalanobis_distance(held_out, mu, sigma_inv)

    return {
        'mu': mu,
        'sigma_inv': sigma_inv,
        'pca': pca,
        'self_distances': self_dists,
    }


# ============================================================================
#  3d. CKA (Centered Kernel Alignment) in Representation Space
# ============================================================================

def cka_analysis(dataset_types, device='cpu', prior_source='both',
                 n_synthetic=500, n_context=50, n_bootstrap=10, seed=42,
                 layers=None):
    """Multi-layer CKA between neurostim and synthetic reference embeddings.

    Measures whether TabPFN routes neurostim data through the same internal
    circuitry as its pretraining data.  High CKA (->1) means representations
    are geometrically similar; low CKA (->0) means they diverge.

    Computing CKA at multiple phase-aligned layers reveals WHERE in the
    network representations diverge — connecting to the three-phase attention
    structure from Ye et al. (2025, arXiv:2502.17361):
      - Layer 4 (end of early phase): attribute identity internalized
      - Layer 13 (deep phase onset): selective attention begins
      - Layer 17 (final): full context-aware representation

    Args:
        dataset_types: list of dataset names.
        device: 'cpu' or 'cuda'.
        prior_source: 'gp' | 'tabpfn_prior' | 'both'.
        n_synthetic: number of synthetic datasets for reference.
        n_context: context size for embedding extraction.
        n_bootstrap: subsampling rounds to align row counts for CKA.
        seed: random seed.
        layers: list of transformer layer indices to analyze.
            Defaults to ID_OOD_LAYERS ([4, 13, 17]).
            Use LAYERWISE_HEATMAP_LAYERS for dense Tier 2 sweep.

    Returns:
        dict with structure:
            results[dataset_type][subj_idx][emg_idx][layer_idx] = {
                'cka_gp': float, 'cka_noise': float, ...
            }
        Plus results['layers'] = list of analyzed layer indices.
    """
    if layers is None:
        layers = ID_OOD_LAYERS
    try:
        return _cka_analysis_inner(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context,
            n_bootstrap=n_bootstrap, seed=seed, layers=layers,
        )
    except _CudaError:
        if device == 'cpu':
            raise
        print("  [CKA] CUDA error in embedding extraction, "
              "falling back to CPU...")
        torch.cuda.empty_cache()
        return _cka_analysis_inner(
            dataset_types, device='cpu', prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context,
            n_bootstrap=n_bootstrap, seed=seed, layers=layers,
        )


def _cka_analysis_inner(dataset_types, device, prior_source, n_synthetic,
                        n_context, n_bootstrap, seed, layers):
    """Core multi-layer CKA implementation.

    Computes CKA at each layer in ``layers``, producing per-layer reference
    embeddings and per-layer neurostim scores.  The result structure nests
    by layer index so downstream code can build layer-wise heatmaps or
    select a single layer for scalar summaries.
    """
    model = TabPFNRegressor(device=device)
    model.n_estimators = 1

    # Phase 1: Build reference embeddings per layer
    # ref_embeddings[layer_idx][ref_name] = np.ndarray
    ref_embeddings: dict[int, dict[str, np.ndarray]] = {}

    for layer_idx in layers:
        lname = _layer_name(layer_idx)
        layer_refs: dict[str, np.ndarray] = {}

        if prior_source in ('gp', 'both'):
            gp_bank = generate_synthetic_gp_bank(
                n_datasets=n_synthetic, n_features=2, seed=seed,
            )
            layer_refs['gp'] = _embeddings_from_bank(
                model, gp_bank, n_context, lname,
            )

        if prior_source in ('tabpfn_prior', 'both'):
            prior_bank = generate_tabpfn_prior_bank(
                n_datasets=n_synthetic, n_features=2, seed=seed,
            )
            layer_refs['prior'] = _embeddings_from_bank(
                model, prior_bank, n_context, lname,
            )

        noise_bank = generate_noise_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed + 10000,
        )
        layer_refs['noise'] = _embeddings_from_bank(
            model, noise_bank, n_context, lname,
        )

        ref_embeddings[layer_idx] = layer_refs

    # Phase 2: Compute CKA for neurostim data at each layer
    rng = np.random.RandomState(seed)
    results: dict = {}

    for dataset_type in dataset_types:
        subjects = ALL_SUBJECTS[dataset_type]
        dataset_results = {}

        for subj_idx in subjects:
            data = load_data(dataset_type, subj_idx)
            coords = data['ch2xy']
            scaler_x = MinMaxScaler()
            X = scaler_x.fit_transform(coords).astype(np.float32)
            n_emgs = data['sorted_respMean'].shape[1]

            subj_results = {}
            for emg_idx in range(n_emgs):
                resp_all = data['sorted_resp'][:, emg_idx, :]
                scaler_y = StandardScaler()
                scaler_y.fit(resp_all.reshape(-1, 1))
                y_mean = data['sorted_respMean'][:, emg_idx]
                y = scaler_y.transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)

                n_ctx = min(n_context, len(X) - 1)
                X_ctx, y_ctx = X[:n_ctx], y[:n_ctx]
                X_tst = X[n_ctx:]

                if len(X_tst) == 0:
                    continue

                # Per-layer CKA scores for this EMG channel
                emg_result: dict[int, dict[str, float]] = {}

                for layer_idx in layers:
                    lname = _layer_name(layer_idx)
                    embeddings = extract_embeddings_frozen(
                        model, X_ctx, y_ctx, X_tst,
                        layer_name=lname,
                    )
                    n_test = len(embeddings)

                    layer_scores: dict[str, float] = {}
                    for ref_name, ref_emb in ref_embeddings[layer_idx].items():
                        cka_scores = []
                        for _ in range(n_bootstrap):
                            idx = rng.choice(
                                len(ref_emb), min(n_test, len(ref_emb)),
                                replace=False,
                            )
                            ref_sub = ref_emb[idx]
                            test_sub = embeddings[:len(ref_sub)]

                            score = linear_cka(
                                torch.from_numpy(ref_sub).float(),
                                torch.from_numpy(test_sub).float(),
                            )
                            cka_scores.append(score)

                        layer_scores[f'cka_{ref_name}'] = float(
                            np.mean(cka_scores)
                        )

                    emg_result[layer_idx] = layer_scores

                subj_results[emg_idx] = emg_result

            dataset_results[subj_idx] = subj_results

        results[dataset_type] = dataset_results

    # Store analyzed layers as metadata for visualization
    results['layers'] = list(layers)

    return results


# ============================================================================
#  3d-ii. RSA (Representational Similarity Analysis)
# ============================================================================

def compute_rsa(
    Z1: np.ndarray,
    Z2: np.ndarray,
    n_subsample: int = 300,
    seed: int = 42,
) -> float:
    """Compute RSA Spearman rho between two embedding clouds.

    Subsamples both Z1 and Z2 to n_subsample rows, computes pairwise
    Euclidean RDMs, vectorizes the upper triangle (k=1), and returns the
    Spearman rank correlation between the two RDM vectors.

    Args:
        Z1: First embedding matrix, shape (n1, D).
        Z2: Second embedding matrix, shape (n2, D).
        n_subsample: Max rows to draw from each matrix. Capped to
            min(n1, n2, n_subsample) automatically.
        seed: Random seed for subsampling reproducibility.

    Returns:
        Spearman rho in [-1, 1]. Returns 0.0 if either RDM vector has
        zero variance (degenerate case).
    """
    rng = np.random.RandomState(seed)
    n = min(len(Z1), len(Z2), n_subsample)
    idx1 = rng.choice(len(Z1), n, replace=False)
    idx2 = rng.choice(len(Z2), n, replace=False)
    Z1s = Z1[idx1]   # [n, D]
    Z2s = Z2[idx2]   # [n, D]

    rdm1 = cdist(Z1s, Z1s, metric='euclidean')  # [n, n]
    rdm2 = cdist(Z2s, Z2s, metric='euclidean')  # [n, n]

    triu_idx = np.triu_indices(n, k=1)
    v1 = rdm1[triu_idx]
    v2 = rdm2[triu_idx]

    if v1.std() == 0.0 or v2.std() == 0.0:
        return 0.0

    return float(spearmanr(v1, v2).statistic)


def rsa_analysis(
    dataset_types: list[str],
    device: str = 'cpu',
    prior_source: str = 'both',
    n_synthetic: int = 500,
    n_context: int = 50,
    n_subsample: int = 300,
    seed: int = 42,
    layers: list[int] | None = None,
) -> dict:
    """Multi-layer RSA between neurostim and synthetic reference embeddings.

    Computes RSA Spearman rho at each transformer layer, revealing where
    neurostim geometry tracks the prior's geometry vs. the noise baseline.
    High rho (->1) at a given layer means the pairwise distance structure
    of neurostim embeddings matches the reference; near-zero or negative
    rho means geometric dissimilarity.

    Args:
        dataset_types: List of dataset names.
        device: 'cpu' or 'cuda'.
        prior_source: 'gp' | 'tabpfn_prior' | 'both'.
        n_synthetic: Number of synthetic datasets for reference embeddings.
        n_context: Context size for embedding extraction.
        n_subsample: Points to subsample per cloud for RDM computation.
        seed: Random seed.
        layers: Transformer layer indices to analyze.
            Defaults to ID_OOD_LAYERS ([4, 13, 17]).

    Returns:
        dict with structure::

            results[dataset_type][subj_idx][emg_idx][layer_idx] = {
                'rsa_gp': float, 'rsa_prior': float, 'rsa_noise': float
            }

        Plus results['layers'] = list of analyzed layer indices.
    """
    if layers is None:
        layers = ID_OOD_LAYERS
    try:
        return _rsa_analysis_inner(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context,
            n_subsample=n_subsample, seed=seed, layers=layers,
        )
    except _CudaError:
        if device == 'cpu':
            raise
        print("  [RSA] CUDA error in embedding extraction, "
              "falling back to CPU...")
        torch.cuda.empty_cache()
        return _rsa_analysis_inner(
            dataset_types, device='cpu', prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context,
            n_subsample=n_subsample, seed=seed, layers=layers,
        )


def _rsa_analysis_inner(
    dataset_types: list[str],
    device: str,
    prior_source: str,
    n_synthetic: int,
    n_context: int,
    n_subsample: int,
    seed: int,
    layers: list[int],
) -> dict:
    """Core multi-layer RSA implementation.

    Mirrors _cka_analysis_inner() but calls compute_rsa() instead of
    linear_cka(), omitting the bootstrap loop (RSA subsamples internally).
    """
    model = TabPFNRegressor(device=device)
    model.n_estimators = 1

    # Phase 1: Build reference embeddings per layer
    ref_embeddings: dict[int, dict[str, np.ndarray]] = {}

    for layer_idx in layers:
        lname = _layer_name(layer_idx)
        layer_refs: dict[str, np.ndarray] = {}

        if prior_source in ('gp', 'both'):
            gp_bank = generate_synthetic_gp_bank(
                n_datasets=n_synthetic, n_features=2, seed=seed,
            )
            layer_refs['gp'] = _embeddings_from_bank(
                model, gp_bank, n_context, lname,
            )

        if prior_source in ('tabpfn_prior', 'both'):
            prior_bank = generate_tabpfn_prior_bank(
                n_datasets=n_synthetic, n_features=2, seed=seed,
            )
            layer_refs['prior'] = _embeddings_from_bank(
                model, prior_bank, n_context, lname,
            )

        noise_bank = generate_noise_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed + 10000,
        )
        layer_refs['noise'] = _embeddings_from_bank(
            model, noise_bank, n_context, lname,
        )

        ref_embeddings[layer_idx] = layer_refs

    # Phase 2: Compute RSA for neurostim data at each layer
    results: dict = {}

    for dataset_type in dataset_types:
        subjects = ALL_SUBJECTS[dataset_type]
        dataset_results: dict = {}

        for subj_idx in subjects:
            data = load_data(dataset_type, subj_idx)
            coords = data['ch2xy']
            scaler_x = MinMaxScaler()
            X = scaler_x.fit_transform(coords).astype(np.float32)
            n_emgs = data['sorted_respMean'].shape[1]

            subj_results: dict = {}
            for emg_idx in range(n_emgs):
                resp_all = data['sorted_resp'][:, emg_idx, :]
                scaler_y = StandardScaler()
                scaler_y.fit(resp_all.reshape(-1, 1))
                y_mean = data['sorted_respMean'][:, emg_idx]
                y = scaler_y.transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)

                n_ctx = min(n_context, len(X) - 1)
                X_ctx, y_ctx = X[:n_ctx], y[:n_ctx]
                X_tst = X[n_ctx:]

                if len(X_tst) == 0:
                    continue

                emg_result: dict[int, dict[str, float]] = {}

                for layer_idx in layers:
                    lname = _layer_name(layer_idx)
                    embeddings = extract_embeddings_frozen(
                        model, X_ctx, y_ctx, X_tst,
                        layer_name=lname,
                    )

                    layer_scores: dict[str, float] = {}
                    for ref_name, ref_emb in ref_embeddings[layer_idx].items():
                        rho = compute_rsa(
                            embeddings, ref_emb,
                            n_subsample=n_subsample,
                            seed=seed,
                        )
                        layer_scores[f'rsa_{ref_name}'] = rho

                    emg_result[layer_idx] = layer_scores

                subj_results[emg_idx] = emg_result

            dataset_results[subj_idx] = subj_results

        results[dataset_type] = dataset_results

    results['layers'] = list(layers)
    return results


# ============================================================================
#  3d-iii. Procrustes BO-Trajectory (B7)
# ============================================================================

# Default BO budget steps for the trajectory sweep.  Budget=2 is the minimum
# valid context size for TabPFN.fit() and serves as the disparity baseline.
PROC_BUDGETS_DEFAULT: list[int] = [2, 10, 30, 50, 100]


def compute_procrustes_disparity(
    Z1: np.ndarray,
    Z2: np.ndarray,
    n_subsample: int = 300,
    seed: int = 42,
) -> float:
    """Procrustes disparity between two embedding clouds with row correspondence.

    ``scipy.spatial.procrustes`` centers each cloud, scales to unit Frobenius
    norm, and finds the rotation/reflection R minimizing ``||X - YR||_F`` via
    SVD.  Disparity = ``1 - ||YR^T X^T||_F^2`` in ``[0, 1]``.

    Procrustes assumes rows are ordered (row *i* of Z1 corresponds to row *i*
    of Z2).  The intended use is to compare embeddings of the *same* query
    points under two different contexts — rows naturally align by query
    point index.

    Args:
        Z1: First embedding matrix, shape ``(n, D)``.
        Z2: Second embedding matrix, shape ``(n, D)`` — must match Z1's shape.
        n_subsample: Subsample this many rows (applied to both clouds with
            the same indices so correspondence is preserved).
        seed: Random seed for subsampling.

    Returns:
        Disparity in ``[0, 1]``.  Returns ``np.nan`` if either cloud contains
        non-finite values or has zero Frobenius norm after centering.
    """
    if Z1.shape != Z2.shape:
        raise ValueError(
            f"Procrustes requires matching shapes: {Z1.shape} vs {Z2.shape}"
        )
    n = min(len(Z1), n_subsample)
    if n < 2:
        return float('nan')

    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(Z1), n, replace=False))

    X = Z1[idx].astype(np.float64)
    Y = Z2[idx].astype(np.float64)

    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        return float('nan')

    try:
        _, _, disparity = scipy_procrustes(X, Y)
    except ValueError:
        return float('nan')
    return float(disparity)


def _trajectory_disparities(
    model: TabPFNRegressor,
    X: np.ndarray,
    y: np.ndarray,
    budgets: list[int],
    rng: np.random.RandomState,
    layer_name: str,
    n_subsample: int,
) -> list[float] | None:
    """Single BO-trajectory disparity curve.

    Picks a random permutation of X.  At each budget t, uses the first t
    permuted points as TabPFN context and extracts embeddings for the full
    search space at ``layer_name``.  Returns the Procrustes disparity between
    E(budgets[0]) and each E(t).

    Returns:
        List of len(budgets) disparities, or None if any embedding extraction
        fails or produces non-finite values (signals caller to skip this
        trajectory).
    """
    n = len(X)
    perm = rng.permutation(n)

    embeddings: list[np.ndarray] = []
    for t in budgets:
        t_clamped = max(2, min(int(t), n - 1))
        ctx = perm[:t_clamped]
        try:
            emb = extract_embeddings_frozen(
                model, X[ctx], y[ctx], X, layer_name=layer_name,
            )                                                       # [n, d_hidden]
        except Exception as exc:
            if _is_cuda_error(exc):
                raise _CudaError(str(exc)) from exc
            return None
        if not np.all(np.isfinite(emb)):
            return None
        embeddings.append(emb)

    baseline = embeddings[0]
    return [
        compute_procrustes_disparity(baseline, emb, n_subsample=n_subsample,
                                     seed=0)
        for emb in embeddings
    ]


def embedding_trajectory_analysis(
    dataset_types: list[str],
    device: str = 'cpu',
    prior_source: str = 'tabpfn_prior',
    n_synthetic: int = 20,
    budgets: list[int] | None = None,
    layer: int = 17,
    n_subsample: int = 300,
    seed: int = 42,
) -> dict:
    """Procrustes BO-trajectory analysis (B7).

    For each (subject, EMG) pair and each synthetic reference dataset, extracts
    embeddings of the full search space at transformer ``layer`` for a sweep
    of BO budgets.  Disparity is computed against the smallest-budget
    embedding — measuring how the internal geometry evolves as more context
    accumulates.  Smooth convergence is predicted for ID data; erratic drift
    is predicted for OOD data.

    Args:
        dataset_types: Dataset names (e.g. ``['rat', 'nhp']``).
        device: ``'cpu'`` or ``'cuda'``.  Auto-falls back to CPU on CUDA error.
        prior_source: ``'gp' | 'tabpfn_prior' | 'both'``.
        n_synthetic: Number of synthetic datasets per reference source.
            Kept small (default 20) because each dataset yields a full
            budget sweep.
        budgets: BO budget steps.  Default ``PROC_BUDGETS_DEFAULT``.
        layer: Transformer layer index for embedding extraction.
        n_subsample: Subsample size for Procrustes computation.
        seed: Random seed.

    Returns:
        dict with structure::

            {
                'budgets': [...],
                'layer': int,
                <dataset_type>: {subj_idx: {emg_idx: [disparities]}},
                'synthetic_gp':    [ [disparities], ... ],     # optional
                'synthetic_prior': [ [disparities], ... ],     # optional
                'synthetic_noise': [ [disparities], ... ],
            }
    """
    if budgets is None:
        budgets = list(PROC_BUDGETS_DEFAULT)
    try:
        return _embedding_trajectory_inner(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=n_synthetic, budgets=budgets, layer=layer,
            n_subsample=n_subsample, seed=seed,
        )
    except _CudaError:
        if device == 'cpu':
            raise
        print("  [Procrustes] CUDA error in embedding extraction, "
              "falling back to CPU...")
        torch.cuda.empty_cache()
        return _embedding_trajectory_inner(
            dataset_types, device='cpu', prior_source=prior_source,
            n_synthetic=n_synthetic, budgets=budgets, layer=layer,
            n_subsample=n_subsample, seed=seed,
        )


def _embedding_trajectory_inner(
    dataset_types: list[str],
    device: str,
    prior_source: str,
    n_synthetic: int,
    budgets: list[int],
    layer: int,
    n_subsample: int,
    seed: int,
) -> dict:
    """Core trajectory implementation (called by embedding_trajectory_analysis)."""
    model = TabPFNRegressor(device=device)
    model.n_estimators = 1

    layer_name = _layer_name(layer)
    max_budget = max(budgets)

    results: dict = {'budgets': list(budgets), 'layer': layer}

    # ── Neurostim trajectories ────────────────────────────────────────────
    for dataset_type in dataset_types:
        subjects = ALL_SUBJECTS[dataset_type]
        ds_traj: dict = {}
        for subj_idx in subjects:
            data = load_data(dataset_type, subj_idx)
            coords = data['ch2xy']
            scaler_x = MinMaxScaler()
            X = scaler_x.fit_transform(coords).astype(np.float32)   # [n, 2]
            n_emgs = data['sorted_respMean'].shape[1]

            if len(X) < max_budget + 1:
                print(f"  [Procrustes] Skipping {dataset_type} S{subj_idx}: "
                      f"grid size {len(X)} < max budget {max_budget}")
                continue

            subj_traj: dict = {}
            for emg_idx in range(n_emgs):
                resp_all = data['sorted_resp'][:, emg_idx, :]
                scaler_y = StandardScaler()
                scaler_y.fit(resp_all.reshape(-1, 1))
                y_mean = data['sorted_respMean'][:, emg_idx]
                y = scaler_y.transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)                         # [n]

                rng_pair = np.random.RandomState(
                    seed + 1000 * (subj_idx + 1) + 11 * emg_idx,
                )
                traj = _trajectory_disparities(
                    model, X, y, budgets, rng_pair, layer_name, n_subsample,
                )
                if traj is not None:
                    subj_traj[emg_idx] = traj
            ds_traj[subj_idx] = subj_traj
        results[dataset_type] = ds_traj

    # ── Synthetic reference trajectories ──────────────────────────────────
    synthetic_banks: dict[str, list] = {}
    if prior_source in ('gp', 'both'):
        synthetic_banks['gp'] = generate_synthetic_gp_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
    if prior_source in ('tabpfn_prior', 'both'):
        synthetic_banks['prior'] = generate_tabpfn_prior_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
    synthetic_banks['noise'] = generate_noise_bank(
        n_datasets=n_synthetic, n_features=2, seed=seed + 10000,
    )

    for src_name, bank in synthetic_banks.items():
        src_trajs: list[list[float]] = []
        n_skipped = 0
        for ds_idx, (X_raw, y_raw) in enumerate(bank):
            normed = _normalize_for_tabpfn(X_raw, y_raw)
            if normed is None:
                n_skipped += 1
                continue
            X_n, y_n = normed
            if len(X_n) < max_budget + 1:
                n_skipped += 1
                continue
            rng_pair = np.random.RandomState(seed + 100_000 + ds_idx)
            traj = _trajectory_disparities(
                model, X_n, y_n, budgets, rng_pair, layer_name, n_subsample,
            )
            if traj is not None:
                src_trajs.append(traj)
            else:
                n_skipped += 1
        print(f"  [Procrustes] synthetic_{src_name}: {len(src_trajs)} valid, "
              f"{n_skipped} skipped")
        results[f'synthetic_{src_name}'] = src_trajs

    return results


# ============================================================================
#  3e. Gradient L2-Norm at Step 0
# ============================================================================

def compute_gradient_norm_frozen(model, X_train, y_train, X_test, y_test):
    """Gradient L2-norm from one backward pass through frozen TabPFN.

    Fits TabPFN on (X_train, y_train), predicts on X_test to get bar-
    distribution logits, computes cross-entropy loss against discretized
    y_test, backprops, and returns total L2 gradient norm.

    If the data is in-distribution, the pretrained weights are already near
    a local minimum → gradient norms will be small.  OOD data produces
    large gradients because the model needs significant updates.

    Args:
        model: TabPFNRegressor (will temporarily enable gradients)
        X_train: (n_train, d) context features
        y_train: (n_train,) context targets
        X_test: (n_test, d) query features
        y_test: (n_test,) query targets

    Returns:
        float: total gradient L2 norm across all parameters.
    """
    # Fit to establish context and preprocessing pipeline
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='overflow',
                                category=RuntimeWarning)
        model.fit(X_train, y_train)

    inner_model = model.models_[0]

    # Enable gradients on frozen parameters
    for p in inner_model.parameters():
        p.requires_grad_(True)

    try:
        # Bypass TabPFN's predict() which wraps in inference mode.
        # Call the inner PerFeatureTransformer directly:
        #   x: [N_train + N_test, 1, n_features]
        #   y: [N_train, 1]
        X_train_t = torch.tensor(
            X_train, dtype=torch.float32,
        ).unsqueeze(1)
        y_train_t = torch.tensor(
            y_train, dtype=torch.float32,
        ).unsqueeze(1)
        X_test_t = torch.tensor(
            X_test, dtype=torch.float32,
        ).unsqueeze(1)
        x_input = torch.cat([X_train_t, X_test_t], dim=0)

        with torch.enable_grad():
            output = inner_model(
                x_input, y_train_t, only_return_standard_out=True,
            )
        # output: [n_test, 1, n_bars] — squeeze the singleton dim
        logits = output.squeeze(1)  # (n_test, n_bars)

        # Discretize y_test into bar-distribution bin indices
        borders = model.znorm_space_bardist_.borders  # 1D tensor of bin edges
        y_tensor = torch.tensor(y_test, dtype=torch.float32,
                                device=logits.device)
        target_indices = torch.searchsorted(borders, y_tensor).clamp(
            0, logits.shape[1] - 1,
        )

        loss = F.cross_entropy(logits, target_indices)
        loss.backward()

        # Collect total L2 norm
        total_norm_sq = 0.0
        for p in inner_model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.norm().item() ** 2

        return float(np.sqrt(total_norm_sq))

    finally:
        # Cleanup: zero grads and disable requires_grad
        inner_model.zero_grad()
        for p in inner_model.parameters():
            p.requires_grad_(False)


def _gradient_norm_from_bank(model, bank, n_context):
    """Compute gradient norms for a bank of synthetic datasets."""
    all_norms = []
    for X, y in bank:
        normed = _normalize_for_tabpfn(X, y)
        if normed is None:
            continue
        X, y = normed
        n_ctx = min(n_context, len(X) - 1)
        if n_ctx < 2 or len(X) - n_ctx < 1:
            continue
        X_ctx, y_ctx = X[:n_ctx], y[:n_ctx]
        X_tst, y_tst = X[n_ctx:], y[n_ctx:]
        try:
            norm = compute_gradient_norm_frozen(
                model, X_ctx, y_ctx, X_tst, y_tst,
            )
            if np.isfinite(norm):
                all_norms.append(norm)
        except Exception:
            continue
    return np.array(all_norms) if all_norms else np.array([])


def gradient_norm_analysis(dataset_types, device='cpu', prior_source='both',
                           n_synthetic=500, n_context=50, seed=42):
    """Gradient L2-norm at step 0 for neurostim and synthetic data.

    Creates a fresh TabPFNRegressor to avoid corrupting state for other
    analyses.

    Args:
        dataset_types: list of dataset names
        device: 'cpu' or 'cuda'
        prior_source: 'gp' | 'tabpfn_prior' | 'both'
        n_synthetic: number of synthetic datasets for reference
        n_context: context size
        seed: random seed

    Returns:
        dict with per-dataset/subject/EMG gradient norms + synthetic baselines.
    """
    # Fresh model to avoid state corruption from gradient computation
    model = TabPFNRegressor(device=device)
    results = {}

    # --- Neurostim data ---
    for dataset_type in dataset_types:
        subjects = ALL_SUBJECTS[dataset_type]
        dataset_results = {}

        for subj_idx in subjects:
            data = load_data(dataset_type, subj_idx)
            coords = data['ch2xy']
            scaler_x = MinMaxScaler()
            X = scaler_x.fit_transform(coords).astype(np.float32)
            n_emgs = data['sorted_respMean'].shape[1]

            subj_results = {}
            for emg_idx in range(n_emgs):
                resp_all = data['sorted_resp'][:, emg_idx, :]
                scaler_y = StandardScaler()
                scaler_y.fit(resp_all.reshape(-1, 1))
                y_mean = data['sorted_respMean'][:, emg_idx]
                y = scaler_y.transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)

                n_ctx = min(n_context, len(X) - 1)
                X_ctx, y_ctx = X[:n_ctx], y[:n_ctx]
                X_tst, y_tst = X[n_ctx:], y[n_ctx:]

                if len(X_tst) == 0:
                    continue

                try:
                    norm = compute_gradient_norm_frozen(
                        model, X_ctx, y_ctx, X_tst, y_tst,
                    )
                    subj_results[emg_idx] = norm
                except Exception as e:
                    print(f"  [GradNorm] Failed {dataset_type} S{subj_idx} "
                          f"EMG{emg_idx}: {e}")
                    continue

            dataset_results[subj_idx] = subj_results

        results[dataset_type] = dataset_results

    # --- Synthetic references ---
    if prior_source in ('gp', 'both'):
        gp_bank = generate_synthetic_gp_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        results['synthetic_gp'] = _gradient_norm_from_bank(
            model, gp_bank, n_context,
        )

    if prior_source in ('tabpfn_prior', 'both'):
        prior_bank = generate_tabpfn_prior_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        results['synthetic_prior'] = _gradient_norm_from_bank(
            model, prior_bank, n_context,
        )

    noise_bank = generate_noise_bank(
        n_datasets=n_synthetic, n_features=2, seed=seed + 10000,
    )
    results['noise'] = _gradient_norm_from_bank(model, noise_bank, n_context)

    return results


# ============================================================================
#  3f. Unified Runner
# ============================================================================

def run_id_ood_analysis(dataset_types=None, analyses=None,
                        prior_source='both', device='cpu',
                        n_synthetic=500, n_context=100, seed=42,
                        save=False, output_dir=None,
                        cka_layers=None,
                        proc_budgets=None,
                        proc_layer=17,
                        proc_n_synthetic=20):
    """Orchestrate all ID/OOD analyses.

    Generates synthetic bank(s) once, shared across analyses.

    Args:
        dataset_types: list of dataset names (default: ['rat', 'nhp'])
        analyses: list of analysis names (default: ['entropy', 'mmd', 'mahalanobis']).
            Additional: 'cka', 'wasserstein', 'gradient_norm', 'rsa',
            'procrustes'.
        prior_source: 'gp' | 'tabpfn_prior' | 'both'
        device: 'cpu' or 'cuda'
        n_synthetic: number of synthetic datasets
        n_context: context size for entropy/Mahalanobis
        seed: random seed
        save: whether to save results to disk
        output_dir: base output directory
        cka_layers: transformer layer indices for CKA analysis. None uses
            ID_OOD_LAYERS default inside cka_analysis().
        proc_budgets: BO budget steps for Procrustes trajectory (B7).
            None uses ``PROC_BUDGETS_DEFAULT``.
        proc_layer: Transformer layer for Procrustes embedding extraction
            (default 17).
        proc_n_synthetic: Synthetic datasets per reference source for the
            Procrustes trajectory (default 20 — smaller than other analyses
            because each dataset yields a full budget sweep).

    Returns:
        dict with results per analysis type.
    """
    if dataset_types is None:
        dataset_types = ['rat', 'nhp']
    if analyses is None:
        analyses = ['entropy', 'mmd', 'mahalanobis']

    if output_dir is None:
        output_dir = os.path.join('output', 'id_ood')

    all_results = {}

    if 'entropy' in analyses:
        print("=" * 60)
        print("[ID/OOD] Running entropy analysis...")
        print("=" * 60)
        entropy_results = entropy_analysis(
            dataset_types, device=device, n_context=n_context,
            prior_source=prior_source, n_synthetic=n_synthetic, seed=seed,
        )
        all_results['entropy'] = entropy_results

        if save:
            out = os.path.join(output_dir, 'entropy')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'entropy_results.pkl'), 'wb') as f:
                pickle.dump(entropy_results, f)
            print(f"Saved entropy results -> {out}")

    if 'mmd' in analyses:
        print("=" * 60)
        print("[ID/OOD] Running MMD analysis...")
        print("=" * 60)
        mmd_results = mmd_analysis(
            dataset_types, prior_source=prior_source,
            n_synthetic=n_synthetic, n_permutations=500, seed=seed,
        )
        all_results['mmd'] = mmd_results

        if save:
            out = os.path.join(output_dir, 'mmd')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'mmd_results.pkl'), 'wb') as f:
                pickle.dump(mmd_results, f)
            print(f"Saved MMD results -> {out}")

    if 'mahalanobis' in analyses:
        print("=" * 60)
        print("[ID/OOD] Running Mahalanobis analysis...")
        print("=" * 60)
        mahalanobis_results = mahalanobis_analysis(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context,
            regularization=1e-2, seed=seed,
        )
        all_results['mahalanobis'] = mahalanobis_results

        if save:
            out = os.path.join(output_dir, 'mahalanobis')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'mahalanobis_results.pkl'), 'wb') as f:
                pickle.dump(mahalanobis_results, f)
            print(f"Saved Mahalanobis results -> {out}")

    if 'cka' in analyses:
        print("=" * 60)
        print("[ID/OOD] Running CKA analysis...")
        print("=" * 60)
        cka_results = cka_analysis(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context, seed=seed,
            layers=cka_layers,
        )
        all_results['cka'] = cka_results

        if save:
            out = os.path.join(output_dir, 'cka')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'cka_results.pkl'), 'wb') as f:
                pickle.dump(cka_results, f)
            print(f"Saved CKA results -> {out}")

    if 'wasserstein' in analyses:
        print("=" * 60)
        print("[ID/OOD] Running Wasserstein analysis...")
        print("=" * 60)
        wasserstein_results = wasserstein_analysis(
            dataset_types, prior_source=prior_source,
            n_synthetic=n_synthetic, seed=seed,
        )
        all_results['wasserstein'] = wasserstein_results

        if save:
            out = os.path.join(output_dir, 'wasserstein')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'wasserstein_results.pkl'), 'wb') as f:
                pickle.dump(wasserstein_results, f)
            print(f"Saved Wasserstein results -> {out}")

    if 'gradient_norm' in analyses:
        print("=" * 60)
        print("[ID/OOD] Running gradient norm analysis...")
        print("=" * 60)
        gradient_results = gradient_norm_analysis(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context, seed=seed,
        )
        all_results['gradient_norm'] = gradient_results

        if save:
            out = os.path.join(output_dir, 'gradient_norm')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'gradient_norm_results.pkl'), 'wb') as f:
                pickle.dump(gradient_results, f)
            print(f"Saved gradient norm results -> {out}")

    if 'rsa' in analyses:
        print("=" * 60)
        print("[ID/OOD] Running RSA analysis...")
        print("=" * 60)
        rsa_results = rsa_analysis(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context, seed=seed,
        )
        all_results['rsa'] = rsa_results

        if save:
            out = os.path.join(output_dir, 'rsa')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'rsa_results.pkl'), 'wb') as f:
                pickle.dump(rsa_results, f)
            print(f"Saved RSA results -> {out}")

    if 'procrustes' in analyses:
        print("=" * 60)
        print("[ID/OOD] Running Procrustes BO-trajectory analysis (B7)...")
        print("=" * 60)
        procrustes_results = embedding_trajectory_analysis(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=proc_n_synthetic, budgets=proc_budgets,
            layer=proc_layer, seed=seed,
        )
        all_results['procrustes'] = procrustes_results

        if save:
            out = os.path.join(output_dir, 'trajectory')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'procrustes_results.pkl'), 'wb') as f:
                pickle.dump(procrustes_results, f)
            print(f"Saved Procrustes trajectory results -> {out}")

    return all_results


