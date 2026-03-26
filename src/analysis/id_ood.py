"""
Core ID/OOD analysis: Shannon entropy, MMD, and Mahalanobis distance.

Tests whether neurostim data lies within TabPFN's pre-training prior by
comparing against synthetic reference distributions (GP and/or Prior Bag).
"""
import os
import pickle
import warnings

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabpfn import TabPFNRegressor

from utils.data_utils import load_data, ALL_SUBJECTS

from analysis.synthetic_gp import generate_synthetic_gp_bank
from analysis.synthetic_noise import generate_noise_bank
from analysis.synthetic_tabpfn_prior import generate_tabpfn_prior_bank


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


def entropy_analysis(dataset_types, device='cpu', n_context=50,
                     prior_source='both', n_synthetic=500, seed=42):
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
                y_mean = data['sorted_respMean'][:, emg_idx]
                scaler_y = StandardScaler()
                y = scaler_y.fit_transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)

                # Split: first n_context as context, rest as test
                n_ctx = min(n_context, len(X) - 1)
                X_ctx, y_ctx = X[:n_ctx], y[:n_ctx]
                X_tst = X[n_ctx:]

                if len(X_tst) == 0:
                    continue

                entropy = compute_bar_distribution_entropy(
                    model, X_ctx, y_ctx, X_tst,
                )
                subj_results[emg_idx] = entropy

            dataset_results[subj_idx] = subj_results

        results[dataset_type] = dataset_results

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
        n_datasets=n_synthetic, n_features=2, seed=seed + 10000,
    )
    results['noise'] = _entropy_from_bank(model, noise_bank, n_context)

    return results


def _entropy_from_bank(model, bank, n_context):
    """Compute entropy for a bank of synthetic datasets."""
    all_entropies = []
    for X, y in bank:
        normed = _normalize_for_tabpfn(X, y)
        if normed is None:
            continue
        X, y = normed
        n_ctx = min(n_context, len(X) - 1)
        if n_ctx < 2 or len(X) - n_ctx < 1:
            continue
        X_ctx, y_ctx = X[:n_ctx], y[:n_ctx]
        X_tst = X[n_ctx:]
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
                y_mean = data['sorted_respMean'][:, emg_idx]
                scaler_y = StandardScaler()
                y = scaler_y.fit_transform(
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
        # Encoder output: (n_train + n_test, n_features, hidden_dim)
        # Flatten per-point: (n_points, n_features * hidden_dim)
        n_points = act.shape[0]
        activations['target'] = act.detach().cpu().reshape(n_points, -1).float()

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
                         n_synthetic=500, n_context=50, regularization=1e-5,
                         seed=42):
    """Full Mahalanobis pipeline.

    Phase 1: Build reference embedding distribution from synthetic data.
    Phase 2: Compute D_M for neurostim data against each reference.

    Automatically falls back to CPU if CUDA errors are encountered during
    embedding extraction (forward hooks are more sensitive to CUDA issues
    than regular inference).

    Args:
        dataset_types: list of dataset names
        device: 'cpu' or 'cuda'
        prior_source: 'gp' | 'tabpfn_prior' | 'both'
        n_synthetic: number of synthetic datasets for reference
        n_context: context size for embedding extraction
        regularization: Tikhonov regularization for covariance inversion
        seed: random seed

    Returns:
        dict with per-dataset Mahalanobis distances + reference statistics.
    """
    try:
        return _mahalanobis_analysis_inner(
            dataset_types, device=device, prior_source=prior_source,
            n_synthetic=n_synthetic, n_context=n_context,
            regularization=regularization, seed=seed,
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
            regularization=regularization, seed=seed,
        )


def _mahalanobis_analysis_inner(
    dataset_types, device, prior_source, n_synthetic, n_context,
    regularization, seed,
):
    """Core Mahalanobis implementation (called by mahalanobis_analysis)."""
    model = TabPFNRegressor(device=device)
    # Use single estimator for speed
    model.n_estimators = 1

    # Phase 1: Build reference embeddings
    ref_stats = {}

    if prior_source in ('gp', 'both'):
        gp_bank = generate_synthetic_gp_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        gp_embeds = _embeddings_from_bank(model, gp_bank, n_context)
        ref_stats['gp'] = _fit_reference(gp_embeds, regularization)

    if prior_source in ('tabpfn_prior', 'both'):
        prior_bank = generate_tabpfn_prior_bank(
            n_datasets=n_synthetic, n_features=2, seed=seed,
        )
        prior_embeds = _embeddings_from_bank(model, prior_bank, n_context)
        ref_stats['prior'] = _fit_reference(prior_embeds, regularization)

    # Noise baseline (always included)
    noise_bank = generate_noise_bank(
        n_datasets=n_synthetic, n_features=2, seed=seed + 10000,
    )
    noise_embeds = _embeddings_from_bank(model, noise_bank, n_context)
    ref_stats['noise'] = _fit_reference(noise_embeds, regularization)

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
                y_mean = data['sorted_respMean'][:, emg_idx]
                scaler_y = StandardScaler()
                y = scaler_y.fit_transform(
                    y_mean.reshape(-1, 1),
                ).ravel().astype(np.float32)

                n_ctx = min(n_context, len(X) - 1)
                X_ctx, y_ctx = X[:n_ctx], y[:n_ctx]
                X_tst = X[n_ctx:]

                if len(X_tst) == 0:
                    continue

                embeddings = extract_embeddings_frozen(
                    model, X_ctx, y_ctx, X_tst,
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

    # Include reference self-distances for comparison
    results['ref_stats'] = {}
    for ref_name, stats in ref_stats.items():
        results['ref_stats'][ref_name] = {
            'self_distances': stats['self_distances'],
            'mean_self_dist': float(np.mean(stats['self_distances'])),
        }

    return results


class _CudaError(RuntimeError):
    """CUDA errors corrupt the device context and are unrecoverable."""
    pass


def _is_cuda_error(exc):
    """Check if an exception is a CUDA error (unrecoverable)."""
    msg = str(exc).lower()
    return 'cuda' in msg or 'cublas' in msg or 'cudnn' in msg


def _embeddings_from_bank(model, bank, n_context):
    """Extract embeddings for all datasets in a bank."""
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
            embeds = extract_embeddings_frozen(model, X_ctx, y_ctx, X_tst)
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


def _fit_reference(embeddings, regularization, max_pca_components=50):
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
#  3d. Unified Runner
# ============================================================================

def run_id_ood_analysis(dataset_types=None, analyses=None,
                        prior_source='both', device='cpu',
                        n_synthetic=500, n_context=50, seed=42,
                        save=False, output_dir=None):
    """Orchestrate all ID/OOD analyses.

    Generates synthetic bank(s) once, shared across analyses.

    Args:
        dataset_types: list of dataset names (default: ['rat', 'nhp'])
        analyses: list of analysis names (default: ['entropy', 'mmd', 'mahalanobis'])
        prior_source: 'gp' | 'tabpfn_prior' | 'both'
        device: 'cpu' or 'cuda'
        n_synthetic: number of synthetic datasets
        n_context: context size for entropy/Mahalanobis
        seed: random seed
        save: whether to save results to disk
        output_dir: base output directory

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
            regularization=1e-5, seed=seed,
        )
        all_results['mahalanobis'] = mahalanobis_results

        if save:
            out = os.path.join(output_dir, 'mahalanobis')
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, 'mahalanobis_results.pkl'), 'wb') as f:
                pickle.dump(mahalanobis_results, f)
            print(f"Saved Mahalanobis results -> {out}")

    return all_results
