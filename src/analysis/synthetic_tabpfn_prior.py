"""
TabPFN Prior Bag reimplementation — GP + random MLP mixture.

Reimplements TabPFN v1's actual pre-training data generator so we can
build a reference distribution that matches what the model saw during
pre-training.  Source: https://github.com/automl/tabpfn-v1-prior

Prior Bag mixing: GP weight ~1.0, MLP weight ~2.0-10.0 (uniform),
then softmax → MLP dominates (~70-90%).
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from analysis.synthetic_gp import generate_synthetic_gp_dataset


# ---------------------------------------------------------------------------
#  Hyperparameter sampling (matching TabPFN's meta-learned distributions)
# ---------------------------------------------------------------------------

def _sample_hp_log_normal(rng, min_mean, max_mean):
    """Meta-learned truncated log-normal hyperparameter.

    log_mean ~ U(log(min_mean), log(max_mean))
    log_std  ~ U(log(0.01), log(1.0))
    hp = exp(Normal(log_mean, exp(log_std)))  (truncated to positive)
    """
    log_mean = rng.uniform(np.log(min_mean), np.log(max_mean))
    log_std = rng.uniform(np.log(0.01), np.log(1.0))
    sample = np.exp(log_mean + np.exp(log_std) * rng.randn())
    return max(sample, 1e-8)


def _sample_hp_gamma(rng, lower_bound, max_alpha, max_scale):
    """Gamma-distributed hyperparameter with lower bound.

    alpha ~ U(0.5, max_alpha), scale ~ U(0.5, max_scale)
    hp = lower_bound + Gamma(alpha, scale)
    """
    alpha = rng.uniform(0.5, max_alpha)
    scale = rng.uniform(0.5, max_scale)
    return lower_bound + rng.gamma(alpha, scale)


# ---------------------------------------------------------------------------
#  GP Component
# ---------------------------------------------------------------------------

def _generate_gp_dataset(n_features, n_samples, rng):
    """GP component of the Prior Bag.

    X ~ Uniform[0,1].  Kernel: ScaleKernel(RBF) with sampled outputscale
    and lengthscale.  Noise from [1e-5, 1e-4, 0.01].

    Returns:
        (X, y) tuple, y is StandardScaler-normalized.
    """
    lengthscale = _sample_hp_log_normal(rng, 1e-5, 10.0)
    outputscale = _sample_hp_log_normal(rng, 1e-5, 10.0)
    noise_std = rng.choice([1e-5, 1e-4, 0.01])

    seed = int(rng.randint(0, 2**31))
    return generate_synthetic_gp_dataset(
        n_features=n_features, n_samples=n_samples,
        kernel_type='rbf', lengthscale=lengthscale,
        outputscale=outputscale, noise_std=noise_std, seed=seed,
    )


# ---------------------------------------------------------------------------
#  MLP Component
# ---------------------------------------------------------------------------

def _random_mlp_forward(X_torch, n_layers, hidden_dim, init_std, noise_std,
                        dropout_prob, activation_fn, is_causal, rng):
    """Forward pass through a random (untrained) MLP.

    Weights ~ N(0, init_std), with Bernoulli dropout masking and
    per-layer Gaussian noise injection.

    Args:
        X_torch: (n_samples, n_features) float tensor
        n_layers: number of layers (>= 2)
        hidden_dim: hidden layer width
        init_std: weight initialization std
        noise_std: per-layer Gaussian noise std
        dropout_prob: dropout probability
        activation_fn: callable activation function
        is_causal: if True, intermediate activations become features (SCM)
        rng: numpy RandomState

    Returns:
        y as (n_samples,) numpy array, or (X_new, y) if is_causal.
    """
    n_samples, n_features = X_torch.shape

    # Build layers: input → hidden → ... → output(1)
    dims = [n_features] + [hidden_dim] * (n_layers - 1) + [1]

    h = X_torch
    causal_features = []

    for layer_i in range(len(dims) - 1):
        in_d, out_d = dims[layer_i], dims[layer_i + 1]

        # Random weights
        W = torch.tensor(
            rng.randn(in_d, out_d) * init_std, dtype=torch.float32,
        )
        b = torch.zeros(out_d)
        h = h @ W + b

        # Clamp to prevent overflow cascading through layers
        h = h.clamp(-1e6, 1e6)

        # Not the last layer
        if layer_i < len(dims) - 2:
            # Activation
            h = activation_fn(h)

            # Dropout (Bernoulli mask)
            if dropout_prob > 0:
                mask = torch.tensor(
                    rng.binomial(1, 1.0 - dropout_prob, size=h.shape),
                    dtype=torch.float32,
                )
                h = h * mask / max(1.0 - dropout_prob, 1e-8)

            # Per-layer Gaussian noise
            if noise_std > 0:
                noise = torch.tensor(
                    rng.randn(*h.shape) * noise_std, dtype=torch.float32,
                )
                h = h + noise

            # Save intermediate activations for causal mode
            if is_causal and layer_i < 3:
                causal_features.append(h.detach())

    y = h.squeeze(-1).detach().numpy()

    if is_causal and causal_features:
        # SCM structure: X = early hidden activations
        X_causal = torch.cat(causal_features, dim=-1)
        # Take first n_features columns
        X_causal = X_causal[:, :n_features].numpy().astype(np.float32)
        # Clamp to finite range
        X_causal = np.clip(np.nan_to_num(X_causal, nan=0.0, posinf=1e6, neginf=-1e6),
                           -1e6, 1e6)
        return X_causal, np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    return None, np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def _sample_feature_distribution(n_samples, n_features, rng):
    """Sample X from one of TabPFN's feature distributions.

    Options: uniform [0,1], standard normal, or mixed (per-feature random).
    """
    choice = rng.randint(3)
    if choice == 0:
        return rng.rand(n_samples, n_features).astype(np.float32)
    elif choice == 1:
        return rng.randn(n_samples, n_features).astype(np.float32)
    else:
        # Mixed: per-feature random distribution
        X = np.empty((n_samples, n_features), dtype=np.float32)
        for f in range(n_features):
            dist = rng.randint(3)
            if dist == 0:
                X[:, f] = rng.rand(n_samples)
            elif dist == 1:
                X[:, f] = rng.randn(n_samples)
            else:
                # Multinomial-like: discrete uniform
                n_cats = rng.randint(2, 10)
                X[:, f] = rng.randint(0, n_cats, size=n_samples).astype(np.float32)
        return X


def _generate_mlp_dataset(n_features, n_samples, rng):
    """MLP component of the Prior Bag.

    Samples all hyperparameters from meta-learned distributions, then:
    X ~ Uniform/Normal/Mixed. y = random_MLP(X) + noise.
    When is_causal=True, X = intermediate MLP activations (SCM).

    Returns:
        (X, y) tuple, y is StandardScaler-normalized.
    """
    # Sample hyperparameters from meta-learned distributions
    n_layers = max(2, int(round(_sample_hp_gamma(rng, 2, 2.0, 3.0))))
    n_layers = min(n_layers, 8)  # cap for stability
    hidden_dim = max(4, int(round(_sample_hp_gamma(rng, 4, 2.0, 100.0))))
    hidden_dim = min(hidden_dim, 256)  # cap for memory
    init_std = _sample_hp_log_normal(rng, 0.01, 10.0)
    noise_std = _sample_hp_log_normal(rng, 1e-4, 0.3)
    dropout_prob = min(rng.beta(1.0, 1.0) * 0.6, 0.9)

    activation_name = rng.choice(['tanh', 'identity', 'relu'])
    activation_map = {
        'tanh': torch.tanh,
        'identity': lambda x: x,
        'relu': torch.relu,
    }
    activation_fn = activation_map[activation_name]
    is_causal = rng.choice([True, False])

    # Generate features
    X = _sample_feature_distribution(n_samples, n_features, rng)
    X_torch = torch.tensor(X, dtype=torch.float32)

    X_causal, y = _random_mlp_forward(
        X_torch, n_layers, hidden_dim, init_std, noise_std,
        dropout_prob, activation_fn, is_causal, rng,
    )

    if is_causal and X_causal is not None:
        X = X_causal

    # Handle degenerate outputs
    if np.std(y) < 1e-10:
        y = y + rng.randn(len(y)).astype(np.float32) * 0.01

    # Normalize y
    scaler = StandardScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    return X, y


# ---------------------------------------------------------------------------
#  Prior Bag
# ---------------------------------------------------------------------------

def generate_tabpfn_prior_dataset(n_features=2, n_samples=100, seed=0):
    """Single (X, y) from TabPFN's Prior Bag.

    Randomly selects GP or MLP component (MLP ~70-90% probability via
    softmax weighting: GP_weight=1.0, MLP_weight~U(2,10)).

    Args:
        n_features: input dimensionality
        n_samples: number of data points
        seed: random seed

    Returns:
        (X, y) tuple, y is StandardScaler-normalized.
    """
    rng = np.random.RandomState(seed)

    # Prior Bag mixing weights
    gp_weight = 1.0
    mlp_weight = rng.uniform(2.0, 10.0)
    # Softmax
    weights = np.array([gp_weight, mlp_weight])
    probs = np.exp(weights - weights.max())
    probs = probs / probs.sum()

    if rng.rand() < probs[0]:
        return _generate_gp_dataset(n_features, n_samples, rng)
    else:
        return _generate_mlp_dataset(n_features, n_samples, rng)


def generate_tabpfn_prior_bank(n_datasets=500, n_features=2, n_samples=100, seed=42):
    """Bank of Prior Bag datasets.

    Each dataset independently samples GP vs MLP component and all
    hyperparameters from their respective meta-learned distributions.

    Args:
        n_datasets: number of datasets to generate
        n_features: input dimensionality
        n_samples: number of data points per dataset
        seed: random seed

    Returns:
        List of (X, y) tuples.
    """
    bank = []
    for i in range(n_datasets):
        X, y = generate_tabpfn_prior_dataset(
            n_features=n_features, n_samples=n_samples, seed=seed + i,
        )
        bank.append((X, y))
    return bank
