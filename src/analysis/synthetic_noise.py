"""
OOD noise generation for ID/OOD reference distributions.

Generates synthetic datasets that are clearly out-of-distribution relative
to TabPFN's training prior.  Provides an upper bound for calibrating
ID/OOD metrics.

Design rationale
----------------
Simple independent noise (X ~ U[0,1], y ~ N(0,1)) has *identical marginals*
to GP priors, making it invisible to MMD (raw feature space) and Mahalanobis
(encoder embeddings are X-driven).  The OOD bank mixes several distribution
families whose shapes remain distinguishable even after MinMaxScaler /
StandardScaler normalization applied by _normalize_for_tabpfn():

  clustered   — tight Gaussian clusters in feature space (non-uniform X)
  correlated  — features on a 1D manifold + bimodal y
  concentrated — Beta-distributed X (peaked, non-uniform density)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
#  Noise type generators — each returns raw (X, y) arrays.
#  _normalize_for_tabpfn() in id_ood.py handles scaling before TabPFN.
# ---------------------------------------------------------------------------

def _clustered_noise(n_samples, n_features, rng):
    """X in tight Gaussian clusters, y uniform.

    Produces a highly non-uniform spatial distribution that remains
    clustered after MinMaxScaler.  Targets Mahalanobis (different encoder
    embeddings) and MMD (different X marginal).
    """
    n_clusters = rng.randint(2, 6)
    centers = rng.rand(n_clusters, n_features)
    labels = rng.randint(0, n_clusters, size=n_samples)
    X = centers[labels] + rng.randn(n_samples, n_features) * 0.02
    y = rng.uniform(-3, 3, size=n_samples)
    return X, y


def _correlated_noise(n_samples, n_features, rng):
    """X features on a 1D manifold (highly correlated), y bimodal.

    Redundant features produce a different encoder representation.
    Bimodal y is non-Gaussian.  Targets all three metrics.
    """
    t = rng.rand(n_samples, 1)
    X = np.hstack([t + rng.randn(n_samples, 1) * 0.01
                   for _ in range(n_features)])
    component = rng.binomial(1, 0.5, size=n_samples)
    y = np.where(component,
                 rng.normal(2, 0.3, n_samples),
                 rng.normal(-2, 0.3, n_samples))
    return X, y


def _concentrated_noise(n_samples, n_features, rng):
    """X ~ Beta(a, b) with random concentration, y uniform.

    Beta distributions create a peaked, non-uniform density in feature
    space that persists after MinMaxScaler (shape is preserved, only
    range changes).  Targets Mahalanobis and MMD.
    """
    alpha = rng.uniform(5, 20)
    beta = rng.uniform(5, 20)
    X = rng.beta(alpha, beta, size=(n_samples, n_features))
    y = rng.uniform(-3, 3, size=n_samples)
    return X, y


_NOISE_TYPES = [_clustered_noise, _correlated_noise, _concentrated_noise]


# ---------------------------------------------------------------------------
#  Public API (matches synthetic_gp.py convention)
# ---------------------------------------------------------------------------

def generate_noise_dataset(n_features=2, n_samples=100, seed=0):
    """Single (X, y) OOD noise dataset.

    Randomly selects a noise type that produces distributions clearly
    distinguishable from GP/Prior even after normalization.
    y is StandardScaler-normalized (matching bank convention).

    Args:
        n_features: input dimensionality
        n_samples: number of data points
        seed: random seed

    Returns:
        (X, y) tuple of float32 numpy arrays.
    """
    rng = np.random.RandomState(seed)
    noise_fn = _NOISE_TYPES[rng.randint(len(_NOISE_TYPES))]
    X, y = noise_fn(n_samples, n_features, rng)

    scaler = StandardScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    return X.astype(np.float32), y.astype(np.float32)


def generate_noise_bank(n_datasets=500, n_features=2, n_samples=100, seed=42):
    """Bank of OOD noise datasets with varied types and sample sizes.

    Each dataset independently picks a noise type (clustered, correlated,
    or concentrated) and a sample size from {32, 64, 96, 100}.

    Args:
        n_datasets: number of datasets to generate
        n_features: input dimensionality
        n_samples: ignored (varied per dataset), kept for API consistency
        seed: random seed

    Returns:
        List of (X, y) tuples.
    """
    rng = np.random.RandomState(seed)
    sample_sizes = [32, 64, 96, 100]
    bank = []

    for i in range(n_datasets):
        ns = sample_sizes[rng.randint(len(sample_sizes))]
        X, y = generate_noise_dataset(
            n_features=n_features, n_samples=ns, seed=seed + i,
        )
        bank.append((X, y))

    return bank
