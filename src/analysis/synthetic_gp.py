"""
Pure GP prior generation for ID/OOD reference distributions.

Generates synthetic datasets from Gaussian Process priors as a simple
baseline reference for comparing against neurostim data distributions.
"""
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler


def generate_synthetic_gp_dataset(
    n_features=2, n_samples=100, kernel_type='rbf',
    lengthscale=0.3, outputscale=1.0, noise_std=0.1, seed=0,
):
    """Single (X, y) from a GP prior.

    X: uniform [0,1]^n_features.
    y: GP sample + noise, StandardScaler-normalized.

    Args:
        n_features: input dimensionality
        n_samples: number of data points
        kernel_type: 'rbf', 'matern32', or 'matern52'
        lengthscale: kernel lengthscale
        outputscale: kernel output scale
        noise_std: observation noise standard deviation
        seed: random seed

    Returns:
        (X, y) tuple of numpy arrays, y is StandardScaler-normalized.
    """
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features).astype(np.float64)

    X_torch = torch.tensor(X, dtype=torch.float64)

    # Build kernel
    if kernel_type == 'rbf':
        base_kernel = gpytorch.kernels.RBFKernel().double()
    elif kernel_type == 'matern32':
        base_kernel = gpytorch.kernels.MaternKernel(nu=1.5).double()
    elif kernel_type == 'matern52':
        base_kernel = gpytorch.kernels.MaternKernel(nu=2.5).double()
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    base_kernel.lengthscale = lengthscale
    kernel = gpytorch.kernels.ScaleKernel(base_kernel).double()
    kernel.outputscale = outputscale

    # Compute covariance and sample
    with torch.no_grad():
        K = kernel(X_torch).evaluate()
        K += (noise_std ** 2 + 1e-6) * torch.eye(n_samples, dtype=torch.float64)

    torch.manual_seed(seed)
    dist = torch.distributions.MultivariateNormal(
        torch.zeros(n_samples, dtype=torch.float64), K,
    )
    y = dist.sample().numpy()

    # Normalize y to match neurostim preprocessing
    scaler = StandardScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    return X.astype(np.float32), y.astype(np.float32)


def generate_synthetic_gp_bank(
    n_datasets=500, n_features=2, n_samples=100, seed=42,
):
    """Bank of GP datasets with varied hyperparameters.

    Per dataset, randomly samples:
      - kernel: RBF (50%), Matern-3/2 (25%), Matern-5/2 (25%)
      - lengthscale: log-uniform [0.05, 1.0]
      - outputscale: log-uniform [0.1, 10.0]
      - noise_std: choice from [1e-5, 1e-4, 0.01] (matching TabPFN)
      - n_samples: randomly from {32, 64, 96, 100}

    Args:
        n_datasets: number of datasets to generate
        n_features: input dimensionality
        n_samples: ignored (varied per dataset), kept for API consistency
        seed: random seed

    Returns:
        List of (X, y) tuples.
    """
    rng = np.random.RandomState(seed)
    bank = []

    kernel_types = ['rbf', 'matern32', 'matern52']
    kernel_weights = [0.5, 0.25, 0.25]
    noise_choices = [1e-5, 1e-4, 0.01]
    sample_sizes = [32, 64, 96, 100]

    for i in range(n_datasets):
        kt = rng.choice(kernel_types, p=kernel_weights)
        ls = np.exp(rng.uniform(np.log(0.05), np.log(1.0)))
        os_ = np.exp(rng.uniform(np.log(0.1), np.log(10.0)))
        ns = noise_choices[rng.randint(len(noise_choices))]
        ns_ = sample_sizes[rng.randint(len(sample_sizes))]

        X, y = generate_synthetic_gp_dataset(
            n_features=n_features, n_samples=ns_,
            kernel_type=kt, lengthscale=ls, outputscale=os_,
            noise_std=ns, seed=seed + i,
        )
        bank.append((X, y))

    return bank
