"""
TabPFN Prior Bag — uses the official tabpfn-v1-prior submodule.

Generates synthetic datasets from the exact prior distribution used during
TabPFN v1's meta-training (GP + MLP mixture via Prior Bag).

Source: https://github.com/automl/tabpfn-v1-prior  (libs/tabpfn-v1-prior)

Prior Bag mixing: GP and MLP components are weighted via softmax, with
MLP dominating (~70-90%) — matching TabPFN v1's pretraining distribution.
"""
import os
import random
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

# Add the tabpfn-v1-prior submodule to the path
_PRIOR_LIB = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'tabpfn-v1-prior'),
)
if _PRIOR_LIB not in sys.path:
    sys.path.insert(0, _PRIOR_LIB)

from tabpfn_prior import build_tabpfn_prior  # noqa: E402


def generate_tabpfn_prior_dataset(n_features=2, n_samples=100, seed=0):
    """Single (X, y) from TabPFN's Prior Bag.

    Uses the official tabpfn-v1-prior implementation with the prior_bag
    prior type (GP + MLP mixture, matching pretraining distribution).

    Args:
        n_features: input dimensionality
        n_samples: number of data points
        seed: random seed

    Returns:
        (X, y) tuple — X float32, y StandardScaler-normalized float32.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    prior = build_tabpfn_prior(
        prior_type='prior_bag',
        num_steps=1,
        batch_size=1,
        num_datapoints_max=n_samples,
        num_features=n_features,
        max_num_classes=0,
        device='cpu',
        flexible=False,
        differentiable=False,
    )

    batch = next(iter(prior))
    X = batch['x'].squeeze(0).numpy().astype(np.float32)
    y = batch['y'].squeeze(0).numpy().astype(np.float32)

    # Handle degenerate constant outputs
    if np.std(y) < 1e-10:
        rng = np.random.RandomState(seed)
        y = y + rng.randn(len(y)).astype(np.float32) * 0.01

    # Normalize y (matching convention used by preprocess_neural_data)
    scaler = StandardScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

    return X, y


def generate_tabpfn_prior_bank(n_datasets=500, n_features=2,
                                n_samples=100, seed=None):
    """Bank of Prior Bag datasets.

    Each dataset independently samples the GP vs MLP component and all
    hyperparameters from the v1 prior's internal distributions.

    Args:
        n_datasets: number of datasets to generate
        n_features: input dimensionality
        n_samples: number of data points per dataset
        seed: optional integer seed.  When None (default), draws from system
            entropy so consecutive calls produce independent realizations.

    Returns:
        List of (X, y) tuples.
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))
    bank = []
    for i in range(n_datasets):
        X, y = generate_tabpfn_prior_dataset(
            n_features=n_features, n_samples=n_samples, seed=seed + i,
        )
        bank.append((X, y))
    return bank
