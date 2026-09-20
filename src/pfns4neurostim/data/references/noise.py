"""
OOD noise generation for ID/OOD reference distributions.

Generates i.i.d. uniform noise datasets (X ~ U[0,1], y ~ U[0,1]) as a
maximally simple out-of-distribution upper bound for calibrating ID/OOD
metrics.

Design rationale
----------------
X ~ U[0,1]^d and y ~ U[0,1] are independent by construction, so TabPFN
receives zero X→y signal from the context.  The uniform y-marginal gives
TabPFN nothing to exploit via in-context density estimation either — uniform
is the maximum-entropy prior over a bounded range.  Together these push the
bar-distribution output toward its theoretical maximum entropy.

Seeding: generators accept an optional integer seed.  When seed is None
(default), the generator draws from system entropy so consecutive runs
produce independent realizations.  When seed is an int, the generator is
deterministic — enabling reproducibility from a logged top-level seed.
"""
from __future__ import annotations

import numpy as np


def generate_noise_dataset(
    n_features: int = 2,
    n_samples: int = 100,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single (X, y) i.i.d. uniform OOD noise dataset.

    Args:
        n_features: input dimensionality.
        n_samples: number of data points.
        seed: optional integer seed.  When None, draws from system entropy
            so consecutive calls produce independent realizations.

    Returns:
        (X, y) tuple of float32 arrays; X in [0, 1]^(n_samples, n_features),
        y in [0, 1]^n_samples.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    y = rng.uniform(0.0, 1.0, size=n_samples).astype(np.float32)
    return X, y


def generate_noise_bank(
    n_datasets: int = 500,
    n_features: int = 2,
    n_samples: int = 100,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Bank of i.i.d. uniform OOD noise datasets with varied sample sizes.

    Sample size per dataset is drawn uniformly from {32, 64, 96, 100} to
    match the GP bank convention.  Each dataset gets an independent seed
    derived from the bank-level RNG so individual datasets are reproducible
    when ``seed`` is supplied, and fully random when ``seed`` is None.

    Args:
        n_datasets: number of datasets to generate.
        n_features: input dimensionality.
        n_samples: ignored (varied per dataset), kept for API consistency.
        seed: optional integer seed for the bank-level RNG.  When None,
            draws from system entropy so consecutive analysis runs produce
            different noise realizations.

    Returns:
        List of (X, y) tuples.
    """
    rng = np.random.default_rng(seed)
    sample_sizes = [32, 64, 96, 100]
    bank = []

    for _ in range(n_datasets):
        ns = int(rng.choice(sample_sizes))
        ds_seed = int(rng.integers(0, 2**31))
        X, y = generate_noise_dataset(
            n_features=n_features, n_samples=ns, seed=ds_seed,
        )
        bank.append((X, y))

    return bank
