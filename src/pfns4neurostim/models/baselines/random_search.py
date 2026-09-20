"""Random-search baseline (task #1 Step 3).

Moved from the flat ``src/models/regressors.py``. Queries uniformly at random,
ignoring the data entirely: the lower bound every surrogate must beat. Its
"predictions" are placeholders, which is why the registry marks it as having no
predictive distribution and its calibration metrics are reported as not computed.
"""
from __future__ import annotations

import numpy as np

__all__ = ["RandomSearchSurrogate"]


# ---------------------------------------------------------------------------
# RandomSearchSurrogate — random-acquisition lower bound (§16, L2)
# ---------------------------------------------------------------------------

class RandomSearchSurrogate:
    """Random-search baseline conforming to the ``SurrogateModel`` protocol.

    Acquisition is uniformly random: ``predict_ucb`` / ``predict_ts`` return
    i.i.d. noise so ``argmax`` selects a random candidate each step. The
    ``predict`` readout used for the exploitation recommendation and the final
    R² is a 1-nearest-neighbour lookup over the points observed so far — the
    honest "best you can do with random queries and no model" reference.

    This is the true lower bound for the §16 baseline ladder: matching random
    search would falsify any claim that either surrogate is learning structure.

    Args:
        seed: Optional seed for the internal RNG (independent of global seeds
            so the acquisition stream is reproducible per surrogate instance).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._X: Optional[np.ndarray] = None  # [N, D] observed
        self._y: Optional[np.ndarray] = None  # [N]    observed

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store observed points for the nearest-neighbour readout.

        Args:
            X: Observed feature matrix, shape [N, D].
            y: Observed targets, shape [N].
        """
        if np.isnan(X).any() or np.isnan(y).any():
            raise RuntimeError("RandomSearchSurrogate.fit received NaN inputs.")
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a 1-NN readout over observed points and zero std.

        Args:
            X: Query feature matrix, shape [M, D].

        Returns:
            Tuple of (mean, std), each shape [M].  ``mean[j]`` is the observed
            target of the nearest observed point to ``X[j]``; ``std`` is all
            zeros (random search carries no calibrated uncertainty).
        """
        if self._X is None or self._y is None:
            raise RuntimeError("RandomSearchSurrogate.predict called before fit.")
        Xq = np.asarray(X, dtype=np.float64)                     # [M, D]
        # Pairwise squared distances query→observed, take nearest observed.
        d2 = ((Xq[:, None, :] - self._X[None, :, :]) ** 2).sum(-1)  # [M, N]
        nn = np.argmin(d2, axis=1)                                # [M]
        mean = self._y[nn]                                        # [M]
        return mean, np.zeros_like(mean)                         # [M], [M]

    def predict_ucb(
        self, X: np.ndarray, kappa: float, t: int, n_steps: int,
    ) -> np.ndarray:
        """Return i.i.d. random acquisition values (uniform random selection).

        Args:
            X: Candidate feature matrix, shape [M, D].
            kappa: Unused (random search ignores exploration coefficients).
            t: Unused.
            n_steps: Unused.

        Returns:
            Random values, shape [M].
        """
        return self._rng.standard_normal(X.shape[0])  # [M]

    def predict_ts(self, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Return i.i.d. random acquisition values (uniform random selection).

        Args:
            X: Candidate feature matrix, shape [M, D].
            temperature: Unused.

        Returns:
            Random values, shape [M].
        """
        return self._rng.standard_normal(X.shape[0])  # [M]


