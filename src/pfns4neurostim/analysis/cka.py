"""Centred kernel alignment (task #1 Step 3; redesign pending in task #5).

``linear_cka`` is the **biased** estimator moved verbatim from
``src/models/regressors.py``. Task #5 replaces it with a debiased CKA built on the
unbiased HSIC estimator (Song et al., 2012) and keeps this one only as
``biased_linear_cka`` for comparison; until then it is used as-is and its results
are provisional.
"""
from __future__ import annotations

import numpy as np

__all__ = ["linear_cka"]


def linear_cka(X, Y):
    """Linear CKA between activation matrices X, Y of shape [n, d]."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = (Y.T @ X).norm() ** 2
    hsic_xx = (X.T @ X).norm()
    hsic_yy = (Y.T @ Y).norm()
    return (hsic_xy / (hsic_xx * hsic_yy + 1e-10)).item()


