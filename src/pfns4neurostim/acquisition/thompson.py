"""Thompson sampling: the symmetric marginal headline and the GP-only joint reference.

Design decision (task #4, 2026-09-17, after the Phase-2 prototype):

* ``ts_marginal`` is the **headline** for both GP and PFN. Each model draws
  independently per site from its own predictive marginal — a GP from
  N(mu_i, sigma_i^2 + sigma_n^2), a PFN from its bar distribution. This is the
  same family of object for both, which is what makes the comparison symmetric.
* ``ts_joint`` is a **GP-only reference row**: one draw from the joint latent
  posterior, keeping the inter-site correlations the kernel implies. It measures
  how much the marginal approximation costs on the real grids.
* PFN joint TS (sequential fantasization) is **parked**. The Phase-2 prototype
  (``tests/shadow/_ts_prototype/``) showed it is not better than marginal TS on
  two of the three co-primary regrets, at 51-134x the per-step latency, and its
  fantasized joint is built from conditionals the network was never trained to
  keep consistent.

Both types live in :mod:`pfns4neurostim.acquisition.registry`; this module
re-exports them and documents the decision next to the code that implements it.
"""
from __future__ import annotations

from .registry import (
    TSJointParams,
    TSMarginalParams,
    _score_ts_joint,
    _score_ts_marginal,
)

__all__ = [
    "TSMarginalParams",
    "TSJointParams",
    "score_ts_marginal",
    "score_ts_joint",
]

#: Marginal Thompson sampling score (GP and PFN headline).
score_ts_marginal = _score_ts_marginal

#: Joint Thompson sampling score (GP only; raises for a PFN surrogate).
score_ts_joint = _score_ts_joint
