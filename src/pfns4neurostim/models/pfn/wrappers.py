"""Per-model wrappers for the five external amortized surrogates (task #8 Step 3).

One class per benchmark model, each reaching its **official implementation**
through :mod:`~pfns4neurostim.models.pfn.external`. The classes exist and are
registered now so that configs, the registry and the contract tests can name them;
their bodies raise :class:`NotImplementedError` with the specific work that
remains, which is per-model and not uniform:

* **PFNs4BO** — backend already installed (``pfns4bo`` 0.1.5, weights vendored in
  ``libs/PFNs4BO``). Remaining work is mapping our ``(X_pool, observations)``
  layout onto its transformer input and reading back its bar distribution.
* **TabPFN v1** — needs ``tabpfn<2``, which cannot coexist with the pinned
  ``tabpfn==6.3.2``; it therefore needs its own environment (task #8 Step 1).
  Classification-only, so it uses the bucketized adapter.
* **TabFM** — released 2026-06-30 with an sklearn-style API; its Python/JAX
  requirements must first be checked against Python 3.9.25 / torch 2.5.1.
* **Mitra** — inside AutoGluon >= 1.4 (heavy install). Open question: whether its
  API exposes a predictive *distribution*; without one it can only serve greedy
  acquisition, which would make its Hyp 0 row incomparable.
* **TabFlex** — in ``microsoft/ticl``, to be added as ``libs/ticl``.
  Classification-focused upstream, so it uses the bucketized adapter.
* **TabICL** — ``soda-inria/tabicl`` (pip ``tabicl``). Added 2026-09-20 on
  request; it was not in the original H0-1 table. Registered on the bucketized
  route because upstream is classification-focused, which must be re-checked
  against the installed version before any TabICL number is reported.

Rows produced by the two adapter-based models must be labelled
**"classification-head adaptation"** in every table and caption.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .external import BucketizedClassifierSurrogate, ExternalSurrogate

__all__ = [
    "PFNs4BOSurrogate",
    "TabICLSurrogate",
    "TabPFNv1Surrogate",
    "TabFMSurrogate",
    "MitraSurrogate",
    "TabFlexSurrogate",
]


class PFNs4BOSurrogate(ExternalSurrogate):
    """PFNs4BO (HEBO prior) surrogate over a discrete candidate pool.

    Args:
        device: Torch device string.
        model_name: Which vendored checkpoint to load.
        **backend_kwargs: Forwarded to the backend.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "hebo_morebudget_9_unused_features_3",
        **backend_kwargs: Any,
    ) -> None:
        super().__init__("pfns4bo", device=device, **backend_kwargs)
        self.model_name = model_name

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Store the context for the PFN forward pass. **Not implemented.**

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].

        Raises:
            NotImplementedError: Task #8 Step 3.
        """
        raise NotImplementedError(
            "PFNs4BOSurrogate is not implemented yet (task #8 Step 3). The backend is "
            "installed; what remains is mapping our (X_pool, observations) layout onto "
            "pfns4bo's transformer input and reading back its bar distribution."
        )


class TabPFNv1Surrogate(BucketizedClassifierSurrogate):
    """TabPFN v1 as a bucketized regressor (classification-head adaptation).

    Args:
        device: Torch device string.
        n_bins: Number of response bins.
        **backend_kwargs: Forwarded to ``TabPFNClassifier``.
    """

    def __init__(self, device: str = "cpu", n_bins: int = 32, **backend_kwargs: Any) -> None:
        super().__init__("tabpfn_v1", device=device, n_bins=n_bins, **backend_kwargs)

    def _make_classifier(self) -> Any:
        """Construct the v1 classifier. **Not implemented.**

        Raises:
            NotImplementedError: Task #8 Step 3; needs the isolated ``tabpfn<2`` extra.
        """
        raise NotImplementedError(
            "TabPFNv1Surrogate is not implemented yet (task #8 Step 3): it needs the "
            "tabpfn<2 extra, which conflicts with the pinned tabpfn 6.3.2, so it must "
            "run in its own environment."
        )


class TabFMSurrogate(ExternalSurrogate):
    """Google TabFM regression surrogate (native regression).

    Args:
        device: Torch device string.
        **backend_kwargs: Forwarded to the backend regressor.
    """

    def __init__(self, device: str = "cpu", **backend_kwargs: Any) -> None:
        super().__init__("tabfm", device=device, **backend_kwargs)

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the TabFM regressor. **Not implemented.**

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].

        Raises:
            NotImplementedError: Task #8 Step 3.
        """
        raise NotImplementedError(
            "TabFMSurrogate is not implemented yet (task #8 Step 3); first check its "
            "Python/JAX requirements against Python 3.9.25 / torch 2.5.1."
        )


class MitraSurrogate(ExternalSurrogate):
    """Mitra regression surrogate (native regression, via AutoGluon).

    Args:
        device: Torch device string.
        **backend_kwargs: Forwarded to the backend regressor.
    """

    def __init__(self, device: str = "cpu", **backend_kwargs: Any) -> None:
        super().__init__("mitra", device=device, **backend_kwargs)

    def _fit_backend(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Mitra regressor. **Not implemented.**

        Args:
            X: Observed coordinates, shape [n, D].
            y: Observed responses, shape [n].

        Raises:
            NotImplementedError: Task #8 Step 3; predictive-distribution access
                must be confirmed first.
        """
        raise NotImplementedError(
            "MitraSurrogate is not implemented yet (task #8 Step 3); confirm first that "
            "AutoGluon exposes a predictive distribution and not only a point prediction."
        )


class TabFlexSurrogate(BucketizedClassifierSurrogate):
    """TabFlex as a bucketized regressor (classification-head adaptation).

    Args:
        device: Torch device string.
        n_bins: Number of response bins.
        **backend_kwargs: Forwarded to the backend classifier.
    """

    def __init__(self, device: str = "cpu", n_bins: int = 32, **backend_kwargs: Any) -> None:
        super().__init__("tabflex", device=device, n_bins=n_bins, **backend_kwargs)

    def _make_classifier(self) -> Any:
        """Construct the TabFlex classifier. **Not implemented.**

        Raises:
            NotImplementedError: Task #8 Step 3; needs ``libs/ticl`` as a submodule.
        """
        raise NotImplementedError(
            "TabFlexSurrogate is not implemented yet (task #8 Step 3); add libs/ticl "
            "as a submodule first."
        )


class TabICLSurrogate(BucketizedClassifierSurrogate):
    """TabICL as a bucketized regressor (classification-head adaptation, provisional).

    Args:
        device: Torch device string.
        n_bins: Number of response bins.
        **backend_kwargs: Forwarded to the backend classifier.
    """

    def __init__(self, device: str = "cpu", n_bins: int = 32, **backend_kwargs: Any) -> None:
        super().__init__("tabicl", device=device, n_bins=n_bins, **backend_kwargs)

    def _make_classifier(self) -> Any:
        """Construct the TabICL classifier. **Not implemented.**

        Raises:
            NotImplementedError: Task #8 Step 3. Confirm first whether the
                installed version exposes a native regressor: if it does, this
                model should not go through the bucketized adapter at all.
        """
        raise NotImplementedError(
            "TabICLSurrogate is not implemented yet (task #8 Step 3). Before "
            "implementing, check whether the installed tabicl exposes a native "
            "regressor - if so, drop the bucketized route for it."
        )
