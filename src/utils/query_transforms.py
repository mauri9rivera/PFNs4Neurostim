"""
Query data transforms for synthetic function outputs.

Provides transforms (e.g., z-score normalization) that can be applied to
synthetic function outputs to normalize/scale values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy import stats
from scipy.special import inv_boxcox
from torch import Tensor


class QueryDataTransform(ABC):
    """
    Abstract base class for synthetic function output transforms.

    Transforms are applied to query data (function outputs) to normalize
    or scale values. All transforms must be invertible and serializable.

    :ivar fitted: Whether the transform has been fitted to data.
    """

    def __init__(self) -> None:
        self._fitted: bool = False

    @property
    def fitted(self) -> bool:
        """Whether transform has been fitted to data."""
        return self._fitted

    @abstractmethod
    def fit(self, data: Tensor) -> QueryDataTransform:
        """
        Fit transform parameters from data.

        :param data: Target values tensor, shape (N,), (N, 1), or (N, D).
        :return: Self for method chaining.
        """
        pass

    @abstractmethod
    def transform(self, data: Tensor) -> Tensor:
        """
        Apply forward transform to data.

        :param data: Values to transform.
        :return: Transformed values (same shape).
        :raises RuntimeError: If transform not fitted.
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: Tensor) -> Tensor:
        """
        Apply inverse transform to recover original scale.

        :param data: Transformed values.
        :return: Values in original scale.
        :raises RuntimeError: If transform not fitted.
        """
        pass

    def fit_transform(self, data: Tensor) -> Tensor:
        """
        Fit and transform in one step.

        :param data: Values to fit and transform.
        :return: Transformed values.
        """
        return self.fit(data).transform(data)

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize transform parameters to dictionary.

        :return: Dictionary with transform parameters.
        :raises RuntimeError: If transform not fitted.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> QueryDataTransform:
        """
        Reconstruct transform from serialized dictionary.

        :param d: Dictionary with transform parameters.
        :return: Reconstructed transform instance.
        """
        pass

    def __call__(self, data: Tensor) -> Tensor:
        """
        Apply transform (alias for transform method).

        :param data: Values to transform.
        :return: Transformed values.
        """
        return self.transform(data)


class ZScoreTransform(QueryDataTransform):
    """
    Z-score (standardization) transform for query data.

    Transforms data to zero mean and unit variance::

        z = (x - mean) / (std + epsilon)

    :ivar mean: Fitted mean value.
    :ivar std: Fitted standard deviation.
    :ivar epsilon: Small constant for numerical stability.
    """

    def __init__(self, epsilon: float = 1e-8, name: str = "zscore") -> None:
        """
        Initialize Z-score transform.

        :param epsilon: Added to std to prevent division by zero.
        :param name: Transform name for identification.
        """
        super().__init__()
        self.epsilon = epsilon
        self.name = name
        self._mean: Optional[Tensor] = None
        self._std: Optional[Tensor] = None

    @property
    def mean(self) -> float:
        """Fitted mean value."""
        if self._mean is None:
            raise RuntimeError("Transform not fitted")
        return self._mean.item()

    @property
    def std(self) -> float:
        """Fitted standard deviation."""
        if self._std is None:
            raise RuntimeError("Transform not fitted")
        return self._std.item()

    def fit(self, data: Tensor) -> ZScoreTransform:
        """
        Compute mean and std from data.

        :param data: Target values, shape (N,) or (N, 1).
        :return: Self for method chaining.
        """
        flat = data.flatten().to(torch.float64)
        self._mean = flat.mean()
        self._std = flat.std()
        self._fitted = True
        return self

    def transform(self, data: Tensor) -> Tensor:
        """
        Apply z-score normalization.

        :param data: Values to transform.
        :return: Standardized values: (x - mean) / (std + epsilon).
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Transform must be fitted before use")
        return (data - self._mean) / (self._std + self.epsilon)

    def inverse_transform(self, data: Tensor) -> Tensor:
        """
        Reverse z-score normalization.

        :param data: Standardized values.
        :return: Original-scale values: x * (std + epsilon) + mean.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Transform must be fitted before use")
        return data * (self._std + self.epsilon) + self._mean

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for saving.

        :return: Dictionary with transform parameters.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Cannot serialize unfitted transform")
        return {
            "type": "ZScoreTransform",
            "name": self.name,
            "mean": self._mean.cpu(),
            "std": self._std.cpu(),
            "epsilon": torch.tensor(self.epsilon),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ZScoreTransform:
        """
        Reconstruct from serialized dictionary.

        :param d: Dictionary with transform parameters.
        :return: Reconstructed ZScoreTransform instance.
        """
        transform = cls(
            epsilon=d["epsilon"].item(),
            name=d.get("name", "zscore"),
        )
        transform._mean = d["mean"]
        transform._std = d["std"]
        transform._fitted = True
        return transform


class MinMaxTransform(QueryDataTransform):
    """
    Min-max scaling to configurable bounds.

    Transforms data to a target range::

        x_scaled = (x - x_min) / (x_max - x_min) * (upper - lower) + lower

    Supports multi-dimensional inputs with per-dimension scaling.

    :ivar lower: Target range lower bound.
    :ivar upper: Target range upper bound.
    :ivar epsilon: Small constant for numerical stability.
    :ivar data_min: Fitted per-dimension minimum values.
    :ivar data_max: Fitted per-dimension maximum values.
    """

    def __init__(
        self,
        lower: float = -1.0,
        upper: float = 1.0,
        epsilon: float = 1e-8,
        name: str = "minmax",
    ) -> None:
        """
        Initialize Min-max transform.

        :param lower: Target range lower bound.
        :param upper: Target range upper bound.
        :param epsilon: Added to range to prevent division by zero.
        :param name: Transform name for identification.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.epsilon = epsilon
        self.name = name
        self._data_min: Optional[Tensor] = None
        self._data_max: Optional[Tensor] = None

    @property
    def data_min(self) -> Tensor:
        """Fitted per-dimension minimum values."""
        if self._data_min is None:
            raise RuntimeError("Transform not fitted")
        return self._data_min

    @property
    def data_max(self) -> Tensor:
        """Fitted per-dimension maximum values."""
        if self._data_max is None:
            raise RuntimeError("Transform not fitted")
        return self._data_max

    def fit(self, data: Tensor) -> MinMaxTransform:
        """
        Compute min and max from data per dimension.

        :param data: Values tensor, shape (N,), (N, 1), or (N, D).
        :return: Self for method chaining.
        """
        data = data.to(torch.float64)
        if data.ndim == 1:
            data = data.unsqueeze(-1)  # [N] -> [N, 1]

        self._data_min = data.min(dim=0).values  # [D]
        self._data_max = data.max(dim=0).values  # [D]
        self._fitted = True
        return self

    def transform(self, data: Tensor) -> Tensor:
        """
        Apply min-max scaling.

        :param data: Values to transform.
        :return: Scaled values in [lower, upper] range.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Transform must be fitted before use")

        original_shape = data.shape
        if data.ndim == 1:
            data = data.unsqueeze(-1)

        data_range = self._data_max - self._data_min
        # Avoid division by zero for constant dimensions
        data_range = torch.where(
            data_range > self.epsilon,
            data_range,
            torch.ones_like(data_range),
        )

        normalized = (data - self._data_min) / data_range
        scaled = normalized * (self.upper - self.lower) + self.lower

        if len(original_shape) == 1:
            scaled = scaled.squeeze(-1)

        return scaled

    def inverse_transform(self, data: Tensor) -> Tensor:
        """
        Reverse min-max scaling to original scale.

        :param data: Scaled values.
        :return: Values in original scale.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Transform must be fitted before use")

        original_shape = data.shape
        if data.ndim == 1:
            data = data.unsqueeze(-1)

        data_range = self._data_max - self._data_min
        data_range = torch.where(
            data_range > self.epsilon,
            data_range,
            torch.ones_like(data_range),
        )

        normalized = (data - self.lower) / (self.upper - self.lower)
        original = normalized * data_range + self._data_min

        if len(original_shape) == 1:
            original = original.squeeze(-1)

        return original

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for saving.

        :return: Dictionary with transform parameters.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Cannot serialize unfitted transform")
        return {
            "type": "MinMaxTransform",
            "name": self.name,
            "lower": torch.tensor(self.lower),
            "upper": torch.tensor(self.upper),
            "epsilon": torch.tensor(self.epsilon),
            "data_min": self._data_min.cpu(),
            "data_max": self._data_max.cpu(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MinMaxTransform:
        """
        Reconstruct from serialized dictionary.

        :param d: Dictionary with transform parameters.
        :return: Reconstructed MinMaxTransform instance.
        """
        transform = cls(
            lower=d["lower"].item(),
            upper=d["upper"].item(),
            epsilon=d["epsilon"].item(),
            name=d.get("name", "minmax"),
        )
        transform._data_min = d["data_min"]
        transform._data_max = d["data_max"]
        transform._fitted = True
        return transform


class YeoJohnsonTransform(QueryDataTransform):
    """
    Yeo-Johnson power transform.

    Transforms data to approximate normal distribution. Handles both
    positive and negative values (unlike Box-Cox).

    Uses scipy.stats.yeojohnson to fit lambda parameter per dimension
    for multivariate data.

    :ivar lambdas_: Fitted lambda parameters per dimension.
    """

    def __init__(self, name: str = "yeojohnson") -> None:
        """
        Initialize Yeo-Johnson transform.

        :param name: Transform name for identification.
        """
        super().__init__()
        self.name = name
        self._lambdas: Optional[Tensor] = None

    @property
    def lambdas_(self) -> Tensor:
        """Fitted lambda parameters per dimension."""
        if self._lambdas is None:
            raise RuntimeError("Transform not fitted")
        return self._lambdas

    def fit(self, data: Tensor) -> YeoJohnsonTransform:
        """
        Fit Yeo-Johnson lambda parameters from data.

        :param data: Values tensor, shape (N,), (N, 1), or (N, D).
        :return: Self for method chaining.
        """
        data_np = data.cpu().numpy()
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)  # [N] -> [N, 1]

        D = data_np.shape[1]
        lambdas = []

        for d in range(D):
            _, lmbda = stats.yeojohnson(data_np[:, d])
            lambdas.append(lmbda)

        self._lambdas = torch.tensor(lambdas, dtype=torch.float64)
        self._fitted = True
        return self

    def transform(self, data: Tensor) -> Tensor:
        """
        Apply Yeo-Johnson transform.

        :param data: Values to transform.
        :return: Transformed values approximating normal distribution.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Transform must be fitted before use")

        original_shape = data.shape
        data_np = data.cpu().numpy()
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        D = data_np.shape[1]
        transformed = np.zeros_like(data_np)

        for d in range(D):
            transformed[:, d] = stats.yeojohnson(
                data_np[:, d], lmbda=self._lambdas[d].item()
            )

        result = torch.tensor(transformed, dtype=data.dtype, device=data.device)
        if len(original_shape) == 1:
            result = result.squeeze(-1)

        return result

    def inverse_transform(self, data: Tensor) -> Tensor:
        """
        Reverse Yeo-Johnson transform.

        :param data: Transformed values.
        :return: Values in original scale.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Transform must be fitted before use")

        original_shape = data.shape
        data_np = data.cpu().numpy()
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        D = data_np.shape[1]
        original = np.zeros_like(data_np)

        for d in range(D):
            original[:, d] = _yeojohnson_inverse(
                data_np[:, d], self._lambdas[d].item()
            )

        result = torch.tensor(original, dtype=data.dtype, device=data.device)
        if len(original_shape) == 1:
            result = result.squeeze(-1)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for saving.

        :return: Dictionary with transform parameters.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Cannot serialize unfitted transform")
        return {
            "type": "YeoJohnsonTransform",
            "name": self.name,
            "lambdas": self._lambdas.cpu(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> YeoJohnsonTransform:
        """
        Reconstruct from serialized dictionary.

        :param d: Dictionary with transform parameters.
        :return: Reconstructed YeoJohnsonTransform instance.
        """
        transform = cls(name=d.get("name", "yeojohnson"))
        transform._lambdas = d["lambdas"]
        transform._fitted = True
        return transform


def _yeojohnson_inverse(y: np.ndarray, lmbda: float) -> np.ndarray:
    """
    Compute inverse Yeo-Johnson transform.

    Implements inverse of Yeo-Johnson power transform.

    :param y: Transformed values.
    :param lmbda: Lambda parameter.
    :return: Original scale values.
    """
    x = np.zeros_like(y)

    # For y >= 0
    pos_mask = y >= 0
    if lmbda == 0:
        x[pos_mask] = np.exp(y[pos_mask]) - 1
    else:
        x[pos_mask] = np.power(y[pos_mask] * lmbda + 1, 1 / lmbda) - 1

    # For y < 0
    neg_mask = ~pos_mask
    if lmbda == 2:
        x[neg_mask] = 1 - np.exp(-y[neg_mask])
    else:
        x[neg_mask] = 1 - np.power(-(2 - lmbda) * y[neg_mask] + 1, 1 / (2 - lmbda))

    return x


class BoxCoxTransform(QueryDataTransform):
    """
    Box-Cox power transform.

    Transforms data to approximate normal distribution. Requires strictly
    positive values (use YeoJohnsonTransform for data with negative values).

    Uses scipy.stats.boxcox to fit lambda parameter per dimension
    for multivariate data.

    :ivar lambdas_: Fitted lambda parameters per dimension.
    """

    def __init__(self, name: str = "boxcox") -> None:
        """
        Initialize Box-Cox transform.

        :param name: Transform name for identification.
        """
        super().__init__()
        self.name = name
        self._lambdas: Optional[Tensor] = None

    @property
    def lambdas_(self) -> Tensor:
        """Fitted lambda parameters per dimension."""
        if self._lambdas is None:
            raise RuntimeError("Transform not fitted")
        return self._lambdas

    def fit(self, data: Tensor) -> BoxCoxTransform:
        """
        Fit Box-Cox lambda parameters from data.

        :param data: Values tensor, shape (N,), (N, 1), or (N, D).
            All values must be strictly positive.
        :return: Self for method chaining.
        :raises ValueError: If any dimension contains non-positive values.
        """
        data_np = data.cpu().numpy()
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)  # [N] -> [N, 1]

        D = data_np.shape[1]
        lambdas = []

        for d in range(D):
            if np.any(data_np[:, d] <= 0):
                raise ValueError(
                    f"Box-Cox requires strictly positive values. "
                    f"Dimension {d} contains non-positive values. "
                    f"Use YeoJohnsonTransform for data with negative values."
                )
            _, lmbda = stats.boxcox(data_np[:, d])
            lambdas.append(lmbda)

        self._lambdas = torch.tensor(lambdas, dtype=torch.float64)
        self._fitted = True
        return self

    def transform(self, data: Tensor) -> Tensor:
        """
        Apply Box-Cox transform.

        :param data: Values to transform (must be positive).
        :return: Transformed values approximating normal distribution.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Transform must be fitted before use")

        original_shape = data.shape
        data_np = data.cpu().numpy()
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        D = data_np.shape[1]
        transformed = np.zeros_like(data_np)

        for d in range(D):
            transformed[:, d] = stats.boxcox(
                data_np[:, d], lmbda=self._lambdas[d].item()
            )

        result = torch.tensor(transformed, dtype=data.dtype, device=data.device)
        if len(original_shape) == 1:
            result = result.squeeze(-1)

        return result

    def inverse_transform(self, data: Tensor) -> Tensor:
        """
        Reverse Box-Cox transform.

        :param data: Transformed values.
        :return: Values in original scale.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Transform must be fitted before use")

        original_shape = data.shape
        data_np = data.cpu().numpy()
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        D = data_np.shape[1]
        original = np.zeros_like(data_np)

        for d in range(D):
            original[:, d] = inv_boxcox(
                data_np[:, d], self._lambdas[d].item()
            )

        result = torch.tensor(original, dtype=data.dtype, device=data.device)
        if len(original_shape) == 1:
            result = result.squeeze(-1)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for saving.

        :return: Dictionary with transform parameters.
        :raises RuntimeError: If transform not fitted.
        """
        if not self._fitted:
            raise RuntimeError("Cannot serialize unfitted transform")
        return {
            "type": "BoxCoxTransform",
            "name": self.name,
            "lambdas": self._lambdas.cpu(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BoxCoxTransform:
        """
        Reconstruct from serialized dictionary.

        :param d: Dictionary with transform parameters.
        :return: Reconstructed BoxCoxTransform instance.
        """
        transform = cls(name=d.get("name", "boxcox"))
        transform._lambdas = d["lambdas"]
        transform._fitted = True
        return transform


# Registry for transform types (for deserialization)
TRANSFORM_REGISTRY: Dict[str, type] = {
    "ZScoreTransform": ZScoreTransform,
    "MinMaxTransform": MinMaxTransform,
    "YeoJohnsonTransform": YeoJohnsonTransform,
    "BoxCoxTransform": BoxCoxTransform,
}


def load_transform(path: Path | str) -> QueryDataTransform:
    """
    Load a saved transform from disk.

    :param path: Path to .pt file containing serialized transform.
    :return: Reconstructed transform instance.
    :raises FileNotFoundError: If path does not exist.
    :raises ValueError: If transform type unknown.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")

    d = torch.load(path, weights_only=False)

    transform_type = d.get("type")
    if transform_type not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform type: {transform_type}")

    return TRANSFORM_REGISTRY[transform_type].from_dict(d)
