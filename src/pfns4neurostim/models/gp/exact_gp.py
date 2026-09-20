import torch
import gpytorch


# ============================================
#         GPYTorch Model Definition
# ============================================

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
        self.likf = likelihood
        self.name = 'ExactGP'

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ============================================
#     Non-Stationary (Deep-Kernel) GP — §16
# ============================================

class _FeatureExtractor(torch.nn.Module):
    """Small MLP that warps inputs into a learned feature space.

    Implements the input-warping stage of Deep Kernel Learning (Wilson et al.,
    2016, "Deep Kernel Learning", AISTATS). Composing a stationary RBF kernel
    with this learned non-linear map yields an effectively *non-stationary*
    kernel in the original coordinate space — the model can shrink its
    lengthscale near sharp EMG hotspots and stretch it over flat regions,
    which a single-lengthscale ARD RBF cannot.

    Args:
        d_in: Input dimensionality (electrode/parameter coordinates).
        hidden: Width of the two hidden layers.
        d_out: Output feature dimensionality fed to the base kernel.
    """

    def __init__(self, d_in: int, hidden: int = 32, d_out: int = 2) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),      # [N, d_in] -> [N, hidden]
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),    # [N, hidden] -> [N, hidden]
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, d_out),     # [N, hidden] -> [N, d_out]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [N, d_out]


class DeepKernelGP(gpytorch.models.ExactGP):
    """Deep-kernel (non-stationary) exact GP with a learned input warping.

    A shared MLP (:class:`_FeatureExtractor`) maps inputs into a low-dimensional
    feature space on which a standard ARD RBF kernel operates. Trained jointly
    with the GP hyperparameters via the exact marginal likelihood, this gives a
    non-stationary baseline (limitation L2/L9 control) that the stationary
    :class:`ExactGP` cannot express.

    Args:
        train_x: Training inputs, shape [N, D].
        train_y: Training targets, shape [N].
        likelihood: A gpytorch Gaussian likelihood.
        feature_dim: Output dimensionality of the feature extractor.
        hidden: Hidden width of the feature-extractor MLP.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: "gpytorch.likelihoods.GaussianLikelihood",
        feature_dim: int = 2,
        hidden: int = 32,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        d_in = train_x.shape[-1]
        self.feature_extractor = _FeatureExtractor(d_in, hidden=hidden, d_out=feature_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
        )
        # Constrain warped features to [-1, 1] so the GP sees a bounded domain
        # regardless of MLP scale — stabilises marginal-likelihood training.
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
        self.likf = likelihood
        self.name = 'DeepKernelGP'

    def forward(self, x: torch.Tensor) -> "gpytorch.distributions.MultivariateNormal":
        z = self.feature_extractor(x)         # [N, feature_dim]
        z = self.scale_to_bounds(z)           # [N, feature_dim], bounded
        mean_x = self.mean_module(z)          # [N]
        covar_x = self.covar_module(z)        # [N, N]
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

