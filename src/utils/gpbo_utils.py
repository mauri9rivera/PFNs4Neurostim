import torch
from torch.distributions import Normal
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np


def expected_improvement(model, likelihood, X_candidates, y_best, device):
    """
    Computes EI for the GP model on discrete candidates.
    """
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        # Predictive posterior
        posterior = likelihood(model(X_candidates))
        mean = posterior.mean
        sigma = posterior.stddev
        
        # Avoid div by zero
        sigma = torch.clamp(sigma, min=1e-9)
        
        # EI Formula
        z = (mean - y_best) / sigma
        # Using PyTorch Normal distribution for cdf/pdf
        dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
        ei = (mean - y_best) * dist.cdf(z) + sigma * dist.log_prob(z).exp()
        
    return ei

def expected_improvement2(gp, X_candidates, y_best, device):
    """
    Computes EI for the GP model on discrete candidates.
    """
    
    posterior = gp.predict(X_candidates, return_std=True)
        
    mean = torch.tensor(posterior[0], device=device)
    sigma = posterior[1]
        
    # Avoid div by zero
    sigma = torch.tensor(np.clip(sigma, 0.0, np.inf), device=device)
        
    # EI Formula
    z = (mean - y_best) / sigma
    # Using PyTorch Normal distribution for cdf/pdf
    dist = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
    ei = (mean - y_best) * dist.cdf(z) + sigma * dist.log_prob(z).exp()
        
    return ei



