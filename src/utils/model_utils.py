"""Utilities for saving and loading finetuned TabPFN models."""
import torch
from tabpfn import TabPFNRegressor
from tabpfn.base import RegressorModelSpecs


def save_finetuned_model(model, path):
    """Save finetuned weights from an extracted inference model.

    Args:
        model: A fitted TabPFNRegressor (e.g. from extract_inference_model()).
        path: File path to save the checkpoint (.pt).
    """
    if not hasattr(model, 'models_') or model.models_ is None:
        raise AttributeError("Model has no initialized weights. "
                             "Ensure the model has been fitted or extracted.")
    state_dict = {k: v.cpu() for k, v in model.models_[0].state_dict().items()}
    torch.save({'state_dict': state_dict, 'n_estimators': model.n_estimators}, path)


def load_finetuned_model(path, device='cpu'):
    """Load finetuned weights into a TabPFNRegressor ready for inference.

    Creates a scaffold TabPFNRegressor to obtain the architecture, then injects
    finetuned weights via RegressorModelSpecs so that subsequent .fit() calls
    preserve the finetuned weights instead of reloading pretrained ones.

    Args:
        path: File path to the saved checkpoint (.pt).
        device: Device for the loaded model ('cpu' or 'cuda').

    Returns:
        A TabPFNRegressor with finetuned weights, ready for .fit() + .predict().
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Scaffold: fresh model to get architecture + configs
    scaffold = TabPFNRegressor(device='cpu')
    scaffold._initialize_model_variables()
    scaffold.models_[0].load_state_dict(checkpoint['state_dict'])

    # Wrap in RegressorModelSpecs so .fit() preserves finetuned weights
    specs = RegressorModelSpecs(
        model=scaffold.models_[0],
        architecture_config=scaffold.configs_[0],
        inference_config=scaffold.inference_config_,
        norm_criterion=scaffold.znorm_space_bardist_,
    )

    n_est = checkpoint.get('n_estimators', 8)
    return TabPFNRegressor(model_path=specs, device=device, n_estimators=n_est)
