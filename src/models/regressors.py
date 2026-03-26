"""
Finetuned TabPFN regressor wrappers and utilities.

- GradientMonitoredRegressor: FinetunedTabPFNRegressor with per-epoch diagnostics
- LoRAFinetunedRegressor: GradientMonitoredRegressor with LoRA adapter injection
- _make_finetuned_regressor(): factory that returns the appropriate regressor class
- extract_inference_model(): deep-copy the finetuned inference regressor
- linear_cka(): linear CKA between activation matrices
"""
import copy
import os

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.base import RegressorModelSpecs
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

from models.lora import (
    apply_lora, merge_lora, count_params, save_lora_checkpoint,
    load_lora_checkpoint,
)


def linear_cka(X, Y):
    """Linear CKA between activation matrices X, Y of shape [n, d]."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = (Y.T @ X).norm() ** 2
    hsic_xx = (X.T @ X).norm()
    hsic_yy = (Y.T @ Y).norm()
    return (hsic_xy / (hsic_xx * hsic_yy + 1e-10)).item()


class GradientMonitoredRegressor(FinetunedTabPFNRegressor):
    """
    FinetunedTabPFNRegressor with comprehensive per-epoch finetuning diagnostics.

    Always collects three categories of metrics each epoch:
      1. Gradient-based: gradient norm, gradient/weight ratio, update-to-parameter ratio
      2. Weight-based: weight displacement (L2), cosine similarity vs pretrained
      3. Representation-based: CKA similarity to pretrained activations (via forward hooks)

    All metrics are stored in self._diagnostics_ (list of per-epoch dicts).

    All metrics are printed to stdout each epoch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diagnostics_ = []
        self._cka_ref_X_ = None
        self._cka_ref_y_ = None
        self._pretrained_acts_ = {}
        self._cka_hook_names_ = []
        self._cka_ref_inputs_ = None

    def fit(self, X, y, **kwargs):
        self._grad_log_ = []
        self._pretrained_params_ = {}
        self._diagnostics_ = []
        self._pretrained_acts_ = {}

        # Store reference batch for CKA (fixed seed for reproducibility)
        n_ref = min(100, len(X))
        rng = np.random.RandomState(0)
        ref_idx = rng.choice(len(X), n_ref, replace=False)
        self._cka_ref_X_ = X[ref_idx]
        self._cka_ref_y_ = y[ref_idx]

        result = super().fit(X, y, **kwargs)

        if self._pretrained_params_:
            # Compute total weight change from pretrained snapshot
            weight_change = {}
            for name, p in self.finetuned_estimator_.model_.named_parameters():
                if name not in self._pretrained_params_:
                    continue
                layer = name.split('.')[0]
                delta = (p.data.cpu() - self._pretrained_params_[name]).norm().item()
                base  = self._pretrained_params_[name].norm().item() + 1e-8
                weight_change.setdefault(layer, []).append(delta / base * 100)
            weight_change_pct = {k: float(np.mean(v)) for k, v in weight_change.items()}

            print("\n[GradientMonitor] Total weight change from pretrained:")
            for layer, pct in sorted(weight_change_pct.items(), key=lambda x: -x[1]):
                bar = '█' * min(40, max(1, int(pct * 4)))
                print(f"  {layer:<30} {pct:6.2f}%  {bar}")

            self._weight_change_pct_ = weight_change_pct

        # Build backward-compat _grad_log_ from _diagnostics_
        self._grad_log_ = [
            {'epoch': d['epoch'], **d['grad_weight_ratio']}
            for d in self._diagnostics_ if d['epoch'] >= 0
        ]

        return result

    def _capture_activations(self, model):
        """Run forward pass with hooks, return {hook_name: flattened_activation [n, d]}."""
        activations = {}
        handles = []

        def _make_hook(name):
            def hook_fn(module, input, output):
                act = output if isinstance(output, torch.Tensor) else output[0]
                # Flatten all dims except first → [n, d]
                activations[name] = act.detach().cpu().reshape(act.shape[0], -1).float()
            return hook_fn

        for name in self._cka_hook_names_:
            try:
                module = model
                for attr in name.split('.'):
                    module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
                handles.append(module.register_forward_hook(_make_hook(name)))
            except (AttributeError, IndexError, KeyError):
                print(f"[GradientMonitor] Warning: could not hook '{name}', skipping CKA for this layer")

        x_input, y_input = self._cka_ref_inputs_
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(x_input, y_input, only_return_standard_out=True)
        if was_training:
            model.train()

        for h in handles:
            h.remove()

        return activations

    def _compute_weight_metrics(self, model):
        """Per-layer weight displacement and cosine similarity vs pretrained."""
        displacement = {}
        cosine_sim = {}
        update_ratio = {}
        for name, p in model.named_parameters():
            if name not in self._pretrained_params_:
                continue
            layer = name.split('.')[0]
            w = p.data.cpu().flatten()
            w0 = self._pretrained_params_[name].flatten()
            diff = (w - w0).norm().item()
            base = w0.norm().item() + 1e-8
            displacement.setdefault(layer, []).append(diff)
            update_ratio.setdefault(layer, []).append(diff / base * 100)
            cos = torch.nn.functional.cosine_similarity(
                w.unsqueeze(0), w0.unsqueeze(0)
            ).item()
            cosine_sim.setdefault(layer, []).append(cos)
        return (
            {k: float(np.mean(v)) for k, v in displacement.items()},
            {k: float(np.mean(v)) for k, v in cosine_sim.items()},
            {k: float(np.mean(v)) for k, v in update_ratio.items()},
        )

    def _log_epoch_evaluation(self, epoch, eval_result, mean_train_loss):
        model = self.finetuned_estimator_.model_

        if epoch == -1:
            # Snapshot pretrained weights
            self._pretrained_params_ = {
                name: p.data.clone().cpu()
                for name, p in model.named_parameters()
            }

            # Discover CKA hook layers
            n_transformer_layers = len(model.transformer_encoder.layers)
            mid = n_transformer_layers // 2
            self._cka_hook_names_ = [
                'encoder',
                'transformer_encoder.layers.0',
                f'transformer_encoder.layers.{mid}',
                f'transformer_encoder.layers.{n_transformer_layers - 1}',
            ]

            # Prepare CKA reference tensors
            device = next(model.parameters()).device
            n_ref = len(self._cka_ref_X_)
            n_ctx = n_ref // 2
            x_input = torch.tensor(
                self._cka_ref_X_, dtype=torch.float32, device=device
            ).unsqueeze(1)  # [N, 1, n_feat]
            y_input = torch.tensor(
                self._cka_ref_y_[:n_ctx], dtype=torch.float32, device=device
            ).unsqueeze(1)  # [N//2, 1]
            self._cka_ref_inputs_ = (x_input, y_input)

            # Capture pretrained activations
            self._pretrained_acts_ = self._capture_activations(model)

        else:
            # --- Gradient-based metrics ---
            grad_norm = {}
            grad_weight_ratio = {}
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                layer = name.split('.')[0]
                g_norm = param.grad.norm().item()
                w_norm = param.data.norm().item() + 1e-8
                grad_norm.setdefault(layer, []).append(g_norm)
                grad_weight_ratio.setdefault(layer, []).append(g_norm / w_norm * 100)

            grad_norm_avg = {k: float(np.mean(v)) for k, v in grad_norm.items()}
            grad_ratio_avg = {k: float(np.mean(v)) for k, v in grad_weight_ratio.items()}

            # --- Weight-based metrics ---
            w_disp, w_cos, w_update = self._compute_weight_metrics(model)

            # --- CKA ---
            cka_scores = {}
            if self._pretrained_acts_ and self._cka_ref_inputs_ is not None:
                current_acts = self._capture_activations(model)
                for hook_name in self._cka_hook_names_:
                    if hook_name in self._pretrained_acts_ and hook_name in current_acts:
                        cka_scores[hook_name] = linear_cka(
                            self._pretrained_acts_[hook_name],
                            current_acts[hook_name],
                        )

            epoch_entry = {
                'epoch': epoch,
                'grad_norm': grad_norm_avg,
                'grad_weight_ratio': grad_ratio_avg,
                'update_to_param_ratio': w_update,
                'weight_displacement': w_disp,
                'cosine_similarity': w_cos,
                'cka': cka_scores,
            }
            self._diagnostics_.append(epoch_entry)

            print(f"\n[GradientMonitor] Epoch {epoch + 1} diagnostics:")
            print(f"  Grad/weight ratio: {', '.join(f'{k}={v:.4f}%' for k, v in sorted(grad_ratio_avg.items()))}")
            print(f"  Cosine sim:        {', '.join(f'{k}={v:.6f}' for k, v in sorted(w_cos.items()))}")
            if cka_scores:
                print(f"  CKA:               {', '.join(f'{k}={v:.4f}' for k, v in cka_scores.items())}")

        # Always call original implementation to preserve logging/early stopping
        super()._log_epoch_evaluation(epoch, eval_result, mean_train_loss)


class LoRAFinetunedRegressor(GradientMonitoredRegressor):
    """GradientMonitoredRegressor with LoRA adapter injection.

    Injects low-rank adapters into target nn.Linear layers after model
    initialisation but before optimizer creation, so only LoRA parameters
    receive gradient updates.  Merges adapters back before the inference
    model is cloned so downstream code sees plain nn.Linear layers.

    Args (extra vs GradientMonitoredRegressor):
        lora_rank: rank of the low-rank matrices (default 8)
        lora_alpha: scaling factor (default 16)
        lora_target: which layer group to adapt (default 'decoder_dict')
    """

    def __init__(self, *args, lora_rank=8, lora_alpha=16,
                 lora_target='decoder_dict', **kwargs):
        super().__init__(*args, **kwargs)
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._lora_target = lora_target
        self._lora_n_replaced = 0

    # ------------------------------------------------------------------
    #  Hook 1: inject LoRA after model weights are loaded
    # ------------------------------------------------------------------
    def _setup_estimator(self) -> None:
        """Monkey-patch _initialize_model_variables to inject LoRA.

        The base _fit flow is:
            _create_estimator()          → self.finetuned_estimator_
            _setup_estimator()           → (this method)
            ...
            _initialize_model_variables() → loads pretrained weights
            model_.to(device)
            get_and_init_optimizer(model_.parameters(), ...)

        We wrap _initialize_model_variables so that after it loads the
        pretrained weights, we inject LoRA adapters and freeze the base.
        The optimizer then sees only LoRA params as trainable.
        """
        super()._setup_estimator()

        estimator = self.finetuned_estimator_
        original_init = estimator._initialize_model_variables

        def _init_with_lora():
            original_init()
            model = estimator.model_
            self._lora_n_replaced = apply_lora(
                model,
                target=self._lora_target,
                rank=self._lora_rank,
                alpha=self._lora_alpha,
            )
            trainable, total = count_params(model)
            print(f"  [LoRA] Injected {self._lora_n_replaced} adapters "
                  f"(rank={self._lora_rank}, alpha={self._lora_alpha}, "
                  f"target={self._lora_target!r})")
            print(f"  [LoRA] Trainable: {trainable:,} / {total:,} "
                  f"({trainable / total * 100:.2f}%)")

        estimator._initialize_model_variables = _init_with_lora

    # ------------------------------------------------------------------
    #  Hook 2: merge LoRA before inference model clone
    # ------------------------------------------------------------------
    def _setup_inference_model(self, final_inference_eval_config) -> None:
        """Merge LoRA adapters into base weights before cloning.

        clone_model_for_evaluation() deep-copies models_[0], so we need
        the underlying nn.Linear layers to have the merged weights.
        """
        model = self.finetuned_estimator_.model_
        n_merged = merge_lora(model)
        print(f"  [LoRA] Merged {n_merged} adapters into base weights")
        super()._setup_inference_model(final_inference_eval_config)

    # ------------------------------------------------------------------
    #  LoRA checkpoint saving (called externally after fit)
    # ------------------------------------------------------------------
    def save_lora(self, save_dir, extra_config=None):
        """Save LoRA weights + config to disk.

        Must be called BEFORE merge (i.e. before _setup_inference_model).
        For normal usage this is called from finetune_tabpfn() between
        the training loop and inference model setup — but since TabPFN
        calls _setup_inference_model internally at the end of _fit, we
        capture the state dict during training instead.

        This method works with the already-merged model by re-applying
        LoRA from saved state — but it's simpler to call it from the
        _log_epoch_evaluation override at the last epoch.
        """
        config = {
            'lora_rank': self._lora_rank,
            'lora_alpha': self._lora_alpha,
            'lora_target': self._lora_target,
            'n_replaced': self._lora_n_replaced,
        }
        if extra_config:
            config.update(extra_config)
        return save_lora_checkpoint(
            self.finetuned_estimator_.model_, save_dir, config,
        )

    def fit(self, X, y, **kwargs):
        """Fit with LoRA, capturing adapter state before merge."""
        from models.lora import get_lora_state_dict
        result = super().fit(X, y, **kwargs)
        # After super().fit(), _setup_inference_model has already merged.
        # We save the pre-merge state by capturing it in _log_epoch_evaluation.
        return result

    def _log_epoch_evaluation(self, epoch, eval_result, mean_train_loss):
        """Capture LoRA state dict at every epoch (overwritten each time).

        The last captured state is the final pre-merge adapter weights.
        """
        if epoch >= 0:
            from models.lora import get_lora_state_dict
            self._lora_state_dict_ = get_lora_state_dict(
                self.finetuned_estimator_.model_
            )
        super()._log_epoch_evaluation(epoch, eval_result, mean_train_loss)


def _make_finetuned_regressor(silence_diagnostics=True, use_lora=False,
                               lora_rank=8, lora_alpha=16,
                               lora_target='decoder_dict', **kwargs):
    """Factory: returns the appropriate regressor class.

    - use_lora=True  → LoRAFinetunedRegressor (always includes diagnostics)
    - silence_diagnostics=False → GradientMonitoredRegressor
    - otherwise → plain FinetunedTabPFNRegressor
    """
    if use_lora:
        return LoRAFinetunedRegressor(
            lora_rank=lora_rank, lora_alpha=lora_alpha,
            lora_target=lora_target, **kwargs,
        )
    if silence_diagnostics:
        return FinetunedTabPFNRegressor(**kwargs)
    return GradientMonitoredRegressor(**kwargs)


def extract_inference_model(finetuned_regressor):
    """Extract a TabPFNRegressor with finetuned weights for in-context learning.

    After finetuning completes, the FinetunedTabPFNRegressor stores an internal
    TabPFNRegressor with finetuned weights at `finetuned_inference_regressor_`.
    This function deep-copies it to produce a standalone regressor where:
      - .fit(X, y) stores context (no gradient updates)
      - .predict() supports output_type="quantiles" and all other options
    """
    if not hasattr(finetuned_regressor, 'finetuned_inference_regressor_'):
        raise AttributeError(
            "FinetunedTabPFNRegressor has not been fit yet. "
            "Call .fit() before extracting the inference model."
        )
    inference_model = copy.deepcopy(finetuned_regressor.finetuned_inference_regressor_)
    print(f"  Extracted TabPFNRegressor with finetuned weights "
          f"(fit_mode={inference_model.fit_mode!r})")
    return inference_model


def load_lora_as_inference_model(checkpoint_dir, device='cpu'):
    """Load a LoRA checkpoint and return a TabPFNRegressor ready for inference.

    Initialises a pretrained TabPFNRegressor, applies the saved LoRA adapters,
    merges them into base weights, and wraps the result in a RegressorModelSpecs
    so that subsequent .fit()/.predict() calls do not reload from disk.

    Args:
        checkpoint_dir: directory containing lora_weights.pt and lora_config.json
        device: 'cpu' or 'cuda'

    Returns:
        (model, config) tuple:
          - model: TabPFNRegressor with LoRA-merged weights, fit_mode='fit_preprocessors'
          - config: dict from lora_config.json (rank, alpha, target, dataset_type, …)
    """
    # Validate checkpoint files exist
    for fname in ('lora_config.json', 'lora_weights.pt'):
        path = os.path.join(checkpoint_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"LoRA checkpoint missing {fname} in {checkpoint_dir}")

    # 1. Initialise pretrained model
    tmp = TabPFNRegressor(device=device)
    tmp._initialize_model_variables()

    # 2. Inject LoRA adapters + load saved weights
    config = load_lora_checkpoint(tmp.models_[0], checkpoint_dir)

    # 3. Merge adapters into base weights
    n_merged = merge_lora(tmp.models_[0])
    print(f"  [LoRA] Merged {n_merged} adapters from checkpoint")

    # 4. Wrap in RegressorModelSpecs so .fit() won't reload from disk
    specs = RegressorModelSpecs(
        model=tmp.models_[0],
        architecture_config=tmp.configs_[0],
        inference_config=tmp.inference_config_,
        norm_criterion=tmp.znorm_space_bardist_,
    )

    model = TabPFNRegressor(
        model_path=specs,
        fit_mode='fit_preprocessors',
        ignore_pretraining_limits=True,
        device=device,
        n_estimators=8,
    )

    print(f"  [LoRA] Ready for inference (device={device})")
    return model, config
