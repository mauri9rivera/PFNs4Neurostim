"""
LoRA (Low-Rank Adaptation) for TabPFN.

Parameter-efficient finetuning by injecting trainable low-rank matrices
into frozen Linear layers:  W' = W + (alpha/rank) * B @ A

Only nn.Linear layers are targeted.  TabPFN's attention modules use raw
nn.Parameter tensors (_w_q, _w_k, …), which are intentionally excluded —
the decoder_dict shows the largest gradient signal and is sufficient for
neurostim adaptation.

Usage from regressors.py:
    apply_lora(model, target='decoder_dict', rank=8, alpha=16)
    ...  # training
    merge_lora(model)          # fold adapters into base weights
    save_lora_checkpoint(...)  # persist adapters + config
"""
import json
import math
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  LoRA Linear wrapper
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank adapter.

    Forward:  y = base_linear(x) + (x @ A @ B) * scaling
    where A ∈ R^{in×r}, B ∈ R^{r×out}, scaling = alpha / rank.

    A is Kaiming-init, B is zero-init → initial delta is zero so the
    model starts from pretrained behaviour.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze base
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base_linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out

    def merge_and_return_base(self):
        """Merge LoRA delta into base weights and return the base Linear."""
        with torch.no_grad():
            # nn.Linear stores weight as (out, in), delta is (in, rank)@(rank, out) = (in, out)
            delta = (self.lora_A @ self.lora_B) * self.scaling   # (in, out)
            self.base_linear.weight.data += delta.T
        self.base_linear.weight.requires_grad = True
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = True
        return self.base_linear

    def extra_repr(self):
        return (f"rank={self.rank}, alpha={self.alpha}, "
                f"in={self.base_linear.in_features}, "
                f"out={self.base_linear.out_features}")


# ---------------------------------------------------------------------------
#  Target resolution
# ---------------------------------------------------------------------------

_TARGET_PATTERNS = {
    'decoder_dict': ('decoder_dict.',),
    'decoder_dict+mlp': ('decoder_dict.', '.mlp.'),
}


def _should_adapt(module_name: str, target: str) -> bool:
    """Return True if *module_name* matches the target pattern set."""
    patterns = _TARGET_PATTERNS.get(target)
    if patterns is None:
        raise ValueError(
            f"Unknown LoRA target {target!r}. "
            f"Valid: {sorted(_TARGET_PATTERNS)}"
        )
    return any(p in module_name for p in patterns)


# ---------------------------------------------------------------------------
#  Inject / merge / freeze
# ---------------------------------------------------------------------------

def apply_lora(model: nn.Module, target: str = 'decoder_dict',
               rank: int = 8, alpha: int = 16) -> int:
    """Replace target nn.Linear layers with LoRALinear and freeze base weights.

    Returns the number of Linear layers replaced.
    """
    # Collect (parent, child_key, Linear) tuples to replace
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _should_adapt(name, target):
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_key = parts
            else:
                parent_name, child_key = '', parts[0]
            # Resolve parent
            parent = model
            if parent_name:
                for attr in parent_name.split('.'):
                    parent = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
            replacements.append((parent, child_key, module))

    for parent, child_key, linear in replacements:
        lora_layer = LoRALinear(linear, rank=rank, alpha=alpha)
        if child_key.isdigit():
            parent[int(child_key)] = lora_layer
        else:
            setattr(parent, child_key, lora_layer)

    # Freeze ALL non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    return len(replacements)


def merge_lora(model: nn.Module) -> int:
    """Merge all LoRALinear adapters back into base weights.

    After merging, the model contains only standard nn.Linear layers
    and can be deep-copied / serialised normally.

    Returns the number of layers merged.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_key = parts
            else:
                parent_name, child_key = '', parts[0]
            parent = model
            if parent_name:
                for attr in parent_name.split('.'):
                    parent = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
            replacements.append((parent, child_key, module))

    for parent, child_key, lora_module in replacements:
        merged_linear = lora_module.merge_and_return_base()
        if child_key.isdigit():
            parent[int(child_key)] = merged_linear
        else:
            setattr(parent, child_key, merged_linear)

    return len(replacements)


# ---------------------------------------------------------------------------
#  State dict utilities
# ---------------------------------------------------------------------------

def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA A/B parameters from the model."""
    return {
        name: param.data.cpu().clone()
        for name, param in model.named_parameters()
        if 'lora_' in name
    }


def load_lora_state_dict(model: nn.Module, state_dict: dict) -> list[str]:
    """Load LoRA weights into an already-LoRA-injected model.

    Returns list of keys in state_dict that were not found in the model.
    """
    model_params = dict(model.named_parameters())
    missing = []
    for name, tensor in state_dict.items():
        if name in model_params:
            model_params[name].data.copy_(tensor)
        else:
            missing.append(name)
    return missing


def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ---------------------------------------------------------------------------
#  Checkpoint I/O
# ---------------------------------------------------------------------------

def save_lora_checkpoint(model: nn.Module, save_dir: str, config: dict):
    """Save LoRA weights and experiment config to *save_dir*.

    Creates:
      save_dir/lora_weights.pt  — LoRA A/B tensors
      save_dir/lora_config.json — rank, alpha, target, experiment metadata
    """
    os.makedirs(save_dir, exist_ok=True)

    state_dict = get_lora_state_dict(model)
    weights_path = os.path.join(save_dir, 'lora_weights.pt')
    torch.save(state_dict, weights_path)

    config_path = os.path.join(save_dir, 'lora_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    trainable, total = count_params(model)
    print(f"  [LoRA] Saved {len(state_dict)} tensors -> {weights_path}")
    print(f"  [LoRA] Config -> {config_path}")
    return weights_path


def load_lora_checkpoint(model: nn.Module, checkpoint_dir: str,
                         target: str = 'decoder_dict'):
    """Apply LoRA from a saved checkpoint.

    Reads lora_config.json for rank/alpha, injects adapters, then loads weights.
    Returns the config dict.
    """
    config_path = os.path.join(checkpoint_dir, 'lora_config.json')
    with open(config_path) as f:
        config = json.load(f)

    rank = config['lora_rank']
    alpha = config['lora_alpha']
    target = config.get('lora_target', target)

    n_replaced = apply_lora(model, target=target, rank=rank, alpha=alpha)
    print(f"  [LoRA] Injected {n_replaced} adapters from config "
          f"(rank={rank}, alpha={alpha}, target={target})")

    weights_path = os.path.join(checkpoint_dir, 'lora_weights.pt')
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    missing = load_lora_state_dict(model, state_dict)
    if missing:
        print(f"  [LoRA] Warning: {len(missing)} keys not found in model")

    return config
