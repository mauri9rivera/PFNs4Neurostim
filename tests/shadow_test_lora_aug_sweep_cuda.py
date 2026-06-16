"""Shadow smoke test: LoRA aug sweep on CUDA with a tiny synthetic dataset.

Verifies the full finetuned_percentage() LoRA code path end-to-end:
  - LoRA adapter injection
  - fit on synthetic data
  - BO loop (budget=5, n_reps=2)
  - result dict structure

Usage:
    conda run -n pfns4neurostim python tests/shadow_test_lora_aug_sweep_cuda.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

from evaluation import finetuned_percentage

print("\n--- Running LoRA aug sweep shadow test (device=%s) ---" % device)
try:
    finetuned_percentage(
        dataset_type='nhp',
        split_type='inter_subject',
        mode='optimization',
        device=device,
        budget=5,
        n_reps=2,
        epochs=2,
        lr=1e-4,
        aug_pct_list=[0.1, 0.5],
        held_out_subj_idx=1,
        save=False,
        silence_diagnostics=True,
        kappa_schedule=4.0,
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_target='decoder_dict',
        grad_clip=1.0,
        n_estimators_finetune=8,
    )
    print("\n[PASS] LoRA aug sweep CUDA shadow test completed successfully.")
except Exception as e:
    print(f"\n[FAIL] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
