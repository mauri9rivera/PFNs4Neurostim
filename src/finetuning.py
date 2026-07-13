"""
Fine-tuning orchestration for TabPFN on neurostimulation data.

Two-phase workflow:
  1. Backprop finetuning: finetune_tabpfn() trains a GradientMonitoredRegressor,
     adapting pretrained weights to neurostimulation data via gradient updates.
     Diagnostics (gradient/weight metrics, CKA) are always collected.
  2. Extraction for evaluation: extract_inference_model() deep-copies the
     internal TabPFNRegressor with finetuned weights. This standalone model
     uses in-context learning (.fit() stores context, no gradients) and
     supports the full predict API including output_type="quantiles".
Usage:
    python finetuning.py --dataset rat --device cuda --epochs 30
    python finetuning.py --dataset nhp --device cuda --epochs 30
    python finetuning.py --dataset nhp --mode optimization --budget 100 --n_reps 20
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Any

import yaml

import numpy as np
import pandas as pd
import torch

from models.regressors import _make_finetuned_regressor, extract_inference_model
from evaluation import (
    gp_baseline, finetuned_optimization,
    finetuned_optimization_budget,
    finetuned_percentage, load_sweep_results,
)
from utils.data_utils import (
    build_finetuning_dataset, load_data,
    HELD_OUT_SUBJECTS, TRAIN_SUBJECTS, ALL_SUBJECTS,
    generate_experiment_tag, save_results,
    create_run_dir, write_run_config,
    load_subject_result, save_subject_result,
)
from utils.visualization import (
    r2_by_subject,
    spearman_by_subject, spearman_by_emg,
    regret_with_timing, regret_traces_by_subject, regret_by_emg,
    exploration_by_subject, exploration_by_emg,
    augmentation_sweep_plot,
    visualize_representation, show_emg_map,
    plot_gradient_metrics, plot_weight_metrics, plot_cka_similarity,
    plot_gradient_share,
)


def finetune_tabpfn(dataset_type, device='cuda', epochs=100, lr=1e-4,
                    aug_pct: float = 2.5, subject_indices=None,
                    held_out_emg_idx=None, seed=42,
                    silence_diagnostics=True, output_dir=None,
                    use_lora=False, lora_rank=8, lora_alpha=16,
                    lora_target='decoder_dict', grad_clip=None,
                    n_estimators_finetune=8,
                    aug_transforms: tuple | None = None,
                    loss_weights: dict[str, float] | None = None):
    """
    Fine-tune a TabPFNRegressor on augmented neurostimulation data.

    Args:
        dataset_type: 'rat' or 'nhp'
        device: 'cpu' or 'cuda'
        epochs: number of fine-tuning epochs
        lr: learning rate
        aug_pct: augmentation percentage (1.0 = 100% = 10 maps per EMG)
        subject_indices: list of subject indices to train on (None = all training subjects)
        held_out_emg_idx: EMG index to exclude from training data
        seed: random seed for dataset building
        silence_diagnostics: if True (default), skip gradient/CKA monitoring for faster
            finetuning and lower memory. If False, use GradientMonitoredRegressor.
        output_dir: when set, saves diagnostic plots to {output_dir}/diagnostics/
        use_lora: if True, use LoRA parameter-efficient finetuning
        lora_rank: rank of low-rank matrices (default 8)
        lora_alpha: scaling factor (default 16)
        lora_target: which layers to adapt (default 'decoder_dict')
        grad_clip: per-parameter gradient max-norm for clipping (None = disabled).
            Only active when silence_diagnostics=False or use_lora=True.
            Prevents gradient explosion and the resulting CUDA memory corruption.
        n_estimators_finetune: number of TabPFN ensemble members used during
            finetuning forward/backward passes (default 8). Reduce to 1 on
            consumer GPUs with limited VRAM — fewer simultaneous model copies
            prevents the deepcopy-triggered CUDA illegal memory access.
        aug_transforms: tuple of transform names to sample from per augmented map;
            None defaults to ('none',) — noise only, no spatial transforms.
        loss_weights: optional dict overriding the TabPFN finetuning objective
            term weights. Recognised keys: ``ce_loss_weight`` (bar-distribution
            NLL), ``crps_loss_weight`` (CRPS), ``crls_loss_weight`` (CRLS),
            ``mse_loss_weight`` / ``mae_loss_weight`` (auxiliary point losses on
            the decoded mean). ``None`` (default) keeps TabPFN's built-in defaults
            (CRPS + auxiliary MSE, both weight 1.0). E.g. ``{'crps_loss_weight':
            1.0, 'mse_loss_weight': 0.0}`` finetunes on pure CRPS;
            ``{'ce_loss_weight': 1.0, 'crps_loss_weight': 0.0, 'mse_loss_weight':
            0.0}`` finetunes on pure bar-distribution NLL.

    Returns:
        (ft_model_raw, ft_model) tuple:
          - ft_model_raw: the finetuned regressor (with _diagnostics_ when diagnostics enabled)
          - ft_model: extracted TabPFNRegressor for in-context learning
    """
    print(f"Building augmented dataset for '{dataset_type}' ...")
    X_train, y_train = build_finetuning_dataset(
        dataset_type,
        subject_indices=subject_indices,
        held_out_emg_idx=held_out_emg_idx,
        aug_pct=aug_pct,
        seed=seed,
        aug_transforms=aug_transforms,
    )
    print(f"  Dataset size: {X_train.shape[0]} rows, {X_train.shape[1]} features")

    method = "LoRA" if use_lora else "full"
    print(f"Initializing finetuned regressor ({method}, epochs={epochs}, lr={lr}) ...")

    ft_model_raw = _make_finetuned_regressor(
        silence_diagnostics=silence_diagnostics,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target=lora_target,
        grad_clip=grad_clip,
        device=device,
        epochs=epochs,
        learning_rate=lr,
        n_estimators_finetune=n_estimators_finetune,
        n_estimators_validation=n_estimators_finetune,
        n_estimators_final_inference=n_estimators_finetune,
        **(loss_weights or {}),
    )
    if loss_weights:
        print(f"  [loss] objective overrides: {loss_weights}")

    print("Fine-tuning ...")
    ft_model_raw.fit(X_train, y_train)

    # Flush pending CUDA ops and release cached allocator memory
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Save LoRA checkpoint if applicable
    if use_lora and output_dir:
        lora_dir = os.path.join(output_dir, 'lora')
        # Use pre-merge state dict captured during training
        if hasattr(ft_model_raw, '_lora_state_dict_'):
            os.makedirs(lora_dir, exist_ok=True)
            torch.save(ft_model_raw._lora_state_dict_,
                       os.path.join(lora_dir, 'lora_weights.pt'))
            config = {
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'lora_target': lora_target,
                'n_replaced': ft_model_raw._lora_n_replaced,
                'dataset_type': dataset_type,
                'epochs': epochs,
                'lr': lr,
                'aug_pct': aug_pct,
            }
            with open(os.path.join(lora_dir, 'lora_config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  [LoRA] Checkpoint saved -> {lora_dir}")

    # Always save diagnostic plots when diagnostics are available
    if hasattr(ft_model_raw, '_diagnostics_') and ft_model_raw._diagnostics_:
        diag_dir = os.path.join(output_dir, 'diagnostics') if output_dir else None
        plot_gradient_metrics(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
        plot_weight_metrics(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
        plot_cka_similarity(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
        plot_gradient_share(ft_model_raw._diagnostics_, save=True, output_dir=diag_dir)
        if diag_dir:
            with open(os.path.join(diag_dir, "diagnostics.pkl"), "wb") as _f:
                pickle.dump(ft_model_raw._diagnostics_, _f, protocol=pickle.HIGHEST_PROTOCOL)
            pd.json_normalize(ft_model_raw._diagnostics_, sep="__").to_csv(
                os.path.join(diag_dir, "diagnostics.csv"), index=False
            )

    print("Fine-tuning complete.")

    ft_model = extract_inference_model(ft_model_raw)
    return ft_model_raw, ft_model


def load_diagnostics(run_dir: str) -> list[dict[str, Any]]:
    """Load serialized per-epoch finetuning diagnostics from a run directory.

    Args:
        run_dir: Path to a run directory (e.g. ``output/runs/nhp-optimization-a3f9c``).
            Must contain a ``diagnostics/diagnostics.pkl`` file written by
            ``finetune_tabpfn()`` when ``output_dir`` is set and diagnostics
            are enabled (``silence_diagnostics=False`` or ``use_lora=True``).

    Returns:
        List of per-epoch diagnostic dicts. Each dict contains keys:
        ``epoch``, ``grad_norm``, ``grad_weight_ratio``,
        ``update_to_param_ratio``, ``weight_displacement``,
        ``cosine_similarity``, ``cka``.

    Raises:
        FileNotFoundError: If ``diagnostics/diagnostics.pkl`` is absent in
            ``run_dir``.
    """
    pkl_path = os.path.join(run_dir, "diagnostics", "diagnostics.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(
            f"Diagnostics file not found: {pkl_path!r}. "
            "Re-run finetune_tabpfn() with output_dir set and "
            "silence_diagnostics=False (or use_lora=True)."
        )
    with open(pkl_path, "rb") as _f:
        return pickle.load(_f)


# ============================================
#       High-Level Experiment Runner
# ============================================

_VALID_MODES = {'optimization', 'optimization_budget'}


def run_experiment(
    dataset_type,
    split_type='inter_subject',
    mode=None,
    device='cuda',
    budget=100,
    n_reps=30,
    epochs=100,
    lr=1e-4,
    aug_pct: float = 2.5,
    held_out_emg_idx=None,
    held_out_subj_idx=None,
    budgets=None,
    save=False,
    silence_diagnostics=True,
    use_lora=False,
    lora_rank=8,
    lora_alpha=16,
    lora_target='decoder_dict',
    lora_weights=None,
    kappa_schedule: float = 0.0,
    grad_clip=None,
    n_estimators_finetune: int = 8,
    aug_transforms: tuple | None = None,
    acq_fn: str = 'ucb',
    ts_temperature: float = 1.0,
    loss_weights: dict[str, float] | None = None,
):
    """
    Unified entry point for transfer learning evaluation.

    Args:
        dataset_type: 'rat' or 'nhp'
        split_type: 'inter_subject' — train on TRAIN_SUBJECTS, test on HELD_OUT_SUBJECTS;
                    'intra_emg'     — train on ALL_SUBJECTS (excluding held_out_emg_idx),
                                      test on that EMG across ALL_SUBJECTS.
        mode: str or list of str — any combination of 'optimization' /
              'optimization_budget'. The model is finetuned once and all
              requested modes are evaluated sequentially.
        device: 'cpu' or 'cuda'
        budget: number of BO queries (optimization)
        n_reps: number of repetitions per experiment
        epochs: fine-tuning epochs
        lr: fine-tuning learning rate
        aug_pct: augmentation percentage (1.0 = 100% = 10 maps per EMG)
        held_out_emg_idx: required when split_type='intra_emg'; the EMG index
            held out from training and used as the test set.
        held_out_subj_idx: optional int. When set, overrides the default subject
            split: trains on all subjects except this one and tests on it alone.
        budgets: list of budgets for 'optimization_budget' mode.
        save: if True, persist results to output/results/ (pkl + CSV summary).
        silence_diagnostics: if True (default), skip gradient/CKA monitoring.
        use_lora: if True, use LoRA parameter-efficient finetuning.
        lora_rank: rank of low-rank matrices (default 8).
        lora_alpha: scaling factor (default 16).
        lora_target: which layers to adapt (default 'decoder_dict').
        lora_weights: path to saved LoRA checkpoint directory. When set,
            skips training and loads pre-trained adapters for evaluation.
        kappa_schedule: UCB exploration coefficient shared by both the finetuned
            TabPFN and the GP baseline BO loops.  ``0.0`` (default) = auto
            cosine-annealed schedule (GP-UCB theory scaling); any other value =
            fixed kappa throughout for both models.
        acq_fn: Acquisition function — ``'ucb'`` (default) or ``'ts'``
            (Thompson Sampling).  Passed to all ``run_bo_loop`` calls.
        ts_temperature: Temperature for TabPFN bar-distribution TS sampling.
            ``1.0`` (default) = exact distribution; ``<1.0`` = sharper/greedier;
            ``>1.0`` = more uniform/exploratory.  Only used when ``acq_fn='ts'``.
        loss_weights: optional dict of TabPFN finetuning objective weights
            (``ce_loss_weight`` / ``crps_loss_weight`` / ``crls_loss_weight`` /
            ``mse_loss_weight`` / ``mae_loss_weight``). Forwarded to
            :func:`finetune_tabpfn`; ``None`` keeps TabPFN's CRPS+MSE default.

    Returns:
        dict keyed by mode name, each value being the result of that mode
        ('optimization' → {'TabPFN': [...], 'GP': [...]},
         'optimization_budget' → DataFrame).
    """
    if mode is None:
        mode = ['optimization']
    if isinstance(mode, str):
        mode = [mode]
    invalid = set(mode) - _VALID_MODES
    if invalid:
        raise ValueError(f"Unknown mode(s): {invalid}. Valid: {_VALID_MODES}")

    if budgets is None:
        budgets = [10, 50, 100, 150, 200]

    # --- Resolve train / test sets ---
    if split_type == 'inter_subject':
        if held_out_subj_idx is not None:
            train_subject_indices = [s for s in ALL_SUBJECTS[dataset_type] if s != held_out_subj_idx]
            test_subjects = [held_out_subj_idx]
        else:
            train_subject_indices = TRAIN_SUBJECTS[dataset_type]
            test_subjects = HELD_OUT_SUBJECTS[dataset_type]
        test_emg_indices = None
        ft_held_out_emg = None
    elif split_type == 'intra_emg':
        if held_out_emg_idx is None:
            raise ValueError("held_out_emg_idx must be set when split_type='intra_emg'")
        if held_out_subj_idx is not None:
            train_subject_indices = [s for s in ALL_SUBJECTS[dataset_type] if s != held_out_subj_idx]
            test_subjects = [held_out_subj_idx]
        else:
            train_subject_indices = ALL_SUBJECTS[dataset_type]
            test_subjects = ALL_SUBJECTS[dataset_type]
        test_emg_indices = [held_out_emg_idx]
        ft_held_out_emg = held_out_emg_idx
    else:
        raise ValueError(f"Unknown split_type={split_type!r}. Use 'inter_subject' or 'intra_emg'.")

    # --- Build experiment tag for unique filenames ---
    tag_parts = [split_type]
    if held_out_subj_idx is not None:
        tag_parts.append(f'subj{held_out_subj_idx}')
    if held_out_emg_idx is not None:
        tag_parts.append(f'emg{held_out_emg_idx}')
    exp_tag = '_'.join(tag_parts)

    # Determine experiment family from run parameters
    if use_lora or lora_weights:
        _family = 'lora-ablation'
    elif budgets and 'optimization_budget' in (mode or ['optimization']):
        _family = 'optimization-budget'
    else:
        _family = 'optimization'

    _tag_config: dict[str, Any] = {
        'dataset_type': dataset_type,
        'split_type': split_type,
        'epochs': epochs,
        'lr': lr,
        'aug_pct': aug_pct,
        'held_out_subj_idx': held_out_subj_idx,
        'held_out_emg_idx': held_out_emg_idx,
    }
    if use_lora or lora_weights:
        _tag_config.update({
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'lora_target': lora_target,
            'lora_weights': lora_weights,
        })
    _save_tag = generate_experiment_tag(dataset_type, _family, _tag_config)

    # --- Build test experiment list ---
    experiments = []
    for subj_idx in test_subjects:
        data = load_data(dataset_type, subj_idx)
        n_emgs = data['sorted_respMean'].shape[1]
        emgs = test_emg_indices if test_emg_indices is not None else range(n_emgs)
        for emg_idx in emgs:
            if emg_idx < n_emgs:
                experiments.append((subj_idx, emg_idx))

    # --- Always create per-run output directory so plots land in runs/<tag>/ ---
    run_dir = create_run_dir(_save_tag, tag=_save_tag)
    run_config = {
        'run_type': 'run_experiment',
        'experiment_tag': _save_tag,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'dataset_type': dataset_type,
        'split_type': split_type,
        'mode': mode,
        'device': device,
        'budget': budget,
        'n_reps': n_reps,
        'epochs': epochs,
        'lr': lr,
        'aug_pct': aug_pct,
        'held_out_emg_idx': held_out_emg_idx,
        'held_out_subj_idx': held_out_subj_idx,
        'budgets': budgets,
        'train_subjects': train_subject_indices,
        'test_subjects': test_subjects,
        'test_emg_indices': test_emg_indices,
        'n_experiments': len(experiments),
    }
    if lora_weights:
        run_config['lora_weights'] = lora_weights
    elif use_lora:
        run_config.update({
            'use_lora': True,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'lora_target': lora_target,
        })
    write_run_config(run_dir, run_config)
    print(f"[INFO] Run directory: {run_dir}")

    # --- Build metadata for pkl provenance ---
    _metadata: dict[str, Any] = {
        'family': _family,
        'dataset': dataset_type,
        'tag': _save_tag,
        'date': datetime.now().isoformat(timespec='seconds'),
        'run_type': 'run_experiment',
        'held_out_subj': held_out_subj_idx,
    }

    # --- Obtain finetuned model: load checkpoint or train from scratch ---
    if lora_weights:
        from models.regressors import load_lora_as_inference_model
        print("=" * 60)
        print(f"Loading LoRA checkpoint  [{lora_weights}]")
        print("=" * 60)
        ft_model, lora_cfg = load_lora_as_inference_model(
            lora_weights, device=device,
        )
        ft_model_raw = None
        if lora_cfg.get('dataset_type') and lora_cfg['dataset_type'] != dataset_type:
            print(f"[WARNING] Checkpoint trained on {lora_cfg['dataset_type']!r}, "
                  f"evaluating on {dataset_type!r}")
    else:
        print("=" * 60)
        print(f"Fine-tuning TabPFN  [{dataset_type} | {split_type} | modes={mode}]")
        print("=" * 60)

        ft_model_raw, ft_model = finetune_tabpfn(
            dataset_type,
            device=device,
            epochs=epochs,
            lr=lr,
            aug_pct=aug_pct,
            subject_indices=train_subject_indices,
            held_out_emg_idx=ft_held_out_emg,
            seed=42,
            silence_diagnostics=silence_diagnostics,
            output_dir=run_dir,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target=lora_target,
            grad_clip=grad_clip,
            n_estimators_finetune=n_estimators_finetune,
            aug_transforms=aug_transforms,
            loss_weights=loss_weights,
        )

    # --- Run each requested mode ---
    all_results = {}

    for m in mode:
        print(f"\n{'=' * 60}")
        print(f"Running mode: {m}")
        print('=' * 60)

        if m == 'optimization':
            _ft_model_type = 'lora_tabpfn' if use_lora else 'finetuned_tabpfn'
            _gp_cache_params = {
                'model': 'gp',
                'budget': budget,
                'n_reps': n_reps,
                'kappa_schedule': kappa_schedule,
                'acq_fn': acq_fn,
                'normalization': 'gp',
            }
            _ft_cache_params: dict[str, Any] = {
                'model': _ft_model_type,
                'budget': budget,
                'n_reps': n_reps,
                'epochs': epochs,
                'lr': lr,
                'aug_pct': aug_pct,
                'kappa_schedule': kappa_schedule,
                'acq_fn': acq_fn,
                'normalization': 'pfn',
            }
            if use_lora:
                _ft_cache_params.update({
                    'lora_rank': lora_rank,
                    'lora_alpha': lora_alpha,
                    'lora_target': lora_target,
                })

            results_ft, results_gp = [], []
            for subj_idx, emg_idx in experiments:
                print(f"  Optimization: subject={subj_idx}, emg={emg_idx}")

                res_gp = load_subject_result(
                    dataset_type, subj_idx, emg_idx, 'gp', _gp_cache_params
                )
                if res_gp is not None:
                    print(f"    [CACHE HIT] GP subject={subj_idx}, emg={emg_idx}")
                else:
                    res_gp = gp_baseline(dataset_type, subj_idx, emg_idx, mode='optimization',
                                          device=device, budget=budget, n_reps=n_reps,
                                          kappa_schedule=kappa_schedule,
                                          acq_fn=acq_fn, ts_temperature=ts_temperature)
                    save_subject_result(
                        res_gp, dataset_type, subj_idx, emg_idx, 'gp', _gp_cache_params
                    )

                res_ft = load_subject_result(
                    dataset_type, subj_idx, emg_idx, _ft_model_type, _ft_cache_params
                )
                if res_ft is not None:
                    print(f"    [CACHE HIT] {_ft_model_type} subject={subj_idx}, emg={emg_idx}")
                else:
                    res_ft = finetuned_optimization(dataset_type, subj_idx, emg_idx, ft_model,
                                                     device=device, budget=budget, n_reps=n_reps,
                                                     kappa_schedule=kappa_schedule,
                                                     acq_fn=acq_fn, ts_temperature=ts_temperature)
                    save_subject_result(
                        res_ft, dataset_type, subj_idx, emg_idx, _ft_model_type, _ft_cache_params
                    )

                results_ft.append(res_ft)
                results_gp.append(res_gp)
                print(f"    TabPFN R2={np.mean(res_ft['r2']):.3f}  |  GP R2={np.mean(res_gp['r2']):.3f}")

            results_dict = {'GP': results_gp, 'TabPFN': results_ft}
            r2_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            spearman_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            spearman_by_emg(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            regret_with_timing(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            regret_traces_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            regret_by_emg(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            exploration_by_subject(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            exploration_by_emg(results_dict, split_type=exp_tag, save=True, output_dir=run_dir)
            visualize_representation(results_dict, mode=f'_{exp_tag}',
                                     save=True, output_dir=run_dir)
            for _res in results_gp:
                show_emg_map(_res, model_type='GP', mode=f'_{exp_tag}',
                             save=True, output_dir=run_dir)
            for _res in results_ft:
                show_emg_map(_res, model_type='TabPFN', mode=f'_{exp_tag}',
                             save=True, output_dir=run_dir)

            all_r2 = [np.mean(r['r2']) for r in results_ft]
            print(f"\nDone. {len(results_ft)} experiments.")
            print(f"Finetuned TabPFN mean R²: {np.mean(all_r2):.3f} ± {np.std(all_r2):.3f}")

            if save:
                save_results(results_dict, 'optimization',
                             output_dir=os.path.join(run_dir, 'results'),
                             tag=_save_tag,
                             metadata=_metadata)
            all_results['optimization'] = results_dict

        elif m == 'optimization_budget':
            df = finetuned_optimization_budget(
                dataset_type, ft_model,
                device=device,
                budgets=budgets,
                test_subjects=test_subjects,
                test_emg_indices=test_emg_indices,
                split_type=exp_tag,
                output_dir=run_dir,
                kappa_schedule=kappa_schedule,
                acq_fn=acq_fn,
                ts_temperature=ts_temperature,
            )
            if save:
                results_dir = os.path.join(run_dir, 'results')
                os.makedirs(results_dir, exist_ok=True)
                pkl_path = os.path.join(results_dir, f'{_save_tag}_optimization_budget.pkl')
                df.to_pickle(pkl_path)
                print(f"Saved budget DataFrame -> {pkl_path}")
            all_results['optimization_budget'] = df

    return all_results


# ============================================
#       CLI Entry Point
# ============================================

def _load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML experiment config file.

    Args:
        path: Filesystem path to a ``.yaml`` config file.

    Returns:
        Dict of key-value pairs from the YAML document.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the YAML document is not a mapping.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(cfg).__name__}: {path}")
    return cfg


def run_finetuning():
    parser = argparse.ArgumentParser(
        description='Fine-tune TabPFN on neurostimulation data and run evaluation.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None, metavar='PATH',
                        help='Path to a YAML config file.  All keys are used as defaults; '
                             'any CLI flag that is explicitly provided overrides the YAML value.')
    parser.add_argument('--dataset', type=str, default=None, choices=['rat', 'nhp', 'spinal', '5d_rat'],
                        help='Dataset type (default: nhp)')
    parser.add_argument('--split', type=str, default='inter_subject',
                        choices=['inter_subject', 'intra_emg'],
                        help='Train/test split strategy:\n'
                             '  inter_subject — train on TRAIN_SUBJECTS, test on HELD_OUT_SUBJECTS\n'
                             '  intra_emg     — train on ALL_SUBJECTS excl. held_out_emg, test on that EMG\n'
                             '(default: inter_subject)')
    parser.add_argument('--mode', type=lambda s: s.split(','), default=None,
                        metavar='MODE[,MODE,...]',
                        help='Comma-separated evaluation modes. Valid values: '
                             'optimization, optimization_budget, '
                             'aug_sweep_optimization. '
                             '(default: optimization)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for training: cpu or cuda (default: cuda)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of fine-tuning epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--aug_pct', type=float, default=None,
                        help='Augmentation percentage (1.0 = 100%% = 10 maps per EMG; default: 2.5)')
    parser.add_argument('--budget', type=int, default=100,
                        help='BO query budget (default: 100)')
    parser.add_argument('--n_reps', type=int, default=30,
                        help='Repetitions per experiment (default: 30)')
    parser.add_argument('--held_out_emg', type=int, default=None,
                        help='EMG index to hold out; required when --split intra_emg')
    parser.add_argument('--held_out_subj', type=int, default=None,
                        help='Subject index to hold out as the sole test subject; '
                             'overrides the default HELD_OUT_SUBJECTS split when set')
    parser.add_argument('--subjects', type=str, default=None,
                        choices=['held_out', 'all'],
                        help='Subject evaluation scope:\n'
                             '  held_out (default) — test on HELD_OUT_SUBJECTS only\n'
                             '  all               — LOO cross-validation over ALL_SUBJECTS\n'
                             '                      (applies to optimization and aug_sweep modes)')
    parser.add_argument('--budgets', type=int, nargs='+', default=None,
                        help='Budget sweep values for *_budget modes (default: 10 30 50 100)')
    parser.add_argument('--aug_pct_sweep', type=float, nargs='+', default=None,
                        help='Augmentation percentages to sweep for aug_sweep_* modes '
                             '(default: 0.1 0.2 0.5 0.7 1.0 2.5). '
                             'Vanilla TabPFN (0%%) is always included as baseline.')
    parser.add_argument('--aug_transforms', type=str, nargs='+', default=None,
                        help='Spatial/response transforms to sample from per augmented map. '
                             'Valid: none h_flip v_flip d_flip y_shift. '
                             'Default: none (noise-only, no transforms).')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Persist results to output/results/ (pkl + CSV summary)')
    parser.add_argument('--cluster-diag', action='store_true', default=False,
                        dest='cluster_diag',
                        help='Print HPC efficiency summary at job end (GPU memory, walltime, '
                             'utilisation, grade, warnings). Also activated by CLUSTER_DIAG=1 '
                             'env var. Designed for SLURM .out log visibility.')
    parser.add_argument('--kappa_schedule', type=float, default=None,
                        help='UCB exploration coefficient shared by both TabPFN and GP BO loops.\n'
                             '  0.0 (default) = auto cosine-annealed schedule (GP-UCB theory)\n'
                             '  any other value = fixed kappa throughout for both models')
    parser.add_argument('--acq_fn', type=str, default=None,
                        choices=['ucb', 'ts'],
                        help="Acquisition function: 'ucb' (default) or 'ts' (Thompson Sampling).")
    parser.add_argument('--ts_temperature', type=float, default=None,
                        help='Temperature for TabPFN bar-distribution sampling in TS mode.\n'
                             '  1.0 (default) = exact predictive distribution;\n'
                             '  <1.0 = sharper/greedier; >1.0 = more uniform/exploratory.')
    parser.add_argument('--ce_loss_weight', type=float, default=None,
                        help='Finetuning objective: weight on the bar-distribution NLL term. '
                             'TabPFN default 0.0. E.g. set 1.0 (with others 0) for pure NLL.')
    parser.add_argument('--crps_loss_weight', type=float, default=None,
                        help='Finetuning objective: weight on the CRPS term. TabPFN default 1.0.')
    parser.add_argument('--crls_loss_weight', type=float, default=None,
                        help='Finetuning objective: weight on the CRLS term. TabPFN default 0.0.')
    parser.add_argument('--mse_loss_weight', type=float, default=None,
                        help='Finetuning objective: weight on the auxiliary MSE term (decoded '
                             'mean). TabPFN default 1.0. Set 0.0 to finetune without MSE.')
    parser.add_argument('--mae_loss_weight', type=float, default=None,
                        help='Finetuning objective: weight on the auxiliary MAE term. TabPFN default 0.0.')
    parser.add_argument('--diagnostics', action='store_true', default=False,
                        help='Enable gradient/CKA monitoring via GradientMonitoredRegressor '
                             '(slower finetuning, higher memory). Off by default.')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Per-parameter gradient max-norm for clipping (default: disabled). '
                             'Only active when --diagnostics or --lora is set. '
                             'Use to prevent gradient explosion (e.g. --grad_clip 1.0).')
    parser.add_argument('--n_estimators', type=int, default=None,
                        help='TabPFN ensemble size for finetuning forward/backward passes '
                             '(default: 8). Reduce to 1 on consumer GPUs to avoid VRAM '
                             'exhaustion during the internal model deepcopy.')

    # LoRA options
    parser.add_argument('--lora', action='store_true', default=False,
                        help='Use LoRA (Low-Rank Adaptation) for parameter-efficient finetuning. '
                             'Only adapter weights are trained; base model is frozen.')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='Rank of LoRA low-rank matrices (default: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA scaling factor alpha (default: 16)')
    parser.add_argument('--lora_target', type=str, default='decoder_dict',
                        choices=['decoder_dict', 'decoder_dict+mlp'],
                        help='Which layers to apply LoRA to (default: decoder_dict)')
    parser.add_argument('--lora_weights', type=str, default=None, metavar='PATH',
                        help='Path to saved LoRA checkpoint directory (containing '
                             'lora_weights.pt + lora_config.json). Skips training '
                             'and goes straight to evaluation.')

    args = parser.parse_args()

    # --- YAML config loading: load first, then let explicit CLI args override ---
    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        # For each key in the YAML, set it on args only when the user did NOT
        # explicitly pass the corresponding flag (i.e. the value is still None
        # or the argparse default).  We detect "still default" conservatively
        # by checking for None; boolean flags (store_true) default to False so
        # they are only overridden when absent from the parsed namespace.
        _bool_flags = {'save', 'diagnostics', 'lora', 'cluster_diag'}
        for key, value in yaml_cfg.items():
            if key in _bool_flags:
                # Only apply YAML value when the flag was NOT explicitly set
                # (argparse store_true defaults to False; we can't distinguish
                # "user passed --save" from "default False", so YAML wins for
                # False-valued booleans only).
                if not getattr(args, key, False):
                    setattr(args, key, value)
            elif getattr(args, key, None) is None:
                setattr(args, key, value)
        print(f"[config] Loaded YAML defaults from {args.config}")

    # Apply argparse defaults for any remaining None values
    _defaults = {
        'dataset': 'nhp',
        'split': 'inter_subject',
        'mode': ['optimization'],
        'device': 'cuda',
        'epochs': 100,
        'lr': 1e-4,
        'aug_pct': 2.5,
        'budget': 100,
        'n_reps': 30,
        'held_out_emg': None,
        'held_out_subj': None,
        'budgets': [10, 30, 50, 100],
        'aug_pct_sweep': None,
        'aug_transforms': None,
        'subjects': 'held_out',
        'kappa_schedule': 0.0,
        'acq_fn': 'ts',
        'ts_temperature': 1.0,
        'lora_rank': 8,
        'lora_alpha': 16,
        'lora_target': 'decoder_dict',
        'lora_weights': None,
    }
    for key, default in _defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)

    # Assemble finetuning loss-objective overrides. Only flags the user set (or
    # the YAML provided) are kept; absent entries are dropped so TabPFN keeps
    # its built-in defaults (CRPS + auxiliary MSE, both weight 1.0).
    _loss_weight_keys = (
        'ce_loss_weight', 'crps_loss_weight', 'crls_loss_weight',
        'mse_loss_weight', 'mae_loss_weight',
    )
    loss_weights = {
        k: float(getattr(args, k)) for k in _loss_weight_keys
        if getattr(args, k, None) is not None
    } or None

    # Activate cluster diagnostics via env var even when flag not passed
    import os as _os
    if not args.cluster_diag and _os.environ.get('CLUSTER_DIAG', '0') == '1':
        args.cluster_diag = True

    _CLI_MODES = _VALID_MODES | {'aug_sweep_optimization'}
    invalid = set(args.mode) - _CLI_MODES
    if invalid:
        parser.error(f"Invalid mode(s): {', '.join(sorted(invalid))}. "
                     f"Valid: {', '.join(sorted(_CLI_MODES))}")

    # Validate --lora_weights combinations
    if args.lora_weights:
        if 'aug_sweep_optimization' in args.mode:
            parser.error("--lora_weights is not compatible with aug_sweep_optimization "
                         "(it requires training)")
        if args.lora:
            print("[WARNING] --lora is ignored when --lora_weights is set "
                  "(config is read from lora_config.json)")

    silence_diagnostics = not args.diagnostics

    exp_modes = [m for m in args.mode if m in _VALID_MODES]

    if exp_modes:
        if args.subjects == 'all':
            loo_subjects = ALL_SUBJECTS[args.dataset]
        else:
            # None → run_experiment uses default HELD_OUT_SUBJECTS
            loo_subjects = [args.held_out_subj]

        from utils.cluster_diagnostics import ClusterDiagnostics as _CD
        with _CD(tag=f"{args.dataset}-finetuning", device=args.device,
                 n_planned=len(loo_subjects),
                 enabled=args.cluster_diag) as _diag:
            for held_subj in loo_subjects:
                run_experiment(
                    dataset_type=args.dataset,
                    split_type=args.split,
                    mode=exp_modes,
                    device=args.device,
                    budget=args.budget,
                    n_reps=args.n_reps,
                    epochs=args.epochs,
                    lr=args.lr,
                    aug_pct=args.aug_pct,
                    held_out_emg_idx=args.held_out_emg,
                    held_out_subj_idx=held_subj,
                    budgets=args.budgets,
                    save=args.save,
                    silence_diagnostics=silence_diagnostics,
                    use_lora=args.lora,
                    lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_target=args.lora_target,
                    lora_weights=args.lora_weights,
                    kappa_schedule=args.kappa_schedule,
                    grad_clip=args.grad_clip,
                    n_estimators_finetune=args.n_estimators if args.n_estimators is not None else 8,
                    aug_transforms=tuple(args.aug_transforms) if args.aug_transforms else None,
                    acq_fn=args.acq_fn,
                    ts_temperature=args.ts_temperature,
                    loss_weights=loss_weights,
                )
                _diag.record_experiment(n_completed=1)

    if 'aug_sweep_optimization' in args.mode:
        finetuned_percentage(
            dataset_type=args.dataset,
            split_type=args.split,
            device=args.device,
            budget=args.budget,
            n_reps=args.n_reps,
            epochs=args.epochs,
            lr=args.lr,
            aug_pct_list=args.aug_pct_sweep,
            held_out_emg_idx=args.held_out_emg,
            held_out_subj_idx=args.held_out_subj,
            subjects_mode=args.subjects,
            save=args.save,
            silence_diagnostics=silence_diagnostics,
            kappa_schedule=args.kappa_schedule,
            aug_transforms=tuple(args.aug_transforms) if args.aug_transforms else None,
            use_lora=args.lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target=args.lora_target,
            grad_clip=args.grad_clip,
            n_estimators_finetune=args.n_estimators if args.n_estimators is not None else 8,
            loss_weights=loss_weights,
        )


if __name__ == '__main__':
    run_finetuning()
