from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ============================================
#           Visualization
# ============================================

PALETTE = {
    'GP': 'sandybrown',
    'PFN': 'royalblue',
    'TabPFN': 'seagreen'
}


def _diag_save_dir(output_dir):
    """Return diagnostics output directory, creating it if needed."""
    base = output_dir if output_dir else os.path.join('output', 'diagnostics')
    os.makedirs(base, exist_ok=True)
    return base


def _extract_metric(diagnostics, key):
    """Extract per-epoch metric dict from diagnostics list → {layer: [values]}."""
    epochs = []
    layer_values = {}
    for d in diagnostics:
        if d['epoch'] < 0 or key not in d:
            continue
        epochs.append(d['epoch'] + 1)
        for layer, val in d[key].items():
            layer_values.setdefault(layer, []).append(val)
    return epochs, layer_values


def plot_gradient_metrics(diagnostics, save=True, output_dir=None):
    """3-panel figure: gradient norm, gradient/weight ratio, update-to-parameter ratio."""
    if not diagnostics:
        return

    metrics = [
        ('grad_norm', 'Gradient Norm (L2)'),
        ('grad_weight_ratio', 'Gradient / Weight Ratio (%)'),
        ('update_to_param_ratio', 'Update-to-Parameter Ratio (%)'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    for ax, (key, title) in zip(axes, metrics):
        epochs, layer_values = _extract_metric(diagnostics, key)
        if not epochs:
            continue
        for layer, values in sorted(layer_values.items()):
            ax.plot(epochs[:len(values)], values, marker='o', markersize=3, label=layer)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    base = _diag_save_dir(output_dir)
    if save:
        path = os.path.join(base, 'gradient_metrics.svg')
        plt.savefig(path, format='svg')
        print(f"Saved gradient metrics plot -> {path}")
    plt.close()


def plot_weight_metrics(diagnostics, save=True, output_dir=None):
    """2-panel figure: weight displacement (L2) and cosine similarity vs pretrained."""
    if not diagnostics:
        return

    metrics = [
        ('weight_displacement', 'Weight Displacement (L2 from pretrained)'),
        ('cosine_similarity', 'Cosine Similarity to Pretrained'),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    for ax, (key, title) in zip(axes, metrics):
        epochs, layer_values = _extract_metric(diagnostics, key)
        if not epochs:
            continue
        for layer, values in sorted(layer_values.items()):
            ax.plot(epochs[:len(values)], values, marker='o', markersize=3, label=layer)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if key == 'cosine_similarity':
            ax.set_ylim(bottom=0, top=1.05)

    fig.tight_layout()

    base = _diag_save_dir(output_dir)
    if save:
        path = os.path.join(base, 'weight_metrics.svg')
        plt.savefig(path, format='svg')
        print(f"Saved weight metrics plot -> {path}")
    plt.close()


def plot_cka_similarity(diagnostics, save=True, output_dir=None):
    """CKA similarity to pretrained representations per hooked layer vs epoch."""
    if not diagnostics:
        return

    epochs, layer_values = _extract_metric(diagnostics, 'cka')
    if not epochs or not layer_values:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for layer, values in layer_values.items():
        ax.plot(epochs[:len(values)], values, marker='o', markersize=4, label=layer)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CKA')
    ax.set_title('CKA Similarity to Pretrained Representations')
    ax.set_ylim(bottom=0, top=1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    base = _diag_save_dir(output_dir)
    if save:
        path = os.path.join(base, 'cka_similarity.svg')
        plt.savefig(path, format='svg')
        print(f"Saved CKA similarity plot -> {path}")
    plt.close()


def _aug_label(v):
    """Human-readable x-axis label for an n_aug value."""
    if v == 0:
        return 'Vanilla'
    elif 0 < v < 1:
        return f'{int(round(v * 100))}%'
    else:
        return str(int(round(v)))


def _normalize_results_dict(first_arg, second_arg=None):
    """
    Backward-compatible helper: accept either a results_dict (dict mapping
    model name -> list of result dicts) or the old positional (gp_results,
    pfn_results) signature.

    Returns a dict[str, list].
    """
    if isinstance(first_arg, dict):
        return first_arg
    # Legacy two-list call
    results = {'GP': first_arg}
    if second_arg is not None:
        results['PFN'] = second_arg
    return results


def r2_per_muscle(results_dict_or_gp, pfn_results=None, mode='', save=False, output_dir=None, eval_type='fit'):
    """Bar plot of R² per subject-EMG pair (muscle), one bar per model."""
    results_dict = _normalize_results_dict(results_dict_or_gp, pfn_results)

    data = []
    n_experiments = 0

    for model_name, results_list in results_dict.items():
        n_experiments = max(n_experiments, len(results_list))
        for res in results_list:
            for score in res['r2']:
                data.append({
                    'muscle': f"S{res['subject']} EMG {res['emg']}",
                    'R2': score,
                    'Model': model_name
                })

    df = pd.DataFrame(data)
    plt.figure(figsize=(1.4 * n_experiments, 6))
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    sns.barplot(data=df, x='muscle', y='R2', hue='Model', palette=PALETTE,
                errorbar=('ci', 95))

    model_names = ' vs '.join(results_dict.keys())
    plt.title(f"R2 Score Comparison: {model_names}")

    # Determine dataset from first available result
    first_results = next(iter(results_dict.values()))
    base = os.path.join(output_dir, 'fitness') if output_dir else \
           os.path.join('output', 'fitness', first_results[0]['dataset'])
    os.makedirs(base, exist_ok=True)
    plot_path = os.path.join(base, f'r2_comparison{mode}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


# Backward-compatible alias
r2_comparison = r2_per_muscle


def show_emg_map(results, idx, model_type, mode='', save=False, output_dir=None, eval_type='fit'):
    res = results[idx]

    y_true = res['y_test']
    y_pred = res['y_pred']
    r2_score = np.mean(np.array(res['r2']))

    # Determine Grid Shape
    n_channels = len(y_true)
    if n_channels == 100: grid_shape = (10, 10)
    elif n_channels == 64: grid_shape = (8, 8)
    elif n_channels == 32: grid_shape = (4, 8)
    else: grid_shape = (1, n_channels)

    v_min = min(y_true.min(), y_pred.min())
    v_max = max(y_true.max(), y_pred.max())

    map_true = y_true.reshape(grid_shape)
    map_pred = y_pred.reshape(grid_shape)

    max_idx_true = np.unravel_index(np.argmax(map_true), grid_shape)
    max_idx_pred = np.unravel_index(np.argmax(map_pred), grid_shape)

    dataset, subject, emg = res['dataset'], res['subject'], res['emg']

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_type} EMG Map | {dataset} Subj {subject} EMG {emg}')

    heatmap_kwargs = {
        'cmap': 'viridis',
        'vmin': v_min,
        'vmax': v_max,
    }

    sns.heatmap(map_true, ax=ax[0], **heatmap_kwargs)
    ax[0].set_title(f"Ground Truth (S{subject} EMG{emg})")
    ax[0].plot(max_idx_true[1] + 0.5, max_idx_true[0] + 0.5, 'ro', markersize=8)

    sns.heatmap(map_pred, ax=ax[1], **heatmap_kwargs)
    ax[1].set_title(f"Prediction | R2:{r2_score:.2f}")
    ax[1].plot(max_idx_pred[1] + 0.5, max_idx_pred[0] + 0.5, 'ro', markersize=8)

    if eval_type == 'optimization':
        base = os.path.join(output_dir, 'optimization', 'emg_maps') if output_dir else \
               os.path.join('output', 'optimization', 'emg_maps')
    else:
        base = os.path.join(output_dir, 'fitness', 'emg_maps') if output_dir else \
               os.path.join('output', 'fitness', dataset, 'emg_maps')
    os.makedirs(base, exist_ok=True)
    plot_path = os.path.join(base, f'emg_map_{dataset}_s{subject}_emg{emg}_{model_type}{mode}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def _infer_grid_shape(n_channels):
    """Return (rows, cols) for a given number of channels."""
    if n_channels == 100: return (10, 10)
    elif n_channels == 96: return (8, 12)
    elif n_channels == 64: return (8, 8)
    elif n_channels == 32: return (4, 8)
    else: return (1, n_channels)


def visualize_representation(results_dict, mode='', save=False, output_dir=None):
    """
    Heatmap grid showing model predictions evolving across BO iterations.

    Row 0: ground truth (repeated across columns).
    Rows 1+: one row per model, columns at log2-spaced snapshot iterations.
    Each cell shows the predicted EMG map; subtitle shows R².

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts.
                      Each result dict must have 'snapshots' (from optimization mode).
        mode: string suffix for the output filename.
        save: whether to save the figure to disk.
        output_dir: run directory (saves under optimization/emg_maps/).
    """
    results_dict = _normalize_results_dict(results_dict)

    # Pick ONE random experiment index (same for all models)
    first_results = next(iter(results_dict.values()))
    n_experiments = len(first_results)

    # Find an experiment where at least one model has snapshots
    candidates = []
    for idx in range(n_experiments):
        has_snap = any(
            results_list[idx].get('snapshots') is not None
            for results_list in results_dict.values()
        )
        if has_snap:
            candidates.append(idx)

    if not candidates:
        print("[visualize_representation] No snapshots available, skipping.")
        return

    idx = candidates[np.random.randint(len(candidates))]

    # Collect snapshot iterations (union across models)
    all_iters = set()
    for results_list in results_dict.values():
        snaps = results_list[idx].get('snapshots')
        if snaps:
            all_iters.update(snaps.keys())
    snapshot_iters = sorted(all_iters)

    if not snapshot_iters:
        return

    ref_res = first_results[idx]
    y_test = ref_res['y_test']
    grid_shape = _infer_grid_shape(len(y_test))
    v_min, v_max = float(y_test.min()), float(y_test.max())

    model_names = list(results_dict.keys())
    n_models = len(model_names)
    n_cols = len(snapshot_iters)
    n_rows = 1 + n_models  # ground truth row + one row per model

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows),
                             squeeze=False)

    heatmap_kw = dict(cmap='viridis', vmin=v_min, vmax=v_max,
                      cbar=False, xticklabels=False, yticklabels=False)

    subject = ref_res.get('subject', '?')
    emg = ref_res.get('emg', '?')

    # Ground truth optimum location (shared across all columns)
    gt_idx = int(np.argmax(y_test))
    gt_row_2d, gt_col_2d = gt_idx // grid_shape[1], gt_idx % grid_shape[1]

    # Row 0: ground truth
    for col in range(n_cols):
        ax = axes[0, col]
        sns.heatmap(y_test.reshape(grid_shape), ax=ax, **heatmap_kw)
        ax.plot(gt_col_2d + 0.5, gt_row_2d + 0.5, 'g*', markersize=10,
                markeredgecolor='white', markeredgewidth=0.5)
        if col == 0:
            ax.set_ylabel(f'Ground Truth\nS{subject} EMG{emg}', fontsize=8)
        ax.set_title(f'Iter {snapshot_iters[col]}', fontsize=8)

    # Rows 1+: model predictions
    for row_i, model_name in enumerate(model_names):
        snaps = results_dict[model_name][idx].get('snapshots')
        for col, it in enumerate(snapshot_iters):
            ax = axes[1 + row_i, col]
            if snaps and it in snaps:
                pred = snaps[it]['y_pred']
                r2_val = snaps[it]['r2']
                sns.heatmap(pred.reshape(grid_shape), ax=ax, **heatmap_kw)
                opt_idx = int(np.argmax(pred.ravel()))
                opt_row_2d = opt_idx // grid_shape[1]
                opt_col_2d = opt_idx % grid_shape[1]
                ax.plot(opt_col_2d + 0.5, opt_row_2d + 0.5, 'r*', markersize=10,
                        markeredgecolor='white', markeredgewidth=0.5)
                ax.set_title(f'R²={r2_val:.2f}', fontsize=7)
            else:
                ax.set_visible(False)
            if col == 0:
                ax.set_ylabel(model_name, fontsize=8)

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=v_min, vmax=v_max))
    fig.colorbar(sm, cax=cbar_ax)

    dataset = ref_res.get('dataset', '')
    fig.suptitle(f'Representation Evolution | {dataset} S{subject} EMG{emg}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.89, 0.95])

    base = os.path.join(output_dir, 'optimization', 'emg_maps') if output_dir else \
           os.path.join('output', 'optimization', 'emg_maps')
    os.makedirs(base, exist_ok=True)
    plot_path = os.path.join(base, f'visualize_representation{mode}.svg')
    if save:
        plt.savefig(plot_path, format='svg')
        print(f"Saved plot to {plot_path}")

    plt.close()


def r2_by_subject(results_dict, split_type='', save=False, output_dir=None,
                  output_subdir='fitness'):
    """Box plot of R² values grouped by subject index, one bar per model.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
        output_subdir: subfolder under ``output_dir`` to write into.
            Use ``'fitness'`` for fit-mode results and ``'optimization'`` for
            optimization-mode results.
    """
    results_dict = _normalize_results_dict(results_dict)

    data = []
    for model_name, results_list in results_dict.items():
        for res in results_list:
            for score in res['r2']:
                data.append({
                    'Subject': f"S{res['subject']}",
                    'R2': score,
                    'Model': model_name
                })

    df = pd.DataFrame(data)
    n_subjects = df['Subject'].nunique()
    plt.figure(figsize=(max(6, 1.8 * n_subjects), 5))
    sns.boxplot(data=df, x='Subject', y='R2', hue='Model', palette=PALETTE)
    plt.ylim(0, 1)
    plt.title("R² by Subject")
    plt.xlabel("Subject")
    plt.ylabel("R²")
    plt.legend(title='Model')
    plt.grid(True, alpha=0.3, axis='y')

    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', '')
    base = os.path.join(output_dir, output_subdir) if output_dir else \
           os.path.join('output', output_subdir, dataset)
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'r2_by_subject{suffix}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def _final_normalized_regret(res: dict) -> np.ndarray:
    """Compute per-rep final simple regret normalized by the response range.

    Uses the running best (not the last queried value) so the metric is
    monotonically non-increasing.  Normalizes by ``y_test`` range so results
    are comparable across subjects/EMGs regardless of absolute scale.

    Returns:
        1-D array of shape [n_reps] with values in [0, 1].
        0 = optimum found; 1 = never improved beyond the worst observed.
    """
    y_range = float(res['y_test'].max() - res['y_test'].min())
    if y_range < 1e-8:
        return np.array([])
    optimal = float(res['y_test'].max())
    running_best = np.maximum.accumulate(np.array(res['values']), axis=1)  # [n_reps, budget]
    best_found = running_best[:, -1]                                        # [n_reps]
    return (optimal - best_found) / y_range                                 # [n_reps]



def regret_by_emg(results_dict, split_type='', save=False, output_dir=None):
    """Box plot of final simple regret grouped by EMG index.

    Regret is normalized by the response range so values are comparable across
    EMG channels with different absolute magnitudes.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
                      (optimization mode; each result must have 'values' and 'y_test')
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
    """
    results_dict = _normalize_results_dict(results_dict)

    data = []
    for model_name, results_list in results_dict.items():
        for res in results_list:
            if 'values' not in res:
                continue
            final_regrets = _final_normalized_regret(res)
            for regret in final_regrets:
                data.append({
                    'EMG': f"EMG {res['emg']}",
                    'Normalized Regret': float(regret),
                    'Model': model_name
                })

    if not data:
        return

    df = pd.DataFrame(data)
    n_emgs = df['EMG'].nunique()
    plt.figure(figsize=(max(6, 1.8 * n_emgs), 5))
    sns.boxplot(data=df, x='EMG', y='Normalized Regret', hue='Model', palette=PALETTE)
    plt.ylim(bottom=0)
    plt.title("Final Simple Regret by EMG Channel")
    plt.xlabel("EMG")
    plt.ylabel("Final Simple Regret\n(normalized by response range, lower is better)")
    plt.legend(title='Model')
    plt.grid(True, alpha=0.3, axis='y')

    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', '')
    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'regret_by_emg{suffix}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def regret_by_subject(results_dict, split_type='', save=False, output_dir=None):
    """Box plot of final simple regret grouped by subject index.

    Regret is normalized by the response range so values are comparable across
    subjects with different absolute magnitudes.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
                      (optimization mode; each result must have 'values' and 'y_test')
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
        output_dir: run-level directory (saves under optimization/)
    """
    results_dict = _normalize_results_dict(results_dict)

    data = []
    for model_name, results_list in results_dict.items():
        for res in results_list:
            if 'values' not in res:
                continue
            final_regrets = _final_normalized_regret(res)
            for regret in final_regrets:
                data.append({
                    'Subject': f"S{res['subject']}",
                    'Normalized Regret': float(regret),
                    'Model': model_name,
                })

    if not data:
        return

    df = pd.DataFrame(data)
    n_subjects = df['Subject'].nunique()
    plt.figure(figsize=(max(6, 1.8 * n_subjects), 5))
    sns.boxplot(data=df, x='Subject', y='Normalized Regret', hue='Model', palette=PALETTE)
    plt.ylim(bottom=0)
    plt.title('Final Simple Regret by Subject')
    plt.xlabel('Subject')
    plt.ylabel('Final Simple Regret\n(normalized by response range, lower is better)')
    plt.legend(title='Model')
    plt.grid(True, alpha=0.3, axis='y')

    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', '')
    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'regret_by_subject{suffix}.svg')
    if save:
        plt.savefig(plot_path, format='svg')
        print(f'Saved plot to {plot_path}')

    plt.close()


def regret_curve(results_dict, split_type='', save=False, output_dir=None):
    """Average regret curve across all subjects and EMGs, with 95% CI shading.

    Aggregates the running-best regret trajectory across every experiment
    (subject × EMG pair) and every repetition.  Each experiment's regret is
    range-normalized before averaging so that channels with different absolute
    magnitudes contribute equally.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
                      (optimization mode; must have 'values' and 'y_test').
        split_type: string suffix for the output filename.
        save: whether to save the figure to disk.
        output_dir: run-level directory (saves under optimization/).
    """
    results_dict = _normalize_results_dict(results_dict)

    fig, ax = plt.subplots(figsize=(9, 5))

    dataset = ''
    for model_name, results_list in results_dict.items():
        color = PALETTE.get(model_name, 'gray')
        per_exp_mean_curves = []

        for res in results_list:
            if 'values' not in res or 'y_test' not in res:
                continue
            dataset = res.get('dataset', dataset)
            y_range = float(res['y_test'].max() - res['y_test'].min())
            if y_range < 1e-8:
                continue
            optimal = float(res['y_test'].max())
            running_best = np.maximum.accumulate(
                np.array(res['values']), axis=1
            )                                                  # [n_reps, budget]
            regret_norm = (optimal - running_best) / y_range  # [n_reps, budget]
            per_exp_mean_curves.append(np.mean(regret_norm, axis=0))  # [budget]

        if not per_exp_mean_curves:
            continue

        # Pad to common length (in case budgets differ across experiments)
        max_len = max(len(c) for c in per_exp_mean_curves)
        padded = np.array([
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in per_exp_mean_curves
        ])                                                     # [n_exp, budget]

        mean_curve = np.nanmean(padded, axis=0)               # [budget]
        # SE across experiments (not reps — reps already averaged per experiment)
        n_exp = np.sum(~np.isnan(padded), axis=0)
        se_curve = np.nanstd(padded, axis=0) / np.sqrt(np.maximum(n_exp, 1))

        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, color=color, linewidth=2.5, label=model_name)
        ax.fill_between(
            x,
            mean_curve - 1.96 * se_curve,
            mean_curve + 1.96 * se_curve,
            color=color, alpha=0.2,
        )

    ax.set_xlabel('BO Iteration')
    ax.set_ylabel('Simple Regret\n(normalized by response range, lower is better)')
    ax.set_ylim(bottom=0)
    ax.set_title(f'Average Regret Curve — All Subjects & EMGs ({dataset})')
    ax.legend(title='Model')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'regret_curve{suffix}.svg')
    if save:
        plt.savefig(plot_path, format='svg')
        print(f"Saved plot to {plot_path}")

    plt.close()


def kappa_regret_curves(
    kappa_results: dict,
    gp_results: list,
    dataset: str = '',
    split_type: str = '',
    save: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """Aggregated regret curves for each tested kappa value, with GP reference.

    Each curve shows mean ± 95 % CI normalized regret (by response range)
    averaged across all experiments (subjects × EMGs) and all repetitions.
    A sequential colormap encodes kappa magnitude; the GP reference is shown
    as a dashed line in its canonical colour.

    Args:
        kappa_results: ``{kappa_val: list[result_dict]}`` — TabPFN results for
            each tested fixed kappa value.
        gp_results: List of result dicts for the GP reference (kappa_schedule=0.0).
        dataset: Dataset name used in title and output filename.
        split_type: String suffix appended to the output filename.
        save: If True, save the figure to disk as SVG.
        output_dir: Run-level directory; figure is saved under
            ``{output_dir}/optimization/``.
    """
    kappa_vals = sorted(kappa_results.keys())
    n_kappas = len(kappa_vals)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.plasma
    colors = [cmap(i / max(n_kappas - 1, 1)) for i in range(n_kappas)]

    def _mean_regret_curve(results_list):
        per_exp = []
        for res in results_list:
            if 'values' not in res or 'y_test' not in res:
                continue
            y_range = float(res['y_test'].max() - res['y_test'].min())
            if y_range < 1e-8:
                continue
            optimal = float(res['y_test'].max())
            rb = np.maximum.accumulate(np.array(res['values']), axis=1)  # [n_reps, budget]
            per_exp.append(np.mean((optimal - rb) / y_range, axis=0))    # [budget]
        if not per_exp:
            return None, None, None
        max_len = max(len(c) for c in per_exp)
        padded = np.array([
            np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
            for c in per_exp
        ])                                                                # [n_exp, budget]
        mean_c = np.nanmean(padded, axis=0)
        n_exp_arr = np.sum(~np.isnan(padded), axis=0)
        se_c = np.nanstd(padded, axis=0) / np.sqrt(np.maximum(n_exp_arr, 1))
        return np.arange(len(mean_c)), mean_c, se_c

    for kappa_val, color in zip(kappa_vals, colors):
        x, mean_c, se_c = _mean_regret_curve(kappa_results[kappa_val])
        if mean_c is None:
            continue
        label = f'κ={kappa_val:.2g}'
        ax.plot(x, mean_c, color=color, linewidth=2, label=label)
        ax.fill_between(x, mean_c - 1.96 * se_c, mean_c + 1.96 * se_c,
                        color=color, alpha=0.15)

    if gp_results:
        x, mean_c, se_c = _mean_regret_curve(gp_results)
        if mean_c is not None:
            ax.plot(x, mean_c, color=PALETTE.get('GP', 'sandybrown'),
                    linewidth=2.5, linestyle='--', label='GP (auto-κ)')
            ax.fill_between(x, mean_c - 1.96 * se_c, mean_c + 1.96 * se_c,
                            color=PALETTE.get('GP', 'sandybrown'), alpha=0.15)

    ax.set_xlabel('BO Iteration')
    ax.set_ylabel('Simple Regret\n(normalized by response range, lower is better)')
    ax.set_ylim(bottom=0)
    ax.set_title(f'Kappa Search — Aggregated Regret Curves ({dataset})')
    ax.legend(title='Model / κ', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'kappa_search_regret{suffix}.svg')
    if save:
        plt.savefig(plot_path, format='svg')
        print(f'Saved plot to {plot_path}')
    plt.close()


def kappa_auc_bar(
    auc_df: "pd.DataFrame",
    gp_auc: float,
    dataset: str = '',
    split_type: str = '',
    save: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """Bar chart of AUC scores per kappa value, with GP reference line.

    Lower AUC = faster convergence to the optimum.  The GP reference is shown
    as a horizontal dashed line to allow direct comparison.

    Args:
        auc_df: DataFrame with columns ``['kappa', 'mean_auc', 'std_auc']``.
            Rows correspond to individual TabPFN fixed-kappa runs.
        gp_auc: Mean AUC for the GP reference (single float).
        dataset: Dataset name for title and filename.
        split_type: String suffix appended to the output filename.
        save: If True, save the figure to disk as SVG.
        output_dir: Run-level directory; figure is saved under
            ``{output_dir}/optimization/``.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    kappa_vals = auc_df['kappa'].tolist()
    means = auc_df['mean_auc'].tolist()
    stds = auc_df['std_auc'].tolist()
    x = np.arange(len(kappa_vals))

    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(kappa_vals) - 1, 1)) for i in range(len(kappa_vals))]

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
                  error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

    if gp_auc > 0:
        ax.axhline(gp_auc, color=PALETTE.get('GP', 'sandybrown'),
                   linewidth=2, linestyle='--', label=f'GP ref (AUC={gp_auc:.3f})')
        ax.legend(fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{k:.2g}' for k in kappa_vals])
    ax.set_xlabel('Fixed κ')
    ax.set_ylabel('Mean AUC (norm. regret, lower is better)')
    ax.set_title(f'Kappa Search — AUC Scores ({dataset})')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'kappa_search_auc{suffix}.svg')
    if save:
        plt.savefig(plot_path, format='svg')
        print(f'Saved plot to {plot_path}')
    plt.close()


def budget_sweep_plot(df, eval_type, dataset='', split_type='', save=False, output_dir=None):
    """
    Budget sweep with per-subject light traces and bold cross-subject mean.

    Light semi-transparent lines show each subject's mean score at each budget;
    the bold line shows the mean across subjects +/- SE. One trace per model.

    Args:
        df: DataFrame with columns Budget, Model, ID, and R2 and/or Regret.
            ID format: '{subject}_{emg}'.
        eval_type: 'fit' -> R² panel; 'optimization' -> Regret panel (+ R² if present).
        dataset, split_type, save, output_dir: plotting options.
    """
    df = df.copy()
    df['Subject'] = df['ID'].str.split('_').str[0]

    metrics = []
    if eval_type == 'fit' and 'R2' in df.columns:
        metrics = [('R2', 'R² Score', (0, 1.05), 'R² vs Budget')]
    elif eval_type == 'optimization':
        if 'R2' in df.columns:
            metrics.append(('R2', 'R² Score', (0, 1.05), 'R² vs Budget'))
        if 'Regret' in df.columns:
            metrics.append(('Regret', 'Final Simple Regret (%)', None, 'Regret vs Budget'))

    if not metrics:
        return

    n_panels = len(metrics)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 5 * n_panels), squeeze=False)

    models = sorted(df['Model'].unique())
    for ax, (y_col, y_label, ylim, panel_title) in zip(axes[:, 0], metrics):
        for model in models:
            color = PALETTE.get(model, 'gray')
            mdf = df[df['Model'] == model]

            subj_means = mdf.groupby(['Subject', 'Budget'])[y_col].mean().reset_index()
            for _, sdf in subj_means.groupby('Subject'):
                sdf = sdf.sort_values('Budget')
                ax.plot(sdf['Budget'], sdf[y_col],
                        color=color, alpha=0.25, linewidth=1, label='_nolegend_')

            grand = subj_means.groupby('Budget')[y_col].agg(['mean', 'sem']).reset_index()
            grand = grand.sort_values('Budget')
            ax.plot(grand['Budget'], grand['mean'],
                    color=color, linewidth=2.5, marker='o', markersize=5, label=model)
            ax.fill_between(grand['Budget'],
                            grand['mean'] - grand['sem'],
                            grand['mean'] + grand['sem'],
                            color=color, alpha=0.2)

        ax.set_xlabel('Budget')
        ax.set_ylabel(y_label)
        ax.set_title(f'{panel_title} ({dataset})')
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Model')

    fig.tight_layout()

    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    if eval_type == 'fit':
        base = os.path.join(output_dir, 'fitness') if output_dir else \
               os.path.join('output', 'fitness', dataset)
        plot_path = os.path.join(base, f'budget_sweep_fit{suffix}.svg')
    else:
        base = os.path.join(output_dir, 'optimization') if output_dir else \
               os.path.join('output', 'optimization')
        plot_path = os.path.join(base, f'budget_sweep_optimization{suffix}.svg')

    os.makedirs(base, exist_ok=True)
    if save:
        plt.savefig(plot_path, format='svg')
        print(f"Saved plot to {plot_path}")

    plt.close()


def regret_with_timing(results_dict, split_type='', save=False, output_dir=None):
    """
    2-row figure: top = regret curves (95% CI bands), bottom = per-step inference time.
    One column per experiment (subject/EMG pair).

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
                      (optimization mode; each result must have 'values', 'y_test', 'times')
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
    """
    results_dict = _normalize_results_dict(results_dict)

    first_results = next(iter(results_dict.values()))
    n_experiments = len(first_results)

    def get_regret_stats(values_list, optimal_val):
        raw_vals = np.maximum.accumulate(np.array(values_list), axis=1)  # running best
        regret_all = (optimal_val - raw_vals) / optimal_val * 100
        mean_regret = np.mean(regret_all, axis=0)
        se_regret = np.std(regret_all, axis=0) / np.sqrt(raw_vals.shape[0])
        return mean_regret, se_regret

    fig, axes = plt.subplots(3, n_experiments,
                             figsize=(4 * n_experiments, 12),
                             squeeze=False)

    for idx in range(n_experiments):
        ax_reg = axes[0, idx]
        ax_time = axes[1, idx]
        ax_ee = axes[2, idx]

        ref_res = first_results[idx]
        optimal_val = ref_res['y_test'].max()

        for model_name, results_list in results_dict.items():
            res = results_list[idx]
            color = PALETTE.get(model_name, 'gray')

            # --- regret row ---
            if 'values' in res:
                mean_reg, se_reg = get_regret_stats(res['values'], optimal_val)
                x_axis = range(len(mean_reg))
                ax_reg.plot(x_axis, mean_reg, color=color, label=model_name, linewidth=2)
                ax_reg.fill_between(x_axis,
                                    mean_reg - 1.96 * se_reg,
                                    mean_reg + 1.96 * se_reg,
                                    color=color, alpha=0.2)

            # --- timing row ---
            times = res['times']
            if np.ndim(times) == 0:
                # scalar — skip per-step timing
                ax_time.axhline(float(times), color=color, linewidth=2,
                                label=model_name, linestyle='--')
            else:
                times_arr = np.array(times)
                ax_time.plot(times_arr, color=color, linewidth=2, label=model_name)

            # --- exploration score row ---
            # exploration(t) = max(model.predict(X_pool)[0]) at iter t / optimal
            # best_pred_val is the model's predicted maximum over the full candidate pool.
            snaps = res.get('snapshots')
            if snaps and optimal_val > 1e-8:
                iters = sorted(snaps.keys())
                scores = [
                    snaps[it]['best_pred_val'] / optimal_val
                    for it in iters
                ]
                ax_ee.plot(iters, scores, color=color, linewidth=2,
                           marker='o', markersize=4, label=model_name)

        ax_reg.set_title(f"S{ref_res['subject']} EMG {ref_res['emg']}", fontsize=9)
        ax_reg.set_xlabel('Iteration')
        ax_reg.grid(True, alpha=0.3)
        ax_reg.set_ylim(bottom=0)

        ax_time.set_xlabel('Iteration')
        ax_time.set_ylabel('Time (s)')
        ax_time.grid(True, alpha=0.3)

        budget = len(ref_res['values'][0]) if 'values' in ref_res else 0
        ax_ee.set_xlabel('Iteration')
        ax_ee.set_ylim(0, 1.05)
        if budget:
            ax_ee.set_xlim(0, budget)
            ax_ee.xaxis.set_major_locator(plt.MultipleLocator(max(1, budget // 10)))
        ax_ee.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
        ax_ee.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel('Recommendation Regret (%)')
    axes[1, 0].set_ylabel('Inference Time (s)')
    axes[2, 0].set_ylabel('Exploration Score\n(max predicted / optimal)')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=9)

    dataset = first_results[0].get('dataset', '')
    fig.suptitle(f'Regret & Inference Time | {dataset}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'regret_timing{suffix}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def augmentation_sweep_plot(df, eval_type, dataset='', split_type='', save=False, output_dir=None):
    """
    Point plot of R² and (optionally) Final Regret vs number of augmentations.

    n_aug=0 represents vanilla TabPFN (no finetuning); n_aug>0 represents
    finetuned TabPFN with that many augmentations per subject-EMG pair.

    Args:
        df: DataFrame with columns n_aug, R2, (Regret), ID
        eval_type: 'fit' (R² only) or 'optimization' (R² + Regret, 2 rows)
        dataset: dataset name used for output path and title
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
    """
    color = sns.color_palette("muted")[0]

    # Build ordered x-axis labels: 0 → 'Vanilla', fractions → 'X%', integers as-is
    aug_values = sorted(df['n_aug'].unique())
    x_labels = [_aug_label(v) for v in aug_values]

    # Map numeric n_aug to display label for plotting
    df = df.copy()
    label_map = {v: _aug_label(v) for v in aug_values}
    df['Aug'] = df['n_aug'].map(label_map)

    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'

    if eval_type == 'fit':
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        sns.pointplot(data=df, x='Aug', y='R2', order=x_labels,
                      color=color, capsize=0.15, errorbar=('ci', 95), ax=ax)
        ax.set_xlabel('Number of Augmentations')
        ax.set_ylabel('R² Score')
        ax.set_ylim(0, 1)
        ax.set_title(f'R² vs Augmentations ({dataset})')
        ax.grid(True, alpha=0.3, axis='y')

        base = os.path.join(output_dir, 'fitness') if output_dir else \
               os.path.join('output', 'fitness', dataset)
        os.makedirs(base, exist_ok=True)
        plot_path = os.path.join(base, f'aug_sweep_fit{suffix}.svg')

    else:  # optimization
        fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

        sns.pointplot(data=df, x='Aug', y='R2', order=x_labels,
                      color=color, capsize=0.15, errorbar=('ci', 95), ax=axes[0])
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f'R² vs Augmentations ({dataset})')
        axes[0].grid(True, alpha=0.3, axis='y')

        sns.pointplot(data=df, x='Aug', y='Regret', order=x_labels,
                      color=color, capsize=0.15, errorbar=('ci', 95), ax=axes[1])
        axes[1].set_ylabel('Final Simple Regret')
        axes[1].set_xlabel('Number of Augmentations')
        axes[1].set_title(f'Final Regret vs Augmentations ({dataset})')
        axes[1].grid(True, alpha=0.3, axis='y')

        fig.tight_layout()

        base = os.path.join(output_dir, 'optimization') if output_dir else \
               os.path.join('output', 'optimization')
        os.makedirs(base, exist_ok=True)
        plot_path = os.path.join(base, f'aug_sweep_optimization{suffix}.svg')

    if save:
        plt.savefig(plot_path, format='svg')
        print(f"Saved plot to {plot_path}")

    plt.close()
