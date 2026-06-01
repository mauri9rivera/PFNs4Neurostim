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
    'TabPFN': 'seagreen',
    'LoRA TabPFN': 'seagreen',
    'Full FT': 'royalblue',
}


def _to_grid(values: np.ndarray, ch2xy: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """Scatter ``values`` (one per real electrode) into a NaN-padded display grid.

    ``ch2xy[i] = (row, col)`` is the grid position of the *i-th* electrode in
    ``values``.  Cells without an electrode stay NaN; downstream heatmaps
    render those as the colormap's "bad colour" (set by ``cmap.set_bad``)
    so empty grid positions are visibly absent rather than spuriously zero.
    """
    img = np.full(tuple(grid_shape), np.nan, dtype=float)
    values = np.asarray(values).ravel()
    for i, (r, c) in enumerate(ch2xy):
        img[int(r), int(c)] = values[i]
    return img


def _grid_argmax(values: np.ndarray, ch2xy: np.ndarray) -> tuple:
    """Return ``(row, col)`` of the maximum value, looked up via ``ch2xy``."""
    flat_idx = int(np.argmax(np.asarray(values).ravel()))
    return int(ch2xy[flat_idx, 0]), int(ch2xy[flat_idx, 1])


def show_emg_map(
    result: dict,
    model_type: str,
    mode: str = '',
    save: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """Side-by-side heatmap: ground truth vs surrogate final prediction on the EMG grid.

    Args:
        result: Result dict from gp_baseline / finetuned_optimization /
            evaluate_optimization.  Must contain ``'y_test'``, ``'y_pred'``,
            ``'r2'``, ``'ch2xy'``, ``'grid_shape'``, ``'dataset'``,
            ``'subject'``, ``'emg'``.
        model_type: Label for the plot title and filename (e.g. ``'GP'``,
            ``'TabPFN'``).
        mode: Optional suffix appended to the output filename.
        save: If True, write SVG under ``<output_dir>/optimization/emg_maps/``.
        output_dir: Run directory root.
    """
    ch2xy = result.get('ch2xy')
    grid_shape = result.get('grid_shape')
    if ch2xy is None or grid_shape is None:
        return

    y_true = np.asarray(result['y_test'])
    y_pred = np.asarray(result['y_pred'])
    r2_val = float(np.mean(result['r2']))
    dataset = result['dataset']
    subject = result['subject']
    emg = result['emg']

    map_true = _to_grid(y_true, ch2xy, grid_shape)   # [R, C]
    map_pred = _to_grid(y_pred, ch2xy, grid_shape)   # [R, C]

    vmin = float(np.nanmin([map_true, map_pred]))
    vmax = float(np.nanmax([map_true, map_pred]))

    max_r_true, max_c_true = _grid_argmax(y_true, ch2xy)
    max_r_pred, max_c_pred = _grid_argmax(y_pred, ch2xy)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_type} — {dataset} Subj {subject} EMG {emg}')

    hm_kw = dict(cmap=cmap, vmin=vmin, vmax=vmax)
    sns.heatmap(map_true, ax=axes[0], mask=np.isnan(map_true), **hm_kw)
    axes[0].set_title('Ground Truth')
    axes[0].plot(max_c_true + 0.5, max_r_true + 0.5, 'ro', markersize=8)

    sns.heatmap(map_pred, ax=axes[1], mask=np.isnan(map_pred), **hm_kw)
    axes[1].set_title(f'Prediction  R²={r2_val:.3f}')
    axes[1].plot(max_c_pred + 0.5, max_r_pred + 0.5, 'ro', markersize=8)

    fig.tight_layout()

    if save:
        base = (
            os.path.join(output_dir, 'optimization', 'emg_maps')
            if output_dir
            else os.path.join('output', 'optimization', 'emg_maps')
        )
        os.makedirs(base, exist_ok=True)
        fname = f'emg_map_{dataset}_s{subject}_emg{emg}_{model_type}{mode}.svg'
        path = os.path.join(base, fname)
        plt.savefig(path, format='svg')
        print(f"Saved EMG map -> {path}")
    plt.close()


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
    y_test = np.asarray(ref_res['y_test'])
    ch2xy = ref_res['ch2xy']
    grid_shape = ref_res['grid_shape']
    v_min, v_max = float(y_test.min()), float(y_test.max())

    model_names = list(results_dict.keys())
    n_models = len(model_names)
    n_cols = len(snapshot_iters)
    n_rows = 1 + n_models  # ground truth row + one row per model

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows),
                             squeeze=False)

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgrey')
    heatmap_kw = dict(cmap=cmap, vmin=v_min, vmax=v_max,
                      cbar=False, xticklabels=False, yticklabels=False)

    subject = ref_res.get('subject', '?')
    emg = ref_res.get('emg', '?')

    # Ground truth optimum location (shared across all columns)
    gt_row_2d, gt_col_2d = _grid_argmax(y_test, ch2xy)
    y_test_grid = _to_grid(y_test, ch2xy, grid_shape)

    # Row 0: ground truth
    for col in range(n_cols):
        ax = axes[0, col]
        sns.heatmap(y_test_grid, ax=ax, **heatmap_kw)
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
                pred = np.asarray(snaps[it]['y_pred'])
                r2_val = snaps[it]['r2']
                sns.heatmap(_to_grid(pred, ch2xy, grid_shape), ax=ax, **heatmap_kw)
                opt_row_2d, opt_col_2d = _grid_argmax(pred, ch2xy)
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


def r2_by_subject(results_dict, split_type='', save=False, output_dir=None):
    """Box plot of R² values grouped by subject index, one bar per model.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
    """
    output_subdir = 'optimization'
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


def _exploration_curves_for(
    results_dict: dict,
    group_key: str,
) -> dict:
    """Extract exploitation score curves keyed by group (subject or emg).

    Returns:
        dict mapping group_value -> dict[model_name -> list[curve_dict]].
        Each curve_dict has keys:
          'dense' (bool), 'x' (np.ndarray), 'y' (np.ndarray), 'se' (np.ndarray|None).
    """
    from collections import defaultdict
    by_group: dict = defaultdict(lambda: defaultdict(list))
    for model_name, results_list in results_dict.items():
        for res in results_list:
            optimal = float(res['y_test'].max())
            if optimal < 1e-8:
                continue
            n_init = res.get('n_init', 0)
            perf = res.get('perf_explore')
            if perf is not None:
                # Dense path: perf shape [n_reps, n_steps]
                perf_arr = np.asarray(perf)                              # [n_reps, n_steps]
                curve = {
                    'dense': True,
                    'x': np.arange(n_init, n_init + perf_arr.shape[1]),
                    'y': np.mean(perf_arr, axis=0),
                    'se': np.std(perf_arr, axis=0) / np.sqrt(max(perf_arr.shape[0], 1)),
                }
            else:
                snaps = res.get('snapshots')
                if not snaps:
                    continue
                iters = sorted(snaps.keys())
                curve = {
                    'dense': False,
                    'x': np.array(iters),
                    'y': np.array([snaps[it]['best_pred_val'] / optimal for it in iters]),
                    'se': None,
                }
            by_group[res[group_key]][model_name].append(curve)
    return by_group


def _plot_exploration_panel(
    ax,
    model_curves: dict,
    col_idx: int,
    title: str,
) -> None:
    """Render one exploitation-score subplot."""
    for model_name, curves in model_curves.items():
        color = PALETTE.get(model_name, 'gray')
        dense = [c for c in curves if c['dense']]
        sparse = [c for c in curves if not c['dense']]
        if dense:
            max_len = max(len(c['y']) for c in dense)
            ys = np.array([np.pad(c['y'], (0, max_len - len(c['y'])), constant_values=np.nan)
                           for c in dense])                              # [n_curves, n_steps]
            x = dense[0]['x'][:max_len]
            mean_c = np.nanmean(ys, axis=0)
            se_c = np.nanstd(ys, axis=0) / np.sqrt(
                np.sum(~np.isnan(ys), axis=0).clip(1)
            )
            ax.plot(x, mean_c, color=color, linewidth=2, label=model_name)
            ax.fill_between(x, mean_c - 1.96 * se_c, mean_c + 1.96 * se_c,
                            color=color, alpha=0.2)
        elif sparse:
            # Aggregate sparse curves at shared snapshot iterations
            ref_x = sparse[0]['x']
            stacked = np.array([c['y'] for c in sparse
                                 if len(c['y']) == len(ref_x)])
            if stacked.ndim == 2 and stacked.shape[0] > 1:
                mean_c = np.nanmean(stacked, axis=0)
                ax.plot(ref_x, mean_c, color=color, linewidth=2.5,
                        marker='o', markersize=5, label=model_name)
                for c in sparse:
                    if len(c['y']) == len(ref_x):
                        ax.plot(c['x'], c['y'], color=color, linewidth=1,
                                marker='o', markersize=3, alpha=0.35,
                                label='_nolegend_')
            else:
                for c in sparse:
                    ax.plot(c['x'], c['y'], color=color, linewidth=2,
                            marker='o', markersize=4, label=model_name)

    ax.set_title(title)
    ax.set_xlabel('BO Iteration')
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
    ax.grid(True, alpha=0.3)
    if col_idx == 0:
        ax.set_ylabel('Exploitation Score\n(best recommendation / optimal)')
    ax.legend(title='Model', fontsize=7)


def exploration_by_subject(results_dict, split_type='', save=False, output_dir=None):
    """Exploitation score trajectories faceted by subject.

    Plots how well the surrogate's pure-exploitation recommendation tracks the
    true optimum over BO iterations, aggregated over EMGs per subject.

    Dense path (new runs): uses ``perf_explore`` stored at every step.
    Sparse fallback (legacy pkl files): uses log-spaced snapshot ``best_pred_val``.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
        output_dir: run-level directory (saves under optimization/)
    """
    results_dict = _normalize_results_dict(results_dict)
    by_subject = _exploration_curves_for(results_dict, 'subject')

    subjects = sorted(by_subject.keys())
    n_subj = len(subjects)
    if n_subj == 0:
        return

    fig, axes = plt.subplots(1, n_subj, figsize=(5 * n_subj, 5), squeeze=False)
    for col, subj in enumerate(subjects):
        _plot_exploration_panel(axes[0, col], by_subject[subj], col, f'Subject {subj}')

    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', '')
    fig.suptitle(f'Exploitation Score Trajectory by Subject ({dataset})')
    fig.tight_layout()

    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'exploration_by_subject{suffix}.svg')
    if save:
        plt.savefig(plot_path, format='svg')
        print(f'Saved plot to {plot_path}')
    plt.close()


def exploration_by_emg(results_dict, split_type='', save=False, output_dir=None):
    """Exploitation score trajectories faceted by EMG channel.

    Same as ``exploration_by_subject`` but grouped by EMG index.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
        output_dir: run-level directory (saves under optimization/)
    """
    results_dict = _normalize_results_dict(results_dict)
    by_emg = _exploration_curves_for(results_dict, 'emg')

    emgs = sorted(by_emg.keys())
    n_emgs = len(emgs)
    if n_emgs == 0:
        return

    n_cols = min(n_emgs, 8)
    n_rows = int(np.ceil(n_emgs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    ax_flat = axes.ravel()

    for i, emg in enumerate(emgs):
        _plot_exploration_panel(ax_flat[i], by_emg[emg], i % n_cols, f'EMG {emg}')

    for j in range(n_emgs, len(ax_flat)):
        ax_flat[j].set_visible(False)

    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', '')
    fig.suptitle(f'Exploitation Score Trajectory by EMG Channel ({dataset})')
    fig.tight_layout()

    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'exploration_by_emg{suffix}.svg')
    if save:
        plt.savefig(plot_path, format='svg')
        print(f'Saved plot to {plot_path}')
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


def budget_sweep_plot(df, dataset='', split_type='', save=False, output_dir=None,
                      eval_type='optimization'):
    """
    Budget sweep with per-subject light traces and bold cross-subject mean.

    Light semi-transparent lines show each subject's mean score at each budget;
    the bold line shows the mean across subjects +/- SE. One trace per model.

    Args:
        df: DataFrame with columns Budget, Model, ID, R2, Regret.
            ID format: '{subject}_{emg}'.
        dataset, split_type, save, output_dir: plotting options.
        eval_type: kept for callsite back-compat; only ``'optimization'`` is supported.
    """
    if eval_type != 'optimization':
        raise ValueError(
            f"budget_sweep_plot supports only eval_type='optimization', got {eval_type!r}."
        )
    df = df.copy()
    df['Subject'] = df['ID'].str.split('_').str[0]

    metrics = []
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

        for model_name, results_list in results_dict.items():
            res = results_list[idx]
            color = PALETTE.get(model_name, 'gray')
            # Use each model's own raw-scale optimal so cross-model comparison is valid.
            optimal_val = float(res['y_test'].max())

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

            # --- exploitation score row ---
            # Dense path: perf_explore[t] = cummax(y_test_raw[argmax(mean_pred)] / optimal)
            # Fallback: sparse best_pred_val from log-spaced snapshots (legacy pkl files).
            perf_explore = res.get('perf_explore')
            n_init_res = res.get('n_init', 0)
            if perf_explore is not None and optimal_val > 1e-8:
                explore_arr = np.asarray(perf_explore)          # [n_steps]
                x_explore = np.arange(n_init_res, n_init_res + len(explore_arr))
                ax_ee.plot(x_explore, explore_arr, color=color, linewidth=2,
                           label=model_name)
            else:
                snaps = res.get('snapshots')
                if snaps and optimal_val > 1e-8:
                    iters = sorted(snaps.keys())
                    scores = [
                        snaps[it]['best_pred_val'] / optimal_val
                        for it in iters
                    ]
                    ax_ee.plot(iters, scores, color=color, linewidth=2,
                               marker='o', markersize=4, label=model_name)

        ref_res = first_results[idx]
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
    axes[2, 0].set_ylabel('Exploitation Score\n(running best recommendation / optimal)')

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


def augmentation_sweep_plot(df, dataset='', split_type='', save=False, output_dir=None):
    """
    Point plot of R² and Final Regret vs number of augmentations (optimization mode).

    n_aug=0 represents vanilla TabPFN (no finetuning); n_aug>0 represents
    finetuned TabPFN with that many augmentations per subject-EMG pair.

    Args:
        df: DataFrame with columns n_aug, R2, Regret, ID
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


def r2_barplot_by_subject(
    results_dict: dict,
    split_type: str = '',
    save: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """Grouped bar chart of mean R² per subject plus an aggregate Average group.

    One bar per model per subject; an extra 'Avg' group shows the cross-subject
    mean.  Error bars are the standard deviation pooled across all repetitions
    and EMG channels for each (subject, model) pair.  Raw R² values are used
    (no clipping) so negative values remain visible.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts.
            Each result dict must contain 'r2' (list[float]), 'subject' (int),
            and 'dataset' (str).
        split_type: String suffix appended to the output filename.
        save: If True, write SVG to disk under '<output_dir>/optimization/'.
        output_dir: Run-level directory root.
    """
    results_dict = _normalize_results_dict(results_dict)

    records = []
    for model_name, results_list in results_dict.items():
        for res in results_list:
            for score in res['r2']:
                records.append({
                    'Subject': f"S{res['subject']}",
                    'R2': float(score),
                    'Model': model_name,
                })

    df = pd.DataFrame(records)
    subjects = sorted(df['Subject'].unique())
    models = list(results_dict.keys())
    x_labels = list(subjects) + ['Avg']

    n_models = len(models)
    bar_width = 0.7 / n_models
    x_base = np.arange(len(x_labels))  # [n_subjects + 1]

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(x_labels)), 5))

    all_means, all_stds = [], []
    for i, model_name in enumerate(models):
        color = PALETTE.get(model_name, f'C{i}')
        mdf = df[df['Model'] == model_name]

        means, stds = [], []
        for lbl in subjects:
            vals = mdf.loc[mdf['Subject'] == lbl, 'R2'].values  # [n_reps * n_emg]
            means.append(float(np.mean(vals)) if len(vals) else 0.0)
            stds.append(float(np.std(vals)) if len(vals) else 0.0)
        # Avg bar: pool across all subjects
        all_vals = mdf['R2'].values  # [n_reps * n_emg * n_subj]
        means.append(float(np.mean(all_vals)) if len(all_vals) else 0.0)
        stds.append(float(np.std(all_vals)) if len(all_vals) else 0.0)

        all_means.extend(means)
        all_stds.extend(stds)

        offset = (i - (n_models - 1) / 2.0) * bar_width
        ax.bar(
            x_base + offset,
            means,
            width=bar_width,
            yerr=stds,
            capsize=4,
            color=color,
            label=model_name,
            alpha=0.85,
            error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
        )

    # Dynamic y-axis bounds that accommodate negative values
    y_lo = min(0.0, min(m - s for m, s in zip(all_means, all_stds))) - 0.05
    y_hi = max(1.0, max(m + s for m, s in zip(all_means, all_stds))) + 0.05
    ax.set_ylim(y_lo, y_hi)
    ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)

    ax.set_xticks(x_base)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Subject')
    ax.set_ylabel('R² Score')
    ax.set_title('R² by Subject')
    ax.legend(title='Model')
    ax.grid(True, alpha=0.3, axis='y')

    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', '')
    fig.tight_layout()

    base = (
        os.path.join(output_dir, 'optimization')
        if output_dir
        else os.path.join('output', 'optimization', dataset)
    )
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'r2_barplot_by_subject{suffix}.svg')
    if save:
        plt.savefig(plot_path, format='svg', bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    plt.close()


def regret_traces_by_subject(
    results_dict: dict,
    split_type: str = '',
    save: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """Regret curve traces per subject plus an Average panel.

    For each subject panel:
      - Faint thin lines: per-EMG mean regret curve (averaged across n_reps).
      - Bold line: cross-EMG mean regret trajectory.
      - Shaded band: +-1 std across EMGs.

    A final 'Average' panel pools all subjects x EMGs with the same layout.

    Args:
        results_dict: dict[str, list[dict]] -- model name -> list of result dicts.
            Each dict must have 'values' (list[list[float]], shape [n_reps, budget]),
            'y_test' (np.ndarray), 'subject' (int), and 'emg' (int).
        split_type: String suffix appended to the output filename.
        save: If True, write SVG to disk under '<output_dir>/optimization/'.
        output_dir: Run-level directory root.
    """
    results_dict = _normalize_results_dict(results_dict)

    # by_subject[subj][model] = list of per-EMG mean regret curves, each [budget]
    by_subject: dict = {}

    for model_name, results_list in results_dict.items():
        for res in results_list:
            if 'values' not in res or 'y_test' not in res:
                continue
            y_range = float(res['y_test'].max() - res['y_test'].min())
            if y_range < 1e-8:
                continue
            optimal = float(res['y_test'].max())
            rb = np.maximum.accumulate(np.array(res['values']), axis=1)  # [n_reps, budget]
            regret = (optimal - rb) / y_range                            # [n_reps, budget]
            curve = np.mean(regret, axis=0)                              # [budget]
            subj = res['subject']
            if subj not in by_subject:
                by_subject[subj] = {}
            if model_name not in by_subject[subj]:
                by_subject[subj][model_name] = []
            by_subject[subj][model_name].append(curve)

    subjects = sorted(by_subject.keys())
    n_subj = len(subjects)
    if n_subj == 0:
        return

    n_panels = n_subj + 1  # one per subject + Average
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)

    def _panel(ax, model_curves, title, col_idx):
        for model_name, curves in model_curves.items():
            color = PALETTE.get(model_name, 'gray')
            max_len = max(len(c) for c in curves)
            padded = np.array([                                          # [n_emg, budget]
                np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
                for c in curves
            ])
            mean_c = np.nanmean(padded, axis=0)                          # [budget]
            std_c = np.nanstd(padded, axis=0)                            # [budget]
            x = np.arange(len(mean_c))

            # Faint per-EMG traces
            for c in curves:
                ax.plot(np.arange(len(c)), c,
                        color=color, linewidth=0.8, alpha=0.3)
            # Bold mean + +-1 std band
            ax.plot(x, mean_c, color=color, linewidth=2.5, label=model_name)
            ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                            color=color, alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel('BO Iteration')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel('Simple Regret\n(normalized by response range, lower is better)')
        ax.legend(title='Model', fontsize=8)

    for col, subj in enumerate(subjects):
        _panel(axes[0, col], by_subject[subj], f'Subject {subj}', col)

    # Average panel: pool all subject x EMG curves
    all_curves: dict = {}
    for subj in subjects:
        for model_name, curves in by_subject[subj].items():
            if model_name not in all_curves:
                all_curves[model_name] = []
            all_curves[model_name].extend(curves)
    _panel(axes[0, n_subj], all_curves, 'Average', n_subj)

    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', '')
    fig.suptitle(f'Regret Curves by Subject ({dataset})')
    fig.tight_layout()

    base = (
        os.path.join(output_dir, 'optimization')
        if output_dir
        else os.path.join('output', 'optimization')
    )
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'regret_traces_by_subject{suffix}.svg')
    if save:
        plt.savefig(plot_path, format='svg', bbox_inches='tight')
        print(f"Saved plot to {plot_path}")
    plt.close()
