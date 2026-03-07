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

# Light colors for individual-trace overlays
_TRACE_COLORS = {
    'GP': 'red',
    'PFN': 'blue',
    'TabPFN': 'green'
}


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


def r2_comparison(results_dict_or_gp, pfn_results=None, mode='', save=False, output_dir=None, eval_type='fit'):

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


def plot_runtime_trajectory(results_dict_or_gp, pfn_results=None, split_type='', save=False, output_dir=None):
    """
    Plots the inference time at each BO step for all models.
    """
    results_dict = _normalize_results_dict(results_dict_or_gp, pfn_results)

    plt.figure(figsize=(8, 6))

    for model_name, results_list in results_dict.items():
        trace_color = _TRACE_COLORS.get(model_name, 'gray')
        line_color = PALETTE.get(model_name, 'black')

        all_times = []
        for res in results_list:
            plt.plot(res['times'], color=trace_color, alpha=0.1)
            all_times.append(res['times'])

        avg_times = np.mean(all_times, axis=0)
        plt.plot(avg_times, color=line_color, linewidth=2, label=model_name)

    plt.title("Inference Time for BO")
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', '')
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    plot_path = os.path.join(base, f'runtime_trajectory{suffix}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def show_emg_map(results, idx, model_type, mode='', save=False, output_dir=None, eval_type='fit'):
    res = results[idx]

    y_true = res['y_test']
    y_pred = res['y_pred']
    r2_score = np.mean(np.array(res['r2']))

    # Determine Grid Shape
    n_channels = len(y_true)
    if n_channels == 96: grid_shape = (8, 12)
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
        base = os.path.join(output_dir, 'optimization') if output_dir else \
               os.path.join('output', 'optimization')
    else:
        base = os.path.join(output_dir, 'fitness') if output_dir else \
               os.path.join('output', 'fitness', dataset)
    os.makedirs(base, exist_ok=True)
    plot_path = os.path.join(base, f'emg_map_{dataset}_s{subject}_emg{emg}_{model_type}{mode}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def regret_curve(results_dict, split_type='', save=False, output_dir=None):
    """
    Plot regret curves for all models across all experiments in a 1×N grid.

    Each subplot corresponds to one experiment index (subject/EMG pair).
    All models are overlaid on each subplot. The y-axis scale is shared
    across subplots for direct comparison.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
        save: whether to save the figure to disk
    """
    first_results = next(iter(results_dict.values()))
    n_experiments = len(first_results)

    def get_regret_stats(values_list, optimal_val):
        raw_vals = np.array(values_list)
        best_so_far = np.maximum.accumulate(raw_vals, axis=1)
        regret_all = optimal_val - best_so_far
        mean_regret = np.mean(regret_all, axis=0)
        std_regret = np.std(regret_all, axis=0)
        se_regret = std_regret / np.sqrt(raw_vals.shape[0])
        return mean_regret, se_regret

    # First pass: compute global y-limits
    y_min_global, y_max_global = np.inf, -np.inf
    for idx in range(n_experiments):
        optimal_val = first_results[idx]['y_test'].max()
        for results_list in results_dict.values():
            mean_reg, se_reg = get_regret_stats(results_list[idx]['values'], optimal_val)
            lo = (mean_reg - 1.96 * se_reg).min()
            hi = (mean_reg + 1.96 * se_reg).max()
            y_min_global = min(y_min_global, lo)
            y_max_global = max(y_max_global, hi)

    y_pad = 0.05 * (y_max_global - y_min_global)
    y_min_global = max(0, y_min_global - y_pad)
    y_max_global = y_max_global + y_pad

    # Second pass: plot
    fig, axes = plt.subplots(1, n_experiments,
                             figsize=(4 * n_experiments, 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for idx, ax in enumerate(axes):
        optimal_val = first_results[idx]['y_test'].max()

        for model_name, results_list in results_dict.items():
            res = results_list[idx]
            color = PALETTE[model_name]
            mean_reg, se_reg = get_regret_stats(res['values'], optimal_val)
            x_axis = range(len(mean_reg))

            ax.plot(x_axis, mean_reg, color=color, label=model_name, linewidth=2)
            ax.fill_between(x_axis,
                            mean_reg - 1.96 * se_reg,
                            mean_reg + 1.96 * se_reg,
                            color=color, alpha=0.2)

        ref_res = first_results[idx]
        ax.set_title(f"S{ref_res['subject']} EMG {ref_res['emg']}", fontsize=9)
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, y_max_global)

    axes[0].set_ylabel('Simple Regret')
    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=9)

    dataset = first_results[0].get('dataset', '')
    fig.suptitle(f'Regret Curves | {dataset}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    base = os.path.join(output_dir, 'optimization') if output_dir else \
           os.path.join('output', 'optimization')
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'regret_curves{suffix}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def r2_by_subject(results_dict, split_type='', save=False, output_dir=None):
    """
    Box plot of R² values grouped by subject index, one bar per model.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
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
    base = os.path.join(output_dir, 'fitness') if output_dir else \
           os.path.join('output', 'fitness', dataset)
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'r2_by_subject{suffix}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def r2_by_emg(results_dict, split_type='', save=False, output_dir=None):
    """
    Box plot of R² values grouped by EMG index, one bar per model.

    Args:
        results_dict: dict[str, list[dict]] — model name -> list of result dicts
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
    """
    results_dict = _normalize_results_dict(results_dict)

    data = []
    for model_name, results_list in results_dict.items():
        for res in results_list:
            for score in res['r2']:
                data.append({
                    'EMG': f"EMG {res['emg']}",
                    'R2': score,
                    'Model': model_name
                })

    df = pd.DataFrame(data)
    n_emgs = df['EMG'].nunique()
    plt.figure(figsize=(max(6, 1.8 * n_emgs), 5))
    sns.boxplot(data=df, x='EMG', y='R2', hue='Model', palette=PALETTE)
    plt.ylim(0, 1)
    plt.title("R² by EMG Channel")
    plt.xlabel("EMG")
    plt.ylabel("R²")
    plt.legend(title='Model')
    plt.grid(True, alpha=0.3, axis='y')

    first_results = next(iter(results_dict.values()))
    dataset = first_results[0].get('dataset', '')
    base = os.path.join(output_dir, 'fitness') if output_dir else \
           os.path.join('output', 'fitness', dataset)
    os.makedirs(base, exist_ok=True)
    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    plot_path = os.path.join(base, f'r2_by_emg{suffix}.svg')
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def regret_by_subject(results_dict, split_type='', save=False, output_dir=None):
    """
    Box plot of final simple regret grouped by subject index.

    Final simple regret = optimal_value - best_observed_at_last_step.

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
            optimal = float(res['y_test'].max())
            raw_vals = np.array(res['values'])
            best_so_far = np.maximum.accumulate(raw_vals, axis=1)
            final_regrets = optimal - best_so_far[:, -1]
            for regret in final_regrets:
                data.append({
                    'Subject': f"S{res['subject']}",
                    'Regret': float(regret),
                    'Model': model_name
                })

    if not data:
        return

    df = pd.DataFrame(data)
    n_subjects = df['Subject'].nunique()
    plt.figure(figsize=(max(6, 1.8 * n_subjects), 5))
    sns.boxplot(data=df, x='Subject', y='Regret', hue='Model', palette=PALETTE)
    plt.title("Final Simple Regret by Subject")
    plt.xlabel("Subject")
    plt.ylabel("Final Simple Regret")
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
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def regret_by_emg(results_dict, split_type='', save=False, output_dir=None):
    """
    Box plot of final simple regret grouped by EMG index.

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
            optimal = float(res['y_test'].max())
            raw_vals = np.array(res['values'])
            best_so_far = np.maximum.accumulate(raw_vals, axis=1)
            final_regrets = optimal - best_so_far[:, -1]
            for regret in final_regrets:
                data.append({
                    'EMG': f"EMG {res['emg']}",
                    'Regret': float(regret),
                    'Model': model_name
                })

    if not data:
        return

    df = pd.DataFrame(data)
    n_emgs = df['EMG'].nunique()
    plt.figure(figsize=(max(6, 1.8 * n_emgs), 5))
    sns.boxplot(data=df, x='EMG', y='Regret', hue='Model', palette=PALETTE)
    plt.title("Final Simple Regret by EMG Channel")
    plt.xlabel("EMG")
    plt.ylabel("Final Simple Regret")
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


def budget_sweep_plot(df, eval_type, dataset='', split_type='', save=False, output_dir=None):
    """
    Line plot of R² (eval_type='fit') or Regret (eval_type='optimization')
    vs Budget, with both GP and TabPFN shown with 95% CI bands.

    Args:
        df: DataFrame with columns Budget, Model, R2 or Regret, ID
        eval_type: 'fit' (uses R2 column) or 'optimization' (uses Regret column)
        dataset: dataset name used for output path and title
        split_type: string suffix for the output filename
        save: whether to save the figure to disk
    """
    y_col = 'R2' if eval_type == 'fit' else 'Regret'
    y_label = 'R² Score (Test Set)' if eval_type == 'fit' else 'Final Simple Regret'
    title = f'Fit Quality vs. Training Budget ({dataset})' if eval_type == 'fit' \
        else f'Optimization: Final Regret vs Budget ({dataset})'

    models_in_df = df['Model'].unique().tolist()
    palette = {m: PALETTE.get(m, 'gray') for m in models_in_df}

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x='Budget',
        y=y_col,
        hue='Model',
        palette=palette,
        marker='o',
        errorbar=('ci', 95),
        err_kws={'alpha': 0.2},
        linewidth=2
    )
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('Budget (Number of Training Points)' if eval_type == 'fit'
               else 'Budget (Number of Queries)')
    if eval_type == 'fit':
        plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Model')

    suffix = f'_{dataset}_{split_type}' if split_type else f'_{dataset}'
    if eval_type == 'fit':
        base = os.path.join(output_dir, 'fitness') if output_dir else \
               os.path.join('output', 'fitness', dataset)
        os.makedirs(base, exist_ok=True)
        plot_path = os.path.join(base, f'budget_sweep_fit{suffix}.svg')
    else:
        base = os.path.join(output_dir, 'optimization') if output_dir else \
               os.path.join('output', 'optimization')
        os.makedirs(base, exist_ok=True)
        plot_path = os.path.join(base, f'budget_sweep_optimization{suffix}.svg')

    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")

    plt.close()


def regret_with_timing(results_dict, split_type='', save=False, output_dir=None):
    """
    2-row figure: top = regret curves (95% CI bands), bottom = per-step inference time.
    One column per experiment (subject/EMG pair).

    Replaces separate regret_curve + plot_runtime_trajectory calls when both
    metrics are needed on one figure.

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
        raw_vals = np.array(values_list)
        best_so_far = np.maximum.accumulate(raw_vals, axis=1)
        regret_all = optimal_val - best_so_far
        mean_regret = np.mean(regret_all, axis=0)
        se_regret = np.std(regret_all, axis=0) / np.sqrt(raw_vals.shape[0])
        return mean_regret, se_regret

    fig, axes = plt.subplots(2, n_experiments,
                             figsize=(4 * n_experiments, 8),
                             squeeze=False)

    for idx in range(n_experiments):
        ax_reg = axes[0, idx]
        ax_time = axes[1, idx]

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

        ax_reg.set_title(f"S{ref_res['subject']} EMG {ref_res['emg']}", fontsize=9)
        ax_reg.set_xlabel('Iteration')
        ax_reg.grid(True, alpha=0.3)
        ax_reg.set_ylim(bottom=0)

        ax_time.set_xlabel('Iteration')
        ax_time.set_ylabel('Time (s)')
        ax_time.grid(True, alpha=0.3)

    axes[0, 0].set_ylabel('Simple Regret')
    axes[1, 0].set_ylabel('Inference Time (s)')

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

    # Build ordered x-axis labels: 0 → 'Vanilla', rest as strings
    aug_values = sorted(df['n_aug'].unique())
    x_labels = ['Vanilla' if v == 0 else str(v) for v in aug_values]

    # Map numeric n_aug to display label for plotting
    df = df.copy()
    label_map = {v: ('Vanilla' if v == 0 else str(v)) for v in aug_values}
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
