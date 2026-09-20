"""
Dedicated plots for ID/OOD analysis results.

Follows conventions from src/utils/visualization.py:
matplotlib/seaborn, SVG output, save flag, output_dir.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = {
    'Neurostim': 'royalblue',
    'Neurostim GT': 'steelblue',
    'Synthetic GP': 'sandybrown',
    'TabPFN Prior': 'seagreen',
    'Noise (OOD)': 'firebrick',
    'Clustered OOD': 'darkorange',
    'Correlated OOD': 'mediumorchid',
}

# Shared reference-comparison legend for dashboard panels. Passing a dict keyed
# by the hue label (not a positional list) keeps each reference on its semantic
# color regardless of which references are present in a given run — e.g. with
# prior_source=prior_bag the 'vs GP' level is absent, but 'vs Prior'/'vs Noise'
# still map to seagreen/firebrick instead of shifting to the missing GP slot.
REF_HUE_ORDER = ['vs GP', 'vs Prior', 'vs Noise']
REF_PALETTE = {
    'vs GP': PALETTE['Synthetic GP'],
    'vs Prior': PALETTE['TabPFN Prior'],
    'vs Noise': PALETTE['Noise (OOD)'],
}


def _save_dir(output_dir, subdir):
    """Return output directory, creating if needed."""
    base = os.path.join(output_dir, subdir) if output_dir else \
           os.path.join('output', 'id_ood', subdir)
    os.makedirs(base, exist_ok=True)
    return base


# ============================================================================
#  Entropy Plots
# ============================================================================

def plot_entropy_distribution(entropy_results, save=False, output_dir=None):
    """Overlaid KDE plot: entropy distributions for neurostim vs references.

    One KDE per group: each neurostim dataset (pooled across subjects),
    GP ref, Prior ref, Noise.  Replaces the prior violin + axhspan approach
    for clearer comparison of distribution shapes and overlap.

    Args:
        entropy_results: dict from entropy_analysis()
        save: whether to save figure
        output_dir: base output directory
    """
    _ref_keys = {'synthetic_gp', 'synthetic_prior', 'noise'}
    # Separate in-context keys (e.g. 'nhp') from ground-truth keys (e.g. 'nhp_gt')
    _gt_suffix = '_gt'
    gt_keys = {k for k in entropy_results if k.endswith(_gt_suffix)}
    dataset_types = [k for k in entropy_results
                     if k not in _ref_keys and k not in gt_keys]

    fig, ax = plt.subplots(figsize=(8, 5))

    def _pool_nested_entropies(nested_data):
        """Flatten {subj: {emg: array}} into a single list of values."""
        vals = []
        for subj_data in nested_data.values():
            for entropy_arr in subj_data.values():
                vals.extend(entropy_arr.tolist()
                            if hasattr(entropy_arr, 'tolist')
                            else list(entropy_arr))
        return vals

    # Neurostim KDEs — in-context (one per dataset, pooled across subjects/EMGs)
    for dataset_type in dataset_types:
        all_vals = _pool_nested_entropies(entropy_results[dataset_type])
        if all_vals:
            sns.kdeplot(all_vals, ax=ax, fill=True, alpha=0.3,
                        color=PALETTE['Neurostim'],
                        label=f'{dataset_type.upper()} (in-context)',
                        linewidth=2)

    # Neurostim KDEs — ground truth (full map as context)
    for gt_key in sorted(gt_keys):
        ds_name = gt_key[:-len(_gt_suffix)]
        all_vals = _pool_nested_entropies(entropy_results[gt_key])
        if all_vals:
            sns.kdeplot(all_vals, ax=ax, fill=True, alpha=0.15,
                        color=PALETTE['Neurostim GT'],
                        label=f'{ds_name.upper()} (ground truth)',
                        linestyle='--', linewidth=2)

    # Synthetic reference KDEs
    ref_plot_config = {
        'synthetic_gp': ('GP Reference', PALETTE['Synthetic GP'], '--'),
        'synthetic_prior': ('TabPFN Prior', PALETTE['TabPFN Prior'], '-.'),
        'noise': ('Noise (OOD)', PALETTE['Noise (OOD)'], ':'),
    }
    for ref_key, (label, color, ls) in ref_plot_config.items():
        if ref_key in entropy_results and len(entropy_results[ref_key]) > 0:
            ref_vals = entropy_results[ref_key]
            sns.kdeplot(ref_vals.tolist() if hasattr(ref_vals, 'tolist')
                        else list(ref_vals),
                        ax=ax, fill=True, alpha=0.15,
                        color=color, label=label,
                        linestyle=ls, linewidth=2)

    ax.set_xlabel('Shannon Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Bar-Distribution Entropy: Neurostim vs Synthetic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'entropy')
        path = os.path.join(base, 'entropy_distribution.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def plot_entropy_heatmap(entropy_results, dataset_type, save=False, output_dir=None):
    """Heatmap: mean entropy per (subject x EMG) for one dataset type.

    Args:
        entropy_results: dict from entropy_analysis()
        dataset_type: which dataset to plot
        save: whether to save figure
        output_dir: base output directory
    """
    ds_data = entropy_results.get(dataset_type, {})
    if not ds_data:
        return

    # Build matrix
    subjects = sorted(ds_data.keys())
    all_emgs = set()
    for subj_data in ds_data.values():
        all_emgs.update(subj_data.keys())
    emgs = sorted(all_emgs)

    matrix = np.full((len(subjects), len(emgs)), np.nan)
    for i, subj in enumerate(subjects):
        for j, emg in enumerate(emgs):
            if emg in ds_data[subj]:
                matrix[i, j] = np.mean(ds_data[subj][emg])

    fig, ax = plt.subplots(figsize=(max(6, len(emgs) * 0.8),
                                     max(3, len(subjects) * 0.6)))
    sns.heatmap(matrix, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f',
                xticklabels=[f'EMG {e}' for e in emgs],
                yticklabels=[f'S{s}' for s in subjects])
    ax.set_title(f'Mean Entropy | {dataset_type.upper()}')
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('Subject')

    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'entropy')
        path = os.path.join(base, f'entropy_heatmap_{dataset_type}.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


# ============================================================================
#  MMD / Wasserstein Heatmap
# ============================================================================

def plot_mmd_heatmap(mmd_results, wasserstein_results=None,
                     save=False, output_dir=None):
    """Pairwise heatmap: rows = (dataset, subject), cols = reference types.

    Shows MMD^2 values as color-coded heatmap with significance annotations.
    If wasserstein_results provided, creates side-by-side MMD + W2 panels.

    Args:
        mmd_results: dict from mmd_analysis()
        wasserstein_results: optional dict from wasserstein_analysis()
        save: whether to save figure
        output_dir: base output directory
    """
    has_mmd = mmd_results is not None
    has_w2 = wasserstein_results is not None
    n_panels = int(has_mmd) + int(has_w2)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5),
                              squeeze=False)

    panel_idx = 0
    if has_mmd:
        _build_distance_heatmap(
            axes[0, panel_idx], mmd_results, metric_prefix='mmd2',
            p_prefix='p', title='MMD²', cmap='YlOrRd',
        )
        panel_idx += 1

    if has_w2:
        _build_distance_heatmap(
            axes[0, panel_idx], wasserstein_results, metric_prefix='w2',
            p_prefix=None, title='Sliced Wasserstein-2', cmap='YlOrRd',
        )

    fig.suptitle('Distributional Distance: Neurostim vs References',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        base = _save_dir(output_dir, 'mmd')
        path = os.path.join(base, 'mmd_heatmap.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def _build_distance_heatmap(ax, results, metric_prefix, p_prefix,
                            title, cmap):
    """Build a single distance heatmap on the given axes.

    Rows = (dataset, subject), columns = reference types.
    """
    ref_names = ['gp', 'prior', 'noise']
    ref_labels = {'gp': 'Synthetic GP', 'prior': 'TabPFN Prior',
                  'noise': 'Noise (OOD)'}

    # Collect rows
    row_labels = []
    matrix_vals = []
    annot_strs = []

    for dataset_type, ds_data in results.items():
        for subj_idx in sorted(ds_data.keys()):
            subj_data = ds_data[subj_idx]
            row_labels.append(f'{dataset_type} S{subj_idx}')
            row_vals = []
            row_annots = []

            for ref in ref_names:
                metric_key = f'{metric_prefix}_{ref}'
                # Average across EMGs
                vals = [emg_data[metric_key]
                        for emg_data in subj_data.values()
                        if metric_key in emg_data]
                mean_val = np.mean(vals) if vals else np.nan
                row_vals.append(mean_val)

                # Significance annotation (if p-values available)
                if p_prefix:
                    p_key = f'{p_prefix}_{ref}'
                    p_vals = [emg_data[p_key]
                              for emg_data in subj_data.values()
                              if p_key in emg_data]
                    mean_p = np.mean(p_vals) if p_vals else 1.0
                    if mean_p < 0.01:
                        star = '**'
                    elif mean_p < 0.05:
                        star = '*'
                    else:
                        star = 'ns'
                    row_annots.append(f'{mean_val:.3f}\n{star}')
                else:
                    row_annots.append(f'{mean_val:.3f}')

            matrix_vals.append(row_vals)
            annot_strs.append(row_annots)

    if not matrix_vals:
        return

    matrix = np.array(matrix_vals)
    annot = np.array(annot_strs)
    col_labels = [ref_labels.get(r, r) for r in ref_names]

    sns.heatmap(matrix, ax=ax, cmap=cmap, annot=annot, fmt='',
                xticklabels=col_labels, yticklabels=row_labels,
                linewidths=0.5)
    ax.set_title(title)


# Backward-compatible alias
def plot_mmd_barplot(mmd_results, save=False, output_dir=None):
    """Deprecated: redirects to plot_mmd_heatmap."""
    plot_mmd_heatmap(mmd_results, save=save, output_dir=output_dir)


# ============================================================================
#  CKA Heatmap
# ============================================================================

def _get_cka_layers(cka_results):
    """Extract analyzed layer indices from CKA results metadata."""
    return cka_results.get('layers', [17])


def _get_cka_dataset_types(cka_results):
    """Extract dataset type keys (excluding metadata keys like 'layers')."""
    return [k for k in cka_results if k != 'layers']


def plot_cka_heatmap(cka_results, save=False, output_dir=None, layer=None):
    """Heatmap: CKA similarity scores (subjects x reference types).

    One panel per dataset type, at a single layer.  Green = high CKA
    (in-distribution), red = low CKA (out-of-distribution).

    Args:
        cka_results: dict from cka_analysis() with multi-layer structure.
        save: whether to save figure.
        output_dir: base output directory.
        layer: transformer layer index to display.
            Defaults to the last (deepest) analyzed layer.
    """
    layers = _get_cka_layers(cka_results)
    if layer is None:
        layer = max(layers)

    dataset_types = _get_cka_dataset_types(cka_results)
    if not dataset_types:
        return

    n_panels = len(dataset_types)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4),
                              squeeze=False)

    ref_names = ['gp', 'prior', 'noise']
    ref_labels = {'gp': 'Synthetic GP', 'prior': 'TabPFN Prior',
                  'noise': 'Noise (OOD)'}

    for ax_i, dataset_type in enumerate(dataset_types):
        ax = axes[0, ax_i]
        ds_data = cka_results[dataset_type]
        subjects = sorted(ds_data.keys())

        matrix = np.full((len(subjects), len(ref_names)), np.nan)
        for i, subj in enumerate(subjects):
            subj_data = ds_data[subj]
            for j, ref in enumerate(ref_names):
                key = f'cka_{ref}'
                vals = [emg_data[layer][key]
                        for emg_data in subj_data.values()
                        if layer in emg_data and key in emg_data[layer]]
                if vals:
                    matrix[i, j] = np.mean(vals)

        sns.heatmap(matrix, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                    annot=True, fmt='.2f',
                    xticklabels=[ref_labels.get(r, r) for r in ref_names],
                    yticklabels=[f'S{s}' for s in subjects],
                    linewidths=0.5)
        ax.set_title(f'CKA Similarity | {dataset_type.upper()}')

    fig.suptitle(f'CKA: Representation Alignment (Layer {layer})', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        base = _save_dir(output_dir, 'cka')
        path = os.path.join(base, f'cka_heatmap_layer{layer}.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def plot_cka_layerwise_heatmap(cka_results, save=False, output_dir=None):
    """Layer-wise CKA heatmap: layers x references, averaged across subjects/EMGs.

    Reveals WHERE in the network neurostim representations diverge from
    reference distributions.  One panel per dataset type.

    Args:
        cka_results: dict from cka_analysis() with multi-layer structure.
        save: whether to save figure.
        output_dir: base output directory.
    """
    layers = _get_cka_layers(cka_results)
    dataset_types = _get_cka_dataset_types(cka_results)
    if not dataset_types or len(layers) < 2:
        return

    ref_names = ['gp', 'prior', 'noise']
    ref_labels = {'gp': 'Synthetic GP', 'prior': 'TabPFN Prior',
                  'noise': 'Noise (OOD)'}

    n_panels = len(dataset_types)
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(4 + 2 * len(ref_names), 1 + 0.5 * len(layers)),
                              squeeze=False)

    for ax_i, dataset_type in enumerate(dataset_types):
        ax = axes[0, ax_i]
        ds_data = cka_results[dataset_type]

        # matrix: layers x references, averaged across all subjects/EMGs
        matrix = np.full((len(layers), len(ref_names)), np.nan)
        for li, layer_idx in enumerate(layers):
            for rj, ref in enumerate(ref_names):
                key = f'cka_{ref}'
                vals = []
                for subj_data in ds_data.values():
                    for emg_data in subj_data.values():
                        if layer_idx in emg_data and key in emg_data[layer_idx]:
                            vals.append(emg_data[layer_idx][key])
                if vals:
                    matrix[li, rj] = np.mean(vals)

        sns.heatmap(matrix, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                    annot=True, fmt='.2f',
                    xticklabels=[ref_labels.get(r, r) for r in ref_names],
                    yticklabels=[f'Layer {l}' for l in layers],
                    linewidths=0.5)
        ax.set_title(f'{dataset_type.upper()}')
        ax.set_ylabel('Transformer Layer')

    fig.suptitle('Layer-wise CKA: Where Do Representations Diverge?',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        base = _save_dir(output_dir, 'cka')
        path = os.path.join(base, 'cka_layerwise_heatmap.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


# ============================================================================
#  Gradient Norm Bar Plot
# ============================================================================

def plot_gradient_norm_barplot(gradient_results, save=False, output_dir=None):
    """Grouped bar plot: gradient L2 norms with log y-axis.

    X-axis: (dataset, subject) groups.  Bars: neurostim gradient norm
    (mean across EMGs).  Horizontal reference lines for synthetic baselines.

    Args:
        gradient_results: dict from gradient_norm_analysis()
        save: whether to save figure
        output_dir: base output directory
    """
    _ref_keys = {'synthetic_gp', 'synthetic_prior', 'noise'}
    dataset_types = [k for k in gradient_results if k not in _ref_keys]

    # Collect neurostim bars
    labels = []
    values = []
    for dt in dataset_types:
        ds_data = gradient_results[dt]
        for subj_idx in sorted(ds_data.keys()):
            subj_data = ds_data[subj_idx]
            norms = [v for v in subj_data.values() if np.isfinite(v)]
            if norms:
                labels.append(f'{dt} S{subj_idx}')
                values.append(np.mean(norms))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))

    x = np.arange(len(labels))
    ax.bar(x, values, color=PALETTE['Neurostim'], alpha=0.8,
           label='Neurostim')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Reference baselines as horizontal lines
    ref_line_config = {
        'synthetic_gp': ('GP Ref', PALETTE['Synthetic GP'], '--'),
        'synthetic_prior': ('Prior Ref', PALETTE['TabPFN Prior'], '-.'),
        'noise': ('Noise OOD', PALETTE['Noise (OOD)'], ':'),
    }
    for ref_key, (label, color, ls) in ref_line_config.items():
        if ref_key in gradient_results:
            ref_vals = gradient_results[ref_key]
            if len(ref_vals) > 0:
                ref_mean = np.mean(ref_vals)
                ax.axhline(ref_mean, color=color, linestyle=ls,
                           linewidth=2, label=f'{label} (mean={ref_mean:.2f})')

    ax.set_yscale('log')
    ax.set_ylabel('Gradient L2 Norm (log scale)')
    ax.set_title('Step-0 Gradient Norms: Neurostim vs Synthetic')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'gradient_norm')
        path = os.path.join(base, 'gradient_norm_barplot.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


# ============================================================================
#  Mahalanobis Violin Plot
# ============================================================================

def plot_mahalanobis_distribution(mahalanobis_results, save=False, output_dir=None):
    """Violin plot: Mahalanobis distance distributions per reference.

    Each panel (vs GP / vs Prior / vs Noise) shows three groups:
      • Neurostim        — the data under investigation
      • Reference self   — how the reference distribution looks to itself (calibration)
      • OOD contrast     — a clearly-OOD distribution for comparison
                           (vs GP and vs Prior: noise OOD bank;
                            vs Noise: two structurally-different noise types)

    Args:
        mahalanobis_results: dict from mahalanobis_analysis()
        save: whether to save figure
        output_dir: base output directory
    """
    dataset_types = [k for k in mahalanobis_results if k != 'ref_stats']
    if not dataset_types:
        return

    # ---- Neurostim distances (one row per test-point per EMG) ----
    _ref_label = {'gp': 'vs GP', 'prior': 'vs Prior', 'noise': 'vs Noise'}
    plot_data = []

    for dataset_type in dataset_types:
        ds_data = mahalanobis_results[dataset_type]
        for subj_idx, subj_data in ds_data.items():
            for emg_idx, emg_data in subj_data.items():
                for ref_name in ['gp', 'prior', 'noise']:
                    dist_key = f'distances_{ref_name}'
                    if dist_key in emg_data:
                        for d in emg_data[dist_key]:
                            plot_data.append({
                                'Dataset': dataset_type.upper(),
                                'Distance': float(d),
                                'Source': 'Neurostim',
                                'Reference': _ref_label[ref_name],
                            })

    # ---- Reference self-distances + OOD contrasts ----
    _self_source = {
        'gp': 'Synthetic GP',
        'prior': 'TabPFN Prior',
        'noise': 'Noise (OOD)',
    }
    ref_stats = mahalanobis_results.get('ref_stats', {})

    for ref_name, stats in ref_stats.items():
        ref_col = _ref_label.get(ref_name, f'vs {ref_name}')
        self_label = _self_source.get(ref_name, ref_name)

        # Self-distances (same for every neurostim dataset → duplicate across all)
        for d in stats.get('self_distances', []):
            for dataset_type in dataset_types:
                plot_data.append({
                    'Dataset': dataset_type.upper(),
                    'Distance': float(d),
                    'Source': self_label,
                    'Reference': ref_col,
                })

        # OOD contrast for GP and Prior panels: single noise OOD bank
        if ref_name in ('gp', 'prior'):
            for d in stats.get('ood_distances', []):
                for dataset_type in dataset_types:
                    plot_data.append({
                        'Dataset': dataset_type.upper(),
                        'Distance': float(d),
                        'Source': 'Noise (OOD)',
                        'Reference': ref_col,
                    })


    if not plot_data:
        return

    df = pd.DataFrame(plot_data)

    refs = ['vs GP', 'vs Prior', 'vs Noise']
    refs = [r for r in refs if r in df['Reference'].unique()]
    n_panels = len(refs)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)

    source_palette = {k: v for k, v in PALETTE.items()}

    for ax_i, ref in enumerate(refs):
        ax = axes[0, ax_i]
        ref_df = df[df['Reference'] == ref]
        hue_order = [s for s in source_palette if s in ref_df['Source'].unique()]
        sns.violinplot(data=ref_df, x='Dataset', y='Distance', hue='Source',
                       hue_order=hue_order, palette=source_palette,
                       ax=ax, inner='box', alpha=0.7)
        ax.set_title(f'Mahalanobis Distance ({ref})')
        ax.set_ylabel('D_M')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Mahalanobis Distance in TabPFN Representation Space', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        base = _save_dir(output_dir, 'mahalanobis')
        path = os.path.join(base, 'mahalanobis_distribution.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


# ============================================================================
#  Summary Dashboard
# ============================================================================

def plot_summary_dashboard(entropy_results=None, mmd_results=None,
                           mahalanobis_results=None, cka_results=None,
                           wasserstein_results=None, gradient_results=None,
                           rsa_results=None, procrustes_results=None,
                           save=False, output_dir=None):
    """Multi-panel paper-ready summary.

    Dynamically includes panels for each available result set.
    Backward-compatible with old 3-positional-arg calls.

    Args:
        entropy_results: dict from entropy_analysis()
        mmd_results: dict from mmd_analysis()
        mahalanobis_results: dict from mahalanobis_analysis()
        cka_results: dict from cka_analysis()
        wasserstein_results: dict from wasserstein_analysis()
        gradient_results: dict from gradient_norm_analysis()
        rsa_results: dict from rsa_analysis()
        save: whether to save figure
        output_dir: base output directory
    """
    # Build list of panels to render
    panels = []
    if entropy_results is not None:
        panels.append(('A', 'Entropy', entropy_results))
    if mmd_results is not None:
        panels.append(('B', 'MMD²', mmd_results))
    if mahalanobis_results is not None:
        panels.append(('C', 'Mahalanobis', mahalanobis_results))
    if cka_results is not None:
        panels.append(('D', 'CKA', cka_results))
    if wasserstein_results is not None:
        panels.append(('E', 'Wasserstein-2', wasserstein_results))
    if gradient_results is not None:
        panels.append(('F', 'Gradient Norm', gradient_results))
    if rsa_results is not None:
        panels.append(('G', 'RSA', rsa_results))
    if procrustes_results is not None:
        panels.append(('H', 'Procrustes', procrustes_results))

    if not panels:
        return

    n = len(panels)
    if n <= 3:
        nrows, ncols = 1, n
    elif n <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 5 * nrows),
                              squeeze=False)

    for idx, (letter, title, data) in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if title == 'Entropy':
            _panel_entropy(ax, data)
        elif title == 'MMD²':
            _panel_mmd(ax, data)
        elif title == 'Mahalanobis':
            _panel_mahalanobis(ax, data)
        elif title == 'CKA':
            _panel_cka(ax, data)
        elif title == 'Wasserstein-2':
            _panel_wasserstein(ax, data)
        elif title == 'Gradient Norm':
            _panel_gradient_norm(ax, data)
        elif title == 'RSA':
            _panel_rsa(ax, data)
        elif title == 'Procrustes':
            _panel_procrustes(ax, data)

        ax.set_title(f'{letter}) {title}')

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("ID/OOD Analysis: Is Neurostim Data Within TabPFN's Prior?",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        base = output_dir if output_dir else os.path.join('output', 'id_ood')
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, 'summary_dashboard.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


# --- Dashboard panel helpers ------------------------------------------------

def _panel_entropy(ax, entropy_results):
    """Compact entropy boxplot for dashboard."""
    _ref_keys = {'synthetic_gp', 'synthetic_prior', 'noise'}
    dataset_types = [k for k in entropy_results if k not in _ref_keys]
    box_data = []
    for dt in dataset_types:
        for subj_data in entropy_results[dt].values():
            for entropy_arr in subj_data.values():
                for val in entropy_arr:
                    box_data.append({'Dataset': dt.upper(),
                                     'Entropy': float(val)})
    for synth_key, label in [('synthetic_gp', 'GP Ref'),
                              ('synthetic_prior', 'Prior Ref'),
                              ('noise', 'Noise OOD')]:
        if synth_key in entropy_results:
            for val in entropy_results[synth_key]:
                box_data.append({'Dataset': label, 'Entropy': float(val)})

    if box_data:
        df = pd.DataFrame(box_data)
        palette = {dt.upper(): PALETTE['Neurostim'] for dt in dataset_types}
        palette['GP Ref'] = PALETTE['Synthetic GP']
        palette['Prior Ref'] = PALETTE['TabPFN Prior']
        palette['Noise OOD'] = PALETTE['Noise (OOD)']
        sns.boxplot(data=df, x='Dataset', y='Entropy', palette=palette,
                    ax=ax, fliersize=2)
    ax.grid(True, alpha=0.3, axis='y')


def _panel_mmd(ax, mmd_results):
    """Compact MMD barplot for dashboard."""
    mmd_data = []
    ref_label_map = {'gp': 'vs GP', 'prior': 'vs Prior', 'noise': 'vs Noise'}
    for dt, ds_data in mmd_results.items():
        for subj_data in ds_data.values():
            for emg_data in subj_data.values():
                for ref_name in ['gp', 'prior', 'noise']:
                    mmd_key = f'mmd2_{ref_name}'
                    if mmd_key in emg_data:
                        mmd_data.append({
                            'Dataset': dt.upper(),
                            'MMD²': emg_data[mmd_key],
                            'Reference': ref_label_map[ref_name],
                        })
    if mmd_data:
        df = pd.DataFrame(mmd_data)
        sns.barplot(data=df, x='Dataset', y='MMD²', hue='Reference',
                    palette=REF_PALETTE, hue_order=REF_HUE_ORDER,
                    ax=ax, capsize=0.1, errorbar=('ci', 95))
    ax.grid(True, alpha=0.3, axis='y')


def _panel_mahalanobis(ax, mahalanobis_results):
    """Compact Mahalanobis boxplot for dashboard."""
    m_dataset_types = [k for k in mahalanobis_results if k != 'ref_stats']
    mah_data = []
    ref_label_map = {'gp': 'vs GP', 'prior': 'vs Prior', 'noise': 'vs Noise'}
    for dt in m_dataset_types:
        for subj_data in mahalanobis_results[dt].values():
            for emg_data in subj_data.values():
                for ref_name in ['gp', 'prior', 'noise']:
                    dist_key = f'distances_{ref_name}'
                    if dist_key in emg_data:
                        for d in emg_data[dist_key]:
                            mah_data.append({
                                'Dataset': dt.upper(),
                                'D_M': float(d),
                                'Reference': ref_label_map[ref_name],
                            })
    if mah_data:
        df = pd.DataFrame(mah_data)
        sns.boxplot(data=df, x='Dataset', y='D_M', hue='Reference',
                    palette=REF_PALETTE, hue_order=REF_HUE_ORDER,
                    ax=ax, fliersize=2)
    ax.grid(True, alpha=0.3, axis='y')


def _panel_cka(ax, cka_results):
    """Compact CKA grouped bar for dashboard (uses deepest analyzed layer)."""
    layers = _get_cka_layers(cka_results)
    layer = max(layers)

    cka_data = []
    ref_label_map = {'gp': 'vs GP', 'prior': 'vs Prior', 'noise': 'vs Noise'}
    for dt in _get_cka_dataset_types(cka_results):
        ds_data = cka_results[dt]
        for subj_data in ds_data.values():
            for emg_data in subj_data.values():
                if layer not in emg_data:
                    continue
                layer_data = emg_data[layer]
                for ref_name in ['gp', 'prior', 'noise']:
                    key = f'cka_{ref_name}'
                    if key in layer_data:
                        cka_data.append({
                            'Dataset': dt.upper(),
                            'CKA': layer_data[key],
                            'Reference': ref_label_map[ref_name],
                        })
    if cka_data:
        df = pd.DataFrame(cka_data)
        sns.barplot(data=df, x='Dataset', y='CKA', hue='Reference',
                    palette=REF_PALETTE, hue_order=REF_HUE_ORDER,
                    ax=ax, capsize=0.1, errorbar=('ci', 95))
        ax.set_ylim(0, 1)
    ax.set_title(f'CKA (Layer {layer})')
    ax.grid(True, alpha=0.3, axis='y')


def _panel_wasserstein(ax, wasserstein_results):
    """Compact Wasserstein barplot for dashboard."""
    w_data = []
    ref_label_map = {'gp': 'vs GP', 'prior': 'vs Prior', 'noise': 'vs Noise'}
    for dt, ds_data in wasserstein_results.items():
        for subj_data in ds_data.values():
            for emg_data in subj_data.values():
                for ref_name in ['gp', 'prior', 'noise']:
                    key = f'w2_{ref_name}'
                    if key in emg_data:
                        w_data.append({
                            'Dataset': dt.upper(),
                            'W2': emg_data[key],
                            'Reference': ref_label_map[ref_name],
                        })
    if w_data:
        df = pd.DataFrame(w_data)
        sns.barplot(data=df, x='Dataset', y='W2', hue='Reference',
                    palette=REF_PALETTE, hue_order=REF_HUE_ORDER,
                    ax=ax, capsize=0.1, errorbar=('ci', 95))
    ax.grid(True, alpha=0.3, axis='y')


def _panel_gradient_norm(ax, gradient_results):
    """Compact gradient norm bars for dashboard."""
    _ref_keys = {'synthetic_gp', 'synthetic_prior', 'noise'}
    dataset_types = [k for k in gradient_results if k not in _ref_keys]

    labels, values = [], []
    for dt in dataset_types:
        ds_data = gradient_results[dt]
        for subj_idx in sorted(ds_data.keys()):
            norms = [v for v in ds_data[subj_idx].values() if np.isfinite(v)]
            if norms:
                labels.append(f'{dt} S{subj_idx}')
                values.append(np.mean(norms))

    if values:
        x = np.arange(len(labels))
        ax.bar(x, values, color=PALETTE['Neurostim'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yscale('log')

        ref_line_config = {
            'synthetic_gp': ('GP', PALETTE['Synthetic GP'], '--'),
            'synthetic_prior': ('Prior', PALETTE['TabPFN Prior'], '-.'),
            'noise': ('Noise', PALETTE['Noise (OOD)'], ':'),
        }
        for ref_key, (label, color, ls) in ref_line_config.items():
            if ref_key in gradient_results and len(gradient_results[ref_key]) > 0:
                ax.axhline(np.mean(gradient_results[ref_key]),
                           color=color, linestyle=ls, linewidth=1.5,
                           label=label)
        ax.legend(fontsize=6)

    ax.set_ylabel('Grad L2 (log)')
    ax.grid(True, alpha=0.3, axis='y')


# ============================================================================
#  RSA Line Plot
# ============================================================================

def plot_rsa_layerwise(
    rsa_results: dict,
    save: bool = False,
    output_dir: str | None = None,
) -> None:
    """Line plot: RSA Spearman rho vs transformer layer index.

    One subplot per dataset type; one line per reference type.
    Reveals where neurostim geometry tracks the prior's geometry.

    Args:
        rsa_results: dict from rsa_analysis().
        save: Whether to save the figure to disk.
        output_dir: Base output directory. Saves to
            <output_dir>/rsa/rsa_by_layer.svg.
    """
    dataset_types = [k for k in rsa_results if k != 'layers']
    layers = rsa_results.get('layers', [4, 13, 17])
    refs = ['gp', 'prior', 'noise']
    ref_labels = {
        'gp': 'Synthetic GP',
        'prior': 'TabPFN Prior',
        'noise': 'Noise (OOD)',
    }

    n_datasets = len(dataset_types)
    if n_datasets == 0:
        return

    fig, axes = plt.subplots(1, n_datasets,
                              figsize=(6 * n_datasets, 4),
                              squeeze=False)

    for col, dataset_type in enumerate(dataset_types):
        ax = axes[0, col]
        for ref in refs:
            rho_per_layer = []
            for layer_idx in layers:
                rhos = [
                    rsa_results[dataset_type][subj][emg][layer_idx].get(
                        f'rsa_{ref}', np.nan,
                    )
                    for subj in rsa_results[dataset_type]
                    for emg in rsa_results[dataset_type][subj]
                    if layer_idx in rsa_results[dataset_type][subj][emg]
                ]
                finite = [r for r in rhos if not np.isnan(r)]
                rho_per_layer.append(np.mean(finite) if finite else np.nan)

            label = ref_labels[ref]
            ax.plot(layers, rho_per_layer,
                    label=label,
                    color=PALETTE[label],
                    marker='o',
                    linewidth=1.8)

        ax.set_xlabel('Transformer layer')
        ax.set_ylabel('RSA Spearman ρ')
        ax.set_title(f'{dataset_type.upper()} — RSA by Layer')
        ax.set_ylim(-1, 1)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('RSA: Neurostim Geometry vs Synthetic Reference Embeddings',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        base = _save_dir(output_dir, 'rsa')
        path = os.path.join(base, 'rsa_by_layer.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


# ============================================================================
#  Procrustes BO-Trajectory (B7)
# ============================================================================

def plot_procrustes_trajectory(
    trajectory_results: dict,
    save: bool = False,
    output_dir: str | None = None,
) -> None:
    """Plot Procrustes disparity vs BO budget (B7).

    One line per source (neurostim datasets + synthetic GP / TabPFN Prior /
    Noise).  Mean ± 1 std across trajectories (one trajectory per neurostim
    (subject, EMG) pair or per synthetic dataset).  Disparity at
    budgets[0] is 0 by construction (self-comparison baseline).

    Args:
        trajectory_results: dict from ``embedding_trajectory_analysis()``.
        save: Whether to save the figure.
        output_dir: Base output directory.  Saves to
            ``<output_dir>/trajectory/procrustes_disparity_vs_budget.svg``.
    """
    budgets = trajectory_results.get('budgets', [])
    layer = trajectory_results.get('layer', 17)
    if not budgets:
        print("  [Procrustes] No budgets in results, skipping plot")
        return

    # Group trajectories per labelled source
    curves: dict[str, np.ndarray] = {}

    meta_keys = {'budgets', 'layer', 'cross_distribution'}
    synthetic_prefix = 'synthetic_'
    dataset_types = [k for k in trajectory_results
                     if k not in meta_keys and not k.startswith(synthetic_prefix)]

    # Neurostim: one curve per dataset, pooling subj/emg pairs
    for dt in dataset_types:
        trajs = []
        for subj_data in trajectory_results[dt].values():
            for traj in subj_data.values():
                trajs.append(traj)
        if trajs:
            curves[f'{dt.upper()} Neurostim'] = np.asarray(trajs, dtype=float)

    # Synthetic references
    synth_labels = {
        'synthetic_gp': 'Synthetic GP',
        'synthetic_prior': 'TabPFN Prior',
        'synthetic_noise': 'Noise (OOD)',
    }
    for key, label in synth_labels.items():
        trajs = trajectory_results.get(key, [])
        if trajs:
            curves[label] = np.asarray(trajs, dtype=float)

    if not curves:
        print("  [Procrustes] No trajectories to plot")
        return

    neurostim_palette = {
        'NHP Neurostim': PALETTE['Neurostim'],
        'RAT Neurostim': PALETTE['Neurostim GT'],
        'SPINAL Neurostim': '#2f6fa0',
    }
    synth_palette = {
        'Synthetic GP': PALETTE['Synthetic GP'],
        'TabPFN Prior': PALETTE['TabPFN Prior'],
        'Noise (OOD)': PALETTE['Noise (OOD)'],
    }
    palette = {**neurostim_palette, **synth_palette}

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, arr in curves.items():
        # arr: [n_trajectories, n_budgets]
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        color = palette.get(label, 'gray')
        ax.plot(budgets, mean, label=f'{label} (n={len(arr)})',
                color=color, marker='o', linewidth=1.8)
        ax.fill_between(budgets, mean - std, mean + std,
                        color=color, alpha=0.18)

    ax.set_xlabel('BO budget (# context observations)')
    ax.set_ylabel(f'Procrustes disparity vs E(t={budgets[0]})')
    ax.set_title(
        f'B7 — Embedding Geometry Trajectory during BO (layer {layer})',
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'trajectory')
        path = os.path.join(base, 'procrustes_disparity_vs_budget.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def _panel_procrustes(ax: plt.Axes, trajectory_results: dict) -> None:
    """Compact Procrustes trajectory line plot for summary dashboard."""
    budgets = trajectory_results.get('budgets', [])
    if not budgets:
        ax.set_visible(False)
        return

    meta_keys = {'budgets', 'layer', 'cross_distribution'}
    synthetic_prefix = 'synthetic_'
    dataset_types = [k for k in trajectory_results
                     if k not in meta_keys and not k.startswith(synthetic_prefix)]

    sources: dict[str, tuple[np.ndarray, str]] = {}
    for dt in dataset_types:
        trajs = [traj for subj in trajectory_results[dt].values()
                 for traj in subj.values()]
        if trajs:
            sources[f'{dt.upper()}'] = (np.asarray(trajs, dtype=float),
                                        PALETTE['Neurostim'])
    for key, (label, color) in {
        'synthetic_gp':    ('GP',    PALETTE['Synthetic GP']),
        'synthetic_prior': ('Prior', PALETTE['TabPFN Prior']),
        'synthetic_noise': ('Noise', PALETTE['Noise (OOD)']),
    }.items():
        if trajectory_results.get(key):
            sources[label] = (np.asarray(trajectory_results[key], dtype=float),
                              color)

    for label, (arr, color) in sources.items():
        mean = np.nanmean(arr, axis=0)
        ax.plot(budgets, mean, label=label, color=color,
                marker='o', linewidth=1.5)

    ax.set_xlabel('BO budget')
    ax.set_ylabel('Procrustes disp.')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)


def plot_procrustes_auc_bar(
    auc_results: dict,
    save: bool = False,
    output_dir: str | None = None,
) -> None:
    """Bar chart of AUC (area under Procrustes disparity curve) per source.

    Integrating each disparity curve collapses the temporal dimension into a
    single scalar per trajectory.  Bars are sorted ascending so that ID
    sources (low AUC, geometrically stable) appear on the left and OOD
    sources (high AUC, erratic drift) appear on the right.

    Args:
        auc_results: dict from :func:`compute_procrustes_auc`.
        save: Whether to save the figure.
        output_dir: Base output directory.  Saves to
            ``<output_dir>/trajectory/procrustes_auc_bar.svg``.
    """
    if not auc_results:
        print("  [Procrustes AUC] No AUC results to plot")
        return

    labels = list(auc_results.keys())
    means = np.array([auc_results[l]['mean'] for l in labels])
    stds = np.array([auc_results[l]['std'] for l in labels])

    order = np.argsort(means)
    labels = [labels[i] for i in order]
    means = means[order]
    stds = stds[order]

    colors = []
    for lbl in labels:
        if 'Noise' in lbl:
            colors.append(PALETTE['Noise (OOD)'])
        elif 'GP' in lbl and 'Neuro' not in lbl:
            colors.append(PALETTE['Synthetic GP'])
        elif 'Prior' in lbl:
            colors.append(PALETTE['TabPFN Prior'])
        else:
            colors.append(PALETTE['Neurostim'])

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors,
           alpha=0.85, edgecolor='black', linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('AUC (Procrustes disparity × budget)')
    ax.set_title('Procrustes Trajectory AUC — geometric drift per source')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'trajectory')
        path = os.path.join(base, 'procrustes_auc_bar.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def plot_procrustes_per_subject(
    trajectory_results: dict,
    save: bool = False,
    output_dir: str | None = None,
) -> None:
    """Procrustes disparity curves split by subject (neurostim datasets only).

    Per-subject traces shown as faint lines (one per subject, averaged over
    that subject's EMG channels).  Bold line = grand mean ± 1 std band across
    subjects.  One subplot per dataset type.  Reveals inter-subject
    heterogeneity in geometric drift rates.

    Args:
        trajectory_results: dict from :func:`embedding_trajectory_analysis`.
        save: Whether to save the figure.
        output_dir: Base output directory.  Saves to
            ``<output_dir>/trajectory/procrustes_per_subject.svg``.
    """
    budgets = trajectory_results.get('budgets', [])
    if not budgets:
        print("  [Procrustes] No budgets in results, skipping per-subject plot")
        return

    meta_keys = {'budgets', 'layer', 'cross_distribution'}
    synthetic_prefix = 'synthetic_'
    dataset_types = [k for k in trajectory_results
                     if k not in meta_keys and not k.startswith(synthetic_prefix)]

    if not dataset_types:
        print("  [Procrustes] No neurostim datasets found, skipping per-subject plot")
        return

    n_panels = len(dataset_types)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    layer = trajectory_results.get('layer', 17)

    for col, dt in enumerate(dataset_types):
        ax = axes[0, col]
        subj_means: list[np.ndarray] = []

        for subj_idx, subj_data in trajectory_results[dt].items():
            emg_trajs = [
                np.array(traj, dtype=float)
                for traj in subj_data.values()
                if np.all(np.isfinite(traj))
            ]
            if not emg_trajs:
                continue
            subj_mean = np.nanmean(np.array(emg_trajs), axis=0)  # [n_budgets]
            subj_means.append(subj_mean)
            ax.plot(budgets, subj_mean, color=PALETTE['Neurostim'],
                    alpha=0.35, linewidth=1.0, label=f'S{subj_idx}')

        if subj_means:
            arr = np.array(subj_means)                    # [n_subj, n_budgets]
            mean_all = np.nanmean(arr, axis=0)
            std_all = np.nanstd(arr, axis=0)
            ax.plot(budgets, mean_all, color=PALETTE['Neurostim'],
                    linewidth=2.2,
                    label=f'Mean (n={len(subj_means)} subjects)')
            ax.fill_between(budgets,
                            mean_all - std_all, mean_all + std_all,
                            color=PALETTE['Neurostim'], alpha=0.18)

        ax.set_xlabel('BO budget (# context observations)')
        ax.set_ylabel('Procrustes disparity')
        ax.set_title(f'{dt.upper()} — Subject-level Trajectory (layer {layer})')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'trajectory')
        path = os.path.join(base, 'procrustes_per_subject.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def plot_cross_distribution_disparity(
    cross_results: dict,
    save: bool = False,
    output_dir: str | None = None,
) -> None:
    """Cross-distribution geometric distance vs BO budget.

    Plots the absolute difference in mean Procrustes disparity between each
    neurostim dataset and each synthetic reference (GP, Noise) at each budget
    step.  A neurostim curve close to Synthetic GP (low Δ) indicates ID-like
    geometric behaviour; a curve close to Noise (high Δ) indicates OOD-like
    behaviour.

    Args:
        cross_results: dict from :func:`compute_cross_distribution_procrustes`,
            with keys ``'budgets'`` and one entry per dataset_type.
        save: Whether to save the figure.
        output_dir: Base output directory.  Saves to
            ``<output_dir>/trajectory/cross_distribution_disparity.svg``.
    """
    budgets = cross_results.get('budgets', [])
    dataset_types = [k for k in cross_results if k != 'budgets']

    if not budgets or not dataset_types:
        print("  [Procrustes] Empty cross-distribution results, skipping plot")
        return

    ref_styles: dict[str, dict] = {
        'Synthetic GP': {'color': PALETTE['Synthetic GP'], 'ls': '-'},
        'Noise (OOD)':  {'color': PALETTE['Noise (OOD)'],  'ls': '--'},
    }
    dt_markers = {'nhp': 'o', 'rat': 's', 'spinal': '^'}

    fig, ax = plt.subplots(figsize=(8, 5))

    for dt in dataset_types:
        for ref_label, diffs in cross_results[dt].items():
            style = ref_styles.get(ref_label, {'color': 'gray', 'ls': ':'})
            marker = dt_markers.get(dt, 'D')
            ax.plot(budgets, diffs,
                    color=style['color'], linestyle=style['ls'],
                    linewidth=1.8, marker=marker, markersize=5,
                    label=f'{dt.upper()} vs {ref_label}')

    ax.set_xlabel('BO budget (# context observations)')
    ax.set_ylabel('|Δ mean Procrustes disparity|')
    ax.set_title('Cross-Distribution Geometric Distance vs BO Budget')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'trajectory')
        path = os.path.join(base, 'cross_distribution_disparity.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


# ============================================================================
#  Tier 1 Consensus Analyses — B8, B9, B10
# ============================================================================

# Metric display labels for consensus plots.
_METRIC_LABELS = {
    'entropy': 'Entropy',
    'mmd': 'MMD²',
    'wasserstein': 'Wasserstein',
    'mahalanobis': 'Mahalanobis',
    'cka': 'CKA',
    'rsa': 'RSA',
}

# Metrics where higher = more ID (need inversion for [0=ID,1=OOD] normalization).
_SIMILARITY_METRICS = {'cka', 'rsa'}


def _extract_scalar_per_emg(
    all_results: dict,
) -> tuple[list[tuple[str, int, int]], dict[str, list[float]]]:
    """Extract one scalar score per (dataset, subject, emg) for each metric.

    Used by both B8 (consensus matrix) and B10 (heterogeneity heatmap).
    All returned lists are co-indexed: ``labels[i]`` corresponds to
    ``scores[metric][i]`` for every metric.

    For CKA and RSA, uses the deepest available transformer layer.

    Args:
        all_results: dict from :func:`run_id_ood_analysis`.

    Returns:
        ``(labels, scores)`` where ``labels`` is a list of
        ``(dataset_type, subj_idx, emg_idx)`` tuples and ``scores`` is a
        dict mapping metric name → list of floats (same length as labels).
        Entries where a metric is unavailable for a given (dataset,subj,emg)
        are stored as NaN.
    """
    ref_skip = {'ref_stats', 'layers', 'synthetic_gp', 'synthetic_prior',
                'noise', 'budgets', 'layer', 'cross_distribution'}

    # Determine the union of (dataset, subj, emg) present in any metric.
    # Use entropy or mmd as the primary key source; fall back to any available.
    key_source = None
    for candidate in ('entropy', 'mmd', 'wasserstein', 'mahalanobis', 'cka', 'rsa'):
        if candidate in all_results:
            key_source = candidate
            break
    if key_source is None:
        return [], {}

    primary = all_results[key_source]
    labels: list[tuple[str, int, int]] = []
    for ds_key, ds_data in primary.items():
        if ds_key in ref_skip or not isinstance(ds_data, dict):
            continue
        for subj_idx, subj_data in ds_data.items():
            if not isinstance(subj_data, dict):
                continue
            for emg_idx in subj_data:
                labels.append((ds_key, subj_idx, emg_idx))

    scores: dict[str, list[float]] = {m: [] for m in _METRIC_LABELS}

    # ── Entropy ──────────────────────────────────────────────────────────
    for ds, subj, emg in labels:
        val = np.nan
        if 'entropy' in all_results:
            node = (all_results['entropy']
                    .get(ds, {}).get(subj, {}).get(emg))
            if node is not None:
                arr = np.asarray(node, dtype=float)
                valid = arr[np.isfinite(arr)]
                val = float(np.mean(valid)) if len(valid) > 0 else np.nan
        scores['entropy'].append(val)

    # ── MMD ──────────────────────────────────────────────────────────────
    for ds, subj, emg in labels:
        val = np.nan
        if 'mmd' in all_results:
            node = (all_results['mmd']
                    .get(ds, {}).get(subj, {}).get(emg, {}))
            v = node.get('mmd2_gp')
            if v is not None:
                val = float(v)
        scores['mmd'].append(val)

    # ── Wasserstein ───────────────────────────────────────────────────────
    for ds, subj, emg in labels:
        val = np.nan
        if 'wasserstein' in all_results:
            node = (all_results['wasserstein']
                    .get(ds, {}).get(subj, {}).get(emg, {}))
            v = node.get('w2_gp')
            if v is not None:
                val = float(v)
        scores['wasserstein'].append(val)

    # ── Mahalanobis ───────────────────────────────────────────────────────
    for ds, subj, emg in labels:
        val = np.nan
        if 'mahalanobis' in all_results:
            node = (all_results['mahalanobis']
                    .get(ds, {}).get(subj, {}).get(emg, {}))
            v = node.get('mean_dist_gp')
            if v is not None:
                val = float(v)
        scores['mahalanobis'].append(val)

    # ── CKA (deepest layer) ───────────────────────────────────────────────
    cka_layer = None
    if 'cka' in all_results:
        cka_layer = max(all_results['cka'].get('layers', [17]))
    for ds, subj, emg in labels:
        val = np.nan
        if 'cka' in all_results and cka_layer is not None:
            node = (all_results['cka']
                    .get(ds, {}).get(subj, {}).get(emg, {}))
            layer_node = node.get(cka_layer, {})
            v = layer_node.get('cka_gp')
            if v is not None:
                val = float(v)
        scores['cka'].append(val)

    # ── RSA (deepest layer) ───────────────────────────────────────────────
    rsa_layer = None
    if 'rsa' in all_results:
        rsa_layer = max(all_results['rsa'].get('layers', [17]))
    for ds, subj, emg in labels:
        val = np.nan
        if 'rsa' in all_results and rsa_layer is not None:
            node = (all_results['rsa']
                    .get(ds, {}).get(subj, {}).get(emg, {}))
            layer_node = node.get(rsa_layer, {})
            v = layer_node.get('rsa_gp')
            if v is not None:
                val = float(v)
        scores['rsa'].append(val)

    # Drop metrics with no data at all.
    scores = {m: v for m, v in scores.items() if any(np.isfinite(x) for x in v)}

    return labels, scores


def _normalize_scores(
    scores: dict[str, list[float]],
) -> dict[str, np.ndarray]:
    """Min-max normalize each metric to [0=ID, 1=OOD].

    Similarity metrics (CKA, RSA) are inverted so that 0 always means
    most ID-like and 1 means most OOD-like.

    Args:
        scores: dict from :func:`_extract_scalar_per_emg`.

    Returns:
        dict mapping metric name → normalized float array.
    """
    norm: dict[str, np.ndarray] = {}
    for metric, vals in scores.items():
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if len(finite) < 2:
            norm[metric] = arr
            continue
        lo, hi = float(finite.min()), float(finite.max())
        if hi == lo:
            norm[metric] = np.zeros_like(arr)
        else:
            normalized = (arr - lo) / (hi - lo)
            if metric in _SIMILARITY_METRICS:
                normalized = 1.0 - normalized
            norm[metric] = normalized
    return norm


def plot_metric_consensus(
    all_results: dict,
    save: bool = False,
    output_dir: str | None = None,
) -> None:
    """B8 — Metric consensus heatmap + composite ID score bar chart.

    Heatmap: 6×6 Spearman ρ between normalized per-(dataset,subj,emg)
    metric scores.  ρ > 0.7 = redundant metrics; ρ < 0 = contradicting
    evidence.

    Bar chart: composite ID score (mean of normalized metrics) per
    (dataset,subj,emg) sorted ascending; lower = more ID-like.

    Args:
        all_results: dict from :func:`run_id_ood_analysis`.
        save: Whether to save figures.
        output_dir: Base output directory.  Saves to
            ``<output_dir>/consensus/``.
    """
    from scipy.stats import spearmanr

    labels, scores = _extract_scalar_per_emg(all_results)
    if len(scores) < 2:
        print("  [B8] Need ≥2 metrics for consensus matrix — skipping.")
        return

    norm = _normalize_scores(scores)
    metric_names = list(norm.keys())
    n = len(metric_names)

    # ── Heatmap ──────────────────────────────────────────────────────────
    rho_matrix = np.full((n, n), np.nan)
    for i, mi in enumerate(metric_names):
        for j, mj in enumerate(metric_names):
            vi = norm[mi]
            vj = norm[mj]
            both = np.isfinite(vi) & np.isfinite(vj)
            if both.sum() >= 4:
                rho_matrix[i, j] = float(spearmanr(vi[both], vj[both]).statistic)

    disp_labels = [_METRIC_LABELS.get(m, m) for m in metric_names]
    fig, ax = plt.subplots(figsize=(5 + 0.5 * n, 4 + 0.4 * n))
    im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                   aspect='auto')
    plt.colorbar(im, ax=ax, label='Spearman ρ')
    ax.set_xticks(range(n)); ax.set_xticklabels(disp_labels, rotation=35, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(disp_labels)
    for i in range(n):
        for j in range(n):
            if np.isfinite(rho_matrix[i, j]):
                ax.text(j, i, f'{rho_matrix[i, j]:.2f}',
                        ha='center', va='center',
                        fontsize=8,
                        color='white' if abs(rho_matrix[i, j]) > 0.5 else 'black')
    ax.set_title('B8 — Metric Consensus (Spearman ρ)')
    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'consensus')
        path = os.path.join(base, 'metric_consensus_heatmap.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")
    plt.close()

    # ── Composite score bar chart ─────────────────────────────────────────
    stacked = np.column_stack([norm[m] for m in metric_names])  # [N, n_metrics]
    composite = np.nanmean(stacked, axis=1)                     # [N]

    order = np.argsort(composite)
    sorted_composite = composite[order]
    sorted_labels = [labels[i] for i in order]

    # Color by dataset type
    ds_palette = {'nhp': PALETTE['Neurostim'],
                  'rat': PALETTE['Neurostim GT'],
                  'spinal': '#2f6fa0'}
    colors = [ds_palette.get(lbl[0], 'gray') for lbl in sorted_labels]
    x_ticks = [f"{lbl[0].upper()} S{lbl[1]} E{lbl[2]}" for lbl in sorted_labels]

    fig2, ax2 = plt.subplots(figsize=(max(8, len(sorted_labels) * 0.35), 4))
    ax2.bar(range(len(sorted_labels)), sorted_composite, color=colors, alpha=0.85)
    ax2.set_xticks(range(len(sorted_labels)))
    ax2.set_xticklabels(x_ticks, rotation=70, ha='right', fontsize=6)
    ax2.set_ylabel('Composite ID Score (0=ID, 1=OOD)')
    ax2.set_title('B8 — Composite ID Score per (Dataset, Subject, EMG)')
    ax2.grid(True, axis='y', alpha=0.3)
    fig2.tight_layout()

    if save:
        base = _save_dir(output_dir, 'consensus')
        path2 = os.path.join(base, 'composite_score_bar.svg')
        plt.savefig(path2, format='svg')
        print(f"Saved plot -> {path2}")
    plt.close()


def plot_synthetic_validation(
    cohens_d_results: dict,
    save: bool = False,
    output_dir: str | None = None,
) -> None:
    """B9 — Synthetic reference validation: Cohen's d per metric.

    Two-panel figure:
    - Left: violin plot of GP vs Noise score distributions per metric.
    - Right: Cohen's d bar chart with 0.8 reference line (strong discriminator
      threshold per Cohen 1988).

    Also writes a CSV table: metric × {mean_gp, mean_noise, cohens_d, verdict}.

    Args:
        cohens_d_results: dict from :func:`compute_cohens_d_per_metric`.
        save: Whether to save figures and CSV.
        output_dir: Base output directory.  Saves to
            ``<output_dir>/consensus/``.
    """
    if not cohens_d_results:
        print("  [B9] No Cohen's d results to plot.")
        return

    metrics = list(cohens_d_results.keys())

    # ── Violin plot ───────────────────────────────────────────────────────
    violin_rows = []
    for metric, info in cohens_d_results.items():
        label = _METRIC_LABELS.get(metric, metric)
        for v in info['gp_scores']:
            violin_rows.append({'Metric': label, 'Group': 'GP (ID)', 'Score': float(v)})
        for v in info['noise_scores']:
            violin_rows.append({'Metric': label, 'Group': 'Noise (OOD)', 'Score': float(v)})

    disp_labels = [_METRIC_LABELS.get(m, m) for m in metrics]
    d_vals = [cohens_d_results[m]['cohens_d'] for m in metrics]
    verdicts = [cohens_d_results[m]['verdict'] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(4 + 2 * len(metrics), 5))

    # Panel 1: violin
    if violin_rows:
        df_v = pd.DataFrame(violin_rows)
        sns.violinplot(data=df_v, x='Metric', y='Score', hue='Group',
                       order=disp_labels,
                       palette={'GP (ID)': PALETTE['Synthetic GP'],
                                'Noise (OOD)': PALETTE['Noise (OOD)']},
                       ax=axes[0], inner='box', alpha=0.75, split=False)
        axes[0].set_title('B9 — Score Distributions: GP vs Noise')
        axes[0].set_xlabel('')
        axes[0].tick_params(axis='x', rotation=30)
        axes[0].grid(True, alpha=0.3, axis='y')

    # Panel 2: Cohen's d bar
    ax2 = axes[1]
    bar_colors = [
        'forestgreen' if v == 'strong' else ('goldenrod' if v == 'moderate' else 'firebrick')
        for v in verdicts
    ]
    finite_d = [d if np.isfinite(d) else 0.0 for d in d_vals]
    ax2.bar(disp_labels, finite_d, color=bar_colors, alpha=0.85, edgecolor='black', lw=0.6)
    ax2.axhline(0.8, color='black', linestyle='--', linewidth=1.2,
                label="Cohen's d = 0.8 (strong)")
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1.0,
                label="Cohen's d = 0.5 (moderate)")
    ax2.set_ylabel("Cohen's d")
    ax2.set_title("B9 — Discriminability (Cohen's d)")
    ax2.tick_params(axis='x', rotation=30)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'consensus')
        path = os.path.join(base, 'synthetic_validation.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

        # CSV table
        rows = []
        for metric in metrics:
            info = cohens_d_results[metric]
            rows.append({
                'metric': metric,
                'mean_gp': f"{info['mean_gp']:.4f}",
                'mean_noise': f"{info['mean_noise']:.4f}",
                'cohens_d': f"{info['cohens_d']:.4f}",
                'verdict': info['verdict'],
            })
        csv_path = os.path.join(base, 'synthetic_validation.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved table -> {csv_path}")

    plt.close()


def plot_id_ood_heterogeneity_heatmap(
    all_results: dict,
    dataset_type: str,
    save: bool = False,
    output_dir: str | None = None,
) -> None:
    """B10 — EMG/Subject heterogeneity heatmap for one dataset.

    Plots a Subject × EMG matrix colored by composite normalized ID score
    (0 = most ID-like, 1 = most OOD-like).  Extends the
    ``plot_entropy_heatmap()`` pattern to all available metrics.

    Cells with no data are left blank (NaN → colormap bad color).

    Args:
        all_results: dict from :func:`run_id_ood_analysis`.
        dataset_type: Which dataset to plot (e.g. ``'nhp'``).
        save: Whether to save the figure.
        output_dir: Base output directory.  Saves to
            ``<output_dir>/consensus/heterogeneity_heatmap_{dataset_type}.svg``.
    """
    labels, scores = _extract_scalar_per_emg(all_results)
    if not scores:
        print(f"  [B10] No metric scores for {dataset_type} — skipping.")
        return

    norm = _normalize_scores(scores)

    # Filter to this dataset type only.
    ds_labels = [(ds, subj, emg)
                 for ds, subj, emg in labels if ds == dataset_type]
    if not ds_labels:
        print(f"  [B10] No (subj,emg) pairs found for dataset '{dataset_type}'.")
        return

    idx_map = {lbl: i for i, lbl in enumerate(labels)}

    subjects = sorted({subj for _, subj, _ in ds_labels})
    emgs = sorted({emg for _, _, emg in ds_labels})

    matrix = np.full((len(subjects), len(emgs)), np.nan)
    n_metrics = len(norm)
    for si, subj in enumerate(subjects):
        for ei, emg in enumerate(emgs):
            key = (dataset_type, subj, emg)
            if key not in idx_map:
                continue
            idx = idx_map[key]
            vals = [norm[m][idx] for m in norm if np.isfinite(norm[m][idx])]
            if vals:
                matrix[si, ei] = float(np.mean(vals))

    fig, ax = plt.subplots(
        figsize=(max(5, len(emgs) * 0.9), max(3, len(subjects) * 0.7)),
    )
    sns.heatmap(
        matrix, ax=ax, cmap='YlOrRd',
        vmin=0.0, vmax=1.0,
        annot=True, fmt='.2f',
        xticklabels=[f'EMG {e}' for e in emgs],
        yticklabels=[f'S{s}' for s in subjects],
        linewidths=0.4,
    )
    ax.set_title(
        f'B10 — ID/OOD Heterogeneity | {dataset_type.upper()} '
        f'(n_metrics={n_metrics}, 0=ID, 1=OOD)'
    )
    ax.set_xlabel('EMG Channel')
    ax.set_ylabel('Subject')
    fig.tight_layout()

    if save:
        base = _save_dir(output_dir, 'consensus')
        path = os.path.join(base, f'heterogeneity_heatmap_{dataset_type}.svg')
        plt.savefig(path, format='svg')
        print(f"Saved plot -> {path}")

    plt.close()


def _panel_rsa(ax: plt.Axes, rsa_results: dict) -> None:
    """Compact RSA barplot for summary dashboard at deepest analyzed layer.

    Args:
        ax: Axes to draw on.
        rsa_results: dict from rsa_analysis().
    """
    layers = rsa_results.get('layers', [4, 13, 17])
    layer = max(layers)
    dataset_types = [k for k in rsa_results if k != 'layers']
    refs = ['gp', 'prior', 'noise']
    ref_labels = {'gp': 'vs GP', 'prior': 'vs Prior', 'noise': 'vs Noise'}

    rsa_data = []
    for dt in dataset_types:
        for subj_data in rsa_results[dt].values():
            for emg_data in subj_data.values():
                if layer not in emg_data:
                    continue
                for ref in refs:
                    key = f'rsa_{ref}'
                    if key in emg_data[layer]:
                        rsa_data.append({
                            'Dataset': dt.upper(),
                            'RSA ρ': emg_data[layer][key],
                            'Reference': ref_labels[ref],
                        })

    if rsa_data:
        df = pd.DataFrame(rsa_data)
        sns.barplot(data=df, x='Dataset', y='RSA ρ', hue='Reference',
                    palette=REF_PALETTE, hue_order=REF_HUE_ORDER,
                    ax=ax, capsize=0.1, errorbar=('ci', 95))

    ax.set_ylim(-1, 1)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(f'RSA ρ (Layer {layer})')
    ax.grid(True, alpha=0.3, axis='y')
