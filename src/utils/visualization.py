import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ============================================
#           Visualization
# ============================================

def r2_comparison(gp_results, pfn_results, save=False):

    data = []

    for res_gp, res_pfn in zip(gp_results, pfn_results):

        for score in res_gp['r2']:
            data.append({
                'muscle': f"S{res_gp['subject']} EMG {res_gp['emg']}",
                'R2': score,
                'Model': 'GP'
            })

        for score in res_pfn['r2']:
            data.append({
                'muscle': f"S{res_pfn['subject']}EMG {res_pfn['emg']}",
                'R2': score,
                'Model': 'PFN'
            })

    custom_palette = {
    'GP': 'sandybrown',
    'PFN': 'royalblue'
    }
  
    df = pd.DataFrame(data)
    plt.figure(figsize=(1.4*len(gp_results), 6))
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    sns.barplot(data=df, x='muscle', y='R2', hue='Model', palette=custom_palette, errorbar=('ci', 95))
    plt.title("R2 Score Comparison: GP vs PFN")
    output_dir = os.path.join('output', 'fitness', f'{gp_results[0]["dataset"]}')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f'r2_comparison.svg'
    )
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")
    plt.show()
    plt.close()

def plot_runtime_trajectory(gp_results, pfn_results, save=False):
    """
    Plots the inference time at each BO step.
    Expects results to be a list, we will aggregate or plot just the first one as example.
    """
    plt.figure(figsize=(8, 6))
    
    
    all_gp_times = []
    all_pfn_times = []
    
    for res in gp_results:
        plt.plot(res['times'], color='red', alpha=0.1)
        all_gp_times.append(res['times'])
        
    for res in pfn_results:
        plt.plot(res['times'], color='blue', alpha=0.1)
        all_pfn_times.append(res['times'])
        
    # Plot Mean
    avg_gp = np.mean(all_gp_times, axis=0)
    avg_pfn = np.mean(all_pfn_times, axis=0)
    
    plt.plot(avg_gp, color='sandybrown', linewidth=2, label='GP (GPyTorch)')
    plt.plot(avg_pfn, color='royalblue', linewidth=2, label='PFN')
    
    plt.title("Inference Time for BO")
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_dir = os.path.join('output', 'optimization')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f'runtime_trajectory.svg'
    )
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")
    plt.show()
    plt.close()
    
def show_emg_map(results, idx, model_type, save=False):
    res = results[idx]
    
    y_true = res['y_test']
    y_pred = res['y_pred']
    r2_score = np.mean(np.array(res['r2']))
    
    # Determine Grid Shape
    n_channels = len(y_true)
    if n_channels == 96: grid_shape = (8, 12) 
    elif n_channels == 32: grid_shape = (4, 8)
    else: grid_shape = (1, n_channels) # Fallback

    # Calculate global scale for a unified cmap
    v_min = min(y_true.min(), y_pred.min())
    v_max = max(y_true.max(), y_pred.max())

    map_true = y_true.reshape(grid_shape)
    map_pred   = y_pred.reshape(grid_shape)

    # Find coordinates of the max values
    max_idx_true = np.unravel_index(np.argmax(map_true), grid_shape)
    max_idx_pred = np.unravel_index(np.argmax(map_pred), grid_shape)
    
    # Setup Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Common keyword arguments for consistency
    heatmap_kwargs = {
        'cmap': 'viridis',
        'vmin': v_min,
        'vmax': v_max,
    }
    
    # Plot Ground Truth
    sns.heatmap(y_true.reshape(8, 12), ax=ax[0], **heatmap_kwargs)
    ax[0].set_title(f"Ground Truth (EMG {res['emg']})")
    ax[0].plot(max_idx_true[1] + 0.5, max_idx_true[0] + 0.5, 'ro', markersize=8)
    
    # Plot Prediction
    sns.heatmap(y_pred.reshape(8, 12), ax=ax[1], **heatmap_kwargs)
    ax[1].set_title(f"{model_type} Prediction | R2:{r2_score:.2f}")
    ax[1].plot(max_idx_pred[1] + 0.5, max_idx_pred[0] + 0.5, 'ro', markersize=8)
    
    output_dir = os.path.join('output', 'fitness')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f'emg_map{idx}_{model_type}.svg'
    )
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")
    plt.show()
    plt.close()

def regret_curve(gp_vals, pfn_vals, idx, save=False):
    
    # Get metadata
    res_gp = gp_vals[idx]
    res_pfn = pfn_vals[idx]
    
    # Determine Optimal Value (Global Max)
    # Ideally this is the max of the ground truth test set
    optimal_val = res_gp['y_test'].max()
    
    def get_regret_stats(values_list):
        # 1. Convert to array: Shape (n_reps, n_steps)
        raw_vals = np.array(values_list)
        
        # 2. Calculate Best-So-Far for EACH repetition individually
        # We accumulate max along axis 1 (the time steps)
        best_so_far = np.maximum.accumulate(raw_vals, axis=1)
        
        # 3. Calculate Regret for EACH repetition
        # Regret = Optimal - Best_Observed_So_Far
        regret_all = optimal_val - best_so_far
        
        # 4. Compute Mean and Standard Error across repetitions (axis 0)
        mean_regret = np.mean(regret_all, axis=0)
        std_regret = np.std(regret_all, axis=0)
        n_reps = raw_vals.shape[0]
        
        # Standard Error for 95% CI
        se_regret = std_regret / np.sqrt(n_reps)
        
        return mean_regret, se_regret

    # Get Stats
    gp_mean, gp_se = get_regret_stats(res_gp['values'])
    pfn_mean, pfn_se = get_regret_stats(res_pfn['values'])
    
    x_axis = range(len(gp_mean))

    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    
    # 1. GP Plot
    plt.plot(x_axis, gp_mean, color='sandybrown', label='GP', linewidth=2)
    plt.fill_between(x_axis, 
                     gp_mean - 1.96 * gp_se, 
                     gp_mean + 1.96 * gp_se, 
                     color='sandybrown', alpha=0.2)

    # 2. PFN Plot
    plt.plot(x_axis, pfn_mean, color='royalblue', label='PFN', linewidth=2)
    plt.fill_between(x_axis, 
                     pfn_mean - 1.96 * pfn_se, 
                     pfn_mean + 1.96 * pfn_se, 
                     color='royalblue', alpha=0.2)

    dataset, subject, emg = res_gp['dataset'], res_gp['subject'], res_gp['emg']

    plt.xlabel('Iteration')
    plt.ylabel('Simple Regret')
    plt.title(f'Regret | {dataset} Subj {subject} EMG {emg}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_dir = os.path.join('output', 'optimization')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir,
        f'regret_curve{idx}.svg'
    )
    if save:
        plt.savefig(plot_path, format="svg")
        print(f"Saved plot to {plot_path}")
    plt.show()
    plt.close()
