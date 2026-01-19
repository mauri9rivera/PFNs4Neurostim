import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ============================================
#           Visualization
# ============================================

def r2_comparison(gp_results, pfn_results):

    data = []

    for res_gp, res_pfn in zip(gp_results, pfn_results):

        data.append({
                'EMG': f"EMG {res_gp['emg']}",
                'R2': res_gp['r2'],
                'Model': 'GP'
            })
        data.append({
                'EMG': f"EMG {res_pfn['emg']}",
                'R2': res_pfn['r2'],
                'Model': 'PFN'
            })
        
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 1)
    sns.barplot(data=df, x='EMG', y='R2', hue='Model')
    plt.title("R2 Score Comparison: GP vs PFN")
    plt.show()

def plot_runtime_trajectory(gp_results, pfn_results):
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
    
    plt.plot(avg_gp, color='red', linewidth=2, label='GP (GPyTorch)')
    plt.plot(avg_pfn, color='blue', linewidth=2, label='PFN')
    
    plt.title("Inference Time per BO")
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def show_emg_map(results, idx, model_type):
    res = results[idx]
    
    y_true = res['y_test']
    y_pred = res['y_pred']
    
    # Determine Grid Shape
    n_channels = len(y_true)
    if n_channels == 96: grid_shape = (8, 12) 
    elif n_channels == 32: grid_shape = (4, 8)
    else: grid_shape = (1, n_channels) # Fallback

    map_true = y_true.reshape(grid_shape)
    map_pred   = y_pred.reshape(grid_shape)
    
    # Setup Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Ground Truth
    sns.heatmap(y_true.reshape(8, 12), ax=ax[0], cmap='viridis')
    ax[0].set_title(f"Ground Truth (EMG {res['emg']})")
    
    # Plot Prediction
    sns.heatmap(y_pred.reshape(8, 12), ax=ax[1], cmap='viridis')
    ax[1].set_title(f"{model_type} Prediction")
    
    plt.show()

    plt.suptitle(f"Model Fit: {res['dataset']} | Subj {res['subject']} | EMG {res['emg']}", fontsize=14)
    plt.show()

def regret_curve(gp_vals, pfn_vals, optimal_val):

    best_so_far_gp = np.maximum.accumulate(gp_vals)
    best_so_far_pfn = np.maximum.accumulate(pfn_vals)

    regret_gp = optimal_val - best_so_far_gp
    regret_pfn = optimal_val - best_so_far_pfn
    
    # 4. Plotting
    plt.step(range(len(regret_pfn)), regret_gp, where='post', label='gp')
    plt.step(range(len(regret_pfn)), regret_pfn, where='post', label='pfn')
    plt.xlabel('Iteration')
    plt.ylabel('Simple Regret')
    plt.title('Optimization Regret Curve')
    plt.legend()
    plt.show()
