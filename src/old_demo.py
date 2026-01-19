import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import gzip
import time
import scipy.io
import seaborn as sns
import pickle
import pandas as pd
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod


# ============================================
#           Data Loader and Preprocessing
# ============================================

def load_data2(dataset_type, m_i):
    '''
    Input: 
        - dataset_type: str characterizing the modality of the experiment
        - m_i: int of subject

    Output:
        - dictionary of neurostimulation data
    
        Important: sorted response shape: (nChan, nEmgs, nReps)
    '''
    path_to_dataset = f'./data/monkeys'
    if dataset_type=='nhp':
        if m_i==0:
            data = scipy.io.loadmat(path_to_dataset+'/Cebus1_M1_190221.mat')['Cebus1_M1_190221'][0][0]
        elif m_i==1:
            data = scipy.io.loadmat(path_to_dataset+'/Cebus2_M1_200123.mat')['Cebus2_M1_200123'][0][0]
        elif m_i==2:    
            data = scipy.io.loadmat(path_to_dataset+'/Macaque1_M1_181212.mat')['Macaque1_M1_181212'][0][0]
        elif m_i==3:
            data  = scipy.io.loadmat(path_to_dataset+'/Macaque2_M1_190527.mat')['Macaque2_M1_190527'][0][0]

        if m_i >= 2:
            #macaques
            mapping = {
                'emgs': 0, 'emgsabr': 1, 'nChan': 2, 'stimProfile': 3, 'stim_channel': 4, 
                'evoked_emg': 5, 'response': 6, 'isvalid': 7, 'sorted_isvalid': 8, 'sorted_resp': 9, 
                'sorted_evoked': 10, 'sampFreqEMG': 11, 'resp_region': 12, 'map': 13, 'ch2xy': 14, 
                'sorted_respMean': 15, 'sorted_respSD': 16
            }
        else:
            # cebus
            mapping = {
                'emgs': 0, 'emgsabr': 1, 'nChan': 2, 'stimProfile': 3, 'stim_channel': 4, 
                'evoked_emg': 5, 'response': 6, 'isvalid': 7, 'sorted_isvalid': 8, 'sorted_resp': 9, 
                'sorted_respMean': 10, 'sorted_respSD': 11, 'sorted_evoked': 12, 'sampFreqEMG': 13, 
                'resp_region': 14, 'map': 15, 'ch2xy': 16
            }

        nChan = data[mapping['nChan']][0][0]

        rN = data[mapping['sorted_isvalid']]
        j1, j2, j3 = rN.shape[0], rN.shape[1], rN[0][0].shape[0]
        sorted_isvalid = np.stack([np.squeeze(rN[i, j]) for i in range(j1) for j in range(j2)], axis=0)
        sorted_isvalid = sorted_isvalid.reshape(j1, j2, j3)

        ch2xy = data[mapping['ch2xy']] - 1
        se = data[mapping['sorted_evoked']]
        i1, i2, i3, i4 = se.shape[0], se.shape[1], se[0][0].shape[0], se[0][0].shape[1]
        sorted_evoked = np.stack([np.squeeze(se[i, j]) for i in range(i1) for j in range(i2)], axis=0)
        sorted_evoked = sorted_evoked.reshape(i1, i2, i3, i4)
        sorted_filtered = sorted_evoked

        stim_channel = data[mapping['stim_channel']]
        if stim_channel.shape[0] == 1:
            stim_channel = stim_channel[0]

        fs = data[mapping['sampFreqEMG']][0][0]
        resp_region = data[mapping['resp_region']][0]

        stimProfile = data[mapping['stimProfile']][0]
        
        # compute baseline
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 30 * 10**(-3))
        baseline = []
        for iChan in range(nChan):
            reps = np.where(stim_channel == iChan + 1)[0]
            n_rep = len(reps)
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, where_zero - window_size : where_zero], axis=-1)
            baseline.append(mean_baseline)
        
        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)
        
        sorted_filtered = sorted_filtered - baseline[..., np.newaxis]
        sorted_resp = np.max(sorted_filtered[:,:,:n_rep,resp_region[0]:resp_region[1]], axis=-1)

        # Create a masked array where invalid points are masked
        masked_resp = np.ma.masked_where(sorted_isvalid == 0, sorted_resp)
        
        # Compute the mean over the last axis, ignoring masked (invalid) values
        sorted_respMean = masked_resp.mean(axis=-1)

        emgs = data[0][0]

        return {
        'emgs': emgs,
        'nChan': nChan, 
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'ch2xy': ch2xy,
        'DimSearchSpace': 96
        }
    elif dataset_type=='rat':  # rat dataset has 6 subjects
        if m_i==0:
            data = scipy.io.loadmat(path_to_dataset+'/rat1_M1_190716.mat')['rat1_M1_190716'][0][0]
        elif m_i==1:
            data = scipy.io.loadmat(path_to_dataset+'/rat2_M1_190617.mat')['rat2_M1_190617'][0][0]     
        elif m_i==2:
            data = scipy.io.loadmat(path_to_dataset+'/rat3_M1_190728.mat')['rat3_M1_190728'][0][0]                  
        elif m_i==3:
            data = scipy.io.loadmat(path_to_dataset+'/rat4_M1_191109.mat')['rat4_M1_191109'][0][0]                  
        elif m_i==4:
            data = scipy.io.loadmat(path_to_dataset+'/rat5_M1_191112.mat')['rat5_M1_191112'][0][0]                  
        elif m_i==5:
            data = scipy.io.loadmat(path_to_dataset+'/rat6_M1_200218.mat')['rat6_M1_200218'][0][0]   

        mapping = {
                'emgs': 0, 'emgsabr': 1, 'nChan': 2, 'stimProfile': 3, 'stim_channel': 4, 
                'evoked_emg': 5, 'response': 6, 'isvalid': 7, 'sorted_isvalid': 8, 'sorted_resp': 9, 
                'sorted_evoked': 10, 'sampFreqEMG': 11, 'resp_region': 12, 'map': 13, 'ch2xy': 14, 
                'sorted_respMean': 15, 'sorted_respSD': 16
            }
        
        nChan = data[mapping['nChan']][0][0]

        rN = data[mapping['sorted_isvalid']]
        j1, j2, j3 = rN.shape[0], rN.shape[1], rN[0][0].shape[0]
        sorted_isvalid = np.stack([np.squeeze(rN[i, j]) for i in range(j1) for j in range(j2)], axis=0)
        sorted_isvalid = sorted_isvalid.reshape(j1, j2, j3)

        ch2xy = data[mapping['ch2xy']] - 1
        se = data[mapping['sorted_evoked']]
        i1, i2, i3, i4 = se.shape[0], se.shape[1], se[0][0].shape[0], se[0][0].shape[1]
        sorted_evoked = np.stack([np.squeeze(se[i, j]) for i in range(i1) for j in range(i2)], axis=0)
        sorted_evoked = sorted_evoked.reshape(i1, i2, i3, i4)
        sorted_filtered = sorted_evoked

        stim_channel = data[mapping['stim_channel']]
        if stim_channel.shape[0] == 1:
            stim_channel = stim_channel[0]

        fs = data[mapping['sampFreqEMG']][0][0]
        resp_region = data[mapping['resp_region']][0]

        stimProfile = data[mapping['stimProfile']][0]
        
        # compute baseline
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 30 * 10**(-3))
        baseline = []
        for iChan in range(nChan):
            reps = np.where(stim_channel == iChan + 1)[0]
            n_rep = len(reps)
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, where_zero - window_size : where_zero], axis=-1)
            baseline.append(mean_baseline)
        
        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)
        
        sorted_filtered = sorted_filtered - baseline[..., np.newaxis]
        sorted_resp = np.max(sorted_filtered[:,:,:n_rep,resp_region[0]:resp_region[1]], axis=-1)
        # Create a masked array where invalid points are masked
        masked_resp = np.ma.masked_where(sorted_isvalid == 0, sorted_resp)
        
        # Compute the mean over the last axis, ignoring masked (invalid) values
        sorted_respMean = masked_resp.mean(axis=-1)

        emgs = data[0][0]

        return {
        'emgs': emgs,
        'nChan': nChan, 
        'sorted_isvalid': sorted_isvalid,
        'sorted_resp': sorted_resp,
        'sorted_respMean': sorted_respMean,
        'ch2xy': ch2xy,
        'DimSearchSpace': 32
        } 
    elif dataset_type =='spinal':

        subject_map = {
            0: 'rat0_C5_500uA.pkl', 1: 'rat1_C5_500uA.pkl', 2: 'rat1_C5_700uA.pkl', 3: 'rat1_midC4_500uA.pkl',
            4: 'rat2_C4_300uA.pkl', 5: 'rat2_C5_300uA.pkl', 6: 'rat2_C6_300uA.pkl', 7: 'rat3_C4_300uA.pkl',
            8: 'rat3_C5_200uA.pkl', 9: 'rat3_C5_350uA.pkl', 10: 'rat3_C6_300uA.pkl' 
        }
        
        #load data
        with open(f'{path_to_dataset}/{subject_map[m_i]}', "rb") as f:
            data = pickle.load(f)
        
        ch2xy, emgs = data['ch2xy'], data['emgs']
        evoked_emg, filtered_emg = data['evoked_emg'], data['filtered_emg']
        maps = data['map']
        parameters = data['parameters']
        resp_region = data['resp_region']
        fs = data['sampFreqEMG']
        sorted_evoked = data['sorted_evoked']
        sorted_filtered = data['sorted_filtered']
        sorted_resp = data['sorted_resp']
        sorted_isvalid = data['sorted_isvalid']
        sorted_respMean = data['sorted_respMean']
        sorted_respSD = data['sorted_respSD']
        stim_channel = data['stim_channel']
        stimProfile=data['stimProfile']
        n_muscles = emgs.shape[0]

        #?# We are removing lots of reps here print(f'sorted response: {sorted_resp.shape}') 
        #Computing baseline for filtered signal
        nChan = parameters['nChan'][0]
        where_zero = np.where(abs(stimProfile) > 10**(-50))[0][0]
        window_size = int(fs * 35 * 10**(-3))
        baseline = []
        n_rep = 10000 # First, determine n_reps global
        for iChan in range(nChan):
            reps= np.where(stim_channel == iChan + 1)[0]
            if len(reps) < n_rep:
                n_rep = len(reps)
        for iChan in range(nChan):
            mean_baseline = np.mean(sorted_filtered[iChan, :, :n_rep, 0 : where_zero], axis=-1)
            baseline.append(mean_baseline)
        
        baseline = np.stack(baseline, axis=0)

        #remove baseline from filtered signal
        sorted_filtered[:, :, :n_rep, :] = sorted_filtered[:, :, :n_rep, :] - baseline[..., np.newaxis]
        sorted_resp = np.nanmax(sorted_filtered[:, :, :n_rep, int(resp_region[0]): int(resp_region[1])], axis=-1)
        masked_resp = np.ma.masked_where(sorted_isvalid[:, :, :n_rep] == 0, sorted_resp)
        sorted_respMean = masked_resp.mean(axis=-1)

         # compute baseline for evoked signal
        baseline = []
        for iChan in range(nChan):
            # Compute mean over the last dimension (time), across those repetitions
            mean_baseline = np.mean(sorted_evoked[iChan, :, :n_rep, 0 : where_zero], axis=-1)
            baseline.append(mean_baseline)
        baseline = np.stack(baseline, axis=0)  # shape: (nChan, nSamples)
        
        #remove baseline from evoked signal
        sorted_evoked[:, :, :n_rep, :] = sorted_evoked[:, :, :n_rep, :] - baseline[..., np.newaxis]
        sorted_resp = np.nanmax(sorted_evoked[:,:,:n_rep,int(resp_region[0]) :int(resp_region[1])], axis=-1)
        masked_resp = np.ma.masked_where(sorted_isvalid[:,:,:n_rep] == 0, sorted_resp)

        #mask sorted_isvalid by n_rep
        sorted_isvalid = sorted_isvalid[:, :, :n_rep]

        subject = {
            'emgs': emgs,
            'nChan': 64,
            'DimSearchSpace': 64,
            'sorted_respMean': sorted_respMean,
            'ch2xy': ch2xy,
            'evoked_emg': evoked_emg, 'filtered_emg':filtered_emg, 'sorted_resp': sorted_resp,  
            'sorted_isvalid': sorted_isvalid, 'sorted_respSD': sorted_respSD,
            'sorted_filtered': sorted_filtered, 'stim_channel': stim_channel, 'fs': fs,
        'parameters': parameters, 'n_muscles': n_muscles, 'maps': maps,
        'resp_region': resp_region, 'stimProfile': stimProfile,  'baseline' : baseline    
        }
        
        return subject   
    else:
        raise ValueError('The dataset type should be 5d_rat, nhp, rat or spinal' )
 
def preprocess_neural_data(subject_data, emg_idx=0):
    """
    Ideally, you would preprocess neural data so that Y_test is the resp_mean,
    but we train (X_train) on sorted_resp.
    """

    coords = subject_data['ch2xy'] # Shape (96, 2)
    resp_all = subject_data['sorted_resp'][:, emg_idx, :] # Shape (96, repetitions)
    resp_mean = subject_data['sorted_respMean'][:, emg_idx] # Shape (96,)

    scaler_x = MinMaxScaler()
    X_unique_scaled = scaler_x.fit_transform(coords)
    n_reps = resp_all.shape[1]
    X_train =  np.repeat(X_unique_scaled, n_reps, axis=0)

    scaler_y = StandardScaler()
    y_train_flat = resp_all.flatten()
    Y_train_scaled = scaler_y.fit_transform(y_train_flat.reshape(-1, 1))
    Y_train = Y_train_scaled.flatten()
    
    Y_test_scaled = scaler_y.transform(resp_mean.reshape(-1, 1))
    Y_test = Y_test_scaled.flatten()

    X_test = X_unique_scaled

    return X_train, Y_train, X_test, Y_test

# ============================================
#           Model evaluation
# ============================================

def evaluate_models(dataset, subject_idx, emg_idx,

                   device='cpu', n_iters=200):
    data = load_data2(dataset, subject_idx)
    X_train, y_train, X_test, y_test = preprocess_neural_data(data, emg_idx)

    # subsample context
    indices = np.random.choice(len(y_train), n_iters, replace=False)
    X_train=X_train[indices]
    y_train=y_train[indices]

    # --- 1. Gaussian Process ---
    start_gp = time.time()
    kernel = 1.0 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
    gp.fit(X_train, y_train)
    y_pred_gp = gp.predict(X_test)
    time_gp = time.time() - start_gp
    r2_gp = r2_score(y_test, y_pred_gp)
    


    # --- 2. PFNs4BO Model ---

    model_path = './libs/PFNS4BO/pfns4bo/final_models/model_hebo_morebudget_9_unused_features_3.pt.gz'
    with gzip.open(model_path, 'rb') as f:
        model_weights = torch.load(f, map_location=device)

    pfn_bo = TransformerBOMethod(
        model_weights,
        device=device,
        acq_function='mean' # CRITICAL: This tells the PFN to output posterior mean
        )

    start_pfn = time.time()

    _, y_pred_pfn = pfn_bo.observe_and_suggest(
        X_obs=X_train,
        y_obs=y_train,
        X_pen=X_test,
        return_actual_ei=True
    )

    if isinstance(y_pred_pfn, torch.Tensor): y_pred_pfn = y_pred_pfn.numpy()

    time_pfn = time.time() - start_pfn
    r2_pfn = r2_score(y_test, y_pred_pfn)

    return {
        'r2_gp': r2_gp, 'time_gp': time_gp,
        'r2_pfn': r2_pfn, 'time_pfn': time_pfn,
        'y_test': y_test, 'y_pred_gp': y_pred_gp, 'y_pred_pfn': y_pred_pfn
    }




# ============================================
#           Visualization
# ============================================

def r2_comparison(results):

    df = pd.DataFrame(results)

    # --- Plot 1: R^2 Comparison (Bar Chart) ---
    plt.figure(figsize=(12, 6))
    df_melt = df.melt(id_vars=['dataset', 'subject', 'emg'],
                    value_vars=['r2_gp', 'r2_pfn'],
                    var_name='Model', value_name='R2')  

    sns.barplot(data=df_melt, x='emg', y='R2', hue='Model', ci=None)
    plt.title("R^2 Comparison: GP vs PFN across Datasets")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def runtime_comparison(results):

    df = pd.DataFrame(results)
    plt.figure(figsize=(6, 6))
    avg_times = df[['time_gp', 'time_pfn']].mean()
    avg_times.plot(kind='bar', color=['red', 'blue'], alpha=0.7)
    plt.title("Average Inference Time (s)")
    plt.ylabel("Seconds")
    plt.show()

def show_emg_map(results, idx):
    """
    Visualizes the True Mean, GP Prediction, and PFN Prediction as 2D Heatmaps.
    Automatically handles 96-channel (8x12) and 32-channel (4x8) grids.
    """
    res = results[idx]


    # Extract data
    y_true = res['y_test']
    y_gp = res['y_pred_gp']
    y_pfn = res['y_pred_pfn']

    # --- 1. Determine Grid Shape ---
    # NHP data is usually 96 channels (8x12)
    # Rat data is usually 32 channels (4x8)
    n_channels = len(y_true)

    if n_channels == 96:
        grid_shape = (8, 12)
    elif n_channels == 32:
        grid_shape = (4, 8)
    else:
        # Fallback: Try to make a square-ish grid
        side = int(np.sqrt(n_channels))
        if n_channels % side == 0:
            grid_shape = (side, n_channels // side)
        else:
            print(f"Warning: Cannot reshape {n_channels} into a grid. Plotting as 1D strip.")
            grid_shape = (1, n_channels)

    # --- 2. Reshape Data ---
    # Note: 'order' depends on how your channels are numbered.
    # 'C' (Row-major) is standard if ch1 is top-left, ch12 is top-right.
    # 'F' (Column-major) if ch1 is top-left, ch2 is below it.
    map_true = y_true.reshape(grid_shape)
    map_gp   = y_gp.reshape(grid_shape)
    map_pfn  = y_pfn.reshape(grid_shape)

   
    # --- 3. Setup Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
   
    # Determine global min/max for SHARED colorbar (Critical for comparison)
    vmin = min(y_true.min(), y_gp.min(), y_pfn.min())
    vmax = max(y_true.max(), y_gp.max(), y_pfn.max())

    # Helper to clean up heatmap style
    heatmap_args = {
        'vmin': vmin,
        'vmax': vmax,
        'cmap': 'viridis',
        'square': True,
        'cbar': True,
        'annot': False # Set True if you want numbers in boxes
    }

    # --- 4. Plot Heatmaps ---

    # Plot A: Ground Truth
    sns.heatmap(map_true, ax=axes[0], **heatmap_args)
    axes[0].set_title("Ground Truth (Mean Response)")
    axes[0].set_ylabel("Grid Rows")
    axes[0].set_xlabel("Grid Cols")

    # Plot B: GP Prediction
    sns.heatmap(map_gp, ax=axes[1], **heatmap_args)
    axes[1].set_title(f"GP Prediction\n(R² = {res['r2_gp']:.2f})")
    axes[1].set_yticks([]) # Hide Y axis for cleanliness

    # Plot C: PFN Prediction
    sns.heatmap(map_pfn, ax=axes[2], **heatmap_args)
    axes[2].set_title(f"PFN Prediction\n(R² = {res['r2_pfn']:.2f})")
    axes[2].set_yticks([])

    # Header
    plt.suptitle(f"Model Comparison: {res['dataset']} | Subj {res['subject']} | {res['emg']}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()



# ============================================
#           Runner
# ============================================

def main(datasets):


    results = []

    for experiment in datasets:

        dataset_type=experiment[0]
        subject_idx = experiment[1]
        emg_idx = experiment[2]

        res = evaluate_models(dataset_type, subject_idx, emg_idx)

        results.append({
                'dataset': dataset_type, 'subject': subject_idx, 'emg': emg_idx,
                **res
            })

    return results




if __name__ == '__main__':

    datasets = [
    ('nhp', 0, 0), #('nhp', 0, 1),  ('nhp', 0, 2), ('nhp', 0, 3)
            ]

    results = main(datasets)
    r2_comparison(results)
    exit(0)
    runtime_comparison(results)