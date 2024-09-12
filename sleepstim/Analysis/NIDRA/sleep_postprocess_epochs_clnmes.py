#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:08:02 2024

@author: administrator
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
import scipy
import glob
import os
from tqdm import tqdm
from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.clustering import Potato
# from autoreject import AutoReject
mne.set_log_level('ERROR')

def interpolate_cubic(data, times, win):
    from scipy import interpolate
    """
    Performs cubic interpolation on a 2D time series data.

    Parameters:
    - data: 2D numpy array representing the time series (channels x time points).
    - times: 1D numpy array representing the time points.
    - win: Tuple (start_time, end_time) representing the window for interpolation in seconds.

    Returns:
    - Interpolated 2D time series data.
    """
    # Convert window to indices; +1 because it should include the last timepoint
    idx1 = np.argmin(np.abs(times - win[0]))
    idx2 = np.argmin(np.abs(times - win[1])) + 1

    # Delete timepoints that should be interpolated from the times array
    x = np.delete(times, np.s_[idx1:idx2], 0)

    # Initialize the array for interpolated data
    interpolated_data = np.copy(data)

    # Perform interpolation for each channel
    for channel in range(data.shape[0]):
        # Delete timepoints that should be interpolated from the channel data
        y = np.delete(data[channel], np.s_[idx1:idx2])

        # Fit the cubic interpolation
        p = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")

        # Get the interpolation values for the timepoints of interest
        interp_values = p(times[idx1:idx2])

        # Replace the corresponding timepoints in the data with interpolated values
        interpolated_data[channel, idx1:idx2] = interp_values

    return interpolated_data

def clean_ica(epochs):
    # PICARD - FastICA solution
    ica = mne.preprocessing.ICA(n_components=32, method='picard', random_state=42,
                                fit_params = dict(ortho=True, extended=True)) 
    ica.fit(epochs.copy())
    
    # Automated component detection assistance 
    from mne_icalabel import label_components
    ic_labels = label_components(epochs, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    print(f"Would exclude these ICA components: {exclude_idx}")
    
    try:
        ica.plot_overlay(epochs[['nmes_stim_c3','nmes_stim_fz']].average(), 
                         exclude=exclude_idx, picks="eeg")
        ica.plot_overlay(epochs[['nmes_sham_c3','nmes_sham_fz']].average(), 
                         exclude=exclude_idx, picks="eeg")
            
        EDC = epochs[['nmes_stim_c3','nmes_stim_fz']].get_data('EDC', units='uV').squeeze()
        sources = ica.get_sources(epochs[['nmes_stim_c3','nmes_stim_fz']]).get_data()
        
        sfreq = epochs.info['sfreq']  # Sampling frequency
        start_sample = int((-0.05 - epochs.tmin) * sfreq)
        end_sample = int((0.15 - epochs.tmin) * sfreq)
       
        r_values = []
        for idx1, source in enumerate(sources):
            for idx2, comp in enumerate(source):
                r = scipy.stats.pearsonr(EDC[idx1,start_sample:end_sample], 
                                         comp[start_sample:end_sample])[0]
                r_values.append([idx1, idx2, r])
        
        df = pd.DataFrame(r_values)
        max_corr = np.abs(df.groupby([1]).mean()[2]).argmax()
        corr = np.abs(df.groupby([1]).mean()[2]).max()
        
        print(f'Component : {max_corr}')
        print(f'Max component and EDC correlation: {np.round(corr, 2)}')
        
        # ica.plot_sources(epochs[['nmes_stim_c3','nmes_stim_fz']], 
        #                  show_scrollbars=False)
        ica.plot_overlay(epochs[['nmes_stim_c3','nmes_stim_fz']].average(), 
                         exclude=[max_corr], picks="eeg")
        
        plt.close('all')
        
        # ica.exclude = [max_corr]
        if max_corr >= .5:
            ica.exclude = exclude_idx + [max_corr]
            ica.plot_overlay(epochs[['nmes_stim_c3','nmes_stim_fz']].average(), 
                             exclude=ica.exclude, picks="eeg")
        else:
            ica.exclude = exclude_idx
    
    except:
        
        ica.plot_overlay(epochs[['pn_stim_c3','pn_stim_fz']].average(), 
                         exclude=exclude_idx, picks="eeg")
        ica.plot_overlay(epochs[['pn_sham_c3','pn_sham_fz']].average(), 
                         exclude=exclude_idx, picks="eeg")
                
        # ica.plot_sources(epochs[['pn_stim_c3','pn_stim_fz']], 
        #                  show_scrollbars=False)
               
        ica.exclude = exclude_idx
        
    
    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    clean_epochs_ = epochs.copy()
    
    # # Apply ICA if corr strong enough
    # if corr >= .5:
    #     clean_epochs = ica.apply(clean_epochs_['nmes_stim_c3','nmes_stim_fz'])
    #     # Define the event IDs you want to replace
    #     event_names_to_replace = ['nmes_stim_c3', 'nmes_stim_fz']
    
    #     # Map event names to their IDs using the event_id dictionary in epochs
    #     event_ids_to_replace = [epochs.event_id[name] for name in event_names_to_replace if name in epochs.event_id]
    
    #     # Get all indices where the event id is one of those specified
    #     indices_to_replace = [i for i, event in enumerate(epochs.events[:, 2]) if event in event_ids_to_replace]
    
    #     # Check if clean_epochs has the correct number of epochs and data structure
    #     if len(indices_to_replace) == len(clean_epochs) and epochs.get_data().shape[1:] == clean_epochs.get_data().shape[1:]:
    #         # Update datapoints of interest
    #         data_update = clean_epochs.get_data()
    #         data_replace = epochs._data[indices_to_replace]
    #         data_replace[:,:, start_sample:end_sample] = data_update[:,:,start_sample:end_sample]
    #         # Replace data for the selected indices
    #         #clean_epochs_._data[indices_to_replace] = clean_epochs.get_data()
    #         clean_epochs_._data[indices_to_replace] = data_replace
    #     else:
    #         print("Mismatch in number of epochs or data dimensions.")
    
    # else:
    #     print('Nothing changed....')

    # return clean_epochs_
    
    # Apply ICA 
    clean_epochs = ica.apply(clean_epochs_)

    return clean_epochs
    
def identify_epoch_outliers(epoch):
    # Check for characteristics indicative of CSD data
    if 'csd' in epoch.get_channel_types():
        data = epoch.get_data(picks='csd') * 1e3
    
    # Check for characteristics indicative of EEG data
    elif 'eeg' in epoch.get_channel_types():
        data = epoch.get_data(picks='eeg', units='uV')

    # Calculate the covariance matrices (n_epochs, n_chan, n_chan)
    covmats = Covariances().fit_transform(data)
    
    # Shrink the covariance matrix (ensure positive semi-definite)
    covmats = Shrinkage().fit_transform(covmats)
    
    # Define Potato instance: 0 = clean, 1 = art
    # To increase speed we set the max number of iterations from 10 to 100
    potato = Potato(metric='riemann', threshold=2, pos_label=0,
                    neg_label=1, n_iter_max=100)
    
    # Apply Potato algorithm, extract z-scores and labels
    zs = potato.fit_transform(covmats)
    art = potato.predict(covmats).astype(int)
    
    # percent of epochs rejected
    perc_reject = 100 * (art.sum() / art.size)
    
    # Print total rejected epochs 
    print(f"Total: {art.sum()} / {art.size} "
          f"epochs rejected ({perc_reject:.2f}%)")
    
    # Convert epoch_is_art to boolean [0, 0, 1] -- > [False, False, True]
    epoch_is_art = art.astype(bool)
    
    return epoch_is_art, zs
     
def postprocess_sleep_nmes_epochs(file, save_path, fig_path, save=True):
    # 0. Get subject, night, rec info
    subject, night, rec = file.split('/')[-1].split('-')[0].split('_')
   
    # 1. Load data & filter more narrowly
    epochs = mne.read_epochs(file)
    epochs.filter(0.5, None) #30
    
    # # 1b. Get stim condition
    # mode = list(epochs.event_id)[0].split('_')[0]
    
    # 2. Identify bads once again
    bads, _ = nk.eeg_badchannels(epochs.get_data(picks='eeg', units='uV').mean(0))
    try:
        epochs.info['bads'] = list(np.asarray(epochs.ch_names)[bads])
    except:
        epochs.info['bads'] = []
    
    # 3. Interpolate bad channels, if applicable
    epochs.interpolate_bads()
              
    # 4. Reject epochs with non-plausible voltages
    reject_criteria = dict(eeg=550e-6)
    epochs.drop_bad(reject = reject_criteria) 
    
    # 5a. Apply ICA to remove stim artifact in NMES if applicable
    try:
        clean_epochs = clean_ica(epochs)
        #plt.savefig(fig_path + f'nmes_sham_c3_{subject}_{night}_{rec}_ica.png')
        plt.close('all')
        if subject == 'LiJo' and night == '1':
            clean_epochs.info['bads'] = ['AF4','P5','CP1','P8']
            clean_epochs.interpolate_bads()
        elif subject == 'DoGi' and night == '1':
            clean_epochs.info['bads'] = ['P4']
            clean_epochs.interpolate_bads()
        elif subject == 'SvSchm' and night == '2':
            clean_epochs.info['bads'] = ['C1','Cz']
            clean_epochs.interpolate_bads()
        elif subject == 'KeWo' and night == '1':
            clean_epochs.info['bads'] = ['C2','AF8']
            clean_epochs.interpolate_bads()
        elif subject == 'KaBr' and night == '1':
            clean_epochs.info['bads'] = ['PO8','FC3']
            clean_epochs.interpolate_bads()
        elif subject == 'SvSchm' and night == '1':
            clean_epochs.info['bads'] = ['T8']
            clean_epochs.interpolate_bads()
        elif subject == 'UyDe' and night == '2':
            clean_epochs.info['bads'] = ['FC2','FC3','C1','FC1','O2']
            clean_epochs.interpolate_bads()
    except:
        clean_epochs = epochs.copy()
        
    # # 5aa. Bad channel detection with local outlier factor
    # from sklearn.neighbors import LocalOutlierFactor
    # ch_names = clean_epochs.ch_names[0:64]
    # data = clean_epochs.average().get_data(picks='eeg')
    # clf = LocalOutlierFactor(n_neighbors=20, metric='euclidean')
    # clf.fit_predict(data)
    # scores_lof = clf.negative_outlier_factor_
    # bad_channel_indices = [
    #     i for i, v in enumerate(np.abs(scores_lof)) if v >= 1.5
    # ]
    # bads = [ch_names[idx] for idx in bad_channel_indices]
    # clean_epochs.info['bads'] = bads
    # clean_epochs.interpolate_bads()
    
    # 5b. Cubic interpolation of stim artifact
    # if 'nmes_stim_c3' in list(epochs.event_id):
    #     epochs.apply_function(fun=interpolate_cubic, 
    #                           picks='eeg', 
    #                           n_jobs=-1,
    #                           channel_wise=True,
    #                           **dict(times = epochs.times, 
    #                                win = [-.02, .15]))
        
    # 5c. Autoreject for epoch wise interpolation - DNU
    # ar = AutoReject(picks='eeg', n_jobs=-1)
    # clean_epochs = ar.fit_transform(clean_epochs) 
               
    # 6. Apply baseline correction
    clean_epochs.apply_baseline(baseline=(-3, -1.5))
    
    # 7. Appy surface Laplacian interpretation
    epochs_csd = mne.preprocessing.compute_current_source_density(clean_epochs) 
   
    # 8. Save 
    if save:
        clean_epochs.save(save_path + f'{subject}_{night}_{rec}_lm-epo.fif', overwrite=True)
        epochs_csd.save(save_path + f'{subject}_{night}_{rec}_csd-epo.fif', overwrite=True)
        
        # 10a. SO filter data for plotting
        epochs_csd.filter(None, 2)
        
        # 10b. Plot data 
        ts_args = dict(spatial_colors=True, gfp=True, time_unit='s')
        topomap_args = dict(outlines='head', time_unit='s', time_format="%0.2f s")
        times = [-0.5, 0, 0.5, 1]  
        try:
            epochs_csd['nmes_sham_c3'].average().plot_joint(title=f'nmes_sham_c3_{subject}', 
                                                            times=times, ts_args=ts_args,
                                                            topomap_args = topomap_args)
            plt.savefig(fig_path + f'nmes_sham_c3_{subject}_{night}_{rec}.png')
            epochs_csd['nmes_stim_c3'].average().plot_joint(title=f'nmes_stim_c3_{subject}', 
                                                            times=times, ts_args=ts_args,
                                                            topomap_args = topomap_args)
            plt.savefig(fig_path + f'nmes_stim_c3_{subject}_{night}_{rec}.png')
            epochs_csd['nmes_stim_fz'].average().plot_joint(title=f'nmes_stim_fz_{subject}',
                                                            times=times, ts_args=ts_args,
                                                            topomap_args = topomap_args)
            plt.savefig(fig_path + f'nmes_stim_fz_{subject}_{night}_{rec}.png')
            epochs_csd['nmes_sham_fz'].average().plot_joint(title=f'nmes_sham_fz_{subject}',
                                                            times=times, ts_args=ts_args,
                                                            topomap_args = topomap_args)
            plt.savefig(fig_path + f'nmes_sham_fz_{subject}_{night}_{rec}.png')
        except:
            epochs_csd['pn_sham_c3'].average().plot_joint(title=f'pn_sham_c3_{subject}', 
                                                            times=times, ts_args=ts_args,
                                                            topomap_args = topomap_args)
            plt.savefig(fig_path + f'pn_sham_c3_{subject}_{night}_{rec}.png')
            epochs_csd['pn_stim_c3'].average().plot_joint(title=f'pn_stim_c3_{subject}', 
                                                            times=times, ts_args=ts_args,
                                                            topomap_args = topomap_args)
            plt.savefig(fig_path + f'pn_stim_c3_{subject}_{night}_{rec}.png')
            epochs_csd['pn_stim_fz'].average().plot_joint(title=f'pn_stim_fz_{subject}',
                                                            times=times, ts_args=ts_args,
                                                            topomap_args = topomap_args)
            plt.savefig(fig_path + f'pn_stim_fz_{subject}_{night}_{rec}.png')
            epochs_csd['pn_sham_fz'].average().plot_joint(title=f'pn_sham_fz_{subject}',
                                                            times=times, ts_args=ts_args,
                                                            topomap_args = topomap_args)
            plt.savefig(fig_path + f'pn_sham_fz_{subject}_{night}_{rec}.png')
        plt.close('all')
 
def group_files_by_subject_night(files):
    """
    Group files by subject and night.
    """
    file_groups = {}
    for file in files:
        parts = file.split('/')[-1].split('-')[0].split('_')
        subject_night = '_'.join(parts[:2])  # Assuming first two parts are subject and night
        if subject_night not in file_groups:
            file_groups[subject_night] = []
        file_groups[subject_night].append(file)
    return file_groups
        
def combine_processed_epochs(file_group, save_directory, ref='csd', save=True):
    """
    Load processed epochs for a group of files belonging to the same subject and night, and combine them.
    """
    processed_epochs_list = []
    for file in file_group:
        print(f"Loading processed file: {file}")
        epochs = mne.read_epochs(file, preload=True)
        processed_epochs_list.append(epochs)

    # Combine the processed epochs
    subject, night = file_group[0].split('/')[-1].split('_')[:2]
    if ref == 'csd':
        final_filename = f'{subject}_{night}_csd-epo.fif'
    elif ref == 'lm':
        final_filename = f'{subject}_{night}_lm-epo.fif'
    final_save_path = os.path.join(save_directory, final_filename)

    if save:   
        if len(processed_epochs_list) > 1:
            combined_epochs = mne.concatenate_epochs(processed_epochs_list)
            print(f"Saving combined processed epochs to: {final_save_path}")
            combined_epochs.save(final_save_path, overwrite=True)
            return combined_epochs
        else:
            print(f"Renaming processed epochs to: {final_save_path}")
            epochs.save(final_save_path, overwrite=True)
            return epochs
       
#%%
if __name__ == '__main__':
    path = '/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/*-epo.fif' 
    save_path = '/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/'
    fig_path = '/media/administrator/Sleep_Data/Processed/Figures/Sleep/'
    for file in tqdm(glob.glob(path)):
        if 'csd' not in file and 'lm' not in file:
            print(file)
            # epochs = mne.read_epochs(file, preload=False)
            # print(list(epochs.event_id)[0])
            postprocess_sleep_nmes_epochs(file, save_path, fig_path, save=True)
   
    new_path = '/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/*_csd-epo.fif'
    save_path = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/'
    file_groups = group_files_by_subject_night(glob.glob(new_path))
    for subject_night, file_group in tqdm(file_groups.items()):
        print(f"Combining recordings, if necessary: {subject_night}")
        combine_processed_epochs(file_group, save_path, ref='csd', save=True)
        
    new_path = '/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/*_lm-epo.fif'
    save_path = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/'
    file_groups = group_files_by_subject_night(glob.glob(new_path))
    for subject_night, file_group in tqdm(file_groups.items()):
        print(f"Combining recordings, if necessary: {subject_night}")
        combine_processed_epochs(file_group, save_path, ref='lm', save=True)
