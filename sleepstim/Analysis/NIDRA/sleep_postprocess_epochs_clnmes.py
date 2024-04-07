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
import glob
import os
from tqdm import tqdm
from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.clustering import Potato
# from autoreject import AutoReject
mne.set_log_level('ERROR')

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
    epochs.filter(0.5, 30)
    
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
    
    # 5. Autoreject for epoch wise interpolation - DNU
    # ar = AutoReject(picks='eeg', n_jobs=-1)
    # epochs = ar.fit_transform(epochs) 
    
    # 6. SO filter data - DNU
    # epochs.filter(None, 2)
    
    # 7. Apply baseline correction
    epochs.apply_baseline(baseline=(-3, -1.5))
    
    # 8. Appy surface Laplacian interpretation
    epochs_csd = mne.preprocessing.compute_current_source_density(epochs) 
   
    # 9. Save 
    if save:
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
        
def combine_processed_epochs(file_group, save_directory, save=True):
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
    final_filename = f'{subject}_{night}_csd-epo.fif'
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
        postprocess_sleep_nmes_epochs(file, save_path, fig_path, save=True)
    
    new_path = '/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/*_csd-epo.fif'
    save_path = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/'
    file_groups = group_files_by_subject_night(glob.glob(new_path))
    for subject_night, file_group in tqdm(file_groups.items()):
        print(f"Combining recordings, if necessary: {subject_night}")
        combine_processed_epochs(file_group, save_path, save=True)
