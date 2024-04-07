#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:32:40 2023

@author: administrator
"""

import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fooof
from tqdm import tqdm
import yasa
import neurokit2 as nk
from glob import glob
mne.set_log_level('ERROR')

def analyze_rs_data(file, figure_path, plot=False): 
    # Get subject, night info
    subject, night, session, _, _ = file.split('/')[-1].split('-')[0].split('_')
     
    # Read epochs
    epochs = mne.read_epochs(file, preload=False)
    
    # Get stimulation condition from epoch events
    try:  
        events = mne.read_events(f'/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/{subject}_{night}_csd-epo.fif', 
                                 return_event_id=True)
        if 'pn_sham_c3' in events[-1]:
            mode = 'pn'
        elif 'nmes_sham_c3' in events[-1]:
            mode = 'nmes'
    except Exception as e:
        print(f"Error reading events: {e}")
        return None
    
    # If mode is not set, return None or skip further processing
    if mode is None:
        print("No relevant events found.")
        return None
    
    # Initialize list
    df = []
    
    # Compute power spectral and fit FOOOF models for eyes open and closed 
    for cond in list(epochs.event_id):
        
        # Compute PSD using the Welch method
        epo_spectrum = epochs[cond].compute_psd(method='welch', fmin=1, fmax=30, n_jobs=-1, 
                                                picks='csd', **dict(average='median', 
                                                                    n_fft=int(2*epochs.info['sfreq'])))
        psds, freqs = epo_spectrum.get_data(return_freqs=True)
        
        # Get relative power with YASA 
        power_df = yasa.bandpower_from_psd(psd=psds.mean(0), freqs=freqs, 
                                           ch_names=epo_spectrum.ch_names,
                                           bands=[(1, 4, 'Delta'), (4, 8, 'Theta'),
                                                  (8, 12, 'Alpha'), (12, 30, 'Beta')])
        power_df.insert(0, 'Subject', subject)
        power_df.insert(1, 'Night', night)
        power_df.insert(2, 'Condition', cond)
        power_df.insert(3, 'Session', session)
        power_df.insert(4, 'Mode', mode)
        
        # FOOOF data                 
        fm = fooof.FOOOFGroup(max_n_peaks=5)
        fm.fit(freqs, psds.mean(0))  
        
        # Extract aperiodic components for all channel
        power_df['Offset'] = fm.get_params(name='aperiodic_params', col='offset')
        power_df['Aperiodic'] = fm.get_params(name='aperiodic_params', col='exponent')           
    
        # Define and plot frequency bands of interest
        if plot:
            bands = {'Delta (1-4 Hz)': (1, 4), 'Theta (4-8 Hz)': (4, 8), 
                     'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30)}
            
            epo_spectrum.plot_topomap(bands=bands, normalize=True)
            plt.suptitle(f'{cond.capitalize()}')
            plt.close('all')
            
        df.append(power_df)
    
    # Create df
    df = pd.concat(df).reset_index(drop=True)
            
    return df

def analyze_rs_hrv(file, figure_path, plot=False):
    # Get subject, night info
    subject, night, session, _, _ = file.split('/')[-1].split('-')[0].split('_')
     
    # Read raw
    raw = mne.io.read_raw(file)
    
    # Crop to only include RS recording phases for ECG eyes open/closed
    raw.crop(tmin = raw.annotations[0]['onset'], 
             tmax = raw.annotations[-1]['onset'] + 30).pick('ECG')
    
    # Get stimulation condition from epoch events
    try:  
        events = mne.read_events(f'/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/{subject}_{night}_csd-epo.fif', 
                                 return_event_id=True)
        if 'pn_sham_c3' in events[-1]:
            mode = 'pn'
        elif 'nmes_sham_c3' in events[-1]:
            mode = 'nmes'
            
    except Exception as e:
        print(f"Error reading events: {e}")
        return None
    
    # If mode is not set, return None or skip further processing
    if mode is None:
        print("No relevant events found.")
        return None
    
    # Get ECG peaks
    # peaks, info = nk.ecg_peaks(raw.get_data(units='uV').squeeze(), 
    #                             sampling_rate=raw.info['sfreq'], 
    #                             correct_artifacts=True)
    from sleepecg import detect_heartbeats
    # modified version of the beat detection algorithm using adaptive thresholds (Pan & Tompkins, 1985)
    peaks = detect_heartbeats(raw.get_data(units='uV').squeeze(), 
                              fs=raw.info['sfreq'])
       
    # Compute heart rate
    hr_rate = nk.signal_rate(peaks, sampling_rate=raw.info['sfreq'],
                             interpolation_method="monotone_cubic",
                             desired_length=None)
    
    # Skip particularly bad recordings 
    import statistics
    # Calculate the mode
    try:
        mode_hr_rate = statistics.mode(hr_rate)
    except statistics.StatisticsError:
        # Handle cases with no unique mode
        print("No unique mode found.")
        return None
    
    # Check if the mode is NaN
    if np.isnan(mode_hr_rate):
        print("No ECG peaks found.")
        return None
    
    # compute ecg signal quality
    hr_quality = nk.ecg_quality(raw.get_data(units='uV').squeeze(), 
                                sampling_rate=raw.info['sfreq'],
                                method="zhao2018", approach="fuzzy")
    
    # Compute HRV metrics
    if plot:
        hrv_indices = nk.hrv(peaks, sampling_rate=raw.info['sfreq'], show=True)       
        # Get the current figure
        fig = plt.gcf()
        # Set the desired size (width, height) in inches
        fig.set_size_inches(12, 8)  # for example, 12 inches by 8 inches
        # Optionally adjust layout
        plt.tight_layout()
        # Save the figure
        plt.savefig(figure_path + f'{subject}_{night}_{session}_hrv.png')
        # Close the plot
        plt.close('all')
    else:
        hrv_indices = nk.hrv(peaks, sampling_rate=raw.info['sfreq'], show=False)
    
    # Append info to HRV df
    hrv_indices.insert(0, 'Subject', subject)
    hrv_indices.insert(1, 'Night', night)
    hrv_indices.insert(2, 'Session', session)
    hrv_indices.insert(3, 'Mode', mode)
    hrv_indices.insert(4, 'HR', np.median(hr_rate))
    hrv_indices.insert(5, 'Quality', hr_quality)
    
    return hrv_indices
                           
#%%
if __name__ == '__main__':
    
    ## 1. Power Analysis
    path = '/media/administrator/Sleep_Data/Processed/RS/*csd-epo.fif'
    # path = '/mnt/server/data03/2023_NIDRA/Recordings/*/*/*'
    save_path = '/media/administrator/Sleep_Data/Processed/Statistics/'
    figure_path = '/media/administrator/Sleep_Data/Processed/Figures/RS/'
    power_dfs = []
    for file in tqdm(glob(path)):
        print(file)
        power_df = analyze_rs_data(file, figure_path, plot=False) 
        power_dfs.append(power_df)
        
    # Concatenate power dataframes
    power_dfs_all = pd.concat(power_dfs).reset_index(drop=True)
    power_dfs_all.to_csv(save_path + 'df_rs.csv') 
    
    ## 2. HRV Analyis
    raw_path = '/media/administrator/Sleep_Data/Processed/RS/*csd-raw.fif'
    hrv_dfs = []
    for file in tqdm(glob(raw_path)):
        print(file)
        hrv_df = analyze_rs_hrv(file, figure_path, plot=True) 
        hrv_dfs.append(hrv_df)
        
    # Concatenate HRV dataframes
    hrv_dfs_all = pd.concat(hrv_dfs).reset_index(drop=True)
    hrv_dfs_all.to_csv(save_path + 'df_rs_hrv.csv')   