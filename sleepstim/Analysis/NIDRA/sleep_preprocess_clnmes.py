#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:32:40 2023

@author: administrator
"""

import mne
import os
import numpy as np
import liesl
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import find_peaks
from tqdm import tqdm
import neurokit2 as nk
import gc
from glob import glob
from scipy import interpolate
import pyprep
mne.set_log_level('ERROR')

from sleepstim.core.io import channel_parser

def identify_bad_eegs(raw, 
                      deviation_threshold = 5.0,
                      extensive = False,
                      frac_bad = 0.4,
                      corr_thresh = 0.75,
                      corr_window_secs = 5,
                      run_ransac = False,
                      ):

    ## Noisy channels according to pyprep
    eeg_data = raw.resample(128).pick_types(eeg=True, exclude='bads')
    noisy_ch = pyprep.NoisyChannels(eeg_data, do_detrend=False,
                                    random_state=2022)
    noisy_ch.find_bad_by_nan_flat()
    noisy_ch.find_bad_by_deviation(deviation_threshold=deviation_threshold)
 
    if extensive:
        noisy_ch.find_bad_by_correlation(correlation_secs=corr_window_secs)
        noisy_ch.find_bad_by_hfnoise()
        noisy_ch.find_bad_by_SNR()
    if run_ransac:
        noisy_ch.find_bad_by_ransac(channel_wise=True, corr_thresh=corr_thresh,
                                    corr_window_secs=corr_window_secs, frac_bad=frac_bad)
 
    return noisy_ch.get_bads()  
   
def interpolate_cubic(data, times, win):
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

def tkeo(data, sf, normalize=True, plot=True, threshold_factor=3):
    """
    TKEO with max points significantly larger than the background signal.
    """

    # Vectorized TKEO computation
    emgf = np.zeros_like(data)
    emgf[1:-1] = data[1:-1]**2 - data[:-2] * data[2:]

    # Normalization
    emgZ = stats.zscore(data) if normalize else data
    emgE = stats.zscore(emgf) if normalize else emgf

    # Determine threshold
    threshold = np.mean(emgf) + threshold_factor * np.std(emgf)

    # Find peaks above threshold
    peaks, _ = find_peaks(emgf, height=threshold)

    # Plotting
    if plot:       
        # Add times
        times = np.arange(0, len(emgf)) / sf
        peak_times = times[peaks]        
        # Create figure
        plt.figure()
        plt.plot(times, emgZ, label='Raw Signal')
        plt.plot(times, emgE, 'm', label='TKEO Energy')
        plt.scatter(peak_times, emgf[peaks], marker='o', color='r', label='Significant Peaks')
        plt.title("TKEO energy plotted over EMG/EEG")
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.legend()
        plt.show()

    return peaks, emgf[peaks]
  
def epoch_sleep_nmes_data(raw, events_merged, event_id,
                          tmin=-3, tmax=3, csd=False, 
                          baseline=None, preload=True,
                          #the below is necessary for recordings where eeg stream disappeared and triggers still sent 
                          event_repeated='drop'): 
    
    # Create epochs based on nmes markers
    epochs = mne.Epochs(raw, events = events_merged, event_repeated = event_repeated,
                        event_id = event_id, baseline = baseline,  
                        tmin = tmin, tmax = tmax, preload = preload)
       
    # Compute CSD for epoched data
    if csd:
        epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
    
        return epochs, epochs_csd
    else:
        return epochs

def preprocess_sleep_nmes_data(filename, save_path, save=True):
    ## 1. Load and extract data streams from file

    # Load the data 
    xdf_file = liesl.api.XDFFile(filename)
    
    # Extract recording data and information
    eego = xdf_file['eego']
    eego_data = eego.time_series
    # Use the imported channel_parser
    ch_names, ch_types = channel_parser(eego['info'], eego_data)
    sfreq = int(eego.nominal_srate)
    times = eego.time_stamps.copy() 
    
    # Extract timestamps of classification markers
    classification_markers = [xdf_file['reiz-marker'].time_stamps[idx] for idx in 
                              range(len(xdf_file['reiz-marker'].time_series)) if 'stage_predict' in xdf_file['reiz-marker'].time_series[idx][0]]
    
    if 'pinknoise_trigger_send_c3_stim' in np.unique(xdf_file['reiz-marker'].time_series):
        night = 'pinknoise'
    else:
        night = 'nmes'
        del xdf_file, eego
        gc.collect()     
        return []
        
    if night == 'nmes':
        # Extract timestamps of nmes sham markers
        # nmes_sham_markers = [xdf_file['reiz-marker'].time_stamps[idx] 
        #                      for idx in range(len(xdf_file['reiz-marker'].time_series))
        #                      if 'nmes_sham_send' in xdf_file['reiz-marker'].time_series[idx][0]
        #                      or 'nmes_sham' in xdf_file['reiz-marker'].time_series[idx][0]]
        # Check if 'nmes_sham_send' is present in any description
        nmes_sham_send_exists = any('nmes_sham_send' in desc[0] for desc in xdf_file['reiz-marker'].time_series)
        
        # Now use list comprehension to select the appropriate markers
        nmes_markers = dict()
        for chan in ('c3','fz'):
            # Initialize a sub-dictionary for the channel if it doesn't exist
            if chan not in nmes_markers:
                nmes_markers[chan] = {'sham': [], 'stim': []}
                
            nmes_sham_marker = [xdf_file['reiz-marker'].time_stamps[idx] 
                                 for idx in range(len(xdf_file['reiz-marker'].time_series))
                                 if (f'nmes_sham_send_{chan}_sham' in xdf_file['reiz-marker'].time_series[idx][0]) or
                                    (f'nmes_sham_{chan}_sham' in xdf_file['reiz-marker'].time_series[idx][0] and not nmes_sham_send_exists)]
            nmes_markers[chan]['sham'] = nmes_sham_marker
            
            # Extract timestamps of nmes stim markers
            nmes_stim_start_marker = [xdf_file['reiz-marker'].time_stamps[idx] for idx in 
                                       range(len(xdf_file['reiz-marker'].time_series)) 
                                       if f'nmes_trigger_send_{chan}' in xdf_file['reiz-marker'].time_series[idx][0]]
    
            nmes_markers[chan]['stim'] = nmes_stim_start_marker
            
    elif night == 'pinknoise':
        ## OLD
        # nmes_sham_send_exists = any('pinknoise_sham_send' in desc[0] for desc in xdf_file['reiz-marker'].time_series)
        
        # # Now use list comprehension to select the appropriate markers
        # nmes_markers = dict()
        # for chan in ('c3','fz'):
        #     # Initialize a sub-dictionary for the channel if it doesn't exist
        #     if chan not in nmes_markers:
        #         nmes_markers[chan] = {'sham': [], 'stim': []}
                
        #     nmes_sham_marker = [xdf_file['reiz-marker'].time_stamps[idx] 
        #                          for idx in range(len(xdf_file['reiz-marker'].time_series))
        #                          if (f'pinknoise_sham_send_{chan}_sham' in xdf_file['reiz-marker'].time_series[idx][0]) or
        #                             (f'pinknoise_sham_{chan}_sham' in xdf_file['reiz-marker'].time_series[idx][0] and not nmes_sham_send_exists)]
        #     nmes_markers[chan]['sham'] = nmes_sham_marker
            
        #     # Extract timestamps of nmes stim markers
        #     nmes_stim_start_marker = [xdf_file['reiz-marker'].time_stamps[idx] for idx in 
        #                                range(len(xdf_file['reiz-marker'].time_series)) 
        #                                if f'pinknoise_trigger_send_{chan}' in xdf_file['reiz-marker'].time_series[idx][0]]
    
        #     nmes_markers[chan]['stim'] = nmes_stim_start_marker
                        
        ## New and considers only 'real' pinknoise timestamps     
        # Initialize a dictionary to store the markers for NMES and pink noise
        nmes_sham_send_exists = any('pinknoise_sham_send' in desc[0] for desc in xdf_file['reiz-marker'].time_series)

        # Now use list comprehension to select the appropriate markers
        nmes_markers = dict()
        for chan in ('c3','fz'):
            # Initialize a sub-dictionary for the channel if it doesn't exist
            if chan not in nmes_markers:
                nmes_markers[chan] = {'sham': [], 'stim': []}
            
            # Extract successive pink noise timestamps for stim
            pinknoise_stim_markers = []
            for idx in range(len(xdf_file['reiz-marker'].time_series) - 1):
                if f'pinknoise_trigger_send_{chan}' in xdf_file['reiz-marker'].time_series[idx][0]:
                    # Find the next 'pinknoise' marker
                    next_idx = idx + 1
                    while next_idx < len(xdf_file['reiz-marker'].time_series) and 'pinknoise' not in xdf_file['reiz-marker'].time_series[next_idx][0]:
                        next_idx += 1
                    # Check if we found a valid pink noise marker
                    if next_idx < len(xdf_file['reiz-marker'].time_series) and 'pinknoise' in xdf_file['reiz-marker'].time_series[next_idx][0]:
                        pinknoise_stim_markers.append(xdf_file['reiz-marker'].time_stamps[next_idx])
            nmes_markers[chan]['stim'] = pinknoise_stim_markers
           
            # Extract successive pink noise timestamps for sham
            pinknoise_sham_markers = []
            for idx in range(len(xdf_file['reiz-marker'].time_series) - 1):
                if (f'pinknoise_sham_send_{chan}_sham' in xdf_file['reiz-marker'].time_series[idx][0]) or \
                   (f'pinknoise_sham_{chan}_sham' in xdf_file['reiz-marker'].time_series[idx][0] and not nmes_sham_send_exists):
                    # Find the next 'pinknoise' marker
                    next_idx = idx + 1
                    while next_idx < len(xdf_file['reiz-marker'].time_series) and 'pinknoise' not in xdf_file['reiz-marker'].time_series[next_idx][0]:
                        next_idx += 1
                    # Check if we found a valid pink noise marker
                    if next_idx < len(xdf_file['reiz-marker'].time_series) and 'pinknoise' in xdf_file['reiz-marker'].time_series[next_idx][0]:
                        pinknoise_sham_markers.append(xdf_file['reiz-marker'].time_stamps[next_idx])
            nmes_markers[chan]['sham'] = pinknoise_sham_markers
            
    # Delete unused objects 
    del xdf_file, eego
    gc.collect()    
    
    # The following lines for manual ch_names/ch_types manipulation are now
    # handled by the imported channel_parser function.
    # # Rename certain channels
    # ch_names[ch_names.index('chan_7')] = 'ECG'
    # ch_names[ch_names.index('chan_19')] = 'EDC'
    # ch_names[ch_names.index('chan_20')] = 'EOG'
    # #ch_names[ch_names.index('chan_20')] = 'EOG_L'
    # #ch_names[ch_names.index('chan_21')] = 'EOG_R'
    # ch_names[ch_names.index('chan_22')] = 'EMG_L'
    # ch_names[ch_names.index('chan_23')] = 'EMG_R'
    # 
    # # Reassign channel types
    # ch_types[0:64] = ['eeg']*64
    # ch_types[ch_names.index('ECG')] = 'ecg'
    # ch_types[ch_names.index('EDC')] = 'emg'
    # ch_types[ch_names.index('EOG')] = 'eog'
    # # ch_types[ch_names.index('EOG_L')] = 'eog'
    # # ch_types[ch_names.index('EOG_R')] = 'eog'
    # ch_types[ch_names.index('EMG_L')] = 'emg'
    # ch_types[ch_names.index('EMG_R')] = 'emg'
    
    # Extract TKEO peaks from NMES stimulation
    #peaks, _ = tkeo(eego_data[:, ch_names.index('EDC')], sfreq)
                       
    for chan in nmes_markers.keys():
        # Convert sham timestamps into indices and adjust
        sham_timestamps = nmes_markers[chan]['sham']
        sham_indices = [np.argmin(np.abs(times - ts)) for ts in sham_timestamps]
        sham_indices_adjusted = list(np.asarray(sham_indices)) #+ int(0.0125 * sfreq))
        nmes_markers[chan]['sham_idx'] = sham_indices_adjusted
    
        # Convert stim timestamps into indices and adjust
        stim_timestamps = nmes_markers[chan]['stim']
        stim_indices = [np.argmin(np.abs(times - ts)) for ts in stim_timestamps]
        stim_indices_adjusted = list(np.asarray(stim_indices)) #+ int(0.0125 * sfreq))
        nmes_markers[chan]['stim_idx'] = stim_indices_adjusted
      
    # # Convert timestamps into indices
    # classification_idx = [np.argmin(np.abs(times - ts)) for ts in classification_markers]
    # nmes_sham_idx = [np.argmin(np.abs(times - ts)) for ts in nmes_sham_markers]
    # nmes_sham_idx = list(np.asarray(nmes_sham_idx) + int(0.0125*sfreq)) #- int(0.0875*sfreq))
    # events_stim_start_idx = [np.argmin(np.abs(times - ts)) for ts in nmes_stim_start_markers]
    # events_stim_start_idx = list(np.asarray(events_stim_start_idx) + int(0.0125*sfreq))
    
    # Create Events Arrays
    # Classification events
    classification_idx = [np.argmin(np.abs(times - ts)) for ts in classification_markers]
    events_classification = np.array([[idx, 0, 1] for idx in classification_idx])

    # Sham and Stim events for 'c3' and 'fz' channels
    events_sham_c3 = np.array([[idx, 0, 2] for idx in nmes_markers['c3']['sham_idx']])
    events_sham_fz = np.array([[idx, 0, 3] for idx in nmes_markers['fz']['sham_idx']])
    events_stim_c3 = np.array([[idx, 0, 4] for idx in nmes_markers['c3']['stim_idx']])
    events_stim_fz = np.array([[idx, 0, 5] for idx in nmes_markers['fz']['stim_idx']])
    
    # Merge and Sort the Events Arrays
    try:
        
        events_merged = np.concatenate([events_classification, events_sham_c3, events_sham_fz, events_stim_c3, events_stim_fz], axis=0)
        events_merged = events_merged[np.argsort(events_merged[:, 0])]
        
        # # Sort the combined events array by the sample indices
        # events_merged = events_merged[np.argsort(events_merged[:, 0])]
    
        # # Create events array and combine
        # events_classification = np.array([[sample, 0, 1] for sample in classification_idx])
        # events_sham = np.array([[sample, 0, 2] for sample in nmes_sham_idx])
        # events_stim_start = np.array([[sample, 0, 3] for sample in events_stim_start_idx])
        # events_merged = np.concatenate([
        #                                 events_classification, 
        #                                 events_sham, 
        #                                 events_stim_start], 
        #                                axis=0)
        # events_merged = events_merged[np.argsort(events_merged[:, 0])]
        
        # Create mapping of stimuli for MNE
        if night == 'nmes':
            mapping = {
                1: 'classification', 
                # 2: 'nmes_sham', 
                # 3: 'nmes_stim', 
                2: 'nmes_sham_c3',
                3: 'nmes_sham_fz',
                4: 'nmes_stim_c3',
                5: 'nmes_stim_fz'
            }
        
            event_id = {
                # 'classification': 1, 
                # 'nmes_sham': 2, 
                # 'nmes_stim': 3, 
                'nmes_sham_c3' : 2,
                'nmes_sham_fz': 3,
                'nmes_stim_c3': 4,
                'nmes_stim_fz': 5
            }
            
        elif night == 'pinknoise':
            mapping = {
                1: 'classification', 
                2: 'pn_sham_c3',
                3: 'pn_sham_fz',
                4: 'pn_stim_c3',
                5: 'pn_stim_fz'
            }
        
            event_id = {
                'pn_sham_c3' : 2,
                'pn_sham_fz': 3,
                'pn_stim_c3': 4,
                'pn_stim_fz': 5
            }
        
        # Info about recording
        print(f'The length of the recording : {np.round(eego_data.shape[0]/60/sfreq, 2)} minutes of data at {sfreq} Hz sampling frequency.')
        
        # Channel Index
        chan_idx = [idx for idx in range(len(ch_names)) if 'EEG' not in ch_types[idx]]
         
        ## 2. MNE procedure
        
        # Create MNE object
        info = mne.create_info(ch_names = list(np.asarray(ch_names)[chan_idx]), 
                               sfreq = sfreq,
                               ch_types = list(np.asarray(ch_types)[chan_idx]))
        
        # Create raw object
        raw = mne.io.RawArray(eego_data[:,chan_idx].T, info)
        
        # Delete unused objects 
        del eego_data, times
        gc.collect() 
        
        # Set montage
        raw.set_montage(mne.channels.make_standard_montage('standard_1005')) 
        
        # Add events 
        annot_from_events = mne.annotations_from_events(
            events=events_merged,
            event_desc=mapping,
            sfreq=raw.info["sfreq"],
            orig_time=None,
        )
        raw.set_annotations(annot_from_events)
          
        ## 3. Pre-process data with MNE
        
        # Apply cubic interpolation for stimulation artifacts
        # if night == 'nmes':
        #     raw.apply_function(fun=interpolate_cubic, 
        #                        picks='eeg', 
        #                        n_jobs=-1,
        #                        channel_wise=True,
        #                        #annotations=raw.annotations[np.logical_or(raw.annotations.description == 'nmes_stim_fz',
        #                        #                                          raw.annotations.description == 'nmes_stim_c3')],
        #                        **dict(times = raw.times, 
        #                               win = [-.02, .15]))
        
            # raw.apply_function(fun=interpolate_cubic, 
            #                    picks='eeg', 
            #                    n_jobs=-1, 
            #                    channel_wise=False,
            #                    times=raw.times, 
            #                    annotations=raw.annotations[raw.annotations.description == 'nmes_stim'],
            #                    win=(-0.016, 0.117),
            #                    sfreq=raw.info['sfreq'])
            
        # Bandpass filter data
        raw.filter(0.3, 45, picks='eeg', n_jobs=1)
        raw.filter(0.3, 35, picks='eog', n_jobs=1)
        raw.filter(10, 100, picks=['EMG_L','EMG_R'], n_jobs=1)
        # raw.filter(0.3, 70, picks='ecg', n_jobs=1)
        # Process ECG separately 
        raw.apply_function(picks='ECG', fun=nk.ecg_clean, 
                           **dict(sampling_rate=raw.info['sfreq']))
        
        # Remove 'EDC' and 'ECG' from notch filtering
        channels_to_filter = [chan for chan in raw.ch_names if chan not in 'EDC']
    
        # Notch filter the data
        raw.notch_filter([50, 100, 150], 
                         picks=channels_to_filter, 
                         # method='spectrum_fit', 
                         n_jobs=1)
        
        # Re-reference the data to linked mastoids
        raw.set_eeg_reference(['M1', 'M2'])
           
        # Bad channel detection
        bads, _ = nk.eeg_badchannels(raw.get_data(picks='eeg', units='uV'), distance_threshold=0.99)
        raw.info['bads'] = list(np.asarray(raw.ch_names)[bads])
        # raw.info['bads'] = identify_bad_eegs(raw, extensive = False, run_ransac = False)
        
        # Interpolate bads
        raw.interpolate_bads()
        
        ## 4. Epoch data
        epochs = epoch_sleep_nmes_data(raw, events_merged, event_id,
                                       tmin=-3, tmax=3, csd=False, 
                                       baseline=None, preload=True)
        
        ## 5. Resample data
        raw.resample(128)
        epochs.resample(128)
        
        ## 6. Save Raw/Epochs 
        # Save into path
        if save:
            subj = filename.split('/')[-2]
            rec = filename.split('/')[-1].split('_')[-1].split('.')[0][-1]
            raw.save(save_path + f'{subj}_{rec}-raw.fif', overwrite=True)
            epochs.save(save_path + f'{subj}_{rec}-epo.fif', overwrite=True)
    
        # Delete unused objects 
        del raw, epochs  
        gc.collect() 
        
    except:
        pass
                                    
#%%

path = '/media/administrator/Sleep_Data/Raw/*/*'
if __name__ == '__main__':
    for file in tqdm(glob(path)):
        if 'sleepstim' in file:
            subj_night = file.split('/')[-2]
            rec = file.split('/')[-1].split('_')[-1].split('.')[0][-1]
            # save_path = '/media/administrator/data/Study_2_data/NIDRA/CLNMES/Sleep/'
            save_path = '/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/'
            if not os.path.exists(save_path + f'{subj_night}_{rec}-raw.fif'):
                print(f' Processing data from subject: {subj_night} for night: {subj_night.split("_")[1]}')
                preprocess_sleep_nmes_data(file, save_path, save=True)    
