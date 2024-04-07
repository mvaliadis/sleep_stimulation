#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:32:40 2023

@author: administrator
"""

import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import numpy as np
import liesl
from tqdm import tqdm
import neurokit2 as nk
import gc
from glob import glob
import pyprep
import os
mne.set_log_level('ERROR')
   
def preprocess_rs_data(filename, save_path, figure_path, save=True):  
    ## 0. Extract basic subject info
    subject = filename.split('/')[-2].split("_")[0]
    night = filename.split('/')[-2].split("_")[1]
    session = filename.split('/')[-1].split('_')[-2]
    print(f' Processing RS data from subject: {subject} for night: {night} for session: {session}')
    
    ## 1. Load and extract data streams from file
    # Load the data 
    xdf_file = liesl.api.XDFFile(filename)
    
    # Extract recording data and information
    eego = xdf_file['eego']
    eego_data = eego.time_series
    ch_names = eego.channel_labels.copy()
    ch_types = eego.channel_types.copy()
    sfreq = int(eego.nominal_srate)
    times = eego.time_stamps.copy() 
    
    # Extract timestamps of classification markers with list comprehension
    markers = dict()
    for cue in ('augen_auf','augen_zu'):            
        marker = [xdf_file['reiz-marker'].time_stamps[idx] for idx in 
                  range(len(xdf_file['reiz-marker'].time_series)) if cue in xdf_file['reiz-marker'].time_series[idx][0]]
        markers[cue] = marker
        
    # Delete unused objects 
    del xdf_file, eego
    gc.collect()    
    
    # Rename certain channels
    ch_names[ch_names.index('chan_7')] = 'ECG'
    ch_names[ch_names.index('chan_19')] = 'EDC'
    
    # Reassign channel types
    ch_types[0:64] = ['eeg']*64
    ch_types[ch_names.index('ECG')] = 'ecg'
    ch_types[ch_names.index('EDC')] = 'emg'
    
    # Convert timestamps into indices
    for cue in markers.keys():
        timestamps = markers[cue]
        indices = [np.argmin(np.abs(times - ts)) for ts in timestamps]
        indices_adjusted = list(np.asarray(indices))
        markers[cue] = indices_adjusted
    
    # Create Events Arrays
    events_open = np.array([[idx, 0, 1] for idx in markers['augen_auf']])
    events_closed = np.array([[idx, 0, 2] for idx in markers['augen_zu']])
    
    # Merge and Sort the Events Arrays
    events_merged = np.concatenate([events_open, events_closed], axis=0)
    events_merged = events_merged[np.argsort(events_merged[:, 0])]
        
    # Create mapping of stimuli for MNE
    mapping = {
        1: 'eyes_open', 
        2: 'eyes_closed',

    }

    event_id = {
        'eyes_open' : 1,
        'eyes_closed': 2,
    }
    
    # Info about recording
    print(f'The length of the recording : {np.round(eego_data.shape[0]/60/sfreq, 2)} minutes of data at {sfreq} Hz sampling frequency.')
    # Break preprocessing if length is less than 5 minutes
    if np.round(eego_data.shape[0]/60/sfreq, 2) >= 5:
                  
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
        # Bandpass filter the eeg data
        raw.filter(1, 45, picks='eeg', n_jobs=-1)
        # raw.filter(0.3, 70, picks='ecg', n_jobs=-1)
        # Process ECG separately 
        raw.apply_function(picks='ECG', fun=nk.ecg_clean, 
                           **dict(sampling_rate=raw.info['sfreq']))
        
        # Remove 'EDC' and 'ECG' from notch filtering
        channels_to_filter = [chan for chan in raw.ch_names if chan not in 'EDC']
    
        # Notch filter the data
        raw.notch_filter([50, 100, 150], 
                         picks=channels_to_filter, 
                         # method='spectrum_fit', 
                         n_jobs=-1)
                
        # Bad channel detection, interpolation, and robust rereferencing -> pyprep
        prep_params = {"ref_chs": ch_names[0:64],
                       "reref_chs": ch_names[0:64]}
    
        pyprep_obj = pyprep.reference.Reference(raw, prep_params, ransac=False)
        pyprep_obj.perform_reference()
        
        # Update the original 'raw' object with processed EEG data
        eeg_data = pyprep_obj.raw.get_data(picks='eeg')
        eeg_indices = [raw.ch_names.index(ch_name) for ch_name in ch_names[0:64]]
        for idx, ch_idx in enumerate(eeg_indices):
            raw._data[ch_idx, :] = eeg_data[idx, :]
    
        # Bad channel detection, interpolation, and car rereferencing -> neurokit2/mne
        # bads, _ = nk.eeg_badchannels(raw.get_data(picks='eeg', units='uV'))
        # raw.info['bads'] = list(np.asarray(raw.ch_names)[bads])
        
        # # Interpolate bads
        # raw.interpolate_bads()
        
        # # Re-reference the data to common average 
        # raw.set_eeg_reference(ref_channels='average')
            
        ## 5. Perform ICA to remove eye blinks/muscle artifacts (PICARD - FastICA solution)
        ica = ICA(n_components=32, method='picard', random_state=42,
                  fit_params = dict(ortho=True, extended=True)) 
        print('\n***** Performing ICA ...\n')
        ica.fit(raw, picks='eeg')
    
        # Determine ecg component to remove
        ecg_idx, ecg_scores = ica.find_bads_ecg(raw, method = 'ctps', threshold= 'auto')
        
        # Automated component detection assistance 
        ic_labels = label_components(raw, ica, method="iclabel")
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        print(f"Excluding these ICA components: {exclude_idx}")
        
        # components to exclude 
        ica.exclude = exclude_idx + ecg_idx
        
        # # Plot signal with removed components
        # ica.plot_overlay(raw, exclude=ica.exclude)
        
        # Apply ICA
        raw_clean = ica.apply(raw)
        
        ## 6. Epoch data
        epochs = mne.Epochs(raw_clean, events = events_merged, 
                            event_id = event_id, baseline = None,  
                            tmin = 0, tmax = 30, preload = True)
                  
        # Compute CSD for raw/epoched data
        raw_csd = mne.preprocessing.compute_current_source_density(raw)
        epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
                    
        ## 7. Resample data
        raw_csd.resample(128)
        epochs.resample(128)
        epochs_csd.resample(128)
        
        ## 8. Save Raw/Epochs 
        # Save into path
        if save:
            raw_csd.save(save_path + f'{subject}_{night}_{session}_rs_csd-raw.fif', overwrite=True)
            epochs.save(save_path + f'{subject}_{night}_{session}_rs-epo.fif', overwrite=True)
            epochs_csd.save(save_path + f'{subject}_{night}_{session}_rs_csd-epo.fif', overwrite=True)
            
        ## 9. Plot cleaned data
        plot = raw_clean.plot(duration=max(raw_csd.get_data().shape),
                              start=0, color={'eeg': 'steelblue'},
                              n_channels = 64,
                              title='Cleaned Signal', 
                              show_scalebars=True, show = False)
        plot.grab().save(figure_path + f'{subject}_{night}_{session}_rs_cleaned_signal.png')
    
        # Delete unused objects 
        del raw, epochs  
        gc.collect()   
                               
#%%

path = '/media/administrator/Sleep_Data/Raw/*/*/*'
# path = '/mnt/server/data03/2023_NIDRA/Recordings/*/*/*'
save_path = '/media/administrator/Sleep_Data/Processed/RS/'
figure_path = '/media/administrator/Sleep_Data/Processed/Figures/RS/'
if __name__ == '__main__':
    for file in tqdm(glob(path)):
        if 'resting_state' in file:  
            subject, night = file.split('/')[-2].split('-')[0].split('_')
            session = file.split('/')[-1].split('_')[-2]
            if not os.path.exists(save_path + f'{subject}_{night}_{session}_rs_csd-raw.fif'):
                print(f' Processing data from subject: {subject} for night: {night} and session: {session}')
                preprocess_rs_data(file, save_path, figure_path, save=True)               