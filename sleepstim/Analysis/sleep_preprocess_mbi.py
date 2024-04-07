#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:43:35 2023

@author: administrator
"""

import mne
import os
import numpy as np
import neurokit2 as nk
mne.set_log_level('ERROR')

# Set the directory containing the sleep files -> CHANGE THIS! 
directory = '/mnt/server/data03/2023_NMTI_course/'

# Get files in directory
files = os.listdir(directory)

#%%

def preprocess_sleep_mbi_data(filename):
    
    # Load the data 
    raw = mne.io.read_raw_edf(os.path.join(directory, filename), 
                              preload=True, verbose=False)
    print(f'Processing file {filename.split(".")[0]}')
       
    # Find the channel names and rename them
    ch_names_og = raw.info['ch_names']
    print(ch_names_og)
    
    channel_mapping = {
        'EEG Fp1-Ref': 'Fp1',
        'EEG Fp2-Ref': 'Fp2',
        'EEG F3-Ref': 'F3',
        'EEG F4-Ref': 'F4',
        'EEG C3-Ref': 'C3',
        'EEG C4-Ref': 'C4',
        'EEG Fpz-Ref': 'Fpz',
        'EEG M2-Ref': 'M2'
        }
    
    # Rename channels
    raw.rename_channels(channel_mapping)
           
    # Bandpass filter the eeg data
    raw.filter(0.5, 30)
    
    # Notch filter the data
    raw.notch_filter([50, 100, 150])
    
    # Set montage
    raw.set_montage(mne.channels.make_standard_montage('standard_1020')) 
              
    # Re-reference the data to linked mastoids
    raw.set_eeg_reference(['M2'])
        
    # Bad channel detection and interpolation
    bads, _ = nk.eeg_badchannels(raw.get_data(picks='eeg', units='uV'))
    print(f'Detected bad channels: {list(np.asarray(raw.ch_names)[bads])} ')
    
    # Resample data
    raw.resample(100)
        
    # Save Raw 
    save_path = '/media/administrator/data/Study_2_data/NIDRA/'
    raw.save(save_path + f'{filename}-raw.fif', overwrite=True)
      
    
#%%

if __name__ == '__main__':
    for filename in files:
        if filename.endswith('.edf'):
            preprocess_sleep_mbi_data(filename) 
