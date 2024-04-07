#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:54:03 2024

@author: administrator
"""

import glob
import liesl
import matplotlib.pyplot as plt

path = path = '/media/administrator/Sleep_Data/Raw/*/*/*.xdf'
for file in glob.glob(path)[80:]:
    if 'sleepstim' in file:
        # file = '/media/administrator/Sleep_Data/Raw/CLNMES/IsEb_2/2_sleepstim_R001.xdf'
        subj_night = file.split('/')[-2]
        rec = file.split('/')[-1].split('_')[-1].split('.')[0][-1]
        print(f' Processing data from subject: {subj_night} for night: {subj_night.split("_")[1]}')
        xdf_file = liesl.api.XDFFile(file)
        
        # Extract recording data and information
        eego = xdf_file['eego']
        eego_data = eego.time_series
        ch_names = eego.channel_labels.copy()
        ch_types = eego.channel_types.copy()
        sfreq = int(eego.nominal_srate)
        times = eego.time_stamps.copy()
        
        # Reassign channel names
        ch_names[ch_names.index('chan_7')] = 'ECG'
        ch_names[ch_names.index('chan_19')] = 'EDC'
        ch_names[ch_names.index('chan_20')] = 'EOG'
        ch_names[ch_names.index('chan_22')] = 'EMG_L'
        ch_names[ch_names.index('chan_23')] = 'EMG_R'
        
        # Reassign channel types
        ch_types[0:64] = ['eeg']*64
        ch_types[ch_names.index('ECG')] = 'ecg'
        ch_types[ch_names.index('EDC')] = 'emg'
        ch_types[ch_names.index('EOG')] = 'eog'
        ch_types[ch_names.index('EMG_L')] = 'emg'
        ch_types[ch_names.index('EMG_R')] = 'emg'
    
        # Plot 
        plt.figure()
        plt.plot(times, eego_data[:, ch_names.index('EMG_R')]*1e6, label=f'EMG_R_{subj_night}')
        plt.plot(times, eego_data[:, ch_names.index('EMG_L')]*1e6, label=f'EMG_L_{subj_night}')
        plt.legend()
        plt.savefig(f'/home/administrator/Downloads/{subj_night}_EMG.png')
        plt.close('all')
        
        del xdf_file, eego_data