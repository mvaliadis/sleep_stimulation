#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:41:47 2021

@author: administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import seaborn as sns
import os 
import yasa
import mne
from os import chdir as cd
from os import listdir
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, preprocess_sleep_data, compare_hypnograms, check_match_data_hypno_elements,
                                                     label_artifacts, load_preprocessed_data, Data_SW)

#%%
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    for i, data_files in enumerate(files_list):
    # if data_files.endswith('_sham.p'):
        pass
        Data = pickle.load(open(data_files,"rb")) 
        Data.pinknoise_timestamps_sync = [Data.times[np.abs(Data.times - Data.pinknoise_times[ix]).argmin()] for ix in range(len(Data.pinknoise_times))]
        Data.pinknoise_timestamps_sync_adj = np.asarray(Data.pinknoise_timestamps_sync)*Data.sfreq
        first_bursts = Data.pinknoise_timestamps_sync_adj[::2]
        insp_ix = [np.where(Data.times*Data.sfreq == first_bursts[i])[0][0] for i in range(len(first_bursts))]
        center, _ = yasa.get_centered_indices(Data.data[:,Data.chans.index('C3')], np.asarray(insp_ix), 
                                              npts_before = Data.sfreq*1.5, npts_after = Data.sfreq*2)
        info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
        epochs = mne.EpochsArray(np.swapaxes(Data.data[center]/1e6, 1, 2), info, tmin = -1.5, baseline=(-1.5, -1.0), proj=False) 
        # epochs.set_montage(mne.channels.make_standard_montage('standard_1005'))
        
        # ## compute CSD 
        # epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
        # evoked_csd = epochs_csd.average()
        
        ## plotting without CSD
        # epochs.plot_image(combine='mean', picks='C3', title = f'{data_files.split("/")[-1].split(".")[0]} - C3')
        epochs.average(method='mean', picks='C3').plot()
        plt.title(f'{data_files.split("/")[-1].split(".")[0]} - C3')
        
        ## plotting with CSD
        # epochs_csd.plot_image(picks='C3')
        epochs_csd.average(method='mean', picks='C3').plot()
        plt.title(f'{data_files.split("/")[-1].split(".")[0]} - C3 - CSD')
        
        ## compartive plotting
        epochs.average().plot_joint(title='Average Mastoids', show=False)
        evoked_csd.plot_joint(title='Current Source Density')
    

    
#%%

ptps = [np.ptp(C3_pn[i,0:Data.sfreq*2]) for i in range(np.shape(C3_pn)[0])]
loc = np.where(np.asarray(ptps)>75)[0]

for i in loc:
    plt.figure()
    plt.plot(C3_pn_narrow[i,0:Data.sfreq*3])
    plt.vlines(Data.sfreq*1.5, ymin=-75, ymax=40)

