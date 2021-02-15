#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:40:22 2021

@author: administrator
"""

# =============================================================================
# Bad channel detection - comparing RANSAC, Bandpower ratio (my approach), Max's
# method 
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle
import seaborn as sns
import os 
import yasa
import mne
from autoreject import Ransac
from autoreject.utils import interpolate_bads
from os import chdir as cd
from os import listdir
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, preprocess_sleep_data, compare_hypnograms, check_match_data_hypno_elements, 
                                                     label_artifacts, load_preprocessed_data, Data_SW)

#%%
path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
for i, data_files in enumerate(files_list):
    #pass
    Data = pickle.load(open(data_files,"rb")) 
 
    # epoched data
    _, epoched_data = yasa.sliding_window(Data.data.T, sf=Data.sfreq, window=30)
    
    ## RANSAC method
    # mne-ize 
    info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)
    epochs = mne.EpochsArray(epoched_data/1e6, info, tmin = 0, baseline=(None)) 
    epochs.set_montage(mne.channels.make_standard_montage('standard_1005'))

    picks = mne.pick_types(epochs.info, eeg=True, stim=False, eog=False,
                           include=[], exclude=[])
    ransac = Ransac(verbose='progressbar', picks=picks, n_jobs=-1)
    epochs_clean = ransac.fit_transform(epochs)
    
    print(f'The following dataset: {data_files.split("/")[-1].split(".")[0]} has the following bad channels: {ransac.bad_chs_} ')
    print(f'The aforementioned dataset has a rectified 99th perentile amplitude of: {np.percentile(abs(Data.data[:,[Data.chans.index("M1"), Data.chans.index("M2")]].mean(-1)), 99)}')
    
    #del epochs, Data, epoched_data, epochs_clean, ransac
    
    evoked = epochs.average()
    evoked_clean = epochs_clean.average()
    
    # detected and repaired channels
    evoked.info['bads'] = ransac.bad_chs_
    evoked_clean.info['bads'] = ransac.bad_chs_
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    
    for ax in axes:
        ax.tick_params(axis='x', which='both', bottom='off', top='off')
        ax.tick_params(axis='y', which='both', left='off', right='off')
    
    evoked.plot(exclude=[], axes=axes[0], show=False)
    axes[0].set_title('Before RANSAC')
    evoked_clean.plot(exclude=[], axes=axes[1])
    axes[1].set_title('After RANSAC')
    fig.tight_layout()
    print('\n'.join(ransac.bad_chs_))
    
    ##
    ch_names = [epochs.ch_names[ii] for ii in ransac.picks][7::10]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(ransac.bad_log, cmap='Reds',
              interpolation='nearest')
    ax.grid(False)
    ax.set_xlabel('Sensors')
    ax.set_ylabel('Trials')
    plt.setp(ax, xticks=range(7, len(ransac.picks), 10),
             xticklabels=ch_names)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    fig.tight_layout(rect=[None, None, None, 1.1])
    plt.show()
