#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:54:35 2021

@author: administrator
"""

from multitaper_toolbox.python.multitaper_spectrogram_python import multitaper_spectrogram
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, load_preprocessed_data)
import yasa
import numpy as np
import matplotlib.pyplot as plt
import os

Data = load_preprocessed_data('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/D1BOI2AY_3_preproc_data.p')[0]

#%% multitaper toolbox
spect, stimes, sfreqs = multitaper_spectrogram(data = Data.data[:,Data.chans.index('F3')], fs = Data.sfreq, 
                                               frequency_range=(0,25), time_bandwidth=4, num_tapers=None, 
                                               window_params=(4, 1), min_nfft=0, detrend_opt='constant', 
                                               multiprocess=True, cpus=16, weighting='unity', plot_on=True, 
                                               clim_scale = True, cmap_percentile = (10, 90), verbose=True, 
                                               xyflip=False)

#%% yasa toolbox (plotting implementation only)

path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/'
files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
for file in files_list:
    Data = load_preprocessed_data(file)[0]
    plot = yasa.plot_spectrogram(data = Data.data[:,Data.chans.index('F3')], sf = Data.sfreq, win_sec = 30, 
                                 fmin = 0.5, fmax = 25, cmap = 'Spectral_r', trimperc=2.5)

