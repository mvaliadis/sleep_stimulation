#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:24:59 2020

@author: mvaliadis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import yasa 
from scipy.signal import welch
from os import chdir as cd
from os import listdir
from sleepstim.sleep_funs import process_raw_EDF, process_raw_EDF_cfs, bandpower
cd('/media/administrator/data/cfs/polysomnography')

#%%
## Physionet data
#cd('/home/administrator/Documents/Physionet_data')

files = list(set([f.split('-')[0] for f in sorted(listdir()) if f.endswith('.edf')]))

allArrays = []
stageArrays = []

for idx, fname in enumerate(files):
    
    datArray, stageArray = process_raw_EDF(fname)
     
    ## compute PSD with welch's method + yasa absolute/relative power extraction 
    # compute power spectral density with welch's method
    data = bandpower(datArray, fs=100, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                 (8, 12, 'Alpha'),(12, 16, 'Sigma'), 
                                 (16, 30, 'Beta')], relative=True)
    
    # reshape data for classifier, must be (epochs x (nchans*bands))
    data = np.swapaxes(data, 0, 1)
    nepochs, nbands, nchans = np.shape(data)
    data = data.reshape(nepochs, nchans*nbands, order='F')
    
    pickle.dump(data, open(fname + "-allArrays.p", "wb"))
    
    pickle.dump(stageArray, open(fname + "-stageArrays.p", "wb"))
    
#%%      
%reset -f

#%%
## NSRR Cleveland Sleep Study
#cd('/users/neuro/cfs/polysomnography')

#files = list(set([f.split('.')[0] for f in sorted(listdir()) if f.endswith('.xml') or f.endswith('.edf')]))
files = list(set([f.split('.')[0] for f in sorted(listdir()) if f.endswith('.edf')]))

allArrays = []
stageArrays = []

for idx, fname in enumerate(files): 
    print(idx)
    datArray, stageArray = process_raw_EDF_cfs(fname)
     
    ## compute PSD with welch's method + yasa absolute/relative power extraction 
    # compute power spectral density with welch's method
    data = bandpower(datArray, fs=128, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                 (8, 12, 'Alpha'),(12, 16, 'Sigma'), 
                                 (16, 30, 'Beta'), (59, 61, 'Line noise')], relative=True)
    
    # reshape data for classifier, must be (epochs x (nchans*bands))
    data = np.swapaxes(data, 0, 1)
    nepochs, nbands, nchans = np.shape(data)
    data = data.reshape(nepochs, nchans*nbands, order='F')
    
    # save as pickle files
    pickle.dump(data, open(fname + "_allArrays.p", "wb"))
    pickle.dump(stageArray, open(fname + "_stageArrays.p", "wb"))

#%%      
%reset -f
