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
#cd('/users/neuro/sleep_stimulation')
from sleep_funs import process_raw_EDF_cfs, bandpower

#%%
## Physionet data
#cd('/home/administrator/Documents/Physionet_data')

files = list(set([f.split('-')[0] for f in sorted(listdir())]))

allArrays = []
stageArrays = []

for f in enumerate(files): 
    fname = f[1]
    
    datArray, stageArray = process_raw_EDF(fname)
     
    ## compute PSD with welch's method + yasa absolute/relative power extraction 
    # compute power spectral density with welch's method
    data = bandpower(datArray, fs=100, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                 (8, 12, 'Alpha'),(12, 16, 'Sigma'), 
                                 (16, 30, 'Beta')], relative=True)
    
    pickle.dump(dataArray, open(fname + "-allArrays.p", "wb"))
    
    pickle.dump(stageArray, open(fname + "-stageArrays.p", "wb"))
    
#%%      
%reset -f

#%%
## NSRR Cleveland Sleep Study
#cd('/users/neuro/cfs/polysomnography')

files = list(set([f.split('.')[0] for f in sorted(listdir())]))

allArrays = []
stageArrays = []

for f in enumerate(files): 
    fname = f[1]
    
    datArray, stageArray = process_raw_EDF(fname)
     
    ## compute PSD with welch's method + yasa absolute/relative power extraction 
    # compute power spectral density with welch's method
    data = bandpower(datArray, fs=100, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                 (8, 12, 'Alpha'),(12, 16, 'Sigma'), 
                                 (16, 30, 'Beta')], relative=True)
    
    pickle.dump(dataArray, open(fname + "-allArrays.p", "wb"))
    
    pickle.dump(stageArray, open(fname + "-stageArrays.p", "wb"))

