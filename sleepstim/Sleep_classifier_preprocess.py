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
from tqdm import tqdm
from scipy.signal import welch
from os import chdir as cd
from os import listdir
from sleepstim.sleep_funs import (process_raw_EDF, process_raw_EDF_cfs, bandpower, 
                                  feature_extraction, ecg_feature_extraction)
cd('/media/administrator/data/cfs/polysomnography')

#%%
## Physionet data
#cd('/home/administrator/Documents/Physionet_data')

files = list(set([f.split('-')[0] for f in sorted(listdir()) if f.endswith('.edf')]))

allArrays = []
stageArrays = []

for idx, fname in enumerate(files):
    print(f'Processing dataset: {fname}')
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


for idx, fname in enumerate(tqdm(files)): 
    print(f'Processing dataset: {fname}')
    datArray, stageArray = process_raw_EDF_cfs(fname)
     
    ## compute PSD with welch's method + yasa absolute/relative power extraction 
    # compute power spectral density with welch's method
    # data = bandpower(datArray, fs=128, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
    #                                           (8, 12, 'Alpha'), (12, 16, 'Sigma'), 
    #                                           (16, 30, 'Beta'), (59, 61, 'Line noise')], 
    #                  relative=True)
       
    # # reshape data for classifier, must be (epochs x (nchans*bands))
    # data = np.swapaxes(data, 0, 1)
    # nepochs, nbands, nchans = np.shape(data)
    # data = data.reshape(nepochs, nchans*nbands, order='F')
    
    # more extesive feature extraction
    data2 = feature_extraction(datArray, sf=128)
    data2['subject'] = [fname.split('-')[-1]] * len(data2)
    
    # ecg/hrv feature extraction
    # data_ecg = ecg_feature_extraction(datArray[:,-1,:], sf=128, method='neurokit2')
    # data_ecg['subject'] = [fname.split('-')[-1]] * len(data_ecg)
    # data_ecg['stage'] = data_sleepecg['stage']
    
    # data_sleepecg = ecg_feature_extraction(datArray[:,-1,:], sf=128, method='sleepecg')
    # data_sleepecg['subject'] = [fname.split('-')[-1]] * len(data_sleepecg)
    
    # # numpy-fi this 
    # data2_np = []
    # for c, df in data2.groupby(['chan']):
    #     data2_np.append(np.expand_dims(df[df.columns[1:-2]].to_numpy(),1))
    # dfs = np.concatenate(data2_np, axis=1)
    # nepochs, nchans, nfeats = np.shape(dfs)
    # data2_dfs = dfs.reshape(nepochs, nchans*nfeats, order='F')
    
    # save as pickle files
    #pickle.dump(data, open(fname + "_allArrays.p", "wb"))
    pickle.dump(data2, open(fname + "_allArrays_df_new.p", "wb"))
    #pickle.dump(data_ecg, open(fname + "_allArrays_hrv_df.p", "wb"))
    #pickle.dump(data2_dfs, open(fname + "_allArrays_numpy.p", "wb"))
    #pickle.dump(stageArray, open(fname + "_stageArrays.p", "wb"))

#%%      
%reset -f
