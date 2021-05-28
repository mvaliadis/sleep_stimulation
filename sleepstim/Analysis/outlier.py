#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:37:23 2021

@author: administrator
"""

from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.clustering import Potato
from scipy.stats import stats
import yasa
import seaborn as sns

def outlier_detection(data_filt, sf, eeg_index, ch_names, threshold=3, window=2):
    ## Global epoch check 
    _, epochs = yasa.sliding_window(data_filt[:,eeg_index].T, sf=sf, window=window)
    n_epochs, n_chans, n_times = epochs.shape
    epoch_art = np.zeros(n_epochs, dtype='int')
    # Calculate the covariance matrices (n_epochs, n_chan, n_chan)
    covmats = Covariances().fit_transform(epochs)
    # Shrink the covariance matrix (ensure positive semi-definite)
    covmats = Shrinkage().fit_transform(covmats)
    
    # Plot the (average) covariance matrix of epochs
    plt.figure(figsize=(10, 6))
    sns.heatmap(covmats.mean(0), cmap='Blues', square=True, 
                xticklabels=ch_names[0:63], yticklabels=ch_names[0:63],
                vmin = 10, vmax = 90)
    plt.title('Signal variance-covariance matrix')
    plt.xlabel('Channels')
    _ = plt.ylabel('Channels')

    # Define Potato instance: 0 = clean, 1 = art
    # To increase speed we set the max number of iterations from 10 to 100
    potato = Potato(metric='riemann', threshold=threshold, pos_label=0,
                    neg_label=1, n_iter_max=10)
    # Create empty z-scores output (n_epochs)
    zscores = np.zeros(n_epochs, dtype='float') * np.nan
    # Apply Potato algorithm, extract z-scores and labels
    zs = potato.fit_transform(covmats)
    art = potato.predict(covmats).astype(int)
    # Append to global vector
    epoch_art = art
    zscores = zs
    
    
    ## Channel wise epoch rejection
    ch_var = np.asarray([covmats[:,i, i] for i in range(covmats.shape[1])])
    
    return outlier_idx
