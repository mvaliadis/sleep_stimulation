#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 00:02:07 2024

@author: administrator
"""

import numpy as np
import pingouin as pg

# def permutation_test_epochs(epochs, n_iter=100, fdr=0.05):
#     """
#     Perform a permutation test on MNE Epochs object to compute GFP, p-values, and FDR threshold.
    
#     Parameters:
#     - epochs: MNE Epochs object
#     - n_iter: number of permutations for the test
#     - fdr: False Discovery Rate threshold
    
#     Returns:
#     - FDR_threshold: Computed FDR threshold
#     - p_overall: Overall p-value
#     """
#     data = epochs.get_data('csd')*1e3  # Extract data from epochs
#     nObservations, nElectrodes, nDataPoints = data.shape

#     # GFP calculation
#     gfp = np.zeros((n_iter, nDataPoints))
#     gfp[0, :] = np.std(np.mean(data, axis=0), axis=0, ddof=1)

#     rdata = np.zeros_like(data)

#     # Permutation loop
#     for i in range(1, n_iter):
#         for j in range(nObservations):
#             rdata[j, :, :] = data[j, np.random.permutation(nElectrodes), :]
#         gfp[i, :] = np.std(np.mean(rdata, axis=0), axis=0, ddof=1)

#     # Compute p-values
#     p = np.array([(np.sum(gfp[:, t] >= gfp[0, t]) / n_iter) for t in range(nDataPoints)])

#     # FDR handling
#     p_exp = np.arange(1, nDataPoints + 1) / n_iter
#     p_corr = p_exp * fdr
#     p_sorted = np.sort(p)
#     FDR_threshold = p_sorted[np.max(np.where(p_sorted <= p_corr)[0])]

#     # Compute false positives under the null hypothesis
#     p_fake = np.zeros((nDataPoints, n_iter))
#     Hits = np.zeros(n_iter)

#     for i in range(n_iter):
#         for t in range(nDataPoints):
#             p_fake[t, i] = np.sum(gfp[:, t] >= gfp[i, t]) / n_iter
#         Hits[i] = np.sum(p_fake[:, i] < FDR_threshold)

#     # Calculate overall p-value
#     p_overall = np.sum(Hits[0] <= Hits) / n_iter

#     return FDR_threshold, p_overall

# def topographic_consistency_test(epochs, n_iter=100, fdr=0.05):
#     """
#     Perform a Topographic Consistency Test on MNE Epochs object to evaluate the spatial consistency
#     of ERP maps across subjects within an experimental condition.

#     Parameters:
#     - epochs: MNE Epochs object containing data from multiple subjects or multiple trials from the same subject.
#     - n_iter: Number of permutations for the test.
#     - fdr: False Discovery Rate threshold.

#     Returns:
#     - FDR_threshold: Computed FDR threshold.
#     - p_values: Time-series of p-values indicating the probability that the GFP of the grand-mean ERP
#                 across subjects is compatible with the null hypothesis.
#     """
#     # Extract data from epochs, assuming data is preloaded
#     data = epochs.get_data('csd')*1e3  # Shape: (n_epochs, n_channels, n_times)
    
#     # Calculate GFP for the grand-mean ERP map
#     grand_mean_gfp = epochs.average('csd').data.std(axis=0, ddof=0)*1e3
    
#     # Initialize an array to store GFP values for permuted data
#     permuted_gfps = np.zeros((n_iter, data.shape[2]))
    
#     for i in range(n_iter):
#         # Permute data within each epoch (subject/trial) across channels
#         for j in range(data.shape[0]):
#             data[j, :, :] = data[j, np.random.permutation(data.shape[1]), :]
#         # Compute the grand-mean map for the permuted data
#         permuted_mean_map = np.mean(data, axis=0)
#         # Calculate GFP for the permuted grand-mean map
#         permuted_gfps[i, :] = np.std(permuted_mean_map, axis=0, ddof=0)
    
#     # Calculate p-values for each time point
#     p_values = np.array([np.mean(permuted_gfps[:, t] >= grand_mean_gfp[t]) for t in range(data.shape[2])])
    
#     # Adjust p-values for multiple comparisons using FDR
#     # This step can be adapted based on the preferred method for FDR correction
#     sorted_p_values = np.sort(p_values)
#     n_tests = len(p_values)
#     cumulative_fdr = (np.arange(1, n_tests + 1) / n_tests) * fdr
#     below_threshold = sorted_p_values <= cumulative_fdr
#     if np.any(below_threshold):
#         FDR_threshold = sorted_p_values[below_threshold][-1]
#     else:
#         FDR_threshold = 0  # No p-values below FDR threshold

#     return FDR_threshold, p_values

def calculate_gfp_correlation(epochs, method='pearson'):
    """
    Calculate the correlation between the GFP of individual trials and the GFP of the evoked average
    to get a consistency score.
    """
    try:
        # Extract data from epochs
        data = epochs.get_data('eeg')*1e6  # Shape: (n_epochs, n_channels, n_times)
    
        # Calculate GFP for each trial
        trial_gfps = np.std(data, axis=1, ddof=0)  # Shape: (n_epochs, n_times)
    except:
        # Extract data from epochs
        data = epochs.get_data('csd')*1e3  # Shape: (n_epochs, n_channels, n_times)
        
        # Calculate RMS for each trial
        trial_gfps = np.linalg.norm(data, axis=1) / np.sqrt(len(data))
        
    trial_gfps = trial_gfps[:, epochs.time_as_index(0)[0]:epochs.time_as_index(0.2)[0]]
    
    # Calculate the evoked (grand-mean) average across all trials and its GFP
    evoked_average_gfp = epochs.average('csd').data.std(axis=0, ddof=0)*1e3
    evoked_average_gfp = evoked_average_gfp[epochs.time_as_index(0)[0]:epochs.time_as_index(0.2)[0]]
    
    # Calculate correlation between GFP of each trial and the evoked average GFP
    if method=='pearson':
        consistency_scores = np.array([np.corrcoef(trial_gfp, evoked_average_gfp)[0, 1] for trial_gfp in trial_gfps])
    elif method=='spearman': 
        consistency_scores = np.array([pg.corr(trial_gfp, evoked_average_gfp, method="spearman").r 
                                   for trial_gfp in trial_gfps])
    
    # Calculate the overall consistency as the mean of individual consistency scores
    overall_consistency = np.nanmean(consistency_scores)
    
    return overall_consistency

def calculate_topographic_consistency(epochs):
    """
    Calculate topographic consistency as the mean correlation between the scalp maps
    of single trials and the average scalp map of these trials, focusing on spatial patterns at each time point.
    
    Parameters:
    - epochs: MNE Epochs object
    
    Returns:
    - topographic_consistency: The mean correlation coefficient representing the topographic consistency.
    """
    # Extract data from epochs
    try:
        data = epochs.get_data('csd', tmin=0, tmax=0.2)*1e3 
        
        # Compute the grand average map across all epochs
        # This should be a 2D array: (n_channels, n_times) representing the grand average across all epochs
        grand_average_map = epochs.average().get_data('csd', tmin=0, tmax=0.2)*1e3
        
    except:
        data = epochs.get_data('eeg', tmin=0, tmax=0.2)*1e6 
        
        # Compute the grand average map across all epochs
        # This should be a 2D array: (n_channels, n_times) representing the grand average across all epochs
        grand_average_map = epochs.average().get_data('eeg', tmin=0, tmax=0.2)*1e6
    
    # Initialize a list to hold the mean correlation for each epoch
    epoch_correlations = []
    
    # Iterate over each epoch to calculate its topographic consistency
    for epoch_map in data:
        # Initialize a list to hold correlations for each time point in the epoch
        timepoint_correlations = []
        
        # Iterate over each time point to calculate the correlation across all channels
        for timepoint in range(epoch_map.shape[1]):
            # Correlation of the spatial pattern at this time point
            corr = np.corrcoef(epoch_map[:, timepoint], grand_average_map[:, timepoint])[0, 1]
            timepoint_correlations.append(corr)
        
        # The mean correlation across all time points for this epoch represents its topographic consistency
        epoch_correlations.append(np.mean(timepoint_correlations))
    
    # The overall topographic consistency is the mean of the topographic consistencies of all epochs
    topographic_consistency = np.mean(epoch_correlations)
    
    return topographic_consistency

def calculate_gfp_strength(epochs, method='trial_avg'):
    if method=='evoked_avg':
        # Calculate the evoked (grand-mean) average across all trials and its GFP/RMS
        try:
            D = epochs.average('csd').data
            evoked_average_gfp = np.linalg.norm(D, axis=0) / np.sqrt(len(D))*1e3
        except:
            D = epochs.average('eeg').data
            evoked_average_gfp = D.std(axis=0, ddof=0)*1e6
        evoked_average_gfp = evoked_average_gfp[epochs.time_as_index(0)[0]:epochs.time_as_index(0.2)[0]]
    
    elif method=='trial_avg':
        # Extract data from epochs
        try:
            data = epochs.get_data('eeg')*1e6  # Shape: (n_epochs, n_channels, n_times)
            # Calculate GFP for each trial
            trial_gfps = np.std(data, axis=1, ddof=0)  # Shape: (n_epochs, n_times)
        except:
            data = epochs.get_data('csd')*1e3
            # Calculate RMS for each trial
            trial_gfps = np.linalg.norm(data, axis=1) / np.sqrt(len(data))
        evoked_average_gfp = trial_gfps[:, epochs.time_as_index(0)[0]:epochs.time_as_index(0.2)[0]]
    
    return np.nanmean(evoked_average_gfp)
