#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:49:13 2020

@author: administrator
"""


import yasa
import numpy as np
import pandas as pd
import seaborn as sns
from wonambi.detect.spindle import (DetectSpindle, detect_Lacourse2018, detect_Wamsley2012, 
                                    detect_Nir2011, detect_Ferrarelli2007, detect_Moelle2011)
import mne
import matplotlib.pyplot as plt
import os
from scipy.stats import skewnorm
from scipy.linalg import eigh
from scipy.interpolate import RectBivariateSpline
from scipy.signal import find_peaks, welch, detrend
from inerlinc.plugins.eeg.source import surface_laplacian
from sleep_funs import load_preprocessed_data, Data_Struct
sns.set(context='notebook', font_scale=1.3)

#%% Spindle detection


# =============================================================================
# The YASA spindles algorithm is largely inspired by the A7 algorithm described in Lacourse et al. 2018:
# 
# Lacourse, K., Delfrate, J., Beaudry, J., Peppard, P., Warby, S.C., 2018. A sleep spindle detection algorithm 
# that emulates human expert spindle scoring. J. Neurosci. Methods. https://doi.org/10.1016/j.jneumeth.2018.08.014
# 
# The main idea of the algorithm is to compute different thresholds from the broadband-filtered signal (1 to 30 Hz) and  
# the sigma-filtered signal (11 to 16 Hz).
# 
# There are some notable exceptions between YASA and the A7 algorithm:
# 
#     1. YASA uses 3 different thresholds (relative sigma power, root mean square and correlation). 
#        The A7 algorithm uses 4 thresholds (absolute and relative sigma power, covariance and correlation). 
#        Note that it is possible in YASA to disable one or more threshold by putting None instead.
#     2. The windowed detection signals are resampled to the original time vector of the data using cubic interpolation, 
#        thus resulting in a pointwise detection signal (= one value at every sample). 
#        The time resolution of YASA is therefore higher than the A7 algorithm. 
#        This allows for more precision to detect the beginning, end and durations of the spindles
#        (typically, A7 = 100 ms and YASA = 10 ms).
#     3. The relative power in the sigma band is computed using a Short-Term Fourier Transform. 
#        The relative sigma power is not z-scored.
#     4. The median frequency and absolute power of each spindle is computed using an Hilbert transform.
#     5. YASA computes some additional spindles properties, such as the symmetry index and number of oscillations. 
#        These metrics are inspired from Purcell et al. 2017.
#     6. Potential sleep spindles are discarded if their duration is below 0.5 seconds and above 2 seconds. 
#        These values are respectively 0.3 and 2.5 seconds in the A7 algorithm.
#     7. YASA incorporates an automatic rejection of pseudo or fake events based on an Isolation Forest algorithm.
# 
# 
# =============================================================================

path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded_cleaned/'    
hypno_path= '/media/administrator/data/Study_1_data/Hypnograms/Unblinded_clean/'

def Data_spindle(path, hypno_path):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    hypno_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(hypno_path) for i in files])
    for i, (data_files, hypno_files) in enumerate(zip(files_list, hypno_list)):
        Data = load_preprocessed_data(data_files)
        eeg_index = [i for i, x in enumerate(Data.chtypes) if x == "eeg"]
        
        ## Method 1 - Spindles-based slow/fast peak detection
        # =============================================================================
        #       1. Run the multi-channel detection with the default parameters
        #       2. Plot the distribution of spindles frequency for the frontal and parietal channels separately
        #       3. Fit a Gaussian curve to the distributions
        #       4. Identifity the peaks of the Gaussian distributions
        # =============================================================================
        
        # Run the detection on all 9 channels (should take 20 to 60 seconds)
        # Notice how we use .summary() at the end to directly get the full detection dataframe
        sp = yasa.spindles_detect(data = Data.data[:,eeg_index].T, sf = Data.sfreq, ch_names = np.asarray(Data.chans[0:len(eeg_index)]), 
                                  hypno= Data.hypno_with_art, include=(2,3), freq_sp=(10, 16), freq_broad=(1, 30), duration=(0.5, 2), 
                                  min_distance=500, thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5}, multi_only=False, remove_outliers=True, verbose=True)
        summary = sp.summary()
        print('%i spindles detected on %i channels.' % (summary.shape[0], len(eeg_index)))
        # summary.head().round(3)
        
        # Plot the spindles frequency distribution
        plt.figure(figsize=(10, 6))
        sns.distplot(summary.loc[summary['Channel'].isin(['Fz', 'F3', 'F4']), 'Frequency'], kde=False, fit=skewnorm, label='Frontal - Slow')
        sns.distplot(summary.loc[summary['Channel'].isin(['Cz', 'C3', 'C4']), 'Frequency'], kde=False, fit=skewnorm, label='Central - Fast')
        # sns.distplot(summary.loc[summary['Channel'].isin(['P3', 'P4']), 'Frequency'], kde=False, fit=skewnorm, label='Parietal - Fast')
        plt.legend()
        plt.title('Method 1: slow and fast spindles peak frequencies')
        plt.xlabel('Frequency (Hz)')
        _ = plt.ylabel('Density')
        
        # Extract the fit parameters
        _, slow_mu, slow_std = skewnorm.fit(summary.loc[summary['Channel'].isin(['Fz', 'F3', 'F4']), 'Frequency'])
        # _, fast_mu, fast_std = skewnorm.fit(summary.loc[summary['Channel'].isin(['P3', 'P4']), 'Frequency'])
        _, fast_mu, fast_std = skewnorm.fit(summary.loc[summary['Channel'].isin(['Cz', 'C3', 'C4']), 'Frequency'])

        print("Slow spindles (frontal) have a mean frequency of %.2f Hz and a standard deviation of %.2f Hz" % (slow_mu, slow_std))
        # print("Fast spindles (parietal) have a mean frequency of %.2f Hz and a standard deviation of %.2f Hz" % (fast_mu, fast_std))
        print("Fast spindles (central) have a mean frequency of %.2f Hz and a standard deviation of %.2f Hz" % (fast_mu, fast_std))
        
        
        ## Flip data for next 2 methods:
        data = Data.data.T    
        ## Method 2 - N2 sleep power spectrum peak detection  
        # =============================================================================
        #        1. Compute the power spectrum of N2 sleep for each channel
        #        2. Identify the slow and fast peaks in the power spectrum
        #        3. Run the spindles detection with the appropriate parameters
        # =============================================================================

        # Compute Welch spectrum of N2 sleep for each channel
        f, pxx = welch(data[np.expand_dims(np.asarray(eeg_index),1), Data.hypno_with_art == 2], fs=Data.sfreq, nperseg=(5 * Data.sfreq))
        
        # Convert to dB to reduce 1/f (optional)
        pxx = 10 * np.log10(pxx)
        
        # Keep only frequencies of interest
        pxx = pxx[:, np.logical_and(f >= 10, f <= 16)]
        f = f[np.logical_and(f >= 10, f <= 16)]
        
        # Plot average spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(f, pxx.mean(0), 'ko-', lw=3)
        plt.plot(f, np.rollaxis(pxx, axis=1), lw=1.5, ls=':', color='grey')
        plt.xlim(10, 16)
        plt.title('Method 2: channel-based power spectrum')
        plt.xlabel('Frequency (Hz)')
        _ = plt.ylabel('Power (dB)')
        
        # Identify the two peaks in the power spectrum
        idx_peaks, _ = find_peaks(pxx.mean(0))
        print('Slow spindles peak frequency = %.2f Hz' % f[idx_peaks[0]])
        print('Fast spindles peak frequency = %.2f Hz' % f[idx_peaks[1]])
        
        
        ## Method 3 - Topographhy based slow/fast spindle detection 
        # =============================================================================
        #    a1. Bandpass filter data
        #    a2. Apply surface Laplcian filter to raw data
        #    1. Get the slow sigma-filtered data and fast sigma-filtered data
        #    2. Detrend and compute the covariance matrices
        #    3. Apply the generalized eigendecomposition to find eigenvectors that maximally differentiate 
        #       between slow and fast spindles
        #    4. Use the eigenvectors as spatial filters
        #    5. Compute the Welch spectrum of N2 sleep on all the spatially filtered data (= components)
        #    6. Find slow and fast peak frequencies by averaging the power spectrum of n components that 
        #       are the most associated with slow and fast spindles, respectively.
        # =============================================================================
        
        # ------------ APPLY SURFACE LAPLACIAN HERE ------------------ # 
        
        # Get filtered slow (9 - 12 Hz) and fast (12 - 16 Hz) data
        slow_n2_filt = mne.filter.filter_data(data[Data.hypno_with_art == 2, np.expand_dims(np.asarray(eeg_index),1)].astype(np.float64), Data.sfreq, 9, 12, 
                                              h_trans_bandwidth=1, l_trans_bandwidth=1, verbose=0)
        fast_n2_filt = mne.filter.filter_data(data[Data.hypno_with_art == 2, np.expand_dims(np.asarray(eeg_index),1)].astype(np.float64), Data.sfreq, 12, 16, 
                                              h_trans_bandwidth=1, l_trans_bandwidth=1, verbose=0)
        
        # Remove the mean (= detrend)
        slow_n2_filt = detrend(slow_n2_filt, type='constant')
        fast_n2_filt = detrend(fast_n2_filt, type='constant')
        
        # Compute the covariance matrices between channels
        slow_n2_cov = np.cov(slow_n2_filt)
        fast_n2_cov = np.cov(fast_n2_filt)
        
        # Plot the slow covariance matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(fast_n2_cov, cmap='Blues', square=True, 
                    xticklabels=ch_names[0:len(eeg_index)], yticklabels=ch_names[0:len(eeg_index)])
        plt.title('Fast sigma variance-covariance matrix')
        plt.xlabel('Channels')
        _ = plt.ylabel('Channels')
        
        # Get the eigenvalues / eigenvectors
        eigval, eigvec = eigh(slow_n2_cov, fast_n2_cov)
        
        # Flip to descending order
        eigval = np.flip(eigval)
        eigvec = np.fliplr(eigvec)
        
        print('Eigenvalues =', list(np.round(eigval, 2)))
        
        # Apply spatial filters by multiplying data with eigenvectors
        # sf_comp_n2 = np.dot(data[:, hypno == 2].T, eigvec).T
        sf_comp_n2 = np.dot(Data.data[Data.hypno_with_art == 2, np.expand_dims(np.asarray(eeg_index),1)].T, eigvec)
        print(sf_comp_n2.shape)
        
        # Compute Welch spectrum of N2 sleep
        f, pxx = welch(sf_comp_n2, Data.sfreq, nperseg=(5 * Data.sfreq))
        
        pxx = pxx[:, np.logical_and(f >= 10, f <= 16)]
        f = f[np.logical_and(f >= 10, f <= 16)]
        
        # Convert to dB to reduce 1/f (optional)
        pxx = 10 * np.log10(pxx)
        
        # Select the number of components to keep
        n_comp = 2
        
        # Plot first slow component
        plt.figure(figsize=(10, 6))
        plt.plot(f, np.rollaxis(pxx[:n_comp], axis=1), lw=1, ls=':', color='blue')
        plt.plot(f, np.rollaxis(pxx[-n_comp:], axis=1), lw=1, ls=':', color='red')
        
        # Plot average spectrum of four first / last components
        plt.plot(f, pxx[:n_comp].mean(0), 'bo-', lw=3)
        plt.plot(f, pxx[-n_comp:].mean(0), 'ro-', lw=3)
        _ = plt.xlim(10, 16)

        # Identify the peaks
        print('Slow spindles peak frequency: %.2f Hz' % f[pxx[:n_comp].mean(0).argmax()])
        print('Fast spindles peak frequency: %.2f Hz' % f[pxx[-n_comp:].mean(0).argmax()])

        # Running the tuned spindles detection
        sp_slow = yasa.spindles_detect(data, sf, ch_names, freq_sp=(10.20, 12.20)).summary()
        sp_fast = yasa.spindles_detect(data, sf, ch_names, freq_sp=(11.60, 13.60)).summary()
        print('%i spindles detected on %i channels.' % (sp_slow.shape[0], len(ch_names)))
        print('%i spindles detected on %i channels.' % (sp_fast.shape[0], len(ch_names)))
        sp_slow.head().round(3)
        
        # =============================================================================
        # Summary
        # - Method 2 and method 3 yielded exactly the same results in this case (i.e. 11.20 Hz for slow spindles and 12.60 Hz for fast spindles).
        # - Method 1 returned slightly different results. This is probably caused by the fact that the frontal channels do not have exclusively 
        #   slow spindles and the parietal channels exclusively fast spindles. Rather, there must be some level of contamination, which explains 
        #   this "regression to the mean" effect that we observe in the two peaks detected with method 1.
        # - Method 3 is probably the most accurate, granted that you have enough clean and artefact-free channels.
        #   Note that in the original paper by Cox and colleagues, a surface Laplacian filter is also applied on the raw data to enhance spatial precision for topographical analyses.
        # =============================================================================
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        