#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:32:21 2021

@author: administrator
"""

import scipy
import numpy as np 
from yasa.others import _zerocrossings
from sleepstim.Analysis.analysis_pre_process import (load_preprocessed_data, Data_Struct, unravel_hypnogram_visbrain)
import yasa
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict

# load only fully processed, unblinded data
Data = load_preprocessed_data('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/YIOYSRPX_sham.p')
Data.fir_filter(l_freq=None, h_freq=2)

#%%

peaks, _ = scipy.signal.find_peaks(x=Data.data[Data.hypno_with_art==3, Data.chans.index('C3')],  
                                              distance = int(Data.sfreq*0.25),prominence=(None,None))
neg_peaks, _ = scipy.signal.find_peaks(x=-Data.data[Data.hypno_with_art==3,Data.chans.index('C3')],
                                                    distance = int(Data.sfreq*0.25),prominence=(None,None))

peak_thres = np.percentile(Data.data[Data.hypno_with_art==3, Data.chans.index('C3')][peaks], 75)
neg_peak_thres = np.percentile(Data.data[Data.hypno_with_art==3, Data.chans.index('C3')][neg_peaks], 25)

sw_peaks_bool = Data.data[Data.hypno_with_art==3, Data.chans.index('C3')][peaks] >= peak_thres
neg_sw_peaks_bool = Data.data[Data.hypno_with_art==3, Data.chans.index('C3')][neg_peaks] <= neg_peak_thres

sw_peaks_idx = peaks[sw_peaks_bool]
neg_sw_peaks_idx = neg_peaks[neg_sw_peaks_bool]
zerocrossing_idx = _zerocrossings(Data.data[Data.hypno_with_art==3, Data.chans.index('C3')])

#%%Reduce the peaks to make sure they conform to slow wave zerocrossing
def peak_reduction(peaks,zerocrossings,sf=512, l_freq=0.5):
    """
    
    Parameters
    ----------
    peaks : 1-d array
        Array with indices of detected peaks
    zerocrossings : 1-d array
        Array with indices of zerocrossings
    Returns
    -------
    new_peaks: 1-d array
        peaks that have zerocrossings before and after them that are no more than a specific timespan away (half-wave of specific user-determined frequency)
    """
    new_peaks = []
    neg_sorted = np.searchsorted(zerocrossings, peaks)
    previous_zc = zerocrossings[neg_sorted - 1] - peaks
    following_zc = zerocrossings[neg_sorted] - peaks
    for i,peak in enumerate(peaks):
        if abs(previous_zc[i]) < (sf/(l_freq*2)) and abs(following_zc[i]) < (sf/(l_freq*2)) :
            new_peaks = np.append(new_peaks,np.asarray(peaks[i]))
    return new_peaks.astype(int)

#%%

peaks_reduced = peak_reduction(sw_peaks_idx, zerocrossing_idx)
neg_peaks_reduced = peak_reduction(neg_sw_peaks_idx, zerocrossing_idx) 
# Make sure that the last detected peak is a positive one
if peaks_reduced[-1] < neg_peaks_reduced[-1]:
    # If not, append a fake positive peak one sample after the last neg
    peaks_reduced = np.append(peaks_reduced, neg_peaks_reduced[-1] + 1)
           
pk_sorted = np.searchsorted(peaks_reduced, neg_peaks_reduced)
closest_pos_peaks = peaks_reduced[pk_sorted] - neg_peaks_reduced
closest_pos_peaks = closest_pos_peaks[np.nonzero(closest_pos_peaks)]
peaks_reduced = neg_peaks_reduced + closest_pos_peaks

# Now we compute the PTP amplitude and keep only the good peaks
sw_ptp = (np.abs(Data.data[Data.hypno_with_art==3, Data.chans.index('C3')][neg_peaks_reduced]) + 
          Data.data[Data.hypno_with_art==3, Data.chans.index('C3')][peaks_reduced])
good_ptp = sw_ptp >= (peak_thres - neg_peak_thres)

sw_ptp = sw_ptp[good_ptp]
idx_neg_peaks = neg_peaks_reduced[good_ptp]
idx_pos_peaks = peaks_reduced[good_ptp]

# Make sure that there is a zero-crossing after the last detected peak
if zerocrossing_idx[-1] < max(idx_pos_peaks[-1], idx_neg_peaks[-1]):
    # If not, append the index of the last peak
    zerocrossing_idx = np.append(zerocrossing_idx, max(idx_pos_peaks[-1], idx_neg_peaks[-1]))

# Find distance to previous and following zc
neg_sorted = np.searchsorted(zerocrossing_idx, idx_neg_peaks)
previous_neg_zc = zerocrossing_idx[neg_sorted - 1] - idx_neg_peaks
following_neg_zc = zerocrossing_idx[neg_sorted] - idx_neg_peaks

# Distance between the positive peaks and the previous and
# following zero-crossings
pos_sorted = np.searchsorted(zerocrossing_idx, idx_pos_peaks)
previous_pos_zc = zerocrossing_idx[pos_sorted - 1] - idx_pos_peaks
following_pos_zc = zerocrossing_idx[pos_sorted] - idx_pos_peaks

# Duration of the negative and positive phases, in seconds
neg_phase_dur = (np.abs(previous_neg_zc) + following_neg_zc) / Data.sfreq
pos_phase_dur = (np.abs(previous_pos_zc) + following_pos_zc) / Data.sfreq

# We now compute a set of metrics
sw_start = Data.times[idx_neg_peaks + previous_neg_zc]
sw_end = Data.times[idx_pos_peaks + following_pos_zc]
# This should be the same as `sw_dur = pos_phase_dur + neg_phase_dur`
# We round to avoid floating point errr (e.g. 1.9000000002)
sw_dur = (sw_end - sw_start).round(4)
sw_dur_both_phase = (pos_phase_dur + neg_phase_dur).round(4)
sw_midcrossing = Data.times[idx_neg_peaks + following_neg_zc]
sw_idx_neg = Data.times[idx_neg_peaks]  # Location of negative peak
sw_idx_pos = Data.times[idx_pos_peaks]  # Location of positive peak
# Slope between peak trough and midcrossing
sw_slope = sw_ptp / (sw_midcrossing - sw_idx_neg)
# Hypnogram
sw_sta = Data.hypno_with_art[Data.hypno_with_art==3][idx_neg_peaks]
# And we apply a set of thresholds to remove bad slow waves
good_sw = np.logical_and.reduce((
    # Data edges
    previous_neg_zc != 0,
    following_neg_zc != 0,
    previous_pos_zc != 0,
    following_pos_zc != 0,
    # Duration criteria
    sw_dur == sw_dur_both_phase,  # dur = negative + positive
    sw_dur <= 1 + 1,  # dur < max(neg) + max(pos)
    sw_dur >= 0.25 + .1,  # dur > min(neg) + min(pos)
    neg_phase_dur > .25,
    neg_phase_dur < 1,
    pos_phase_dur > .1,
    pos_phase_dur < 1,
    # Sanity checks
    sw_midcrossing > sw_start,
    sw_midcrossing < sw_end,
    sw_slope > 0,
    ))

# Filter good events
idx_neg_peaks = idx_neg_peaks[good_sw]
idx_pos_peaks = idx_pos_peaks[good_sw]
sw_start = sw_start[good_sw]
sw_idx_neg = sw_idx_neg[good_sw]
sw_midcrossing = sw_midcrossing[good_sw]
sw_idx_pos = sw_idx_pos[good_sw]
sw_end = sw_end[good_sw]
sw_dur = sw_dur[good_sw]
sw_ptp = sw_ptp[good_sw]
sw_slope = sw_slope[good_sw]
sw_sta = sw_sta[good_sw]

# Create a dictionnary
sw_params = OrderedDict({
    'Start': sw_start,
    'NegPeak': sw_idx_neg,
    'MidCrossing': sw_midcrossing,
    'PosPeak': sw_idx_pos,
    'End': sw_end,
    'Duration': sw_dur,
    'ValNegPeak': Data.data[Data.hypno_with_art==3,Data.chans.index('C3')][idx_neg_peaks],
    'ValPosPeak': Data.data[Data.hypno_with_art==3,Data.chans.index('C3')][idx_pos_peaks],
    'PTP': sw_ptp,
    'Slope': sw_slope,
    'Frequency': 1 / sw_dur,
    'Stage': sw_sta,
    })

sw = yasa.sw_detect(data = Data.data[:, Data.chans.index('C3')], sf=Data.sfreq, ch_names=['C3'],
                    hypno = Data.hypno_with_art, include=(3), freq_sw=(0.5, 1.5), dur_neg=(0.3, 1.5), 
                    dur_pos=(0.1, 1), amp_neg=(None, 300), amp_pos=(None, 200), amp_ptp=(75, 500),
                    coupling=False, freq_sp=(12, 16), remove_outliers=True, verbose=False)

summary = sw.summary().round(2)
