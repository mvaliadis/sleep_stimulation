#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:56:27 2023

@author: administrator
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sleepstim.sleep_funs import load_xdf, channel_parser
import yasa
import mne
from numba import jit, prange
#from scipy.signal import find_peaks, welch, detrend
import scipy.stats as stats
#import fooof
import pandas as pd
#from scipy.linalg import eigh, eig
#import itertools
#import os
#import glob
#from tqdm import tqdm
import seaborn as sns
#import neurokit2 as nk 
import pingouin as pg
mne.set_log_level("CRITICAL")
sns.set(style='darkgrid', font_scale=1.2)

def subject_cond_parser(file, night, sub):
    # data sheet with true subject condition nights
    sheet = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
    subj_cond = np.loadtxt(sheet, delimiter=',', dtype='str', skiprows=1) 
    cond_dict = {0:'sham', 1:'up', 2:'down'}
    # find condition by night
    sc = sub
    index_name = list(subj_cond[:,0]).index(sc)
    cond = int(subj_cond[index_name,1::][int(night)-1])
    # true condition night name
    condition_night = cond_dict[cond]  
    
    return condition_night

def re_reference(data, ch_names, trans_csd=None,
                 sf=512, reference='common average'):
    """EEG data re_referencing function.

    This function takes a numpy array of the data & rereferences the data based on
    the new selected reference. 

    Parameters
    ----------
    data : numpy array of [n_samples x n_chans]
        The data. Include only EEG channels for this argument!
        
    reference : str {‘common average’, ‘mastoids’}
        The new reference; default is ‘common average’.
        
    Returns
    -------
    data : numpy array of shape [n_samples x n_chans]
        The re-referenced data
    """
    # %timeit = 97 microseconds for 30s of data (fs = 256 Hz)
    # Initial safety check to confirm data shape is [n_samples x n_chans]
    dpnts, chans = data.shape
    if chans > dpnts:
        data = np.transpose(data)
    if reference == 'common average':
        # Compute mean of each epoch per channel then subtract common average
        ref_data = data[..., :].mean(-1, keepdims=True)
        data -= ref_data
        #print('Data will be re-referenced to the common average of all selected EEG electrodes!')
    elif reference == 'mastoids':
        # May change to name indexed instead of numerical index...
        ref_data = data[..., [ch_names.index('M1'), ch_names.index('M2')]].mean(-1, keepdims=True)
        data -= ref_data
        #print('Data will be re-referenced to the average of the mastoids!')
    elif reference == 'csd':
        data = surface_laplacian_rt(data=data, trans_csd=trans_csd) #np.expand_dims(d,1))
    else:
        raise ValueError('Please select a valid reference!')
    
    return data

def surface_laplacian_rt(data, trans_csd):
    #epochs = inst._data
    epochs = np.expand_dims(data.T, 0)
    for epo in epochs:
        csd_data = np.dot(trans_csd, epo)
    
    return csd_data

@jit(nopython=True, parallel=True)
def calculate_metrics(eeg):
    num_channels = eeg.shape[0]
    channel_metrics = np.zeros((num_channels, 2))
 
    # Note the use of prange here instead of range
    for i in prange(num_channels):
        channel = eeg[i, :]
        channel_metrics[i, 0] = np.nanstd(channel)
        channel_metrics[i, 1] = np.nanmean(channel)

    return channel_metrics

def fast_eeg_badchannels(eeg, bad_threshold=0.5, distance_threshold=0.99):
    """
    Find bad channels in a more efficient manner by focusing on the fastest metrics.

    Parameters:
    - eeg : np.ndarray, the EEG data (channels x time points)
    - bad_threshold : float, the proportion criteria for a bad channel
    - distance_threshold : float, the z-score threshold for outlier detection

    Returns:
    - bads : list, indices of bad channels
    """
    # Check input type
    if not isinstance(eeg, np.ndarray):
        raise ValueError("Input 'eeg' must be a numpy ndarray.")

    # Calculate metrics
    channel_metrics = calculate_metrics(eeg)

    # Standardization (z-score calculation)
    means = np.mean(channel_metrics, axis=0)
    stds = np.std(channel_metrics, axis=0, ddof=1)  # ddof=1 to calculate the sample standard deviation
    z = np.abs((channel_metrics - means) / stds)  # broadcasting

    # Outlier detection based on z-scores
    outlier_counts = np.sum(z > stats.norm.ppf(distance_threshold), axis=1)
    bad_channels = np.where(outlier_counts >= np.ceil(bad_threshold * z.shape[1]))[0]

    return list(bad_channels)

def robust_scaling(eeg_data):
    """
    Apply robust scaling (using median and MAD) to the multichannel EEG data.
    Each column in the eeg_data should represent a different channel.
    """
    median = np.median(eeg_data, axis=np.argmax(eeg_data.shape))  # Median per channel
    mad = np.median(np.abs(eeg_data - median), 
                    axis=np.argmin(median.shape))  # MAD per channel
    
    # Avoid division by zero by replacing MAD = 0 with 1
    mad[mad == 0] = 1
    
    scaled_eeg_data = (eeg_data - median) / mad
    return scaled_eeg_data

#%%
# 1. Load pre-recorded dataset

file = '/media/administrator/data/Study_1_data/Raw_data/Experimental/886MCPKG_1/sleepstim_R001.xdf'
stream_dict = load_xdf(file) 

# extract info from file 
sub = file.split('/')[-2].split('_')[0]
rec = file.split('/')[-2].split('_')[1]
night = subject_cond_parser(file, rec, sub)
             
# extract relevant channel data  
info = stream_dict['eego']['info'] 
sf = int(info['nominal_srate'][0])
data = stream_dict['eego']['time_series']
times = stream_dict['eego']['time_stamps'][0:sf*60*220]
ch_names, ch_types = channel_parser(info, data) 
idx = [x for i,x in enumerate(range(0,72)) if i!=ch_names.index('EOG')]
data = data[0:sf*60*220, idx][:,0:64]*1e6
ch_names = list(np.asarray(ch_names)[idx][0:64])
marker = stream_dict['reiz-marker']
pinknoise_times = [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if marker['time_series'][ix][0] == 'pinknoise'] - times[0]
times -= times[0]
if np.all(np.diff(pinknoise_times[0:3]) < 0.5):
    pinknoise_times_sync = [np.argmin(np.abs(times - ts)) for ts in pinknoise_times[3::]] 
else:
    pinknoise_times_sync = [np.argmin(np.abs(times - ts)) for ts in pinknoise_times] 
del stream_dict

# start data at first auditory burst
data = data[pinknoise_times_sync[0]:,:] # start recording at first stimulus

# # mne-ize for mne_lsl testing
# mne_info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='eeg')
# mne_data = mne.io.RawArray(data.T/1e6, mne_info)
# mne_data.set_montage(mne.channels.make_standard_montage('standard_1005'))
# mne_data.resample(100)
# mne_data.save('/media/administrator/data/Study_2_data/mne_lsl_data/test-raw.fif')

#%%
# 2. Load CSD matrix

csd_file_path = '/home/administrator/sleep_stimulation/sleepstim/Analysis/CSD_matrix.npy'
trans_csd = np.load(csd_file_path)

#%%

# 3. Imitate sliding-window procedure of original experiment

# Constants
ch_names = ch_names
reference = 'mastoids'#'csd'
fs = sf  # Sample rate
new_times = times[pinknoise_times_sync[0]:]  # Adjusting the slice to start from the sync point
# Time shift value (e.g., 475 milliseconds)
time_shift = 0.475  # time shift in seconds
threshold = True
threshold_val = -35
# Pre-filter C3 data
C3 = data[:, ch_names.index('C3')]
#ch_idx = [ch_names.index('C3'), ch_names.index('M1'), ch_names.index('M2')]
data_filt = mne.filter.filter_data(data.astype('float64').T, sfreq=sf, 
                                   l_freq=.4, h_freq=30, method='fir', n_jobs=-1).T
# ref_data = data_filt[..., [-2:-1]].mean(-1, keepdims=True)
# data_filt -= ref_data
C3_filt = re_reference(data_filt, ch_names, reference='mastoids')[:, ch_names.index('C3')]

bandpass = False  # This toggle seems to switch between two different filtering methods

# Filters for real-time application
#sos_lp = signal.butter(2, 4, fs = sf, btype='lowpass', output="sos")  # Low-pass filter
b, a = signal.butter(2, 4, fs = sf, btype='lowpass')  # Low-pass filter
d, c = signal.butter(2, (0.3, 30), fs=fs, btype='bandpass')  # Band-pass filter

#%%
# Result storage
crit_reconstruct = []
time_reconstruct = []
crit_reconstruct_up = []
time_reconstruct_up = []

# Window definitions
extraction_window = 10 * fs  # e.g., last 10 seconds of data
slide_interval = 2 * fs  # e.g., move window every 2 seconds

# Main loop
ix = 0
while ix <= len(C3) and len(crit_reconstruct) <= 100:
    # Calculate the current window's start and end, ensuring it doesn't go below zero
    window_end = ix
    window_start = max(0, ix - extraction_window)
    window = data[int(window_start):int(window_end),:]
    
    #ix += slide_interval # remove 

    if len(window) == extraction_window:
        # Apply the chosen filtering technique
        filtered_data = signal.filtfilt(d, c, window, axis=0) if bandpass else signal.filtfilt(b, a, window, axis=0)
        if not bandpass:
            filtered_data -= np.median(filtered_data, axis=0)  # Removing median if low-pass filter is applied
            #filtered_data_rs = robust_scaling(filtered_data) # with MAD -> changes unit scale
        
        # Remove bad channels
        #bads = nk.eeg_badchannels(filtered_data.T)
        
        # Apply spline interpolation
        #filtered_data = online_interpolation(filtered_data)
        
        # Re-reference data 
        if reference == 'mastoids':
            filtered_data_final = re_reference(filtered_data, ch_names, 
                                               reference='mastoids')
        elif reference == 'csd':
            filtered_data_final = re_reference(data=filtered_data/1e3, ch_names=ch_names,
                                               trans_csd=trans_csd, sf=sf, reference='csd').T
         
        # Extract target signal (C3)
        chan_data = filtered_data_final[:, ch_names.index('C3')]
        
        # Criteria check
        if threshold:
            minamp = min(np.percentile(chan_data[-2 * int(sf):], 10), threshold_val)
        else:
            minamp = np.percentile(chan_data[int(sf*-.04):].mean(0), 10)
    
        # Get critical value
        crit = min(chan_data[int(-0.02 * sf):])

        # Check the criterion and time distance
        if crit < minamp and (new_times[ix] - (time_reconstruct[-1] if time_reconstruct else float('-inf')) > 2.99):
            if np.ptp(chan_data) < 600:
                # Store the results
                current_time = new_times[ix]
                time_reconstruct.append(current_time)
                crit_reconstruct.append(crit)
    
                # Calculate the new time and corresponding index for filtered_data
                ts_up = current_time + time_shift  # shifted timestamp
                index_shift = int(time_shift * fs)  # converting time shift to index shift
                index_for_ts_up = window_start + extraction_window - 1 + index_shift  # index in the complete data
    
                # Ensure the index is within the bounds of the complete dataset
                if 0 <= index_for_ts_up < len(C3):
                    amplitude_at_ts_up = C3_filt[index_for_ts_up]  # amplitude at the shifted time
                    crit_reconstruct_up.append(amplitude_at_ts_up)
                    time_reconstruct_up.append(ts_up)  # Storing the time corresponding to ts_up
                           
    ix += slide_interval  # Moving the analysis window forward


#%%
## Plot results
plt.figure()
plt.plot(new_times, C3_filt)
plt.plot(time_reconstruct[1::], crit_reconstruct[1::], 'xr')
plt.plot(time_reconstruct_up[1::], crit_reconstruct_up[1::], 'xg')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Filtered Data')
plt.show()

#%%
# Correcting the approach to handle individual timestamps
crit_true_idx = [int((time - new_times[0]) * fs) for time in time_reconstruct]
crit_true = C3_filt[crit_true_idx]

from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(crit_true, np.asarray(crit_reconstruct))
print(f'MAPE: {mape}')

#%%
# Create a DataFrame
df = pd.DataFrame({
    'True Voltage': crit_true,
    'Reconstructed Voltage': crit_reconstruct,
    'Subject': [f'Subject_{i}' for i, _ in enumerate(crit_true)]
})
df_melted = df.melt(id_vars='Subject', var_name='Type', value_name='Critical Voltage')
pg.plot_paired(df_melted, dv='Critical Voltage',
               within='Type', subject='Subject', boxplot_in_front=True)

#%%
reject_criteria = dict(eeg=550e-6)
ts_pinknoise_times_sync = [np.argmin(np.abs(new_times - ts)) for ts in time_reconstruct]
center_crit, _ = yasa.get_centered_indices(data[:, ch_names.index('C3')], 
                                            np.asarray(ts_pinknoise_times_sync[1::]), 
                                            npts_before = sf*2, npts_after = sf*2)
info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='eeg')
epochs_crit = mne.EpochsArray(np.swapaxes(data[center_crit], 1, 2)/1e6, 
                              info, tmin = -2, baseline=(-2, -1), proj=False)
epochs_crit.filter(0.5, 2)
epochs_crit.set_eeg_reference(['M1','M2'])
epochs_crit.drop_bad(reject = reject_criteria) 
epochs_crit.set_montage(mne.channels.make_standard_montage('standard_1005'))
epochs_crit.plot_image('C3') 
epochs_crit.average().plot_joint()
mne.preprocessing.compute_current_source_density(epochs_crit).average().plot_joint()

#%%
## Epoch plotting with MNE
# down
reject_criteria = dict(eeg=550e-6)
ts_pinknoise_times_sync = [np.argmin(np.abs(new_times - ts)) for ts in time_reconstruct]
ts_pinknoise_times_sync_up = [np.argmin(np.abs(new_times - ts)) for ts in time_reconstruct_up]

center_crit, _ = yasa.get_centered_indices(C3_filt_notch_broad, np.asarray(ts_pinknoise_times_sync[1::]), 
                                            npts_before = sf*4, npts_after = sf*4)
info = mne.create_info(ch_names=['C3'], sfreq=sf, ch_types='eeg')
epochs_crit = mne.EpochsArray(np.expand_dims(C3_filt_notch_broad[center_crit], 1)/1e6, info, tmin = -4, 
                              baseline=(-4, -1.5), proj=False)
art_idx = art_detect(epochs_crit)
epochs_crit.drop(art_idx)
epochs_crit.drop_bad(reject = reject_criteria) 
# epochs_crit.average(method='mean').plot()
epochs_crit.plot_image()

# up
center_up, _ = yasa.get_centered_indices(C3_filt_notch_broad, np.asarray(ts_pinknoise_times_sync_up[1::]), 
                                          npts_before = sf*4, npts_after = sf*4)
info = mne.create_info(ch_names=1, sfreq=sf, ch_types='eeg')
epochs_up = mne.EpochsArray(np.expand_dims(C3_filt_notch_broad[center_up], 1)/1e6, info, tmin = -4, 
                            baseline=(-4, -1.5), proj=False)
art_idx = art_detect(epochs_up)
epochs_up.drop(art_idx)
epochs_up.drop_bad(reject = reject_criteria) 
# epochs_up.average(method='mean').plot()
epochs_up.plot_image()     

#%%
def old_test(data=data, fs=sf, ch_names=ch_names):
    mne.set_log_level("CRITICAL")
    epochs, epochs_csd = [], []
    info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='eeg')
    info_csd = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='csd')
    info_csd.set_montage(mne.channels.make_standard_montage('standard_1005'))
    #crit_reconstruct = [0]; crit_reconstruct_up = [0]
    #time_reconstruct = [0]; time_reconstruct_up = [0]
    fs = 512
    #ix = 2*fs
    b,a = signal.butter(2, 4, fs = fs)
    #while ix <= 512*60*120:
    for ep in neg_peak_idx:
        #ix0 = ix - 2*fs
        #pick 2s window
        #data_wind = data[int(ix0):int(ix),:]
        data_wind = data[ep,:]
        data_wind = signal.filtfilt(b,a,data_wind, axis=0)
        data_wind -= np.median(data_wind, 0)
        #data_wind = scipy.signal.detrend(data_wind, axis=-1, type='constant')
        data_wind = re_reference(data_wind, ch_names, reference='mastoids')
        data_wind_csd = re_reference(data=data_wind/1e3, ch_names=ch_names, 
                                      trans_csd=trans_csd, sf=512, reference='csd').T
        
        # determine 20th percentile of values for threshold of lowest values
        # FLIP BASED ON CONDITION
        percentiles = np.percentile(data_wind_csd[int(sf*-.04):,:].mean(0), 20)
        
        ## Ground truth analysis should probably contain interpolated data
        epoch = mne.EpochsArray(np.expand_dims(data_wind.T, 0)/1e6, 
                                info, tmin = -1.98, baseline=None)
        epoch.set_montage(mne.channels.make_standard_montage('standard_1005'))
        epoch = detect_bad_interpolate(epoch, method='NK')

        # csd epoch
        epoch_csd = mne.preprocessing.compute_current_source_density(epoch, lambda2=1e-03, 
                                                                      verbose=0)
        
        ##
        d = data_wind[:, ch_names.index('C3')]
        #plt.plot(d, 'k')       
        d_csd = data_wind_csd[:, ch_names.index('C3')]        
        #plt.plot(d_csd, 'm')
        #corr = np.corrcoef([d[int(-.5*sf):], d_csd[int(-.5*sf):]]).min()
        #print(f'Correlation between C3 signals in last 500 ms: {corr.round(3)}')
        
        ## Append data
        if d_csd[int(sf*-.04):].mean() < percentiles:
            epochs.append(epoch)
            epochs_csd.append(epoch_csd)
    
    # grand averages (where C3 is in the 80th percentile or higher) 
    gav = mne.grand_average([epochs[i].average() for i in range(len(epochs)-1)])
    gav_csd = mne.grand_average([epochs_csd[i].average() for i in range(len(epochs_csd)-1)])
    gav.plot()
    gav_csd.plot()
        
    # SSD fun
    for name, inst in zip(['lm', 'csd'], [data_wind, data_wind_csd]):
        print(name)
        filters, patterns = compute_ssd(inst.T, signal_bp=(0.3, 2), use_mne=False,
                                        noise_bp=(0.1, 30), noise_bs=(0.1, 30), sf=fs)
        epochs_ssd = apply_filters(inst.T, filters)
        
        # narrowband filter 
        #filtparams = signal.butter(2, 4, fs = sf)        
        #epochs_ssd = signal.filtfilt(*filtparams, epochs_ssd, axis=-1)
        epochs_ssd = signal.filtfilt(b,a, epochs_ssd, axis=-1)
    
        # compute amplitude corrected spatial pattern coefficients
        std_comp = np.std(epochs_ssd, axis=-1)
        weighted_patterns = std_comp * patterns
    
        # spatial complexity
        metric = compute_sensor_complexity(weighted_patterns, 
                                            np.min(weighted_patterns.shape)) #10
        
        # plot spatial complexity
        yasa.topoplot(pd.Series(metric, ch_names), cmap='Spectral_r')
        # plot voltage in last 20 ms
        yasa.topoplot(pd.Series(np.mean(data_wind[int(-0.02*sf):,:],0), ch_names), cmap='Spectral_r')
        # plot spatial patterns
        yasa.topoplot(pd.Series(weighted_patterns[:,0], ch_names), cmap='Spectral_r')
        # correlation between voltage maps and spatial pattern maps
        print(f'Correlation between maps is: {np.corrcoef(weighted_patterns[:,0], np.median(data_wind[int(-0.02*sf):,:],0)).min().round(3)}')
        
        # minamp 
        minamp = min(np.percentile((d[-2* int(sf):]), 10), -35)
        crit = min(d[int(-0.02*sf):])
        
        minamp_csd = min(np.percentile((d_csd[-2* int(sf):]), 10), -35)
        crit_csd = min(d_csd[int(-0.02*sf):])

        if (min(d[-2:]) < -35) and (new_times[ix] - time_reconstruct[-1] > 3):
            if crit < minamp and crit_csd < minamp_csd and (new_times[ix] - time_reconstruct[-1] > 3): 
                if crit_csd < minamp_csd and np.ptp(d_csd) < 500 and (new_times[ix] - time_reconstruct[-1] > 3): 
                    time_reconstruct.append(new_times[ix])
                    crit_reconstruct.append(crit)
                    #ts_up = new_times[ix] + .475
                    #time_reconstruct_up.append(ts_up)
                    #crit_reconstruct_up.append(d[int(ts_up)])
                    #plt.figure()
                    plt.plot(d)
                    #yasa.topoplot(pd.Series(metric, ch_names), cmap='Spectral_r')
                    #yasa.topoplot(pd.Series(np.mean(data_wind[int(-0.02*sf):,:],0), ch_names), cmap='Spectral_r')
                    #yasa.topoplot(pd.Series(weighted_patterns[:,0], ch_names), cmap='Spectral_r', vmax=20)
                    # print(f'Correlation between maps is: {np.corrcoef(weighted_patterns[:,0], np.median(data_wind[int(-0.02*sf):,:],0)).min().round(3)}')
                    
                    if len(time_reconstruct) == 25:
                        break
        
        # recompute every 250 ms, the last 2 seconds
        ix += int(.5*fs)       