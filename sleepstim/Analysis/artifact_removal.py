#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:48:09 2024

@author: administrator
"""

import matplotlib.pyplot as plt
import numpy as np
import mne
import scipy.signal as signal
import numpy.polynomial.polynomial as poly
from scipy import interpolate

# 0. Load data
filename = '/media/administrator/Sleep_Data/Processed/PaJa_1-C3-epo.fif'
epochs = mne.read_epochs(filename, preload=True)
evk = epochs['nmes_stim'].average()

#data = epochs['nmes_stim'].get_data('eeg', units='uV')
data = epochs['nmes_stim'][50].get_data('eeg', units='uV').squeeze()
data_sham = epochs['nmes_sham'][50].get_data('eeg', units='uV').squeeze()

#%%
# Interpolation
def interpolate_artifact(signal, start_artifact, end_artifact, fit_margin=50, poly_degree=3):
    """
    Interpolates over an artifact in each channel of a multi-channel signal using polynomial fitting.

    Parameters:
    - signal: The original multi-channel signal (2D numpy array, channels x time).
    - start_artifact: The starting index of the artifact.
    - end_artifact: The ending index of the artifact.
    - fit_margin: The number of points used around the artifact for fitting the polynomial.
    - poly_degree: The degree of the polynomial used for fitting.

    Returns:
    - The signal with the artifact interpolated in each channel.
    """
    num_channels = signal.shape[0]
    interpolated_signal = np.copy(signal)

    for channel in range(num_channels):
        start_fit = max(0, start_artifact - fit_margin)
        end_fit = min(signal.shape[1], end_artifact + fit_margin)

        # Time points for fitting
        t_fit = np.arange(start_fit, end_fit)
        signal_fit = signal[channel, start_fit:end_fit]

        # Fit a polynomial to the data around the artifact
        coefs = poly.polyfit(t_fit, signal_fit, poly_degree)

        # Interpolate the signal for this channel
        t_artifact = np.arange(start_artifact, end_artifact + 1)
        interpolated_signal[channel, start_artifact:end_artifact + 1] = poly.polyval(t_artifact, coefs)

    return interpolated_signal

def interpolate_with_sham(signal, sham_signal, start_artifact, end_artifact, fit_margin=50, poly_degree=3):
    """
    Interpolates over an artifact in each channel of a multi-channel signal using polynomial fitting,
    informed by a sham condition.

    Parameters:
    - signal: The original multi-channel signal with artifacts (2D numpy array, channels x time).
    - sham_signal: The multi-channel sham signal without artifacts (2D numpy array, channels x time).
    - start_artifact: The starting index of the artifact.
    - end_artifact: The ending index of the artifact.
    - fit_margin: The number of points used around the artifact for fitting the polynomial.
    - poly_degree: The degree of the polynomial used for fitting.

    Returns:
    - The signal with the artifact interpolated in each channel.
    """
    num_channels = signal.shape[0]
    interpolated_signal = np.copy(signal)

    for channel in range(num_channels):
        # Use sham data for polynomial fitting
        start_fit_sham = max(0, start_artifact - fit_margin)
        end_fit_sham = min(sham_signal.shape[1], end_artifact + fit_margin)

        t_fit_sham = np.arange(start_fit_sham, end_fit_sham)
        sham_fit = sham_signal[channel, start_fit_sham:end_fit_sham]

        # Fit a polynomial to the sham data
        coefs_sham = poly.polyfit(t_fit_sham, sham_fit, poly_degree)

        # Interpolate the experimental signal using the sham-based polynomial
        t_artifact = np.arange(start_artifact, end_artifact + 1)
        interpolated_signal[channel, start_artifact:end_artifact + 1] = poly.polyval(t_artifact, coefs_sham)

    return interpolated_signal

def interpolate_cubic(data, times, win):
    """
    Performs cubic interpolation on a 2D time series data.

    Parameters:
    - data: 2D numpy array representing the time series (channels x time points).
    - times: 1D numpy array representing the time points.
    - win: Tuple (start_time, end_time) representing the window for interpolation in seconds.

    Returns:
    - Interpolated 2D time series data.
    """
    # Convert window to indices; +1 because it should include the last timepoint
    idx1 = np.argmin(np.abs(times - win[0]))
    idx2 = np.argmin(np.abs(times - win[1])) + 1

    # Delete timepoints that should be interpolated from the times array
    x = np.delete(times, np.s_[idx1:idx2], 0)

    # Initialize the array for interpolated data
    interpolated_data = np.copy(data)

    # Perform interpolation for each channel
    for channel in range(data.shape[0]):
        # Delete timepoints that should be interpolated from the channel data
        y = np.delete(data[channel], np.s_[idx1:idx2])

        # Fit the cubic interpolation
        p = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")

        # Get the interpolation values for the timepoints of interest
        interp_values = p(times[idx1:idx2])

        # Replace the corresponding timepoints in the data with interpolated values
        interpolated_data[channel, idx1:idx2] = interp_values

    return interpolated_data

# # Artifact indices
# artifact_indices = np.arange(epochs.time_as_index(0)[0], 
#                              epochs.time_as_index(.125)[0])

# # Example usage
# interpolated_signal = interpolate_with_sham(data, 
#                                             data_sham, 
#                                             start_artifact = epochs.time_as_index(0)[0],
#                                             end_artifact = epochs.time_as_index(.12)[0],
#                                             fit_margin=100, poly_degree=3
#                                             )


# # Interpolated_signal now contains the EEG signal with the artifact interpolated for each channel
# interpolated_signal = interpolate_artifact(data, 
#                                            start_artifact = epochs.time_as_index(-.05)[0],
#                                            end_artifact = epochs.time_as_index(.12)[0],
#                                            fit_margin=10, poly_degree=2)


# Example usage
interpolated_signal = interpolate_cubic(data = data, 
                                        times = epochs.times, 
                                        win = [-.016, .117])


#%% Plotting
plt.plot(epochs.times, data.T, 'r')
#plt.plot(epochs.times, data_sham.T, 'b')
plt.plot(epochs.times, interpolated_signal.T, 'k')

#%%
# Power Analysis
f, Pxx = signal.welch(interpolated_signal[epochs.ch_names.index('C3'),:].squeeze(), 
                      fs=epochs.info['sfreq'], 
                      nperseg=int(epochs.info['sfreq']*2/1), average='median')
plt.figure()
plt.semilogy(f, Pxx)
plt.semilogx(f, Pxx)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

#%%

raw = mne.io.RawArray(eego_data[0:512*60*60,chan_idx].T, info)

def interpolate_cubic(data, times, annotations, win, sfreq):
    """
    Performs cubic interpolation on a single channel's data.

    Parameters:
    - data: 1D numpy array representing the channel's data.
    - times: 1D numpy array representing the time points.
    - annotations: Annotations from the MNE Raw object.
    - win: Tuple (start_time, end_time) in seconds for interpolation.
    - sfreq: Sampling frequency.

    Returns:
    - Interpolated data for the channel.
    """
    
    interpolated_data = np.copy(data)

    # Perform interpolation for each annotation time window
    for annotation in annotations:
        # Get the time indices for this annotation
        onset = annotation['onset']
        idx1 = np.argmin(np.abs(times - (onset + win[0]))) - 1
        idx2 = np.argmin(np.abs(times - (onset + win[1]))) + 1
        
        # Delete timepoints that should be interpolated from the times array
        x = np.delete(times, np.s_[idx1:idx2], 0)

        # Perform interpolation for each channel
        for channel in range(data.shape[0]):
           # Delete timepoints that should be interpolated from the channel data
           y = np.delete(data[channel], np.s_[idx1:idx2])

           # Fit the cubic interpolation
           if len(x) > 3 and len(y) > 3:  # Ensure sufficient points for cubic interpolation
               cubic_interp = interpolate.interp1d(x, y, kind='cubic', fill_value="extrapolate")

               # Get the interpolation values for the timepoints of interest
               interp_values = cubic_interp(times[idx1:idx2])

               # Replace the corresponding timepoints in the data with interpolated values
               interpolated_data[channel, idx1:idx2] = interp_values

    return interpolated_data


raw.filter(0.5, 30, picks='eeg', n_jobs=1)
raw.set_eeg_reference(['M1', 'M2'])

# Modify the apply_function call to pass the annotations
raw2 = raw.copy().apply_function(fun=interpolate_cubic, 
                                 picks='eeg', 
                                 n_jobs=-1, 
                                 channel_wise=False,
                                 times=raw.times, 
                                 annotations=raw.annotations[raw.annotations.description == 'nmes_stim'],
                                 win=(-0.016, 0.117),
                                 sfreq=raw.info['sfreq'])
  
epochs = epoch_sleep_nmes_data(raw, events_merged, event_id,
                               tmin=-3, tmax=3, csd=False, 
                               baseline=None, preload=True)
epochs2 = epoch_sleep_nmes_data(raw2, events_merged, event_id,
                               tmin=-3, tmax=3, csd=False, 
                               baseline=None, preload=True)

