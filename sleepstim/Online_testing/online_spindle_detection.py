# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:41:20 2021

@author: mvali
"""

import matplotlib.pyplot as plt 
import numpy as np
import os 
import emd 
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time
from neurodsp.plts.time_series import plot_time_series, plot_instantaneous_measure
import yasa
import scipy
import pywt
import seaborn as sns
from mne.time_frequency import morlet
sns.set(style='darkgrid', font_scale=1.2)

# load data 
#data = np.loadtxt(r'C:\Users\mvali\Documents\GitHub\yasa\notebooks\data_N2_spindles_15sec_200Hz.txt').astype('float32')
data = np.loadtxt(r'/home/administrator/Downloads/yasa-master/notebooks/data_N2_spindles_15sec_200Hz.txt').astype('float32')
sf = 200
times = (np.arange(len(data))/sf).astype('float32')

#%%
def spindles_detect(x, sf, thresh=0.25, wlt_params={'nc': 12, 'cf': 'auto'}):
    """Simple spindles detector based on Morlet wavelet.

    Parameters
    ----------
    x : 1D-array
        EEG signal
    sf : float
        Sampling frequency
    thresh : float
        Threshold (0 - 1)
    wlt_params : dict
        Morlet wavelet parameters ::

        'nc' : number of oscillations
        'cf' : central frequency (int or 'auto')

    Returns
    -------
    supra_thresh_bool : 1D-array (boolean)
        Boolean array indicating for each point if it is a spindles or not.
    sp_params : dict
        Spindles parameters dictionnary.
    """
    from scipy.signal import detrend
    from mne.time_frequency import morlet, psd_array_multitaper

    if wlt_params['cf'] == 'auto':
        # Compute the power spectrum and find the peak 11-16 Hz frequency.
        psd, freqs = psd_array_multitaper(x, sf, fmin=11, fmax=16, verbose=0)
        wlt_params['cf'] = freqs[np.argmax(psd)]
        print('Central frequency: %.2f Hz' % wlt_params['cf'])

    # Compute the wavelet and convolve with data
    wlt = morlet(sf, [wlt_params['cf']], n_cycles=wlt_params['nc'])[0]
    analytic = np.convolve(x, wlt, mode='same')
    phase = np.angle(analytic)

    # Square and normalize the magnitude from 0 to 1 (using the min and max)
    power = np.square(np.abs(analytic))
    norm_power = (power - power.min()) / (power.max() - power.min())

    # Find supra-threshold values and indices
    supra_thresh_bool = norm_power >= thresh
    supra_thresh_idx = np.where(supra_thresh_bool)[0]

    # Extract duration, frequency and amplitude of spindles
    sp = np.split(supra_thresh_idx, np.where(np.diff(supra_thresh_idx) != 1)[0] + 1)
    idx_start_end = np.array([[k[0], k[-1]] for k in sp])
    sp_dur = (np.diff(idx_start_end, axis=1) / sf).flatten() * 1000
    sp_amp, sp_freq = np.zeros(len(sp)), np.zeros(len(sp))
    for i in range(len(sp)):
        sp_amp[i] = np.ptp(detrend(x[sp[i]]))
        sp_freq[i] = np.median((sf / (2 * np.pi) * np.diff(phase[sp[i]])))

    sp_params = {'Duration (ms)' : sp_dur, 'Frequency (Hz)': sp_freq,
                 'Amplitude (uV)': sp_amp}

    return supra_thresh_bool, sp_params

# Run the function
spindles = spindles_detect(x = data, sf = sf, thresh=0.25)

#%%
## Compute instantaneous frequency from a signal
def inst_freq(data, sf, f_range=None, smoothing=True, smoothing_size=25):
    IF = freq_by_time(data, sf, f_range=f_range, hilbert_increase_n=True)
    if smoothing:
        IF = scipy.ndimage.median_filter(IF, size=smoothing_size)
    return IF

IF = inst_freq(data, sf, f_range=None, smoothing=True, smoothing_size=25)

# Plot median filtered IF vs. signal
_, axs = plt.subplots(2, 1, figsize=(15, 9))
plot_time_series(times, data, 'Raw Signal', xlabel=None, ax=axs[0])
plot_instantaneous_measure(times, IF, 'frequency', 
                           label='Instantaneous Frequency',
                           colors='r', ax=axs[1])
# Plot with vlines 
bool_sp = np.logical_and(IF>=11, IF<=16)
sp_loc = np.where(bool_sp)[0]

plt.plot(times, data)
plt.vlines(sp_loc/sf, ymin=min(data), ymax=max(data))

#%%
## EMD --> too slow
# Compute EMD
imf = emd.sift.sift(data, max_imfs=5)

# Visualise the IMFs
emd.plotting.plot_imfs(imf, cmap=True, scale_y=True)

# Extract instantaneous phase, frequency, and amplitude 
IP, IF, IA = emd.spectra.frequency_transform(imf, sf, 'hilbert')

bool_sp = np.logical_and(IF[:,0]>=12, IF[:,0]<=16)
sp_loc = np.where(bool_sp)

#%%
## B-spline wavelet convolution
a,b = pywt.cwt(data = data, scales = frequencies[1], wavelet = "fbsp4-1.5-13")

dt = 0.01  # 100 Hz sampling
frequencies = pywt.scale2frequency("fbsp4-1.5-13", np.linspace(1,12,12)) / dt

wav = pywt.ContinuousWavelet('fbsp4-1.5-13')
int_psi, x = pywt.integrate_wavelet(wav, precision=10)


#%%
## Morelet wavelet convolution
# Parameters
cf = 13     # Central spindles frequency in Hz
nc = 12     # Number of oscillations in the spindles

# Compute the wavelet
wlt = morlet(sf, [cf], n_cycles=nc)[0]
# Convolve the wavelet and extract magnitude and phase
analytic = np.convolve(data, wlt, mode='same')
magnitude = np.abs(analytic)
phase = np.angle(analytic)

power = np.square(magnitude)
norm_power = (power - power.min()) / (power.max() - power.min())
power_bspline = np.square(a.real[0])
norm_power_bspline = (power_bspline - power_bspline.min()) / (power_bspline.max() - power_bspline.min())


# Define the threshold
thresh = 0.25

# Find supra-threshold values
supra_thresh = np.where(norm_power >= thresh)[0]

# Create vector for plotting purposes
val_spindles = np.nan * np.zeros(x.size)
val_spindles[supra_thresh] = x[supra_thresh]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax1.plot(times, x, lw=1.5)
ax1.plot(times, val_spindles, color='indianred', alpha=.8)
ax1.set_xlim(0, times[-1])
ax1.set_ylabel('Voltage [uV]')
ax1.set_title('Cz EEG signal')

ax2.plot(times, norm_power_bspline)
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Normalized wavelet power')
ax2.axhline(thresh, ls='--', color='indianred', label='Threshold')
ax2.fill_between(times, norm_power_bspline, thresh, where = norm_power_bspline >= thresh,
                 color='indianred', alpha=.8)
plt.legend(loc='best')

## Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.suptitle('Signal with B-Spline Wavelet Conv.')

ax1.plot(times[:-1], data[:-1])
ax1.set_ylabel('Signal')

ax2.plot(times[:-1], a.real[0][:-1])
ax2.set_ylabel('Real part of convolution')

ax3.plot(times[:-1], np.diff(data))
ax3.set_ylabel('Derivative of Signal')
ax3.set_xlabel('time (s)')

plt.show()