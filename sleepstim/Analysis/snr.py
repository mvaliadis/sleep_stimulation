#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:59:41 2021

@author: administrator
"""

import mne
#from multitaper_toolbox.python.multitaper_spectrogram_python import multitaper_spectrogram
import matplotlib.pyplot as plt
import yasa
import numpy as np
from scipy.stats import ttest_rel
from sleepstim.Analysis.analysis_pre_process import (Data_Struct, load_preprocessed_data)
import yasa
import numpy as np
import matplotlib.pyplot as plt

Data = load_preprocessed_data('/media/administrator/data/Study_1_data/Pre-processed_data/Adaption/IBYYXKMB_adaption_preproc_data.p')[0]

## The below code has been adapted from mne: 
## https://mne.tools/dev/auto_tutorials/time-freq/plot_ssvep.html#plot-psd-and-snr-spectra

#%%
###############################################################################
# Calculate signal to noise ratio (SNR)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# SNR - as we define it here - is a measure of relative power:
# it's the ratio of power in a given frequency bin - the 'signal' -
# to a 'noise' baseline - the average power in the surrounding frequency bins.
# This approach was initially proposed by
# `Meigen & Bach (1999) <https://doi.org/10.1023/A:1002097208337>`_
#
# Hence, we need to set some parameters for this baseline - how many
# neighboring bins should be taken for this computation, and do we want to skip
# the direct neighbors (this can make sense if the stimulation frequency is not
# super constant, or frequency bands are very narrow).
#
# The function below does what we want.
#

def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(
        mean_noise, pad_width=pad_width, constant_values=np.nan
    )

    return psd / mean_noise


#%%
## SNR computation 

def snr_score(freqs, snrs):
    freq_range = range(0, len(freqs))
    # SNR spectrum
    snr_mean = snrs.mean(axis=(0, 1))[freq_range]
    snr_std = snrs.std(axis=(0, 1))[freq_range]
    
    return snr_mean, snr_std
    
def snr_plot(psds, snrs, freqs):
    freq_range = range(0, len(freqs))
    fig, axes = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(8, 5))
    psds_plot = 10 * np.log10(psds)
    psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
    psds_std = psds_plot.std(axis=(0, 1))[freq_range]
    axes[0].plot(freqs[freq_range], psds_mean, color='b')
    axes[0].fill_between(
        freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std,
        color='b', alpha=.2)
    axes[0].set(title="PSD spectrum", ylabel='Power Spectral Density [dB]')
    
    # SNR spectrum
    snr_mean = snrs.mean(axis=(0, 1))[freq_range]
    snr_std = snrs.std(axis=(0, 1))[freq_range]
    
    axes[1].plot(freqs[freq_range], snr_mean, color='r')
    axes[1].fill_between(
        freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std,
        color='r', alpha=.2)
    axes[1].set(
        title="SNR spectrum", xlabel='Frequency [Hz]',
        ylabel='SNR', ylim=[0,  np.nanmax(snr_mean) + snr_std[np.where(snr_mean == np.nanmax(snr_mean))[0][0]] + 1], 
        xlim=[freqs[0], freqs[-1]])
    fig.show()
    
#%% 
if __name__ == '__main__':
    eeg_index = [i for i, x in enumerate(Data.chtypes) if x == "eeg"]
    emg_index = [i for i, x in enumerate(Data.chtypes) if x == "emg"]
    eog_index = [i for i, x in enumerate(Data.chtypes) if x == "eog"]
    ecg_index = [i for i, x in enumerate(Data.chtypes) if x == "ecg"]
    for idx, freq_low_high in zip([eeg_index, emg_index, eog_index, ecg_index],
                                  [(0.5, 30),(0.5, 30),(10, 100),(0.5, 70)]):
        _, epochs = yasa.sliding_window(data = Data.data[:,idx].T, 
                                        window=30, sf = Data.sfreq)
        psds, freqs = mne.time_frequency.psd_array_welch(epochs, sfreq=Data.sfreq, 
                                                         fmin=freq_low_high[0], fmax=freq_low_high[-1], 
                                                         window='boxcar', n_overlap=0, 
                                                         n_per_seg=None, n_fft=4*Data.sfreq)
        snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3,
                            noise_skip_neighbor_freqs=1)
        snr_mean, snr_std = snr_score(freqs = freqs, snrs = snrs)
        snr_plot(psds = psds, snrs = snrs, freqs = freqs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    