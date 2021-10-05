#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 17:59:39 2021

@author: administrator
"""

import gzip
import numpy as np
from scipy.signal import welch
from matplotlib import pyplot as plt
from mne import create_info
from mne.io import RawArray
from mne.viz import plot_topomap
from mne.preprocessing import ICA
from pyriemann.spatialfilters import AJDC
from pyriemann.utils.viz import plot_cospectra

#%%
## Artifact Correction by AJDC-based Blind Source Separation
## Blind source separation (BSS) based on approximate joint diagonalization of Fourier cospectra (AJDC), 
## applied to artifact correction of EEG [1].
## [1] Q. Barthélemy, L. Mayaud, Y. Renard, D. Kim, S.-W. Kang, J. Gunkelman and M. Congedo, 
## “Online denoising of eye-blinks in electroencephalography” , Neurophysiol Clin, 2017.

## See: https://pyriemann.readthedocs.io/en/latest/auto_examples/artifacts/plot_correct_ajdc_EEG.html#id2

# signal = raw.pick_types(eeg=True)
# data = signal.get_data().T*1e6
# ch_names = signal.ch_names
# ch_types = ch_types[0:61]
# ch_info = signal.info
def AJDC_bss_art_correction(data, ch_names, ch_types, sf, co_spec_freqs = (1,32)):
    ch_info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sf)
    ch_info.set_montage('standard_1005')
    signal = RawArray(data.T, ch_info, verbose=False)
    signal.plot(duration=len(data)/512, start=0, n_channels=len(ch_names),
                color={'eeg': 'steelblue'},
                title='Original EEG signal', show_scalebars=True)
    
    # Compute and diagonalize Fourier cospectral matrices between 1 and 35 Hz
    window, overlap = sf, 0.5
    fmin, fmax = co_spec_freqs[0], co_spec_freqs[1]
    ajdc = AJDC(window=window, overlap=overlap, fmin=fmin, fmax=fmax, fs=sf,
                dim_red={'max_cond': 100})
    ajdc.fit(data.T[np.newaxis, np.newaxis, ...])
    freqs = ajdc.freqs_
    
    # Plot cospectra in channel space, after trace-normalization by frequency: each
    # cospectrum, associated to a frequency, is a covariance matrix
    plot_cospectra(ajdc._cosp_channels, freqs, ylabels=ch_names,
                   title='Cospectra, in channel space')
    
    # Plot diagonalized cospectra in source space
    sr_count = ajdc.n_sources_
    sr_names = ['S' + str(s).zfill(2) for s in range(sr_count)]
    plot_cospectra(ajdc._cosp_sources, freqs, ylabels=sr_names,
                   title='Diagonalized cospectra, in source space')
    
    # Estimate sources S applying forward filters B to signal X: S = B X
    source_raw = ajdc.transform(data.T[np.newaxis, ...])[0]
    
    # Plot sources S
    sr_info = create_info(ch_names=sr_names, ch_types=['misc'] * sr_count,
                          sfreq=sf)
    source = RawArray(source_raw, sr_info, verbose=False)
    source.plot(duration=len(data)/sf, start=0,
                scalings={'misc': 2e2}, title='EEG sources estimated by AJDC',
                show_scalebars=False)
    
    # Identify artifact by eye: blinks are well separated in source S0
    for i in range(10):
        blink_idx = i
    
        # Get normal spectrum, ie power spectrum after trace-normalization
        blink_spectrum_norm = ajdc._cosp_sources[:, blink_idx, blink_idx]
        blink_spectrum_norm /= np.linalg.norm(blink_spectrum_norm)
        
        # Get absolute spectrum, ie raw power spectrum of the source
        f, spectrum = welch(source.get_data(picks=[blink_idx]), fs=sf,
                            nperseg=window, noverlap=int(window * overlap))
        blink_spectrum_abs = spectrum[0, (f >= fmin) & (f <= fmax)]
        blink_spectrum_abs /= np.linalg.norm(blink_spectrum_abs)
        
        # Get topographic map
        blink_filter = ajdc.backward_filters_[:, blink_idx]
        
        # Plot spectrum and topographic map of the blink source separated by AJDC
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        axs[0].set(title='Power spectrum of the blink source estimated by AJDC',
                   xlabel='Frequency (Hz)', ylabel='Power spectral density')
        axs[0].plot(freqs, blink_spectrum_abs[1::], label='Absolute power')
        axs[0].plot(freqs, blink_spectrum_norm, label='Normal power')
        axs[0].legend()
        axs[1].set_title('Topographic map of the blink source estimated by AJDC')
        plot_topomap(blink_filter, pos=ch_info, axes=axs[1], extrapolate='box')
        plt.show()
    
    # BSS denoising: blink source is suppressed in source space using activation
    # matrix D, and then applying backward filters A to come back to channel space
    # Denoised signal: Xd = A D S
    signal_denois_raw = ajdc.inverse_transform(source_raw[np.newaxis, ...],
                                               supp=[blink_idx])[0]
    
    # Plot denoised signal Xd
    signal_denois = RawArray(signal_denois_raw, ch_info, verbose=False)
    signal_denois.plot(duration=len(data)/sf, start=0, n_channels=len(ch_names),
                       scalings={'eeg': 3e1}, color={'eeg': 'steelblue'},
                       title='Denoised EEG signal by AJDC', show_scalebars=False)
    
    return signal_denois

#%%
## ICA comparison
## Infomax-based ICA is a Higher-Order Statistics (HOS)-based BSS, minimizing
# mutual information
# ica = ICA(n_components=ajdc.n_sources_, method='infomax', random_state=42)
# ica.fit(signal, picks='eeg')

# # Plot sources separated by ICA
# ica.plot_sources(signal, title='EEG sources estimated by ICA')

# # Plot topographic maps of sources separated by ICA
# ica.plot_components(title='Topographic maps of EEG sources estimated by ICA')
