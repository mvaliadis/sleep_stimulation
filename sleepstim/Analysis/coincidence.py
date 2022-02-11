#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:58:17 2021

@author: administrator
"""

from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize, ListedColormap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sleepstim.Analysis.analysis_pre_process import load_preprocessed_data
import yasa 
import scipy
import mne 

Data = load_preprocessed_data('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded/YIOYSRPX_sham.p')
# data = Data.data[:, Data.chans.index('C3')]

#%%
spindles = yasa.spindles_detect(data = Data.data[:,0:23].T, sf=512, ch_names=Data.chans[0:23], hypno=Data.hypno_with_art,
                                include=(2), freq_sp=(9, 16), freq_broad=(1, 30), duration=(0.5, 2),
                                min_distance=500, thresh={'rel_pow': 0.2, 'corr': 0.65, 'rms': 1.5},
                                multi_only=False, remove_outliers=True, verbose=False)

#%%
## Plot Spindles with Detection and Spectrogram 
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
ax1.plot(Data.times, Data.data[:,Data.chans.index('C3')])
ax1.set_ylabel('Voltage [uV]')
ax1.set_title('C3 EEG signal')

ax2.plot(Data.times, spindles.get_mask(), color='m')
# ax2.axhline(1, ls='--', color='indianred', label='Threshold')
ax2.set_ylabel('Spindle (Y/N)')

# Calculate multi-taper spectrogram
nperseg = int(4 * Data.sfreq)
f, t, Sxx = spectrogram_lspopt(data.squeeze(), Data.sfreq, nperseg=nperseg, noverl ap=0)
Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

# Select only relevant frequencies (up to 30 Hz)
good_freqs = np.logical_and(f >= 0.5, f <= 25)
Sxx = Sxx[good_freqs, :]
f = f[good_freqs]
t /= 3600  # Convert t to hours

# Normalization
vmin, vmax = np.percentile(Sxx, [0 + 2.5, 100 - 2.5])
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = 'Spectral_r'

im = ax3.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True,
                    shading="auto")
ax3.set_ylabel('Frequency [Hz]')
ax3.set_xlabel('Time [hrs]')
# cbar = fig.colorbar(im, ax=ax3, shrink=0.95, fraction=0.1, aspect=25)
cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)


plt.xlim(0,Data.times[-1])
sns.despine()
plt.tight_layout()

#%% 
## Plot Coincidence matrix
coincidence = spindles.get_coincidence_matrix(scaled=True)
sns.set_theme(style="white")

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(coincidence, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio   
sns.heatmap(coincidence, mask=mask, cmap=cmap, vmax=np.percentile(coincidence, 95), center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#%%
## Plot Coherence
_, epochs = yasa.sliding_window(Data.data[Data.hypno_with_art==2,0:23][0:60*60*512].T, Data.sfreq, window=0.5)
coherence = mne.connectivity.spectral_connectivity(data = epochs, 
                                                   method = 'plv', sfreq=Data.sfreq, fmin=9, fmax=16)
mean_coherence = coherence[0].mean(-1)

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(mean_coherence, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(mean_coherence, mask=mask, cmap=cmap, vmax=np.percentile(mean_coherence, 95), center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, yticklabels=Data.chans[0:23], 
            xticklabels=Data.chans[0:23])