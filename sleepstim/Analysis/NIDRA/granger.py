#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:55:13 2024

@author: administrator
"""

import numpy as np
from matplotlib import pyplot as plt

import mne
from mne_connectivity import spectral_connectivity_epochs, phase_slope_index
import yasa 
import pandas as pd

file = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/RoKi_1_csd-epo.fif'
#file = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Epochs/IsEb_1_csd-epo.fif'
subject, night, ref = file.split('/')[-1].split('-')[0].split('_')
epochs = mne.read_epochs(file)
try:
    epoch = epochs['nmes_stim_fz'].crop(tmin=0, tmax=2).pick('csd')
except:
    epoch = epochs['pn_stim_fz'].crop(tmin=0, tmax=2).pick('csd')
    
lags = 20 

#%%
# motor sensors
signals_a = [epochs.ch_names.index(ch_idx) for ch_idx in (['C5','C3','C1'])]
# frontal sensors
signals_b = [epochs.ch_names.index(ch_idx) for ch_idx in (['F1','Fz','F2'])]

indices_ab = (np.array([signals_a]), np.array([signals_b]))  # A => B
indices_ba = (np.array([signals_b]), np.array([signals_a]))  # B => A

# compute Granger causality
gc_ab = spectral_connectivity_epochs(
    epoch,
    sfreq=epoch.info['sfreq'],
    method=["gc"],
    indices=indices_ab,
    fmin=0.5,
    fmax=30,
    #rank=(np.array([3]), np.array([3])),
    gc_n_lags=lags,
)  # A => B
gc_ba = spectral_connectivity_epochs(
    epoch,
    sfreq=epoch.info['sfreq'],
    method=["gc"],
    indices=indices_ba,
    fmin=0.5,
    fmax=30,
    #rank=(np.array([3]), np.array([3])),
    gc_n_lags=lags,
)  # B => A
freqs = gc_ab.freqs

fig, axis = plt.subplots(1, 1)
axis.plot(freqs, gc_ab.get_data()[0], linewidth=2)
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Connectivity (A.U.)")
fig.suptitle("GC: [A => B]")
plt.show()

fig, axis = plt.subplots(1, 1)
axis.plot(freqs, gc_ba.get_data()[0], linewidth=2)
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Connectivity (A.U.)")
fig.suptitle("GC: [B => A]")
plt.show()

# Drivers and receivers: analysing the net direction of information flow
net_gc = gc_ab.get_data() - gc_ba.get_data()  # [A => B] - [B => A]

fig, axis = plt.subplots(1, 1)
axis.plot((freqs[0], freqs[-1]), (0, 0), linewidth=2, linestyle="--", color="k")
axis.plot(freqs, net_gc[0], linewidth=2)
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Connectivity (A.U.)")
fig.suptitle("Net GC: [A => B] - [B => A]")
plt.show()

#%%
# compute GC on time-reversed signals
gc_tr_ab = spectral_connectivity_epochs(
    epoch,
    sfreq=epoch.info['sfreq'],
    method=["gc_tr"],
    indices=indices_ab,
    fmin=0.5,
    fmax=30,
    #rank=(np.array([3]), np.array([3])),
    gc_n_lags=lags,
)  # TR[A => B]
gc_tr_ba = spectral_connectivity_epochs(
    epoch,
    sfreq=epoch.info['sfreq'],
    method=["gc_tr"],
    indices=indices_ba,
    fmin=0.5,
    fmax=30,
    #rank=(np.array([3]), np.array([3])),
    gc_n_lags=lags,
)  # TR[B => A]

# compute net GC on time-reversed signals (TR[A => B] - TR[B => A])
net_gc_tr = gc_tr_ab.get_data() - gc_tr_ba.get_data()

# compute TRGC
trgc = net_gc - net_gc_tr

fig, axis = plt.subplots(1, 1)
axis.plot((freqs[0], freqs[-1]), (0, 0), linewidth=2, linestyle="--", color="k")
axis.plot(freqs, trgc[0], linewidth=2)
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Connectivity (A.U.)")
fig.suptitle("TRGC: net[A => B] - net time-reversed[A => B]")
plt.show()

#%%%
## wPLI 

# fmin, fmax = 0.5, 2  # compute connectivity within 11-16 Hz
# sfreq = epochs.info["sfreq"]  # the sampling frequency
# tmin = 0.0  # exclude the baseline period

# # Number of channels (assuming 64 channels, indexed from 0 to 63)
# n_channels = 64

# # Seed channel index (e.g., channel 14)
# seed_channel_idx = epoch.ch_names.index('C3')

# # Generate indices for all pairs where the seed channel is connected to each other channel
# # Note: We create a list of tuples (seed_channel_idx, other_channel_idx) for each other channel
# indices = (np.array([seed_channel_idx] * (n_channels - 1)),  # seed channel repeated
#             np.array([idx for idx in range(n_channels) if idx != seed_channel_idx]))  # all other channels


# # Compute wPLI
# con_wpli = spectral_connectivity_epochs(
#     epoch,
#     method="wpli",
#     mode="multitaper",
#     sfreq=sfreq,
#     fmin=fmin,
#     fmax=fmax,
#     indices=indices, 
#     faverage=True,
#     tmin=tmin,
#     mt_adaptive=False,
#     n_jobs=-1,
# )


# psi = phase_slope_index(
#     epoch,
#     mode="multitaper",
#     indices=indices,
#     sfreq=sfreq,
#     fmin=fmin,
#     fmax=fmax,
#     tmin=tmin,
#     n_jobs=-1,
# )

# conn = np.nanmean(con_wpli.get_data("dense").squeeze(), 0)
# conn[seed_channel_idx] = 0 

# yasa.topoplot(pd.Series(conn, index=con_wpli.names, name='wPLI'))

# conn = np.nanmean(psi.get_data("dense").squeeze(), 0)
# conn[seed_channel_idx] = 0 

# yasa.topoplot(pd.Series(conn, index=psi.names, name='PSI'))
