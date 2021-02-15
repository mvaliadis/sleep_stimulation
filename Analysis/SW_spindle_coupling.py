#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:45:22 2021

@author: administrator
"""

import yasa
import numpy as np
import mne 
import pingouin as pg
from scipy.signal import hilbert
from scipy.fftpack import next_fast_len

#%%
## Using the spindles detection to calculate SO/Spindle coupling

## An even more straightforward way to calculate the coupling is to use the spindles detection rather than the 
## slow-waves detection. Here, the SOPhase column correspond to the phase of the slow-oscillation signal 
## (0.1 - 1.25 Hz, defined in freq_so) at the most prominent peak of the spindles. While this approach is conceptually 
## very similar to the previous coupling calculated on the slow-wave (and computationally more efficient because 
## here we do not create epochs centered around the peak), the former method based on slow-wave detection might be
## more adequate in most cases because it ensures that there is a strong oscillatory activity in the lower frequency 
## band, which is one of the pre-requisite for PAC.

"""
TO-DO: add topographical analysis here from spindle module...

"""

# Spindle detection - see note above
sp = yasa.spindles_detect(data, sf, hypno, include=(2, 3)) #coupling=True, freq_so=(0.1, 1.25)

# Get data centered around the most prominent peak of each spindle
events = yasa.get_sync_events(data, sf, detection=sp, center='Peak', time_before=1, time_after=1)

# Plot the peak-locked sleep spindle average across all detected spindles in NEM for this participant
events.groupby("Time")['Amplitude'].mean().plot(color=['tab:blue'])
plt.ylabel('Amplitude')
plt.title("Peak-locked average sleep spindle")
sns.despine()

# Filter slow-oscillations
data_so = mne.filter.filter_data(data, sf, 0.1, 1.25, verbose=0, h_trans_bandwidth=0.1, l_trans_bandwidth=0.1)

# Get filtered SO data centered around the most prominent peak of each spindle
events_so = yasa.get_sync_events(data_so, sf, detection=sp, center='Peak', time_before=1, time_after=1)
events = events.merge(events_so, on=['Time', 'Event'], suffixes=['_sp', '_so'])
events.rename(columns={'Amplitude_sp': 'Spindles', 'Amplitude_so': 'SO'}, inplace=True)


# Overlay spindles + slow-oscillations
events.groupby("Time").mean().plot(color=['tab:blue', 'tab:red'])
plt.ylabel('Amplitude')
plt.title("Peak-locked average sleep spindle")
sns.despine()

# Extract instantaneous phase using Hilbert transform
n = data_so.size
nfast = next_fast_len(n)
inst_phase = np.angle(hilbert(data_so, N=nfast)[:n])

# Get SO phase centered around the most prominent peak of each spindle
events_phase = yasa.get_sync_events(inst_phase, sf, detection=sp, center='Peak', time_before=1, time_after=1)
events_phase.rename(columns={'Amplitude': "SO_phase"}, inplace=True)
events = events.merge(events_phase, on=['Time', 'Event'])
events.head()

# Calculate the circular mean and vector length at spindles most prominent peak
print(events.groupby("Time")["SO_phase"].agg(pg.circ_mean).loc[0])
print(events.groupby("Time")["SO_phase"].agg(pg.circ_r).loc[0])





