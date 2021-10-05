# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 08:29:12 2020

@author: neuro
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sleepstim.sleep_funs import (load_xdf, channel_parser, bfr_butter_filt, thresholdcrossings)
from mne.filter import filter_data, notch_filter
from scipy import signal
import scipy
import time
import seaborn as sns 
import yasa
import mne
import heartpy as hp
sns.set(style='darkgrid', font_scale=1.2)

# parse data
stream_dict = load_xdf('/media/administrator/data/Study_1_data/Raw_data/Experimental/475MQ9BL_1/sleepstim_R001.xdf') 

# extract C3 data  
data = stream_dict['eego']['time_series']
eego_times = stream_dict['eego']['time_stamps']
times = eego_times - eego_times[0]
marker = stream_dict['reiz-marker']
pinknoise_times= [marker['time_stamps'][ix] for ix in range(len(marker['time_stamps'])) if marker['time_series'][ix][0] == 'pinknoise'] - eego_times[0]
if np.all(np.diff(pinknoise_times[0:3]) < 0.5):
    pinknoise_times_sync = [np.argmin(np.abs(times - ts)) for ts in pinknoise_times[3::]] 
else:
    pinknoise_times_sync = [np.argmin(np.abs(times - ts)) for ts in pinknoise_times] 
info = stream_dict['eego']['info'] 
sf = int(info['nominal_srate'][0])
ch_names, ch_types = channel_parser(info, data) 
C3_og = data[:,ch_names.index('C3')]*1e6
# delete stream_dict/data for memory sake
del stream_dict, data, ch_names, ch_types
  
# filter data for later
C3 = C3_og[pinknoise_times_sync[0]::]
new_times = times[pinknoise_times_sync[0]::]
C3_filt = filter_data(C3.astype('float64'), sfreq=sf, l_freq=.3, h_freq=4.0, method='fir')
C3_filt_notch = notch_filter(C3_filt, Fs=sf, method='spectrum_fit', freqs=np.arange(50,50*2+1,50), 
                             mt_bandwidth=2, p_value=0.01, filter_length='10s')

#%%
## Imitate sliding-window procedure of original experiment
crit_reconstruct = [0]; crit_reconstruct_up = [0]
time_reconstruct = [0]; time_reconstruct_up = [0]
fs = 512
ix = 30*fs

b,a = signal.butter(4, 4, fs = fs)
# fnyq = fs/2
# N, beta = signal.kaiserord(60.0, 5/fnyq)
# taps = signal.firwin(N, 4/fnyq, window=('kaiser', beta))
while ix < len(C3):
    ix0 = ix - 30*fs
    #pick 30 s window
    window = C3[int(ix0):int(ix)]
    d = signal.filtfilt(b,a,window)
    #d = signal.filtfilt(taps, 1.0, window)
    d -= np.median(d)
    
    # # median filter
    # d = scipy.ndimage.median_filter(d, size=50)
    # # savitzky golay filter
    # d = hp.smooth_signal(d, sample_rate = fs, window_length=int(fs*1), polyorder=3)
    # sin convolution
    # d = np.convolve(np.sin(1), d)
    
    minamp = min(np.percentile((d[-2* int(sf):]), 10), -35)
    crit = min(d[int(-0.02*sf):])

    # if (min(d[-2:]) < -35) and (new_times[ix] - time_reconstruct[-1] > 3):
    if crit < minamp and (new_times[ix] - time_reconstruct[-1] > 4): 
        time_reconstruct.append(new_times[ix])
        crit_reconstruct.append(crit)
        ts_up = new_times[ix] + .610
        time_reconstruct_up.append(ts_up)
        crit_reconstruct_up.append(d[int(ts_up)])
    ix += 4
    
    if len(crit_reconstruct) > 10:
        break

#%%
## Plot results
plt.figure()
plt.plot(new_times, C3_filt_notch)
plt.plot(time_reconstruct[1::], crit_reconstruct[1::], 'xr')
plt.plot(time_reconstruct_up[1::], crit_reconstruct_up[1::], 'xg')

#%%
## Epoch plotting with MNE
ts_pinknoise_times_sync = [np.argmin(np.abs(new_times - ts)) for ts in time_reconstruct]
ts_pinknoise_times_sync_up = [np.argmin(np.abs(new_times - ts)) for ts in time_reconstruct_up]

center_crit, _ = yasa.get_centered_indices(C3_filt_notch, np.asarray(ts_pinknoise_times_sync[1::]), 
                                      npts_before = sf*3, npts_after = sf*3)
info = mne.create_info(ch_names=1, sfreq=sf, ch_types='eeg')
epochs_crit = mne.EpochsArray(np.expand_dims(C3_filt_notch[center_crit], 1)/1e6, info, tmin = -3, 
                         baseline=(None), proj=False)
# epochs_crit.average(method='mean').plot()
epochs_crit.plot_image()

center_up, _ = yasa.get_centered_indices(C3_filt_notch, np.asarray(ts_pinknoise_times_sync_up[1::]), 
                                         npts_before = sf*3, npts_after = sf*3)
info = mne.create_info(ch_names=1, sfreq=sf, ch_types='eeg')
epochs_up = mne.EpochsArray(np.expand_dims(C3_filt_notch[center_up], 1)/1e6, info, tmin = -3, 
                         baseline=(None), proj=False)
# epochs_up.average(method='mean').plot()
epochs_up.plot_image()
