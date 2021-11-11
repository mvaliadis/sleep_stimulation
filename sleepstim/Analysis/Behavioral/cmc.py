#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:43:22 2021

@author: administrator
"""

import liesl
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os 
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal 
from mne.time_frequency import csd_array_multitaper
from mne_connectivity import spectral_connectivity
import scipy.stats as stats 
import mne 
import neurokit2 as nk
from sleepstim.sleep_funs import (downsample_scaled, load_xdf)
from sleepstim.Analysis.TMS.tms_funcs import tkeo
import yasa 
import seaborn as sns
from detecta import detect_onset
from scipy.integrate import trapz, cumtrapz
from lspopt import spectrogram_lspopt
from tqdm import tqdm
import pandas as pd
import finn.same_frequency_coupling.time_domain.magnitude_squared_coherency as ms_coh
sns.set_theme(color_codes=True)


def CMC(signal1, signal2, sf, foi=(2,40), plot=True, method='welch'): 
    if method=='multitaper':
        x = np.expand_dims(np.vstack([signal1, signal2]), 0)
        csd = csd_array_multitaper(x, sf, fmin=foi[0], fmax=foi[1],
                                    n_fft=None,
                                    n_jobs=1, verbose=0)
               
        Pxx, freqs_s1 = mne.time_frequency.psd_array_multitaper(signal1, sf, fmin=foi[0], fmax=foi[1], 
                                                                normalization='full', verbose=0)

        Pyy, freqs_s2 = mne.time_frequency.psd_array_multitaper(signal2, sf, fmin=foi[0], fmax=foi[1], 
                                                                normalization='full', verbose=0)

        idx_band = np.logical_and(freqs_s1 >= 2, freqs_s1 <= 40)
               
        Cxy = np.abs(csd._data[1, idx_band])**2/ Pxx[idx_band] / Pyy[idx_band]
    
        norm_Cxy = (Cxy - Cxy.min()) / (Cxy.max() - Cxy.min())
    
        return freqs_s1[idx_band], norm_Cxy[idx_band]

        
    if method=='welch':
        if plot:
            plt.figure()
            coh, f = plt.cohere(signal1, signal2, NFFT=int((2/foi[0])*sf), Fs=sf)
            plt.xlabel('frequency [Hz]')
            plt.ylabel('Coherence')
            plt.title('CMC between C3 and EDC_R')
            plt.xlim(foi[0], foi[1])
            plt.show()
        else:
            f, coh = signal.coherence(signal1, signal2, fs=sf, nperseg=(2/foi[0])*sf, 
                                      detrend='constant')
      
    foi_idx = np.logical_and(f >= foi[0], f <= foi[1])
    
    return f[foi_idx], coh[foi_idx]   

  
    
val = [signal.coherence(epoch[i,0,:], epoch[i,1,:], fs=sf, nperseg=500, 
                        detrend='constant') for i in range(epoch.shape[0])]

f_s, cohs = [],[]
for i in range(len(val)):
    f_s.append(val[i][0])
    cohs.append(val[i][1])
    

plt.xlim(2,40)

def channel_parser(chans, chtypes):
       bipolar_names = {'chan_1': 'EDC_L', 'chan_2': 'ECR_L', 'chan_3': 'FCR_L', 
                        'chan_4': 'FDS_L', 'chan_5': 'ECG', 'chan_6': 'EmptyChan', 
                        'chan_7': 'EDC_R', 'chan_8': 'ECR_R', 'chan_9': 'FCR_R', 
                        'chan_10': 'FDS_R'}
       bipolar_types = {'EDC_L' : 'emg', 'ECR_L' : 'emg', 'FCR_L': 'emg', 
                        'FDS_L': 'emg', 'ECG': 'ecg', 'EmptyChan': 'misc', 
                        'EDC_R': 'emg', 'ECR_R': 'emg', 'FCR_R': 'emg', 
                        'FDS_R': 'emg'}
       chans[64::] = list(bipolar_names.values())
       chtypes[0:64] = ['eeg'] * 64
       chtypes[64::] = list(bipolar_types.values())
       return chans, chtypes
  
def linear_envelope(x, sf=1000, fc_bp=[10, 400], fc_lp=8):
    r"""Calculate the linear envelope of a signal.

    Parameters
    ----------
    x     : 1D array_like
            raw signal
    freq  : number
            sampling frequency
    fc_bp : list [fc_h, fc_l], optional
            cutoff frequencies for the band-pass filter (in Hz)
    fc_lp : number, optional
            cutoff frequency for the low-pass filter (in Hz)

    Returns
    -------
    x     : 1D array_like
            linear envelope of the signal

    Notes
    -----
    A 2nd-order Butterworth filter with zero lag is used for the filtering.  

    See this notebook [1]_.

    References
    ----------
    .. [1] https://github.com/demotu/BMC/blob/master/notebooks/Electromyography.ipynb

    """
    
    import numpy as np
    from scipy.signal import butter, filtfilt
    
    if np.size(fc_bp) == 2:
        # band-pass filter
        b, a = butter(2, fc_bp, fs = sf, btype = 'bandpass')
        x = filtfilt(b, a, x)
    if np.size(fc_lp) == 1:
        # full-wave rectification
        x = abs(x)
        # low-pass Butterworth filter
        b, a = butter(2, fc_lp, fs = sf, btype = 'low')
        x = filtfilt(b, a, x)
    
    return x

def integration_time_reset(data, sf=1000, plot=True):
    eegotimes = np.arange(len(data))/sf
    nreset = 400 # reset after this amount of samples
    area = []
    for i in range(int(np.ceil(np.size(data)/nreset))):
        area = np.hstack((area, cumtrapz(data[i*nreset:(i+1)*nreset], initial=0)/sf))
    
    if plot:
        # plot data
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
        ax1.plot(eegotimes, data, 'r')
        ax1.set_title('EMG signal (linear envelope)')
        ax1.set_ylabel('EMG amplitude [V]')
        ax1.set_xlim(eegotimes[0], eegotimes[-1])
        ax2.plot(eegotimes, area, 'y', label='Trapezoid')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Integral of the EMG signal with time reset (t = %s ms)' %nreset)
        ax2.set_ylabel('EMG integral [Vs]')
        plt.locator_params(axis='both', nbins=4)
        plt.tight_layout()
        
    return area

def plot_spectrogram(data, sf, foi=(10,200), method='stft', dB=False):
    """

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    sf : TYPE
        DESCRIPTION.
    foi : TYPE, optional
        DESCRIPTION. The default is (10,200).
    method : TYPE, optional
        DESCRIPTION. The default is 'stft'.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))
    if method=='stft':
        f, t, Zxx = signal.stft(data, fs=sf, nperseg=int((2/foi[0])*sf), noverlap = 64)
    elif method=='multitaper':
        f, t, Zxx = spectrogram_lspopt(data, fs=sf, nperseg=int((2/foi[0])*sf), 
                                       c_parameter=20.0, noverlap = 64)
    if dB:
        Zxx = np.log10(Zxx)
    ## Plotting
    foi_idx = np.logical_and(f >= foi[0], f <= foi[1])
    ax1.pcolormesh(t, f[foi_idx], np.abs(Zxx)[foi_idx,:], vmin=np.percentile(np.abs(Zxx)[foi_idx,:], 5), 
                   vmax=np.percentile(np.abs(Zxx)[foi_idx,:], 95), shading='gouraud', cmap=plt.cm.Spectral_r)
    if method=='stft':
        ax1.set_title('Short-Time Fourier Transform Spectrogram')
    elif method=='multitaper':
        ax1.set_title('Multitaper Spectrogram')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_xlim(t[0], t[-1])
    plt.tight_layout()

    return fig   
           
def process_rawXDF_CMC(file): 
    # load data
    streams = liesl.XDFFile(file)
    
    ## read EEG/EMG data
    eego = streams["eego"]
    sf = eego.nominal_srate
        
    ## get clocks
    eegotimes = eego.time_stamps - eego.time_stamps[0]
    slalom_start = [] #emg movement onset of size x 

    ## adjust chan info
    chans, chtypes = channel_parser(eego.channel_labels, eego.channel_types)
     
    ## label data
    edcix = chans.index("EDC_R")
    fdsix = chans.index("FDS_R")
    c3ix = chans.index("C3")
    
    ## get EMG/C3 data
    edcdat = eego.time_series[:,edcix]*1e6
    edcdat = mne.filter.notch_filter(edcdat.astype('float64'),sf,freqs=(50,100,150,200), method = 'spectrum_fit', verbose = 0) 
    fdsdat = eego.time_series[:,fdsix]*1e6
    fdsdat = mne.filter.notch_filter(fdsdat.astype('float64'),sf,freqs=(50,100,150,200), method = 'spectrum_fit', verbose = 0) 
    c3dat = eego.time_series[:,c3ix]*1e6
    c3dat = mne.filter.notch_filter(c3dat.astype('float64'),sf,freqs=(50,100,150,200), method = 'spectrum_fit', verbose = 0)
    c3dat = mne.filter.filter_data(c3dat, sfreq=1000, l_freq=1, h_freq=45, verbose=0)
    ecgdat = eego.time_series[:,chans.index('ECG')]*1e6
    ecgdat = mne.filter.notch_filter(ecgdat.astype('float64'),sf,freqs=(50,100,150,200), method = 'spectrum_fit', verbose = 0)

    ## CMC between EDC_R & C3
    cmc = CMC(edcdat, c3dat, sf=sf, foi=(2,40), plot=True)
    
    edcdat = linear_envelope(edcdat, sf=1000, fc_bp=[10, 400], fc_lp=8)
    
    return edcdat, cmc 

def emg_onset(data, sf, envelope=False, use_tkeo=False):
    if envelope:
        data_le = linear_envelope(data, freq=sf, fc_bp=[10, 400], fc_lp=8)
    else:
        data_le = data 
    if use_tkeo:
        data_le = abs(tkeo(data_le, normalize=True, plot=True))
        threshold = 0.5*np.std(data_le)
    else:
        threshold = 2*np.std(data_le)
    inds = detect_onset(data_le, threshold=threshold,n_above=50, n_below=10, show=True)
    return inds 

#%%
from neurodsp.rhythm import compute_lagged_coherence
from neurodsp.plts.rhythm import plot_lagged_coherence

# lag_coh_by_f, freqs = compute_lagged_coherence(c3dat, sf, (2, 40),
#                                                 return_spectrum=True)
# plot_lagged_coherence(freqs, lag_coh_by_f)

# lag_coh_by_f, freqs = compute_lagged_coherence(signal2, sf, (2, 40),
#                                                return_spectrum=True)
# plot_lagged_coherence(freqs, lag_coh_by_f)

# ## epoched correlation
# t, ep1 = yasa.sliding_window(signal1, sf, window=2)
# t, ep2 = yasa.sliding_window(signal2, sf, window=2)
# corr_stats = [stats.pearsonr(ep1[i,:], ep2[i,:])[1] for i in range(ep1.shape[0])]


# ## compute time shift based on correlation 
# y1 = signal2
# y2 = signal1
# sr = sf
# n = len(y1)

# corr = signal.correlate(y2, y1, mode='same') 
# corr /= np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])
# delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
# delay = delay_arr[np.argmax(corr)]
# print('y2 is ' + str(delay) + ' behind y1')

# plt.figure()
# plt.plot(delay_arr, corr)
# plt.title('Lag: ' + str(np.round(delay, 3)) + ' s')
# plt.xlabel('Lag')
# plt.ylabel('Correlation coeff')
# plt.tight_layout()
# plt.show()


#%%
# file = '/media/administrator/data/Study_1_data/Pre_post_data/YIOYSRPX/cond_0_slalom_post_R001.xdf'
# slalom_file1 = '/media/administrator/data/Study_1_data/Pre_post_data/YIOYSRPX/postYIOYSRPX_0_0.csv'
# slalom_file2 = '/media/administrator/data/Study_1_data/Pre_post_data/YIOYSRPX/postYIOYSRPX_0_1.csv'
# slalom_file3 = '/media/administrator/data/Study_1_data/Pre_post_data/YIOYSRPX/postYIOYSRPX_0_2.csv'
# slalom_file4 = '/media/administrator/data/Study_1_data/Pre_post_data/YIOYSRPX/postYIOYSRPX_0_3.csv'  

# slalom_1, slalom_2 = np.genfromtxt(slalom_file1, delimiter=','), np.genfromtxt(slalom_file2, delimiter=',')
# slalom_3, slalom_4 = np.genfromtxt(slalom_file3, delimiter=','), np.genfromtxt(slalom_file4, delimiter=',')
# pause = np.zeros(10*60) #approximately

# slaloms = [stats.zscore(slalom[0][0,:]) for slalom in zip((slalom_1, slalom_2, slalom_3, slalom_4))]
# slaloms = np.concatenate([pause, slaloms[0], pause, slaloms[1], pause, slaloms[2], pause, slaloms[3], pause])
            

# data_le = linear_envelope(edcdat, freq=sf, fc_bp=[10, 400], fc_lp=8)
# data_le = downsample_scaled(np.expand_dims(data_le,1), 1000, 100).squeeze()
# inds = emg_onset(data_le, envelope=False, use_tkeo=False)
# plot_spectrogram(edcdat, sf, foi=(10,200), method='multitaper', dB=False)


#%%

maindir = '/media/administrator/data/Study_1_data/Pre_post_data/'
def cmc_results(maindir):  
    files = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(maindir) for i in files if 'slalom_pre' in i or 'slalom_post' in i])  
    results = defaultdict(lambda: [])
    # files = files[30:40]
    for file in tqdm(files):
        print(file)
        # subjname = str(file).split("/")[-2]             
        # night = " ".join(str(file).split('.')[0].split('/')[-1].split('_')[0:2])
        # condition = subject_cond_parser(file, study_phase='tms')
        name = str(file).split('.')[0].split('/')[-1]
        session = name.split('_')[-2]
        # rec = name.split('_')[-1]
        edcdat, cmc = process_rawXDF_CMC(file)
        # integration_time_reset(edcdat, sf=1000, plot=True)
        results["CMC"].extend([cmc])
        # results["Subject"].extend([subjname])
        # results["Night"].extend([night])
        # results["Condition"].extend([condition])
        results["Session"].extend([session])
        # results["File"].extend([rec])

    return results

#%%
results = cmc_results(maindir)
rs = pd.DataFrame([results['CMC'][i][1] for i in range(len(results['CMC']))])
rs['Session'] = results['Session']
freqs = results['CMC'][0][0]

pre = rs[rs['Session'] == 'pre']
post = rs[rs['Session'] == 'post']

plt.figure()
plt.plot(freqs, pre.to_numpy(dtype=object)[:,0:-1].mean(0), label='pre mean')
plt.plot(freqs, post.to_numpy(dtype=object)[:,0:-1].mean(0), label='post mean')
plt.plot(freqs, np.median(pre.to_numpy(dtype=object)[:,0:-1], axis=0), label='pre median')
plt.plot(freqs, np.median(post.to_numpy(dtype=object)[:,0:-1], axis=0), label='post median')
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')
plt.title('CMC between C3 and EDC_R')
plt.xlim(freqs[0], freqs[-1])
plt.legend()
plt.show()

plt.figure()
plt.plot(freqs, rs.to_numpy()[1,:])