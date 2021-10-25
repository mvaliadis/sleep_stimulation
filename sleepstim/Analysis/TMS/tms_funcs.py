#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:46:16 2021

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
import scipy.stats as stats 
import mne 

#%%
def tkeo(data, plot=True):
    """
    Basic z-scored and normalized Teager-Kaiser Energy Operator
    """
    emgf = np.zeros((len(data)))
    # Yt = Xt^2 - Xt-1 * Xt+1
    for i in range(1,len(emgf)-1):
        emgf[i] = data[i]**2 - data[i-1]*data[i+1]
    # convert original signal to z-score from time-zero
    emgZ = stats.zscore(data)
    # same for filtered signal energy
    emgE = stats.zscore(emgf)
    # plot "raw"
    if plot:       
        plt.figure()
        plt.plot(emgZ)
        plt.plot(emgE,'m')
        plt.title("TKEO energy plotted over EMG/EEG")
        plt.xlabel('Time')
        plt.ylabel('Energy')
        # plt.legend()
        plt.show()
    return emgE 

def CMC(signal1, signal2, sf, l_foi=2, h_foi=40, plot=True): 
    if plot:
        plt.figure()
        coh, f = plt.cohere(signal1, signal2, NFFT=int((2/l_foi)*sf), Fs=sf)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Coherence')
        plt.title('CMC between C3 and EDC_R')
        plt.xlim(l_foi, h_foi)
        plt.show()
    else:
        f, coh = signal.coherence(signal1, signal2, fs=sf, nperseg=(2/l_foi)*sf, 
                                  detrend='constant')
        
    return f[l_foi:h_foi], coh[l_foi:h_foi]   

def process_rawXDF(file):   
    # load data
    streams = liesl.XDFFile(file)
    
    ## read EEG/EMG data
    eego = streams["eego"]
    sf = eego.nominal_srate
    
    ## read marker information
    marker = streams['reiz-marker']
    TMStimes = [marker.time_stamps[i] for i,v in enumerate(marker.time_series) if \
                'cse_' in v[0]][1::]
    
    ## checks here for data length and if intensity % given for pulse
    if eego.time_series.shape[0]/sf < 50:
        return [], np.empty(len(TMStimes)), sf, [], [], np.empty(len(TMStimes))
    if marker.time_series[0][0] != '':
        intensity = int(marker.time_series[0][0].split('_')[-1])
    else:
        intensity = []
        return [], np.empty(len(TMStimes)), sf, [], [], np.empty(len(TMStimes))

    ## label data
    edcix = eego.channel_labels.index("chan_7")
    c3ix = eego.channel_labels.index("C3")
    
    ## match clocks
    eegoTMStimes = [np.argmin(np.abs(eego.time_stamps - ts)) for ts in TMStimes]
        
    ## get EMG/C3 data
    edcdat = eego.time_series[:,edcix]*1e6
    # edcdat = mne.filter.notch_filter(edcdat.astype('float64'),sf,freqs=(50,100,150,200)) 
    c3dat = eego.time_series[:,c3ix]*1e6
    # c3dat = mne.filter.notch_filter(c3dat.astype('float64'),sf,freqs=(50,100,150,200))

    ## CMC between EDC_R & C3
    cmc = CMC(edcdat, c3dat, sf=sf, l_foi=10, h_foi=30, plot=False)
    
    # ## Plot TMS artifact 
    # plt.figure()
    # plt.plot(eego.time_stamps,c3dat)
    # plt.plot(TMStimes, [0]*len(TMStimes), 'or')
    
    edcepochs = np.nan*np.zeros((len(eegoTMStimes), 200))
    for i, ts in enumerate(eegoTMStimes):
        c3ep = np.abs(c3dat[ts-100:ts+100])
        # use TKEO to determine peak coil artifact location in EEG
        tartifact = np.argmax(tkeo(c3ep, plot=False))
        ts_corrected = (ts-100)+tartifact
        # if any timestamp doesn't last 200 time points, correct for this and add nans to trial
        last_ts = int((eego.time_stamps - eego.time_stamps[0])[-1] * sf)
        if ts_corrected+100 > last_ts:
            add_nans = np.nan*np.zeros(ts_corrected+100 - last_ts)
            edcepochs[i,:] = np.concatenate([edcdat[ts_corrected-100:last_ts] - np.median(edcdat[ts_corrected-100:last_ts]), add_nans], axis=0)
        else:
            edcepochs[i,:] = edcdat[ts_corrected-100:ts_corrected+100] - np.median(edcdat[ts_corrected-100:ts_corrected+100])
    
    # if less than 20 trials, add nans to remaining VPPs
    if min(edcepochs.shape) < 21:
        add_nans = np.nan*np.zeros(20 - min(edcepochs.shape))
        Vpp = np.concatenate([np.ptp(edcepochs[:, 105:160], axis = 1), add_nans], axis=0)
    else:
        Vpp = np.ptp(edcepochs[:, 105:160], axis = 1)
      
    # flag MEPs that don't exceed 3 std of pre-stimulus signal
    pre_thres_bool = 3*np.std(abs(edcepochs[:, 0:100]), axis=1) < Vpp
    
    # ## Plot EDC response
    # plt.figure()
    # plt.title(" ".join(file.split('/')[-2::]))
    # plt.plot(np.arange(-max(edcepochs.shape)/2, max(edcepochs.shape)/2)/sf, np.mean(edcepochs,0))
    
    return edcepochs, Vpp, sf, cmc, intensity, pre_thres_bool

#%%

maindir = '/media/administrator/data/Study_1_data/Pre_post_data/'
files = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(maindir) for i in files if 'cse' in i])
results = defaultdict(lambda: [])
for file in files:
    print(file)
    subjname = str(file).split("/")[-2]             
    condition = str(file).split('.')[0].split('/')[-1]
    edcepochs, Vpp, sf, cmc, intensity, pre_thres_bool = process_rawXDF(file)
    results["Vpp"].extend(Vpp)
    results["Vpp_True"].extend(pre_thres_bool)
    results["logVpp"].extend(np.log(Vpp))
    results["Intensity"].extend([intensity])
    results["Subject"].extend([subjname]*len(Vpp))
    results["Condition"].extend([condition]*len(Vpp))
    # results["CMC"].extend([cmc]*len(Vpp))

    plt.figure()
    plt.title(" ".join(file.split('/')[-2::]))
    plt.plot(np.arange(-max(edcepochs.shape)/2, max(edcepochs.shape)/2)/sf, np.mean(edcepochs,0))
    plt.vlines(0.005, ymin=np.mean(edcepochs,0).min(), ymax=np.mean(edcepochs,0).max())
    plt.vlines(0.060, ymin=np.mean(edcepochs,0).min(), ymax=np.mean(edcepochs,0).max())
