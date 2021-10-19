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

#%%
def process_rawXDF(file):
    streams = liesl.XDFFile(file)
    
    ## read marker information
    marker = streams['reiz-marker']
    TMStimes = [marker.time_stamps[i] for i,v in enumerate(marker.time_series) if \
                'cse_' in v[0]][1::]
        
    ## read EEG/EMG data
    eego = streams["eego"]
    
    ## 
    sf = eego.nominal_srate
    
    ##
    edcix = eego.channel_labels.index("chan_7")
    c3ix = eego.channel_labels.index("C3")
    
    ## match clocks
    eegoTMStimes = [np.argmin(np.abs(eego.time_stamps - ts)) for ts in TMStimes]
        
    ## get EMG data
    edcdat = eego.time_series[:,edcix]*1e6
    c3dat = eego.time_series[:,c3ix]*1e6
    # b, a = signal.iircomb(w0=50, Q=10, ftype='notch', fs=sf)
    # edcdat_f = signal.filtfilt(b, a, edcdat)
    # c3dat_f = signal.filtfilt(b, a, c3dat)
    # d, c = signal.iircomb(w0=100, Q=10, ftype='notch', fs=sf)
    # edcdat_f = signal.filtfilt(d, c, edcdat_f)
    # c3dat_f = signal.filtfilt(d, c, c3dat_f)
    
    ## Plot CMC between EDC_R & C3
    cmc = CMC(edcdat, c3dat, sf=sf, l_foi=10, h_foi=30, plot=True)
    
    ### Plot TMS artifact 
    # print(TMStimes)
    # print(eego.time_stamps[eegoTMStimes])
    # print(TMStimes - eego.time_stamps[eegoTMStimes])
    # plt.figure()
    # plt.plot(eego.time_stamps,c3dat)
    # plt.plot(TMStimes, [0]*len(TMStimes), 'or')
    
    edcepochs = np.nan*np.zeros((len(eegoTMStimes), 200))
    for i, ts in enumerate(eegoTMStimes):
        c3ep = np.abs(c3dat[ts-100:ts+100])
        # tartifact = np.argmax(c3ep)
        tartifact = np.argmax(tkeo(c3ep, plot=True))
        ts_corrected = (ts-100)+tartifact
        edcepochs[i,:] = edcdat[ts_corrected-100:ts_corrected+100]
        
    Vpp = np.ptp(edcepochs[:, 105:160], axis = 1)
    
    # ## Plot EDC response
    # plt.figure()
    # plt.title(" ".join(file.split('/')[-2::]))
    # plt.plot(np.arange(-max(edcepochs.shape)/2, max(edcepochs.shape)/2)/sf, np.mean(edcepochs,0))
    
    return edcepochs, Vpp, sf, cmc

#%%

maindir = '/media/administrator/data/Study_1_data/Pre_post_data/'
files = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(maindir) for i in files if 'cse' in i])
results = defaultdict(lambda: [])
for file in files:
    print(file)
    subjname = str(file).split("/")[-2]
    condition = str(file).split('.')[0].split('/')[-1]
    edcepochs, Vpp, sf, cmc = process_rawXDF(file)
    results["Vpp"].extend(Vpp)
    results["Subject"].extend([subjname]*len(Vpp))
    results["Condition"].extend([condition]*len(Vpp))
    results["CMC"].extend([cmc]*len(Vpp))

    plt.figure()
    plt.title(" ".join(file.split('/')[-2::]))
    plt.plot(np.arange(-max(edcepochs.shape)/2, max(edcepochs.shape)/2)/sf, np.mean(edcepochs,0))
    plt.vlines(0.005, ymin=np.mean(edcepochs,0).min(), ymax=np.mean(edcepochs,0).max())
    plt.vlines(0.060, ymin=np.mean(edcepochs,0).min(), ymax=np.mean(edcepochs,0).max())
