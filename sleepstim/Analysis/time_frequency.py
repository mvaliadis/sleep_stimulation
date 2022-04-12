#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:25:47 2021

@author: administrator
"""

import mne
import numpy as np
from tensorpac.utils import PeakLockedTF, PSD, ITC, BinAmplitude
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
#%%

def tfr_analysis(Data, l_freq = 0.5, h_freq = 30, steps = 0.25, method = 'wavelet', baseline=[-3, 3], 
                 mode='zscore', chan = 'C3', itc_calculation = 'tensorpac', plot=True, output='avg', 
                 cmap = cm.Spectral_r, save_path=None):
    """

    Parameters
    ----------
    Data : TYPE: mne object
        DESCRIPTION.
    l_freq : TYPE, optional
        DESCRIPTION. The default is 0.5.
    h_freq : TYPE, optional
        DESCRIPTION. The default is 30.
    steps : TYPE, optional
        DESCRIPTION. The default is 0.2.
    method : TYPE, optional
        DESCRIPTION. The default is 'wavelet'.
    baseline : TYPE, optional
        DESCRIPTION. The default is [-3, 3].
    mode : TYPE, optional
        DESCRIPTION. The default is 'zscore'.
    chan : TYPE, optional
        DESCRIPTION. The default is 'C3'.
    itc_calculation : TYPE, optional
        DESCRIPTION. The default is 'tensorpac'.
    plot : TYPE, optional
        DESCRIPTION. The default is True.
    output : TYPE, optional
        DESCRIPTION. The default is 'avg'.

    Returns
    -------
    itc_data : TYPE
        DESCRIPTION.
    Sxx : TYPE
        DESCRIPTION.

    """
    freqs = np.arange(l_freq, h_freq+steps, steps)
    n_cyc = freqs
    if method == 'wavelet':      
        Sxx = mne.time_frequency.tfr_morlet(Data, freqs, n_cycles=n_cyc, picks = chan,
                                            zero_mean=True, use_fft=True, decim=5, 
                                            output='power', n_jobs=16, verbose=None,
                                            average=False, return_itc=False)
    elif method == 'multitaper':
        Sxx = mne.time_frequency.tfr_multitaper(Data, freqs, n_cycles=n_cyc, use_fft=True,
                                                decim=5, picks = chan, return_itc=False,
                                                n_jobs=16, verbose=None, average = False)
    else:
        raise ValueError('Undefined method for TFR')
        
    Sxx.apply_baseline(baseline, mode=mode)
    # Sxx.average().plot(tmin=-2.5, tmax=2.5, fmin=0.5, fmax=20, cmap='Spectral_r')
    Sxx_ = Sxx.average().data[:,:,int(Sxx.info['sfreq']*.5):int(Sxx.info['sfreq']*5.5)].squeeze()
    times = Sxx.times[int(Sxx.info['sfreq']*.5):int(Sxx.info['sfreq']*5.5)]

    if output == 'avg': 
        print('creating new figure ...')
        fig, ax = plt.subplots()
        vmin, vmax = np.percentile(Sxx_, [0 + 0.1, 100 - 0.1])
        norm = Normalize(vmin=vmin, vmax=vmax)
        CM = ax.pcolormesh(times, freqs, Sxx_, 
                           shading='gouraud', cmap=cmap, rasterized = True, 
                           norm = norm, antialiased=True, vmin=vmin, vmax=vmax)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        cbar = plt.colorbar(CM, ax=ax)
        cbar.set_label(mode + ' (dB)', rotation=270)
        if save_path is not None:
            plt.savefig(save_path + '_tf_plot.png')
    
    
    if itc_calculation == 'tensorpac':
        itc = ITC(Data.get_data(picks = 'C3').squeeze()*1e6,
                  sf=int(Data.info['sfreq']), f_pha=(l_freq, h_freq, l_freq, steps), 
                  dcomplex='wavelet', edges=int(Data.info['sfreq']/2))
        plt.figure()
        itc.plot(times = Data.times, cmap='plasma')
        plt.show()
        if save_path is not None:
            itc.savefig(save_path + '_itc_plot.png')
        itc_data = np.asarray(itc.itc.data.tolist())
        
    elif itc_calculation == None:
        itc_data = []
          
    return itc_data, Sxx