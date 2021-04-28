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

#%%

def tfr_analysis(Data, l_freq = 0.5, h_freq = 30, steps = 0.25, method = 'wavelet', baseline=[-2.5, 2.5], 
                 mode='zscore', chan = 'C3', itc_calculation = 'tensorpac', plot=True, output='avg', 
                 zscore=False, cmap = cm.Spectral_r, save_path=None):
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
        DESCRIPTION. The default is [-2.5, 2.5].
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

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    itc_data : TYPE
        DESCRIPTION.
    Sxx : TYPE
        DESCRIPTION.

    """
    freqs = np.arange(l_freq, h_freq, steps)
    n_cyc = freqs + 0.5
    if method == 'wavelet':      
        Sxx, itc = mne.time_frequency.tfr_morlet(Data, freqs, n_cycles=n_cyc, picks = chan,
                                                 zero_mean=False, use_fft=True, decim=1, 
                                                 output='power', n_jobs=16, verbose=None)
        # Sxx.plot(picks='C3',baseline=(-2.5, 2.5), mode='zlogratio' , tmin=-2.5, tmax=2.5, fmin=0.5, 
        #           fmax=30, cmap='Spectral_r')
    elif method == 'multitaper':
        Sxx, itc = mne.time_frequency.tfr_multitaper(Data, freqs, n_cycles=n_cyc, 
                                                     use_fft=True, decim=1, picks = chan, 
                                                     n_jobs=16, verbose=None, time_bandwidth = 2.0)
    else:
        raise ValueError('Undefined method for TFR')
        
    # itc.apply_baseline(baseline, mode=None)   
    Sxx.apply_baseline(baseline, mode=mode)
    Sxx = Sxx._data[:,:,int(Data.info['sfreq']/2):int(len(Data.times))-int(Data.info['sfreq']/2)].squeeze()
    
    if output == 'power':
        # baseline correct window
        edge = int(Data.info['sfreq']/2)
        Sxx = Sxx[:,:,int(edge):int(max(np.shape(Sxx))-edge)]
        # baseline correct 
        Sxx = 10 * np.log(Sxx/np.median(Sxx, axis=-1, keepdims=True))
        if zscore:
              Sxx /= np.std(Sxx, axis=-1, keepdims=True)
        # mean trial
        Sxx = np.median(Sxx, axis=0)

        # plot
        print('creating new figure ...')
        fig, ax = plt.subplots()
        CM = ax.pcolormesh(Data.times[int(Data.info['sfreq']/2):int(max(np.shape(Sxx)))-int(Data.info['sfreq']/2)], 
                           freqs, Sxx[:,int(Data.info['sfreq']/2):int(max(np.shape(Sxx)))-int(Data.info['sfreq']/2)], 
                           shading='gouraud', cmap=cmap, rasterized = True, 
                           vmin=Sxx[:,int(Data.info['sfreq']/2):int(max(np.shape(Sxx)))-int(Data.info['sfreq']/2)].min(),
                           vmax=Sxx[:,int(Data.info['sfreq']/2):int(max(np.shape(Sxx)))-int(Data.info['sfreq']/2)].max())
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        cbar = plt.colorbar(CM, ax=ax)
        cbar.set_label(mode + ' (dB)', rotation=270)
        if save_path is not None:
            plt.savefig(save_path + '_tf_plot.png')

    elif output == 'avg': 
        print('creating new figure ...')
        fig, ax = plt.subplots()
        CM = ax.pcolormesh(np.linspace(-2.5, 2.5, max(np.shape(Sxx))), freqs, 
                           Sxx, 
                           shading='gouraud', cmap=cmap, rasterized = True, 
                           vmin=np.percentile(Sxx.min(),10),
                           vmax=np.percentile(Sxx.max(),90))
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        cbar = plt.colorbar(CM, ax=ax)
        cbar.set_label(mode + ' (dB)', rotation=270)
        if save_path is not None:
            plt.savefig(save_path + '_tf_plot.png')
    
    # plt.plot(Data.times[int(Data.sfreq/2):int(len(Data.times)-Data.sfreq/2)],
    #          Data.get_data(picks='C3').mean(0).squeeze()[int(Data.sfreq/2):int(len(Data.times)-Data.sfreq/2)]*1e6)
    
    # peak = PeakLockedTF(Data.get_data(picks = 'C3').squeeze()*1e6, sf=Data.sfreq, cue=0., 
    #                     times=Data.times, f_pha=[0.5, 30],f_amp=(0.5, 30, 0.5, 0.2), 
    #                     n_jobs=16, verbose=True)
    # peak.plot(zscore=True, baseline=(int(Data.sfreq/2), int(len(Data.times)-Data.sfreq/2)), 
    #           edges=256, cmap='Spectral_r')
    
    if itc_calculation == 'mne':
        itc_data = itc._data.squeeze()[:,int(Data.info['sfreq']/2):int(len(Data.times))-int(Data.info['sfreq']/2)]
    elif itc_calculation == 'tensorpac':
        itc = ITC(Data.get_data(picks = 'C3').squeeze()*1e6, 
                  sf=int(Data.info['sfreq']), f_pha=(l_freq, h_freq, l_freq, steps), 
                  dcomplex='wavelet', edges=int(Data.info['sfreq']/2))
        plt.figure()
        itc.plot(times = Data.times, cmap='plasma')
        plt.show()
        if save_path is not None:
            itc.savefig(save_path + '_itc_plot.png')
        itc_data = np.asarray(itc.itc.data.tolist())
        
    return itc_data, Sxx

# def tfr_plot(Data, Sxx, itc):
    
    
