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
import emd
import scipy.stats as stats
from scipy.fftpack import next_fast_len
from scipy import signal 

#%%

def tfr_analysis(Data, l_freq = 0.5, h_freq = 30, steps = 0.25, method = 'wavelet', baseline=(-4,4), 
                 mode='zscore', chan = 'C3', itc_calculation = 'tensorpac', length=[-3,3], 
                 plot=True, cmap = cm.Spectral_r, save_path=None, inst_power=False, band=None):            

    """

    Parameters
    ----------
    Data : TYPE: mne object
        DESCRIPTION.
    l_freq : TYPE, optional
        DESCRIPTION. The default is 1.
    h_freq : TYPE, optional
        DESCRIPTION. The default is 30.
    steps : TYPE, optional
        DESCRIPTION. The default is 0.25.
    method : TYPE, optional
        DESCRIPTION. The default is 'wavelet'.
    baseline : TYPE, optional
        DESCRIPTION. The default is [-4, 4].
    mode : TYPE, optional
        DESCRIPTION. The default is 'zscore'.
    chan : TYPE, optional
        DESCRIPTION. The default is 'C3'.
    itc_calculation : TYPE, optional
        DESCRIPTION. The default is 'tensorpac'.
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    itc_data : TYPE
        DESCRIPTION.
    Sxx : TYPE
        DESCRIPTION.

    """
    
    # select contramotor channels, all, or given channel
    if chan == 'CM_area':
        chan = ['F3','C3','Cz','T7','P3']
    elif chan == 'all':
        try:
            chan = Data[0].pick('eeg').ch_names
        except:
            chan = Data.pick('eeg').ch_names
    else:
        chan = chan
    
    # give frequencies to determine # of cycles 
    freqs = np.arange(l_freq, h_freq+steps, steps)
    n_cyc = freqs
    
    # compute tfr via wavelet method - multitaper noqa
    if method == 'wavelet':    
        try:
            Sxx = mne.time_frequency.tfr_morlet(Data, freqs, n_cycles=n_cyc, picks = chan,
                                            zero_mean=True, use_fft=True, decim=5, 
                                            output='power', n_jobs=16, verbose=None,
                                            average=False, return_itc=False)
        except:
            Sxx = mne.time_frequency.tfr_morlet(Data[0], freqs, n_cycles=n_cyc, picks = chan,
                                            zero_mean=True, use_fft=True, decim=5, 
                                            output='power', n_jobs=16, verbose=None,
                                            average=False, return_itc=False)

    elif method == 'multitaper':
        Sxx = mne.time_frequency.tfr_multitaper(Data[0], freqs, n_cycles=n_cyc, use_fft=True,
                                                decim=5, picks = chan, return_itc=False,
                                                n_jobs=16, verbose=None, average = False)
    else:
        raise ValueError('Undefined method for TFR')
        
    # apply baseline correction
    Sxx.apply_baseline(baseline, mode=mode)
    # average TFR data, remove edge, and alter timepoints accordingly 
    Sxx_ = Sxx.copy().pick(chan).crop(tmin=length[0],tmax=length[1]).average().data.mean(0) #mean over channels
    times = Sxx.copy().crop(tmin=length[0],tmax=length[1]).times
    
    # Plot TFR
    if plot:
        if inst_power:
            band_corr = {'Delta': [0.5, 4],
                         'Theta': [4, 8],
                         'Slow_Spindle': [8, 12],
                         'Fast_Spindle': [12, 16],
                         'Spindle': [9, 16],
                         'Beta': [16, 30]}
            fig, axs = plt.subplots(2, figsize=(10, 8))
            axs[0].set_title('Average TFR Plot')
            vmin, vmax = np.percentile(Sxx_, [0 + 0.1, 100 - 0.1])
            norm = Normalize(vmin=vmin, vmax=vmax)
            CM = axs[0].pcolormesh(times, freqs, Sxx_, shading='gouraud', 
                                   cmap=cmap, rasterized = True, 
                                   antialiased=True, vmin=vmin, vmax=vmax)
            axs[0].set_ylabel('Frequency (Hz)')
            # cbar = plt.colorbar(CM, ax=axs[0])
            # cbar.set_label(mode, rotation=270)
            
            
            axs[1].set_title(f'Instantaneous Power')
            t_min = (np.abs(Data.times - length[0])).argmin()
            t_max = (np.abs(Data.times - length[1])).argmin()
            for bands in ['Delta', 'Spindle', 'Slow_Spindle', 'Fast_Spindle']:
                pha, freq, amp = analytical_transform(Data, picks='C3', 
                                                      freqs=band_corr[bands], 
                                                      method = 'hilbert')
                axs[1].plot(Data.times[t_min:t_max+1], 
                            amp.mean(0).mean(0)[t_min:t_max+1],
                            #stats.zscore(amp, axis=-1).mean(0).mean(0)[t_min:t_max+1],
                            label=bands)
                plt.legend()
            axs[1].set_xlim([length[0], length[1]])
            axs[1].set_ylabel('Amplitude envelope')
            plt.xlabel('Time (s)')
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(save_path + '_tf_plot.png')
        else:
            fig, ax = plt.subplots()
            vmin, vmax = np.percentile(Sxx_, [0 + 0.1, 100 - 0.1])
            norm = Normalize(vmin=vmin, vmax=vmax)
            CM = ax.pcolormesh(times, freqs, Sxx_, 
                               shading='gouraud', cmap=cmap, rasterized = True, 
                               antialiased=True, vmin=vmin, vmax=vmax)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            cbar = plt.colorbar(CM, ax=ax)
            cbar.set_label(mode + ' (dB)', rotation=270)
            plt.tight_layout()
            if save_path is not None:
                plt.savefig(save_path + '_tf_plot.png')
    
    
    if itc_calculation == 'tensorpac':
        itc = ITC(Data.get_data(picks = chan).squeeze()*1e6,
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

#%%
def analytical_transform(epoch, picks='C3', freqs=(9, 16), method = 'imf_hilbert'):
    ## narrow filter data
    narrow_data = epoch.copy().pick(picks).filter(l_freq=freqs[0], h_freq=freqs[1]).get_data(units='uV')
    
    ## IMF based method
    if method == 'imf_hilbert':
        # phase with imf - use freq transform 
        pha = np.ones(narrow_data.shape)*np.nan
        freq = np.ones(narrow_data.shape)*np.nan
        amp = np.ones(narrow_data.shape)*np.nan
        for ep_idx, ep in enumerate(narrow_data):
            for ch_idx, chan in enumerate(ep):
                imf_ch_ep = emd.sift.sift(chan, 
                                          imf_opts={'sd_thresh': 0.1}, max_imfs=1) 
                pha_, freq_, amp_ = emd.spectra.frequency_transform(imf_ch_ep, epoch.info['sfreq'], 'nht')
                pha[ep_idx, ch_idx, :] = pha_.squeeze()   
                freq[ep_idx, ch_idx, :] = freq_.squeeze() 
                amp[ep_idx, ch_idx, :] = amp_.squeeze()
        
        return pha, freq, amp
    
    ## Hilbert based method, no IMF correction
    if method == 'hilbert':
        pha, freq, amplitude = [],[],[]
        n_samples = max(narrow_data.shape)
        nfast = next_fast_len(n_samples)
        # Now extract the instantaneous phase/amplitude using Hilbert transform
        pha = np.angle(signal.hilbert(narrow_data, N=nfast)[:,:,:n_samples])
        amp = np.abs(signal.hilbert(narrow_data, N=nfast)[:,:,:n_samples])
        
        return pha, [], amp
      
    if method == 'ndsp':
        from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time
        pha = [np.expand_dims(phase_by_time(narrow_data[:,i,:].squeeze(), epoch.info['sfreq'],
                              freqs),1) for i in range(min(narrow_data.shape))]
        freq = [np.expand_dims(freq_by_time(narrow_data[:,i,:].squeeze(), epoch.info['sfreq'],
                              freqs),1) for i in range(min(narrow_data.shape))]
        amp = [np.expand_dims(amp_by_time(narrow_data[:,i,:].squeeze(), epoch.info['sfreq'],
                              freqs),1) for i in range(min(narrow_data.shape))]
        
        return np.concatenate(pha,1), np.concatenate(freq,1), np.concatenate(amp,1)

