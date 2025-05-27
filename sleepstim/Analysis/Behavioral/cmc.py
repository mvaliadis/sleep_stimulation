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
from sleepstim.core.dsp import downsample_scaled
from sleepstim.core.io import load_xdf
from sleepstim.Analysis.Resting_State.rs_preproc import subject_cond_parser
from sleepstim.Analysis.TMS.tms_funcs import tkeo
import yasa 
import seaborn as sns
from detecta import detect_onset
from scipy.integrate import trapz, cumtrapz
from matplotlib.colors import Normalize
from lspopt import spectrogram_lspopt
from tqdm import tqdm
import pandas as pd
import pyprep
from mne.preprocessing import ICA
import pickle 
import pingouin as pg
sns.set_theme(color_codes=True)

def SNR(x, y=None, axis=0):
    """Signal to Noise Ratio, as defined here:
    www.scholarpedia.org/article/Signal-to-noise_ratio_in_neuroscience

    This is equivalent to the variance of the average signal in relation to the 
    total variance.

    Parameters
    ----------
    x : array of any shape
        An array representing a set of signals. The last dimension in the array
        must correspond to the time course of the signal (over which the common
        underlying signal is expected).
    y : None, array of shape == x.shape
        Reference array that will be used for computation of the underlying
        signal. If y = None, x is used as reference array. Defaults to None.
    axis : int or tuple of ints
        The axes over which the signals should be averaged to get the common
        underlying signal. Can include every axis of the array except for the 
        last one, which is assumed to be the time axis of the signal.
        Defaults to 0.

    Returns
    -------
    SNR : float or array of floats
        The signal-to-noise ration over the remaining axes. Is equivalent to 
        the fraction of how much of the total variance in the signal set is
        explained through the common underlying signal.
    """
    y = x if y is None else y
    axis = tuple([axis]) if isinstance(axis, int) else axis
    if np.any([ax in axis for ax in [-1, x.ndim - 1]]):
        raise ValueError("Last array axis can't be passed as axis argument.")
    return np.mean(np.mean(y, axis)**2, axis=-1) / np.var(x, axis=(-1,) + axis)

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
    
    elif method=='multitaper_conn':
        _, x = yasa.sliding_window(np.vstack([signal1, signal2]), window=2, sf=1000)
        indices = (np.array([0]), np.array([1])) 
        import mne_connectivity
        coh = mne_connectivity.EpochSpectralConnectivity(x, names=None, method='coh', indices=indices,
                                                         sfreq=sf, mode='multitaper', fmin=foi[0], fmax=foi[1],
                                                         fskip=1, faverage=False, block_size=1000,
                                                         verbose=0)
        
        return coh.freqs, coh._data.squeeze()


        
    elif method=='welch':
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
        # f, t, Zxx = signal.stft(data, fs=sf, nperseg=int((2/foi[0])*sf), noverlap = 64)
        f, t, Zxx = yasa.stft_power(data, sf, window=int(foi[0]/1), step=.2, 
                                    band=foi, norm=True, interp=True)
    elif method=='multitaper':
        f, t, Zxx = spectrogram_lspopt(data, fs=sf, nperseg=int((2/foi[0])*sf), 
                                       c_parameter=20.0, noverlap = 0)
        if dB:
            Zxx = 10*np.log10(Zxx)
    ## Plotting
    foi_idx = np.logical_and(f >= foi[0], f <= foi[1])
    vmin, vmax = np.percentile(np.abs(Zxx), [0 + 5, 100 - 5])
    norm = Normalize(vmin=vmin, vmax=vmax)
    ax1.pcolormesh(t, f[foi_idx], np.abs(Zxx)[foi_idx,:], vmin=vmin, 
                   vmax=vmax, shading='gouraud', norm = norm,
                   antialiased=True, cmap=plt.cm.Spectral_r)
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
    
    ## Quit if recording is too short
    if max(eego.time_series.shape) < 60*sf:
        trial_dict = []
        return trial_dict
        
    ## get clocks
    eegotimes = eego.time_stamps - eego.time_stamps[0]

    ## adjust chan info
    chans, chtypes = channel_parser(eego.channel_labels, eego.channel_types)
    
    ## mne-ize data
    mne_info = mne.create_info(ch_names=chans, sfreq=sf, ch_types=chtypes)
    raw = mne.io.RawArray(eego.time_series[:,0:len(chans)].T, mne_info) 
    raw = raw.set_montage(mne.channels.make_standard_montage('standard_1005')) 
    
    ## filter data
    raw.filter(l_freq = 1, h_freq = 45, picks='eeg', verbose = 0)
    raw.filter(l_freq = 10, h_freq = 200, picks='emg', verbose = 0)
    raw.filter(l_freq = 0.3, h_freq = 70, picks='ecg', verbose = 0)
    raw.notch_filter(freqs=(50,100,150), verbose = 0) # method = 'spectrum_fit'
    
    ## PREP pipeline bad channel detection, interpolation and subsequent robust rereferencing 
    prep_params = {"ref_chs": chans[0:64],
                   "reref_chs": chans[0:64]
                   }

    raw_ref = pyprep.reference.Reference(raw, prep_params, ransac=False)
    raw_ref.perform_reference()
    
    ## clean via ICA  -- WARNING -- NO ECG component rejection
    fit_ica = ica_slalom_eeg_data(raw_ref.raw, file)
    
    ## CSD, but first drop bads from info
    fit_ica.info['bads'] = []
    fit_ica = mne.preprocessing.compute_current_source_density(fit_ica)
    
    ## get EMG/C3 data
    edcdat = raw.get_data(picks='EDC_R').squeeze()*1e6
    c3dat = fit_ica.get_data(picks='C3').squeeze()*1e3 # if CSD, else 1e6
    ecgdat = raw.get_data(picks='ECG').squeeze()*1e6
   
    ## linear envelope of edc for movement onset
    edcdat_le = linear_envelope(edcdat, sf=1000, fc_bp=[10, 200], fc_lp=8)
    
    ## Plot spectrogram of EDC
    # plot_spectrogram(edcdat, sf, foi=(10,200), method='multitaper', dB=False)
    
    ## Movement onset detection
    move_on = emg_onset(edcdat_le, sf, show=False, envelope=False, use_tkeo=True)
    fig,ax,coords,cid = plot_and_click_emg(edcdat.squeeze(), move_on)
    
    ## Select automated slalom trials or manually clicked ones
    select = input('Select "automated" or "manual" trial duration onset/offsets: ')
    if select == 'manual':
        coords = np.asarray(np.round(coords)).reshape(4,2).astype(int)
        runs, eeg_runs = [], []
        for run in range(len(coords)):
            runs.append(edcdat[coords[run][0]:coords[run][1]])
            eeg_runs.append(c3dat[coords[run][0]:coords[run][1]])
    elif select == 'automated':
        runs, eeg_runs = [], []
        for run in range(len(move_on)):
            runs.append(edcdat[move_on[run][0]:move_on[run][1]])
            eeg_runs.append(c3dat[move_on[run][0]:move_on[run][1]])            
    else:
        print('WARNING: Please select a valid trial onset/offset method!')
        
    ## Epoch data for SNR calculation 
    epochs = [yasa.sliding_window(np.expand_dims(runs[run], 0), window=2, sf=sf)[1] for run in range(len(runs))]
    
    ## Compute SNR for EDC
    snr0 = [SNR(epochs[0][[i],:,:].T, axis=0)[0] for i in range(epochs[0].shape[0])]
    snr1 = [SNR(epochs[1][[i],:,:].T, axis=0)[0] for i in range(epochs[1].shape[0])]
    snr2 = [SNR(epochs[2][[i],:,:].T, axis=0)[0] for i in range(epochs[2].shape[0])]
    snr3 = [SNR(epochs[3][[i],:,:].T, axis=0)[0] for i in range(epochs[3].shape[0])]
         
    ## Extract dictionary of processed EEG/EMG/CMC/SNR values by trials 
    trial_dict = {'Trial1_EMG' : runs[0], 'Trial1_EEG': eeg_runs[0], 'Trial1_SNR' :  snr0,
                  'Trial1_CMC' : CMC(runs[0], eeg_runs[0], sf=sf, foi=(3, 40), plot=False, method='multitaper_conn'), 
                  'Trial2_EMG' : runs[1], 'Trial2_EEG' : eeg_runs[1], 'Trial2_SNR' :  snr1, 
                  'Trial2_CMC' : CMC(runs[1], eeg_runs[1], sf=sf, foi=(3, 40), plot=False, method='multitaper_conn'), 
                  'Trial3_EMG' : runs[2], 'Trial3_EEG': eeg_runs[2], 'Trial3_SNR' :  snr2, 
                  'Trial3_CMC' : CMC(runs[2], eeg_runs[2], sf=sf, foi=(3, 40), plot=False, method='multitaper_conn'),  
                  'Trial4_EMG' : runs[3], 'Trial4_EEG' : eeg_runs[3], 'Trial4_SNR' :  snr3,
                  'Trial4_CMC' : CMC(runs[3], eeg_runs[3], sf=sf, foi=(3, 40), plot=False, method='multitaper_conn')}
                    
    return trial_dict

def emg_onset(data, sf, show=False, envelope=False, use_tkeo=False):
    if envelope:
        data_le = linear_envelope(data, sf=sf, fc_bp=[10, 400], fc_lp=8)
    else:
        data_le = data 
    if use_tkeo:
        data_le = abs(tkeo(data_le, normalize=True, plot=False))
        threshold = 0.5*np.std(data_le)
    else:
        threshold = 2*np.std(data_le)
    inds = detect_onset(data_le, threshold=threshold,n_above=5000, n_below=5000, show=show)
    return inds 

def plot_and_click_emg(edcdat, move_on):
    fig, ax = plt.subplots()
    ax.plot(edcdat)
    ax.vlines(move_on, ymin= -1, ymax=max(edcdat), linestyles='dashed', colors='m')  
    global coords 
    coords = []
    def onclick(event):
        global ix
        ix = event.xdata
        print('%s click: button=%d, x=%d, xdata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.xdata))
        coords.append(ix)   
        if len(coords) >= 8:
            fig.canvas.mpl_disconnect(cid)
            plt.close()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.manager.window.resize(1950, 550)
    plt.show()
    showing = True
    while showing == True:
        if len(coords) == 8:
            showing = False
        else:
            plt.pause(0.1)

    
    return fig,ax,coords,cid
    
def ica_slalom_eeg_data(raw, file):
    # plotting & ica object path
    fig_path = '/media/administrator/data/Study_1_data/Figures/Slalom_EEG/'
    ica_path = '/media/administrator/data/Study_1_data/Slalom_ica_data/'
    subjname = str(file).split("/")[-2]
    night = str(file).split('.')[0].split('/')[-1].split('_')[-1][-1]
    session = str(file).split('.')[0].split('/')[-1].split('_')[-2]
    subj_cond = subjname + '_' + night + '_' + session
    
    # initialize ICA
    ica = ICA(n_components=15, method='picard', random_state=42) #fit_params = dict(ortho=True, extended=True)
    print('\n***** Performing ICA ...\n')
    ica.fit(raw, picks='eeg')
    
    # # Determine ecg component to remove
    # ecg_idx, ecg_scores = ica.find_bads_ecg(raw, method = 'ctps', threshold= 'auto')
    # ecg_score_plot = ica.plot_scores(ecg_scores)
    # ecg_score_plot.savefig(fig_path + subj_cond + '_ecg_component_score.png')
    
    # Automated component detection assistance 
    ica.detect_artifacts(raw)
     
    # Plot sources separated by ICA
    ic_source_plot = ica.plot_sources(raw, show_scrollbars=True, title='EEG sources estimated by ICA')
    ic_source_plot.savefig(fig_path + subj_cond + '_ic_component_source.png')

    # Plot topographic maps of sources separated by ICA
    ic_comp_plot = ica.plot_components(title='Topographic maps of EEG sources estimated by ICA')
    ic_comp_plot[0].savefig(fig_path + subj_cond + '_ic_topo_plot.png')
    
    # components to exclude 
    # ica.exclude = ica.exclude + ecg_idx
     
    # save ICA object
    ica.save(ica_path + subj_cond + '_ica_obj.fif')
    
    # log excluded ICA components
    # out.write(f'The dataset: {subj_cond} had the following components removed: {ica.exclude}' + '\n')
    
    # Plot signal with removed components
    overlay_plot = ica.plot_overlay(raw, exclude=ica.exclude)
    overlay_plot.savefig(fig_path + subj_cond + '_ic_removed_overlay.png')
    
    # Apply ICA
    fit_ica = ica.apply(raw)
    
    # close plots 
    plt.close('all')
    
    return fit_ica         
   
def cmc_results(maindir):  
    files = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(maindir) for i in files if 'slalom_pre' in i or 'slalom_post' in i])  
    results = defaultdict(lambda: [])
    for file in tqdm(files):
        print(file)
        subjname = str(file).split("/")[-2]             
        night = " ".join(str(file).split('.')[0].split('/')[-1].split('_')[0:2])
        condition = subject_cond_parser(file, study_phase='tms')
        name = str(file).split('.')[0].split('/')[-1]
        session = name.split('_')[-2]
        # rec = name.split('_')[-1]
        trial_dict = process_rawXDF_CMC(file)
        # integration_time_reset(edcdat, sf=1000, plot=True)
        if trial_dict == []:
            pass
        else:
            results["Subject"].extend([subjname])
            results["Night"].extend([night])
            results["Condition"].extend([condition])
            results["Session"].extend([session])
            results["Trial1_CMC"].extend([trial_dict['Trial1_CMC']])
            results["Trial2_CMC"].extend([trial_dict['Trial2_CMC']])
            results["Trial3_CMC"].extend([trial_dict['Trial3_CMC']])
            results["Trial4_CMC"].extend([trial_dict['Trial4_CMC']])
            results["Trial1_SNR"].extend([trial_dict['Trial1_SNR']])
            results["Trial2_SNR"].extend([trial_dict['Trial2_SNR']])
            results["Trial3_SNR"].extend([trial_dict['Trial3_SNR']])
            results["Trial4_SNR"].extend([trial_dict['Trial4_SNR']])
               
            return results

def fix_df_cmc(df_cmc):
    night = df_cmc['Night'].to_numpy()
    nights = [int(night[i][-1]) for i in range(len(night))]
    df_cmc['Night'] = nights
    df_cmc.reset_index(inplace=True)
    
    for idx, (band, foi) in enumerate(zip(['Theta', 'Alpha', 'Beta', 'Gamma'],
                                          ((4, 8), (8, 13), (13, 30), (30, 40)))):
        # print(idx, band, foi)
        band_idx = np.logical_and(np.asarray(df_cmc['Trial1_CMC'][0][0]) >= foi[0], 
                                  np.asarray(df_cmc['Trial1_CMC'][0][0]) <= foi[1])

        for trial in range(1,5):
            block_cmc = [df_cmc[f'Trial{trial}_CMC'][i][1][band_idx].mean() for i in 
                         range(len(df_cmc[f'Trial{trial}_CMC']))]
        
            df_cmc[f'{band}_{trial}_CMC'] = block_cmc
            
            snr = [np.asarray(df_cmc[f'Trial{trial}_SNR'][i]).mean() for i in 
                   range(len(df_cmc[f'Trial{trial}_SNR']))]
        
            df_cmc[f'SNR_{trial}'] = snr
        
    for i in ["Trial1_CMC","Trial2_CMC","Trial3_CMC","Trial4_CMC", 
              "Trial1_SNR","Trial2_SNR","Trial3_SNR","Trial4_SNR",
              "index"]:
        del df_cmc[i]
        
    df_cmc.groupby([ "Subject", "Night", "Condition", "Session"]).sum().transpose().stack(0).reset_index()
    df_cmc = df_cmc.set_index(["Subject", "Night", "Condition", "Session"])
    df_cmc = df_cmc.reindex(sorted(df_cmc.columns), axis=1).T
    
    df_cmc["Block"] = [0,1,2,3]*5
    df_cmc = df_cmc.reindex(sorted(df_cmc.columns), axis=1)
    
    freqband = ["alpha"]*4
    freqband.extend(["beta"]*4)
    freqband.extend(["gamma"]*4)
    freqband.extend(["SNR"]*4)
    freqband.extend(["theta"]*4)
    df_cmc["Freq"] = freqband
    
    df_cmc = df_cmc.reset_index()
    del df_cmc["index"]
    
    df_cmc = df_cmc.pivot(columns = ["Freq", "Block"])
    df_cmc = df_cmc.T.reorder_levels(["Subject","Condition","Night",
                                      "Block", "Freq","Session"]).sort_index()
    df_cmc.reset_index()
    
    df_ = df_cmc.copy()
    df_['Total'] = df_.sum(axis=1)
    for i in range(20):
        del df_[i]
        
    _df_ = df_.reset_index()
    _df_ = _df_.pivot(columns = ["Freq","Session"], values = "Total", index = ["Subject","Condition","Night","Block"])
    df_cmc = _df_.reset_index()
    
    return df_cmc

#%%
if __name__ == '__main__':
    maindir = '/media/administrator/data/Study_1_data/Pre_post_data/'
    run = input('Do you wish to restart the CMC analysis? ')
    if run == 'yes':
        results = cmc_results(maindir)
        df = pd.DataFrame(results)
        save_path = '/media/administrator/data/Study_1_data/Statistics/CMC/CMC_results.p'
        pickle.dump(df, open(save_path, "wb"))  
    else:
        df_cmc = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/CMC/CMC_results.p', 'rb'))
        df_cmc = df_cmc.drop([6,11,12,13,14,15,16,37,48,85,86,93,98,99])
        for i in zip(['9PJZ8Z8F','475MQ9BL','5LNKD1MG','CWESJCNJ','RVQL2MRD','6QF3HOJC']):
            df_cmc.drop(df_cmc.loc[df_cmc['Subject']==i[0]].index, inplace=True)
        df_cmc_f = fix_df_cmc(df_cmc.copy()) 
        df_cmc_f.to_csv('/media/administrator/data/Study_1_data/Statistics/CMC/CMC_results_final.csv')
        
    for _, trial in enumerate(['Trial1_SNR','Trial2_SNR','Trial3_SNR','Trial4_SNR']):
        fig, ax = plt.subplots(figsize=(15,10))
        sns.violinplot(data = df_cmc[trial], ax=ax, palette="muted")
        files = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(maindir) for i in files if 'slalom_pre' in i or 'slalom_post' in i])  
        labels = [files[i].split('/')[-2] + '_' + files[i].split('/')[-1].split('_')[-2] for i in range(len(files))]
        plt.title(trial)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylabel("SNR distribution across EDC_R")
        plt.xticks(fontsize=8.0) #rotation=90
        plt.tight_layout()
        plt.savefig('/media/administrator/data/Study_1_data/Figures/Slalom_SNR/' + trial + '.jpg')

    # ## if standardize 
    # for _, trial in enumerate(['Trial1_CMC','Trial2_CMC','Trial3_CMC','Trial4_CMC']):
    #     df[trial] = [stats.zscore(df[trial].to_numpy()[i][1]) for i in range(len(df))]    

    for _, trial in enumerate(zip(['Trial1_CMC'],['Trial2_CMC'],['Trial3_CMC'],['Trial4_CMC'])):
        
        pre_up =  df[df['Session'] == 'pre'][df['Condition'] == 'up'].reset_index()
        pre_sham =  df[df['Session'] == 'pre'][df['Condition'] == 'sham'].reset_index()
        pre_down =  df[df['Session'] == 'pre'][df['Condition'] == 'down'].reset_index()
        
        post_up = df[df['Session'] == 'post'][df['Condition'] == 'up'].reset_index()
        post_sham = df[df['Session'] == 'post'][df['Condition'] == 'sham'].reset_index()
        post_down = df[df['Session'] == 'post'][df['Condition'] == 'down'].reset_index()
        
        ## Concatenate blocks into sessions 
        CMC_vals_pre_up = np.asarray([np.vstack([pre_up[trial[j]][i][1] for i in range(len(pre_up))]) for j in range(4)]).mean(0)
        CMC_vals_pre_sham = np.asarray([np.vstack([pre_sham[trial[j]][i][1] for i in range(len(pre_sham))]) for j in range(4)]).mean(0)
        CMC_vals_pre_down = np.asarray([np.vstack([pre_down[trial[j]][i][1] for i in range(len(pre_down))]) for j in range(4)]).mean(0)
    
        CMC_vals_post_up = np.asarray([np.vstack([post_up[trial[j]][i][1] for i in range(len(post_up))]) for j in range(4)]).mean(0)
        CMC_vals_post_sham = np.asarray([np.vstack([post_sham[trial[j]][i][1] for i in range(len(post_sham))]) for j in range(4)]).mean(0)
        CMC_vals_post_down = np.asarray([np.vstack([post_down[trial[j]][i][1] for i in range(len(post_down))]) for j in range(4)]).mean(0)  

        ## Stack all trials
        # CMC_vals_pre_up = np.vstack([np.vstack([pre_up[trial[j]][i][1] for i in range(len(pre_up))]) for j in range(4)])
        # CMC_vals_pre_sham = np.vstack([np.vstack([pre_sham[trial[j]][i][1] for i in range(len(pre_sham))]) for j in range(4)])
        # CMC_vals_pre_down = np.vstack([np.vstack([pre_down[trial[j]][i][1] for i in range(len(pre_down))]) for j in range(4)])
        
        # CMC_vals_post_up = np.vstack([np.vstack([post_up[trial[j]][i][1] for i in range(len(post_up))]) for j in range(4)])
        # CMC_vals_post_sham = np.vstack([np.vstack([post_sham[trial[j]][i][1] for i in range(len(post_sham))]) for j in range(4)])
        # CMC_vals_post_down = np.vstack([np.vstack([post_down[trial[j]][i][1] for i in range(len(post_down))]) for j in range(4)])
    
        # ## Standardized at each trial level
        # CMC_vals_pre_up = np.asarray([np.vstack([stats.zscore(pre_up[trial[j]][i][1]) for i in range(len(pre_up))]) for j in range(4)]).mean(0)
        # CMC_vals_pre_sham = np.asarray([np.vstack([stats.zscore(pre_sham[trial[j]][i][1]) for i in range(len(pre_sham))]) for j in range(4)]).mean(0)
        # CMC_vals_pre_down = np.asarray([np.vstack([stats.zscore(pre_down[trial[j]][i][1]) for i in range(len(pre_down))]) for j in range(4)]).mean(0)
    
        # CMC_vals_post_up = np.asarray([np.vstack([stats.zscore(post_up[trial[j]][i][1]) for i in range(len(post_up))]) for j in range(4)]).mean(0)
        # CMC_vals_post_sham = np.asarray([np.vstack([stats.zscore(post_sham[trial[j]][i][1]) for i in range(len(post_sham))]) for j in range(4)]).mean(0)
        # CMC_vals_post_down = np.asarray([np.vstack([stats.zscore(post_down[trial[j]][i][1]) for i in range(len(post_down))]) for j in range(4)]).mean(0)
        
    
        
        CMC_freqs = df['Trial1_CMC'][0][0]
        
        # CMC_vals_pre = np.vstack([pre_up[trial][i][1] for i in range(len(pre_up))])
        # CMC_freqs_pre = np.vstack([pre[trial][i][0] for i in range(len(pre))])
        # CMC_vals_post = np.vstack([post[trial][i][1] for i in range(len(post))])
        # CMC_freqs_post = np.vstack([post[trial][i][0] for i in range(len(post))])
    
    for idx, cond in enumerate(zip(([CMC_vals_pre_up, CMC_vals_post_up],
                                    [CMC_vals_pre_sham, CMC_vals_post_sham], 
                                    [CMC_vals_pre_down, CMC_vals_post_down]),
                                   (['Up'],['Sham'],['Down']))):
        plt.figure()
        plt.plot(CMC_freqs, cond[0][0].mean(0), label='pre mean')
        plt.plot(CMC_freqs, cond[0][1].mean(0), label='post mean')
        # plt.plot(CMC_freqs, np.median(cond[0].mean(0), axis=0), label='pre median')
        # plt.plot(CMC_freqs, np.median(cond[1].mean(0), axis=0), label='post median')
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Coherence')
        plt.title(f'{cond[-1][0]} - CMC between C3 and EDC_R')
        plt.xlim(CMC_freqs[0], CMC_freqs[-1])
        plt.legend()
        plt.show()
        plt.savefig('/media/administrator/data/Study_1_data/Figures/Slalom_SNR/' + 'CMC_plot_' + cond[-1][0] + '.jpg')

    norm_up_diff = CMC_vals_post_up - CMC_vals_pre_up #stats.zscore
    norm_sham_diff = CMC_vals_post_sham - CMC_vals_pre_sham
    norm_down_diff = CMC_vals_post_down - CMC_vals_pre_down
    
    #t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test([norm_up_diff,norm_down_diff])
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test([norm_sham_diff, norm_down_diff, norm_up_diff], 
                                                                         n_permutations=1000,
                                                                         tail=1, n_jobs=1,
                                                                         out_type='mask')
    
   
    times = CMC_freqs
    plt.subplot(211)
    plt.plot(times, norm_up_diff.mean(axis=0) - norm_sham_diff.mean(axis=0),
             label="Contrast (Up - Sham")
    plt.plot(times, norm_up_diff.mean(axis=0) - norm_down_diff.mean(axis=0),
          label="Contrast (Up - Down")
    plt.plot(times, norm_down_diff.mean(axis=0) - norm_sham_diff.mean(axis=0),
       label="Contrast (Down - Sham")
    
    plt.ylabel("CMC difference")
    plt.legend()
    plt.xlim(CMC_freqs[0], CMC_freqs[-1])
    plt.subplot(212)
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_pv[i_c] <= 0.05:
            h = plt.axvspan(times[c.start], times[c.stop - 1],
                            color='r', alpha=0.3)
        else:
            plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                        alpha=0.3)
    hf = plt.plot(times, t_obs, 'g')
    # plt.legend((h, ), ('cluster p-value < 0.05', ))
    plt.xlabel("Frequencies (Hz)")
    plt.ylabel("f-values")
    plt.xlim(CMC_freqs[0], CMC_freqs[-1])
    plt.show()
    
    # from scipy.stats import bootstrap
    # data = (CMC_vals_post_up,)  # samples must be in a sequence
    # res = bootstrap(data, np.std, confidence_level=0.95,
    #                 random_state=rng)

    
    # beta = np.logical_and(np.asarray(CMC_freqs) > 16 , np.asarray(CMC_freqs) < 30)
    # beta_diff_up = norm_up_diff[:,beta].mean(1)
    # beta_diff_sham = norm_sham_diff[:,beta].mean(1)
    # beta_diff_down = norm_down_diff[:,beta].mean(1)
    
    # stats.ttest_rel(beta_diff_up, beta_diff_down)
    # stats.ttest_rel(beta_diff_up, beta_diff_sham)
    # stats.ttest_rel(beta_diff_down, beta_diff_sham)
    