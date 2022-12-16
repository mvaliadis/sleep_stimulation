# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 08:29:12 2020

@author: neuro
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sleepstim.sleep_funs import (load_xdf, channel_parser)
#from mne.filter import filter_data, notch_filter
# from scipy import signal
# import scipy
# import time
# import seaborn as sns 
from functools import reduce
import yasa
import mne
#from scipy.signal import find_peaks, welch, detrend
import scipy.stats as stats
import fooof
import pandas as pd
from scipy.linalg import eigh, eig
import mne
import itertools
import os
import glob
import statistics
from tqdm import tqdm
import seaborn as sns
import pingouin as pg
mne.set_log_level("CRITICAL")
sns.set(style='darkgrid', font_scale=1.2)

def surface_laplacian_pre(inst, sphere='auto', lambda2=1e-3,
                          stiffness=4, n_legendre_terms=50):
    picks = mne.pick_types(inst.info, meg=False, eeg=True, exclude=[])
    radius, origin_head, origin_device = mne.bem.fit_sphere_to_headshape(inst.info)
    x, y, z = origin_head - origin_device
    sphere = (x, y, z, radius)
    sphere = np.array(sphere, float)
    x, y, z, radius = sphere
    
    pos = np.array([inst.info['chs'][pick]['loc'][:3] for pick in picks])
    pos -= (x, y, z)
    
    # Project onto a unit sphere to compute the cosine similarity:
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    cos_dist = np.clip(np.dot(pos, pos.T), -1, 1)
    # This is equivalent to doing one minus half the squared Euclidean:
    # from scipy.spatial.distance import squareform, pdist
    # cos_dist = 1 - squareform(pdist(pos, 'sqeuclidean')) / 2.
    del pos
    
    G = mne.preprocessing._csd._calc_g(cos_dist, stiffness=stiffness,
                                       n_legendre_terms=n_legendre_terms)
    H = mne.preprocessing._csd._calc_h(cos_dist, stiffness=stiffness,
                                       n_legendre_terms=n_legendre_terms)
    
    G_precomputed = mne.preprocessing._csd._prepare_G(G, lambda2)
    
    trans_csd = mne.preprocessing._csd._compute_csd(G_precomputed=G_precomputed,
                                                    H=H, radius=radius)
    
    return trans_csd 

def surface_laplacian_rt(data, trans_csd):
    #epochs = inst._data
    epochs = np.expand_dims(data.T, 0)
    for epo in epochs:
        csd_data = np.dot(trans_csd, epo)
    
    return csd_data

def get_coords(channels):
    #channels = np.load('/home/administrator/sleep_stimulation/sleepstim/channels.npy')[:,0]
        
    montage = mne.channels.make_standard_montage('standard_1005')
    names_coords = montage._get_ch_pos()
    eego_names_coords = {k: names_coords[k] for k in channels}
    coords = np.stack(([eego_names_coords[chan] for chan in channels]))

    return coords, channels

def re_reference(data, ch_names, trans_csd=None,
                 sf=512, reference='common average'):
    """EEG data re_referencing function.

    This function takes a numpy array of the data & rereferences the data based on
    the new selected reference. 

    Parameters
    ----------
    data : numpy array of [n_samples x n_chans]
        The data. Include only EEG channels for this argument!
        
    reference : str {‘common average’, ‘mastoids’}
        The new reference; default is ‘common average’.
        
    Returns
    -------
    data : numpy array of shape [n_samples x n_chans]
        The re-referenced data
    """
    # %timeit = 97 microseconds for 30s of data (fs = 256 Hz)
    # Initial safety check to confirm data shape is [n_samples x n_chans]
    dpnts, chans = data.shape
    if chans > dpnts:
        data = np.transpose(data)
    if reference == 'common average':
        # Compute mean of each epoch per channel then subtract common average
        ref_data = data[..., :].mean(-1, keepdims=True)
        data -= ref_data
        #print('Data will be re-referenced to the common average of all selected EEG electrodes!')
    elif reference == 'mastoids':
        # May change to name indexed instead of numerical index...
        ref_data = data[..., [ch_names.index('M1'), ch_names.index('M2')]].mean(-1, keepdims=True)
        data -= ref_data
        #print('Data will be re-referenced to the average of the mastoids!')
    elif reference == 'csd':
        data = surface_laplacian_rt(data=data, trans_csd=trans_csd) #np.expand_dims(d,1))
    else:
        raise ValueError('Please select a valid reference!')
    
    return data

def detect_bad_interpolate(epoch, method='NK', csd_=False):
    if method=='EQI':
        # from sleepstim.Analysis import bad_channel_detection
        from sklearn.ensemble import IsolationForest
        def qc_segment(xch,sf,window = 2):
            """
            Divide into segments of n s according to sampling frequency, selecting the middle part of the data
            ----------
            x : eeg data (1 x timepoints)
            sf : sampling rate
            window : length of epochs in seconds

            Returns
            -------
            xs : eeg data in segments x timepoints .

            """
            
            pps = sf*window
            nseg      = int(len(xch)/pps)    
            remainder = len(xch) - nseg*pps
            if remainder>0:
                part2     = int(remainder/2)
                part1     = remainder - part2
                if part2 == 0: xs = np.reshape(xch[part1:],(nseg, pps))
                else:          xs = np.reshape(xch[part1:-part2],(nseg, pps))
            else:
                xs = np.reshape(xch,(nseg, pps))
            return xs

        def qc_assas(xs, frange, fs = 1000):
            # Calculate FFT
            yf   = fft(xs)
            bins = fftfreq(len(xs), 1/fs)
            # Check fft:
            # plt.plot(bins[:int(fs/2)],yf[:int(fs/2)])
            assas = np.mean(np.abs(yf[np.argwhere(bins == frange[0])[0][0]:np.argwhere(bins == frange[1])[0][0]+1]))
            return assas
            
        def qc_rms(xs):
            # Calculate root mean square of the EEG (NB after de-meaning?)
            rms = (np.sum(np.abs(xs)**2)/len(xs))**0.5
            return rms

        def qc_mg(xs):
            # Calculate maximum gradient
            mg = max(xs[1:]-xs[:-1])
            return mg

        def qc_zcr(xs):
            # Calculate zero-crossing rate
            zcr = np.sum(np.abs( np.sign(xs[1:]) - np.sign(xs[:-1]) ))/len(xs)
            return zcr

        def qc_kurt(xs):
            # Calculate kurtosis
            # kurt = np.sum((xs - np.mean(xs))**4) / np.sum((xs - np.mean(xs))**2)**2
            kurt = stats.kurtosis(xs, axis=-1)
            return kurt

        def qc_calcEQI(data2, fs, frange=(1,45)):
            """
            Combines 6 quality index functions above and returns qc_arr matrix in the shape of
            number of measures x channels x segments
            ----- input -----
            eeg : channels x timepoints
            fs  : sampling frequency

            """
            nch    = data2.shape[0]
            nsegs  = qc_segment(data2[0,:],fs).shape[0]
            qc_arr = np.zeros((6,nch,nsegs))
            
            # Loop over channels and segments and calculate quality indices
            for ich, xch in enumerate(data2):
                xsegs = qc_segment(xch,fs,window=2)
                for idxs, xs in enumerate(xsegs):
                    # 1. Average Single-Sided Amplitude Spectrum in range given range
                    qc_arr[0,ich,idxs] = qc_assas(xs, frange, fs)
                    # 2. Average Single-Sided Amplitude Spectrum in range 49-51 Hz
                    qc_arr[1,ich,idxs] = qc_assas(xs, [49,51], fs)
                    # 3. root mean square
                    qc_arr[2,ich,idxs] = qc_rms(xs)
                    # 4. maximum gradient
                    qc_arr[3,ich,idxs] = qc_mg(xs)
                    # 5. Zero-Crossing Rate 
                    qc_arr[4,ich,idxs] = qc_zcr(xs)
                    # 6. Kurtosis 
                    qc_arr[5,ich,idxs] = qc_kurt(xs)
                
            return qc_arr
        
        eeg_chans = epoch.ch_names
        EEG_qi = qc_calcEQI(epoch.get_data().squeeze()*1e6, 
                            fs=epoch.info['sfreq'], frange=(0.5,30))
        # create dataframe for EQI results 
        eqi_lab = ['avgspec_0.5-30','line_noise','root_mean_sq','max_gradient','zero-crossing_rate','kurtosis']
        df_comb = [pd.DataFrame(data = EEG_qi[:,:,i].T, columns = eqi_lab, index = eeg_chans)
                   for i in range(EEG_qi.shape[-1])]
        for i in range(EEG_qi.shape[-1]): df_comb[i]['Epoch'] = i
        df_comb2 = pd.concat(df_comb)
        ilf = IsolationForest(contamination='auto', max_samples='auto',
                              verbose=0, random_state=42)
        good = ilf.fit_predict(df_comb2.drop(['Epoch'], axis=1))
        good[good == -1] = 0
        df_comb2['Isolation forest score'] = good
        # isolation forest scores by channel
        df_if_scores = pd.DataFrame([df_comb2.loc[(epoch.ch_names[i])]['Isolation forest score'].value_counts(normalize=True) 
                                     for i in range(len(eeg_chans))])
        df_if_scores['Channel'] = eeg_chans
        epoch.info['bads'] = df_if_scores['Channel'][df_if_scores[0] > 0.25].to_list()
        epoch.interpolate_bads()
    elif method == 'NK':
        import neurokit2 as nk
        bads, _ = nk.eeg_badchannels(epoch.get_data().squeeze(), 
                                     distance_threshold=0.95, show=False)
        epoch.info['bads'] = list(np.asarray(epoch.ch_names)[bads])
        if csd_:
            return list(np.asarray(epoch.ch_names)[bads])
        else:
            epoch.interpolate_bads()
            return epoch
   
def subject_cond_parser(file, night, sub):
    # data sheet with true subject condition nights
    sheet = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
    subj_cond = np.loadtxt(sheet, delimiter=',', dtype='str', skiprows=1) 
    cond_dict = {0:'sham', 1:'up', 2:'down'}
    # find condition by night
    sc = sub
    index_name = list(subj_cond[:,0]).index(sc)
    cond = int(subj_cond[index_name,1::][int(night)-1])
    # true condition night name
    condition_night = cond_dict[cond]  
    
    return condition_night

""" Functions to compute Spatial-Spectral Decompostion (SSD).

Reference
---------
Nikulin VV, Nolte G, Curio G.: A novel method for reliable and fast
extraction of neuronal EEG/MEG oscillations on the basis of
spatio-spectral decomposition. Neuroimage. 2011 Apr 15;55(4):1528-35.
doi: 10.1016/j.neuroimage.2011.01.057. Epub 2011 Jan 27. PMID: 21276858.

"""

def compute_ged(cov_signal, cov_noise):
    """Compute a generatlized eigenvalue decomposition maximizing principal
    directions spanned by the signal contribution while minimizing directions
    spanned by the noise contribution.

    Parameters
    ----------
    cov_signal : array, 2-D
        Covariance matrix of the signal contribution.
    cov_noise : array, 2-D
        Covariance matrix of the noise contribution.

    Returns
    -------
    filters : array
        SSD spatial filter matrix, columns are individual filters.

    """

    nr_channels = cov_signal.shape[0]

    # check for rank-deficiency
    [lambda_val, filters] = eig(cov_signal)
    idx = np.argsort(lambda_val)[::-1]
    filters = np.real(filters[:, idx])
    lambda_val = np.real(lambda_val[idx])
    tol = lambda_val[0] * 1e-6
    r = np.sum(lambda_val > tol)

    # if rank smaller than nr_channels make expansion
    if r < nr_channels:
        #print("Warning: Input data is not full rank")
        M = np.matmul(filters[:, :r], np.diag(lambda_val[:r] ** -0.5))
    else:
        M = np.diag(np.ones((nr_channels,)))

    cov_signal_ex = (M.T @ cov_signal) @ M
    cov_noise_ex = (M.T @ cov_noise) @ M

    [lambda_val, filters] = eig(cov_signal_ex, cov_signal_ex + cov_noise_ex)

    idx = np.argsort(lambda_val)[::-1]
    filters = filters[:, idx]
    filters = np.matmul(M, filters)

    return filters

def apply_filters(raw, filters, prefix="ssd", ch_names=None):
    """Apply spatial filters on continuous data.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance with signals to be spatially filtered.
    filters : array, 2-D
        Spatial filters as computed by SSD.
    prefix : string | None
        Prefix for renaming channels for disambiguation. If None: "ssd"
        is used.

    Returns
    -------
    raw_projected : instance of Raw
        Raw instance with projected signals as traces.
    """
    
    if type(raw) == mne.evoked.EvokedArray:
        raw_projected = raw.copy()
        components = filters.T @ raw.get_data().squeeze()
        nr_components = filters.shape[1]
        raw_projected._data = components
    
        ssd_channels = [f"{prefix}{i+1}" for i in range(nr_components)]
        mapping = dict(zip(raw.info["ch_names"], ssd_channels))
        mne.channels.rename_channels(raw_projected.info, mapping)
        raw_projected.drop_channels(raw_projected.info["ch_names"][nr_components:])
    else:
        raw_projected = raw.copy()
        components = filters.T @ raw
        nr_components = filters.shape[1]
        raw_projected = components
    
        #ssd_channels = [f"{prefix}{i+1}" for i in range(nr_components)]
        #mapping = dict(zip(ch_names, ssd_channels))

    return raw_projected

def compute_patterns(cov_signal, filters):
    """Compute spatial patterns for a specific covariance matrix.

    Parameters
    ----------
    cov_signal : array, 2-D
        Covariance matrix of the signal contribution.
    filters : array, 2-D
        Spatial filters as computed by SSD.
    Returns
    -------
    patterns : array, 2-D
        Spatial patterns.
    """

    top = cov_signal @ filters
    bottom = (filters.T @ cov_signal) @ filters
    patterns = top @ np.linalg.pinv(bottom)

    return patterns

def compute_ssd(raw, sf, signal_bp, noise_bp, noise_bs):
    """Compute SSD for a specific peak frequency.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance with signals to be spatially filtered.
    signal_bp : tuple
        Pass-band for defining the signal contribution. E.g. (8, 13)
    noise_bp : tuple
        Pass-band for defining the noise contribution.
    noise_bs : tuple
        Stop-band for defining the noise contribution.


    Returns
    -------
    filters : array, 2-D
        Spatial filters as computed by SSD, each column = 1 spatial filter.
    patterns : array, 2-D
        Spatial patterns, with each pattern being a column vector.
    """

    if type(raw) == mne.evoked.EvokedArray:
        iir_params = dict(order=2, ftype="butter", output="sos")

        # bandpass filter for signal
        raw_signal = raw.copy().filter(
            l_freq=signal_bp[0],
            h_freq=signal_bp[1],
            method="iir",
            iir_params=iir_params,
            verbose=False,
        )
    
        # bandpass filter
        raw_noise = raw.copy().filter(
            l_freq=noise_bp[0],
            h_freq=noise_bp[1],
            method="iir",
            iir_params=iir_params,
            verbose=False,
        )
    
        # bandstop filter
        # raw_noise = raw_noise.filter(
        #     l_freq=noise_bs[1],
        #     h_freq=noise_bs[0],
        #     method="iir",
        #     iir_params=iir_params,
        #     verbose=False,
        # )

        # compute covariance matrices for signal and noise contributions
        if raw_signal._data.ndim == 3:
            cov_signal = mne.compute_covariance(raw_signal, verbose=False).data
            cov_noise = mne.compute_covariance(raw_noise, verbose=False).data
        elif raw_signal._data.ndim == 2:
            cov_signal = np.cov(raw_signal._data.squeeze())
            cov_noise = np.cov(raw_noise._data.squeeze())

        # compute spatial filters
        filters = compute_ged(cov_signal, cov_noise)
    
        # compute spatial patterns
        patterns = compute_patterns(cov_signal, filters)
        
    else:
        filtparams_lp = signal.butter(2, (signal_bp[1]), fs = sf, 
                                      btype='lowpass', output='sos')
        filtparams_noise = signal.butter(2, (noise_bp[1]), fs = sf, 
                                         btype='lowpass', output='sos')        
        #filtparams_bs = signal.butter(2, (noise_bs[0], noise_bs[1]), fs = sf, btype='bandstop')        

        # bandpass filter for signal
        raw_signal = signal.sosfiltfilt(filtparams_lp, raw, axis=-1)
    
        # bandpass filter
        raw_noise = signal.sosfiltfilt(filtparams_noise, raw, axis=-1)
    
        # bandstop filter
        #raw_noise = signal.filtfilt(*filtparams_bs, raw_noise, axis=-1)
        # raw_noise = raw_noise.filter(
        #     l_freq=noise_bs[1],
        #     h_freq=noise_bs[0],
        #     method="iir",
        #     iir_params=iir_params,
        #     verbose=False,
        # )

        # compute covariance matrices for signal and noise contributions
        cov_signal = covariance_and_shrink(raw_signal, shrink=False).mean(0)
        cov_noise = covariance_and_shrink(raw_noise, shrink=False).mean(0)
        
        # compute spatial filters
        filters = compute_ged(cov_signal, cov_noise)
    
        # compute spatial patterns
        patterns = compute_patterns(cov_signal, filters)
        
    return filters, patterns

def ssd_offline(epochs):
    try:
        data = epochs.copy().get_data(units='uV')
    except:
        data = epochs.copy().get_data()*1e3
    # SSD fun
    filters, patterns = compute_ssd(data, signal_bp=(0.5, 1.5), 
                                    noise_bp=(0.1, 30), 
                                    noise_bs=(None, None),
                                    sf=epochs.info['sfreq'])
    epochs_ssd = apply_filters(data, filters)
    # narrowband filter 
    sos = signal.butter(2, 1.5, btype='lowpass', fs=epochs.info['sfreq'], 
                        output='sos')
    epochs_ssd = signal.sosfiltfilt(sos, epochs_ssd, axis=-1)
    # compute amplitude corrected spatial pattern coefficients
    std_comp = np.std(epochs_ssd, axis=-1).mean(0)
    weighted_patterns = std_comp * patterns
    # spatial complexity
    metric = compute_sensor_complexity(weighted_patterns, 
                                       5) #np.min(weighted_patterns.shape)) #10 for alpha

    return weighted_patterns, metric
            
def plot_patterns(patterns, raw, nr_patterns=10):
    """Convenience plotting function for checking spatial patterns.

    Args:
        patterns : array, 2-D
            Spatial patterns.
        raw : instance of Raw
            Raw instance containing electrode positions.
        nr_patterns :  int (optional)
            Number of patterns to be plotted. Defaults to 10.
    """

    nr_cols = 4
    nr_rows = int(np.ceil(nr_patterns/4))

    fig, ax = plt.subplots(nr_rows, nr_cols)

    for i in range(nr_patterns):
        ax1 = ax.flatten()[i]
        mne.viz.plot_topomap(patterns[:, i], raw.info, axes=ax1)#, 
                             #names=raw.ch_names)

    fig.show()

def compute_norm_coefficients(weighted_patterns, nr_components):
    """Computes normalized spatial pattern coefficients.

    Parameters
    ----------
        weighted_patterns : array, patterns weighted by amplitude.
        nr_components : measure is calculated using this many components.

    Returns
    -------
        M : array, normalized spatial pattern coefficient.
    """

    weighted_patterns = weighted_patterns.astype("float32")

    # take absolute value
    M = np.abs(weighted_patterns)
    M = M[:, :nr_components]

    # normalize across dipoles
    norm_M = np.sum(M, axis=1)
    M = (M.T / norm_M).T
    return M

def compute_sensor_complexity(weighted_patterns, nr_components):
    """Computes sensor complexity as a proxy for assessing spatial mixing.

    Parameters
    ----------
        weighted_patterns : array, patterns weighted by amplitude.
        nr_components : measure is calcualted using this many components.

    Returns
    -------
        sensor_complexity : array, sensor complexity for each sensor.
    """
    M = compute_norm_coefficients(weighted_patterns, nr_components)
    sensor_complexity = -np.sum(M * np.log(M), axis=1)

    return sensor_complexity

def covariance_and_shrink(data, shrink=True):
    from pyriemann.estimation import Covariances, Shrinkage
    if 'mne' in str(type(data)):
        try:
            data = data.get_data(units='uV')
        except:
            data = data.get_data()*1e3
    else:
        if data.ndim == 2:
            data = np.expand_dims(data, 0)
    # Calculate the covariance matrices,
    # shape (n_epochs, n_chan, n_chan)
    covmats = Covariances().fit_transform(data)
    if shrink:
        # Shrink the covariance matrix (ensure positive semi-definite)
        covmats = Shrinkage().fit_transform(covmats)
    
    return covmats
    
def plot_covs(cov_signal, cov_noise):
    ### plot the two covariance matrices
    _,axs = plt.subplots(1,3,figsize=(8,4))
    
    # S matrix
    axs[0].imshow(cov_signal,cmap='jet')
    axs[0].set_title('S matrix')
    
    # R matrix
    axs[1].imshow(cov_noise,cmap='jet')
    axs[1].set_title('R matrix')
    
    # R^{-1}S
    axs[2].imshow(np.linalg.inv(cov_noise)@cov_signal,cmap='jet')
    axs[2].set_title('$R^{-1}S$ matrix')
    
    plt.tight_layout()
    plt.show()

def pre_process_so_local_epochs():
    # Load CSD matrix
    csd_file_path = '/home/administrator/sleep_stimulation/sleepstim/Analysis/CSD_matrix.npy'
    trans_csd = np.load(csd_file_path)
    # Path containing data
    path = '/media/administrator/data/Study_1_data/Raw_data/*/*/*.xdf'
    for i, f in enumerate(tqdm(glob.glob(path)[191::])): 
        if os.stat(f).st_size > 2.5*1e9:
            print(i, f)
            stream_dict = load_xdf(f) 
            #break
            sub = f.split('/')[-2].split('_')[0]
            night = f.split('/')[-2].split('_')[1]
            if night!='adaption':
                night = subject_cond_parser(f, night, sub)
            # extract C3,C4,or Fz data  
            data = stream_dict['eego']['time_series']
            eego_times = stream_dict['eego']['time_stamps']
            times = eego_times - eego_times[0]
            info = stream_dict['eego']['info'] 
            sf = int(info['nominal_srate'][0])
            ch_names, ch_types = channel_parser(info, data) 
            idx = [x for i,x in enumerate(range(0,72)) if i!=ch_names.index('EOG')]
            data = data[0:sf*60*220, idx][:,0:64]*1e6
            ch_names = list(np.asarray(ch_names)[idx][0:64])
            # delete stream_dict/data for memory sake
            del stream_dict
            
            ## SW ground determination
            data_filt = mne.filter.filter_data(data[:,[ch_names.index('C3'), ch_names.index('C4'),
                                                       ch_names.index('Fz'), ch_names.index('M1'), 
                                                       ch_names.index('M2')]].T.astype('float64'), 
                                               l_freq=0.3, h_freq=30, sfreq=sf)
            data_filt_mastoids = re_reference(data_filt, ['C3','C4','Fz','M1','M2'], reference='mastoids')
            sw = yasa.sw_detect(data=data_filt_mastoids.T, ch_names=('C3','C4','Fz','M1','M2'), 
                                sf=sf, remove_outliers=True, coupling=True, freq_sw=(0.3, 1.5),
                                dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_ptp=(50, 500))
            #sw.compare_channels(score='recall', max_distance_sec=.1)
            for ind, (chan, peak) in enumerate(zip(['C3','C3','C4','C4','Fz','Fz'],
                                                   ['NegPeak','PosPeak','NegPeak','PosPeak','NegPeak','PosPeak'])):
                print(chan, peak)
                peak_idx, _ = yasa.get_centered_indices(data_filt_mastoids[:,0], 
                                                        np.asarray(sw.summary()[sw.summary().Channel==chan][peak]*sf), 
                                                        npts_before = int(sf*1.98), npts_after = int(sf*0.02))
            
                ## compute CSD matrix with example epoch -- could be loaded as well into memory instead
                # b,a = signal.butter(2, 4, fs = fs)
                # data_wind = data[peak_idx[0],:]
                # data_wind = signal.filtfilt(b,a,data_wind, axis=0)
                # data_wind -= np.median(data_wind, 0)
                # data_wind = re_reference(data_wind, ch_names, reference='mastoids')
                # info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='eeg')
                # epoch = mne.EpochsArray(np.expand_dims(data_wind.T, 0)/1e6, 
                #                         info, tmin = -1.98, baseline=None)
                # epoch.set_montage(mne.channels.make_standard_montage('standard_1005'))
                #trans_csd = surface_laplacian_pre(epoch, sphere='auto', lambda2=1e-5,
                #                                  stiffness=4, n_legendre_terms=50)
                #np.save(arr=trans_csd, file='/home/administrator/sleep_stimulation/sleepstim/Analysis/CSD_matrix.npy')
                #del data_wind
            
                ##
                epochs, epochs_csd = [], []
                info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='eeg')
                info_csd = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='csd')
                info_csd.set_montage(mne.channels.make_standard_montage('standard_1005'))
                fs = sf
                b,a = signal.butter(2, 4, fs = fs)
                sos_lp = signal.butter(2, 4, fs = sf, btype='lowpass', output="sos")
                for ep in peak_idx:
                    #pick 2s window from ground truth SOs
                    data_wind = data[ep,:]                 
                    #data_wind = signal.filtfilt(b,a,data_wind, axis=0)
                    data_wind = signal.sosfiltfilt(sos_lp, data_wind, axis=0)
                    data_wind -= np.median(data_wind, 0)
                    data_wind = re_reference(data_wind, ch_names, reference='mastoids')
                    data_wind_csd = re_reference(data=data_wind/1e3, ch_names=ch_names, 
                                                 trans_csd=trans_csd, sf=512, reference='csd').T
                    
                    # determine 20th/80th percentile of values for threshold of lowest values
                    if peak=='PosPeak':
                        percentiles = np.percentile(data_wind_csd[int(sf*-.04):,:].mean(0), 80)
                    elif peak=='NegPeak':
                        percentiles = np.percentile(data_wind_csd[int(sf*-.04):,:].mean(0), 20)
    
                    ## Ground truth analysis should probably contain interpolated data
                    epoch = mne.EpochsArray(np.expand_dims(data_wind.T, 0)/1e6, 
                                            info, tmin = -1.98, baseline=None)
                    epoch.set_montage(mne.channels.make_standard_montage('standard_1005'))
                    epoch = detect_bad_interpolate(epoch, method='NK')
            
                    # csd epoch
                    epoch_csd = mne.preprocessing.compute_current_source_density(epoch, lambda2=1e-05, 
                                                                                 verbose=0)
                    
                    ## Forward framework
                    d = data_wind[:, ch_names.index(chan)]
                    #plt.plot(d, 'k')       
                    d_csd = data_wind_csd[:, ch_names.index(chan)]        
                    #plt.plot(d_csd, 'm')
                    
                    ## Append data
                    if np.abs(d_csd[int(sf*-.04):].mean(0)) >= np.abs(percentiles):
                        epochs.append(epoch)
                        epochs_csd.append(epoch_csd) 

                if len(epochs) > 0:
                    # grand averages (where chan is in the 80th percentile or higher) 
                    epochs_combined = mne.concatenate_epochs(epochs)
                    epochs_csd_combined = mne.concatenate_epochs(epochs_csd)
                                            
                    gav = mne.grand_average([epochs[i].average() for i in range(len(epochs))])
                    gav_csd = mne.grand_average([epochs_csd[i].average() for i in range(len(epochs_csd))])
                    del epochs, epochs_csd 
                    
                    # save and plot gav images 
                    save_path = '/media/administrator/data/Study_2_data/processed_data/'
                    gav.save(fname=f'{save_path}{sub}_{night}_{chan}_{peak}_targeted_gav.fif',
                             overwrite=True)
                    gav_csd.save(fname=f'{save_path}{sub}_{night}_{chan}_{peak}_targeted_csd_gav.fif',
                                 overwrite=True)
                    epochs_combined.save(fname=f'{save_path}{sub}_{night}_{chan}_{peak}_targeted_epo.fif',
                                         overwrite=True)
                    epochs_csd_combined.save(fname=f'{save_path}{sub}_{night}_{chan}_{peak}_targeted_csd_epo.fif',
                                             overwrite=True)
                    del epochs_combined, epochs_csd_combined
                    
                    # args for plotting
                    fig_path = '/media/administrator/data/Study_2_data/figures/subject/'
                    ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
                    topomap_args = dict(outlines = 'head', time_unit='s', 
                                        time_format = "%0.2f s")
                    times = np.asarray([-0.50, -0.25, 0, .015])
         
                    gav.plot_joint(times, ts_args = ts_args, 
                                   topomap_args = topomap_args,
                                   title = f'{peak} {chan} - LM')
                    plt.savefig(fig_path + f'{sub}_{night}_{chan}_{peak}_LM.png')
                    gav_csd.plot_joint(times, ts_args = ts_args, 
                                       topomap_args = topomap_args,
                                       title = f'{peak} {chan} - CSD')
                    plt.savefig(fig_path + f'{sub}_{night}_{chan}_{peak}_CSD.png')
                    plt.close('all')

def group_stats(gavs, obj, title=None):
    ## do stats
    try:
        adjacency, ch_names = mne.channels.find_ch_adjacency(gavs.info, ch_type='eeg')
        eps = [list(obj[1].Epochs)[i].get_data(units='uV', tmin=-0.01).mean(1) for i in range(len(obj[1]))]
        contrast = np.concatenate([eps])
    except:
        adjacency, ch_names = mne.channels.find_ch_adjacency(gavs.info, ch_type=None)
        eps = [list(obj[1].Epochs)[i].get_data(tmin=-0.01).mean(1)*1e3 for i in range(len(obj[1]))]
        contrast = np.concatenate([eps])
        
    # spatial permuation cluster test
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        contrast,                           # numpy array for contrast [n_subjects, n_voltage, n_channels]
        n_permutations=1024,                # 1000 is the minimum
        threshold=dict(start=0, step=0.2),  # TFCE, starting at 0, in 0.2 steps (in t-values)
        tail=0,                             # two-tailed test (1 or -1 for one-tailed)
        n_jobs=-1,                          # increase value to speed up computations
        adjacency=adjacency,                # sparse matrix for channel adjacency as computed above
        buffer_size=None,
        out_type='mask',                    # returns a mask map instead of indices of sig. points
        seed=1503
    )
    
    df_stats = pd.DataFrame({'Chan': ch_names, 'T-Stat': t_obs, 
                             'Pval': cluster_p_values}) 
    df_stats = df_stats.set_index("Chan")
    df_stats['Sig'] = (df_stats['Pval'] < 0.05)

    if 'csd' in obj[0]:
        unit='mV/m2'
    else:
        unit='uV'
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8,6))
    im1, _ = mne.viz.plot_topomap(contrast.mean(0), 
                                  pos=gavs.info,
                                  axes=ax[0], show=0, cmap='RdBu_r',
                                  names=None, show_names=False)
    cbar1 = fig.colorbar(im1, fraction=0.05, ax=ax[0])   
    cbar1.ax.set_ylabel(unit, rotation=270)
    plt.tight_layout()
    im2, _ = mne.viz.plot_topomap(t_obs, 
                                  pos=gavs.info, mask=df_stats['Sig'],
                                  axes=ax[1], show=0, cmap='RdBu_r',
                                  names=None, show_names=False, 
                                  mask_params=dict(markersize=8, markerfacecolor='y'))
    cbar2 = fig.colorbar(im2, fraction=0.05, ax=ax[1])   
    cbar2.ax.set_ylabel('t-stat', rotation=270)
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
    
def pre_process_so_local_gavs():
    path = '/media/administrator/data/Study_2_data/processed_data/*.fif'
    fig_path = '/media/administrator/data/Study_2_data/figures/group/'
    df = pd.DataFrame(columns=['Subject','Target_Chan','Condition',
                               'Peak','Reference','Trials','Epochs'])
                               #'Spatial_patterns', 'Spatial_complexity'])

    for i, f in enumerate(tqdm(glob.glob(path))): 
        print(i, f.split('/')[-1])
        if f.endswith('_gav.fif'):
            #gav = mne.read_evokeds(f)[0]
            pass
        elif f.endswith('_epo.fif'):
            epochs = mne.read_epochs(f)
            #break
            
            # # process them SSDs
            # weighted_patterns, metric = ssd_offline(epochs)
            
            # # ssds with mne
            # from mne.decoding import SSD
            # ssd_epochs = SSD(info=epochs.info,
            #                  reg='oas',
            #                  filt_params_signal=dict(l_freq=.5,
            #                                          h_freq=1.5),
            #                  filt_params_noise=dict(l_freq=.1,
            #                                         h_freq=30))
            # ssd_epochs.fit(X=epochs.get_data())

            # # Plot topographies
            # pattern_epochs = mne.EvokedArray(data=ssd_epochs.patterns_[:4].T,
            #                                  info=ssd_epochs.info)
            # pattern_epochs.plot_topomap(time_format='')


            # yasa.topoplot(pd.Series(metric, epochs.ch_names), cmap='Spectral_r')
            # # plot voltage in last 20 ms
            # try:
            #     yasa.topoplot(pd.Series(epochs.copy().get_data(tmin=-0.02, units='uV').mean(0).mean(-1), 
            #                             epochs.ch_names), cmap='Spectral_r')
            # except:
            #     yasa.topoplot(pd.Series(epochs.copy().get_data(tmin=-0.02).mean(0).mean(-1)*1e3, 
            #                             epochs.ch_names), cmap='Spectral_r')
            # # plot spatial patterns
            # yasa.topoplot(pd.Series(weighted_patterns[:,0], epochs.ch_names), cmap='Spectral_r')
            
            
            df = df.append({'Subject': f.split('/')[-1].split('_')[0],
                            'Target_Chan': f.split('/')[-1].split('_')[2],
                            'Condition': f.split('/')[-1].split('_')[1],
                            'Peak': f.split('/')[-1].split('_')[3],
                            'Reference': f.split('/')[-1].split('_')[-2],
                            'Trials': int(len(epochs)),
                            'Epochs': epochs.average()},
                            #'Spatial_patterns': weighted_patterns[:, 0:32],
                            #'Spatial_complexiy': metric}, 
                            ignore_index=True)
        
    # remove nights with too few trials
    df = df[df.Trials>=50].reset_index(drop=True)
    
    # create contrast for 2nd statistical query
    df_evoked = create_subj_comp_evoked(df)
    
    # compute gavs per target chan, peak, and reference
    for idx, item in enumerate(df.groupby(['Target_Chan','Peak','Reference'])):
        print(item[0])
        # grand averages of all nights
        gavs = mne.grand_average(list(item[1].Epochs))
        # spatial pattern averages
        #sp_patterns = list(item[1].Spatial_patterns).to_numpy()[0]
        # plotting args
        ts_args = dict(spatial_colors = True, gfp=True, time_unit='s')
        topomap_args = dict(outlines = 'head', time_unit='s', 
                            time_format = "%0.2f s")
        times = np.asarray([-0.50, -0.25, 0, .015])
 
        gavs.plot_topomap(times=times, title=f'{item[0][0]} {item[0][1]} {item[0][2]}')
        plt.savefig(fig_path + f'{item[0][0]}_{item[0][1]}_{item[0][2]}_group_topo.png')
                  
        gavs.plot_joint(times, ts_args = ts_args, topomap_args = topomap_args,
                        title=f'{item[0][0]} {item[0][1]} {item[0][2]}')
        plt.savefig(fig_path + f'{item[0][0]}_{item[0][1]}_{item[0][2]}_group_joint.png')
               
        group_stats(gavs, obj=item,
                    title=f'{item[0][0]} {item[0][1]} {item[0][2]}')
        plt.savefig(fig_path + f'{item[0][0]}_{item[0][1]}_{item[0][2]}_group_stats.png')
        
        plt.close('all')
            
    # plot and compute 2nd statistical query
    group_stats_contrast(df_evoked, gavs=gavs, path=fig_path, title=None)
        
    return df, df_evoked 

def create_subj_comp_evoked(df):
    subject = list(dict.fromkeys(list(df.Subject)))
    condition = list(dict.fromkeys(list(df.Condition)))
    peaks = list(dict.fromkeys(list(df.Peak)))
    refs = list(dict.fromkeys(list(df.Reference)))
    comp_df = pd.DataFrame(columns=['Subject','Condition','Peak','Reference',
                                    'Difference_c3_c4','Difference_c3_fz',
                                    'Difference_fz_c4'])
    for sub in subject:
        sub_res = df[df.Subject == sub]
        for cond in condition:
            cond_sub_res = sub_res[sub_res.Condition == cond]
            for peak in peaks:
                p_cond_sub_res = cond_sub_res[cond_sub_res.Peak == peak]
                for ref in refs:
                    ref_p_cond_sub_res = p_cond_sub_res[p_cond_sub_res.Reference == ref]
                    ## Create comparision object
                    # c3 c4 comparison  
                    try:
                        gav_comp_c3_c4 = mne.combine_evoked([list(ref_p_cond_sub_res[p_cond_sub_res.Target_Chan=='C3'].Epochs)[0], 
                                                             list(ref_p_cond_sub_res[p_cond_sub_res.Target_Chan=='C4'].Epochs)[0]],
                                                         weights=[1, -1])
                    except:
                        gav_comp_c3_c4 = 0
                        
                    # c3 fz comparison  
                    try:
                        gav_comp_c3_fz =  mne.combine_evoked([list(ref_p_cond_sub_res[p_cond_sub_res.Target_Chan=='C3'].Epochs)[0], 
                                                              list(ref_p_cond_sub_res[p_cond_sub_res.Target_Chan=='Fz'].Epochs)[0]],
                                                            weights=[1, -1])
                    except:
                        gav_comp_c3_fz = 0
                    
                    # fz c4 comparison  
                    try:
                        gav_comp_fz_c4 =  mne.combine_evoked([list(ref_p_cond_sub_res[p_cond_sub_res.Target_Chan=='Fz'].Epochs)[0], 
                                                              list(ref_p_cond_sub_res[p_cond_sub_res.Target_Chan=='C4'].Epochs)[0]],
                                                            weights=[1, -1])
                    except:
                        gav_comp_fz_c4 = 0
                        
                        
                    # log subject/bad channels
                    new_row = {'Subject': sub, 
                               'Condition': cond,
                               'Peak': peak,
                               'Reference': ref,
                               'Difference_c3_c4': gav_comp_c3_c4,
                               'Difference_c3_fz': gav_comp_c3_fz,
                               'Difference_fz_c4': gav_comp_fz_c4}
                        
                    # update dataframe 
                    comp_df = comp_df.append(new_row, ignore_index=True)
        
    return comp_df
   
def group_stats_contrast(df_evoked, gavs, path=None, title=None, save=True):
    ## do stats
    df_rmanova = []
    for it in ('Difference_c3_c4', 'Difference_c3_fz','Difference_fz_c4'):
        #print(it)
        for (peak, ref), obj in df_evoked.groupby(['Peak','Reference'])[it]:
            #print(peak, ref)
            eps, eps_df = [], []
            obj_ = [i for i in list(obj) if i != 0]
            for k, item in enumerate(obj_):
                #print(item)
                try:
                    adjacency, ch_names = mne.channels.find_ch_adjacency(gavs.info, ch_type='eeg')
                    it_mean = item.get_data(units='uV', tmin=-0.01).mean(1)
                    it_mean_df = item.copy().crop(tmin=-0.01).to_data_frame(long_format=True).groupby(["channel"], sort=False).mean()
                    it_mean_df['subject'] = f'Subject_{k}'
                    unit='uV'
                    eps_df.append(it_mean_df)
                    eps.append(it_mean)
                except:
                    adjacency, ch_names = mne.channels.find_ch_adjacency(gavs.info, ch_type=None)
                    it_mean = item.get_data(tmin=-0.01).mean(1)*1e3
                    it_mean_df = item.copy().crop(tmin=-0.01).to_data_frame(long_format=True).groupby(["channel"], sort=False).mean()
                    it_mean_df['subject'] = f'Subject_{k}'
                    unit='mV/m2'
                    eps_df.append(it_mean_df)
                    eps.append(it_mean)
                               
            contrast = np.concatenate([eps])
            df_contrast = pd.concat(eps_df)
            
            # rm anova
            rm = df_contrast.reset_index().rm_anova(dv='value', within='channel', 
                                                    subject='subject')
            rm['Condition'] = it + '_' + peak + '_' + ref 
            df_rmanova.append(rm)
            
            # spatial permuation cluster test
            t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
                contrast,                           # numpy array for contrast [n_subjects, n_voltage, n_channels]
                n_permutations=1024,                # 1000 is the minimum
                threshold=dict(start=0, step=0.2),  # TFCE, starting at 0, in 0.2 steps (in t-values)
                tail=0,                             # two-tailed test (1 or -1 for one-tailed)
                n_jobs=-1,                          # increase value to speed up computations
                adjacency=adjacency,                # sparse matrix for channel adjacency as computed above
                buffer_size=None,
                out_type='mask',                    # returns a mask map instead of indices of sig. points
                seed=1503
            )
            
            # compute effect size
            ds = np.asarray([pg.compute_effsize_from_t(t_obs[i], N=min(contrast.shape), eftype='cohen') 
                             for i in range(max(contrast.shape))])
            
            # stats table for plotting
            df_stats = pd.DataFrame({'Chan': ch_names, 'T-Stat': t_obs, 
                                     "Cohen's_d" : ds, 'Pval': cluster_p_values}) 
            df_stats = df_stats.set_index("Chan")
            df_stats['Sig'] = (df_stats['Pval'] <= 0.01)  
            #df_stats['Max_stat'] = np.abs(df_stats['T-Stat']) >= np.percentile(np.abs(df_stats['T-Stat'].sort_values()), 80)
            
            # sample size estimation
            effects_of_int = df_stats["Cohen's_d"].loc[[it.split('_')[2].capitalize(), 
                                                        it.split('_')[2].capitalize()]].to_numpy()
            if np.sign(effects_of_int.mean()) == -1:
                sample = pg.power_ttest(d=max(effects_of_int), alpha=0.05, 
                                        power=0.95, contrast='paired')
                print(it, peak, ref)
                print('n: %.2f' % sample)
            elif np.sign(effects_of_int.mean()) == 1:
                sample = pg.power_ttest(d=min(effects_of_int), alpha=0.05, 
                                        power=0.95, contrast='paired')
                print(it, peak, ref)
                print('n: %.2f' % sample)
            
            # Plot
            fig, ax = plt.subplots(1, 3, figsize=(16, 6))
            im1, _ = mne.viz.plot_topomap(contrast.mean(0), 
                                          pos=gavs.info,
                                          axes=ax[0], show=0, cmap='RdBu_r',
                                          names=None, show_names=False)
            cbar1 = fig.colorbar(im1, fraction=0.05, ax=ax[0])   
            cbar1.ax.set_ylabel(unit, rotation=270)
            plt.tight_layout()
            im2, _ = mne.viz.plot_topomap(t_obs, 
                                          pos=gavs.info, mask=df_stats['Sig'],
                                          axes=ax[1], show=0, cmap='RdBu_r',
                                          names=None, show_names=False, 
                                          mask_params=dict(markersize=8, markerfacecolor='y'))
            cbar2 = fig.colorbar(im2, fraction=0.05, ax=ax[1])   
            cbar2.ax.set_ylabel('t-stat', rotation=270)
            plt.tight_layout()
            im3, _ = mne.viz.plot_topomap(ds, pos=gavs.info, 
                                          mask=None, #df_stats['Max_stat'],
                                          axes=ax[2], show=0, cmap='RdBu_r',
                                          names=None, show_names=False, 
                                          mask_params=dict(markersize=8, markerfacecolor='y'))
            cbar3 = fig.colorbar(im3, fraction=0.05, ax=ax[2])   
            cbar3.ax.set_ylabel("Cohen's d", rotation=270)
            plt.tight_layout()
            plt.suptitle(f'{it} {peak} {ref}')
            if save:
                plt.savefig(path + f'{it}_{peak}_{ref}_group_stats.png')
            plt.show()
            plt.close('all')
        
        df_ranova_save = pd.concat(df_rmanova)
        df_ranova_save.to_csv('/media/administrator/data/Study_2_data/stats/rmanova.csv')
        
def group_stats_topo_correlations(df, p_path):
    ## Validation stats with topomap correlations 
    mega = []
    # compute gavs per target chan, peak, and reference
    for idx, item in enumerate(df.groupby(['Target_Chan','Peak','Reference'])):
        print(item[0])
        # grand averages of all nights
        #gavs = mne.grand_average(list(item[1].Epochs))
        try:
            eps = [list(item[1].Epochs)[i].get_data(units='uV', tmin=-0.01).mean(1) for i in range(len(item[1]))]
            contrast = np.concatenate([eps])
            mega.append([item[0], list(item[1].Subject + '_' + item[1].Condition), 
                          #gavs.get_data(units='uV', tmin=-0.01).mean(1), 
                          contrast])
        except:
            eps = [list(item[1].Epochs)[i].get_data(tmin=-0.01).mean(1)*1e3 for i in range(len(item[1]))]
            contrast = np.concatenate([eps])
            mega.append([item[0], list(item[1].Subject + '_' + item[1].Condition),
                          #gavs.get_data(tmin=-0.01).mean(1)*1e3, 
                          contrast])
    
    # combos to compute pairwise correlations
    combos_csd_neg = list(map(list, [('C3', 'NegPeak', 'csd'),
                                      ('C4', 'NegPeak', 'csd'),
                                      ('Fz', 'NegPeak', 'csd')]))
    combos_csd_pos = list(map(list, [('C3', 'PosPeak', 'csd'),
                                      ('C4', 'PosPeak', 'csd'),
                                      ('Fz', 'PosPeak', 'csd')]))
    combos_lm_neg = list(map(list, [('C3', 'NegPeak', 'targeted'),
                                    ('C4', 'NegPeak', 'targeted'),
                                    ('Fz', 'NegPeak', 'targeted')]))
    combos_lm_pos = list(map(list, [('C3', 'PosPeak', 'targeted'),
                                    ('C4', 'PosPeak', 'targeted'),
                                    ('Fz', 'PosPeak', 'targeted')]))    
    
    coll = np.asarray([list(np.asarray(mega[i][0]))for i in range(len(mega))])   
    df_corr = pd.DataFrame(columns=['Subject_Cond','Target','Target_means',
                                    'Peak','Reference','Spearman_rho',
                                    'Fishers_ztransformed_rho'])
    for idx, combo in enumerate(zip([combos_csd_neg*2, combos_csd_pos*2,
                                      combos_lm_neg*2, combos_lm_pos*2])):
        combos = list(itertools.combinations(combo[0],2))
        within = [combos[i][0] for i in range(len(combos))]
        between = [combos[i][1] for i in range(len(combos))]
        for x,y in zip(within, between):
            #print(x,y)
            within_index = statistics.mode(np.where((x == coll))[0])
            between_index = statistics.mode(np.where((y == coll))[0])
                
            unity = list(set(mega[between_index][1]) & set(mega[within_index][1]))
    
            pos_within = [unity.index(mega[within_index][1][i]) for i in range(len(mega[within_index][1])) if mega[within_index][1][i] in unity]
            pos_between = [unity.index(mega[between_index][1][i]) for i in range(len(mega[between_index][1])) if mega[between_index][1][i] in unity]
            
            within_mean = mega[within_index][-1][pos_within,:].mean(0)
            between_comparison = mega[between_index][-1][pos_between,:]
            
            comparison = [stats.spearmanr([within_mean, between_comparison[i,:]], axis=1)[0] 
                          for i in range(len(between_comparison))]
            fishers_comparison = np.arctanh(comparison)
            
            print(f' Within: {mega[within_index][0]}, Between: {mega[between_index][0]}')
            
            
            df_corr = df_corr.append(pd.DataFrame({'Subject_Cond': unity,
                                                    'Target': [mega[within_index][0][0] + '_' + mega[between_index][0][0]]*len(unity),
                                                    'Target_means': [mega[within_index][0][0]]*len(unity),
                                                    'Within': [mega[between_index][0][0]]*len(unity), 
                                                    'Peak': [mega[within_index][0][1]]*len(unity),
                                                    'Reference': [mega[within_index][0][2]]*len(unity),
                                                    'Spearman_rho': comparison, 
                                                    'Fishers_ztransformed_rho': list(fishers_comparison)}))
    # reomve duplicates
    df_corr = df_corr.groupby(['Subject_Cond', 'Target', 'Target_means',
                                'Peak', 'Reference']).mean().reset_index()
            
    # mark within versus between
    comparisons = []
    for elem in range(len(df_corr)):
        if len(np.unique(df_corr['Target'][elem].split('_'))) > 1:
            comparisons.append("Between")
        else:
            comparisons.append("Within") 
    df_corr['Comparison'] = comparisons
    
    # mark when within is bigger than between --> descriptive stat
    df_corr_stacked = df_corr.groupby(['Subject_Cond','Peak',
                                        'Reference','Target_means']).mean().reset_index()
    statuses = []
    for (sub, peak, ref, target), df_ in df_corr.groupby(['Subject_Cond','Peak',
                                                          'Reference','Target_means']):
        df_['order'] = df_.Fishers_ztransformed_rho.rank()
        if len(df_['order']) < 3:
            limit = 2
        else:
            limit = 3
        if df_[df_['Comparison']=='Within'].order.to_numpy() == limit:
            statuses.append(True)
        else:
            statuses.append(False)
    df_corr_stacked['Within_v_Between'] = statuses
    wb = df_corr_stacked.groupby(['Peak','Reference','Target_means']).mean()['Within_v_Between']
    print(wb)
    
    # stack into within and between columns for t-test stats and paired plots
    # df_new = df_corr.pivot_table(columns='Comparison', index=['Subject_Cond','Peak',
    #                                                           'Reference','Target_means']).reset_index()
    # df_new.columns = ['_'.join(col) for col in df_new.columns.values]
    # fig, axs = plt.subplots(ncols=4, nrows=3, sharex=True, sharey=True, 
    #                         **dict(figsize=(20, 12)))
    fig, axs = plt.subplots(ncols=3, nrows=2, sharex=True, sharey=True, 
                            **dict(figsize=(20, 12)))
    ttests = []
    for j, [(peak, ref, targets), (df_)] in enumerate(df_corr.groupby(['Peak','Reference','Target_means'])):
        print(j, peak, ref, targets)
        df_new_ = df_.groupby(['Subject_Cond','Target_means', 'Peak', 
                               'Reference','Comparison']).mean().reset_index()
        # mark when within is bigger than between --> descriptive stat
        statuses = []
        for (sub, peak, ref, target), df_ in df_new_.groupby(['Subject_Cond','Peak',
                                                              'Reference','Target_means']):
            df_['order'] = df_.Fishers_ztransformed_rho.rank()
            if len(df_['order']) < 2:
                limit = 1
            else:
                limit = 2
            if df_[df_['Comparison']=='Within'].order.to_numpy() == limit:
                statuses.append(True)
            else:
                statuses.append(False)
        print(f"Proportion of subjects with higher correlation values within versus between: {np.mean(statuses).round(3)}")
        # df_new_['Within_v_Between'] = statuses
        # wb = df_new_.groupby(['Peak','Reference','Target_means']).mean()['Within_v_Between']
        # print(wb)
                
        # do stats here as well, warum nicht
        # pg.ttest(df_new_[df_new_.Comparison=='Within'].Fishers_ztransformed_rho, 
        #          df_new_[df_new_.Comparison=='Between'].Fishers_ztransformed_rho, paired=True)
        ttest = pg.pairwise_tests(dv='Fishers_ztransformed_rho', within='Comparison',
                                  subject='Subject_Cond', padjust=None, data=df_new_)
        pg.plot_paired(df_new_, dv='Fishers_ztransformed_rho', within='Comparison',
                       subject='Subject_Cond', order=None, boxplot=True,
                       boxplot_in_front=True, orient='v', figsize=(4, 4), dpi=100, 
                       ax=axs.flatten()[j], colors=['green', 'grey', 'indianred'], 
                       pointplot_kwargs={'scale': 0.6, 'marker': '.'}, 
                       boxplot_kwargs={'color': 'lightslategrey', 'width': 0.2})   
        axs.flatten()[j].title.set_text(f'{peak}_{ref}_{targets} (p = {ttest["p-unc"].to_numpy()[0].round(10)})')
        ttest['Comparison'] = f'{peak}_{ref}_{targets}'
        ttests.append(ttest)
    
    plt.tight_layout()
    plt.show()
    res = pd.concat(ttests).reset_index(drop=True)
    res.to_csv(p_path + 'topo_ttests.csv')
    fig_path = '/media/administrator/data/Study_2_data/figures/group/'
    plt.savefig(fig_path + 'Paired_plots_topo_correlations.png')
    
    # Stats - compute for each peak, reference, seperately
    from pymer4.models import Lmer
    lmers = []
    for (peak, ref), df_ in df_corr.groupby(['Peak','Reference']):
        print(peak, ref)
        # rm_anova = df_.rm_anova(dv='Fishers_ztransformed_rho', 
        #                         within=['Target_means', 'Comparison'], 
        #                         subject='Subject_Cond', detailed=True)
        model = Lmer('Fishers_ztransformed_rho ~ Target_means*Comparison + (1|Subject_Cond)', 
                      data=df_)
        model.fit(factors={'Target_means': ['C3','C4','Fz'],
                            'Comparison': ['Within','Between']}, summarize=False)
        anova = model.anova(force_orthogonal=True)
        anova['Tested_effect'] = ['Target_means', 'Comparison', 'Interaction']
        anova['Comparison'] = [f'{peak}_{ref}_{targets}']*3
        print(anova)
        lmers.append(anova)
    
    res_lmm = pd.concat(lmers).reset_index(drop=True)
    res_lmm.to_csv(p_path + 'topo_lmms.csv')
    
#%%
## 1. Pre-process epochs first
#pre_process_so_local_epochs()

## 2. Run statistical test on subject vs. group correlations 
# df, df_evoked = pre_process_so_local_gavs()
p_path = '/media/administrator/data/Study_2_data/stats/'
# df.to_pickle(p_path + 'df.p')
# df_evoked.to_pickle(p_path + 'df_contrast.p')

## 3. Load dataframes 
df = pd.read_pickle('/media/administrator/data/Study_2_data/stats/df.p')
df_evoked = pd.read_pickle('/media/administrator/data/Study_2_data/stats/df_contrast.p')
df = df[df.Reference=='csd']
df_evoked = df_evoked[df_evoked.Reference=='csd']

## 4. Topo correlation stats
#group_stats_topo_correlations(df, p_path)

#%%
# #%%
# file = '/media/administrator/data/Study_1_data/Raw_data/Experimental/YIOYSRPX_3/sleepstim_R001.xdf'

# raw.set_montage(mne.channels.make_standard_montage('standard_1005')) 
# raw.set_eeg_reference(['M1','M2'])

# # compute power
# power = raw.compute_psd(method='multitaper', fmin=0.3, fmax=2.0, picks='eeg')
# fm = fooof.FOOOFGroup(max_n_peaks=SPEC_NR_PEAKS)
# fm.fit(power.freqs, power._data.mean(0))
# delta_bands = fooof.analysis.get_band_peak_fg(fm, [0.3, 2.0])
# peak = np.nanmean(delta_bands[:, 0])

# #SSD
# filters, patterns = compute_ssd(raw, signal_bp=(0.5, 2), sf=sf, use_mne=False,
#                                 noise_bp=(0.1, 30), noise_bs=(49, 51))
# raw_ssd = apply_filters(raw, filters, use_mne=False)
# #raw_ssd.filter(0.3, 2, verbose=False)
# filtparams = signal.butter(2, (0.3, 2), fs = sf, btype='bandpass')
# raw_ssd = signal.filtfilt(*filtparams, raw_ssd[0:min(patterns.shape),:], axis=-1)

# # compute amplitude corrected spatial pattern coefficients
# #std_comp = np.std(raw_ssd._data, axis=-1)
# std_comp = np.std(raw_ssd, axis=1)
# weighted_patterns = std_comp * patterns

# # spatial complexity
# metric = compute_sensor_complexity(weighted_patterns, 10)

# # epoch based computation
# events = mne.make_fixed_length_events(raw, id=1, duration=2.0, overlap=1.0)
# # Epoch length is 5 seconds.
# epochs = mne.Epochs(raw, events, tmin=0., tmax=2,
#                     baseline=None, preload=True)
# epochs.set_montage(mne.channels.make_standard_montage('standard_1005')) 

# #SSD
# filters, patterns = compute_ssd(epochs, signal_bp=(12, 16), 
#                                 noise_bp=(0, 30), noise_bs=(49, 51))
# epochs_ssd = apply_filters(epochs, filters)

# # compute amplitude corrected spatial pattern coefficients
# std_comp = np.std(epochs_ssd._data, axis=-1)
# weighted_patterns = std_comp.mean(0) * patterns

# # spatial complexity
# metric = compute_sensor_complexity(weighted_patterns, 10)

# # plot patterns and spatial complexity
# plot_patterns(weighted_patterns, epochs, 4)
# yasa.topoplot(pd.Series(metric, epochs.ch_names), cmap='Spectral_r')

# #%%
# ## Imitate sliding-window procedure of original experiment
# crit_reconstruct = [0]; crit_reconstruct_up = [0]
# time_reconstruct = [0]; time_reconstruct_up = [0]
# fs = 512
# ix = 2*fs

# b,a = signal.butter(2, 4, fs = fs)
# d,c = signal.butter(2, (0.2, 30), fs = fs, btype='bandpass')
# # fnyq = fs/2
# # N, beta = signal.kaiserord(60.0, 5/fnyq)
# # taps = signal.firwin(N, 4/fnyq, window=('kaiser', beta))
# while ix < len(C3):
# #while ix <= 512*60*10:
#     ix0 = ix - 2*fs
#     #pick 30 s window
#     window = C3[int(ix0):int(ix)]
#     d = signal.filtfilt(b,a,window)
#     #d = signal.filtfilt(taps, 1.0, window)
#     d -= np.median(d)
#     #plt.plot(d, 'k')
    
#     # # median filter
#     # d = scipy.ndimage.median_filter(d, size=50)
#     # # savitzky golay filter
#     # d = hp.smooth_signal(d, sample_rate = fs, window_length=int(fs*1), polyorder=3)
#     # sin convolution
#     # d = np.convolve(np.sin(1), d)
    
#     # SSD fun
#     data_wind = data[int(ix0):int(ix),:]
#     #data_wind = signal.filtfilt(d,c,data_wind, axis=0)
#     data_wind = signal.filtfilt(b,a,data_wind, axis=0)
#     data_wind -= np.median(data_wind, 0)
#     filters, patterns = compute_ssd(data_wind.T, signal_bp=(0.3, 2), use_mne=False,
#                                     noise_bp=(0.1, 30), noise_bs=(0.1, 30), sf=fs)
#     epochs_ssd = apply_filters(data_wind.T, filters)

#     # compute amplitude corrected spatial pattern coefficients
#     std_comp = np.std(epochs_ssd, axis=-1)
#     weighted_patterns = std_comp * patterns

#     # spatial complexity
#     metric = compute_sensor_complexity(weighted_patterns, len(weighted_patterns.shape)) #10

#     ix += 2*fs #remove after testing 
    
#     # plot patterns and spatial complexity
#     #plot_patterns(weighted_patterns, epoch, 4)
#     #yasa.topoplot(pd.Series(metric, epoch.ch_names), cmap='Spectral_r')
#     yasa.topoplot(pd.Series(weighted_patterns[:,0], ch_names))
#     yasa.topoplot(pd.Series(metric, ch_names), cmap='Spectral_r')
    
#     minamp = min(np.percentile((d[-2* int(sf):]), 10), -35)
#     crit = min(d[int(-0.02*sf):])

#     # if (min(d[-2:]) < -35) and (new_times[ix] - time_reconstruct[-1] > 3):
#     if crit < minamp and (new_times[ix] - time_reconstruct[-1] > 4): 
#         time_reconstruct.append(new_times[ix])
#         crit_reconstruct.append(crit)
#         ts_up = new_times[ix] + .475
#         time_reconstruct_up.append(ts_up)
#         crit_reconstruct_up.append(d[int(ts_up)])
#     ix += 2*fs
    
#     if len(crit_reconstruct) > 250:
#         break

# #%%
# ## Plot results
# plt.figure()
# plt.plot(new_times, C3_filt_notch)
# plt.plot(time_reconstruct[1::], crit_reconstruct[1::], 'xr')
# plt.plot(time_reconstruct_up[1::], crit_reconstruct_up[1::], 'xg')

# #%%
# reject_criteria = dict(eeg=550e-6)
# ts_pinknoise_times_sync = [np.argmin(np.abs(new_times - ts)) for ts in time_reconstruct]
# center_crit, _ = yasa.get_centered_indices(data[:, ch_names.index('C3')], 
#                                            np.asarray(ts_pinknoise_times_sync[1::]), 
#                                            npts_before = sf*2, npts_after = sf*2)
# info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='eeg')
# epochs_crit = mne.EpochsArray(np.swapaxes(data[center_crit], 1, 2)/1e6, 
#                               info, tmin = -2, baseline=(-2, -1), proj=False)
# epochs_crit.filter(0.5, 2)
# epochs_crit.set_eeg_reference(['M1','M2'])
# epochs_crit.drop_bad(reject = reject_criteria) 
# epochs_crit.set_montage(mne.channels.make_standard_montage('standard_1005'))
# epochs_crit.plot_image('C3') 
# epochs_crit.average().plot_joint()
# mne.preprocessing.compute_current_source_density(epochs_crit).average().plot_joint()

# #%%
# ## Epoch plotting with MNE
# # down
# reject_criteria = dict(eeg=550e-6)
# ts_pinknoise_times_sync = [np.argmin(np.abs(new_times - ts)) for ts in time_reconstruct]
# ts_pinknoise_times_sync_up = [np.argmin(np.abs(new_times - ts)) for ts in time_reconstruct_up]

# center_crit, _ = yasa.get_centered_indices(C3_filt_notch_broad, np.asarray(ts_pinknoise_times_sync[1::]), 
#                                            npts_before = sf*4, npts_after = sf*4)
# info = mne.create_info(ch_names=['C3'], sfreq=sf, ch_types='eeg')
# epochs_crit = mne.EpochsArray(np.expand_dims(C3_filt_notch_broad[center_crit], 1)/1e6, info, tmin = -4, 
#                               baseline=(-4, -1.5), proj=False)
# art_idx = art_detect(epochs_crit)
# epochs_crit.drop(art_idx)
# epochs_crit.drop_bad(reject = reject_criteria) 
# # epochs_crit.average(method='mean').plot()
# epochs_crit.plot_image()

# # up
# center_up, _ = yasa.get_centered_indices(C3_filt_notch_broad, np.asarray(ts_pinknoise_times_sync_up[1::]), 
#                                          npts_before = sf*4, npts_after = sf*4)
# info = mne.create_info(ch_names=1, sfreq=sf, ch_types='eeg')
# epochs_up = mne.EpochsArray(np.expand_dims(C3_filt_notch_broad[center_up], 1)/1e6, info, tmin = -4, 
#                             baseline=(-4, -1.5), proj=False)
# art_idx = art_detect(epochs_up)
# epochs_up.drop(art_idx)
# epochs_up.drop_bad(reject = reject_criteria) 
# # epochs_up.average(method='mean').plot()
# epochs_up.plot_image()     

# #%%
# def old_test(data=data, fs=sf, ch_names=ch_names):
#     mne.set_log_level("CRITICAL")
#     epochs, epochs_csd = [], []
#     info = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='eeg')
#     info_csd = mne.create_info(ch_names=ch_names, sfreq=sf, ch_types='csd')
#     info_csd.set_montage(mne.channels.make_standard_montage('standard_1005'))
#     #crit_reconstruct = [0]; crit_reconstruct_up = [0]
#     #time_reconstruct = [0]; time_reconstruct_up = [0]
#     fs = 512
#     #ix = 2*fs
#     b,a = signal.butter(2, 4, fs = fs)
#     #while ix <= 512*60*120:
#     for ep in neg_peak_idx:
#         #ix0 = ix - 2*fs
#         #pick 2s window
#         #data_wind = data[int(ix0):int(ix),:]
#         data_wind = data[ep,:]
#         data_wind = signal.filtfilt(b,a,data_wind, axis=0)
#         data_wind -= np.median(data_wind, 0)
#         #data_wind = scipy.signal.detrend(data_wind, axis=-1, type='constant')
#         data_wind = re_reference(data_wind, ch_names, reference='mastoids')
#         data_wind_csd = re_reference(data=data_wind/1e3, ch_names=ch_names, 
#                                      trans_csd=trans_csd, sf=512, reference='csd').T
        
#         # determine 20th percentile of values for threshold of lowest values
#         # FLIP BASED ON CONDITION
#         percentiles = np.percentile(data_wind_csd[int(sf*-.04):,:].mean(0), 20)
        
#         ## Ground truth analysis should probably contain interpolated data
#         epoch = mne.EpochsArray(np.expand_dims(data_wind.T, 0)/1e6, 
#                                 info, tmin = -1.98, baseline=None)
#         epoch.set_montage(mne.channels.make_standard_montage('standard_1005'))
#         epoch = detect_bad_interpolate(epoch, method='NK')

#         # csd epoch
#         epoch_csd = mne.preprocessing.compute_current_source_density(epoch, lambda2=1e-03, 
#                                                                      verbose=0)
        
#         ##
#         d = data_wind[:, ch_names.index('C3')]
#         #plt.plot(d, 'k')       
#         d_csd = data_wind_csd[:, ch_names.index('C3')]        
#         #plt.plot(d_csd, 'm')
#         #corr = np.corrcoef([d[int(-.5*sf):], d_csd[int(-.5*sf):]]).min()
#         #print(f'Correlation between C3 signals in last 500 ms: {corr.round(3)}')
        
#         ## Append data
#         if d_csd[int(sf*-.04):].mean() < percentiles:
#             epochs.append(epoch)
#             epochs_csd.append(epoch_csd)
    
#     # grand averages (where C3 is in the 80th percentile or higher) 
#     gav = mne.grand_average([epochs[i].average() for i in range(len(epochs)-1)])
#     gav_csd = mne.grand_average([epochs_csd[i].average() for i in range(len(epochs_csd)-1)])
#     gav.plot()
#     gav_csd.plot()
        
#     # SSD fun
#     for name, inst in zip(['lm', 'csd'], [data_wind, data_wind_csd]):
#         print(name)
#         filters, patterns = compute_ssd(inst.T, signal_bp=(0.3, 2), use_mne=False,
#                                         noise_bp=(0.1, 30), noise_bs=(0.1, 30), sf=fs)
#         epochs_ssd = apply_filters(inst.T, filters)
        
#         # narrowband filter 
#         #filtparams = signal.butter(2, 4, fs = sf)        
#         #epochs_ssd = signal.filtfilt(*filtparams, epochs_ssd, axis=-1)
#         epochs_ssd = signal.filtfilt(b,a, epochs_ssd, axis=-1)
    
#         # compute amplitude corrected spatial pattern coefficients
#         std_comp = np.std(epochs_ssd, axis=-1)
#         weighted_patterns = std_comp * patterns
    
#         # spatial complexity
#         metric = compute_sensor_complexity(weighted_patterns, 
#                                            np.min(weighted_patterns.shape)) #10
        
#         # plot spatial complexity
#         yasa.topoplot(pd.Series(metric, ch_names), cmap='Spectral_r')
#         # plot voltage in last 20 ms
#         yasa.topoplot(pd.Series(np.mean(data_wind[int(-0.02*sf):,:],0), ch_names), cmap='Spectral_r')
#         # plot spatial patterns
#         yasa.topoplot(pd.Series(weighted_patterns[:,0], ch_names), cmap='Spectral_r')
#         # correlation between voltage maps and spatial pattern maps
#         print(f'Correlation between maps is: {np.corrcoef(weighted_patterns[:,0], np.median(data_wind[int(-0.02*sf):,:],0)).min().round(3)}')
        
        
#         # # minamp 
#         # minamp = min(np.percentile((d[-2* int(sf):]), 10), -35)
#         # crit = min(d[int(-0.02*sf):])
        
#         # minamp_csd = min(np.percentile((d_csd[-2* int(sf):]), 10), -35)
#         # crit_csd = min(d_csd[int(-0.02*sf):])

#         # if (min(d[-2:]) < -35) and (new_times[ix] - time_reconstruct[-1] > 3):
#         #if crit < minamp and crit_csd < minamp_csd and (new_times[ix] - time_reconstruct[-1] > 3): 
#         # if crit_csd < minamp_csd and np.ptp(d_csd) < 500 and (new_times[ix] - time_reconstruct[-1] > 3): 
#         #     time_reconstruct.append(new_times[ix])
#         #     crit_reconstruct.append(crit)
#         #     #ts_up = new_times[ix] + .475
#         #     #time_reconstruct_up.append(ts_up)
#         #     #crit_reconstruct_up.append(d[int(ts_up)])
#         #     #plt.figure()
#         #     plt.plot(d)
#         #     #yasa.topoplot(pd.Series(metric, ch_names), cmap='Spectral_r')
#         #     #yasa.topoplot(pd.Series(np.mean(data_wind[int(-0.02*sf):,:],0), ch_names), cmap='Spectral_r')
#         #     #yasa.topoplot(pd.Series(weighted_patterns[:,0], ch_names), cmap='Spectral_r', vmax=20)
#         #     # print(f'Correlation between maps is: {np.corrcoef(weighted_patterns[:,0], np.median(data_wind[int(-0.02*sf):,:],0)).min().round(3)}')
            
#         #     if len(time_reconstruct) == 25:
#         #         break
        
#         # # recompute every 250 ms, the last 2 seconds
#         # ix += int(.5*fs)          



