#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:43:55 2022

@author: administrator
"""

from scipy.linalg import eigh
from scipy.interpolate import RectBivariateSpline
from scipy.signal import find_peaks, welch, detrend
import yasa
import numpy
import fooof
runcell(0, '/home/administrator/sleep_stimulation/sleepstim/Analysis/auditory_stim_validation.py')

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

""" Functions to compute Spatial-Spectral Decompostion (SSD).

Reference
---------
Nikulin VV, Nolte G, Curio G.: A novel method for reliable and fast
extraction of neuronal EEG/MEG oscillations on the basis of
spatio-spectral decomposition. Neuroimage. 2011 Apr 15;55(4):1528-35.
doi: 10.1016/j.neuroimage.2011.01.057. Epub 2011 Jan 27. PMID: 21276858.

"""

from scipy.linalg import eig
import mne
import matplotlib.pyplot as plt

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
        print("Warning: Input data is not full rank")
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

def apply_filters(raw, filters, prefix="ssd"):
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

    raw_projected = raw.copy()
    components = filters.T @ raw.get_data()
    nr_components = filters.shape[1]
    raw_projected._data = components

    ssd_channels = [f"{prefix}{i+1}" for i in range(nr_components)]
    mapping = dict(zip(raw.info["ch_names"], ssd_channels))
    mne.channels.rename_channels(raw_projected.info, mapping)
    raw_projected.drop_channels(raw_projected.info["ch_names"][nr_components:])

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

def compute_ssd(raw, signal_bp, noise_bp, noise_bs):
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
    raw_noise = raw_noise.filter(
        l_freq=noise_bs[1],
        h_freq=noise_bs[0],
        method="iir",
        iir_params=iir_params,
        verbose=False,
    )

    # compute covariance matrices for signal and noise contributions

    if raw_signal._data.ndim == 3:
        cov_signal = mne.compute_covariance(raw_signal, verbose=False).data
        cov_noise = mne.compute_covariance(raw_noise, verbose=False).data
    elif raw_signal._data.ndim == 2:
        cov_signal = np.cov(raw_signal._data)
        cov_noise = np.cov(raw_noise._data)

    # compute spatial filters
    filters = compute_ged(cov_signal, cov_noise)

    # compute spatial patterns
    patterns = compute_patterns(cov_signal, filters)

    return filters, patterns

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
        mne.viz.plot_topomap(patterns[:, i], raw.info, axes=ax1)

    fig.show()

#file = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental/886MCPKG_2_preproc_data.p'
file = '/media/administrator/data/Study_1_data/Raw_data/Experimental/YIOYSRPX_3/sleepstim_R001.xdf'

#%%
if 'Pre-processed_data' in file:
    Data = load_preprocessed_data(file)[0]
    Data.downsample(128)
    
    hypno_file = '/media/administrator/data/Study_1_data/Pre-processed_data/EDFs/Auto_hypnograms/886MCPKG_1_hypnogram.npy'
    hypno = yasa.hypno_upsample_to_data(np.load(hypno_file), sf_hypno=1, sf_data=Data.sfreq, 
                                        data=Data.data)
    
    signal = mne.filter.filter_data(Data.data.T, sfreq=Data.sfreq, l_freq=None, h_freq=4., 
                                    verbose=0)
    noise = mne.filter.filter_data(Data.data.T, sfreq=Data.sfreq, l_freq=None, h_freq=20., 
                                   verbose=0)
    
    Data.data = Data.data[hypno==3,:]
    
    signal_filt = detrend(signal[0:64,:], type='constant')
    noise_filt = detrend(noise[0:64,:], type='constant')
    signal_cov = np.cov(signal_filt)
    noise_cov = np.cov(noise_filt)
    eeg_index = [i for i, x in enumerate(Data.chtypes) if x == "eeg"]
    eeg_chans = [Data.chans[i] for i in eeg_index]
    
    # Get the eigenvalues / eigenvectors
    eigval, eigvec = eigh(noise_cov, signal_cov)
    
    # Flip to descending order
    eigval = np.flip(eigval)
    eigvec = np.fliplr(eigvec)
    
    print('Eigenvalues =', list(np.round(eigval, 2)))
    
    # Apply spatial filters by multiplying data with eigenvectors
    sf_comp_nrem = np.dot(Data.data.T, eigvec).T
    print(sf_comp_nrem.shape)#

else:
    import liesl
    streams = liesl.XDFFile(file)
    raw_eeg = streams['eego'].time_series
    sf = streams['eego'].nominal_srate
    
    chans = streams['eego'].channel_labels
    types = streams['eego'].channel_types
    types[0:23] = ['eeg']*23 
    if 'EOG' in chans:
        types[chans.index('EOG')] = 'misc'

    del streams
    
    raw_eeg = signal.resample_poly(raw_eeg, sf/4, sf)
    sf = sf/4
    #hypno_file = '/media/administrator/data/Study_1_data/Pre-processed_data/EDFs/Auto_hypnograms/886MCPKG_1_hypnogram.npy'
    #hypno = yasa.hypno_upsample_to_data(np.load(hypno_file), sf_hypno=1, sf_data=sf, data=raw_eeg)
    hypno_file = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/YIOYSRPX_3_hypno.txt'
    hypno = yasa.hypno_upsample_to_data(unravel_hypnogram_visbrain(hypno_file), sf_hypno=1, sf_data=sf, 
                                        data=raw_eeg)

    filtparams_lp = signal.butter(4, (0.5, 30), fs = sf, btype='bandpass')
    filt_data = signal.filtfilt(*filtparams_lp, raw_eeg, axis=0)
    filt_data = detrend(filt_data, type='constant')
    n3_data = filt_data[hypno==3, :]
    del raw_eeg, filt_data
    n3_data = n3_data[:, np.where(np.asarray(types)=='eeg')[0]]
    
    info = mne.create_info(ch_names=list(np.asarray(chans)[np.where(np.asarray(types)=='eeg')[0]]), 
                           ch_types=['eeg']*23, sfreq=128)
    raw = mne.io.RawArray(data=n3_data.T, info=info)
    raw.set_montage(mne.channels.make_standard_montage('standard_1005')) 
    raw.set_eeg_reference(['M1','M2'])
    
    # compute power
    power = raw.compute_psd(method='multitaper', fmin=0.3, fmax=2.0, picks='eeg')
    fm = fooof.FOOOFGroup(max_n_peaks=SPEC_NR_PEAKS)
    fm.fit(power.freqs, power._data.mean(0))
    delta_bands = fooof.analysis.get_band_peak_fg(fm, [0.3, 2.0])
    peak = np.nanmean(delta_bands[:, 0])
 
    #SSD
    filters, patterns = compute_ssd(raw, signal_bp=(0.5, 2), 
                                    noise_bp=(0, 30), noise_bs=(49, 51))
    raw_ssd = apply_filters(raw, filters)
    raw_ssd.filter(peak - 0.5, peak + 0.5, verbose=False)
    
    # compute amplitude corrected spatial pattern coefficients
    std_comp = np.std(raw_ssd._data, axis=1)
    weighted_patterns = std_comp * patterns
    
    # spatial complexity
    metric = compute_sensor_complexity(weighted_patterns, 10)
    
    # epoch based computation
    events = mne.make_fixed_length_events(raw, id=1, duration=2.0, overlap=1.0)
    # Epoch length is 5 seconds.
    epochs = mne.Epochs(raw, events, tmin=0., tmax=2,
                        baseline=None, preload=True)
    epochs.set_montage(mne.channels.make_standard_montage('standard_1005')) 

    #SSD
    filters, patterns = compute_ssd(epochs, signal_bp=(12, 16), 
                                    noise_bp=(0, 30), noise_bs=(49, 51))
    epochs_ssd = apply_filters(epochs, filters)
    
    # compute amplitude corrected spatial pattern coefficients
    std_comp = np.std(epochs_ssd._data, axis=-1)
    weighted_patterns = std_comp.mean(0) * patterns
    
    # spatial complexity
    metric = compute_sensor_complexity(weighted_patterns, 10)
    
    # plot patterns and spatial complexity
    plot_patterns(weighted_patterns, epochs, 4)
    yasa.topoplot(pd.Series(metric, epochs.ch_names), cmap='Spectral_r')
        
    # iterative epochs
    sc = pd.DataFrame()
    for idx, epoch in enumerate(epochs):
        #print(idx)
        filters, patterns = compute_ssd(epochs[idx], signal_bp=(0.5, 2), 
                                        noise_bp=(0, 30), noise_bs=(49, 51))
        epochs_ssd = apply_filters(epochs[idx], filters)
    
        # compute amplitude corrected spatial pattern coefficients
        std_comp = np.std(epochs_ssd._data, axis=-1).squeeze()
        weighted_patterns = std_comp * patterns
        
        # spatial complexity
        metric = compute_sensor_complexity(weighted_patterns, 10)
        
        # 
        sc = pd.concat([sc, pd.DataFrame(metric, epochs[idx].ch_names).T])
        
        # plot patterns and spatial complexity
        plot_patterns(weighted_patterns, epochs[idx], 4)
        yasa.topoplot(pd.Series(metric, epochs[idx].ch_names), cmap='Spectral_r')
        

#%% Plotting
for cov in (signal_cov,noise_cov,np.linalg.inv(noise_cov)@signal_cov):
    plt.figure(figsize=(10, 6))
    sns.heatmap(cov, cmap='Blues', square=True, 
                xticklabels=Data.chans[0:64], yticklabels=Data.chans[0:64],
                vmin = np.percentile(np.linalg.inv(noise_cov)@signal_cov, 1), 
                vmax = np.percentile(np.linalg.inv(noise_cov)@signal_cov, 99))
    plt.title('SO variance-covariance matrix')
    plt.xlabel('Channels')
    _ = plt.ylabel('Channels')

_,axs = plt.subplots(1,3,figsize=(8,4))
axs[0].imshow(signal_cov, vmin=np.percentile(signal_cov, 10), 
                          vmax=np.percentile(signal_cov, 90), cmap='jet')
axs[0].set_title('S matrix')
# R matrix
axs[1].imshow(noise_cov, vmin=np.percentile(signal_cov, 10), 
                         vmax=np.percentile(signal_cov, 90), cmap='jet')
axs[1].set_title('R matrix')
# R^{-1}S
axs[2].imshow(np.linalg.inv(noise_cov)@signal_cov,vmin=np.percentile(signal_cov, 10), 
                                                  vmax=np.percentile(signal_cov, 90), cmap='jet')
axs[2].set_title('$R^{-1}S$ matrix')
plt.tight_layout()
plt.show()

#%% SSD decoding
from mne.decoding import SSD
info = mne.create_info(ch_names=Data.chans, ch_types=Data.chtypes, sfreq=128)
raw = mne.io.RawArray(data=Data.data.T, info=info)
# Build epochs as sliding windows over the continuous raw file.
events = mne.make_fixed_length_events(raw, id=1, duration=2.0, overlap=1.0)
# Epoch length is 5 seconds.
epochs = mne.Epochs(raw, events, tmin=0., tmax=2,
                    baseline=None, preload=True)
epochs.drop_channels('EOG')
epochs.set_montage(mne.channels.make_standard_montage('standard_1005')) 
for idx, epoch in enumerate(epochs):
    print(idx)
    ssd = SSD(info=epochs[idx].copy().pick('eeg').info,
              reg='shrinkage',
              sort_by_spectral_ratio=True,  # False for purpose of example.
              filt_params_signal=dict(l_freq=0, h_freq=4,
                                      l_trans_bandwidth=1, h_trans_bandwidth=1),
              filt_params_noise=dict(l_freq=0, h_freq=25,
                                     l_trans_bandwidth=1, h_trans_bandwidth=1))
    ssd.fit(X=epoch[0:64, :])

    # Plot topographies.
    pattern_epochs = mne.EvokedArray(data=ssd.patterns_[:4].T,
                                     info=ssd.info)
    pattern_epochs.plot_topomap(time_format='', 
                                title=f'SO spatial patterns')
    
    # Transform
    ssd_sources = ssd.transform(X=epoch[0:64,:])

    # Get psd of SSD-filtered signals.
    # psd, freqs = mne.time_frequency.psd_array_welch(
    #     ssd_sources, sfreq=up.copy().pick('eeg').info['sfreq'], 
    #     n_fft=int(up.copy().pick('eeg').info['sfreq']*4))

    # Get spec_ratio information (already sorted).
    # Note that this is not necessary if sort_by_spectral_ratio=True (default).
    spec_ratio, sorter = ssd.get_spectral_ratio(ssd_sources)

    # Plot spectral ratio (see Eq. 24 in Nikulin 2011).
    fig, ax = plt.subplots(1)
    ax.plot(spec_ratio, color='black')
    ax.plot(spec_ratio[sorter], color='orange', label='sorted eigenvalues')
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel(r"Spectral Ratio $\frac{P_f}{P_{sf}}$")
    ax.legend()
    ax.axhline(1, linestyle='--')
    plt.show()