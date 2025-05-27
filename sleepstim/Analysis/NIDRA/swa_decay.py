#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:31:54 2022

@author: administrator
"""

import yasa
import mne
import logging
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.optimize import curve_fit, OptimizeWarning
from scipy import stats
from yasa.io import set_log_level
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 

logger = logging.getLogger('yasa')

__all__ = ['swa_decay']

def _decay_func(t, asym, intercept, tau):
    """Exponential decay equation."""
    #return (intercept - asym) * np.exp(- tau * t) + asym
    return (intercept - asym) * np.exp(- t / tau) + asym
    
def swa_decay(data, hypno, *, sf=None, ch_names=None, include=(2, 3), freq_swa=(0.5, 4),
              freq_broad=(0.5, 30), epoch_length="5min", win_sec=4, bandpass=True,
              log_transform=False, kwargs_welch=dict(average='median', window='hamming'), verbose=True):
    """
    Calculate the exponential decline of process S across the night using NREM sleep EEG slow-wave activity (SWA).
    """
    ###############################################################################################
    # PREPROCESSING
    ###############################################################################################

    set_log_level(verbose)
    assert isinstance(freq_swa, (tuple, list)), 'band must be a list or a tuple'
    assert isinstance(freq_broad, (tuple, list)), 'band must be a list or a tuple'
    assert isinstance(bandpass, bool), 'bandpass must be a boolean'
    assert isinstance(epoch_length, str), "epoch_length must be a string."

    # Check if input data is a MNE Raw object
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info['sfreq']  # Extract sampling frequency
        ch_names = data.ch_names  # Extract channel names
        data = data.get_data() * 1e3  # Convert from V to uV
        _, npts = data.shape
    else:
        assert isinstance(data, np.ndarray), 'Data must be a numpy array.'
        data = np.atleast_2d(data)
        assert data.ndim == 2, 'Data must be of shape (nchan, n_samples).'
        nchan, npts = data.shape
        assert sf is not None, 'sf must be specified if passing a numpy array.'
        assert isinstance(sf, (int, float))
        if ch_names is None:
            ch_names = ['CHAN' + str(i).zfill(3) for i in range(nchan)]
        else:
            ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
            assert ch_names.ndim == 1, 'ch_names must be 1D.'
            assert len(ch_names) == nchan, 'ch_names must match data.shape[0].'

    if bandpass:
        # Apply FIR bandpass filter
        fmin, fmax = min(freq_broad), max(freq_broad)
        data = mne.filter.filter_data(data, sf, fmin, fmax, verbose=0)

    hypno = np.asarray(hypno)
    assert include is not None, 'include cannot be None if hypno is given'
    include = np.atleast_1d(np.asarray(include))
    assert hypno.ndim == 1, 'Hypno must be a 1D array.'
    assert hypno.size == npts, 'Hypno must have same size as data.shape[1]'
    assert include.size >= 1, '`include` must have at least one element.'
    assert np.in1d(hypno, include).any(), 'None of the stages specified in `include` are present in hypno.'
    assert float(sf).is_integer(), "Sampling frequency must be an integer."
    mask = np.in1d(hypno, include).astype(int)

    ###############################################################################################
    # FIND "CLEAN" NREM PERIODS, i.e. at least X min of consecutive NREM
    ###############################################################################################

    # Use yasa.hypno_find_periods to find equal-length epochs
    epochs = yasa.hypno_find_periods(hypno=hypno, sf_hypno=sf, 
                                     threshold=epoch_length, 
                                     equal_length=True)
    
    epochs = epochs[epochs["values"].isin(include)]

    # Ensure that the columns match what we expect
    if 'start' not in epochs.columns or 'length' not in epochs.columns:
        raise KeyError("Expected 'Start' and 'Length' columns in epochs DataFrame.")

    # Check that we have enough epochs
    n_epochs = epochs.shape[0]
    logger.info(f"{n_epochs} NREM epochs longer than {epoch_length} were found in hypno.")
    if n_epochs < 4:
        print(f"Less than 4 NREM epochs > {epoch_length} were found in hypno")
        raise ValueError(f"Less than 4 NREM epochs > {epoch_length} were found in hypno. SWA decay cannot be calculated. Please decrease {epoch_length}.")

    # Calculate the onset time (relative to sleep onset) of each NREM period
    epochs['time_onset_hrs'] = epochs['start'] / sf / 3600
    epochs['length_hrs'] = epochs['length'] / sf / 3600
    epochs['time_mid_hrs'] = epochs['time_onset_hrs'] + epochs['length_hrs'] / 2

    # Calculate continuous mask with unique value for each epoch
    mask_epoch = np.zeros_like(mask, dtype=int)
    for i, row in epochs.iterrows():
        start = int(row['start'])
        end = int(row['start'] + row['length'])
        mask_epoch[start:end] = i + 1

    ###############################################################################################
    # EXPONENTIAL DECLINE IN SWA, for each channel separately
    ###############################################################################################

    # Calculate SWA absolute power in each period, for each channel
    bp = yasa.bandpower(
        data=data, sf=sf, ch_names=ch_names, hypno=mask_epoch,
        include=list(range(1, max(mask_epoch) + 1)),
        bands=[(freq_swa[0], freq_swa[1], "SWA"), (freq_broad[0], freq_broad[1], "Broad")],
        win_sec=win_sec, relative=True, bandpass=False, kwargs_welch=kwargs_welch)

    # Initialize output
    df_decay = {"Intercept": [], "Asym": [], "Tau": [], "Decay": [], "MAE": [],
                "Amplitude": [], "SWE" : [], "X-data": [], "Y-data": [], "Recording_Length": []}

    # Calculate exponential decline, for each channel
    xdata = epochs['time_mid_hrs'].to_numpy()
    for chan in ch_names:
        # if log_transform:
        #     ydata = np.log(bp.xs(chan, level=-1)["SWA"].to_numpy())
        # else:
        ydata = bp.xs(chan, level=-1)["SWA"].to_numpy()
        try:
            popt, _ = curve_fit(
                _decay_func, xdata, ydata, p0=(0.5, 0.8, 1), bounds=((0, 0, 0), (1, 1, 4)))
            mae = np.mean(np.abs(ydata - _decay_func(xdata, *popt)))
        except (ValueError, RuntimeError, OptimizeWarning) as e:
            logger.error(f"Exponential fit failed. Returning NaN for channel {chan}\nError: {e}")
            popt = np.array([np.nan, np.nan, np.nan])
            mae = np.nan
            
        # Calculate normalized cumulative power
        swe = np.cumsum(ydata) / np.sum(ydata)

        # Append to dict
        df_decay["Asym"].append(popt[0]) #Est. relative SWA power at sleep offset
        df_decay["Intercept"].append(popt[1]) #Est. relative SWA power at sleep onset
        df_decay["Tau"].append(popt[2]) #Time constant of the exponential decline, in hours
        df_decay["Decay"].append(1 / popt[2]) #Exp. slope (= 1 / tau). Larger values indicate a more rapid decay of SWA across the night.
        df_decay["MAE"].append(mae) #Mean absolute error of the exponential fit. Lower is better
        df_decay['Amplitude'] = popt[1] - popt[0] #Degree of decay from sleep onset to offset
        df_decay["X-data"].append(xdata)
        df_decay["Y-data"].append(ydata)
        df_decay["SWE"].append(swe)
        df_decay["Recording_Length"].append(xdata[-1])

    # Convert to dataframe
    return pd.DataFrame(df_decay, index=ch_names)

# Removed local definition of drop_bads_df, will be imported from sleepstim.core.utils

#%%
## Additional funcs
def swa_decay_stim(file, plot=True):
    # Example
    #file = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Raw/SvSchm_1_csd-raw.fif'
        
    # Get subject, night info
    subject, night, _ = file.split('/')[-1].split('-')[0].split('_')

    # Read raw
    raw = mne.io.read_raw(file)
    mode = [annot for annot in raw.annotations if annot['description'].startswith(('pn_stim_', 'nmes_stim_'))][0]['description'].split('_')[0]
                
    # Load hypnogram
    hypno_path = '/media/administrator/Sleep_Data/Processed/Hypnograms/' 
    hypno = np.load(hypno_path + f'{subject}_{night}_hypno.npy')
        
    # Extract homeostatic decay
    df = swa_decay(raw.get_data('csd')*1e3, 
                   hypno=hypno, 
                   sf=raw.info['sfreq'], 
                   ch_names=raw.ch_names[0:64],
                   include=(2, 3),
                   freq_swa=(0.5, 4), 
                   freq_broad=(0.5, 30),
                   epoch_length="5min",
                   win_sec=4, 
                   bandpass=False,
                   kwargs_welch=dict(average='median', window='hamming'), 
                   log_transform=False,
                   verbose=True) 
    df['Mode'] = mode
    df['Subject'] = subject
    df['Night'] = night
    
    # Plot SWA and SWE
    if plot:
        plot_swa_and_swe(df, raw, mode)
        figure_path = '/media/administrator/Sleep_Data/Processed/Figures/Sleep/'
        plt.savefig(figure_path + f'{subject}_{mode}_SWA_decay.png')
        plt.close('all')
    
    return df
        
def plot_swa_and_swe(df, raw, mode):
    # Extract stimulation times with specific prefixes
    stim_annotations = [annot for annot in raw.annotations if annot['description'].startswith(('pn_stim_', 'nmes_stim_'))]
    
    # Get stimulation start times in hours
    stim_times = [annot['onset'] / 3600 for annot in stim_annotations]
    
    # Your existing data
    timepoints = df.loc['C3']['X-data']
    
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 14), layout='constrained')
    
    # Plot SWA
    delta = np.concatenate([df.loc[ch]['Y-data'].reshape(1, -1) for ch in df.index])
    mean_delta = np.nanmean(delta, 0)
    
    axes[0].scatter(timepoints, mean_delta)
    axes[0].plot(timepoints, mean_delta, 'k')
    axes[0].plot(timepoints, delta.T, linewidth=0.3, alpha=0.3, color='k')
    
    # Add small ticks for stimulation times below the x-axis
    for stim_time in stim_times:
        axes[0].vlines(stim_time, ymin=-.2, ymax=0, color='r', linewidth=1)
    
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('SWA')
    axes[0].set_title('Average SWA with Stimulation Times')
    
    # Plot Cumulative power (aka SWE)
    cumsum_power = np.concatenate([np.expand_dims(df.loc[ch]['SWE'], 0) 
                                   for ch in df.index], 
                                  axis=0)
    mean_cumsum_power = np.nanmean(cumsum_power, 0)
    
    axes[1].scatter(timepoints, mean_cumsum_power)
    axes[1].plot(timepoints, mean_cumsum_power, 'k')
    axes[1].plot(timepoints, cumsum_power.T, linewidth=0.3, alpha=0.3, color='k')

    # Add small ticks for stimulation times below the x-axis
    for stim_time in stim_times:
        axes[1].vlines(stim_time, ymin=-0.1, ymax=-0.01, color='r', linewidth=1)
    
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Normalized Cumulative SWA (SWE)')
    axes[1].set_title('Average SWE with Stimulation Times')
    
    plt.show()
           
def plot_group_swa_decay(df, channel=None, log=False):
    if channel:
        df = df[df.Chan == channel]
      
    if log:
        # Apply np.log to the 'Y-data' column
        df['Y-data'] = df['Y-data'].apply(lambda y: np.log(y))
        
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 14), layout='constrained')
    
    # Define a common time grid
    common_time_grid = np.linspace(0, 8, num=480)  # 8 hours, 480 points (every minute)
    
    # Initialize lists to store mean data
    mean_swa_data_pn = []
    mean_swe_data_pn = []
    mean_swa_data_nmes = []
    mean_swe_data_nmes = []

    # Separate data based on mode and interpolate to the common time grid
    for subject in df['Subject'].unique():
        subject_df_pn = df[(df['Subject'] == subject) & (df['Mode'] == 'pn')]
        subject_df_nmes = df[(df['Subject'] == subject) & (df['Mode'] == 'nmes')]
        
        if not subject_df_pn.empty:
            # Interpolate SWA
            y_data_pn = np.vstack(subject_df_pn['Y-data'])
            f_pn = interp1d(subject_df_pn['X-data'].iloc[0], y_data_pn, axis=1, bounds_error=False)
            mean_swa_pn = np.nanmean(f_pn(common_time_grid), axis=0)
            mean_swa_data_pn.append(mean_swa_pn)
            
            # Interpolate SWE
            swe_pn = np.vstack(subject_df_pn['SWE'])
            f_pn_swe = interp1d(subject_df_pn['X-data'].iloc[0], swe_pn, axis=1, bounds_error=False)
            mean_swe_pn = np.nanmean(f_pn_swe(common_time_grid), axis=0)
            mean_swe_data_pn.append(mean_swe_pn)
        
        if not subject_df_nmes.empty:
            # Interpolate SWA
            y_data_nmes = np.vstack(subject_df_nmes['Y-data'])
            f_nmes = interp1d(subject_df_nmes['X-data'].iloc[0], y_data_nmes, axis=1, bounds_error=False)
            mean_swa_nmes = np.nanmean(f_nmes(common_time_grid), axis=0)
            mean_swa_data_nmes.append(mean_swa_nmes)
            
            # Interpolate SWE
            swe_nmes = np.vstack(subject_df_nmes['SWE'])
            f_nmes_swe = interp1d(subject_df_nmes['X-data'].iloc[0], swe_nmes, axis=1, bounds_error=False)
            mean_swe_nmes = np.nanmean(f_nmes_swe(common_time_grid), axis=0)
            mean_swe_data_nmes.append(mean_swe_nmes)
    
    # Convert lists to arrays for easier handling
    mean_swa_data_pn = np.array(mean_swa_data_pn)
    mean_swe_data_pn = np.array(mean_swe_data_pn)
    mean_swa_data_nmes = np.array(mean_swa_data_nmes)
    mean_swe_data_nmes = np.array(mean_swe_data_nmes)
    
    # Average across subjects
    avg_swa_pn = np.nanmean(mean_swa_data_pn, axis=0)
    avg_swe_pn = np.nanmean(mean_swe_data_pn, axis=0)
    avg_swa_nmes = np.nanmean(mean_swa_data_nmes, axis=0)
    avg_swe_nmes = np.nanmean(mean_swe_data_nmes, axis=0)
    
    # Plot SWA
    axes[0].plot(common_time_grid, avg_swa_pn, 'b', label='Average SWA (CLAS)')
    axes[0].plot(common_time_grid, avg_swa_nmes, 'r', label='Average SWA (CLNMES)')
    for swa in mean_swa_data_pn:
        axes[0].plot(common_time_grid, swa, linewidth=0.3, alpha=0.3, color='b')
    for swa in mean_swa_data_nmes:
        axes[0].plot(common_time_grid, swa, linewidth=0.3, alpha=0.3, color='r')
    
    # axes[0].set_ylim(0, 25)
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('Relative SWA')
    axes[0].set_title('Group Average Relative SWA')
    axes[0].legend()
    
    # Plot SWE
    axes[1].plot(common_time_grid, avg_swe_pn, 'b', label='Average SWE (CLAS)')
    axes[1].plot(common_time_grid, avg_swe_nmes, 'r', label='Average SWE (CLNMES)')
    for swe in mean_swe_data_pn:
        axes[1].plot(common_time_grid, swe, linewidth=0.3, alpha=0.3, color='b')
    for swe in mean_swe_data_nmes:
        axes[1].plot(common_time_grid, swe, linewidth=0.3, alpha=0.3, color='r')
    
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Normalized Cumulative SWA (SWE)')
    axes[1].set_title('Group Average SWE')
    axes[1].legend()
    
    plt.show()
       
# # Plot group SWA and SWE
# df = pd.read_pickle('/media/administrator/Sleep_Data/Processed/Statistics/df_swa_decay.p')
# df = drop_bads_df(df) # This would now use the imported version
# plot_group_swa_decay(df, channel=None)
