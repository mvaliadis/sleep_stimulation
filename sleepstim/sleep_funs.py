# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:58:01 2020

@author: neuro
"""

import numpy as np
# Only essential imports for remaining functions.
# yasa, mne, etc. are now dependencies of the core modules or not used by epoch_psd.

from sleepstim.core.dsp import re_reference, bfr_butter_filt, bandpower


def epoch_psd(data, fs, ch_names_for_mastoid_ref=None):
    """PSG bandpower feature and classification pre-processing function.

    This function takes a numpy array of the lsl buffer selected data, partitions it into EEG, EMG, 
    and EOG streams, filters the data, then subsequently recombines the data 

    Parameters
    ----------
    data : numpy array of [n_samples x n_chans]
        The lsl derived data. 
        
    fs : int/float
        Sampling rate derived from lsl buffer.
        
    ch_names_for_mastoid_ref : list, optional
        List of channel names. Required if 'mastoids' reference is used in the
        `re_reference` function. If None, 'common average' might be used as fallback
        depending on `re_reference` implementation.

    Returns
    -------
    win : numpy array of shape [n_epochs x (n_chans*n_bands)]
        An array containing features for classifier with bands of interest spread across each 
        channel (i.e., an index of [0 - epoch 1,: 0:5 - bands relative delta-line for channel 1],
        [1 - epoch 2,: 5:10 - bands relative delta-line for channel 2], etc.) 
    """
    # Data selection and filtering
    # Assuming data[:, [0,1,2,3,4]] are EEG channels for referencing
    # and data[:, [0,1,2]] are the specific EEG channels of interest after referencing.
    # data[:, [5,6]] are EOG, data[:, [7,8]] are EMG.
    
    # Handle referencing:
    # The re_reference function in dsp.py expects ch_names if 'mastoids' is used.
    # We need to ensure this is handled correctly.
    # If specific ch_names for data[:,[0,1,2,3,4]] are needed for mastoid ref,
    # they must be passed or this will default/fail in re_reference.
    # For now, using 'common average' as a safe default if ch_names_for_mastoid_ref is None
    # and 'mastoids' was intended. Or, the caller must provide appropriate ch_names.
    
    ref_type = 'common average'
    eeg_channels_for_ref = data[:,[0,1,2,3,4]] # Channels used for computing reference
    
    if ch_names_for_mastoid_ref is not None:
        # This assumes ch_names_for_mastoid_ref corresponds to the channels in eeg_channels_for_ref
        # and that M1/M2 are within these first 5 channels if that's the hardcoded assumption in re_reference
        # This is a bit fragile. A better epoch_psd would take ch_names for `data` directly.
        EEG = re_reference(eeg_channels_for_ref, ch_names=ch_names_for_mastoid_ref, reference='mastoids')
    else:
        EEG = re_reference(eeg_channels_for_ref, ch_names=None, reference='common average')
    
    EEG = EEG[:,[0,1,2]] # Assuming these are the primary EEG channels post-referencing
    
    # EOG data selection
    EOG = np.expand_dims(np.mean(data[:,[5,6]], axis=-1), 1)
    # combine EEG & EOG for filtering (necessary if re-referencing online)
    EEG_EOG = np.concatenate([EEG, EOG], axis=1)
    # bandpass filter data (defaults to 4th order filt, 0.5 - 35 Hz bandpass)
    EEG_EOG = bfr_butter_filt(EEG_EOG, fs)
    
    # select and filter EMG 
    EMG = np.mean(data[:,[7,8]], axis=-1)
    EMG = bfr_butter_filt(np.expand_dims(EMG, 1), fs, lfreq = 10, hfreq = 100)
    
    # combine all data streams back into one array
    # Resulting data_processed will have 3 EEG + 1 EOG + 1 EMG = 5 channels
    data_processed = np.transpose(np.concatenate([EEG_EOG, EMG], axis=1))
    
    ## Safety checks
    assert data_processed.ndim == 2, 'Data must be of 2D [n_chans, n_samples].'
    nchan, npts = data_processed.shape
    if npts < nchan: # Should not happen if input `data` is (samples, chans) and processing is correct
        data_processed = np.transpose(data_processed)
        nchan, npts = data_processed.shape
        print(f'Data was transposed to be in shape [{nchan} channels x {npts} data points].')
        
    ## Compute bandpower of epoch, input: [n_chans, n_samples]
    # bandpower from dsp.py is called here. It computes Welch and then yasa.bandpower_from_psd_ndarray
    win = bandpower(data_processed, fs) # bands is default in dsp.bandpower (6 bands)
    
    ## Reshape data for classifier [epochs (1 for this func), (n_chans*n_bands)]
    # If win is (n_chans, n_bands), then win.T.reshape(1, n_chans * n_bands) is correct.
    # nchan here is 5 (3 EEG, 1 EOG, 1 EMG). Default bands are 6. So 5*6=30 features.
    win = win.T.reshape(1, nchan * 6, order='F') 
    return win
