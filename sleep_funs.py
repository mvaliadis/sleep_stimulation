# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:58:01 2020

@author: neuro
"""

import pyaudio
import numpy as np
import pandas as pd
from reiz import clock
import reiz
import threading
import mne
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, welch, resample, resample_poly
from scipy.integrate import simps
import pickle
import liesl
import yasa
import random
import xml.etree.ElementTree as ET

#%%
## NSRR Cleveland Sleep Dataset Classifier training functions
def process_raw_EDF_cfs(file):
    # import raw edf file
    raw_train = mne.io.read_raw_edf(file + '.edf', eog = ['LOC','ROC'], preload = 'True')
    
    # create dictionary of channels we are interested in  
    mapping = {'C3': 'eeg',
               'C4': 'eeg',
               'M1': 'eeg',
               'M2': 'eeg',
               'LOC': 'eog',
               'ROC': 'eog',
               'EMG2': 'emg',
               'EMG3': 'emg'}
    
    # select channels in object and give labels for channel type
    raw_train.pick_channels(ch_names=list(mapping))
    raw_train.set_channel_types(mapping) 
    
    # rereference eeg data to average of mastoids
    raw_train.set_eeg_reference(ref_channels=['M1','M2'])
     
    # bipolarize eog and emg data 
    raw_train = mne.set_bipolar_reference(raw_train, 'LOC', 'ROC')
    raw_train = mne.set_bipolar_reference(raw_train, 'EMG2', 'EMG3')

    
    # extract data, time, sampling rate information
    EEG = raw_train.get_data(picks='eeg', return_times=False)*1e6
    EOG = raw_train.get_data(picks='eog', return_times=False)*1e6
    EMG = raw_train.get_data(picks='emg', return_times=False)*1e6
    times = raw_train.times
    fs = raw_train.info['sfreq']
    
    # delete mne object as it is no longer necessary
    del raw_train
    
    # filter data 
    EEG = bfr_butter_filt(EEG, fs, lfreq=0.5, hfreq=35)
    EOG = bfr_butter_filt(EOG, fs, lfreq=0.5, hfreq=35)
    EMG = bfr_butter_filt(EMG, fs, lfreq=10, hfreq=100)

    # combine data 
    data = np.concatenate([EEG, EOG, EMG])
    
    # epoch data into 30s segments 
    times, epochs = yasa.sliding_window(data, fs, window=30)
     
    # downsample data to 128 Hz (up-sampled due to ECG being sampled at 512 Hz) 
    epoched_data = []
    for index in range(np.size(epochs, 1)):
        dat = np.expand_dims(downsample_scaled((epochs[:,index,:]), fs, new_sf=128), 1)            
        epoched_data.append(dat)
    epoched_data = np.swapaxes(np.concatenate(epoched_data, axis=1), 2, 0)
    
    # import hypnogram
    stages, stagelens = read_xml(file + '.xml')
            
    # unravel hypnogram
    hypnogram = unravel_hypnogram(stages, stagelens)
  
    return epoched_data, hypnogram


def read_xml(file):
    # import xml annotation file to extract hypnogram
    tree = ET.parse(file)
    
    # obtain roots from xml annotation tree
    root = tree.getroot()
    
    # extract sleep stageing related information
    stages = [] 
    stagelens = []
    for child in root.iter('ScoredEvent'):
        var = child[0].text
        if var == None:
            pass
        elif 'Stages' in var:
            stage = child[1].text
            # append stages (0-5)
            stages.append(int(stage[-1]))
            stagelen = int(float(child[3].text))
            # append epoch lengths in increments of 30s
            stagelens.append(int(stagelen/30))
        else:
            pass
        
    # return numpy arrays with stages and corresponding length info
    return np.array(stages).astype(int), np.array(stagelens).astype(int)


def unravel_hypnogram(stages, stagelens):  
    stage_len = []
    # parse stageing information to fit total of epochs by stages 
    for index, length in enumerate(stagelens):
        if stages[index] == 0:
            stage_len.append(np.zeros(length))
        elif stages[index] == 1:
            stage_len.append(np.ones(length))
        elif stages[index] == 2:
            stage_len.append(2*np.ones(length))
        elif stages[index] == 3:
            stage_len.append(3*np.ones(length))
        # collapse stage 3 and 4
        elif stages[index] == 4:
            stage_len.append(3*np.ones(length))
        elif stages[index] == 5:
            stage_len.append(4*np.ones(length))
    
    hypnogram = np.concatenate(stage_len)
    
    # sanity check - does hypnogram 
    if len(hypnogram) != stagelens.sum():
        raise ValueError('The length of the scaled hypnogram does not match the amount of total epochs')
    
    return hypnogram 


#%%
## Analysis functions

def load_xdf(path: str):
    streams, fileheader = pyxdf.load_xdf(path)
    stream_dict = {}
    for ix, stream in enumerate(streams):
        streamname = stream['info']['name'][0]
        stream_dict[streamname] = stream
    return stream_dict

def unravel_hypnogram_visbrain(hypnogram_file, data):  
    ## TO-DO LATER: integate with other unravel function for NSRR dataset
    # load hypnogram file
    hypno = np.genfromtxt(hypnogram_file, delimiter='\t', dtype=str)
    # define stages 
    stages = hypno[1::,0]
    # define stage duration
    stage_len = np.round(hypno[1::,1].astype(dtype=float))
    # define length of 30s epoched stage as difference between durations
    diff = np.diff(stage_len/30)
    
    # add first epoch as difference between first epoch and 0 and to iterated list
    total_diff = [stage_len[0]/30] + list(diff)
    # parse stageing information to fit total of epochs by stages 
    stagelens = []
    for index, length in enumerate(total_diff):
        print(length)
        if stages[index] == 'Wake':
            stagelens.append(np.zeros(int(length)))
        elif stages[index] == 'N1':
            stagelens.append(np.ones(int(length)))
        elif stages[index] == 'N2':
            stagelens.append(2*np.ones(int(length)))
        elif stages[index] == 'N3':
            stagelens.append(3*np.ones(int(length)))
        elif stages[index] == 'REM':
            stagelens.append(5*np.ones(int(length)))
            
    hypnogram = np.concatenate(stagelens)
   
    # sanity check - does length of hypnogram match data epoch length
    if data.shape[0] != len(hypnogram):
        raise ValueError('The length of the scaled hypnogram does not match the amount of total epochs in the data')
    
    return hypnogram 


def hjorth_mobility(x):
    return np.sqrt(np.var(np.diff(x))/np.var(x))
def hjorth_complexity(x):
    """ calculates Hjorth complexity of the input time series vector x"""
    return hjorth_mobility(np.diff(x))/hjorth_mobility(x)

def downsample_scaled(data, old_sf, new_sf, nint_method='none'):
    """Downsample function 
    
    The following function allows the user to downsample the data, so long 
    as the new sampling rate is a multiple of 100 or 128. The function can 
    now also downsample on non-integer scales with scipy fft resampling and
    polyphase resampling, although the latter is preferable with respect to
    computation time. 
    
    Parameters
    ----------
    data : np.array of shape [n_epochs, n_samples]
          The epoched data.
           
    old_sf : int
            Initial sampling rate.
             
    new_sf : int
            Requested sampling rate.
             
    nint_method : str 'none' (default), 'resample_fft', 'resample_poly'
                Non-integer resampling method from scipy.signal.
                
    Returns
    -------
    data: np.array of shape [n_epochs, n_samples]
        Downsampled data.
    """
      
    # first check if we can downsample to 100 or 128 Hz
    if old_sf >= 128 and new_sf >= 100: 
        if old_sf % 100 == 0 or old_sf % 128 == 0:
            if new_sf % 100 == 0 or new_sf % 128 == 0:
                decim = old_sf/new_sf
                epochs, dpnts = data.shape
                # the following holds true so long as recording > ~25 hrs
                if epochs < dpnts:
                    # forcibly alter shape of data 
                    data = np.transpose(data)
                if old_sf % new_sf == 0:
                    data = data[::int(decim)]
                    print(f'Downsampled data by a factor of {int(decim)}')
                else:
                    if nint_method != 'none':
                        if nint_method == 'resample_fft':
                            data = signal.resample(data, int(len(data)*1/decim)) 
                            print(f'Downsampled data by a factor of {decim}')
                        elif nint_method == 'resample_poly':
                            data = signal.resample_poly(data, new_sf, old_sf)
                            print(f'Downsampled data by a factor of {decim}')
                        print('Please note that the ratio of the old and new sampling rate is a '
                              'non-integer value, and will be resampled with the resampling method you selected.')
                    else:
                        raise ValueError('When downsampling by a non-integer value, please select a valid resampling method.')
            else:
                raise ValueError('The requested sampling rate must be a multiple of 100 or 128')
        else:
            raise ValueError('The initial sampling rate is not divisible by 100 or 128!') 
    else:
        raise ValueError('The initial or requested sampling rate must both be larger than 100 Hz!')
         
    return data


def thresholdcrossings(x, threshold):
    """Find indices of threshold-crossings in a 1D array. 
    
    This function can be utilized to determine zero crossing as well as any 
    negative or postive crossing. 

    Parameters
    ----------
    x : np.array
        One dimensional data vector.

    Returns
    -------
    idx_zc : np.array
        Indices of threshold-crossings

    Examples
    --------

        >>> import numpy as np
        >>> from sleep_funs import thresholdcrossings
        >>> a = np.array([20, 29, -43, -10, 16, 37, 45, -36, -29]) #amplitude values 
        >>> thresholdcrossings(a)
            array([1, 2, 6, 7], dtype=int64)
    """
    pos = x > threshold
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]


def process_raw_EDF(fname):
    """
    The following function exlusively preprocesses physionet datasets and 
    extracts the epoched data along with the corresponding sleep stages for 
    and eventual classification
    
        Parameters
    ----------
    fname : EDF data file(s) 
        fname is root and will append both PSG.edf & Hypnogram.edf

    Returns
    -------
    datArray : np.array
        Epoched data array
    stageArray : np.array
        Epoched stage array
    
    """
    raw_train = mne.io.read_raw_edf(fname + '-PSG.edf', preload=True)
    
    # load in hypnogram
    annot_train = mne.read_annotations(fname + '-Hypnogram.edf')
        
    # filter data and select channels to use
    raw_train.pick_types(include=(['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal', 'EMG submental'])) 
    raw_train.filter(0.5, 35, picks = ['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal'])
    raw_train.notch_filter(49)
    raw_train.filter(20, 45, picks = (['EMG submental']))
    
    # select channels for annotations
    mapping = {'EEG Fpz-Cz':'eeg',
               'EEG Pz-Oz': 'eeg',
               'EOG horizontal': 'eog',
               'EMG submental': 'emg'}
     
    raw_train.set_annotations(annot_train, emit_warning=False)
    raw_train.set_channel_types(mapping)           
       
    # extract 30s events from annotations
    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}
        
    events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)
        
    # create a new event_id that unifies stages 3 and 4
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}

    # create Epochs from the data based on the events found in the annotations
    tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included
        
    epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None)
            
    datArray = epochs_train.get_data()*1e6
    stageArray = epochs_train.events[:,2]
    return datArray, stageArray


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def ROC_curve_plot(rf_roc_auc, fpr, tpr, thresholds):
    plt.figure()
    plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig('Log_ROC')
    plt.show()

# def standardize_features(X_train, X_test):
#     from sklearn.preprocessing import StandardScaler
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
    
#     return X_train, X_test

#%%
## Subject code based parameter pulling function

def subject_param_pull(file, subjID, evening, mean_pk2pk):
    """ This function pulls the participant parameter information from excel file to
        prevent experimenter from directly observing condition relevant information 
        for blinding purposes.

    Parameters
    ----------
    file : str
        Input file name with subject information minus .csv ending.
    subjID : str
        Subject ID #.
    evening : int, either: {0, 1, 2}
        Select the experimental recording evening for the participant. 

    Returns
    -------
    time_delay : float
        Returns requisite time delay based on recording evening, either 0 ms for
        down-state targeting or 500 ms for both sham and up-state targeting.
    volume : int
        Returns volume based on recording evening condition, either 0 for sham and
        1 for both up-state and down-state targeting.

    """
    # load file with subject specific parameters
    subj_cond = np.loadtxt(file + '.csv', delimiter=',', dtype='str', skiprows=1) 
    # determine stimulation condition based on recording evening and subject code 
    index_pos = []
    # obtain index position of subject from excel file
    while index_pos == []:
        index_pos = [i for i,item in enumerate(subj_cond) if subjID in item]
        if index_pos == []:
            raise NameError('Subject code is invalid, please enter a valid subject code! ')
        else:    
            ## time delay + volume based on condition
            # 0 corresponds to sham (same trigger as up, without volume)
            if int(subj_cond[index_pos[0],:][evening]) == 0:
                time_delay = mean_pk2pk
                volume = 0
            # 1 corresponds to up-state targeting 
            elif int(subj_cond[index_pos[0],:][evening]) == 1:
                time_delay = mean_pk2pk
                volume = 1
            # 2 corresponds to down-state targeting     
            elif int(subj_cond[index_pos[0],:][evening]) == 2:
                time_delay = 0
                volume = 1 
                
    return time_delay, volume

#%%
## Online signal pre-processing functions
def bandpower(epochs, fs, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                 (8, 12, 'Alpha'),(12, 16, 'Sigma'), 
                                 (16, 30, 'Beta'), (49, 51, 'Line noise')], relative=True):
    """PSG absolute/relative power band feature extraction.

    This function takes a numpy array of the data & creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn. The function first computes the spectral density based on Welch's 
    method then by wrapping the bandpower_from_psd_ndarray from yasa, a package 
    written by Raphael Vallat, computes the absolute or relative power per frequency
    band of interest. 

    Parameters
    ----------
    epochs : numpy array --> [n_epochs, n_chans, n_samples] or [n_chans, n_samples]
        The epoched data.
        
    fs : int
        Sampling rate. 
    
    bands : Frequency bands of interest.
        Feed in bands of interest up to and limited by the Nyquist limit (half of the sampling rate), 
        beyond the default above. Line noise PSD is used for channel failure detection. 
    
    relative : bool 
        Default is 'True', which takes relative PSD values per epoch of time, otherwise
        select 'False' for absolute PSD values.
        
    see 'yasa.bandpower_from_psd_ndarray' for further documentation related to bandpower extraction.
    https://raphaelvallat.com/yasa/build/html/generated/yasa.bandpower_from_psd_ndarray.html
    
    Returns
    -------
    bp : numpy array of shape --> [n_epochs, n_bands, n_chans] or [n_bands, n_chans] 
        relative/absolute power of data.
    """
    # Define window length sufficiently long encompassing at least two 2 cycles of the lowest frequency of interest
    nperseg = (2 / bands[0][0]) * fs

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(epochs, fs, nperseg=nperseg, average='median')
    
    # extract relative or absolute spectral density values for frequency bands of interest
    bp = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative)
    return bp


def bfr_butter_filt(data, fs, order = 4, lfreq = 0.5, hfreq = 35, btype='pass', method='none'):
    """PSG data buffer extraction butterworth filter.

    This function takes a numpy array of the data & filters it based on given
    parameters. The function primarily wraps scipy butterworth filter construction
    and zero-phase filtfilt implementation. 

    Parameters
    ----------
    data : numpy array of [n_samples x n_chans]
        The data.
        
    fs : int/float
        Sampling rate. 
    
    order : int 
        Order of the filter
        
    lfreq : int
        High-pass filter frequency
    
    hfreq : int
        Low-pass filter frequency
    
    btype : str {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        The type of filter. Default is ‘bandpass’
        
    method : str {'none', 'mne'}
        If data will be later used with mne, please select 'mne'. 
        
    Returns
    -------
    data : numpy array of shape [n_samples x n_chans]
        The filtered data
    """
    ## Safety check - check is necessary only when later utilizing MNE, 
    #  which requires float64 type data 
    if method == 'mne':
        if data.dtype != np.float64:
            data == np.asarray(data, dtype=np.float64)
    ## Construct butter filter (scipy)
    filtparams = butter(order, (lfreq, hfreq), btype=btype, fs = fs)
    # run zero phase digitial filter with butterworth parameters, 
    # but first confirm order of dims
    dpnts, chans = data.shape
    if chans < dpnts:
        data = filtfilt(*filtparams, data, axis=0, padtype='odd')
    else:
        data = filtfilt(*filtparams, data, axis=-1, padtype='odd')
    return data


def re_reference(data, reference='common average'):
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
        ref_data = data[..., [3,4]].mean(-1, keepdims=True)
        data -= ref_data
        #print('Data will be re-referenced to the average of the mastoids!')
    else:
        raise ValueError('Please select a valid reference!')
    
    return data

        
def epoch_psd(data, fs):
    """PSG bandpower feature and classification pre-processing function.

    This function takes a numpy array of the lsl buffer selected data, partitions it into EEG, EMG, 
    and EOG streams, filters the data, then subsequently recombines the data 

    Parameters
    ----------
    data : numpy array of [n_samples x n_chans]
        The lsl derived data. 
        
    fs : int/float
        Sampling rate derived from lsl buffer.
        
    Returns
    -------
    win : numpy array of shape [n_epochs x (n_chans*n_bands)]
        An array containing features for classifier with bands of interest spread across each 
        channel (i.e., an index of [0 - epoch 1,: 0:5 - bands relative delta-line for channel 1],
        [1 - epoch 2,: 5:10 - bands relative delta-line for channel 2], etc.) 
    """
    ## Data selection and filtering
    # TO-DO: Index based on channel names...
    # select and re-reference EEG signal to the common average, where 
    # first 3 should correspond to EEG channels based lsl buffer get_data
    EEG = re_reference(data[:,[0,1,2,3,4]], reference='mastoids')  #Cz, C3, C4
    EEG = EEG[:,[0,1,2]]
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
    data = np.transpose(np.concatenate([EEG_EOG, EMG], axis=1))
    ## Safety checks
    assert data.ndim == 2, 'Data must be of 2D [n_chans, n_samples].'
    nchan, npts = data.shape
    if npts < nchan:
        data = np.transpose(data)
        nchan, npts = data.shape
        print(f'Data was transposed to be in shape [{nchan} channels x {npts} data points].')
    ## Compute bandpower of epoch, input: [n_chans, n_samples]
    win = bandpower(data, fs)
    ## Reshape data for classifier [epochs, (nchans*bands)]
    win = win.reshape(1, nchan*6, order='F')
    return win  
  
def sleep_staging(bfr, indices_to_pull):
    """Sleep Stageing function.

    This function takes an lsl buffer argument and will be the sole argument in the 
    sleep stageing thread, integrating the selection, filtering, re-referencing, channel
    failure, and classification feature creation functions. Classification occurs every 
    15 seconds for the last 30 seconds of data.

    Parameters
    ----------
    bfr : Object
        lsl derived data and info object. 
              
    Returns
    -------
    stage_predictArrays : list of integer values
        A list containg global variable stage_predictArrays, which corresponds to
        the classified stage for 30s epoch of extracted data and could either be:
        0 --> Wake, N1, REM, or 1 --> N2, SWS
        e.g., [1,1,1,0,1,0,0,....].    
    channel_failure : list of integer values
        A list containing the status of selected channels, either if they are intact --> 1,
        or whether they have failed --> 0.
    """
    # %timeit measured at 9.5 ms per 30s before classification 
    global stage_predictArrays
    global channel_failure
    global channel_failureArray
    stage_predictArrays = []
    tz = reiz.clock.now()
    ## Loop till the end ~ 8 hours + extra 15 minutes (12600s)
    #%% load classifier models
    rf_1 = pickle.load(open("rf_model_1_cfs.p", "rb"))            #2 EEG, 1 EOG, 1 EMG
    rf_2 = pickle.load(open("rf_model_2_cfs.p", "rb"))            #1 EEG, 1 EOG, 1 EMG
    rf_3 = pickle.load(open("rf_model_3_cfs.p", "rb"))            #2 EEG, 1 EOG
    rf_4 = pickle.load(open("rf_model_4_cfs.p", "rb"))            #2 EEG, 1 EMG
    rf_5 = pickle.load(open("rf_model_5_cfs.p", "rb"))            #2 EEG
    rf_6 = pickle.load(open("rf_model_6_cfs.p", "rb"))            #1 EEG
    
    while reiz.clock.now() - tz < 29700: 
        ## Pull data from EEG, EOG, and EMG
        # data = bfr.get_data()[:,[8,9,10,22,23,24,25,26,27]]*1e6 #cap
        data = bfr.get_data()[:,indices_to_pull]*1e6
        ## Extract bandpower and relative power per epoch
        epoch_stage_features = epoch_psd(data, bfr.fs)
        
        ## Baseline and ongoing channel failure calculation comparison
        # function produces an array matching the functional channels [1,1,1,1,1,1], 
        # where 0 is dysfunctional and 1 is functional.
        channel_failure = channel_failure_test(epoch_stage_features)  
        #print(f'{channel_failure} - channel failure')
        
        # select classifier
        stage_predict = int(rf_model_select(channel_failure, epoch_stage_features, rf_1, rf_2, rf_3, rf_4, rf_5, rf_6)) 
                
        ## append stage arrays (1 = N2/3; 0 = Wake/N1/REM)
        print(f'Sleep Stage: {stage_predict}')
        reiz.marker.push('stage_predict')
        
        ## 1 for N2/SWS, 0 for all else
        if stage_predict == 2 or stage_predict == 3:
            stage_predict_binary = 1
        else:
            stage_predict_binary = 0
        
        stage_predictArrays.append(stage_predict_binary)
        print(f'{stage_predictArrays} - previously classified epochs')
            
        # pull data every ~15s
        reiz.clock.sleep(15)

        
def channel_failure_test(epochs):
    """Channel failure detection

    This function assess the stability of a given input of spectral density features,
    where the input corresponds to an array, typically consisting of a [1 x 30] array
    with spectral features for a given epoched. These are then parsed and made into a 
    ratio of line noise (49 - 51 Hz) to alpha (8 - 12 Hz) 

    Parameters
    ----------
    epochs :  numpy array of [n_epochs x n_chans*n_spectral_feats]
        Spectral density feature array.  
              
    Returns
    -------
    stage_predictArrays : list of integer values
        A list containing the status of selected channels, either if they are intact --> 1,
        or whether they have failed --> 0. They reflect the status of C3, Cz, C4, EOG_average,
        and EMG_average. For example, if all channels are functional --> [1,1,1,1,1], if all
        have failed --> [0,0,0,0,0]    
    """
    # ~39.7 microseconds computation time, before appending added...
    global channel_failure_featArray
    channel_failure_featArray = []
    channel_failureArray = []
    ## Initialize all channels as functioning -> 1
    channel_failureArray = np.ones(5)
        
    ## Baseline spectral density ratio levels
    # calculate relative alpha power for all channels
    rel_alpha = epochs[:,2::6]
    # calculate relative line noise power for all channels
    rel_ln = epochs[:,5::6]
    # calculate alpha/line ratio for all channels
    ln_alpha = rel_ln/rel_alpha
    # combine the above into a list to be able to do baseline/ongoing comparisons
    channel_failure_feat = list(ln_alpha) 
    # append into list
    channel_failure_featArray.append(channel_failure_feat[0])
    
    ## Ongoing check to see if threshold is exceeded
    # individual epoch check (change in ln_alpha ratio)
    epoch_check = channel_failure_feat[-1] > 5
    if any(epoch_check): 
        channel_failureArray[np.where(epoch_check)[0]] = 0
    else:
        # channel failure array remains the same
        channel_failureArray = channel_failureArray
    # epoch to epoch change in ln_alpha ratio, can only occur 
    # if list of channel_failure_feats contains more than one index
    if len(channel_failure_featArray) < 1:
        epoch_diff = channel_failure_featArray[-2] - channel_failure_featArray[-1] > 2                                                    
        if any(epoch_diff): 
            channel_failureArray[np.where(epoch_diff)[0]] = 0
        else:
            # channel failure array remains the same
            channel_failureArray = channel_failureArray
    # compare baseline and present epoch (ln_alpha ratio)
    baseline_check = ln_alpha[-1] - channel_failure_featArray[0] > channel_failure_featArray[0]*1.3
    if any(baseline_check):
        channel_failureArray[np.where(baseline_check)[0]] = 0
    else:
        # channel failure array remains the same
        channel_failureArray = channel_failureArray

    ## Return updated channel_failureArray
    return channel_failureArray        
        
def rf_model_select(channel_failure, epoch_stage_features, rf_1, rf_2, rf_3, rf_4, rf_5, rf_6):
    stage_predict = 0
    ## Classifier selection:
    # remove line noise variable for classification
    epoch_stage_features = np.delete(epoch_stage_features, np.arange(5, epoch_stage_features.size, 6), axis=-1)
    # Classifer to use when all channels are functional (2 EEG, 1 EOG, 1 EMG)
    if all(channel_failure==1):
        index = np.where(channel_failure[0:3] == 1)[0]
        # randomly select two EEG channels
        new_index = random.sample(list(index), k=2)
        if [0 and 1] in new_index:
            index = np.r_[0:10,15:25]
            epoch_stage_features = epoch_stage_features[:,[index]][0]
            stage_predict = rf_1.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_base_model_C3_Cz')
        elif [0 and 2] in new_index:
            index = np.r_[0:5,10:25]
            epoch_stage_features = epoch_stage_features[:,[index]][0]
            stage_predict = rf_1.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_base_model_C3_C4')
        elif [1 and 2] in new_index:
            epoch_stage_features = epoch_stage_features[:,5:25]
            stage_predict = rf_1.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_base_model_Cz_C4')
        else:
            print('No available classification streams; waiting until the next iteration of loop...')
    
    # Classifier if one EEG fails (1 EEG, 1 EMG, 1 EOG)
    elif sum(channel_failure[0:3]) == 2 and sum(channel_failure[4:5]) == 2:
        index = np.where(channel_failure[0:3] == 1)[0]
        # randomly select one functioning EEG channel
        new_index = random.choice(list(index))
        if new_index == 0:
            index = np.r_[0:5,15:20]
            epoch_stage_features2 = epoch_stage_features[:,[index]][0]
            stage_predict = rf_2.predict(epoch_stage_features2)[0]
            reiz.marker.push('rf_base2_model_C3')
        if new_index == 1:
            index = np.r_[5:10,15:20]
            epoch_stage_features2 = epoch_stage_features[:,[index]][0]
            stage_predict = rf_2.predict(epoch_stage_features2)[0]
            reiz.marker.push('rf_base2_model_Cz')
        if new_index == 2:
            epoch_stage_features2 = epoch_stage_features[:,10:25]
            stage_predict = rf_2.predict(epoch_stage_features2)[0]    
            reiz.marker.push('rf_base2_model_C4')
        else:
            print('No available classification streams; waiting until the next iteration of loop...')
    
    # Classifier with 2 EEG, 1 EOG 
    elif sum(channel_failure[0:3]) >= 2 and channel_failure[3] == 1:
        index = np.where(channel_failure[0:3] == 1)[0]
        # randomly select two EEG channels
        new_index = random.sample(list(index), k=2)
        if [0 and 1] in new_index:
            index = np.r_[0:10,15:20]
            epoch_stage_features = epoch_stage_features[:,[index]][0]
            stage_predict = rf_3.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_2EEG_EOG_model_C3_Cz')
        elif [0 and 2] in new_index:
            index = np.r_[0:5,10:20]
            epoch_stage_features = epoch_stage_features[:,[index]][0]
            stage_predict = rf_3.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_2EEG_EOG_model_C3_C4')
        elif [1 and 2] in new_index:
            epoch_stage_features = epoch_stage_features[:,5:20]
            stage_predict = rf_3.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_2EEG_EOG_model_Cz_C4')
        else:
            print('No available classification streams; waiting until the next iteration of loop...')
        
    # Classifier with 2 EEG, 1 EMG
    elif sum(channel_failure[0:3]) >= 2 and channel_failure[4] == 1:
        index = np.where(channel_failure[0:3] == 1)[0]
        # randomly select two EEG channels
        new_index = random.sample(list(index), k=2)
        if [0, 1] == new_index:
            index = np.r_[0:10,20:25]
            epoch_stage_features = epoch_stage_features[:,[index]][0]
            stage_predict = rf_4.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_2EEG_EMG_model_C3_Cz')
        elif [0, 2] == new_index:
            index = np.r_[0:5,10:15,20:25]
            epoch_stage_features = epoch_stage_features[:,[index]][0]
            stage_predict = rf_4.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_2EEG_EMG_model_C3_C4')
        elif [1, 2] == new_index:
            index = np.r_[5:15,20:25]
            epoch_stage_features = epoch_stage_features[:,[index]][0]
            stage_predict = rf_4.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_2EEG_EMG_model_Cz_C4')
        else:
            print('No available classification streams; waiting until the next iteration of loop...')
        
    # Classifier to use if bipolar channels fail (2 EEG)          
    elif all(channel_failure[3::]==0) and sum(channel_failure[0:3]) >= 2:
        index = np.where(channel_failure[0:3] == 1)[0]
        # randomly select two functioning EEG channels
        new_index = sorted(random.sample((list(index)), k=2))
        if [0, 1] == new_index:
            epoch_stage_features = epoch_stage_features[:,0:10]
            stage_predict = rf_5.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_2EEG_model_C3_Cz')
        elif [0, 2] == new_index:
            index = np.r_[0:5,10:15]
            epoch_stage_features = epoch_stage_features[:,[index]][0]
            stage_predict = rf_5.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_2EEG_model_C3_C4')
        elif [1, 2] == new_index:
            epoch_stage_features = epoch_stage_features[:,5:15]
            stage_predict = rf_5.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_2EEG_model_Cz_C4')
        else:
            print('No available classification streams; waiting until the next iteration of loop...')
    
    # Classifer to use if bipolar channels + 2 EEG channels fail (1 EEG)
    elif sum(channel_failure[0:3])==1 and sum(channel_failure[3:5])==0:
        index = np.where(channel_failure[0:3] == 1)[0]
        new_index = random.choice(list(index))
        if new_index == 0:
            epoch_stage_features = epoch_stage_features[:,0:5]
            stage_predict = rf_6.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_1EEG_model_C3')
        elif new_index == 1:
            epoch_stage_features = epoch_stage_features[:,5:10]
            stage_predict = rf_6.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_1EEG_model_Cz')
        elif new_index == 2:
            epoch_stage_features = epoch_stage_features[:,10:15]
            stage_predict = rf_6.predict(epoch_stage_features)[0]
            reiz.marker.push('rf_1EEG_model_C4')
        else:
            print('No available classification streams; waiting until the next iteration of loop...') 

    else:
        print('No available classification streams; waiting until the next iteration of loop...')
        stage_predict = 0
        # index = np.r_[0:5,10:15]
        # epoch_stage_features = epoch_stage_features[:,[index]][0]
        # stage_predict = rf_5.predict(epoch_stage_features)[0]
        # reiz.marker.push('rf_2EEG_model_C3_C4_backup')
    
    return stage_predict


def SO_detection(time_delay, volume, nepochsthresh = 4, minamp = -35, 
                 winshift_in_ms = 20, totalruntime = 12600):
    sinfo = liesl.get_streaminfos_matching(type = 'EEG')

    bfr2 = liesl.RingBuffer(sinfo[0], duration_in_ms = 30000) 
    bfr2.start()

    bfr2.await_running()
    
    filtparams_lp = signal.butter(4, 4, fs = bfr2.fs)
    #filtparams_hp = signal.butter(4, 0.5, fs = bfr2.fs, btype='highpass')
    
    n = PinkNoise(volume)

    block_auditory_stim = False
    tblock = 0
    tz = reiz.clock.now()
    while reiz.clock.now() - tz < totalruntime:
        reiz.clock.tick()                   
        if sum(stage_predictArrays[-nepochsthresh:]) >= nepochsthresh:
            #print('SO detection has been invoked!')
            if clock.now() - tblock > 2.99:
                block_auditory_stim = False
            #%% Proper channel needs to be selected based on channel failure index from classification thread
            if channel_failure[0] == 1:
                d = bfr2.get_data()[:,9]*1e6 #C3, main recording channel - TO DO: use index
                #print('SO detection with channel: C3')
            elif channel_failure[1] == 1: 
                d = bfr2.get_data()[:,8]*1e6 #Cz, alternative recording channel - TO DO: use index
                print('SO detection with channel: Cz')
            elif channel_failure[2] == 1:
                d = bfr2.get_data()[:,10]*1e6 #C4, second alternative recording channel - TO DO: use index
                print('SO detection with channel: C4')
            else:
                continue
                                                               
            #%% online SWS detection pre-processing
            d = signal.filtfilt(*filtparams_lp, d)
            #d = signal.filtfilt(*filtparams_hp, d)
            
            ## Minamp reflects deviation from the last 5 seconds of data within the 10th percentile of the signal
            minamp = min(np.percentile((d[-5* int(bfr2.fs):] - np.median(d)), 10), -35)
            #print(f'minimum amplitude value: {minamp}')
            
            ## Linear drift detection
            # Check the peak-to-peak maximum of the current epoch, if it exceeds 500 µV
            # (and -300 µV negative amplitude), reset threshold to -35 & block stimulation for 10s
            if minamp < -300 and np.ptp(d[-2*int(bfr2.fs):] - np.median(d[-2*int(bfr2.fs):])) < 500:
                minamp = -35
                reiz.clock.sleep(10)
             
            # criterion for SO occurence
            crit = min(d[int(-0.02*bfr2.fs):]) - np.median(d)
            #print(f'critical value: {crit}')
            #reiz.marker.push('crit: {}'.format(crit))
            
            if crit < minamp and block_auditory_stim == False: 
                print('target reached')
                # wait for 0ms, ~500ms, depending on Up/Downstate    
                clock.sleep(time_delay)
                # deliver tone twice with 1.075s interval
                n.play()
                clock.sleep(1.075)
                n.play()
                # blocking auditory stimulation for 3s refractory period
                block_auditory_stim = True 
                tblock = clock.now()
                
        clock.sleep_debiased(winshift_in_ms/1000)

 
class PinkNoise():
    def generate_noise(self,duration_in_s = 0.05, fs = 48000, ncols=16):
        """Generates pink noise using the Voss-McCartney algorithm.
        
        nrows: number of values to generate
        rcols: number of random sources to add
        
        returns: NumPy array
        """
        nrows = int(fs * duration_in_s)
        array = np.empty((nrows, ncols))
        array.fill(np.nan)
        array[0, :] = np.random.random(ncols)
        array[:, 0] = np.random.random(nrows)
        
        # the total number of changes is nrows
        n = nrows
        cols = np.random.geometric(0.5, n)
        cols[cols >= ncols] = 0
        rows = np.random.randint(nrows, size=n)
        array[rows, cols] = np.random.random(n)
        
        flank_samples = int(0.005*fs)
        rising = np.linspace(0,1,flank_samples)
        falling = np.linspace(1,0,flank_samples)
        plateau = np.ones(nrows-2*flank_samples)
        window = np.concatenate((rising,plateau,falling))
        df = pd.DataFrame(array)
        df.fillna(method='ffill', axis=0, inplace=True)
        total = df.sum(axis=1)
        noise = total.values
        noise /= max(noise)
        return noise*window  
    
    def open_stream(self):  
        fs = 48000
        self.stream = self.p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=fs,
                        output=True)
      
    def play(self):
        if self.reizmarker:
            reiz.marker.push('pinknoise')
        self.stream.write(self.samples)
        self.stream.stop_stream()
        self.stream.start_stream()
        
    def __init__(self, volume):
        threading.Thread.__init__(self)
        self.samples = self.generate_noise()*volume
        self.p = pyaudio.PyAudio()
        self.open_stream()
        self.reizmarker = True
        # self.play()
        # self.play()
        # self.play()
        if not reiz.marker.available():
            print('Marker Server not available!')
            self.reizmarker = False
   
