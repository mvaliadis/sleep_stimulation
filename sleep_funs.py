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
            stagelens.append(4*np.ones(int(length)))
            
    hypnogram = np.concatenate(stagelens)

    # padding for last epoch if it doesn't last 30s
    #hypnogram = np.pad(hypnogram, (0, npts_diff), mode='edge')
    
    # sanity check - does length of hypnogram match data epoch length
    if data.shape[0] != len(hypnogram):
        raise ValueError('The length of the scaled hypnogram does not match the amount of total epochs in the data')
    
    return hypnogram 


def hjorth_mobility(x):
    return np.sqrt(np.var(np.diff(x))/np.var(x))
def hjorth_complexity(x):
    """ calculates Hjorth complexity of the input time series vector x"""
    return hjorth_mobility(np.diff(x))/hjorth_mobility(x)

def downsample_scaled(data, old_sf, new_sf):
    """Downsample function 
    
    The following function allows the user to downsample the data, so long 
    as the new sampling rate is a multiple of 100 or 128 and the ratio of 
    the old/new sampling rate is an integer number.

    Parameters
    ----------
    data : np.array
           The data.
           
    old_sf : int
             initial sampling rate.
             
    new_sf : int
             requested sampling rate.
             
    Returns
    -------
    data: np.array of shape [n_samples, chans]
        Downsampled data.
    """
    
    # first Check if we can downsample to 100 or 128 Hz
    if old_sf >= 128 and new_sf >= 128: #to-do edit to allow for 100 Hz minimum fs...
        if old_sf % 100 == 0 or old_sf % 128 == 0:
            if new_sf % 100 == 0 or new_sf % 128 == 0:
                if old_sf % new_sf == 0:
                    decim = int(old_sf/new_sf)
                    dpnts, epochs = data.shape
                    # so long as recording is less than 25 hrs, the below is valid
                    if epochs > dpnts:
                        data = np.transpose(data)
                    data = data[::decim]
                    print(f'Downsampled data by a factor of {decim}')
                else:
                    # add resample/resample_poly for non integer numbers
                    raise ValueError('The ratio of the old and new sampling rate is not an integer value')
            else:
                raise ValueError('The requested sampling rate must be a multiple of 100 or 128')
        else:
            # add resample/resample_poly for non integer numbers
            raise ValueError('The initial sampling rare is not divisible by 100 or 128!') 
    else:
        raise ValueError('The initial or requested sampling rate must both be larger than 128 Hz!')
         
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
    
#%%
## Online signal pre-processing functions
def bandpower(epochs, fs, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                 (8, 12, 'Alpha'),(12, 16, 'Sigma'), 
                                 (16, 30, 'Beta'), (49, 51, 'Line noise')], relative=True):
    """EEG absolute/relative power band feature extraction.

    This function takes a numpy array of the data & creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn. The function first computes the spectral density based on Welch's 
    method then by wrapping the bandpower_from_psd_ndarray from yasa, a package 
    written by Raphael Vallat, computes the absolute or relative power per frequency
    band of interest. 

    Parameters
    ----------
    epochs : Epochs
        The data.
        
    fs : sampling rate 
    
    see 'yasa.bandpower_from_psd_ndarray' for remaining parameters and documentation

    Returns
    -------
    bp : numpy array of shape [n_bands, chans]
        relative/absolute power of data.
    """
    # Define window length sufficiently long encompassing at least two 2 cycles of the lowest frequency of interest
    nperseg = (2 / bands[0][0]) * fs

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(epochs, fs, nperseg=nperseg, average='median')
    
    # extract relative or absolute spectral density values for frequency bands of interest
    bp = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative)
    return bp


def bfr_butter_filt(data, fs, order = 4, lfreq = 0.5, hfreq = 35, btype='pass'):
    ## Safety check 
    # check data type, convert to float64 if necessary 
    #if data.dtype != np.float64:
    #    data == np.asarray(data, dtype=np.float64)
   
    ## Construct butter filter (scipy)
    filtparams = butter(order, (lfreq, hfreq), btype=btype, fs = fs)
    # run zero phase digitial filter with butterworth parameters, 
    # but first confirm order of dims
    chans, dpnts = data.shape
    if chans > dpnts:
        data = filtfilt(*filtparams, data, axis=0, padtype='odd')
    else:
        data = filtfilt(*filtparams, data, axis=-1, padtype='odd')
    return data


def re_reference(data, reference='common average'):
    # common average method takes ~1.5 ms for 30s of data
    # to-do: use loop method
    if reference == 'common average':
        mean_vec = np.mean(data, axis = 1)
        data -= np.tile(mean_vec, (np.shape(data)[1],1)).T
    # mastoid average method takes ~115 micro s for 30s of data
    elif reference == 'mastoids':
        # not true yet, please avoid selecting this option
        data = data[:,[9,10,11]] - (data[:, [18]] + data[:,[20]])/2
    else:
        raise ValueError('Please select a valid reference!')

        
def epoch_stage(data, fs):
    ## Data selection and filtering
    # select and re-reference EEG signal to the common average
    # to-do: take mean of all channels in rereference function
    EEG = re_reference(data[:,[0,1,2]], reference='common average')  #Cz, C3, C4
    # EOG data selection
    EOG = data[:,[3]] 
    # combine EEG & EOG for filtering (necessary if re-referencing online)
    EEG_EOG = np.concatenate([EEG, EOG], axis=1)
    # bandpass filter data (defaults to 4th order filt, 0.5 - 35 Hz bandpass)
    EEG_EOG = bfr_butter_filt(EEG_EOG, fs)
    # notch filter data
    EEG_EOG = bfr_butter_filt(EEG_EOG, fs, lfreq = 49, hfreq = 51, btype='stop')
    # select and filter EMG 
    EMG = bfr_butter_filt(data[:,[4]], fs, lfreq = 10, hfreq = 100)
    # combine all data streams back into one array
    data = np.transpose(np.concatenate([EEG_EOG, EMG], axis=1))#*1e6  
    ## Safety checks
    assert data.ndim == 2, 'Data must be of shape (nchan, n_samples).'
    nchan, npts = data.shape
    if npts < nchan:
        data = np.transpose(data) 
        nchan, npts = data.shape
    ## Compute bandpower of epoch
    win = bandpower(data, fs)
    ## Reshape data for classifier (epochs x (nchans*bands))
    win = win.reshape(1, nchan*6, order='F')
    return win  
 
    
def sleep_staging(bfr):
    global stage_predictArrays
    global channel_failure
    stage_predictArrays = []
    tz = reiz.clock.now()
    ## Loop for 210 minutes (12600s)
    while reiz.clock.now() - tz < 12600:
    #while True:  
        ## Pull data from EEG, EOG, and EMG
        data = bfr.get_data()[:,[9,10,11,20,21]]*1e6 
        
        ## Extract bandpower and relative power per epoch
        epoch_psd = epoch_stage(data, bfr.fs)
        
        ## Baseline and ongoing channel failure calculation comparison
        # function produces an array matching the functional channels [1,1,1,1,1], 
        # where 0 is dysfunctional and 1 is functional.
        channel_failure = channel_failure_test(epoch_psd)         
        
        ## Classifier selection:
        # Classifer to use when all channels are functional (1 EEG, 1 EOG, 1 EMG)
        if sum(channel_failure) == 5: 
            # predict sleep stage of epoch
            index = np.r_[0:5,18:29] #bands for: ch0, ch3, ch4
            epoch_psd = epoch_psd[:,index][0]
            stage_predict = rf.predict(epoch_psd)[0] 
        # Classifer to use if bipolar channels fail (2 EEG)
        # to-do: edit to just select two of three functioning EEG channels 
        elif sum(channel_failure[[3,4]]) <= 1:
            if channel_failure[0] == 1 and channel_failure[1] == 1:
                epoch_psd = epoch_psd[:,0:11][0] #bands for: ch0, ch1
                stage_predict = rf_2EEG.predict(epoch_psd)[0]
            elif channel_failure[0] == 1 and channel_failure[2] == 1:
                index = np.r_[0:5,12:17] #bands for: ch0, ch2
                epoch_psd = epoch_psd[:,index][0]
                stage_predict = rf_2EEG.predict(epoch_psd)[0]
            elif channel_failure[1] == 1 and channel_failure[2] == 1:
                epoch_psd = epoch_psd[:,[6:17]][0] #bands for: ch0, ch2
                stage_predict = rf_2EEG.predict(epoch_psd)[0]
        # Classifer to use if EMG fails (1 EEG, 1 EOG)
        # WARNING: won't be reached 
        elif channel_failure[4] == 0:
            if channel_failure[3] == 1 and channel_failure[0] == 1:
                index = np.r_[18:23,0:5] #bands for: ch3, ch0
                epoch_psd = epoch_psd[:,index][0] 
                stage_predict = rf_EEG_EOG.predict(epoch_psd)[0]
            elif channel_failure[3] == 1 and channel_failure[1] == 1:
                index = np.r_[18:23,6:11] #bands for: ch3, ch1
                epoch_psd = epoch_psd[:,index][0]
                stage_predict = rf_EEG_EOG.predict(epoch_psd)[0]
            elif channel_failure[3] == 1 and channel_failure[2] == 1:
                index = np.r_[18:23,12:17] #bands for: ch3, ch2
                epoch_psd = epoch_psd[:,index][0] 
                stage_predict = rf_EEG_EOG.predict(epoch_psd)[0]
        # Classifer to use if bipolar channels + 2 EEG channels fail (1 EEG)
        elif sum(channel_failure[[0,1,2]]) == 1 and sum(channel_failure[[3,4]]) == 0:
            if channel_failure[0] == 1:
                stage_predict = rf_1EEG.predict(epoch_psd[:,0:5)[0]
            elif channel_failure[1] == 1:
                stage_predict = rf_1EEG.predict(epoch_psd[:,6:11])[0]
            elif channel_failure[2] == 2:
                stage_predict = rf_1EEG.predict(epoch_psd[:,6:11])[0]

        ## append stage arrays (1 = N2/3; 0 = Wake/N1/REM)
        print(stage_predict)
        #stage_predict = 1 # for testing
        stage_predictArrays.append(stage_predict)
        # pull data every ~15s
        reiz.clock.sleep(15)

        
def channel_failure_test(epochs):
    # ~24 microseconds computation time
    ## Initialize all channels as functioning -> 1
    channel_failureArray = np.ones(5)
    
    ## Baseline spectral density ratio levels
    # calculate relative alpha power
    rel_alpha = epochs[:,2] 
    # calculate relative line noise power
    rel_ln = epochs[:,5]  
    # calculate alpha/line ratio
    ln_alpha = rel_ln/rel_alpha

    ## Ongoing check to see if threshold is exceeded
    # epoch to epoch check
    # to-do: make into loop                                                     
    if ln_alpha[-1] < 5: 
        # compare baseline and present epoch
        if ln_alpha[-1] - ln_alpha[0] > np.mean(ln_alpha)*1.3:
            # if EEG channel 1 fails, then switch to EEG channel 2
            channel_failureArray[0] = 0
            ln_alpha = epochs[:,8]/epochs[:,11]
            if ln_alpha[-1] < 5:
                if ln_alpha[-1] - ln_alpha[0] > np.mean(ln_alpha)*1.3:
                    # if EEG channel 2 fails, then switch to EEG channel 3
                    channel_failureArray[1] = 0
                    ln_alpha = epochs[:,14]/epochs[:,17]
                    if ln_alpha[-1] < 5:
                        if ln_alpha[-1] - ln_alpha[0] > np.mean(ln_alpha)*1.3:
                            # if EEG channel 3 fails, then.....
                            channel_failureArray[2] = 0
    else:
        channel_failureArray = channel_failureArray
        
    ## Check EOG and EMG channels for failure 
    # EOG check 
    ln_alpha_EOG = epochs[:,20] /epochs[:,23]
    if ln_alpha_EOG[-1] < 5:
        if ln_alpha_EOG[-1] - ln_alpha_EOG[0] > np.mean(ln_alpha_EOG)*1.3:
            channel_failureArray[3] = 0
    # EMG check    
    ln_alpha_EMG = epochs[:,26] /epochs[:,29]
    if ln_alpha_EMG[-1] < 5:
        if ln_alpha_EMG[-1] - ln_alpha_EMG[0] > np.mean(ln_alpha_EMG)*1.3:
            channel_failureArray[4] = 0
            
    ## Return updated channel_failureArray  
    return channel_failureArray    


def SO_detection(nepochsthresh = 0, minamp = -35, 
                 winshift_in_ms = 20, totalruntime = 12600,
                 time_delay = 500, volume = 1  ):
    sinfo = liesl.get_streaminfos_matching(type = 'EEG')

    bfr2 = liesl.RingBuffer(sinfo[0], duration_in_ms = 30000) 
    bfr2.start()

    bfr2.await_running()
    
    filtparams = signal.butter(4, 4, fs = bfr2.fs)

    n = PinkNoise(volume)

    block_auditory_stim = False
    tblock = 0
    tz = reiz.clock.now()
    while reiz.clock.now() - tz < totalruntime:
        reiz.clock.tick()
        if sum(stage_predictArrays[-5:]) > nepochsthresh: 
            if clock.now() - tblock > 2.4:
                block_auditory_stim = False
                
            #%% Proper channel needs to be selected based on channel failure index from classification thread
            if channel_failure[0] == 1:
                d = bfr2.get_data()[:,9]*1e6 #C3, main recording channel
            elif channel_failure[1] == 1: 
                d = bfr2.get_data()[:,10]*1e6 #Cz, alternative recording channel
            else:
                continue
                                                          
            #%% online SWS detection pre-processing
            d = signal.filtfilt(*filtparams, d)
          
            #
            crit = min(d[-6:]) - np.median(d)
            # reiz.marker.push('crit: {}'.format(crit))
            
            ## Change the minamp to reflect the most negative median amplitude from the last 5 seconds
            minamp = min(min(d[-5* bfr2.fs:]) - np.median(d), -35)
            
            ## Linear drift detection
            # Check the peak-to-peak maximum of the current epoch, if it exceeds 500 µV
            # (and -300 µV negative amplitude), reset threshold to -35 & block stimulation for 10s
            if minamp < -300 and np.ptp(d[-2*bfr2.fs:] - np.median(d[-2*bfr2.fs:])) < 500:
                minamp = -35
                block_auditory_stim = True 
                tblock = clock.now()
            if clock.now() - tblock > 10:
                block_auditory_stim = False 
            
                if crit < minamp and block_auditory_stim == False: 
                    # wait for 0ms, 500ms, depending on Up/Downstate    
                    clock.sleep(time_delay)
                    # deliver tone twice with 1.075s interval
                    n.play()
                    clock.sleep(1.075)
                    n.play()
                    # blocking auditory stimulation for 2.5s
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

#        stream.close()
#        return t
#        self.p.terminate()
        
    def __init__(self, volume):
        threading.Thread.__init__(self)
        self.samples = self.generate_noise()*volume
        self.p = pyaudio.PyAudio()
        self.open_stream()
        self.reizmarker = True
        self.play()
        self.play()
        self.play()
        if not reiz.marker.available():
            print('Marker Server not available!')
            self.reizmarker = False
    
