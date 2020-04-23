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
from scipy import signal
from scipy.signal import welch, resample, resample_poly
from scipy.integrate import simps
import pickle
import liesl
import yasa


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
    if old_sf > 128 and new_sf > 128: #to-do edit to allow for 100 Hz minimum fs...
        if old_sf % 100 == 0 or old_sf % 128 == 0:
            if new_sf % 100 == 0 or new_sf % 128 == 0:
                if old_sf % new_sf == 0:
                    decim = int(old_sf/new_sf)
                    data = data[::decim]
                    print(f'Downsampled data by a factor of {decim}')
                else:
                    # resample/resample_poly not integer numbers
                    raise ValueError('The ratio of the old and new sampling rate is not an integer value')
            else:
                raise ValueError('The requested sampling rate must be a multiple of 100 or 128')
        else:
            # resample/resample_poly not integer numbers
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
    raw_train.pick_types(include=(['EEG Fpz-Cz', 'EOG horizontal', 'EMG submental'])) 
    raw_train.filter(0.5, 35, picks = ['EEG Fpz-Cz', 'EOG horizontal'])
    raw_train.notch_filter(49)
    raw_train.filter(20, 45, picks = (['EMG submental']))
    
    # select channels for annotations
    mapping = {'EEG Fpz-Cz':'eeg',
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
    freqs, psd = welch(data_win, fs, nperseg=nperseg, average='median')
    
    # extract relative or absolute spectral density values for frequency bands of interest
    bp = bandpower_from_psd_ndarray(psd, freqs, bands, relative)
    return bp


def bfr_butter_filt(data, fs, order = 4, lfreq = 0.5, hfreq = 35, btype='pass'):
    ## Safety check 
    # check data type, convert to float64 if necessary
    if data.dtype != np.float64:
        data == np.asarray(data, dtype=np.float64)
    ## Construct butter filter (scipy)
    filtparams = butter(order, (lfreq, hfreq), btype=btype, fs = fs)
    # run zero phase digitial filter with butterworth parameters
    data = filtfilt(*filtparams, data, axis=0, padtype='odd')
    return data


def re_reference(data, reference='common average'):
    # common average method takes ~1.5 ms for 30s of data
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
    EEG = re_reference(data[:,[9,10,11]], reference='common average')  #C3, Cz, C4
    # EOG data selection
    EOG = data[:,[21]] 
    # combine EEG & EOG for filtering (necessary if re-referencing online)
    EEG_EOG = np.concatenate([EEG, EOG], axis=1)
    # bandpass filter data (defaults to 4th order filt, 0.5 - 35 Hz bandpass)
    EEG_EOG = bfr_butter_filt(EEG_EOG, fs)
    # notch filter data
    EEG_EOG = bfr_butter_filt(EEG_EOG, fs, lfreq = 49, hfreq = 51, btype='stop')
    # select and filter EMG 
    EMG = bfr_butter_filt(data[:,[20]], fs, lfreq = 10, hfreq = 100)
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
    stage_predictArrays = []
    tz = reiz.clock.now()
    ## Loop for 210 minutes (12600s)
    while reiz.clock.now() - tz < 12600:
    #while True:    
        # to-do: incorporate baseline channel failure calculation comparison
        dat = bfr.get_data()[:,4]*1e6 #to-do: decide which channels can be replacements, also include SW detection channel to preprocess
        ## to-do: combine the epoch_psd calculation and channel detection failure test
        # extract bandpower and relative power per epoch
        epoch_psd = epoch_stage(dat, bfr.fs)

        # check to see if channel failure occured
        channel_failure = channel_failure_test(rel_alpha/rel_ln) #to-do: integrate with new function to compute failure based on ratio
        if channel_failure = 1: #and channel is frontal, (elif) central, (elif) parietal:
            dat = bfr.get_data()[:,..]*1e6 # pick possible channel changes
            # extract bandpower/relative power again.... 
            # what if bipolar channel fails and new classifier is necessary?
            rf
        # predict sleep stage of epoch
        stage_predict = rf.predict(epoch_psd[:,])[0]
        print(stage_predict)
        # append stage arrays (1 = N2/3; 0 = Wake/N1/REM)
        #stage_predict = 1
        stage_predictArrays.append(stage_predict)
        # pull data every ~15s
        reiz.clock.sleep(15)

def channel_failure_test(d):
    global channel_failureArray #21 elements set to 1, if failed fifth position -> 0
    channel_failureArray = []
    global rel_alpha_power # see note below
    rel_alpha_power = [] # see note below 
    #baseline ratio levels
    # extract bandpower (parse into seperate function that works with classfication)
    freqs, psd = welch(d, sf=bfr.fs, nperseg=int(4 * bfr.fs), average='median')
    # calculate relative delta/theta/alpha/sigma/beta (may have to transpose psds based on bfr data shape)    
    rel_all = yasa.bandpower_from_psd_ndarray(psd, freqs, bands=[(0.5, 4, 'Delta'),(4, 8, 'Theta'),
                                                                 (8, 12, 'Alpha'),(12, 16, 'Sigma'), 
                                                                 (16, 30, 'Beta'), (49, 51, 'Line Noise')], relative=True)
    # calculate relative alpha power
    rel_alpha = rel_all[2,:] 
    # calculate relative line noise power
    rel_ln = rel_all[5,:]  
    # calculate alpha/line ratio
    ln_alpha = rel_ln/rel_alpha
    # create a list of relative alpha power values (may need to add to different function)
    rel_alpha_power.append(rel_alpha)
    # threshold(s) necessary to reflect channel failure 
    if diff(rel_alpha_power) > np.mean(rel_alpha_power)*.2 and rel_ln > .01 and ln_alpha < 5: #epoch to epoch checks
        if rel_alpha_power[-1] - rel_alpha_power[0] and : #compare baseline and present epoch
            channel_failure = 1
    channel_failureArray.append(channel_failure)
    # don't extract power here
    return channel_failureArray #rel_all, rel_alpha_power 

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
        if sum(stage_predictArrays[-10:]) > nepochsthresh: 
            if clock.now() -tblock > 2.4:
                block_auditory_stim = False
                
            #%% proper channel needs to be picked here
            d = bfr2.get_data()[:,9]*1e6 #C3, test channel is 1Hz sinusoid
            
            #%% online SWS detection pre-processing
            d = signal.filtfilt(*filtparams, d)
          
            crit = min(d[-6:]) - np.median(d)
            # reiz.marker.push('crit: {}'.format(crit))
            # change the minamp to reflect the most negative median amplitude from the last 5 seconds
            minamp = min((d[-5* bfr2.fs:]) - np.median(d), -35)
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
    
