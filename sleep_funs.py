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
import pickle
import liesl
import yasa


def hjorth_mobility(x):
    return np.sqrt(np.var(np.diff(x))/np.var(x))
def hjorth_complexity(x):
    """ calculates Hjorth complexity of the input time series vector x"""
    return hjorth_mobility(np.diff(x))/hjorth_mobility(x)

def downsample_scaled(data, old_sf, new_sf):
    ## The following function allows the user to downsample the data, so long as the new sampling rate is a multiple of 100 or 128
    ## and the ratio of the old/new sampling rate is an integer number.
    
    # Check if we can downsample to 100 or 128 Hz
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
        >>> a = np.array([20, 29, -43, -10, 16, 37, 45, -36, -29])
        >>> thresholdcrossings(a)
            array([1, 2, 6, 7], dtype=int64)
    """
    pos = x > threshold
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]


def process_raw_EDF(fname):
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

def bandpower(epochs, fs):
    """EEG relative power band feature extraction.

    This function takes a numpy array of the data & creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4],
                  "theta": [4, 8],
                  "alpha": [8, 11],
                  "sigma": [11, 16],
                  "beta": [16, 30]}

    freqs,psds = welch(epochs, fs = fs)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

def epoch_stage(dat, fs):
    data = np.float64(dat)
    times = np.arange(len(data)) / fs
    ## select and filter EEG, EOG
    # re-reference EEG (Cz) signal to the average of mastoids
    # EEG = data[:,[15]] - (data[:,[12]] + data[:,[18]])/2
    EEG = data[:,[10]]
    # EOG data selection
    EOG = data[:,[21]] 
    # combine EEG & EOG for filtering
    EEG_EOG = np.transpose(np.concatenate([EEG, EOG], axis=1))
    EEG_EOG = signal.filtfilt(*signal.butter(4, (0.5, 35), fs = fs, btype = 'pass'), EEG_EOG)
    EEG_EOG = signal.filtfilt(*signal.butter(4, (49, 51), fs = fs, btype = 'stop'), EEG_EOG)
    # select and filter EMG 
    EMG = np.transpose(data[:,[20]])  
    EMG = signal.filtfilt(*signal.butter(4, (10, 100), fs = fs, btype = 'pass'), EMG)
    # combine all data streams back into one array
    data = np.concatenate([EEG_EOG, EMG])*1e6
    # partition data into a manner readable by bandpower calculation (epochs x channels X time points)
    data_win = np.expand_dims(data, axis=0) #adds requisite singleton dimension for bandpower calculation
    # compute bandpower of epoch
    win = bandpower(data_win, fs)
    return win

def sleep_staging(bfr):
    rf = pickle.load(open("rf_model.p", "rb"))
    global stage_predictArrays
    stage_predictArrays = []
    tz = reiz.clock.now()
    #loop for 210 minutes (12600s)
    while reiz.clock.now() - tz < 12600:
    #while True:    
        dat = bfr.get_data()
        # calcule PSD per epoch for delta, theta, alpha, sigma, & beta
        epoch_psd = epoch_stage(dat, bfr.fs)
        # predict sleep stage of epoch
        stage_predict = rf.predict(epoch_psd)[0]
        print(stage_predict)
        # append stage arrays (1 = N2/3; 0 = Wake/N1/REM)
        stage_predict = 1
        stage_predictArrays.append(stage_predict)
        # pull data every ~15s
        reiz.clock.sleep(15)


def SO_detection(nepochsthresh = 0, minamp = -35, 
                 winshift_in_ms = 20, totalruntime = 12600,
                 time_delay = 0, volume = 1  ):
    sinfo = liesl.get_streaminfos_matching(type = 'EEG')

    bfr2 = liesl.RingBuffer(sinfo[0], duration_in_ms = 30000) 
    bfr2.start()

    bfr2.await_running()
    
    filtparams = signal.butter(4, 4, fs = bfr2.fs)

    n = PinkNoise(volume)
    
    #baseline levels
    d = bfr2.get_data()[:,9]*1e6 #C3, test channel is 1Hz sinusoid
    #online SWS detection pre-processing
    d = signal.filtfilt(*filtparams, d)
    # extract bandpower 
    freqs, psd = welch(d, sf=bfr2.fs, nperseg=int(4 * bfr2.fs), average='median')
    # calculate relative alpha power
    rel_alpha = yasa.bandpower_from_psd_ndarray(psd, freqs, bands=[(8, 12, 'Alpha')])
    # calculate relative line noise power
    rel_ln = yasa.bandpower_from_psd_ndarray(psd, freqs, bands=[(49, 51, 'LineNoise')])
    # calculate alpha/line ratio
    baseline_ln_alpha = rel_ln/alpha_ln

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
            # extract bandpower 
            freqs, psd = welch(d, sf=bfr2.fs, nperseg=int(4 * bfr2.fs), average='median')
            # calculate relative alpha power
            rel_alpha = yasa.bandpower_from_psd_ndarray(psd, freqs, bands=[(8, 12, 'Alpha')])
            # calculate relative line noise power
            rel_ln = yasa.bandpower_from_psd_ndarray(psd, freqs, bands=[(49, 51, 'LineNoise')])
            # calculate alpha/line ratio
            ln_alpha = rel_ln/alpha_ln
          
            crit = min(d[-6:]) - np.median(d)
            # reiz.marker.push('crit: {}'.format(crit))
            if crit < minamp and block_auditory_stim == False: 
                # change the minamp to reflect the most negative median amplitude from the last 5 seconds
                if np.median(d[-5* bfr2.fs:]) < minamp:
                    minamp = np.median(d[-5* bfr2.fs:])
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
    
