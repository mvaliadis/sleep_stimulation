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
from scipy.signal import welch

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
    
