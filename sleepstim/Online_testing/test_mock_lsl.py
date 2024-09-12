#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:09:17 2023

@author: administrator
"""

import time
import numpy as np
from numpy.polynomial.legendre import legval
from scipy.linalg import pinv
from matplotlib import pyplot as plt
from scipy import signal
import scipy.stats as stats
from scipy.signal import welch
from scipy.stats import norm
from numba import jit, prange
import mne
import pandas as pd
import yasa
from sklearn.ensemble import IsolationForest
import scipy 

from matplotlib import pyplot as plt
#from mne_lsl.datasets import sample
from mne_lsl.lsl import local_clock
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
from mne_lsl.stream_viewer import StreamViewer
from mne_lsl.lsl import StreamInlet, resolve_streams

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

def surface_laplacian_rt(data, trans_csd):
    #epochs = inst._data
    epochs = np.expand_dims(data.T, 0)
    for epo in epochs:
        csd_data = np.dot(trans_csd, epo)
    
    return csd_data

def get_unit_sphere_positions(info):
    
    picks = mne.pick_types(info, meg=False, eeg=True, exclude=[])
    radius, origin_head, origin_device = mne.bem.fit_sphere_to_headshape(info)
    x, y, z = origin_head - origin_device
    sphere = (x, y, z, radius)
    sphere = np.array(sphere, float)
    x, y, z, radius = sphere
    
    pos = np.array([info['chs'][pick]['loc'][:3] for pick in picks])
    pos -= (x, y, z)
    
    # Project onto a unit sphere:
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
       
    return pos
   
class RealTimeSplineInterpolator:
    def __init__(self, all_pos, alpha=1e-5, stiffness=4, n_legendre_terms=50):
        self.all_pos = all_pos  # positions are assumed to be on the unit sphere
        self.alpha = alpha
        self.stiffness = stiffness
        self.n_legendre_terms = n_legendre_terms
        self.precomputed_matrices = self.precompute_interpolation_matrices()

    def _calc_g(self, cosang):
        """Calculate spherical spline g function between points on a sphere."""
        factors = [
            (2 * n + 1) / (n ** self.stiffness * (n + 1) ** self.stiffness * 4 * np.pi)
            for n in range(1, self.n_legendre_terms + 1)
        ]
        return np.polynomial.legendre.legval(cosang, [0] + factors)

    def precompute_interpolation_matrices(self):
        """Precompute interpolation matrices for each channel."""
        matrices = {}
        for i in range(len(self.all_pos)):
            # Consider each channel as bad and compute the interpolation matrix
            matrices[i] = self.compute_interpolation_matrix(i)
        return matrices

    def compute_interpolation_matrix(self, bad_channel_idx):
        """Compute the interpolation matrix for the bad channel."""
        good_pos = np.delete(self.all_pos, bad_channel_idx, axis=0)
        bad_pos = self.all_pos[bad_channel_idx, np.newaxis]

        cosang_from = good_pos.dot(good_pos.T)
        cosang_to_from = bad_pos.dot(good_pos.T)
        
        G_from = self._calc_g(cosang_from)
        G_to_from = self._calc_g(cosang_to_from)

        if self.alpha is not None:
            G_from.flat[::G_from.shape[0] + 1] += self.alpha

        C = np.vstack([np.hstack([G_from, np.ones((G_from.shape[0], 1))]),
                       np.hstack([np.ones((1, G_from.shape[0])), [[0]]])])
        C_inv = pinv(C)

        interpolation_matrix = np.dot(G_to_from, C_inv[:-1, :-1]) - C_inv[-1, :-1]
        return interpolation_matrix
    
    def interpolate(self, data, bad_channel_indices):
        """Interpolate data for multiple bad channels."""
        for bad_channel_idx in bad_channel_indices:
            if bad_channel_idx not in self.precomputed_matrices:
                raise ValueError(f"No precomputed matrix for channel {bad_channel_idx}")
    
            # Get the good data (excluding the current bad channel)
            good_data = np.delete(data, bad_channel_idx, axis=0)
            
            # Get the interpolation matrix for the current bad channel
            interpolation_matrix = self.precomputed_matrices[bad_channel_idx]
            
            # Interpolate the bad channel
            interpolated_data = interpolation_matrix.dot(good_data)
            
            # Insert the interpolated data back into the original data
            data[bad_channel_idx, :] = interpolated_data.squeeze()  # Ensure it has correct shape
        return data

@jit(nopython=True, parallel=True)
def calculate_metrics(eeg):
    num_channels = eeg.shape[0]
    channel_metrics = np.zeros((num_channels, 3))
 
    # Note the use of prange here instead of range
    for i in prange(num_channels):
        channel = eeg[i, :]
        channel_metrics[i, 0] = np.nanstd(channel)
        channel_metrics[i, 1] = np.nanmean(channel)
        channel_metrics[i, 2] = len(np.where(np.diff(np.sign(channel)))[0])

    return channel_metrics

def outlier_detection(channel_metrics):
    ilf = IsolationForest(
        contamination="auto", max_samples="auto", verbose=0, random_state=42
    )
    good = ilf.fit_predict(channel_metrics)
    
    return np.where(good == -1)[0]
                
def fast_eeg_badchannels(eeg, bad_threshold=2/3, distance_threshold=0.99):
    """
    Find bad channels in a more efficient manner by focusing on the fastest metrics.

    Parameters:
    - eeg : np.ndarray, the EEG data (channels x time points)
    - bad_threshold : float, the proportion criteria for a bad channel
    - distance_threshold : float, the z-score threshold for outlier detection

    Returns:
    - bads : list, indices of bad channels
    """
    # Check input type
    if not isinstance(eeg, np.ndarray):
        raise ValueError("Input 'eeg' must be a numpy ndarray.")

    # Calculate metrics
    channel_metrics = calculate_metrics(eeg)
    
    # Create dataframe channel metrics df
    #channel_metrics = pd.DataFrame(channel_metrics, columns=['Mean', 'STD', 'NZC'], 
                                  # index=stream.ch_names)
                                    
    # Isolation forest bad channels -> too slow                                    
    #bad_channels_if = outlier_detection(channel_metrics)

    # Standardization (z-score calculation)
    means = np.mean(channel_metrics, axis=0)
    stds = np.std(channel_metrics, axis=0, ddof=1)  # ddof=1 to calculate the sample standard deviation
    z = np.abs((channel_metrics - means) / stds)  # broadcasting

    # Outlier detection based on z-scores
    outlier_counts = np.sum(z > stats.norm.ppf(distance_threshold), axis=1)
    bad_channels = np.where(outlier_counts >= np.ceil(bad_threshold * z.shape[1]))[0]

    return list(bad_channels)

def center_distances(coordinates, center, ch_names):
    """
    Calculate the distances of electrodes from a specified center electrode 
    and return the indices and names of the closest electrodes.

    Parameters
    ----------
    coordinates : ndarray
        3D coordinates (x, y, z) on the unit sphere for each electrode in the montage.
    center : str
        Name of the center electrode.
    ch_names : list
        List of channel names.

    Returns
    -------
    indices : ndarray
        Indices of channels with the least distance from the center electrode.
    channels : ndarray
        Channel names corresponding to the indices.
    """
    if center not in ch_names:
        raise ValueError(f"Center electrode {center} not found in channel names.")

    center_idx = ch_names.index(center)

    # Calculate cosine distances
    cos_dist = np.sum((coordinates - coordinates[center_idx]) ** 2, axis=1) / 2
    indices = np.argsort(cos_dist)

    # Select the closest 5 electrodes (including the center electrode)
    closest_indices = indices[:5]
    closest_channels = np.array(ch_names)[closest_indices]

    return closest_indices, closest_channels

#%%
## Test speed of func
def fun(stream, times, reference='csd'):
    data, ts = stream.get_data()
    buffer = data.T #np.vstack((buffer, data))
    times.extend(ts)
    
    # Keep only the last 'extraction_window' samples in the buffer
    if len(buffer) > extraction_window:
        buffer = buffer[-extraction_window:, :]
        times = times[-extraction_window:]
    
    ## Process the buffered data
    # Apply filter
    filtered_data = signal.filtfilt(b, a, buffer, axis=0)
    
    # Median correction
    filtered_data -= np.median(filtered_data, axis=0)
    
    # Detect bad channels
    bad_channel_idx = fast_eeg_badchannels(
        filtered_data.T,
        bad_threshold=2/3,
        distance_threshold=0.99,
        )
    
    if len(bad_channel_idx) > 0:
        # Apply spline interpolation
        filtered_data = interpolator.interpolate(filtered_data.T, bad_channel_idx).T

    # Re-reference data 
    if reference == 'mastoids':
        filtered_data_final = re_reference(filtered_data, stream.ch_names, 
                                           reference='mastoids')
    elif reference == 'csd':
        filtered_data_final = re_reference(data=filtered_data/1e3, ch_names=stream.ch_names,
                                           trans_csd=trans_csd, sf=fs, reference='csd').T
     
    # Extract target signal (C3)
    chan_data = filtered_data_final[:, stream.ch_names.index('C3')]
    
    return chan_data
    
def local_SO_detection(trans_csd, surface_laplacian = True, stim_intervall=2.99):

    # Get data
    data = bfr2.get_data()[:,indices_to_pull]*1e6
                               
    # Apply filter
    filtered_data = signal.filtfilt(b, a, data, axis=0)

    # Median correction
    filtered_data -= np.median(filtered_data, axis=0)
                                  
    # Surface Laplacian
    if surface_laplacian == True:
        sl_data = re_reference(data=filtered_data/1e3, ch_names=chanlabels[0:64],
                               trans_csd=trans_csd, sf=bfr2.fs, reference='csd').T
    else:
        sl_data = re_reference(data=filtered_data, ch_names=chanlabels[0:64],
                               sf=bfr2.fs, reference='mastoids').T
        
    # Extract target channel
    chan_data = sl_data[:, chanlabels[0:64].index('C3')]                                              
   
    # Thresholding
    prom_mean = 220
    #prominence threshold is determined (adaptive mean+std)
   
    _, properties = scipy.signal.find_peaks(chan_data,
                                            prominence=(None,None), 
                                            distance=int(0.25*bfr2.fs))
    prominence = (properties['prominences'].mean()+properties['prominences'].std())*0.4
    peak_up = max(chan_data[int(-0.02*bfr2.fs):])
    peak_down = min(chan_data[int(-0.4*bfr2.fs):])
    
    # Linear drift detection -> peak to peak exceeds 500 mV/mm^2
    if np.ptp(chan_data[-2*int(bfr2.fs):]) > 500:
        # reset threshold to prom_mean
        prominence = prom_mean
        block_auditory_stim = True
                           
    # Stimulate if the max. value of last 0.02s minus the min. value of last 400ms
    # is bigger than the prominence threshold and if there has been a zerocrossing inbetween                       
    if (peak_up - peak_down) > prominence and block_auditory_stim == False and (np.sign(peak_up) - np.sign(peak_down)) != 0:
        if random.choice([True, False]):
            # Real stimulation
            reiz.marker.push('nmes_trigger_send')
        else:
            # Sham stimulation
            reiz.marker.push('nmes_sham_send')
   

#%%
# 1. Load pre-recorded dataset
#fname = '/media/administrator/data/Study_2_data/mne_lsl_data/test-raw.fif'
# fname = '/media/administrator/data/Study_2_data/NIDRA/CLNMES/PaJa_1-test.fif'
fname = '/media/administrator/Sleep_Data/Pilot/Processed/PaJa_1-raw.fif'
#fname = sample.data_path() / "sample-ant-raw.fif"

# 2. Load csd matrix
csd_file_path = '/home/administrator/sleep_stimulation/sleepstim/Analysis/CSD_matrix.npy'
trans_csd = np.load(csd_file_path)

# 3. Initialize interpolor object
mne_info = mne.io.read_raw(fname, preload=False).info
all_pos = get_unit_sphere_positions(mne_info)
interpolator = RealTimeSplineInterpolator(all_pos)

#%%
# 4. Initialize LSL player/buffer
player = Player(fname)
player.start()
stream = Stream(bufsize=30, source_id='MNE-LSL')  # 30 seconds of buffer
stream.connect(acquisition_delay=0.2)

# 5. Initialize variables
fs = stream.info['sfreq']  # Sampling frequency
ch_names = stream.info.ch_names 
extraction_window = int(30 * fs)  # e.g., last 30 seconds of data
slide_interval = int(2 * fs)  # e.g., move window every 2 seconds
b, a = signal.butter(2, 4, fs=fs, btype='lowpass')  # Low-pass filter
reference = 'csd' #'mastoids'  
block_auditory_stim = False 
centered = False
target_chan = 'C3'
roi_idx, roi = center_distances(all_pos, center=target_chan, ch_names=ch_names)

# Result storage
crit_reconstruct_stim, crit_reconstruct_sham = [], []
time_reconstruct_stim, time_reconstruct_sham = [], []

# stream.info
# stream.pick(["Fz", "Cz", "Oz"])
# stream.info

#%%
## Main loop
t0 = local_clock()
buffer = []
buffer_lm = []
times = []

# Sleep once in the beginning to allow stream to fill up
time.sleep(extraction_window / fs)
while len(crit_reconstruct_stim) <= 100:
       
    # Make sure stimulation is not blocked
    block_auditory_stim = False 
    
    # Check for new data
    n_samples = stream.n_new_samples
    
    # Extract data 
    data, ts = stream.get_data()
    ts = ts - t0
    data = data.T*1e6 
    
    data_window = data[-extraction_window:, :]
    ts_new = ts[-extraction_window:]
    
    ## Process the buffered data
    # Apply filter
    filtered_data = signal.filtfilt(b, a, data_window, axis=0)
    
    # Median correction
    filtered_data -= np.median(filtered_data, axis=0)
    
    # Detect bad channels
    # bad_channel_idx = fast_eeg_badchannels(
    #     filtered_data.T,
    #     bad_threshold=2/3,
    #     distance_threshold=0.99,
    #     )
    # print(f'Bad channel detected: {np.asarray(ch_names)[bad_channel_idx]}')
    
    # if len(bad_channel_idx) > 0:
    #     # Apply spline interpolation
    #     filtered_data = interpolator.interpolate(filtered_data.T, bad_channel_idx).T
    
    # Re-reference data 
    if reference == 'mastoids':
        filtered_data_final = re_reference(filtered_data, stream.ch_names, 
                                           reference='mastoids')
    elif reference == 'csd':
        filtered_data_final = re_reference(data=filtered_data/1e3, ch_names=stream.ch_names,
                                           trans_csd=trans_csd, sf=fs, reference='csd').T
     
    # run twice
    filtered_data_final_lm = re_reference(filtered_data, stream.ch_names, 
                                          reference='mastoids')
    
    # Extract target signal 
    if centered:
        chan_data = filtered_data_final[:, roi_idx].mean(1)
    else:
        chan_data = filtered_data_final[:, stream.ch_names.index(target_chan)]
    
    # Plot C3
    #plt.plot(ts_new - ts_new[0], chan_data)
    
    # Adaptive prominence thresholding (mean+std)
    prom_mean = 220
    _, properties = scipy.signal.find_peaks(chan_data,
                                            prominence=(None,None), 
                                            distance=int(0.25*fs))
    
    prominence = (properties['prominences'].mean()+properties['prominences'].std())*1
    peak_up = max(chan_data[int(-0.02*fs):])
    #print(peak_up)
    peak_down = min(chan_data[int(-0.4*fs):])
    
    # Linear drift detection -> peak to peak exceeds 500 mV/mm^2
    if np.ptp(chan_data[-2*int(fs):]) > 500:
        # reset threshold to prom_mean
        prominence = prom_mean
        block_auditory_stim = True 
        
    # Stimulate if the max. value of last 0.02s minus the min. value of last 400ms
    # is bigger than the prominence threshold and if there has been a zerocrossing inbetween
    if (peak_up - peak_down) > prominence and block_auditory_stim == False and (np.sign(peak_up) - np.sign(peak_down)) != 0:
        if np.random.choice([True, False]):
            # Real stimulation
            time_reconstruct_stim.append(ts_new[int(-0.01*fs):][0])
            crit_reconstruct_stim.append(max(chan_data[int(-0.01*fs):]))   
            
            print(f'Stim idx : {len(crit_reconstruct_stim)}') 
            time.sleep(slide_interval / fs)   

        else:
            # Sham stimulation
            time_reconstruct_sham.append(ts_new[int(-0.01*fs):][0])
            crit_reconstruct_sham.append(max(chan_data[int(-0.01*fs):]))   
            
            print(f'Stim idx : {len(crit_reconstruct_sham)}') 
            time.sleep(slide_interval / fs)               

        
        # Only append when criteria reached
        times.extend(ts_new)
        buffer.append(np.expand_dims(filtered_data_final, 1))
        buffer_lm.append(np.expand_dims(filtered_data_final_lm, 1))
                   
        #plt.figure()
        #plt.plot(ts_new - ts_new[0], chan_data)
        #plt.vlines(ts_new[-1] - ts_new[0], ymin=-100, ymax=100, linestyle='--') 
        

    # Slide the window forward
    time.sleep(0.2)
    
    # Need to add sleep timer to avoid stimulating for at least 3 seconds
    # time.sleep(3)

# Plotting
#viewer = StreamViewer()
#viewer.start()

# Terminate the LSL stream and player
stream.disconnect()
player.stop()

#%%
if reference=='mastoids':
    epochs = np.concatenate(buffer, axis=1)
else:
    epochs = np.concatenate(buffer_lm, axis=1)
    
#epoch_times = np.concatenate([times])

info = mne.create_info(ch_names, fs, ch_types='eeg')
mne_epochs = mne.EpochsArray(data=np.swapaxes(epochs[:int(fs*30),:,:].T, 0, 1)/1e6, 
                             info=info, tmin=-29.98)
mne_epochs.set_montage(mne.channels.make_standard_montage('standard_1005'))
mne_epochs.info['bads'] = ['Oz']
mne_epochs.interpolate_bads()
# import neurokit2 as nk 
# bads, info = nk.eeg_badchannels(mne_epochs[0].get_data().T)
#mne_epochs.set_eeg_reference(['M1','M2'])
mne_epochs.filter(0.5, None)
mne_epochs.average().plot_joint(**dict(times=[-.75, -.5, -.25, -.1, 0, 0.01]))
mne_epochs_csd = mne.preprocessing.compute_current_source_density(mne_epochs)
mne_epochs_csd.average().plot_joint(**dict(times=[-.75, -.5, -.25, -.1, 0, 0.01]))
mne_epochs_csd.copy().crop(-2, 0.01).average().plot_joint(**dict(times=[-.75, -.5, -.25, -.1, 0, 0.01]))    

#%%
#
plt.figure()
plt.plot(mne_epochs.copy().crop(-2, 0.01).times, 
         mne_epochs.copy().crop(-2, 0.01).average().get_data()[roi_idx,:].T*1e6)

#%%
## Analyze data with respect to recorded
data=mne.io.Raw(fname, preload=True)
data.info['bads'] = ['Oz']
data.interpolate_bads()
data.set_eeg_reference(['M1','M2'])
data.filter(0.5, 4)
data_csd = mne.preprocessing.compute_current_source_density(data)
# Plot the raw data
data_csd.plot(duration=30, n_channels=30, 
              title='EEG (CSD) Data with Reconstructed Times')
# Create epochs
trigger_times = time_reconstruct_stim #- np.float64(t0) 
sample_indices = [int(t * fs) for t in trigger_times]

# Prepare the events array
events = np.array([[sample, 0, 1] for sample in sample_indices])

tmin = -1.98  # 200 ms before the event
tmax = 1.98   # 800 ms after the event
epochs = mne.Epochs(data_csd, events=events, 
                    tmin=tmin, tmax=tmax, preload=True)


# Optionally, you can plot the epochs
epochs.average().plot(picks='C3')

#%%
print(f"Number of new samples: {stream.n_new_samples}")
data, ts = stream.get_data()
time.sleep(0.5)
print(f"Number of new samples: {stream.n_new_samples}")

t0 = local_clock()
f, ax = plt.subplots(64, 1, sharex=True, constrained_layout=True)
for _ in range(3):
    # figure how many new samples are available, in seconds
    winsize = stream.n_new_samples / stream.info["sfreq"]
    # retrieve and plot data
    data, ts = stream.get_data(winsize)
    for k, data_channel in enumerate(data):
        ax[k].plot(ts - t0, data_channel)
    time.sleep(0.5)
for k, ch in enumerate(stream.ch_names):
    ax[k].set_title(f"EEG {ch}")
ax[-1].set_xlabel("Timestamp (LSL time)")
plt.show()