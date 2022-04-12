#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:29:58 2021

@author: administrator
"""
from scipy.linalg import eigh
from scipy.interpolate import RectBivariateSpline
from scipy.signal import find_peaks, welch, detrend

# load dataset
Data = _pre_process_sleep_data(files, reference=None, validation='auditory', stageing=False)

# apply CSD if warranted
if surface_laplacian:
    mne_info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)
    raw = mne.io.RawArray(Data.data.T, mne_info)
    raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
    # ignore non-EEG channels! 
    raw.pick_types(eeg=True)
    Data.data = mne.preprocessing.compute_current_source_density(raw).get_data()
else:
    mne_info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)
    raw = mne.io.RawArray(Data.data.T, mne_info)
    # ignore non-EEG channels! 
    raw.pick_types(eeg=True)
    Data.data = raw.get_data()

del raw

# take first 210 minutes
Data.data = Data.data[:,0:512*60*210]
              
# Get filtered slow (0.5 - 2 Hz) and fast (2 - 4 Hz) data
slow_nrem_filt = mne.filter.filter_data(Data.data, sfreq=Data.sfreq, l_freq=0.5, h_freq=2., 
                                  verbose=0)
fast_nrem_filt = mne.filter.filter_data(Data.data, sfreq=Data.sfreq, l_freq=2., h_freq=4., 
                                  verbose=0)

# Remove the mean (= detrend)
slow_nrem_filt = detrend(slow_nrem_filt, type='constant')
fast_nrem_filt = detrend(fast_nrem_filt, type='constant')

# Compute the covariance matrices between channels
slow_nrem_cov = np.cov(slow_nrem_filt)
fast_nrem_cov = np.cov(fast_nrem_filt)

slow_nrem_cov = np.asarray([np.cov(slow_nrem_filt[i,:,:]) for i in range(np.size(center, 0))]).mean(0)
fast_nrem_cov = np.asarray([np.cov(fast_nrem_filt[i,:,:]) for i in range(np.size(center, 0))]).mean(0)

# Plot the slow covariance matrix
eeg_index = [i for i, x in enumerate(Data.chtypes) if x == "eeg"]
eeg_chans = [Data.chans[i] for i in eeg_index]
   
plt.figure(figsize=(10, 6))
sns.heatmap(slow_nrem_cov, cmap='Blues', square=True, 
            xticklabels=eeg_chans, yticklabels=eeg_chans,
            vmin = np.percentile(slow_nrem_cov, 20), 
            vmax = np.percentile(slow_nrem_cov, 80))
plt.title('SO variance-covariance matrix')
plt.xlabel('Channels')
_ = plt.ylabel('Channels')

# Get the eigenvalues / eigenvectors
eigval, eigvec = eigh(slow_nrem_cov, fast_nrem_cov)

# Flip to descending order
eigval = np.flip(eigval)
eigvec = np.fliplr(eigvec)

print('Eigenvalues =', list(np.round(eigval, 2)))

# Apply spatial filters by multiplying data with eigenvectors
sf_comp_nrem = np.dot(Data.data.T, eigvec).T
print(sf_comp_nrem.shape)


#%%

covmats_lf = Covariances().fit_transform(epochs.copy().filter(l_freq=8, h_freq=12).get_data('eeg'))
covmats_lf = detrend(covmats_lf, type='constant')
covmats_lf = Shrinkage().fit_transform(covmats_lf)

covmats_hf = Covariances().fit_transform(epochs.copy().filter(l_freq=12, h_freq=16).get_data('eeg'))
covmats_hf = detrend(covmats_hf, type='constant')
covmats_hf = Shrinkage().fit_transform(covmats_hf)


sns.heatmap(covmats_lf.mean(0), xticklabels= Data.ch_names[0:23], 
            yticklabels= Data.ch_names[0:23])
plt.figure()
sns.heatmap(covmats_hf.mean(0), xticklabels= Data.ch_names[0:23], 
            yticklabels= Data.ch_names[0:23])

eigval, eigvec = eigh(covmats_lf.mean(0), covmats_hf.mean(0))
