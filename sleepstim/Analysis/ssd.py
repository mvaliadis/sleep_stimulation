#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:51:10 2022

@author: administrator
"""


for times in ([-4,-2],[-2,0],[0,2]):
    ssd = SSD(info=up.copy().pick('eeg').info,
              reg='shrinkage',
              sort_by_spectral_ratio=True,  # False for purpose of example.
              filt_params_signal=dict(l_freq=12, h_freq=16,
                                      l_trans_bandwidth=1, h_trans_bandwidth=1),
              filt_params_noise=dict(l_freq=0, h_freq=25,
                                     l_trans_bandwidth=1, h_trans_bandwidth=1))
    ssd.fit(X=up.get_data('eeg', tmin=times[0], tmax=times[1]))
    
    # Plot topographies.
    pattern_epochs = mne.EvokedArray(data=ssd.patterns_[:4].T,
                                     info=ssd.info)
    pattern_epochs.plot_topomap(time_format='', 
                                title=f'SO spatial patterns: {times[0]} - {times[1]} s')

# Transform
ssd_sources = ssd.transform(X=up.get_data('eeg'))

# Get psd of SSD-filtered signals.
psd, freqs = mne.time_frequency.psd_array_welch(
    ssd_sources, sfreq=up.copy().pick('eeg').info['sfreq'], 
    n_fft=int(up.copy().pick('eeg').info['sfreq']*4))

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

# We can see that the initial sorting based on the eigenvalues
# was already quite good. However, when using few components only
# the sorting might make a difference.

below50 = freqs < 50
# for highlighting the freq. band of interest
bandfilt = (freqs_sig[0] <= freqs) & (freqs <= freqs_sig[1])
fig, ax = plt.subplots(1)
ax.loglog(freqs[below50], psd[0, :, below50], label='max SNR')
ax.loglog(freqs[below50], psd[-1, :, below50], label='min SNR')
ax.loglog(freqs[below50], psd[:, below50].mean(axis=0), label='mean')
ax.fill_between(freqs[bandfilt], 0, 10000, color='green', alpha=0.15)
ax.set_xlabel('log(frequency)')
ax.set_ylabel('log(power)')
ax.legend()
plt.show()

# We can clearly see that the selected component enjoys an SNR that is
# way above the average power spectrum.