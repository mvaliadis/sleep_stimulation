#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 04:08:01 2024

@author: administrator
"""

import yasa
import numpy as np
import pandas as pd
import antropy as ant
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)

# Load EEG data
f = np.load('/home/administrator/Downloads/data_full_6hrs_100Hz_Cz+Fz+Pz.npz')
data, ch_names = f['data'], f['chan']
sf = 100.
times = np.arange(data.size) / sf

# Keep only Cz
data = data[0, :]
print(data.shape, np.round(data[0:5], 3))



# Load the hypnogram data
hypno = np.loadtxt('/home/administrator/Downloads/data_full_6hrs_100Hz_hypno_30s.txt').astype(int)
print(hypno.shape, 'Unique values =', np.unique(hypno))


# Convert the EEG data to 30-sec data
times, data_win = yasa.sliding_window(data, sf, window=30)

# Convert times to minutes
times /= 60

data_win.shape

from numpy import apply_along_axis as apply

df_feat = {
    # Entropy
    'perm_entropy': apply(ant.perm_entropy, axis=1, arr=data_win, normalize=True),
    'svd_entropy': apply(ant.svd_entropy, 1, data_win, normalize=True),
    'sample_entropy': apply(ant.sample_entropy, 1, data_win),
    # Fractal dimension
    'dfa': apply(ant.detrended_fluctuation, 1, data_win),
    'petrosian': apply(ant.petrosian_fd, 1, data_win),
    'katz': apply(ant.katz_fd, 1, data_win),
    'higuchi': apply(ant.higuchi_fd, 1, data_win),
}

df_feat = pd.DataFrame(df_feat)
df_feat.head()


from scipy.signal import welch
freqs, psd = welch(data_win, sf, nperseg=int(4 * sf))
bp = yasa.bandpower_from_psd_ndarray(psd, freqs)
bp = pd.DataFrame(bp.T, columns=['delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma'])
df_feat = pd.concat([df_feat, bp], axis=1)
df_feat.head()



from sklearn.feature_selection import f_classif

# Extract sorted F-values
fvals = pd.Series(f_classif(X=df_feat, y=hypno)[0], 
                  index=df_feat.columns
                 ).sort_values()

# Plot best features
plt.figure(figsize=(6, 6))
sns.barplot(y=fvals.index, x=fvals, palette='RdYlGn')
plt.xlabel('F-values')
plt.xticks(rotation=20);

# Plot hypnogram and higuchi
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

hypno = pd.Series(hypno).map({-1: -1, 0: 0, 1: 2, 2: 3, 3: 4, 4: 1}).values
hypno_rem = np.ma.masked_not_equal(hypno, 1)

# Plot the hypnogram
ax1.step(times, -1 * hypno, color='k', lw=1.5)
ax1.step(times, -1 * hypno_rem, color='r', lw=2.5)
ax1.set_yticks([0, -1, -2, -3, -4])
ax1.set_yticklabels(['W', 'R', 'N1', 'N2', 'N3'])
ax1.set_ylim(-4.5, 0.5)
ax1.set_ylabel('Sleep stage')

# Plot the non-linear feature
ax2.plot(times, df_feat['delta'])
ax2.set_ylabel('Delta activity')
ax2.set_xlabel('Time [minutes]')

ax2.set_xlim(0, times[-1]);

