#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:47:18 2021

@author: administrator
"""

eps = epochs.filter(l_freq = None, h_freq=2.0)
imfs = []
for i in range(epochs._data.shape[0]):
    imf = emd.sift.sift(eps.get_data(picks='C3').squeeze()[i,:]*1e6, max_imfs=3)
    imfs.append(imf)  

IPs = []
for i in range(len(imfs)):
    IP, _, _ = emd.spectra.frequency_transform(imfs[i],512,'nht')
    IPs.append(IP)
    

pn_phase_emd = np.asarray(IPs)[:,:,0][:,512*3]
plt.figure()
ax = plt.subplot(111, projection='polar')
ax.hist(pn_phase_emd)