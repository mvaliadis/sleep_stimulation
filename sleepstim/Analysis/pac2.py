#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:17:59 2022

@author: administrator
"""

## Create ERPAC plot
def ERPAC(data, f_pha=[0.5, 4], f_amp=(4, 30, .25, .25), n_perm=None, smooth=200, 
          method = 'gc', edges=0.5, stationarity_t=False, plot=True, save_path=None):
    rp_obj = EventRelatedPac(f_pha=f_pha, f_amp=f_amp)
    edges=int(edges*data.info['sfreq'])
    if stationarity_t:
        stationary_epochs = unit_root_test(data, p=0.05)
        if len(stationary_epochs) > 0:
            erpac = rp_obj.filterfit(int(data.info['sfreq']), data.get_data(picks='C3').squeeze()[stationary_epochs,:],
                                     method=method, smooth=smooth, edges=edges, n_perm=n_perm)
        else:
            return np.nan
    else:
        erpac = rp_obj.filterfit(int(data.info['sfreq']), data.get_data(picks='C3').squeeze(),
                                 method=method, smooth=smooth, edges=edges, n_perm=n_perm)
    if plot:
        plt.figure(figsize=(8, 6))
        rp_obj.pacplot(erpac.squeeze(), data.times[edges:-1-edges], rp_obj.yvec, xlabel='Time',
                       ylabel='Amplitude frequency (Hz)',
                       title='Event-Related PAC occurring for Delta phase',
                       fz_labels=15, fz_title=18)
        add_stimulus_onset(color = 'white')
        plt.show()
        if save_path is not None:
            plt.savefig(save_path + '_erpac_plot.png')
    
    return erpac


erpac = ERPAC(data = epoch, f_pha=[0.3, 4], f_amp=(5, 25, 0.25, 0.25), 
              smooth=100, method = 'gc', edges=1, stationarity_t=False, 
              plot=True, save_path=None)
              #save_path=fig_path + files.split('/')[-1])
              
      
#%%
import numpy as np
from tensorpac import EventRelatedPac
import matplotlib.pyplot as plt

# define an ERPAC object
p = EventRelatedPac(f_pha=[0.3, 4],  f_amp=(5, 25, 0.25, 0.25))#f_amp='hres')

# compute the erpac
erpac = p.filterfit(128, epoch.get_data(picks='C3', units='uV').squeeze(),
                    method='gc', smooth=None, edges=128, n_perm=None).squeeze()
 
# plot
plt.figure(figsize=(7, 4))
p.pacplot(erpac, epoch.times[128:-1-128], p.yvec, xlabel='Time (second)',
          cmap='Spectral_r', ylabel='Amplitude frequency', title=p.method,
          cblabel='ERPAC', vmin=0., rmaxis=True)
plt.axvline(1.075, linestyle='--', color='w', linewidth=2)

plt.tight_layout()
p.show()
