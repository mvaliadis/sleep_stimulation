#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:12:12 2024

@author: administrator
"""

from sleepstim.Analysis.NIDRA import pci
from glob import glob
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
mne.set_log_level('ERROR')

def analyze_wake_nmes_epochs_pci(epochs, subject, night):  
    # Extract PCI
    par = {'baseline_window':(-0.2, -0.0), 
           'response_window':(0.0, 0.4), 
           'k':1.2, 
           'min_snr':1.1, 
           'max_var':99,
           'embed':False,
           'n_steps':100, 
           'avgref': False}
    # evk = mne.combine_evoked([epochs['nmes_stim'].average(),
    #                           epochs['nmes_sham'].average()],
    #                          weights=[1, -1])
    evk = epochs['nmes_stim'].average()
    pci_ = pci.calc_PCIst(evk.get_data()*1e3, evk.times, full_return=True, **par)
    
    # Create a new dictionary for the current channel and stimulation
    pci_dict = {
        'Subject': subject,
        'Night': night,
        # 'Mode': mode, 
        'PCI': pci_['PCI'],
    } 
            
    # Convert to df
    df = pd.Series(pci_dict) 
    
    return df, pci_

# Load epochs and create grand average
save_path = '/media/administrator/Sleep_Data/Processed/Calibration/'
epochs_files = glob(save_path + 'Epochs/*calibration-epochs.fif')

for epoch_file in epochs_files:
    print(epoch_file)
    # break
    subject = epoch_file.split('/')[-1].split('_')[0]
    night = epoch_file.split('/')[-1].split('_')[1]
    epochs = mne.read_epochs(epoch_file) 
    # EDC = epochs['nmes_stim'].get_data('EDC').squeeze()
    # plt.plot(epochs.times, EDC.mean(0))
    # epochs['nmes_stim'].average().copy().crop(-0.1, 0.3).plot_joint()
    # epochs['nmes_stim'].average().copy().crop(0, 0.25).plot('C3')
    epoch_pci, pci_table = analyze_wake_nmes_epochs_pci(epochs, subject, night)
    print(epoch_pci)
    nst_times = np.linspace(0, 0.4, len(pci_table['NST_diff']))
    plt.plot(nst_times, pci_table['NST_diff'], color='k', alpha=0.50, linewidth=.75)
    plt.plot(nst_times, pci_table['NST_diff'][:, 0], color='r', alpha=1, linewidth=1)
plt.show()
    
    
    
    
