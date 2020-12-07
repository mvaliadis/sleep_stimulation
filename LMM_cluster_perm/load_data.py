#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:25:25 2020

@author: daniela
"""
import os.path
import pickle
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from clusterperm_LMM import permutation_cluster_test_multilevel, mixedlm_cond_grp

path = 'data/'
def group_SW_ERPs(path):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files])
    Up, Down, Sham = [], [], []
    Up_sub, Down_sub, Sham_sub = [], [], []
    for i, data_files in enumerate(files_list):
        if data_files.endswith('up_epochs.p'):
            epochs = pickle.load(open(data_files,"rb"))
            Up.append(epochs.get_data()*1e6)
            Up_sub.append(Up[-1].shape[0]*[data_files[len(path):len(path)+7]])
        elif data_files.endswith('down_epochs.p'):
            epochs = pickle.load(open(data_files,"rb"))
            Down.append(epochs.get_data()*1e6)
            Down_sub.append(Down[-1].shape[0]*[data_files[len(path):len(path)+7]])
        elif data_files.endswith('sham_epochs.p'):
            epochs = pickle.load(open(data_files,"rb"))
            Sham.append(epochs.get_data()*1e6)
            Sham_sub.append(Sham[-1].shape[0]*[data_files[len(path):len(path)+7]])
   
    info = mne.create_info(ch_names=['C3'], sfreq=512, ch_types=['eeg'])
    epochs_up = mne.EpochsArray(np.concatenate(Up)/1e6, info, tmin = -2, baseline=None)
    epochs_down = mne.EpochsArray(np.concatenate(Down)/1e6, info, tmin = -2, baseline=None)
    epochs_sham = mne.EpochsArray(np.concatenate(Sham)/1e6, info, tmin = -2, baseline=None)
      
    df = pd.concat([epochs_down.to_data_frame(), epochs_up.to_data_frame(),
                    epochs_sham.to_data_frame()], keys=['Down', 'Up', 'Sham'],
                    names=['Condition'], axis=0)
    df['time'] = df['time']/1000
    df = df.drop(columns="condition")
    df = df.rename(columns={'C3':'Amplitude (uV)','time': 'Time (s)'})
   
    Up = np.vstack(np.array(Up)).squeeze()
    Down = np.vstack(np.array(Down)).squeeze()
    Sham = np.vstack(np.array(Sham)).squeeze()
    Up_sub = np.hstack(np.array(Up_sub))
    Down_sub = np.hstack(np.array(Down_sub))
    Sham_sub = np.hstack(np.array(Sham_sub))
    return df, Up, Down, Sham, Up_sub, Down_sub, Sham_sub

df, Up, Down, Sham, Up_sub, Down_sub, Sham_sub = group_SW_ERPs(path)


# transforming subject gorupings to int
all_subs = np.unique(np.concatenate((Up_sub, Down_sub, Sham_sub)))
for i, s in enumerate(all_subs):
    print(i, s)
    Up_sub[np.where(Up_sub==s)] = i
    Down_sub[np.where(Down_sub==s)] = i
    Sham_sub[np.where(Sham_sub==s)] = i
Up_sub = Up_sub.astype(np.int64)
Down_sub = Down_sub.astype(np.int64)
Sham_sub = Sham_sub.astype(np.int64)

# making data shorter for testing
Up = Up[:,::10]
Down = Down[:,::10]
Sham = Sham[:,::10]

X_list = [Up, Down, Sham]
groups = [Up_sub, Down_sub, Sham_sub]

# run stats
t_obs, clusters, cluster_pv, H0 = permutation_cluster_test_multilevel(X_list, n_permutations = 10, stat_fun_ml=mixedlm_cond_grp, mm_groups=groups)


# plotting results
for i in range(len(clusters)):
    plt.axvspan(clusters[i][0].start, clusters[i][0].stop, facecolor='r', alpha=0.2)
    
plt.plot(np.mean(Up, axis = 0))
plt.plot(np.mean(Down, axis = 0))
plt.plot(np.mean(Sham, axis = 0))

plt.legend(('Up', 'Down', 'Sham'))