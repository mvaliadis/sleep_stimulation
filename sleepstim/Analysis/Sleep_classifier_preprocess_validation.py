#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 21:40:50 2020

@author: administrator
"""

# =============================================================================
# Classifier validation
# =============================================================================

import lightgbm as lgb 
import xgboost as xgb
import treelite
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import yasa 
import pandas as pd
# from scipy.signal import welch
from os import chdir as cd
from os import listdir
from sleepstim.sleep_funs import bandpower, unravel_hypnogram_visbrain, plot_confusion_matrix
from sleepstim.Analysis.analysis_pre_process import load_preprocessed_data, Data_Struct
#cd('/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_unblinded_cleaned')

#%%
## Study 1 data 

rf_path = '/home/administrator/sleep_stimulation-development/'
rf1 = pickle.load(open(rf_path + "rf_model_1_cfs.p","rb"))
rf2 = pickle.load(open(rf_path + "rf_model_2_cfs.p","rb"))
rf3 = pickle.load(open(rf_path + "rf_model_3_cfs.p","rb"))
rf4 = pickle.load(open(rf_path + "rf_model_4_cfs.p","rb"))
rf5 = pickle.load(open(rf_path + "rf_model_5_cfs.p","rb"))
rf6 = pickle.load(open(rf_path + "rf_model_6_cfs.p","rb"))

files = list(set([f.split('.')[0] for f in sorted(listdir()) if f.endswith('.p')]))

allArrays = []
stageArrays = []
new_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/'

for idx, fname in enumerate(files): 
    print(idx, fname)
    Data = load_preprocessed_data(fname + '.p')
    # epoch data first
    index = np.r_[Data.chans.index('C3'), Data.chans.index('Cz'), Data.chans.index('C4'), 
                  Data.chans.index('EOG_L'), Data.chans.index('EOG_R'), Data.chans.index('EMG_L'), 
                  Data.chans.index('EMG_R')]
    new_chans = ['C3','Cz','C4','EOG_L','EOG_R','EMG_L','EMG_R']
    data = Data.data[:,index]
    _, epochs = yasa.sliding_window(data.T, sf=512, window=30)
    
    # load hypnogram
    hypnogram = unravel_hypnogram_visbrain('/media/administrator/data/Study_1_data/Hypnograms/Unblinded/' + fname + '_hypno.txt', epochs)
     
    # unsampled hypnogram to data
    unsampled_hypno = yasa.hypno_upsample_to_data(hypno=hypnogram, sf_hypno=1/30, data=data.T, sf_data=512)
    
    ## compute PSD with welch's method + yasa absolute/relative power extraction 
    # compute power spectral density with welch's method
    bp = bandpower(epochs, fs=512, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                           (8, 12, 'Alpha'), (12, 16, 'Sigma'), 
                                           (16, 30, 'Beta'), (49, 51, 'Line noise')], relative=True)
    
    # reshape data for classifier, must be (epochs x (nchans*bands))
    bp = np.swapaxes(bp, 0, 1)
    nepochs, nbands, nchans = np.shape(bp)
    bp = bp.reshape(nepochs, nchans*nbands, order='F')
    
    # cut segments by channels
    cutindx = np.linspace(0, nchans*nbands, nchans+1).astype(int)
    C3, Cz, C4, EOG_L, EOG_R, EMG_L, EMG_R = [bp[:,cutindx[i]:cutindx[i+1]] for i in range(nchans)]
     
    # find average values for EMG, EOG
    EMG = np.mean([EMG_L, EMG_R], axis=0)
    EOG = np.mean([EOG_L, EOG_R], axis=0)
    
    # combine EEG, EOG, EMG
    x1 = np.concatenate([C3[:,0:5],C4[:,0:5],EOG[:,0:5],EMG[:,0:5]], axis=1) #2 EEG, 1 EOG, 1 EMG
    x2 = np.concatenate([C3[:,0:5],EOG[:,0:5],EMG[:,0:5]], axis=1)           #1 EEG, 1 EOG, 1 EMG
    x3 = np.concatenate([C3[:,0:5],C4[:,0:5],EOG[:,0:5]], axis=1)            #2 EEG, 1 EOG
    x4 = np.concatenate([C3[:,0:5],C4[:,0:5],EMG[:,0:5]], axis=1)            #2 EEG, 1 EMG
    x5 = np.concatenate([C3[:,0:5],C4[:,0:5]], axis=1)                       #2 EEG
    x6 = np.concatenate([C3[:,0:5]], axis=1)                                 #1 EEG
                            
    # predict based on online classification selection
    y_pred1 = rf1.predict(x1)
    y_pred2 = rf2.predict(x2)
    y_pred3 = rf3.predict(x3)
    y_pred4 = rf4.predict(x4)
    y_pred5 = rf5.predict(x5)
    y_pred6 = rf6.predict(x6)
        
    # Sleep stage names
    event_id ={'Wake':0,
               'Stage 1':1,
               'Stage 2':2,
               'Stage 3':3,
               'REM':4}
    target_names=event_id.keys()
    
    # Prediction model key
    model_id ={'y_pred1':'2 EEG, 1 EOG, 1 EMG',
               'y_pred2':'1 EEG, 1 EOG, 1 EMG',
               'y_pred3':'2 EEG, 1 EOG',
               'y_pred4':'2 EEG, 1 EMG',
               'y_pred5':'2 EEG',
               'y_pred6':'1 EEG'}
    model_names=model_id.keys()
    
    # Current hypnogram becomes test
    y_test = hypnogram
    
    # accuracy report, confusion matrix, classification reports
    for idx, y_pred in enumerate((y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6)):
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy score: {}".format(acc))
        confusion = confusion_matrix(y_test, y_pred)
        # print(confusion_matrix(y_test, y_pred))
        report = classification_report(y_test, y_pred, target_names=event_id.keys())
        print(classification_report(y_test, y_pred, target_names=event_id.keys()))
        
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        np.set_printoptions(precision=1)
        print(f'Confusion matrix: \n {cm}' + '\n')
        plt.figure()
        cd(new_path)
        plot_confusion_matrix(cm, target_names, title=f'Confusion matrix - {fname} - {list(model_id.values())[idx]}', 
                              save=True, save_name=fname + '_' + list(model_id)[idx])

