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
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import cross_val_score, GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import yasa 
import mne
import pandas as pd
import seaborn as sns
from scipy.signal import welch
from scipy import interp
import scipy
from os import chdir as cd
from os import listdir
from sleepstim.sleep_funs import (plot_multiclass_ROC, bandpower, unravel_hypnogram_visbrain,
                                  plot_confusion_matrix, feature_extraction, ecg_feature_extraction  )
from sleepstim.Analysis.analysis_pre_process import load_preprocessed_data, Data_Struct, check_match_data_hypno_elements
sns.set_theme(color_codes=True)

#%%
## Study 1 data 

rf_path = '/home/administrator/sleep_stimulation-development/'
rf1 = pickle.load(open(rf_path + "rf_model_1_cfs.p","rb"))
rf2 = pickle.load(open(rf_path + "rf_model_2_cfs.p","rb"))
rf3 = pickle.load(open(rf_path + "rf_model_3_cfs.p","rb"))
rf4 = pickle.load(open(rf_path + "rf_model_4_cfs.p","rb"))
rf5 = pickle.load(open(rf_path + "rf_model_5_cfs.p","rb"))
rf6 = pickle.load(open(rf_path + "rf_model_6_cfs.p","rb"))

new_path = '/media/administrator/data/cfs/models/'
rf_new = pickle.load(open(new_path + "rf_model_new_cfs.p", "rb"))
nn_new = pickle.load(open(new_path + "lgbm_model_new_cfs.p", "rb"))
lgbm_new = pickle.load(open(new_path + "nn_model_new_cfs.p", "rb"))

def classifier_validation(session='adaption', new_features=True):
    if session == 'adaption':
        data_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Adaption_classifier_validation'
        hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Adaption/'
        new_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/Adaption/'
        hypno_path2 = list()
    elif session == 'experimental':
        data_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_classifier_validation'
        hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/'
        new_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/Experimental/'
        hypno_path2 = '/media/administrator/data/Study_1_data/Pre-processed_data/EDFs/Auto_hypnograms/'
    accArrays = [] 
    exist_exp_files, exist_hyp_files = check_match_data_hypno_elements(data_path, hypno_path, hypno_path2)
    for idx, (fname, hypno_files) in enumerate(zip(exist_exp_files, exist_hyp_files)):
        print(idx, fname, hypno_files)
        Data = load_preprocessed_data(data_path + '/' + fname)[0]
        # parse items for later
        subj = fname.split('_')[0]
        from sleepstim.Analysis.Resting_State.rs_preproc import subject_cond_parser
        if fname.split('_')[1] != 'adaption':            
            cond = subject_cond_parser(fname, study_phase='classifier')
        else:
            cond = 'adaption'
        # epoch data first
        index = np.r_[Data.chans.index('C3'), Data.chans.index('Cz'), Data.chans.index('C4'), 
                      Data.chans.index('EOG_L'), Data.chans.index('EOG_R'), Data.chans.index('EMG_L'), 
                      Data.chans.index('EMG_R')]
        new_chans = ['C3','Cz','C4','EOG_L','EOG_R','EMG_L','EMG_R']
        data_ = Data.data[:,index]
        eeg = data_[:, [new_chans.index('C3'), new_chans.index('Cz'), 
                        new_chans.index('C4')]]
        eog = np.mean(data_[:, [new_chans.index('EOG_L'), new_chans.index('EOG_R')]], 
                      axis=-1, keepdims=True)
        emg = np.mean(data_[:, [new_chans.index('EMG_L'), new_chans.index('EMG_R')]],
                      axis=-1, keepdims=True)
        data = np.column_stack((eeg, emg, eog))
        
        _, epochs = yasa.sliding_window(data.T, sf=Data.sfreq, window=30)
        
        # 30s per stage hypnogram
        try:
            hypnogram = unravel_hypnogram_visbrain(hypno_path + hypno_files, epochs)
        except:
            hypnogram = np.load(hypno_path2 + hypno_files)
            
        # unsampled hypnogram to data
        #hypnogram = Data.hypno[::Data.sfreq*30][:epochs.shape[0]]
        
        ## extract features
        features = feature_extraction(epochs, Data.sfreq, line_noise_freq=(49, 51), 
                                      ch_names=['C3', 'Cz', 'C4', 'EOG', 'EMG'])
        # drop a channel
        #features = features.set_index('chan').drop(index=['Cz']).reset_index()
        
        # concatenate over features
        x0 = features.set_index(['chan','epoch']).loc[['C3', 'C4', 'EOG', 'EMG']].unstack(level='chan').to_numpy()
       
                    
        ## compute PSD with welch's method + yasa absolute/relative power extraction 
        # compute power spectral density with welch's method
        bp = bandpower(epochs, fs=Data.sfreq, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                                     (8, 12, 'Alpha'), (12, 16, 'Sigma'), 
                                                     (16, 30, 'Beta'), (49, 51, 'Line noise')], relative=True)
        
        # reshape data for classifier, must be (epochs x (nchans*bands))
        bp = np.swapaxes(bp, 0, 1)
        nepochs, nbands, nchans = np.shape(bp)
        features = bp.reshape(nepochs, nchans*nbands, order='F')
    
        # cut segments by channels
        cutindx = np.linspace(0, nchans*nbands, nchans+1).astype(int)
        #C3, Cz, C4, EOG_L, EOG_R, EMG_L, EMG_R = [features[:,cutindx[i]:cutindx[i+1]] for i in range(nchans)]
        C3, Cz, C4, EOG, EMG = [features[:,cutindx[i]:cutindx[i+1]] for i in range(nchans)]
          
        # find average values for EMG, EOG
        # EMG = np.mean([EMG_L, EMG_R], axis=0)
        # EOG = np.mean([EOG_L, EOG_R], axis=0)
    
        # combine EEG, EOG, EMG
        x1 = np.concatenate([C3[:,0:5],C4[:,0:5],EOG[:,0:5],EMG[:,0:5]], axis=1) #2 EEG, 1 EOG, 1 EMG
        x2 = np.concatenate([C3[:,0:5],EOG[:,0:5],EMG[:,0:5]], axis=1)           #1 EEG, 1 EOG, 1 EMG
        x3 = np.concatenate([C3[:,0:5],C4[:,0:5],EOG[:,0:5]], axis=1)            #2 EEG, 1 EOG
        x4 = np.concatenate([C3[:,0:5],C4[:,0:5],EMG[:,0:5]], axis=1)            #2 EEG, 1 EMG
        x5 = np.concatenate([C3[:,0:5],C4[:,0:5]], axis=1)                       #2 EEG
        x6 = np.concatenate([C3[:,0:5]], axis=1)                                 #1 EEG
                                    
        # predict based on online classification selection
        y_pred0_rf = rf_new.predict(x0)
        y_score0_rf = rf_new.predict_proba(x0)
        y_pred0_lgbm = lgbm_new.predict(x0)
        y_score0_lgbm = lgbm_new.predict_proba(x0)
        y_pred0_nn = nn_new.predict(x0)
        y_score0_nn = nn_new.predict_proba(x0)
        if len(y_pred0_rf) == len(hypnogram):
            y_pred1 = rf1.predict(x1)
            # y_pred1_mf = scipy.ndimage.median_filter(y_pred1, size=10)
            y_score1 = rf1.predict_proba(x1) 
            y_pred2 = rf2.predict(x2)
            # y_pred2_mf = scipy.ndimage.median_filter(y_pred2, size=10)
            y_score2 = rf2.predict_proba(x2) 
            y_pred3 = rf3.predict(x3)
            # y_pred3_mf = scipy.ndimage.median_filter(y_pred3, size=10)
            y_score3 = rf3.predict_proba(x3) 
            y_pred4 = rf4.predict(x4)
            # y_pred4_mf = scipy.ndimage.median_filter(y_pred4, size=10)
            y_score4 = rf4.predict_proba(x4) 
            y_pred5 = rf5.predict(x5)
            # y_pred5_mf = scipy.ndimage.median_filter(y_pred5, size=10)
            y_score5 = rf5.predict_proba(x5) 
            y_pred6 = rf6.predict(x6)
            # y_pred6_mf = scipy.ndimage.median_filter(y_pred6, size=10)
            y_score6 = rf6.predict_proba(x6) 
            
            
            # yasa classification
            info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, 
                                   ch_types=Data.chtypes)
            raw = mne.io.RawArray(Data.data.T/1e6, info)
            # raw = mne.set_bipolar_reference(raw, 'EOG_L', 'EOG_R')
            # raw = mne.set_bipolar_reference(raw, 'EMG_L', 'EMG_R')
            sls = yasa.SleepStaging(raw, eeg_name="Cz", eog_name="EOG_L", emg_name="EMG_L")
            y_pred_yasa = yasa.hypno_str_to_int(sls.predict())
            yasa_score = sls.predict_proba().to_numpy()
            #yasa_pred_mf = scipy.ndimage.median_filter(y_pred_yasa, size=10)
            
            # Sleep stage names
            event_id ={'Wake':0,
                       'Stage 1':1,
                       'Stage 2':2,
                       'Stage 3':3,
                       'REM':4}
            #target_names=event_id.keys()
            
            # # Prediction model key
            # model_id ={'y_pred0_rf': '2 EEG, 1 EOG, 1 EMG',
            #            'y_pred0_lgbm': '2 EEG, 1 EOG, 1 EMG',
            #            'y_pred0_nn': '2 EEG, 1 EOG, 1 EMG',
            #            'y_pred1':'2 EEG, 1 EOG, 1 EMG',
            #            # 'y_pred1 med':'2 EEG, 1 EOG, 1 EMG',
            #            'y_pred2':'1 EEG, 1 EOG, 1 EMG',
            #            # 'y_pred2 med':'1 EEG, 1 EOG, 1 EMG',
            #            'y_pred3':'2 EEG, 1 EOG',
            #            # 'y_pred3 med':'2 EEG, 1 EOG',
            #            'y_pred4':'2 EEG, 1 EMG',
            #            # 'y_pred4 med':'2 EEG, 1 EMG',
            #            'y_pred5':'2 EEG',
            #            # 'y_pred5 med':'1 EEG',
            #            'y_pred6':'1 EEG',
            #            # 'y_pred6 med':'1 EEG',
            #            'y_pred_yasa': '1 EEG, 1 EMG, 1 EOG'}
                       #'y_pred_yasa med': '1 EEG, 1 EMG, 1 EOG'}
           # model_names=model_id.keys()
            
            # Current hypnogram becomes test
            y_test = hypnogram
            
            # accuracy report, confusion matrix, classification reports
            y_preds = (y_pred0_rf, y_pred0_lgbm, y_pred0_nn, y_pred1, y_pred2, 
                       y_pred3, y_pred4, y_pred5, y_pred6, y_pred_yasa)
            y_scores = (y_score0_rf, y_score0_lgbm, y_score0_nn, y_score1, y_score2, 
                        y_score3, y_score4, y_score5, y_score6, yasa_score)
            for idx, (y_pred, y_score) in enumerate(zip(y_preds, y_scores)):
                acc = accuracy_score(y_test, y_pred)
                print("Accuracy score: {}".format(acc))
                #confusion = confusion_matrix(y_test, y_pred)
                #print(confusion_matrix(y_test, y_pred))
                if len(np.unique(y_pred)) == 5:
                    report = classification_report(y_test, y_pred, target_names=event_id.keys())
                    print(report)
                else:
                    pass
                
                cm = confusion_matrix(y_test, y_pred, normalize='true')
                np.set_printoptions(precision=2)
                print(f'Confusion matrix: \n {cm}' + '\n')
                
                # Compute interrater reliability
                inter_agreement = cohen_kappa_score(y_test, y_pred).round(2)
                print(f'The inter-rate agreement is K = {inter_agreement}' + '\n')
                  
                accArrays.append([subj, cond, y_test, y_pred, 
                                  y_score, inter_agreement, idx])
                
                # plt.figure()
                # plot_confusion_matrix(path = new_path, cm=cm, target_names=target_names, title=f'Confusion matrix - {"_".join(fname.split("_")[0:2])} - {list(model_id.values())[idx]}', save=True, 
                #                       save_name = list(model_id)[idx] + "_".join(fname.split("_")[0:2]))
                # plot_multiclass_ROC(level = 'subject', path = new_path, y_test_all = y_test, y_score_all = y_score, save_name = list(model_id)[idx] + "_".join(fname.split("_")[0:2]))
                # plt.close()
                # sns.heatmap(cm, cmap=plt.cm.Blues,square=True, annot=True, cbar=True)
                # plt.xlabel('predicted value')
                # plt.ylabel('true value');
            
    return accArrays


#%% 

run = input('Do you wish to restart the classifier validation analysis? ')
if run == 'yes':
    df_adaption = pd.DataFrame(classifier_validation(session='adaption'), 
                               columns=['Subject','Condition','Test hypnograms','Pred hypnograms',
                                         'Pred probabilities','Cohens Kappa','model idx'])
    df_experimental = pd.DataFrame(classifier_validation(session='experimental'), 
                                   columns=['Subject','Condition','Test hypnograms','Pred hypnograms',
                                            'Pred probabilities','Cohens Kappa','model idx'])
    df = pd.concat([df_adaption, df_experimental])
    save_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/RF_results_.p'
    pickle.dump(df, open(save_path, "wb"))  
else:
    #df = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/Classifier_validation/RF_results.p', 'rb'))
    #df.to_csv(r'/media/administrator/data/Study_1_data/Statistics/Classifier_validation/RF_results.csv')
    
    df = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/Classifier_validation/RF_results_new.p', 'rb'))
    
#%%

def group_classification_validation(df, model_id, event_id, target_names):     
    new_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/All/'
    
    y_tests_all = []
    for idx in range(len(list(model_id))):
        y_tests_all.append(np.hstack([df[df['model idx']==idx]['Test hypnograms'].to_numpy()[i] for i in range(df[df['model idx']==idx]['Test hypnograms'].shape[0])]))
                   
    y_preds_all = []
    for idx in range(len(list(model_id))):
        y_preds_all.append(np.hstack([df[df['model idx']==idx]['Pred hypnograms'].to_numpy()[i] for i in range(df[df['model idx']==idx]['Pred hypnograms'].shape[0])]))        
       
    y_scores_all = []
    for idx in range(len(list(model_id))):
        y_scores_all.append(np.vstack([df[df['model idx']==idx]['Pred probabilities'].to_numpy()[i] for i in range(df[df['model idx']==idx]['Pred probabilities'].shape[0])]))            
                   
    y_cohen_ks_all = []
    for idx in range(len(list(model_id))):
        y_cohen_ks_all.append(np.hstack([df[df['model idx']==idx]['Cohens Kappa'].to_numpy()[i] for i in range(df[df['model idx']==idx]['Cohens Kappa'].shape[0])]))        
       
    
    for idx, (y_pred_all, y_test_all, y_score_all, y_cohen_k_all) in enumerate(zip(y_preds_all, y_tests_all, y_scores_all, y_cohen_ks_all)):    
        print("Model: {}".format(model_id[list(model_id)[idx]]))
        acc = accuracy_score(y_test_all, y_pred_all)
        print("Accuracy score: {}".format(acc))
        confusion = confusion_matrix(y_test_all, y_pred_all)
        print(confusion)
        report = classification_report(y_test_all, y_pred_all, target_names=event_id.keys())
        print(report)
        cm = confusion_matrix(y_test_all, y_pred_all, normalize='true').round(2)
        np.set_printoptions(precision=2)
        print(f'Confusion matrix: \n {cm}' + '\n')
        
        ############## - compute over [] - #############
        # import scikits.bootstrap as bootstrap
        # CI = bootstrap.ci
        #inter_agreement = df.groupby('Cohens Kappa').apply(lambda x:bootstrap.ci(data=x, statfunction=scipy.mean)).round(2)
        inter_agreement = np.nanmean(y_cohen_k_all)
        print(f'The inter-rate agreement is K = {inter_agreement}' + '\n')
        
        # plt.figure()
        # plot_confusion_matrix(path = new_path, cm=cm, target_names=target_names, title=f'Confusion matrix - All Subjects - {list(model_id.values())[idx]}', 
        #                       save=True, save_name = list(model_id)[idx] + '_all_subj')
        plot_multiclass_ROC(level = 'group level', path = new_path, y_test_all = y_test_all, y_score_all = y_score_all, save_name = list(model_id)[idx] + '_all_subj')
        
        plt.figure()
        sns.heatmap(cm, cmap=plt.cm.Blues,square=True, annot=True, cbar=True, xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=360)
        plt.title(f'Confusion matrix - All Subjects - {list(model_id.values())[idx]}')
        plt.tight_layout()
        plt.savefig(new_path + f"confusion_matrix_{list(model_id)[idx] + '_all_subj'}")
        
    plt.close('all')
    
    sns.violinplot(x = 'model idx', y = 'Cohens Kappa', data = df, cut=1)
    plt.title('Inter-rater reliability of RF models')
    sns.despine()
    plt.savefig(new_path + 'cohens_kappa_all_models.png')
    
    plt.close('all')
        

#%%
# Prediction model key
# model_id ={'y_pred1':'2 EEG, 1 EOG, 1 EMG',
#            'y_pred1 med':'2 EEG, 1 EOG, 1 EMG',
#            'y_pred2':'1 EEG, 1 EOG, 1 EMG',
#            'y_pred2 med':'1 EEG, 1 EOG, 1 EMG',
#            'y_pred3':'2 EEG, 1 EOG',
#            'y_pred3 med':'2 EEG, 1 EOG',
#            'y_pred4':'2 EEG, 1 EMG',
#            'y_pred4 med':'2 EEG, 1 EMG',
#            'y_pred5':'2 EEG',
#            'y_pred5 med':'1 EEG',
#            'y_pred6':'1 EEG',
#            'y_pred6 med':'1 EEG',
#            'y_pred_yasa': '1 EEG, 1 EMG, 1 EOG',
#            'y_pred_yasa med': '1 EEG, 1 EMG, 1 EOG'}

# Prediction model key
model_id ={'y_pred0_rf': '2 EEG, 1 EOG, 1 EMG',
           'y_pred0_lgbm': '2 EEG, 1 EOG, 1 EMG',
           'y_pred0_nn': '2 EEG, 1 EOG, 1 EMG',
           'y_pred1':'2 EEG, 1 EOG, 1 EMG',
           'y_pred2':'1 EEG, 1 EOG, 1 EMG',
           'y_pred3':'2 EEG, 1 EOG',
           'y_pred4':'2 EEG, 1 EMG',
           'y_pred5':'2 EEG',
           'y_pred6':'1 EEG',
           'y_pred_yasa': '1 EEG, 1 EMG, 1 EOG'}

event_id ={'Wake':0,
           'Stage 1':1,
           'Stage 2':2,
           'Stage 3':3,
           'REM':4}
target_names=event_id.keys()
        
exp_df = df[df.Condition != 'adaption'].reset_index(drop=True)
group_classification_validation(exp_df, model_id, event_id, target_names)
