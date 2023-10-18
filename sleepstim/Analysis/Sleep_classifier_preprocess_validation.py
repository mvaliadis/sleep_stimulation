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
#import treelite
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import cross_val_score, GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, cohen_kappa_score, matthews_corrcoef, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import yasa 
import mne
import glob
from natsort import natsorted
import pandas as pd
import seaborn as sns
from scipy.signal import welch
from scipy import interp
import scipy
from os import chdir as cd
from os import listdir
cd('/home/administrator/sleep_stimulation/') 
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

model_path = "/media/administrator/data/cfs/models/"
for file in glob.glob(model_path + "*.p"):
    # assign name of loaded model
    model_name = file.split("/")[-1].split(".")[0]
    print(model_name)
    # load model
    model = pickle.load(open(file, "rb"))
    # reassign variable name based on model_name
    exec(model_name + "= model")
    
# compile list of model names
model_names = [model_name.split("/")[-1].split(".")[0] for model_name in glob.glob(model_path + "*.p")]

def classifier_validation(session='adaption', loaded=True):
    if session == 'adaption':
        data_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Adaption_classifier_validation'
        hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Adaption/'
        #new_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/Adaption/'
        hypno_path2 = list()
    elif session == 'experimental':
        data_path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_classifier_validation'
        hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/'
        #new_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/Experimental/'
        hypno_path2 = '/media/administrator/data/Study_1_data/Pre-processed_data/EDFs/Auto_hypnograms/'
    accArrays = [] 
    exist_exp_files, exist_hyp_files = check_match_data_hypno_elements(data_path, hypno_path, hypno_path2)
    for idx, (fname, hypno_files) in enumerate(zip(exist_exp_files, exist_hyp_files)):
        print(idx, fname, hypno_files)
        # parse items for later
        subj = fname.split('_')[0]
        from sleepstim.Analysis.Resting_State.rs_preproc import subject_cond_parser
        if fname.split('_')[1] != 'adaption':            
            cond = subject_cond_parser(fname, study_phase='classifier')
        else:
            cond = 'adaption'
        if loaded == False:
            Data = load_preprocessed_data(data_path + '/' + fname)[0]
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
            hypnogram = unravel_hypnogram_visbrain(hypno_path + hypno_files, data=None)#, epochs)
        except:
            hypnogram = np.load(hypno_path2 + hypno_files)
            
        # unsampled hypnogram to data
        #hypnogram = Data.hypno[::Data.sfreq*30][:epochs.shape[0]]
        
        ## extract features
        # model path
        feat_path = "/media/administrator/data/Study_1_data/Statistics/Classifier_validation/Features/"   
        if loaded:
            features = pd.read_csv(feat_path + f'features_{subj}_{cond}.csv', index_col=0)
        else:
            features = feature_extraction(epochs, Data.sfreq, line_noise_freq=(49, 51), 
                                      ch_names=['C3', 'Cz', 'C4', 'EOG', 'EMG'])
            # drop a channel
            features = features.set_index('chan').drop(index=['Cz']).reset_index()
            
            # save feats
            features.to_csv(feat_path + f'features_{subj}_{cond}.csv')
        
        # # Standarize features for classification
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler()
        # features[features.columns[1:-1]] = scaler.fit_transform(features[features.columns[1:-1]])
        features = features.reset_index(drop=True)
        
        if loaded == False:
            ## compute PSD with welch's method + yasa absolute/relative power extraction 
            # compute power spectral density with welch's method
            bp = bandpower(epochs, fs=Data.sfreq, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                                         (8, 12, 'Alpha'), (12, 16, 'Sigma'), 
                                                         (16, 30, 'Beta'), (49, 51, 'Line noise')], relative=True)
            
            # reshape data for classifier, must be (epochs x (nchans*bands))
            bp = np.swapaxes(bp, 0, 1)
            nepochs, nbands, nchans = np.shape(bp)
            features_ = bp.reshape(nepochs, nchans*nbands, order='F')
        
            # cut segments by channels
            cutindx = np.linspace(0, nchans*nbands, nchans+1).astype(int)
            #C3, Cz, C4, EOG_L, EOG_R, EMG_L, EMG_R = [features_[:,cutindx[i]:cutindx[i+1]] for i in range(nchans)]
            C3, Cz, C4, EOG, EMG = [features_[:,cutindx[i]:cutindx[i+1]] for i in range(nchans)]
              
            # find average values for EMG, EOG
            # EMG = np.mean([EMG_L, EMG_R], axis=0)
            # EOG = np.mean([EOG_L, EOG_R], axis=0)
            
            # yasa classification
            info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, 
                                   ch_types=Data.chtypes)
            raw = mne.io.RawArray(Data.data.T/1e6, info)
            # raw = mne.set_bipolar_reference(raw, 'EOG_L', 'EOG_R')
            # raw = mne.set_bipolar_reference(raw, 'EMG_L', 'EMG_R')
            sls = yasa.SleepStaging(raw, eeg_name="Cz", eog_name="EOG_L", emg_name="EMG_L")
            pickle.dump(sls, open(feat_path + f'yasa_{subj}_{cond}.p', "wb"))  

            # combine EEG, EOG, EMG
            x1 = np.concatenate([C3[:,0:5],C4[:,0:5],EOG[:,0:5],EMG[:,0:5]], axis=1) #2 EEG, 1 EOG, 1 EMG
            np.save(file = feat_path + f'features_x1_{subj}_{cond}.npy', arr = x1)
            x2 = np.concatenate([C3[:,0:5],EOG[:,0:5],EMG[:,0:5]], axis=1)           #1 EEG, 1 EOG, 1 EMG
            np.save(file = feat_path + f'features_x2_{subj}_{cond}.npy', arr = x2)
            x3 = np.concatenate([C3[:,0:5],C4[:,0:5],EOG[:,0:5]], axis=1)            #2 EEG, 1 EOG
            np.save(file = feat_path + f'features_x3_{subj}_{cond}.npy', arr = x3)
            x4 = np.concatenate([C3[:,0:5],C4[:,0:5],EMG[:,0:5]], axis=1)            #2 EEG, 1 EMG
            np.save(file = feat_path + f'features_x4_{subj}_{cond}.npy', arr = x4)
            x5 = np.concatenate([C3[:,0:5],C4[:,0:5]], axis=1)                       #2 EEG
            np.save(file = feat_path + f'features_x5_{subj}_{cond}.npy', arr = x5)
            x6 = np.concatenate([C3[:,0:5]], axis=1)                                 #1 EEG
            np.save(file = feat_path + f'features_x6_{subj}_{cond}.npy', arr = x6)   
            
            # concatenate over features
            x1_ = features.set_index(['chan','epoch']).loc[['C3', 'C4', 'EOG', 'EMG']].unstack(level='chan')
            x1_.to_csv(feat_path + f'features_x1_new_{subj}_{cond}.csv')   
            x1_ = x1_.to_numpy()
            x2_ = features.set_index(['chan','epoch']).loc[['C3', 'EMG', 'EOG']].unstack(level='chan')
            x2_.to_csv(feat_path + f'features_x2_new_{subj}_{cond}.csv')   
            x2_ = x2_.to_numpy()
            x3_ = features.set_index(['chan','epoch']).loc[['C3', 'C4', 'EOG']].unstack(level='chan')
            x3_.to_csv(feat_path + f'features_x3_new_{subj}_{cond}.csv')   
            x3_ = x3_.to_numpy() 
            x4_ = features.set_index(['chan','epoch']).loc[['C3', 'C4', 'EMG']].unstack(level='chan')
            x4_.to_csv(feat_path + f'features_x4_new_{subj}_{cond}.csv')   
            x4_ = x4_.to_numpy()
            x5_ = features.set_index(['chan','epoch']).loc[['C3', 'C4']].unstack(level='chan')
            x5_.to_csv(feat_path + f'features_x5_new_{subj}_{cond}.csv')   
            x5_ = x5_.to_numpy() 
            x6_ = features.set_index(['chan','epoch']).loc[['C3']].unstack(level='chan')
            x6_.to_csv(feat_path + f'features_x6_new_{subj}_{cond}.csv')   
            x6_ = x6_.to_numpy()  
        
        else:
            sls = pickle.load(open(feat_path + f'yasa_{subj}_{cond}.p', "rb")) 
            
            # combine EEG, EOG, EMG
            x1 = np.load(file = feat_path + f'features_x1_{subj}_{cond}.npy')
            x2 = np.load(file = feat_path + f'features_x2_{subj}_{cond}.npy')
            x3 = np.load(file = feat_path + f'features_x3_{subj}_{cond}.npy')
            x4 = np.load(file = feat_path + f'features_x4_{subj}_{cond}.npy')
            x5 = np.load(file = feat_path + f'features_x5_{subj}_{cond}.npy')
            x6 = np.load(file = feat_path + f'features_x6_{subj}_{cond}.npy')

            # # concatenate over features
            # x1_ = pd.read_csv(feat_path + f'features_x1_new_{subj}_{cond}.csv')   
            # x1_ = x1_.to_numpy()
            # x2_ = pd.read_csv(feat_path + f'features_x2_new_{subj}_{cond}.csv')   
            # x2_ = x2_.to_numpy()
            # x3_ = pd.read_csv(feat_path + f'features_x3_new_{subj}_{cond}.csv')   
            # x3_ = x3_.to_numpy()
            # x4_ = pd.read_csv(feat_path + f'features_x4_new_{subj}_{cond}.csv')   
            # x4_ = x4_.to_numpy()
            # x5_ = pd.read_csv(feat_path + f'features_x5_new_{subj}_{cond}.csv')   
            # x5_ = x5_.to_numpy()
            # x6_ = pd.read_csv(feat_path + f'features_x6_new_{subj}_{cond}.csv')   
            # x6_ = x6_.to_numpy()
            
            x1_ = features.set_index(['chan','epoch']).loc[['C3', 'C4', 'EOG', 'EMG']].unstack(level='chan')
            x1_ = x1_.to_numpy()
            x2_ = features.set_index(['chan','epoch']).loc[['C3', 'EMG', 'EOG']].unstack(level='chan')
            x2_ = x2_.to_numpy()
            x3_ = features.set_index(['chan','epoch']).loc[['C3', 'C4', 'EOG']].unstack(level='chan')
            x3_ = x3_.to_numpy() 
            x4_ = features.set_index(['chan','epoch']).loc[['C3', 'C4', 'EMG']].unstack(level='chan')
            x4_ = x4_.to_numpy()
            x5_ = features.set_index(['chan','epoch']).loc[['C3', 'C4']].unstack(level='chan')  
            x5_ = x5_.to_numpy() 
            x6_ = features.set_index(['chan','epoch']).loc[['C3']].unstack(level='chan')
            x6_ = x6_.to_numpy() 

        y_pred_yasa = yasa.hypno_str_to_int(sls.predict())
        yasa_score = sls.predict_proba().to_numpy()
    
        # predict based on online classification selection
       
        # # Sort the model names
        # sorted_model_names = natsorted(model_names, key=lambda x: x.split('_')[-1])
        # # Iterate over the sorted model names
        # for idx, model_name in enumerate(sorted_model_names): #sorted(model_names):
        #     # for idx, x in enumerate(feature_vectors):     
        #     print(model_name, idx)
        #     # predict for each feature vector x
        #     exec("y_pred_" + model_name + "= " + model_name + f".predict(x{int(model_name[-1])}_)")
        #     output_dict[f"y_pred_{model_name}"] = y_pred
        #     # predict probabilities for each feature vector x
        #     exec("y_score_" + model_name + "= " + model_name + f".predict_proba(x{int(model_name[-1])}_)")
        #     output_dict[f"y_score_{model_name}"] = y_score   
            
        # Sort the model names
        sorted_model_names = natsorted(model_names, key=lambda x: x.split('_')[-1])
        # Initialize the output dictionary
        output_dict_pred = {}
        output_dict_score = {}
        # Iterate over the sorted model names
        for idx, model_name in enumerate(sorted_model_names):
            #print(model_name, idx)
            # Declare y_pred and y_score variables
            y_pred_ = None
            y_score_ = None
            # predict for each feature vector x
            exec("y_pred_ = " + model_name + f".predict(x{int(model_name[-1])}_)")
            output_dict_pred[f"y_pred_{model_name}"] = y_pred_
            # predict probabilities for each feature vector x
            exec("y_score_ = " + model_name + f".predict_proba(x{int(model_name[-1])}_)")
            output_dict_score[f"y_score_{model_name}"] = y_score_

        if len(y_pred_) == len(hypnogram):
            y_pred1 = rf1.predict(x1)
            y_score1 = rf1.predict_proba(x1) 
            y_pred2 = rf2.predict(x2)
            y_score2 = rf2.predict_proba(x2) 
            y_pred3 = rf3.predict(x3)            
            y_score3 = rf3.predict_proba(x3) 
            y_pred4 = rf4.predict(x4)            
            y_score4 = rf4.predict_proba(x4) 
            y_pred5 = rf5.predict(x5)            
            y_score5 = rf5.predict_proba(x5) 
            y_pred6 = rf6.predict(x6)            
            y_score6 = rf6.predict_proba(x6) 
                       
            # Sleep stage names
            event_id ={'Wake':0,
                       'Stage 1':1,
                       'Stage 2':2,
                       'Stage 3':3,
                       'REM':4}
            target_names=event_id.keys()
            
            # # Prediction model key
            # model_id ={'y_pred0_rf': '2 EEG, 1 EOG, 1 EMG',
            #            'y_pred0_lgbm': '2 EEG, 1 EOG, 1 EMG',
            #            'y_pred0_nn': '2 EEG, 1 EOG, 1 EMG',
            #            'y_pred1':'2 EEG, 1 EOG, 1 EMG',
            #            'y_pred2':'1 EEG, 1 EOG, 1 EMG',    ,
            #            'y_pred3':'2 EEG, 1 EOG',
            #            'y_pred4':'2 EEG, 1 EMG',
            #            'y_pred5':'2 EEG',
            #            'y_pred6':'1 EEG',
            #            'y_pred_yasa': '1 EEG, 1 EMG, 1 EOG', 
            #           }
            # model_names=model_id.keys()
            
            # Current hypnogram becomes test
            y_test = hypnogram
            
            # accuracy report, confusion matrix, classification reports
            y_preds = [output_dict_pred.get(key) for key in output_dict_pred.keys()]
            # Append the additional arrays to the grouped list
            y_preds.extend([y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6, y_pred_yasa])
            # do same for scores
            y_scores = [output_dict_score.get(key) for key in output_dict_score.keys()]
            y_scores.extend([y_score1, y_score2, y_score3, y_score4, y_score5, y_score6, yasa_score])
            for idx, (y_pred, y_score) in enumerate(zip(y_preds, y_scores)):
                acc = accuracy_score(y_test, y_pred)
                print("Accuracy score: {}".format(acc))
                #confusion = confusion_matrix(y_test, y_pred)
                #print(confusion_matrix(y_test, y_pred))
                if len(np.unique(y_pred)) == 5:
                    report = classification_report(y_test, y_pred, target_names=event_id.keys())
                    print(report)
                    N3_recall = recall_score(y_test, y_pred, average=None)[3]
                else:
                    N3_recall = np.nan
                    #break
                    pass
                    
                
                cm = confusion_matrix(y_test, y_pred, normalize='true')
                np.set_printoptions(precision=2)
                print(f'Confusion matrix: \n {cm}' + '\n')
                
                # Compute interrater reliability
                inter_agreement = cohen_kappa_score(y_test, y_pred)
                print(f'The inter-rate agreement is K = {inter_agreement}' + '\n')
                
                # Compute Matthews Correlation Coeffient
                phi = matthews_corrcoef(y_test, y_pred)
                print(f'The Phi coefficient is {phi}' + '\n')
                
                # Compute weighted f1 score
                f1 = f1_score(y_test, y_pred, average='weighted')
                print(f'The F1 score is {f1}' + '\n')          
                
                # append all
                accArrays.append([subj, cond, y_test, y_pred, 
                                  y_score, inter_agreement, phi, f1, N3_recall, idx])
                
                print('Passed all steps')
                
                # plt.figure()
                # plot_confusion_matrix(path = new_path, cm=cm, target_names=target_names, title=f'Confusion matrix - {"_".join(fname.split("_")[0:2])} - {model_id[idx]}', save=True, 
                #                       save_name = model_id[idx] + "_".join(fname.split("_")[0:2]))
                # plot_multiclass_ROC(level = 'subject', path = new_path, y_test_all = y_test, y_score_all = y_score, save_name = model_id[idx] + "_".join(fname.split("_")[0:2]))
                # plt.close()
                # sns.heatmap(cm, cmap=plt.cm.Blues,square=True, annot=True, cbar=True)
                # plt.xlabel('predicted value')
                # plt.ylabel('true value');
                
            del y_scores, y_preds
                
        else:
            pass
            
    return accArrays

#%% 

run = input('Do you wish to restart the classifier validation analysis? ')
if run == 'yes':
    df_adaption = pd.DataFrame(classifier_validation(session='adaption', loaded=True), 
                               columns=['Subject','Condition','Test hypnograms','Pred hypnograms',
                                         'Pred probabilities','Cohens Kappa', 'Phi',
                                         'F1-score','N3_recall','model idx'])
    df_adaption['Source'] = 'Adaption'
    df_experimental = pd.DataFrame(classifier_validation(session='experimental'), 
                                   columns=['Subject','Condition','Test hypnograms','Pred hypnograms',
                                            'Pred probabilities','Cohens Kappa','Phi',
                                            'F1-score','N3_recall','model idx'])
    df_experimental['Source'] = 'Experimental'
    df = pd.concat([df_adaption, df_experimental], ignore_index=True)
    save_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/RF_results_.p'
    pickle.dump(df, open(save_path, "wb"))  
else:
    #df = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/Classifier_validation/RF_results.p', 'rb'))
    #df.to_csv(r'/media/administrator/data/Study_1_data/Statistics/Classifier_validation/RF_results.csv')
    
    #df = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/Classifier_validation/RF_results_combined.p', 'rb'))
    df = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/Classifier_validation/Results.p', 'rb'))
    
#%%

def group_classification_validation(df, model_id, event_id, target_names):     
    new_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/All/'
    
    y_tests_all = []
    for idx in range(len(model_id)):
        y_tests_all.append(np.hstack([df[df['model idx']==idx]['Test hypnograms'].to_numpy()[i] for i in range(df[df['model idx']==idx]['Test hypnograms'].shape[0])]))
                   
    y_preds_all = []
    for idx in range(len(model_id)):
        y_preds_all.append(np.hstack([df[df['model idx']==idx]['Pred hypnograms'].to_numpy()[i] for i in range(df[df['model idx']==idx]['Pred hypnograms'].shape[0])]))        
       
    y_scores_all = []
    for idx in range(len(model_id)):
        y_scores_all.append(np.vstack([df[df['model idx']==idx]['Pred probabilities'].to_numpy()[i] for i in range(df[df['model idx']==idx]['Pred probabilities'].shape[0])]))            
                   
    y_cohen_ks_all = []
    for idx in range(len(model_id)):
        y_cohen_ks_all.append(np.hstack([df[df['model idx']==idx]['Cohens Kappa'].to_numpy()[i] for i in range(df[df['model idx']==idx]['Cohens Kappa'].shape[0])]))        
       
    y_phis_all = []
    for idx in range(len(model_id)):
        y_phis_all.append(np.hstack([df[df['model idx']==idx]['Phi'].to_numpy()[i] for i in range(df[df['model idx']==idx]['Phi'].shape[0])]))        
       
    
    for idx, (y_pred_all, y_test_all, y_score_all, y_cohen_k_all, y_phi_all) in enumerate(zip(y_preds_all, y_tests_all, y_scores_all, y_cohen_ks_all, y_phis_all)):    
        print("Model: {}".format(model_id[idx]))
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
        # plot_confusion_matrix(path = new_path, cm=cm, target_names=target_names, title=f'Confusion matrix - All Subjects - {model_id[idx]}', 
        #                       save=True, save_name = model_id[idx] + '_all_subj')
        plot_multiclass_ROC(level = 'group level', path = new_path, y_test_all = y_test_all, y_score_all = y_score_all, save_name = model_id[idx] + '_all_subj')
        
        plt.figure()
        sns.heatmap(cm, cmap=plt.cm.Blues,square=True, annot=True, cbar=True, xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=360)
        plt.title(f'Confusion matrix - All Subjects - {model_id[idx]}')
        plt.tight_layout()
        plt.savefig(new_path + f"confusion_matrix_{model_id[idx] + '_all_subj'}")
        
    plt.close('all')
    
    sns.violinplot(x = 'model idx', y = 'Cohens Kappa', data = df, cut=1)
    plt.title('Inter-rater reliability of RF models')
    sns.despine()
    plt.savefig(new_path + 'cohens_kappa_all_models.png')
    
    plt.close('all')
        

#%%
# model path
model_path = "/media/administrator/data/cfs/models/"   
# compile sorted list of model names
model_id = [model_name.split("/")[-1].split(".")[0] for model_name in glob.glob(model_path + "*.p")]
sorted_model_names = natsorted(model_id, key=lambda x: x.split('_')[-1])
sorted_model_names.extend([f'rf_model_{idx}_cfs' for idx in range(1,7)])
sorted_model_names.extend(['yasa_model'])

# Prediction model key
# model_id ={'y_pred0_rf': '2 EEG, 1 EOG, 1 EMG',
#            'y_pred0_lgbm': '2 EEG, 1 EOG, 1 EMG',
#            'y_pred0_nn': '2 EEG, 1 EOG, 1 EMG',
#            'y_pred1':'2 EEG, 1 EOG, 1 EMG',
#            'y_pred2':'1 EEG, 1 EOG, 1 EMG',
#            'y_pred3':'2 EEG, 1 EOG',
#            'y_pred4':'2 EEG, 1 EMG',
#            'y_pred5':'2 EEG',
#            'y_pred6':'1 EEG',
#            'y_pred_yasa': '1 EEG, 1 EMG, 1 EOG'}

event_id ={'Wake':0,
           'Stage 1':1,
           'Stage 2':2,
           'Stage 3':3,
           'REM':4}
target_names=event_id.keys()
        
exp_df = df[df.Condition != 'adaption'].reset_index(drop=True)
group_classification_validation(exp_df, sorted_model_names, event_id, target_names)
