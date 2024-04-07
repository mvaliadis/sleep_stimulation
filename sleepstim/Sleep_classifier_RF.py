#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:24:59 2020

@author: mvaliadis
"""

from sklearn.impute import SimpleImputer, KNNImputer
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, accuracy_score, matthews_corrcoef, auc,
                             roc_curve, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.feature_selection import f_classif, RFECV
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from lightgbm import LGBMClassifier
import xgboost
from itertools import cycle
import pandas as pd
#import pingouin as pg
import seaborn as sns
import yasa
sns.set(font_scale=1.2)
from os import chdir as cd
from os import listdir
from openTSNE import TSNE
import shap
# cd('/home/administrator/Documents/Physionet_data')
cd('/media/administrator/data/cfs/pre_proc')

#from sklearn.model_selection import cross_val_score

fls = sorted(listdir())
files = list(set([f.split('_')[0] for f in fls]))

#%%

def preprocess_classification(files, use_ecg=True, kind='pandas'):
    if kind == 'numpy':
        allArrays = []
        stageArrays = []
        partArrays = []
        for idx, fname in enumerate(files):
            # import data arrays of features
            datArray = pickle.load(open(fname + "_allArrays_numpy.p", "rb"))
            print(idx, fname, datArray.shape.max(), 'epochs')
                
            # import data array of class labels (hypnograms )
            stageArray = pickle.load(open(fname + "_stageArrays.p", "rb"))
                
            # append all feature arrays
            allArrays.append(datArray)
            stageArrays.append(stageArray)
            partArrays.append([fname.split('-')[-1]]*len(stageArray))
        
        # combine feature, subject, and stage info   
        df = np.concatenate(allArrays)
        stages = np.concatenate(stageArrays)
        parts = np.concatenate(partArrays)
        
        return df, stages, parts
        
    elif kind =='pandas':
        allArrays = []
        for idx, fname in enumerate(files):
            # import data arrays of features
            datArray = pickle.load(open(fname + "_allArrays_df.p", "rb"))
            print(idx, fname, datArray.epoch.max(), 'epochs')
            
            # import data array of class labels (hypnograms )
            stageArray = pickle.load(open(fname + "_stageArrays.p", "rb"))
        
            # add stage label to df
            datArray['stage'] = np.ones(len(datArray))*np.nan
            for ch, df_idx in datArray.groupby(['chan']):
                datArray.loc[df_idx.index, 'stage'] = stageArray
            # also add str label 
            datArray['stage_lab'] = datArray['stage'].replace({0:'W', 1:'N1', 2: 'N2',
                                                               3: 'N3', 4: 'REM'})
            
            # import ecg features 
            if use_ecg:
                hrvArray = pickle.load(open(fname + "_allArrays_hrv_df.p", "rb"))
                print(idx, fname, hrvArray.epoch.max(), 'ecg epochs')
                hrvArray['stage'] = datArray['stage']
                hrvArray['stage_lab'] = datArray['stage_lab']
                # append all feature arrays
                allArrays.append([datArray.reset_index(drop=True), 
                                  hrvArray.reset_index(drop=True)])
                
            else:
                # append all feature arrays
                allArrays.append(datArray.reset_index(drop=True))
            
        if use_ecg:
            # combine feature, subject, and stage info   
            df_eeg = pd.concat([allArrays[i][0] for i in range(len(allArrays))])
            # drop mastoids
            df_eeg = df_eeg.set_index('chan').drop(index=['M1','M2']).reset_index()
            
            # combine feature, subject, and stage info   
            df_ecg = pd.concat([allArrays[i][1] for i in range(len(allArrays))])
            # drop columns with too many nans 
            null_counts = df_ecg.isnull().sum()
            null_percentages = null_counts / len(df_ecg)
            columns_to_drop = null_percentages[null_percentages > .10].index.tolist()
            # drop the columns
            df_ecg.drop(columns_to_drop, axis=1, inplace=True)
            
            return df_eeg, df_ecg
        
        else:
            # combine feature, subject, and stage info   
            df = pd.concat(allArrays)
            # drop mastoids
            df = df.set_index('chan').drop(index=['M1','M2']).reset_index()
 
            # # convert to np array
            # data_all = []
            # for (sub, ch, stage), df_ in df.groupby(['subject','chan','stage']):
            #     print(sub, ch, stage)
            #     data_all.append(np.expand_dims(df_[df_.columns[1:-4]].to_numpy(),1))
            #     print(len(df_))
                
            # dfs = np.concatenate(data_all, axis=1)
            # nepochs, nchans, nfeats = np.shape(dfs)
            # data_all = dfs.reshape(nepochs, nchans*nfeats, order='F')
        
        return df
    
df_eeg, df_ecg = preprocess_classification(files, use_ecg=True, kind='pandas')

#%%
def plot_feature_preprocess(df, df_ecg=None):
    df_ecg['chan'] = ['ecg']*len(df_ecg)
    for df, label in zip((df, df_ecg),('eeg','ecg')):
        print(label)
        # plot this by channel 
        df_mean = df.groupby(['subject','chan','stage','stage_lab','epoch']).mean().reset_index()
        for c, df_ in df_mean.groupby('chan'):
            # Create a color mapping based on the unique values of the x-variable
            color_mapping = {x: f'C{i}' for i, x in enumerate(df_['stage_lab'].unique())}
            df_['color'] = df_['stage_lab'].map(color_mapping)
    
            ## Plotting
            if label == 'eeg':
                fig, axs = plt.subplots(3, 7, figsize=(25, 25), sharey=False)
            elif label == 'ecg':
                fig, axs = plt.subplots(6, 11, sharey=False)
            for idx, (col, ax) in enumerate(zip(df_.columns[5:-1], axs.flatten())):       
                #ax.scatter(df_.stage_lab, df_[col], alpha=.5, s=.8, color=df_.color.to_numpy())   
                sns.scatterplot(ax=ax, x='stage_lab', y=col, hue='stage_lab', #hue='chan',
                                data=df_, alpha=.8, s=1)
                ax.set_ylabel(f'{col}')
                #ax.set_xlabel('Sleep Stage')
                time.sleep(0.1)
                
            plt.tight_layout()
            plt.savefig(f'/media/administrator/data/cfs/figures/raw_features_{label}_{c}.png')
            plt.close('all')
            
            # # Extract sorted r and rho values
            # # get correlation between each feature and sleep stages 
            # r = np.corrcoef(df_.stage, np.squeeze(df_[col])).min().round(2)
            # print(f'Pearson correlation between feature {col} and labels is: {r}')
            # rho = pg.corr(y=df_.stage, x=np.squeeze(df_[col]), 
            #               method="spearman")['r'][0].round(2)
            # print(f'Spearman correlation between feature {col} and labels is: {rho}')
            
            if label == 'ecg':                
                # Replace infinite values with NaN
                X = df_[df_.columns[5:-1]].replace([np.inf, -np.inf], np.nan)
                
                # Create a imputer object
                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                #imputer = KNNImputer(n_neighbors=5)
                
                # Impute NaNs with KNN imputation
                X_imputed = imputer.fit_transform(X)
              
                # Convert the numpy array back to a DataFrame
                df_imputed = pd.DataFrame(X_imputed, columns=df_[df_.columns[5:-1]].columns)
                
                # Calculate f-values
                fvals = pd.Series(f_classif(X=df_imputed, y=df_.stage_lab)[0],
                                  index=df_imputed.columns).sort_values()

            else:
                # Extract sorted F-values
                fvals = pd.Series(f_classif(X=df_[df_.columns[5:-1]], 
                                            y=df_.stage_lab)[0], 
                                  index=df_.columns[5:-1]
                                  ).sort_values()
            
            # Plot best features
            plt.figure(figsize=(6, 6))
            sns.barplot(y=fvals.index, x=fvals, palette='RdYlGn')
            plt.xlabel('F-values')
            plt.xticks(rotation=20);
            plt.tight_layout()
            plt.savefig(f'/media/administrator/data/cfs/figures/raw_features_{c}.png')
            plt.close('all')

def plot_feature_hypnogram(df, df_ecg=None, chan='C3'):
    df_ecg['chan'] = ['ecg']*len(df_ecg)
    # Plot hypnogram and feature (channel x subject)  
    for (ch, sub), _ in df.groupby(['chan','subject']):
        if ch == chan:
            df_ = df.set_index(['subject', 'chan']).loc[sub, ch]
            for idx, col in enumerate(df.columns[2:-3]):
                ex = df_[col].to_numpy()
                hypno = df_.stage.astype(int).to_numpy()
                times = np.sort(df_.epoch.to_numpy()/120)
            
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                
                # Plot the hypnogram
                yasa.plot_hypnogram(hypno, ax=ax1)
                
                # Plot the non-linear feature
                ax2.plot(times, ex)
                ax2.set_ylabel(f'{col}')
                ax2.set_xlabel('Time [hrs]')
                ax2.set_xlim(0, times[-1]);
                
                # save figs
                plt.savefig(f'/media/administrator/data/cfs/figures/subject/{col}_hypno_{sub}_{ch}.png')
                plt.close('all')
                
                del df_

def plot_tsne(x, y, ax=None, title=None, draw_legend=True, draw_centers=False,
              draw_cluster_labels=False, colors=None, legend_kwargs=None, 
              label_order=None, **kwargs):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

def classifier_with_rfe(df_eeg, model='lightGBM'):
    # Step 0: Create new dataframe and impute nans
    rfe_df = df_eeg.set_index(['chan', 'set', 'subject','epoch','stage']).loc[(['C3', 'C4', 'EOG', 'EMG']),
                                                                              df_eeg.columns[2:-4]].unstack(level='chan') 
    rfe_df.columns = list([f"{x[0]}_{x[1]}" for x in rfe_df.columns])
    
    # Impute NaN values with mean again....
    rfe_df = rfe_df.fillna(rfe_df.mean())

    # Step 1: Create an instance of the decision tree model
    if model == 'Random_forest':
        clf = RandomForestClassifier(n_estimators=100, oob_score=True, 
                                     random_state=42)
    elif model == 'lightGBM':
        params = dict(
            boosting_type='gbdt',
            n_estimators=400,
            max_depth=5,
            num_leaves=90,
            colsample_bytree=0.5,
            importance_type='gain',
            )
        clf = LGBMClassifier(**params)
    elif model == 'XGBoost':
        clf = xgboost.XGBClassifier()
    
    # Step 2: Create an instance of the Recursive Feature Elimination with Cross-Validation (RFECV)
    rfe = RFECV(estimator=clf,
                cv=GroupShuffleSplit(n_splits=100, train_size=0.8, random_state=42),
                scoring="f1_weighted",
                min_features_to_select=1,
                n_jobs=-1)

    # Step 3: Apply RFECV to perform feature selection and train the RF model
    rfe.fit(rfe_df, rfe_df.reset_index()['stage'].to_numpy(), 
            groups=rfe_df.reset_index()['subject'].to_numpy())    
      
    # Step 4: Access the selected features
    selected_features = rfe_df.columns[rfe.support_]

    # Step 5: Optionally, access the feature ranking
    feature_ranking = rfe.ranking_

    return rfe, selected_features, feature_ranking

def plot_rfe(rfe):
    n_scores = len(rfe.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Weighted F1-scores")
    plt.errorbar(
        range(1, n_scores + 1),
        rfe.cv_results_["mean_test_score"],
        yerr=rfe.cv_results_["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()

def calculate_vif(df, features):   
    ## Impute data first
    # Create a imputer object
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                                              
    # Impute NaNs with simple imputation
    X_imputed = imputer.fit_transform(df[df.columns[2:-4]])
                                          
    # Convert the numpy array back to a DataFrame
    df_imputed = pd.DataFrame(X_imputed, columns=df.columns[2:-4])
               
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for idx, feature in enumerate(features):

        # extract all the other features you will regress against
        X = [f for f in features if f != feature]        
        X, y = df_imputed[X], df_imputed[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)                
        print(r2)
        
        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1/(tolerance[feature])
        
        print(f'VIF for feature: {feature} is {vif[feature]}')
        
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})

#df_vif = calculate_vif(df_eeg, features=df_eeg.columns[2:-4])

def smote(X, y, method='SMOTE'):
    from collections import Counter
    if method=='SMOTE':
        from imblearn.over_sampling import SMOTE
        # Instantiate SMOTE object
        smote = SMOTE(random_state=42)
        # Resample training data using SMOTE
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(sorted(Counter(y_resampled).items()))
    elif method=='SMOTE-ENN':
        from imblearn.combine import SMOTEENN
        # Instantiate SMOTE-ENN object
        smote_enn = SMOTEENN(random_state=42)
        # Resample training data using SMOTE-ENN
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        print(sorted(Counter(y_resampled).items()))
    elif method=='SMOTE-Tomek':
        from imblearn.combine import SMOTETomek
        # Instantiate SMOTE-Tomek object
        smote_tomek = SMOTETomek(random_state=42)
        # Resample training data using SMOTE-Tomek
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
        print(sorted(Counter(y_resampled).items()))
    
    return X_resampled, y_resampled 

def imputation(X_feat, method='simple'):
    if method=='knn':
        # =============================================================================
        # NAN correction/interpolation (K-nearest neighbor): The KNNImputer class provides
        # imputation for filling in missing values using the k-Nearest Neighbors approach.
        # see ref:
        #   Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor Hastie,
        #   Robert Tibshirani, David Botstein and Russ B. Altman, Missing value estimation
        #   methods for DNA microarrays, BIOINFORMATICS Vol. 17 no. 6, 2001 Pages 520-525.
        # https://scikit-learn.org/stable/modules/impute.html#ol2001
        # =============================================================================
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights="uniform")
    elif method=='simple':
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')        
    return imputer.fit_transform(X_feat)
 
#%%        
def train_test_feature_split(df, df_ecg=None, kind='pandas', pca=False, use_smote=False):
    # Define training / testing
    groups = df['subject']

    #Split the data into training and testing sets using GroupShuffleSplit:
    gss = GroupShuffleSplit(n_splits=100, train_size=0.8, random_state=42)   
    for train_idx, test_idx in gss.split(df, groups=groups):
        # perform split
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]
        #print("GROUP TRAIN:", train_data.subject.unique(),\
        #      "GROUP TEST:", test_data.subject.unique())
        if df_ecg is not None:
            ecg_train = df_ecg[df_ecg.subject.isin(train_data.subject.unique())]
            ecg_test = df_ecg[df_ecg.subject.isin(test_data.subject.unique())]

    # assign label to training/test sets in df
    df.loc[train_idx, "set"] = "training"
    df.loc[test_idx, "set"] = "testing"
        
    if kind=='pandas':
        return df
    
    elif kind=='numpy':   
        # X_train = df.set_index(['chan','set']).loc[(['C3','C4','EOG'],['training']),:][df.columns[5:-4]].to_numpy()    
        # X_test = df.set_index(['chan','set']).loc[(['C3','C4','EOG'],['testing']),:][df.columns[5:-4]].to_numpy() 
        # X_train = df.set_index(['chan', 'set', 'subject','epoch']).loc[(['C3', 'C4', 'EOG', 'EMG'], ['training']), 
        #                                                                df.columns[2:-4]].unstack(level='chan').to_numpy()
        # X_test = df.set_index(['chan', 'set', 'subject','epoch']).loc[(['C3', 'C4', 'EOG', 'EMG'], ['testing']), 
        #                                                                df.columns[2:-4]].unstack(level='chan').to_numpy()
        # y_train = df.set_index(['chan', 'set', 'subject','epoch']).loc[(['C3'], ['training']),
        #                                                                df.columns[-3]].unstack(level='chan').astype(int).to_numpy().squeeze()
        # y_test = df.set_index(['chan', 'set', 'subject','epoch']).loc[(['C3'], ['testing']), 
        #                                                               df.columns[-3]].unstack(level='chan').astype(int).to_numpy().squeeze()
        
        unique_chans = df.chan.unique()
        # X_train = df.set_index(['chan','set']).loc[(['C3','C4','EOG'],['training']),:][df.columns[5:-4]].to_numpy()    
        # X_test = df.set_index(['chan','set']).loc[(['C3','C4','EOG'],['testing']),:][df.columns[5:-4]].to_numpy() 
        X_train = df.set_index(['chan', 'set', 'subject','epoch']).loc[(unique_chans, ['training']), 
                                                                       df.columns[2:-4]].unstack(level='chan').to_numpy()
        X_test = df.set_index(['chan', 'set', 'subject','epoch']).loc[(unique_chans, ['testing']), 
                                                                       df.columns[2:-4]].unstack(level='chan').to_numpy()
        y_train = df.set_index(['chan', 'set', 'subject','epoch']).loc[(unique_chans[0], ['training']),
                                                                       df.columns[-3]].unstack(level='chan').astype(int).to_numpy().squeeze()
        y_test = df.set_index(['chan', 'set', 'subject','epoch']).loc[(unique_chans[0], ['testing']), 
                                                                      df.columns[-3]].unstack(level='chan').astype(int).to_numpy().squeeze()


        if df_ecg is not None:
            X_train = np.concatenate([X_train, ecg_train[ecg_train.columns[0:-4]].to_numpy()], axis=1)
            X_test = np.concatenate([X_test, ecg_test[ecg_test.columns[0:-4]].to_numpy()], axis=1)
        
        # Imputation if necessary   
        # Replace infinite values with NaN
        X_train = np.nan_to_num(X_train, nan=np.nan, posinf=np.nan, neginf=np.nan)
        X_test = np.nan_to_num(X_test, nan=np.nan, posinf=np.nan, neginf=np.nan)
        X_train = imputation(X_train, method='simple')
        X_test = imputation(X_test, method='simple')
        
        # SMOTE-ing
        if use_smote:
            X_train, y_train = smote(X_train, y_train, method='SMOTE')
            
        # PCA-ing
        if pca:
            pca = PCA(n_components=20)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

    return X_train, X_test, y_train, y_test  
    
#df = train_test_feature_split(df_eeg, kind='pandas', use_smote=False)
X_train, X_test, y_train, y_test = train_test_feature_split(df_eeg, df_ecg=None, 
                                                            pca=False,
                                                            kind='numpy', 
                                                            use_smote=False)

#%%
def random_forest_classification(X_train, X_test, y_train, y_test, df=None,
                                 feature_selection=True, selection_method='rfe',
                                 standarize=False, class_weight=False):  
    if feature_selection:
        if selection_method == 'univariate':
            # univariate feature selection based on F-values
            from sklearn.feature_selection import SelectKBest
            # Select top 20 features using f score
            selector = SelectKBest(f_classif, k=20)
            selector.fit(X_train, y_train)
            
            # Transform the data to keep only the selected features
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
        
        if selection_method == 'rfe':
            # recursive feature elimination 
            from sklearn.feature_selection import RFE
            # random state for reproducibility
            rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
            
            # perform RFE to select the top 10 features
            rfe = RFE(estimator=rf, n_features_to_select=20, step=1)
            rfe.fit(X_train, y_train)
            
            # transform the data to include only the selected features
            X_train = rfe.transform(X_train)
            X_test = rfe.transform(X_test)
            
            del rf
    
    if standarize:
        from sklearn.preprocessing import StandardScaler  
        scaler = StandardScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        # apply same transformation to test data
        X_test = scaler.transform(X_test)  
        
    # See the following for more info re: RF modeling
    # https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#remarks
    # also: L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
    event_id = {'Wake': 0,
                'Stage 1': 1,
                'Stage 2': 2,
                'Stage 3': 3,
                'REM': 4}
    
    params = dict(n_estimators=1000, max_depth=20, min_samples_split=2,
                  min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                  random_state=42, oob_score=True)
    
    # add class weight, if desired
    if class_weight:
        params['class_weight'] = {1: 2.2, 2: 1, 3: 1.2, 4: 1.4, 0: 1}
                                #{'N1': 2.2, 'N2': 1, 'N3': 1.2, 'R': 1.4, 'W': 1}
   
    # random state for reproducibility
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    #rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    #y_score = rf.predict_proba(X_test)  # modeling score
    y_pred = rf.predict(X_test)
    
    # Learn to predict each class against the other, see:
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    # May not be necessary actually....
    # rf = []
    # rf = OneVsRestClassifier(RandomForestClassifier(n_estimators = 100, oob_score = True, random_state=42))
    # rf.fit(X_train, y_train)
    # y_score = rf.predict_proba(X_test) #modeling score
    # y_pred = rf.predict(X_test)

    # accuracy report, confusion matrix, classification reports
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score: {}".format(acc))
    #confusion = confusion_matrix(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    #report = classification_report(y_test, y_pred, target_names=event_id.keys())
    print(classification_report(y_test, y_pred, target_names=event_id.keys()))

    return rf 

def rf_plot_pred_probab(rf, X_test):
    score = rf.predict_proba(X_test)
    # Plot the predicted probability for each sleep stage for each 30-sec epoch of data 
    proba = score#[0:1000]
    #palette=["#99d7f1", "#009DDC", "xkcd:twilight blue", "xkcd:rich purple", "xkcd:sunflower"],
    ax = pd.DataFrame(proba).plot(kind="area", figsize=(10, 5), alpha=0.8, 
                                  stacked=True, lw=0) #, colors=palette)
    ax.set_xlim(0, proba.shape[0])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time (30-sec epoch)")
    plt.legend(frameon=False, bbox_to_anchor=(1, 1))
    plt.tight_layout()

def lightgbm_classification(X_train, y_train, X_test, y_test, class_weight=False):
    from lightgbm import LGBMClassifier
    
    # Define hyper-parameters
    params = dict(
        boosting_type='gbdt',
        n_estimators=400,
        max_depth=5,
        num_leaves=90,
        colsample_bytree=0.5,
        importance_type='gain',
        )
    
    # add class weight, if desired
    if class_weight:
        params['class_weight'] = {1: 2.2, 2: 1, 3: 1.2, 4: 1.4, 0: 1}
                                #{'N1': 2.2, 'N2': 1, 'N3': 1.2, 'R': 1.4, 'W': 1}
    
    # Fit
    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    
    # Metrics
    event_id = {'Wake': 0,
                'Stage 1': 1,
                'Stage 2': 2,
                'Stage 3': 3,
                'REM': 4} 
    
    # prediction    
    y_pred = clf.predict(X_test)
    
    # accuracy report, confusion matrix, classification reports
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score: {}".format(acc))
    #confusion = confusion_matrix(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    #report = classification_report(y_test, y_pred, target_names=event_id.keys())
    print(classification_report(y_test, y_pred, target_names=event_id.keys()))
    
    return clf

def xgboost_classification(X_train, y_train, X_test, y_test):
    # Fit
    clf = xgboost.XGBClassifier()
    clf.fit(X_train, y_train)
    
    # Metrics
    event_id = {'Wake': 0,
                'Stage 1': 1,
                'Stage 2': 2,
                'Stage 3': 3,
                'REM': 4} 
    
    # prediction    
    y_pred = clf.predict(X_test)
    
    # accuracy report, confusion matrix, classification reports
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score: {}".format(acc))
    #confusion = confusion_matrix(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    #report = classification_report(y_test, y_pred, target_names=event_id.keys())
    print(classification_report(y_test, y_pred, target_names=event_id.keys()))
    
    return clf
    

def neural_network_classification(X_train, y_train,  X_test, y_test, standarize=True):
    if standarize:
        from sklearn.preprocessing import StandardScaler  
        scaler = StandardScaler()  
        # Don't cheat - fit only on training data
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        # apply same transformation to test data
        X_test = scaler.transform(X_test)  

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5, 
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    event_id = {'Wake': 0,
                'Stage 1': 1,
                'Stage 2': 2,
                'Stage 3': 3,
                'REM': 4} 
    
    # accuracy report, confusion matrix, classification reports
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score: {}".format(acc))
    #confusion = confusion_matrix(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    #report = classification_report(y_test, y_pred, target_names=event_id.keys())
    print(classification_report(y_test, y_pred, target_names=event_id.keys()))
    
    return clf
 
#%%
## Feature Calibration
# feature mean and variance
feat_mean = np.mean(X_train, axis=0)
feat_var = np.var(X_train, axis=0, ddof=1)
# plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel("Mean")
ax.set_ylabel("Variance")
ax.set_xscale("log")
ax.set_yscale("log")
ax.scatter(feat_mean, feat_var, color="blue", alpha=0.1, label="Real data")
ax.plot(
        sorted(feat_mean),
        sorted(feat_mean),
        color="red",
        label="Poisson Distribution",
        )
ax.set_title("Mean-Variance Relationship")
ax.legend()
plt.tight_layout()

# fano factorization
fano = feat_var / feat_mean
# plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Mean")
ax.set_ylabel("Fano factor")
ax.scatter(feat_mean, fano, color="blue", alpha=0.1, label="Real data")
ax.axhline(1, color="red", label="Poisson distribution")
ax.legend()
ax.set_title("Feature mean vs. Fano factor")
plt.tight_layout()

##
# normalize/scale features 
from sklearn.preprocessing import StandardScaler#, RobustScaler
#from sklearn.preprocessing import MaxAbsScaler #for [-1 to 1] if both value ranges
scaler = StandardScaler() #RobustScaler()  
# Don't cheat - fit only on training data
X_train_scaled = scaler.fit_transform(X_train)  
# apply same transformation to test data
X_test_scaled = scaler.transform(X_test) 


norm_mean = np.average(X_train_scaled, axis=0)
norm_var = np.var(X_train_scaled, axis=0, ddof=1)

fig, ax = plt.subplots(figsize=(6, 4))
# add plot
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Mean")
ax.set_ylabel("Fano factor")
ax.set_title("Normalized features")
ax.scatter(norm_mean, norm_var / norm_mean, color="blue", alpha=0.1)
plt.tight_layout()

# Perform PCA three times: on the resulting matrix as is,
# after np.log2(X+1) transform, and after np.sqrt(X) transform
# perform PCA
norm_fano = norm_var / norm_mean
#fano_crit = np.where(fano >= 3)[0]
fano_crit = np.where(norm_fano >= 3)[0]
#feats_fano = X_train[:, fano_crit]
feats_fano = X_train_scaled[:, fano_crit]

# pca
pca1 = PCA(n_components=35)
pca1.fit(X_train_scaled)
pca1_4tsne = pca1.transform(X_train_scaled)
pca_thres = np.where(np.cumsum(pca1.explained_variance_)>=50)[0][0]

#
feats_fano_log = np.log(1e-10 + np.abs(feats_fano))
pca2 = PCA(n_components=65).fit_transform(feats_fano_log)
feats_fano_sqrt = np.sqrt(feats_fano)
#pca3 = PCA(n_components=65).fit_transform(feats_fano_sqrt)

# tsne
tsne1 = TSNE(perplexity=50, metric="euclidean", n_jobs=-1).fit(pca1_4tsne[:,0:pca_thres]).transform(pca1_4tsne[:,0:pca_thres])
# tsne1 = TSNE(perplexity=100, metric="euclidean", n_jobs=-1).fit(X_train).transform(X_train)
# tsne1 = TSNE(perplexity=100, metric="euclidean", n_jobs=-1).fit(pca1).transform(pca1)

# fig, axs = plt.subplots(figsize=(20, 10))
# axs.scatter(tsne1[:, 0], tsne1[:, 1], s=1, c=y_train)
fig, ax = plt.subplots(figsize=(8, 8))
plot_tsne(tsne1, y_train, alpha=0.25, ax=ax)


tsne2 = TSNE(perplexity=100, metric="euclidean", n_jobs=-1).fit(pca2).transform(pca2)

fig, ax = plt.subplots(figsize=(8, 8))
plot_tsne(tsne2, y_train, alpha=0.25, ax=ax)


#%%
## Leiden clustering
import igraph as ig
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import leidenalg as la

# Construct kNN graph with k=15
A = kneighbors_graph(pca1_4tsne[:,0:5], 30)
# Transform it into an igraph object
sources, targets = A.nonzero()
G = ig.Graph(directed=False)
G.add_vertices(A.shape[0])
edges = list(zip(sources, targets))
G.add_edges(edges)

# Run Leiden clustering
# you can use `la.RBConfigurationVertexPartition` as the partition type
partition = la.find_partition(
    G, la.RBConfigurationVertexPartition, resolution_parameter=1, seed=1
    )

fig, ax = plt.subplots(figsize=(4, 4))
plot_tsne(tsne1, partition.membership, alpha=0.25, ax=ax)


# %% Run & save Models/Training/Testing sets
cd('/media/administrator/data/cfs/models/')

# Random forest
rf = random_forest_classification(X_train, X_test, y_train, y_test, df=None,
                                  feature_selection=False, selection_method='rfe',
                                  standarize=False, class_weight=False)
pickle.dump(rf, open("rf_model_new_cfs.p", "wb"))

# Light GBM 
lgbm = lightgbm_classification(X_train, y_train, X_test, y_test, 
                               class_weight=False)
pickle.dump(lgbm, open("lgbm_model_new_cfs.p", "wb"))

# XGBoost 
XGBoost = xgboost_classification(X_train, y_train, X_test, y_test)
pickle.dump(XGBoost, open("xgboost_model_new_cfs.p", "wb"))

# Neural network   
nn = neural_network_classification(X_train, y_train,  X_test, y_test,
                                   standarize=True)
pickle.dump(nn, open("nn_model_new_cfs.p", "wb"))

#%%
feat_path = '/media/administrator/data/cfs/feature_analysis/'
## Feature importance with shapely values
df_eeg = df_eeg[['chan', 'epoch', 'Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Line noise',
                'subject', 'stage', 'stage_lab']]
# Create Tree Explainer object that can calculate shap values
explainer = shap.TreeExplainer(lgbm) #rf
# train data 
shap_df_train = df_eeg.set_index(['chan', 'set', 'subject','epoch']).loc[(['C3', 'C4', 'EOG', 'EMG'], 
                                                                         ['training']), 
                                                                        df_eeg.columns[2:-4]].unstack(level='chan')
shap_df_train.columns = list([f"{x[0]}_{x[1]}" for x in shap_df_train.columns])
# testing data to explain
shap_df_test = df_eeg.set_index(['chan', 'set', 'subject','epoch']).loc[(['C3', 'C4', 'EOG', 'EMG'], 
                                                                         ['testing']), 
                                                                        df_eeg.columns[2:-4]].unstack(level='chan')
shap_values = explainer.shap_values(shap_df_test)
# f_list = list([f"{x[0]}_{x[1]}" for x in shap_df.columns])
shap_df_test.columns = list([f"{x[0]}_{x[1]}" for x in shap_df_test.columns])
shap.summary_plot(shap_values, feature_names=shap_df_test.columns, max_display=30)

# Aggregate SHAP values
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=1)

# Rank the features for each class
feature_rank = pd.DataFrame(mean_abs_shap_values.T, 
                            columns=['Class {}'.format(i) for i in range(5)], 
                            index=shap_df_test.columns)
feature_rank['Average Importance'] = feature_rank.mean(axis=1)
feature_rank = feature_rank.sort_values(by='Average Importance', ascending=False)
feature_rank.to_csv(feat_path + "shap_rank_lgbm.csv")

#%%
## Recursive feature elimination with cross validation
plt.close('all')
rfe, selected_features, feature_ranking = classifier_with_rfe(df_eeg, model='Random_forest')
plot_rfe(rfe)
pickle.dump(rfe, open(feat_path + "rfe_rf.p", "wb"))

#%%
## Reclassify based on Shaply importances
from sklearn.metrics import f1_score
f1_scores = []
for idx in range(1, len(feature_rank.index)):
    # CV to resplit each time 
    df = train_test_feature_split(df_eeg, df_ecg=None, pca=False, 
                                  kind='pandas', use_smote=False)
    
    shap_df_train = df.set_index(['chan', 'set', 'subject','epoch', 'stage']).loc[(['C3', 'C4', 'EOG', 'EMG'], 
                                                                          ['training']),
                                                                         df.columns[2:-4]].unstack(level='chan')
    shap_df_train.columns = list([f"{x[0]}_{x[1]}" for x in shap_df_train.columns])
    
    shap_df_test = df.set_index(['chan', 'set', 'subject','epoch','stage']).loc[(['C3', 'C4', 'EOG', 'EMG'], 
                                                                         ['testing']),
                                                                        df.columns[2:-4]].unstack(level='chan')
    shap_df_test.columns = list([f"{x[0]}_{x[1]}" for x in shap_df_test.columns])
    
    
    shap_df_train_shap = shap_df_train[feature_rank.index[0:idx]]
    shap_df_test_shap = shap_df_test[feature_rank.index[0:idx]]
    clf = lightgbm_classification(X_train = shap_df_train_shap, 
                                  y_train = shap_df_train_shap.reset_index()['stage'].to_numpy(), 
                                  X_test = shap_df_test_shap,
                                  y_test = shap_df_test_shap.reset_index()['stage'].to_numpy(),
                                  class_weight=False)
    
    # Metrics
    event_id = {'Wake': 0,
                'Stage 1': 1,
                'Stage 2': 2,
                'Stage 3': 3,
                'REM': 4} 
    
    # prediction    
    y_pred = clf.predict(shap_df_test_shap)
    
    # accuracy report, confusion matrix, classification reports
    f1_scores.append([idx, f1_score(y_test, y_pred, average='weighted')])
  
# extract counts & f1-scores    
feat_count = [f1_scores[i][0] for i in range(len(f1_scores))]
f1s = [f1_scores[i][1] for i in range(len(f1_scores))]

# Find the saturation point
max_f1_score = max(f1s)
max_f1_score_index = f1s.index(max_f1_score)
thresholds = np.linspace(0, 1, len(f1_scores))
saturation_threshold = thresholds[max_f1_score_index]

# # Find the plateau point
# threshold_diffs = np.diff(thresholds)
# f1_score_diffs = np.diff(f1s)
# f1_score_gradients = f1_score_diffs / threshold_diffs

# # Find the index of the plateau point where the gradient is close to zero
# plateau_indices = np.where(np.isclose(f1_score_gradients, 0))[0]
# plateau_thresholds = thresholds[plateau_indices]

# # Select the first plateau threshold as the plateau point
# plateau_threshold = plateau_thresholds[0]
# plateau_f1_score = f1_scores[plateau_indices[0]]

# Plot curve
plt.figure()
plt.plot(feat_count, f1s)
plt.xlabel('Ranked Feature Count')
plt.ylabel('Weighted F1-scores')
plt.vlines(max_f1_score_index, ymin=min(f1s), ymax=max(f1s)+.01,
           linestyles='dashed', label='Max F1-score')
plt.grid(True)
plt.tight_layout()
plt.legend()

#%%
## Feature importance with permutation random forest 
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=42)
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
tree_indices = np.arange(0, len(rf.feature_importances_)) + 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))
ax1.barh(tree_indices, rf.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticks(tree_indices)
ax1.set_yticklabels(shap_df_test.columns[tree_importance_sorted_idx])
ax1.set_ylim((0, len(rf.feature_importances_)))
ax2.boxplot(
    result.importances[perm_sorted_idx].T,
    vert=False,
    labels=shap_df_test.columns[perm_sorted_idx],
)
fig.tight_layout()
plt.show()

## Handling Multicollinear Features
# When features are collinear, permutating one feature will have little effect 
# on the models performance because it can get the same information from a correlated 
# feature. One way to handle multicollinear features is by performing hierarchical 
# clustering on the Spearman rank-order correlations, picking a threshold, and 
# keeping a single feature from each cluster. First, we plot a heatmap of the 
# correlated features:
# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X_test).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=shap_df_test.columns, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
plt.grid(False)
plt.show()

# Next, we manually pick a threshold by visual inspection of the dendrogram to 
# group our features into clusters and choose a feature from each cluster to keep, 
# select those features from our dataset, and train a new random forest. The test 
# accuracy of the new random forest did not change much compared to the random forest
# trained on the complete dataset
cluster_ids = hierarchy.fcluster(dist_linkage, .5, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

X_train_sel = X_train[:, selected_features]
X_test_sel = X_test[:, selected_features]

clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
clf_sel.fit(X_train_sel, y_train)
print(f"Accuracy on test data with features"
      f"removed: {clf_sel.score(X_test_sel, y_test).round(2)}")
print(f"Selected features: {list(shap_df_test.columns[selected_features])}")

#%%
c1 = list(df_eeg.set_index(['chan', 'set', 'subject','epoch']).loc[(['C3', 'C4', 'EOG', 'EMG'], 
                                                                    ['training']), 
                                                                   df_eeg.columns[2:-4]].unstack(level='chan').columns)
c1_ = [f"{x[0]}_{x[1]}" for x in c1]
#c2 = list(ecg_train[ecg_train.columns[0:-4]].columns)

# c_all = c1_ + c2

# Create bar plot
plt.figure(figsize=(20, 20))
sns.barplot(y=c1_, x=rf.feature_importances_[0:len(c1_)], palette='RdYlGn')    
plt.xlabel('Normalized feature importance')
plt.xticks(rotation=20)
# Set font size of y-tick labels
ax = plt.gca()
ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()


plt.figure(figsize=(20, 20))
#sns.barplot(y=c2, x=rf.feature_importances_[len(c1_):len(c1_) + len(c2)], palette='RdYlGn')    
plt.xlabel('Normalized feature importance')
plt.xticks(rotation=20)
# Set font size of y-tick labels
ax2 = plt.gca()
ax2.tick_params(axis='y', labelsize=10)

plt.tight_layout()

#%%
def compare_ecg_feature_methods(data_sleepecg, data_ecg):
    #from sklearn.impute import SimpleImputer, KNNImputer
    #from sklearn.feature_selection import f_classif
    
    X = data_sleepecg[data_sleepecg.columns[0:-1]].replace([np.inf, -np.inf], np.nan)
     
    # Create a imputer object
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    #imputer = KNNImputer(n_neighbors=5)
    
    # Impute NaNs with KNN imputation
    X_imputed = imputer.fit_transform(X)
    
    # Convert the numpy array back to a DataFrame
    df_imputed = pd.DataFrame(X_imputed, 
                              columns=data_sleepecg.columns[0:-1])
    
    # Calculate f-values
    fvals = pd.Series(f_classif(X=df_imputed, y=data_sleepecg.stage)[0],
                      index=df_imputed.columns).sort_values()
    
    
    plt.figure(figsize=(6, 6))
    sns.barplot(y=fvals.index, x=fvals, palette='RdYlGn')
    plt.xlabel('F-values')
    plt.xticks(rotation=20);
    plt.tight_layout()
    
    
    ###
    X = data_ecg[data_ecg.columns[0:-2]].replace([np.inf, -np.inf], np.nan)
     
    # Create a imputer object
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    #imputer = KNNImputer(n_neighbors=5)
    
    # Impute NaNs with KNN imputation
    X_imputed = imputer.fit_transform(X)
    
    # Convert the numpy array back to a DataFrame
    df_imputed = pd.DataFrame(X_imputed, 
                              columns=data_ecg.columns[0:-2])
    
    # Calculate f-values
    fvals = pd.Series(f_classif(X=df_imputed, y=data_ecg.stage)[0],
                      index=df_imputed.columns).sort_values()
    
    
    plt.figure(figsize=(6, 6))
    sns.barplot(y=fvals.index, x=fvals, palette='RdYlGn')
    plt.xlabel('F-values')
    plt.xticks(rotation=20);
    plt.tight_layout()

# %%
# initialize some variables here
allArrays = []
stageArrays = []
partArray = []
for f in enumerate(files):
    fname = f[1]

    # import data arrays of features
    datArray = pickle.load(open(fname + "_allArrays.p", "rb"))

    # create array corresponding to dataset number
    part_dat = np.ones(len(datArray))
    part_dat = part_dat*f[0]

    # import data array of class labels (hypnograms )
    stageArray = pickle.load(open(fname + "_stageArrays.p", "rb"))

    # append all feature arrays
    allArrays.append(datArray)
    # 'C3','C4','M1','M2','EOG','EMG'

    # append all participant arrays
    partArray.append(part_dat)

    # append all stage labels
    stageArrays.append(stageArray)

# concetanate arrays
part_all = np.concatenate(partArray)
data_all = np.concatenate(allArrays)
stages_All = np.concatenate(stageArrays)

C3 = data_all[:, 0:5]
C4 = data_all[:, 6:11]
EOG = data_all[:, 24:29]
EMG = data_all[:, 30:35]

x = np.concatenate([C3, C4, EOG, EMG], axis=1)

# =============================================================================
# NAN correction/interpolation (K-nearest neighbor): The KNNImputer class provides
# imputation for filling in missing values using the k-Nearest Neighbors approach.
# see ref:
#   Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor Hastie,
#   Robert Tibshirani, David Botstein and Russ B. Altman, Missing value estimation
#   methods for DNA microarrays, BIOINFORMATICS Vol. 17 no. 6, 2001 Pages 520-525.
# https://scikit-learn.org/stable/modules/impute.html#ol2001
# =============================================================================
imputer = KNNImputer(missing_values=np.nan, n_neighbors=5, weights="uniform")
x = imputer.fit_transform(x)
print(imputer.fit_transform(x))

y = stages_All
z = part_all
list_feat_labels = [x, y, z]

# concatenate all N2, SWS stages verusus the rest for binary classification
# y[y == 1] = 0
# y[y == 2] = 0
# y[y == 3] = 1
# y[y == 4] = 1
# y[y == 5] = 0

# =============================================================================
# GroupKFold is a variation of k-fold which ensures that the same group is not represented
# in both testing and training sets. For example if the data is obtained from different
# subjects with several samples per-subject and if the model is flexible enough to
# learn from highly person specific features it could fail to generalize to new subjects.
# GroupKFold makes it possible to detect this kind of overfitting situations.
# =============================================================================

# event_id = {'Sleep stage W/N1/REM': 0,
#             'Sleep stage N2/SWS': 1}

event_id = {'Wake': 1,
            'Stage 1': 2,
            'Stage 2': 3,
            'Stage 3': 4,
            'REM': 5}

gss = GroupShuffleSplit(n_splits=100, train_size=.8)

# test all 100 splits to see variation of classifier results based on splits
for train_index, test_index in gss.split(list_feat_labels[0], list_feat_labels[1], groups=list_feat_labels[2]):
    print("GROUP TRAIN:", np.unique(list_feat_labels[2][train_index]), "GROUP TEST:", np.unique(
        list_feat_labels[2][test_index]))
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

# See the following for more info re: RF modeling
# https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#remarks
# also: L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
rf = []
# random state for reproducibility
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
y_score = rf.predict_proba(X_test)  # modeling score
y_pred = rf.predict(X_test)

# Learn to predict each class against the other, see:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
# May not be necessary actually....
# rf = []
# rf = OneVsRestClassifier(RandomForestClassifier(n_estimators = 100, oob_score = True, random_state=42))
# rf.fit(X_train, y_train)
# y_score = rf.predict_proba(X_test) #modeling score
# y_pred = rf.predict(X_test)

# accuracy report, confusion matrix, classification reports
acc = accuracy_score(y_test, y_pred)
print("Accuracy score: {}".format(acc))
confusion = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
report = classification_report(y_test, y_pred, target_names=event_id.keys())
print(classification_report(y_test, y_pred, target_names=event_id.keys()))

#%%

# Prediction model key
model_id ={'model 1':'2 EEG, 1 EOG, 1 EMG',
           'model 2':'1 EEG, 1 EOG, 1 EMG',
           'model 3':'2 EEG, 1 EOG',
           'model 4':'2 EEG, 1 EMG',
           'model 5':'2 EEG',
           'model 6':'1 EEG'}


new_path = '/media/administrator/data/Study_1_data/Statistics/Classifier_validation/All/'
title = 'Confusion matrix - Test - 2 EEG, 1 EOG, 1 EMG'
     
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay.from_predictions(
     y_test,
     y_pred,
     display_labels=event_id.keys(),
     cmap=plt.cm.Blues,
     normalize='true',
     )

disp.ax_.set_title(title)
disp.ax_.grid(False)
plt.xticks(rotation=45)
plt.yticks(rotation=360)
plt.tight_layout()
for text in disp.text_.ravel():
    value = float(text.get_text())
    text.set_text(f"{value:.{2}f}")

plt.savefig(new_path + "confusion_matrix_model1_test.png")
plt.close('all')

# %% Save RF Models/Training/Testing sets
cd('/home/administrator/Documents/Classifier')
# 2 EEG. 1 EMG, 1 EOG
pickle.dump(rf, open("rf_model_1_cfs.p", "wb"))
pickle.dump(X_train, open("rf_X_train_1_cfs.p", "wb"))
pickle.dump(X_test, open("rf_X_test_1_cfs.p", "wb"))
pickle.dump(y_train, open("rf_y_train_1_cfs.p", "wb"))
pickle.dump(y_test, open("rf_y_test_1_cfs.p", "wb"))
pickle.dump(y_pred, open("rf_y_pred_1_cfs.p", "wb"))
pickle.dump(y_score, open("rf_y_pred_1_cfs.p", "wb"))
# 1 EEG. 1 EMG, 1 EOG
pickle.dump(rf, open("rf_model_2_cfs.p", "wb"))
pickle.dump(X_train, open("rf_X_train_2_cfs.p", "wb"))
pickle.dump(X_test, open("rf_X_test_2_cfs.p", "wb"))
pickle.dump(y_train, open("rf_y_train_2_cfs.p", "wb"))
pickle.dump(y_test, open("rf_y_test_2_cfs.p", "wb"))
pickle.dump(y_pred, open("rf_y_pred_2_cfs.p", "wb"))
pickle.dump(y_score, open("rf_y_pred_2_cfs.p", "wb"))
# 2 EEG, 1 EOG
pickle.dump(rf, open("rf_model_3_cfs.p", "wb"))
pickle.dump(X_train, open("rf_X_train_3_cfs.p", "wb"))
pickle.dump(X_test, open("rf_X_test_3_cfs.p", "wb"))
pickle.dump(y_train, open("rf_y_train_3_cfs.p", "wb"))
pickle.dump(y_test, open("rf_y_test_3_cfs.p", "wb"))
pickle.dump(y_pred, open("rf_y_pred_3_cfs.p", "wb"))
pickle.dump(y_score, open("rf_y_pred_3_cfs.p", "wb"))
# 2 EEG, 1 EMG
pickle.dump(rf, open("rf_model_4_cfs.p", "wb"))
pickle.dump(X_train, open("rf_X_train_4_cfs.p", "wb"))
pickle.dump(X_test, open("rf_X_test_4_cfs.p", "wb"))
pickle.dump(y_train, open("rf_y_train_4_cfs.p", "wb"))
pickle.dump(y_test, open("rf_y_test_4_cfs.p", "wb"))
pickle.dump(y_pred, open("rf_y_pred_4_cfs.p", "wb"))
pickle.dump(y_score, open("rf_y_pred_4_cfs.p", "wb"))
# 2 EEG
pickle.dump(rf, open("rf_model_5_cfs.p", "wb"))
pickle.dump(X_train, open("rf_X_train_5_cfs.p", "wb"))
pickle.dump(X_test, open("rf_X_test_5_cfs.p", "wb"))
pickle.dump(y_train, open("rf_y_train_5_cfs.p", "wb"))
pickle.dump(y_test, open("rf_y_test_5_cfs.p", "wb"))
pickle.dump(y_pred, open("rf_y_pred_5_cfs.p", "wb"))
pickle.dump(y_score, open("rf_y_pred_5_cfs.p", "wb"))
# 1 EEG
pickle.dump(rf, open("rf_model_6_cfs.p", "wb"))
pickle.dump(X_train, open("rf_X_train_6_cfs.p", "wb"))
pickle.dump(X_test, open("rf_X_test_6_cfs.p", "wb"))
pickle.dump(y_train, open("rf_y_train_6_cfs.p", "wb"))
pickle.dump(y_test, open("rf_y_test_6_cfs.p", "wb"))
pickle.dump(y_pred, open("rf_y_pred_6_cfs.p", "wb"))
pickle.dump(y_score, open("rf_y_pred_6_cfs.p", "wb"))

# Re-load the above to perform the next section later on....

# %%
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 5  # class labels
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area - take this calc. with a grain of salt....
# to-do research differences between micro and macro ROC averages
for i in range(n_classes):
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test == i, y_score[:, i].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


##############################################################################
# Plot ROC curves for the multilabel problem
# ..........................................
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# Area under ROC for multiclass problem
y_prob = rf.predict_proba(X_test)
macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                  average="macro")
weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                     average="weighted")
macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                  average="macro")
weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                     average="weighted")
print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
      "(weighted by prevalence)"
      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

# %% Further classifier statistics

# feature importances
feats = ['EEG1_delta', 'EEG1_theta', 'EEG1_alpha', 'EEG1_sigma', 'EEG1_beta', 'EEG2_delta',
         'EEG2_theta', 'EEG2_alpha', 'EEG2_sigma', 'EEG2_beta', 'EOG_delta', 'EOG_theta',
         'EOG_alpha', 'EOG_sigma', 'EOG_beta', 'EMG_delta', 'EMG_theta', 'EMG_alpha',
         'EMG_sigma', 'EMG_beta']
plt.plot(feats, rf.feature_importances_)

# Out-of-bag samples for generalization accuracy estimate
rf.oob_score_

# %%
# ROC curve plot for binary classification
# rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
# fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
# plt.figure()
# plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# #plt.savefig('Log_ROC')
# plt.show()

# Confusion matrix


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
target_names = event_id.keys()
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# visualize classifier - needs work
# def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
#     ax = ax or plt.gca()

#     # Plot the training points
#     ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
#                clim=(y.min(), y.max()), zorder=3)
#     ax.axis('tight')
#     ax.axis('off')
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()

#     # fit the estimator
#     model.fit(X, y)
#     xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
#                          np.linspace(*ylim, num=200))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

#     # Create a color plot with the results
#     n_classes = len(np.unique(y))
#     contours = ax.contourf(xx, yy, Z, alpha=0.3,
#                            levels=np.arange(n_classes + 1) - 0.5,
#                            cmap=cmap, clim=(y.min(), y.max()),
#                            zorder=1)

#     ax.set(xlim=xlim, ylim=ylim)
