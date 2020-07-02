#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:24:59 2020

@author: mvaliadis
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from itertools import cycle

from os import chdir as cd
from os import listdir
#cd('/home/administrator/Documents/Physionet_data')
cd('/media/administrator/data/cfs/pre_proc')

from scipy import interp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer, KNNImputer

fls = sorted(listdir())
files = list(set([f.split('_')[0] for f in fls]))

#%%

# initialize some variables here
allArrays = []
stageArrays = []
partArray = []

for f in enumerate(files): 
    fname = f[1]
    
    # import data arrays of features
    datArray = pickle.load(open (fname + "_allArrays.p","rb"))
    
    # create array corresponding to dataset number 
    part_dat = np.ones(len(datArray))
    part_dat = part_dat*f[0]
    
    # import data array of class labels (hypnograms )
    stageArray = pickle.load(open (fname + "_stageArrays.p","rb"))
    
    # append all feature arrays     
    allArrays.append(datArray)
    #'C3','C4','M1','M2','EOG','EMG'
    
    # append all participant arrays
    partArray.append(part_dat)
    
    # append all stage labels    
    stageArrays.append(stageArray)

# concetanate arrays
part_all = np.concatenate(partArray)
data_all = np.concatenate(allArrays)  
stages_All = np.concatenate(stageArrays)


C3 = data_all[:,0:5] 
C4 = data_all[:,6:11] 
EOG = data_all[:,24:29]
EMG = data_all[:,30:35]
   
x = np.concatenate([C3,C4,EOG,EMG], axis=1)

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
list_feat_labels = [x,y,z]

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

event_id ={'Wake':1,
           'Stage 1':2,
           'Stage 2':3,
           'Stage 3':4,
           'REM':5}

gss = GroupShuffleSplit(n_splits=100, train_size=.8) 

# test all 100 splits to see variation of classifier results based on splits
for train_index, test_index in gss.split(list_feat_labels[0],list_feat_labels[1], groups=list_feat_labels[2]):
    print("GROUP TRAIN:", np.unique(list_feat_labels[2][train_index]), "GROUP TEST:", np.unique(list_feat_labels[2][test_index]))
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

# See the following for more info re: RF modeling 
# https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#remarks
# also: L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
rf = []
rf = RandomForestClassifier(n_estimators = 100, oob_score = True, random_state=42) # random state for reproducibility   
rf.fit(X_train, y_train)
y_score = rf.predict_proba(X_test) #modeling score
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

#%% Save RF Models/Training/Testing sets
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

### Re-load the above to perform the next section later on....

#%%
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 5  #class labels
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test==i, y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area - take this calc. with a grain of salt....   
# to-do research differences between micro and macro ROC averages                                                                                                                      
for i in range(n_classes):
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test==i, y_score[:,i].ravel())
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

#%% Further classifier statistics

# feature importances
feats = ['EEG1_delta','EEG1_theta','EEG1_alpha','EEG1_sigma','EEG1_beta', 'EEG2_delta',
         'EEG2_theta','EEG2_alpha','EEG2_sigma','EEG2_beta', 'EOG_delta','EOG_theta',
         'EOG_alpha','EOG_sigma','EOG_beta', 'EMG_delta','EMG_theta','EMG_alpha',
         'EMG_sigma','EMG_beta']
plt.plot(feats, rf.feature_importances_)

# Out-of-bag samples for generalization accuracy estimate
rf.oob_score_

#%%
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
target_names=event_id.keys()
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