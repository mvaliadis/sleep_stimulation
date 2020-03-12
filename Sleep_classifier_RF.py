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

from os import chdir as cd
from os import listdir
cd('/home/administrator/Documents/Physionet_data')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

fls = sorted(listdir())
files = list(set([f.split('-')[0] for f in fls]))

#%%

# initialize some variables here
allArrays = []
stageArrays = []
partArray = []

for f in enumerate(files): 
    fname = f[1]
    
    # import data arrays of features
    datArray = pickle.load(open (fname + "-allArrays.p","rb"))
    
    # create array corresponding to dataset number 
    part_dat = np.ones(len(datArray))
    part_dat = part_dat*f[0]
    
    # import data array of class labels (hypnograms )
    stageArray = pickle.load(open (fname + "-stageArrays.p","rb"))
    
    # append all feature arrays     
    allArrays.append(datArray)
    
    # append all participant arrays
    partArray.append(part_dat)
    
    # append all stage labels    
    stageArrays.append(stageArray)

# concetanate arrays
part_all = np.concatenate(partArray)
data_all = np.concatenate(allArrays)  
stages_All = np.concatenate(stageArrays)
    
x = data_all
y = stages_All
z = part_all
list_feat_labels = [x,y,z]

# concatenate all N2, SWS stages verusus the rest for binary classification
y[y == 1] = 0
y[y == 2] = 0
y[y == 3] = 1 
y[y == 4] = 1
y[y == 5] = 0

# classify only non-wakefulness epochs 
#x = x[y != 1, :]
#y = y[y != 1]

# =============================================================================
# GroupKFold is a variation of k-fold which ensures that the same group is not represented 
# in both testing and training sets. For example if the data is obtained from different 
# subjects with several samples per-subject and if the model is flexible enough to 
# learn from highly person specific features it could fail to generalize to new subjects. 
# GroupKFold makes it possible to detect this kind of overfitting situations.
# =============================================================================

# Group shuffle split & compare performance of random forests on different splits for robustness     
#allX_train = []
#allX_test = []
#ally_train = []
#ally_test = []
#allRF_models = []
#allAccuracies = []
#allConfusion = []
#allClassif_reports = []

event_id = {'Sleep stage W/N1/REM': 0,
            'Sleep stage N2/SWS': 1}

gss = GroupShuffleSplit(n_splits=100, train_size=.8, random_state=42)

# test all 100 splits to see variation of classifier results based on splits
for train_index, test_index in gss.split(list_feat_labels[0],list_feat_labels[1], groups=list_feat_labels[2]):
    print("GROUP TRAIN:", np.unique(list_feat_labels[2][train_index]), "GROUP TEST:", np.unique(list_feat_labels[2][test_index]))
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

rf = []
rf = RandomForestClassifier(n_estimators = 100)   
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# accuracy report, confusion matrix, classification reports
acc = accuracy_score(y_test, y_pred)
print("Accuracy score: {}".format(acc))
confusion = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
report = classification_report(y_test, y_pred, target_names=event_id.keys())
print(classification_report(y_test, y_pred, target_names=event_id.keys()))
    
#    # append train and test indices 
#    allX_train.append(X_train)
#    allX_test.append(X_test)
#    ally_train.append(y_train)
#    ally_test.append(y_test)
#    
#    # append random forest models 
#    allRF_models.append(rf)
#    
#    # append accuracy reports
#    allAccuracies.append(acc)
#    
#    # append confusion matrices 
#    allConfusion.append(confusion)
#    
#    # append all classification reports
#    allClassif_reports.append(report)


cd('/home/administrator/Documents/Classifier')
#pickle.dump(allAccuracies, open("accuracies.p", "wb"))    
#pickle.dump(allConfusion, open("confusion_matrices.p", "wb"))   
#pickle.dump(allClassif_reports, open("classif_reports.p", "wb")) 
#pickle.dump(allX_train, open("X_train.p", "wb")) 
#pickle.dump(allX_test, open("X_test.p", "wb")) 
#pickle.dump(ally_train, open("y_train.p", "wb")) 
#pickle.dump(ally_test, open("y_test.p", "wb")) 

# save best model
pickle.dump(rf, open("rf_model.p", "wb"))      

# ROC curve plot for binary classification N2/3/4 versus all
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()

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