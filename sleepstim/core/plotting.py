# -*- coding: utf-8 -*-
"""
Core plotting functions for sleepstim project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_curve # precision_recall_curve included as it's related
from itertools import cycle

def plot_confusion_matrix(path, cm, target_names, title='Confusion matrix', cmap=plt.cm.Blues, save=False, save_name='default'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save:
        if path is None:
            print("Warning: Save is True but no path provided for plot_confusion_matrix.")
        else:
            plt.savefig(path + f'confusion_matrix_{save_name}.png') # Added .png extension

def ROC_curve_plot(rf_roc_auc, fpr, tpr, thresholds=None): # thresholds is often an output of roc_curve, not always used for plotting directly
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_multiclass_ROC(level, path, y_test_all, y_score_all, n_classes = 5, multi_class=True, save=True, save_name='bla bla'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # n_classes = 5  # class labels # Parameterized
    enc = OneHotEncoder(handle_unknown='ignore')
    
    # Ensure y_test_all is 2D for OneHotEncoder
    if y_test_all.ndim == 1:
        y_test_all_2d = y_test_all.reshape(-1,1)
    else: # Assuming it might already be one-hot encoded from some contexts
        y_test_all_2d = y_test_all
        
    # Fit encoder only if y_test_all is not already one-hot encoded
    if y_test_all_2d.shape[1] == 1 or not np.array_equal(y_test_all_2d, y_test_all_2d.astype(bool)):
        enc.fit(np.unique(y_test_all_2d).reshape(-1,1))
        y_test_transformed = enc.transform(y_test_all_2d).toarray()
    else: # Data is likely already one-hot encoded
        y_test_transformed = y_test_all_2d
        # Infer n_classes if not one-hot encoded initially this way
        if y_test_all_2d.shape[1] != n_classes and y_test_all.ndim == 2 :
             print(f"Warning: n_classes ({n_classes}) does not match one-hot encoded y_test_all columns ({y_test_all_2d.shape[1]}). Adjusting n_classes.")
             n_classes = y_test_all_2d.shape[1]


    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_transformed[:,i], y_score_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area   
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_transformed.ravel(), y_score_all.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot of a ROC curve for a specific class
    if multi_class==False:  
        stage = int(input('Please select the sleep stage ROC of interest (0-{}): '.format(n_classes-1)))
        if not 0 <= stage < n_classes:
            print(f"Invalid stage selected. Please choose between 0 and {n_classes-1}.")
            return
        plt.figure()
        plt.plot(fpr[stage], tpr[stage], color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc[stage])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC: Sleep Stage {stage} ')
        plt.legend(loc="lower right")
        plt.show()
        
    else: # Multi-class plotting
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        
        # Then interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
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
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']) # Adjusted for up to 5 classes
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC ' + level)
        plt.legend(loc="lower right")
        plt.show()
        if save:
            if path is None:
                print("Warning: Save is True but no path provided for plot_multiclass_ROC.")
            else:
                plt.savefig(path + f'ROC_multiclass_{save_name}.png') # Added .png extension

def transition_matrix_plot(probs, save=False, path=None):
    grid_kws = {"height_ratios": (.9, .05), "hspace": .1}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws, figsize=(5, 5))
    sns.heatmap(probs, ax=ax, square=False, vmin=0, vmax=1, cbar=True,
                cbar_ax=cbar_ax, cmap='YlOrRd', annot=True, fmt='.2f',
                cbar_kws={"orientation": "horizontal", "fraction": 0.1,
                          "label": "Transition probability"})
    ax.set_xlabel("To sleep stage")
    ax.xaxis.tick_top()
    ax.set_ylabel("From sleep stage")
    ax.xaxis.set_label_position('top')
    if save:
        if path is None:
            print("Warning: Save is True but no path provided for transition_matrix_plot.")
        else:
            plt.savefig(path + '_transition.png', dpi=100, bbox_inches='tight') # Added .png extension
