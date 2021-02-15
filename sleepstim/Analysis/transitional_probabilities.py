#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:17:30 2020

@author: administrator
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from os import chdir as cd
from sleepstim.sleep_funs import transition_matrix, transition_matrix_prob, transition_matrix_plot

#%%
def transition_matrix(transitions):
    # the function takes a list with states labeled as successive integers and
    # returns a transition matrix, trans_max of all transitions between given states
    
    # derive number of states
    states = len(np.unique(transitions)) 
    
    # initialize matrix with 0s
    if states == 5:
        trans_matrix = [[0]*states for _ in range(states)] 
    else:
        trans_matrix = [[0]*5 for _ in range(5)] 
        logging.warning(f'One of the datasets contains only {states} sleep stages')
    
    # extract instances of each state
    for (i,j) in zip(transitions,transitions[1:]):
        trans_matrix[i][j] += 1
        
    return trans_matrix

def transition_matrix_prob(trans_matrix):
    # convert occurences to a transitional probability matrix to indicate the 
    # probability of transitioning from one state to the next  
    probs = [row/row.sum(axis=-1, keepdims=True) for row in np.asarray(trans_matrix)]

    return np.round(np.array(probs).astype(float), 4) 
 
def transition_matrix_plot(probs):
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
    # plt.savefig('transition.png', dpi=100, bbox_inches='tight')


#%%
## Apply over all training sets and plot sum of probabilities       
hypno_trans = []
for j in np.unique(z_train):
    hypno_test = [int(i) for i in list(y_train[np.where(z_train==j)])]
    hypno_trans.append(transition_matrix(hypno_test))
all_hypno_trans = sum(np.array(hypno_trans))
probs = transition_matrix_prob(all_hypno_trans)
for row in probs: print(' '.join('{0:.2f}'.format(x) for x in row))
transition_matrix_plot(probs)

