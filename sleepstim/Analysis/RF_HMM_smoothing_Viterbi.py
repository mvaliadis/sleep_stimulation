#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 16:56:32 2020

@author: administrator
"""

import numpy as np
import yasa 

#%%
## Viterbi Hidden Markov Chain Algorithm 
def viterbi(obs, states, start_p, trans_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * obs[st][0], "prev": None}
        #V[0][st] = {"prob": start_p[st] * obs[0][st], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = V[t-1][states[0]]["prob"]*trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t-1][prev_st]["prob"]*trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st

            #max_prob = max_tr_prob * obs[t][st]
            max_prob = max_tr_prob * obs[st][t]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

            
    for line in dptable(V):
        print(line)

    opt = []
    max_prob = 0.0
    previous = None
    
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st

    opt.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print ('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)
    

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

#%%
## Example with weather

obs = ('normal', 'cold', 'dizzy')
states = ('Healthy', 'Fever')
start_p = {'Healthy': 0.6, 'Fever': 0.4}
trans_p = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
emit_p = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }
        
viterbi(obs,
        states,
        start_p,
        trans_p,
        emit_p)

#%%
## RF Classifier smoothing

obs_all = rf.predict_proba(X_test[z_test==0])
obs = {
       'Wake' : obs_all[:,0],
       'N1' : obs_all[:,1],
       'N2' : obs_all[:,2],
       'N3' : obs_all[:,3],
       'REM' : obs_all[:,4]
       }

states_int = rf.predict(X_test[z_test==0])
states = tuple(yasa.hypno.hypno_int_to_str(rf.predict(X_test[z_test==0]), 
                                           mapping_dict={0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}))
    
start_p = {'Wake': 0.2, 'N1': 0.2, 'N2': 0.2, 'N3': 0.2, 'REM': 0.2}
    
# calculate based on existing labels in training set --> calculate average probabilites from state to state
trans_p = {
   'Wake' : {'Wake': probs[0][0], 'N1': probs[0][1], 'N2': probs[0][2], 'N3': probs[0][3], 'REM': probs[0][4]},
   'N1' : {'Wake': probs[1][0], 'N1': probs[1][1], 'N2': probs[1][2], 'N3': probs[1][3], 'REM': probs[1][4]},
   'N2' : {'Wake': probs[2][0], 'N1': probs[2][1], 'N2': probs[2][2], 'N3': probs[2][3], 'REM': probs[2][4]},
   'N3' : {'Wake': probs[3][0], 'N1': probs[3][1], 'N2': probs[3][2], 'N3': probs[3][3], 'REM': probs[3][4]},
   'REM' : {'Wake': probs[4][0], 'N1': probs[4][1], 'N2': probs[4][2], 'N3': probs[4][3], 'REM': probs[4][4]},
   }
            
viterbi(obs,
        states,
        start_p,
        trans_p,
        )


#%%
from prml.markov import CategoricalHMM
categorical_hmm = CategoricalHMM(initial_proba={'Wake': 0.2, 'N1': 0.2, 'N2': 0.2, 'N3': 0.2, 'REM': 0.2},
                                 transition_proba=transition_probs, means = np.vstack([counts, obs]).T)

