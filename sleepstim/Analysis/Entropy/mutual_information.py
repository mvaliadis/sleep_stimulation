#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:12:05 2021

@author: administrator
"""

import scipy.stats as stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
import numpy as np

a = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
b = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 1])

print(stats.entropy([0.5,0.5])) # entropy of 0.69, expressed in nats
print(mutual_info_classif(a.reshape(-1,1), b, discrete_features = True)) # mutual information of 0.69, expressed in nats
print(mutual_info_score(a,b)) # information gain of 0.69, expressed in nats
