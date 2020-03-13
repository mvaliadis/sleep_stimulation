#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:17:18 2020

@author: mvaliadis
"""

import numpy as np
import random
from os import chdir as cd
cd('/home/administrator/Documents/Scripts')
from generate_subject_code import generate_subject_code as sc


subjects=[]
night = []
n1 = []
n2 = []
n3 = []

for i in range(30):
    # generate sequence of subject codes
    code = sc(8)
    print("Automatically generated subject ID by Python: " + code)
    
    # generate sequence of night conditions 
    nights = random.sample(range(3), 3)
    
    # append conditions
    night.append(nights)
    night1 = night[i][0]
    night2 = night[i][1]
    night3 = night[i][2]
    n1.append(night1)
    n2.append(night2)
    n3.append(night3)
    
    # append subject codes
    subjects.append(code)
    
    # write codes and conditions into csv file
    import csv
    filename = "subject_codes.csv"
    fields = ['Subject ID', 'Condition 1', 'Condition 2', 'Condition 3']  
    rows = np.transpose([subjects, n1, n2, n3])
    with open(filename, 'w') as csvfile:   
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        # writing the fields  
        csvwriter.writerow(fields)  
        # writing the data rows  
        csvwriter.writerows(rows)                 