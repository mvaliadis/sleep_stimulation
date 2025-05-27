#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:17:18 2020

@author: mvaliadis
"""

import numpy as np
import random
import csv
from sleepstim.core.utils import generate_subject_code as sc


def generate_subject_condition(subjects, conditions, save_path):
    subject=[]
    night = []
    for i in range(subjects):
        # generate sequence of subject codes
        code = sc(8)
        print("Automatically generated subject ID by Python: " + code)
        
        # generate sequence of night conditions 
        nights = random.sample(range(conditions), conditions)
        
        # append conditions
        #night.append(", ".join( repr(e) for e in nights ))
        night.append(nights)
        
        # append subject codes
        subject.append(code)
        
        # write codes and conditions into csv file
        filename = save_path + "subject_codes2.csv"
        cond_names = [f'Condition {conds}' for conds in range(conditions)]
        fields = ['Subject ID'] + cond_names
        rows = np.transpose([subject, night])
        # =============================================================================
        # ----------  TO-DO: PARTION ROWS TO FIT CONDITIONS ----------    
        # =============================================================================
        with open(filename, 'w') as csvfile:   
            # creating a csv writer object  
            csvwriter = csv.writer(csvfile)  
            # writing the fields  
            csvwriter.writerow(fields)  
            # writing the data rows  
            csvwriter.writerows(rows)                 

if __name__ == '__main__':
    generate_subject_condition(subjects = 50, conditions = 10, save_path = '/media/administrator/data/ADHD_Study_1/')