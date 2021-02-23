#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:22:13 2021

@author: administrator
"""

import os 
from pyedflib import highlevel

path = '/media/administrator/data/Child_psych_data/'
save_path = path # could be the same path as above

files_list = [os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files]
for i, files in enumerate(files_list):
    print(i, files)
    highlevel.anonymize_edf(files, new_file= save_path + 'anonymized_' + str(i) + '.edf', 
                            to_remove=['patientname', 'patientcode', 'birthdate', 'gender', 'admincode'],
                            new_values=['anonymized', 'anonymized', '', '', ''])