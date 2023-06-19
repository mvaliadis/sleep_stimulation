#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:55:04 2023

@author: administrator
"""


from os import listdir
import numpy as np
import pandas as pd 
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt

#%% CFS: https://sleepdata.org/datasets/cfs

## 1. Load dataset information csv and convert to dataframe with relevant info
df_cfs = pd.read_csv(
    '/media/administrator/data/cfs/datasets/cfs-visit5-dataset-0.5.0.csv', 
    usecols=['nsrrid', 'age', 'SEX', 'race', 'ethnicity', 'bmi', 'ahi_a0h3',
             'AbnorEEG', 'AbnorEye', 'QuEEG1', 'QuHR', 'QuEOGL', 'QuChin', 
             'RemNRemPr', 'Stg1Stg2Pr', 'Stg2Stg3Pr', 'WakSlePr', 
             'INSODIAG', 'htndx', 'DIADIAG', 'DIANARC', 'DEPDIAG'])

# Rename for ease-of-use
df_cfs.rename(columns={
    'nsrrid': 'subj',
    # 'SEX': 'male',
    'ahi_a0h3': 'ahi',
    'INSODIAG': 'insomnia',
    'htndx': 'hypertension',
    'DIADIAG': 'diabetes',
    'DIANARC': 'narcolepsy',
    'DEPDIAG': 'depression'
    }, inplace=True)

df_cfs['race'].replace({1: 'caucasian', 2: 'african', 3: 'other'}, inplace=True)
df_cfs.loc[df_cfs['ethnicity'] == 1, 'race'] = 'hispanic'
df_cfs.drop(columns=['ethnicity'], inplace=True)
df_cfs.rename(columns={'race': 'ethnicity'}, inplace=True)

# Drop columns
df_cfs.drop(
    columns=['RemNRemPr', 'AbnorEEG', 'AbnorEye', 'QuEEG1', 'QuHR', 'QuEOGL', 
             'QuChin', 'RemNRemPr', 'Stg1Stg2Pr', 'Stg2Stg3Pr', 'WakSlePr'], 
    inplace=True)

# Convert to str
df_cfs['subj'] = df_cfs['subj'].astype(str)
#df_cfs.set_index('subj', inplace=True)

## 2. Narrow set to selected data only
fls = sorted(listdir('/media/administrator/data/cfs/pre_proc'))
files = list(set([f.split('_')[0] for f in fls]))
file_id = [item.split('-')[-1] for item in files]

# New dataframe, first load train/test dataframe to add set labels
# TODO: add path to train/test dataframe
df = df_eeg #pd.read_csv()
df_cfs = df_cfs[df_cfs.subj.isin(file_id)].reset_index().drop(columns='index')
train_idx = df_cfs.subj.isin(df[df.set=='training'].subject.unique())
test_idx = df_cfs.subj.isin(df[df.set=='testing'].subject.unique())

# Assign label to training/test sets in df
df_cfs.loc[train_idx, "set"] = "training"
df_cfs.loc[test_idx, "set"] = "testing"

## 3. Plot demographics
# Age
sns.barplot(data=df_cfs, x='set', y='age')
plt.tight_layout()

# Sex
sns.barplot(data=df_cfs, x='set', y='SEX')
plt.tight_layout()

## 4. Do stats 
df_cfs.ttest

#%% NMTI Cross-validation dataset

## 1. Load dataset



















