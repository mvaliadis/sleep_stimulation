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
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)


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
df = pd.read_csv('/media/administrator/data/cfs/feature_analysis/eeg_feature_space.csv')
df.rename(columns={'subject': 'subj'}, inplace=True)
df['subj'] = df['subj'].astype(str)
df_cfs = df_cfs[df_cfs.subj.isin(file_id)].reset_index().drop(columns='index')

train_idx = df_cfs.subj.isin(df[df.set=='training'].subj.unique().astype(str))
test_idx = df_cfs.subj.isin(df[df.set=='testing'].subj.unique().astype(str))

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
# df_cfs.ttest

## 4b. Sex based statistics
df_eeg = df[np.logical_or(df.chan=='C3', df.chan=='C4')]

# merge two dataframes on the 'subject' and 'set' columns
df_combined = pd.merge(df_eeg, df_cfs, on=['subj', 'set'], how='inner')
plot_df = df_combined.groupby(['subj','stage']).mean(numeric_only=True)

sns.violinplot(plot_df, hue='SEX', y='lziv', x='stage')

for feat in plot_df.columns[1:25]:
    p_sex = plot_df.mixed_anova(dv=feat, between='SEX', 
                                within='stage', subject='subj')['p-unc'][0].round(4)
    if p_sex < 0.05:
        print(feat)
    
df_sws = plot_df.xs(3.0, level='stage')

r = pg.ttest(
    x=df_sws[df_sws.SEX==0]['lziv'],
    y=df_sws[df_sws.SEX==1]['lziv']
    )

#%%
# Create a FacetGrid with 2 rows and 2 columns for the 4 stages
g = sns.FacetGrid(plot_df.reset_index(), col="stage")

# Map the swarmplot to each facet
g.map(
      sns.boxplot, "SEX", "lziv", 
      **dict(fill=True, palette="Set2")
      )

# Adjust axis labels and titles
g.set_axis_labels("SEX", "LZIV Metric")
g.set_titles(row_template="SEX {row_name}", col_template="Stage {col_name}")

# Adjust space between subplots
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("LZIV Across Sleep Stages by SEX", fontsize=16)

plt.show()
plt.tight_layout()

#%% NMTI Cross-validation dataset

## 1. Load dataset



















