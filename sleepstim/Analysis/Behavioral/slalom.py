#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 11:26:36 2021

@author: administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os 
import pandas as pd
import pingouin as pg
#from scipy.signal import hilbert
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sleepstim.Analysis.Resting_State.rs_preproc import check_match_prepost_data_elements
from pymer4.models import Lmer

sns.set_theme(style="whitegrid")
path = r'/media/administrator/data/Study_1_data/Pre_post_data'

#%%
def slalom_results(path, plot=False):
    files_list = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if '.csv' in i])  
    sheet = '/media/administrator/data/Study_1_data/Data_tracking/subject_codes.csv'
    subj_cond = np.loadtxt(sheet, delimiter=',', dtype='str', skiprows=1) 
    cond_dict = {0:'sham', 1:'up', 2:'down'}
    # sort files to make sure blocks are consistent between subject nights
    pre, post = check_match_prepost_data_elements(path, files=files_list, dtype='slalom')
    slalom_dict = []
    for j, (pre_f, post_f) in enumerate(zip(pre, post)):
        if pre_f.split('/')[-1][-16::] == post_f.split('/')[-1][-16::]:
            # decode night with stimulation condition
            subj = pre_f.split('/')[-1].split('_')[0][-8::]
            sub_pos = np.where(subj == subj_cond[:,0])[0][0]
            if int(pre_f.split('/')[-1].split('_')[-2]) + 1 == 1:
                cond = int(subj_cond[sub_pos,:][1])
            elif int(pre_f.split('/')[-1].split('_')[-2]) + 1 == 2:
                cond = int(subj_cond[sub_pos,:][2])
            elif int(pre_f.split('/')[-1].split('_')[-2]) + 1 == 3:
                cond = int(subj_cond[sub_pos,:][3])
            
            print('Generating report for the following pre/post subject + condition: ' + pre_f.split('/')[-1][-16::])
            data_pre, data_post = np.genfromtxt(pre_f, delimiter=','), np.genfromtxt(post_f, delimiter=',')
                       
            # Calculate RMSE for pre/post with zscores & zscores + amplitude envelope
            error_pre_rms = mean_squared_error(y_true = stats.zscore(data_pre[0,:]), 
                                               y_pred = stats.zscore(data_pre[1,:]), squared=False)

            error_post_rms = mean_squared_error(y_true = stats.zscore(data_post[0,:]), 
                                                y_pred = stats.zscore(data_post[1,:]), squared=False)
            # amplitude_envelope_pre_rms = mean_squared_error(y_true = np.abs(hilbert(stats.zscore(data_pre[0,:]))), 
            #                                                 y_pred = np.abs(hilbert(stats.zscore(data_pre[1,:]))), squared=False)            
            # amplitude_envelope_post_rms = mean_squared_error(y_true = np.abs(hilbert(stats.zscore(data_post[0,:]))), 
            #                                                  y_pred = np.abs(hilbert(stats.zscore(data_post[1,:]))), squared=False)  
            
            # # Calculation of error per time point 
            # error_pre = stats.zscore(data_pre[0,:]) - stats.zscore(data_pre[1,:])
            # error_post = stats.zscore(data_post[0,:]) - stats.zscore(data_post[1,:])
            # amplitude_envelope_pre_diff = np.abs(hilbert(stats.zscore(data_pre[0,:]))) - np.abs(hilbert(stats.zscore(data_pre[1,:])))
            # amplitude_envelope_post_diff = np.abs(hilbert(stats.zscore(data_post[0,:]))) - np.abs(hilbert(stats.zscore(data_post[1,:])))
            
                      
            ## create dictionary with info
            s_dict = {
                "RMSE_Pre": error_pre_rms,
                "RMSE_Post": error_post_rms,
                #"Amplitude envelope RMSE (Pre)": amplitude_envelope_pre_rms,
                #"Amplitude envelope RMSE (Post)": amplitude_envelope_post_rms,
                "RMSE_difference_ratio": ((error_post_rms - error_pre_rms)/error_pre_rms*100),
                "RMSE_difference": error_post_rms - error_pre_rms,
                #"Amplitude envelope RMSE difference": amplitude_envelope_post_rms - amplitude_envelope_pre_rms,
                "Subject": pre_f.split('/')[-1].split('_')[0][-8::],
                "Night": int(pre_f.split('/')[-1].split('_')[1]),
                "Condition": cond_dict[cond], 
                "Block": int(pre_f.split('/')[-1].split('_')[2][0]),
                }
            
            slalom_dict.append(s_dict)
            
            if plot:
                plt.figure()
                plt.plot(stats.zscore(data_pre[0,:]), label = 'Slalom correct trajectory (pre)')
                plt.plot(stats.zscore(data_pre[1,:]), label = 'Slalom attempted trajectory (pre)')
                plt.plot(stats.zscore(data_post[0,:]), label = 'Slalom correct trajectory (post)')
                plt.plot(stats.zscore(data_post[1,:]), label = 'Slalom attempted trajectory (post)')
                plt.legend()
               
        else:
            raise NameError('The given pre- and post- file names do not correspond! ')
        
    # covert to dataframe    
    df = pd.DataFrame(slalom_dict)     
    
    return df

#%%
def slalom_stats(df): 
    # ttests
    #stats.ttest_ind(df[df['Stimulation night']=='sham']['Absolute RMSE difference'], df[df['Stimulation night']=='up']['Absolute RMSE difference'])
    #stats.ttest_ind(df[df['Stimulation night']=='down']['Absolute RMSE difference'], df[df['Stimulation night']=='up']['Absolute RMSE difference'])
    # descriptive difference
    print(df.groupby(['Condition'])['RMSE_difference'].agg(['mean','median','std']).round(2))  
    # Compute repeated measures ANOVA
    rmanova = pg.rm_anova(dv='RMSE_difference', within='Condition', 
                          subject='Subject', detailed = True, data=df)
    # Pretty printing of ANOVA summary
    pg.print_table(rmanova)
    # Post hoc analysis
    posthocs = pg.pairwise_ttests(dv='RMSE_difference', within='Condition',
                                  subject='Subject', data=df)
    pg.print_table(posthocs)
    
def slalom_stats_lmm(df):
    model = Lmer(f"RMSE_difference ~ Condition*Block + (1|Subject)", 
                 data=df)
    # Using dummy-coding; suppress summary output
    model.fit(factors={"Condition": ["sham", "up", "down"],
                       "Block" : ["0","1","2","3"]}, 
              ordered=True, summarize=True)
    
    # Get ANOVA table, but this time force orthogonality for valid SS III inferences
    # In this case the data are balanced so nothing changes
    print(model.anova(force_orthogonal=True))
    

    ## Post-hoc tests 
    marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr",
                                                     marginal_vars='Condition',
                                                     # grouping_vars='Block',
                                                     )
    
    print(marginal_estimates)
    print(comparisons)
        
def violin_plot(df):    
    #sns.set_theme(style="whitegrid")
    sns.set_theme(style="darkgrid")
    ax = sns.violinplot(data=df, x='Condition', y='RMSE_difference', 
                        palette="Set3", bw=.2, linewidth=1)   
    # Finalize the figure
    ax.set_ylabel("\u0394 Root Mean Square Error")
    ax.set_title("Slalom Performance")
    sns.despine(left=True, bottom=True)
    
def violin_plot_comb(df_comb):
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 4, figsize=(2 * 6, 3*2), sharey=True)
    fig.suptitle("Change in CMC during Slalom")
    grk = ['\u03B8', '\u03B1', '\u03B2', '\u03B3']
    for ind, band in enumerate(['theta', 'alpha', 'beta', 'gamma']):
        # Get the PSD diff across channels for the current band
        contrast = df_comb.groupby(['Condition','Subject'])[f'{band}_diff'].mean()
               
        # Create a topomap for the current oscillation band
        ax = sns.violinplot(data=contrast.reset_index(), x='Condition', 
                            y=f'{band}_diff', ax=axes[ind],
                            palette="Set3", bw=.2, linewidth=1); 
       
        # Change the ylabel
        ax.set_ylabel('\u0394 ' + grk[ind] + ' CMC')
       
        # despine layout
        sns.despine(left=True, bottom=True)
        
        # tighten layout
        #plt.tight_layout()

    
#%%
## Bias investigation
# g = sns.lmplot(x='Absolute RMSE (Pre)', y='Absolute RMSE (Post)', data=df, col='Condition')

# # def annotate(data, **kws):
# #     r, p = stats.pearsonr(df['Absolute RMSE (Pre)'], df['Absolute RMSE (Post)'])
# #     ax = plt.gca()
# #     ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
# #             transform=ax.transAxes)
    
# # g.map_dataframe(annotate)
# plt.show()
# plt.tight_layout()

#%%
path = r'/media/administrator/data/Study_1_data/Pre_post_data'
if __name__ == '__main__':
    run = input('Do you wish to restart the Slalom analysis? ')
    if run == 'yes':
        df = slalom_results(path) 
        df.to_csv(r'/media/administrator/data/Study_1_data/Statistics/Slalom/Slalom_results.csv')
    else:
        df = pd.read_csv(r'/media/administrator/data/Study_1_data/Statistics/Slalom/Slalom_results.csv', index_col=0)
        
    # drop the following subjects 
    subjects = df['Subject'].unique()
    for i in zip(['9PJZ8Z8F','475MQ9BL','5LNKD1MG','CWESJCNJ','RVQL2MRD','6QF3HOJC','IBYYXKMB','NW2JV7YA']):
        df.drop(df.loc[df['Subject']==i[0]].index, inplace=True)
        
    df = df.drop([0,1,2,3,180,181,182,183]) 
    # mean over blocks by subject and condition
    df_slalom.groupby(['Condition','Subject']).mean().reset_index(inplace=True)
    
    ## load df
    cmc_path = '/media/administrator/data/Study_1_data/Statistics/CMC/CMC_results_final.csv'
    df_cmc = pd.read_csv(cmc_path)
    df_comb = pd.merge(df, df_cmc, on=["Subject","Condition","Night","Block"])
    
    for idx, band in enumerate(zip(['theta', 'alpha', 'beta', 'gamma','SNR'])):
        df_comb[f'{band[0]}_diff'] = df_comb[(f'{band[0]}', 'post')] - df_comb[(f'{band[0]}', 'pre')]
    
    # save combined df
    df_comb.to_csv('/media/administrator/data/Study_1_data/Statistics/Slalom/Slalom_cmc_results.csv')        
    
    # blah blah    
    slalom_stats_lmm(df)
    violin_plot(df)
    plt.savefig('/media/administrator/data/Study_1_data/Statistics/Slalom/Slalom_violin.jpg')


#%%
model = Lmer(f"Absolute_RMSE_difference ~ (gamma_diff + beta_diff) + (1|Subject)", 
               data=df_comb)
# Using dummy-coding; suppress summary output
model.fit(factors={"Condition": ["sham", "up", "down"]
                   }, 
          ordered=True, summarize=False)

# Get ANOVA table, but this time force orthogonality for valid SS III inferences
# In this case the data are balanced so nothing changes
print(model.anova(force_orthogonal=True))


## Post-hoc tests 
marginal_estimates, comparisons = model.post_hoc(p_adjust="fdr",
                                                 marginal_vars='Condition',
                                                 # grouping_vars='Block',
                                                 )

print(marginal_estimates)
print(comparisons)
