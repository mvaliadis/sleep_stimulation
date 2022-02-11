#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:46:16 2021

@author: administrator
"""

import liesl
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os 
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal 
import scipy.stats as stats 
import mne 
import pandas as pd 
from scipy.optimize import curve_fit
from tqdm import tqdm
import lmfit
import seaborn as sns
from sleepstim.Analysis.Resting_State.rs_preproc import subject_cond_parser
import pickle
from scipy.integrate import trapz, cumtrapz
import pingouin as pg

def tkeo(data, normalize=True, plot=True):
    """
    Basic z-scored and normalized (if desired) Teager-Kaiser Energy Operator
    """
    emgf = np.zeros((len(data)))
    # Yt = Xt^2 - Xt-1 * Xt+1
    for i in range(1,len(emgf)-1):
        emgf[i] = data[i]**2 - data[i-1]*data[i+1]
    if normalize:
        # convert original signal to z-score from time-zero
        emgZ = stats.zscore(data)
        # same for filtered signal energy
        emgE = stats.zscore(emgf)
    else:
        emgZ = data ; emgE = emgf
    # plot "raw"
    if plot:       
        plt.figure()
        plt.plot(emgZ)
        plt.plot(emgE,'m')
        plt.title("TKEO energy plotted over EMG/EEG")
        plt.xlabel('Time')
        plt.ylabel('Energy')
        # plt.legend()
        plt.show()
    return emgE 

def process_rawXDF(file, paired=False):   
    # load data
    streams = liesl.XDFFile(file)
    
    ## read EEG/EMG data
    eego = streams["eego"]
    sf = eego.nominal_srate
    
    ## read marker information
    marker = streams['reiz-marker']
    TMStimes = [marker.time_stamps[i] for i,v in enumerate(marker.time_series) if \
                'cse_' in v[0] or 'icf_' in v[0] or 'sici_' in v[0]][1::]
    
    ## checks here for data length and if intensity % given for pulse
    stim = len(TMStimes)
    if paired:
        stim_min = 7; rep = 12  
    else:
        stim_min = 15; rep = 21
    
    if marker.time_series[0][0] != '':
        intensity = int(marker.time_series[0][0].split('_')[-1])
    else:
        intensity = []
        return [], np.ones(rep)*np.nan, sf, [], intensity, np.ones(rep)*np.nan, np.ones(rep)*np.nan
    
    if eego.time_series.shape[0]/sf < 20 or stim < stim_min:
        return [], np.ones(rep)*np.nan, sf, [], intensity, np.ones(rep)*np.nan, np.ones(rep)*np.nan

    ## label data
    edcix = eego.channel_labels.index("chan_7")
    c3ix = eego.channel_labels.index("C3")
    msix = eego.channel_labels.index("M2")
    
    ## match clocks
    eegoTMStimes = [np.argmin(np.abs(eego.time_stamps - ts)) for ts in TMStimes]
        
    ## get EMG/C3 data
    edcdat = eego.time_series[:,edcix]*1e6
    # edcdat = mne.filter.notch_filter(edcdat.astype('float64'),sf,freqs=(50,100,150,200)) 
    c3dat = eego.time_series[:,c3ix]*1e6
    # c3dat = mne.filter.notch_filter(c3dat.astype('float64'),sf,freqs=(50,100,150,200))

    ## CMC between EDC_R & C3
    # cmc = CMC(edcdat, c3dat, sf=sf, foi=(3, 40), plot=False, method='multitaper_conn')
    cmc = []
    
    # ## Plot TMS artifact 
    # plt.figure()
    # plt.plot(eego.time_stamps,c3dat)
    # plt.plot(TMStimes, [0]*len(TMStimes), 'or')
    
    tkeo_vals = []
    dovs = []
    dovs_bool = []    
    edcepochs = np.nan*np.zeros((len(eegoTMStimes), 200))
    for i, ts in enumerate(eegoTMStimes):
        c3ep = np.abs(c3dat[ts-100:ts+100])
        dov = max(np.diff(c3ep))
        dovs.append(dov)
        dovs_bool.append(dov >= 500)
        print(f'The maximum change in voltage is {dov} and '
              f'occurs at: {np.argmax(np.diff(c3ep))}')
        # use TKEO to determine peak coil artifact location in EEG
        tartifact = np.argmax(tkeo(c3ep, plot=False))
        tkeo_vals.append(max(tkeo(c3ep, plot=False)))
        ts_corrected = (ts-100)+tartifact
        # if any timestamp doesn't last 200 time points, correct for this and add nans to trial
        last_ts = int((eego.time_stamps - eego.time_stamps[0])[-1] * sf)
        if ts_corrected+100 > last_ts:
            add_nans = np.nan*np.zeros(ts_corrected+100 - last_ts)
            edcepochs[i,:] = np.concatenate([edcdat[ts_corrected-100:last_ts] - np.median(edcdat[ts_corrected-100:last_ts]), add_nans], axis=0)
        else:
            edcepochs[i,:] = edcdat[ts_corrected-100:ts_corrected+100] - np.median(edcdat[ts_corrected-100:ts_corrected+100])
    
    tkeo_vals = np.asarray(tkeo_vals) 
    dovs =  np.asarray(dovs) 
    dovs_bool = np.asarray(dovs_bool)   
    
    # if less than 20 trials, add nans to remaining VPPs, tkeo_vals 
    if min(edcepochs.shape) < 21 and paired == False:
        add_nans = np.nan*np.zeros(21 - min(edcepochs.shape))
        Vpp = np.concatenate([np.ptp(edcepochs[:, 115:160], axis = 1), add_nans], axis=0)
        tkeo_vals = np.concatenate([tkeo_vals, add_nans], axis=0)
    elif min(edcepochs.shape) < 12 and paired == True:
        add_nans = np.nan*np.zeros(12 - min(edcepochs.shape))
        Vpp = np.concatenate([np.ptp(edcepochs[:, 125:160], axis = 1), add_nans], axis=0)
        tkeo_vals = np.concatenate([tkeo_vals, add_nans], axis=0)
    else:
        if paired:
            Vpp = np.ptp(edcepochs[:, 125:160], axis = 1)
        else:
            Vpp = np.ptp(edcepochs[:, 115:160], axis = 1)
      
    # flag MEPs that don't exceed 3 std of pre-stimulus signal
    if paired:
        bool_mep = np.logical_and(3*np.std(abs(edcepochs[:, 0:90]), axis=1) < Vpp[~np.isnan(Vpp)],
                                  dovs_bool)
        # always flag first paired pulse as invalid due to technical recording issues!
        bool_mep[0] = False 
    else:
        bool_mep = np.logical_and(3*np.std(abs(edcepochs[:, 0:100]), axis=1) < Vpp[~np.isnan(Vpp)],
                                  dovs_bool)
    if True in np.unique(np.isnan(Vpp)):
        if paired:
            bool_mep.resize((12), refcheck=False)
        else:
            bool_mep.resize((21), refcheck=False)
    
    # ## Plot EDC response
    # plt.figure()
    # plt.title(" ".join(file.split('/')[-2::]))
    # plt.plot(np.arange(-max(edcepochs.shape)/2, max(edcepochs.shape)/2)/sf, np.mean(edcepochs,0))
    
    return edcepochs, Vpp, sf, cmc, intensity, bool_mep, tkeo_vals 
       
def function_sigmoid(intensity, mep_max, s50, k, b):
    y_mep = ((mep_max/(1 + np.exp(k*(s50-intensity)))))+b
    return (y_mep)

def trim_mean(x):
    tm = stats.trim_mean(x, proportiontocut = 0.1)
    return tm

def io_curve_dnu(norm_mep, optimize = False, plot = True):
    xdata = np.linspace(0, 6, 7)
    ydata = norm_mep
    
    ## Approximate slope parameters with non-linear least squares regression
    popt, _ = curve_fit(sigmoid, xdata, norm_mep)  
    x = xdata
    y = sigmoid(xdata, *popt)
    
    ## Additional curve fitting optimization - WARNING - currently does not work 
    if optimize:
        m1 = lmfit.models.Model(boltzmann_sigmoid)
        parms = m1.make_params()
        parms['Amplitude'].set(1., min=0., max=10.)
        parms['Bias'].set(0.001, min=0.001, max=.01)
        parms['Slope'].set(.9, min=0, max=10)
        parms['Threshold'].set(6., min=1., max=16.)   
        out = m1.fit(y, parms, x = x, method = 'leastsq', nan_policy = 'omit')
        y = boltzmann_sigmoid(x, Amplitude = out.best_values['Amplitude'], Bias = out.best_values['Bias'],
                              Slope = out.best_values['Slope'], Threshold = out.best_values['Threshold'])
        
    if plot:      
        plt.figure()
        plt.plot(np.linspace(90, 150, 7), ydata, 'o', label='data')
        plt.plot(np.linspace(90, 150, 7), y, label='fit')
        plt.legend(loc='best', fancybox=True)
        plt.ylabel('Normalized MEP Amplitude') 
        plt.xlabel('% RMT Intensity')
        print('Greatest slope transition at {:.1f} %RMT'.format(np.linspace(90, 150, 7)[np.argmax(np.diff(y))]))
        plt.show()
        plt.tight_layout()
        sns.despine()
        
    return x, y

def tangent_point():
    return tangent_pt

def peak_slope():
    return k

def AUC_io_curve():
    return AOC

def s50():
    return s50

def x_intercept_tanget():
    return x_tan 

#%%
maindir = '/media/administrator/data/Study_1_data/Pre_post_data/'
def TMS_results(maindir):  
    files = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(maindir) for i in files if 'cse' in i or 'icf' in i or 'sici' in i])  
    results = defaultdict(lambda: [])
    mep_data = []
    for file in tqdm(files):
        print(file)
        subjname = str(file).split("/")[-2]             
        night = " ".join(str(file).split('.')[0].split('/')[-1].split('_')[0:2])
        condition = subject_cond_parser(file, study_phase='tms')
        name = str(file).split('.')[0].split('/')[-1]
        session = name.split('_')[-2]
        rec = name.split('_')[-1]
        if 'sici' in name or 'icf' in name:
            paired = True
            protocol = str(file).split('.')[0].split('/')[-1].split('_')[2]
        else:
            paired = False
            protocol = " ".join(str(file).split('.')[0].split('/')[-1].split('_')[2:4])
        edcepochs, Vpp, sf, cmc, intensity, bool_mep, tkeo_vals = process_rawXDF(file, paired=paired)
        
        if edcepochs == []:
            edcepochs = np.ones([len(bool_mep), 200])*np.nan
         
        if edcepochs.shape[0] != len(bool_mep):
            diff = len(bool_mep) - edcepochs.shape[0]
            edcepochs = np.concatenate([edcepochs, np.ones([diff, 200])*np.nan], 0)
            
        mep_data.append(edcepochs)
        
        results["Vpp"].extend(Vpp)
        results["Vpp_True"].extend(bool_mep)
        results["TKEO_value"].extend(tkeo_vals)
        results["logVpp"].extend(np.log(Vpp))
        results["Intensity"].extend([intensity]*len(Vpp))
        results["Subject"].extend([subjname]*len(Vpp))
        results["Night"].extend([night]*len(Vpp))
        results["Condition"].extend([condition]*len(Vpp))
        results["Protocol"].extend([protocol]*len(Vpp))
        results["Session"].extend([session]*len(Vpp))
        results["File"].extend([rec]*len(Vpp))
        # results["CMC"].extend([cmc]*len(Vpp))
    
        def plot_mep(edcepochs, file, paired=False):
            plt.figure()
            plt.title(" ".join(file.split('/')[-2::]))
            plt.plot(np.arange(-max(edcepochs.shape)/2, max(edcepochs.shape)/2)/sf, np.mean(edcepochs,0))
            if paired:
                entry = 0.0125
            else:
                entry = 0.0115
            plt.vlines(entry, ymin=np.mean(edcepochs[:,int(sf * entry)+100:],0).min()*1.1, ymax=np.mean(edcepochs[:,int(sf * entry)+100:],0).max()*1.1, colors='r', linestyles='dotted')
            plt.vlines(0.060, ymin=np.mean(edcepochs[:,int(sf * entry)+100:],0).min()*1.1, ymax=np.mean(edcepochs[:,int(sf * entry)+100:],0).max()*1.1, colors='r', linestyles='dotted')
            plt.xlim(entry - 0.005, .065)
            plt.ylim(np.mean(edcepochs[:,int(sf * entry)+100:],0).min()*1.1, np.mean(edcepochs[:,int(sf * entry)+100:],0).max()*1.1)
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (uV)')
            
    res_df = pd.DataFrame(results)
    # true_idx = np.asarray(res_df[res_df.Vpp_True==True].index)
    mep_data = np.concatenate(mep_data)
    # mep_data_true = mep_data[true_idx,:]
    
    ### Plotting average MEPs
    sns.set_theme(color_codes=True)
    conds = list(dict.fromkeys(list(res_df.Condition)))
    sessions = list(dict.fromkeys(list(res_df.Session)))  
    protocol = list(dict.fromkeys(list(res_df.Protocol)))      
    for p in protocol:
        fig, axs = plt.subplots(2,3, sharex=True, sharey=False, figsize=(18,8))
        fig.suptitle('Average MEP response by condition/protocol')
        df_p = res_df.loc[res_df.Protocol == p]
        for i,c in enumerate(conds):
            df_c = df_p.loc[df_p.Condition == c]
            for s, session in enumerate(sessions):
                df_session = df_c.loc[df_c.Session == session]
                dat = np.nanmean(mep_data[list(df_session.loc[df_session["Vpp_True"]==True].index), :], 0)
                axs[s, i].plot(np.arange(0, 200, 1),
                               dat)
                axs[s, i].set_title(f'{p} {c} {session}')
                
                if 'icf' in p or 'sici' in p:
                    entry = 110
                else:
                    entry = 110
                
                axs[s, i].set_xlabel('Time (ms)')
                axs[s, i].set_ylabel('Voltage (uV)')
        
                axs[s, i].set_xlim([entry, 160])
                axs[s, i].set_ylim([dat[entry:160].min()*1.1, dat[entry:160].max()*1.1])

                plt.show()
                
            plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/TMS/MEP_avg_{p}.jpg')

    plt.close('all')
    
    ####
    
    ## save df with all ##
    
    ####
    
    # drop icf & sici
    ind_drop = res_df[res_df['Protocol'].apply(lambda x: x.startswith('sici') or 
                                               x.startswith('icf') or x.startswith('cse 90'))].index
    io_df = res_df.drop(ind_drop)
    
    # sici/icf
    cse_drop = res_df[res_df['Protocol'].apply(lambda x: x.startswith('cse'))].index
    pulse_df = res_df.drop(cse_drop)
    from sklearn.preprocessing import minmax_scale 
    group_pulse_dfs = []
    pulse_df = pulse_df[pulse_df['Vpp_True']==True].reset_index(drop=True)
    for idx, group in enumerate(pulse_df.groupby(['Subject','Condition','Session','Protocol'])):
        # print(idx, group[0])
        group_dfp = group[1]
        group_dfp['Normalized MEPs'] = minmax_scale(list(group_dfp['Vpp']))
        group_pulse_dfs.append(group_dfp)
    # aggregate everything
    subject_pre_post_pulse_df = pd.concat(group_pulse_dfs).reset_index(drop=True)
    
    
    # group 90 - 150 I/O MEP responses 
    io_df['Protocol'] = io_df['Protocol'].str.replace('cse', '', regex=True).astype(int)
    
    # normalize meps by subject trial 
    def mep_normalization(mep_values : list):
        all_norm_meps = [(mep_values[i] - np.nanmin(mep_values[i])) / 
                         ((np.nanmax(mep_values[i])) - np.nanmin(mep_values[i])) for i in range(len(mep_values))]
        
        return all_norm_meps
    
    # norm totals
    group_dfs = []
    io_df = io_df[io_df['Vpp_True']==True].reset_index(drop=True)
    for idx, group in enumerate(io_df.groupby(['Subject','Condition','Session'])):
        # print(idx, group[0])
        group_df = group[1]
        group_df['Normalized MEPs'] = minmax_scale(list(group_df['Vpp']))
        group_dfs.append(group_df)
    # aggregate everything
    subject_pre_post = pd.concat(group_dfs).reset_index(drop=True)
    
    #####
    # group_pre_post = subject_pre_post.groupby(['Subject','Condition','Session','Protocol','Intensity']).mean()
         
    # groupies = []
    # for idx, group in enumerate(group_pre_post.groupby(['Subject','Condition','Session'])):
    #     # print(group[1]['Vpp'])
    #     groupies.append(group[1]['Vpp'].to_numpy())
    # norm_mep = np.concatenate(mep_normalization(groupies))
    # # reapply normalization?
    # group_pre_post['Normalized MEPs'] = norm_mep
        
    ## group by condition, session, protocol 
    final_res = subject_pre_post.groupby(['Condition','Session','Protocol','Intensity']).mean()['Normalized MEPs'].reset_index()

    return final_res, subject_pre_post, subject_pre_post_pulse_df

#%%
# maindir = '/media/administrator/data/Study_1_data/Pre_post_data/'
if __name__ == '__main__':
    run = input('Do you wish to restart the TMS analysis? ')
    if run == 'yes':
        final_res, subject_pre_post, subject_pre_post_pulse_df = TMS_results(maindir)
        
        save_path = '/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_average.p'
        pickle.dump(final_res, open(save_path, "wb")) 
        final_res.to_csv(r'/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_average.csv')
        
        save_path = '/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_paired.p'
        pickle.dump(subject_pre_post_pulse_df, open(save_path, "wb")) 
        subject_pre_post_pulse_df.to_csv(r'/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_paired.csv')
        
        save_path = '/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_subject.p'
        pickle.dump(subject_pre_post, open(save_path, "wb")) 
        subject_pre_post.to_csv(r'/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_subject.csv')
    else:
        subject_pre_post = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_subject.p', 'rb'))
        final_res = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_average.p', 'rb'))
        subject_pre_post_pulse_df = pickle.load(open('/media/administrator/data/Study_1_data/Statistics/TMS/TMS_results_paired.p', 'rb'))
        
    #%%
    ## Plot I/O curves 
    from statannotations.Annotator import Annotator
    sns.set_theme(color_codes=True)
    order = ['pre','post']
    pairs=[("SICI", "ICF")]
    
    ## Group plots all in 3 conditions (ICF/SICI)
    subject_pre_post_pulse_df_rev = subject_pre_post_pulse_df.groupby(['Subject','Condition','Session','Protocol']).mean()['Normalized MEPs'].reset_index()
    axs_pair = sns.catplot(data = subject_pre_post_pulse_df_rev, x='Session', y='Normalized MEPs', 
                           hue='Protocol', col='Condition', kind='point', join=True, 
                           estimator=trim_mean, ci=95, order = order)
    
    # # annotate significance 
    # annotator = Annotator(axs_pair, pairs, data=subject_pre_post_pulse_df_rev, x='Session', 
    #                       y='Normalized MEPs', order=order)
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    # annotator.apply_and_annotate()

    plt.tight_layout()
    plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/TMS/SICI_ICF_joint.jpg')
            
    ## Group plots all in 3 conditions 
    subject_pre_post_rev = subject_pre_post.groupby(['Subject','Condition','Session','Protocol']).mean()['Normalized MEPs'].reset_index()
    axs = sns.catplot(data = subject_pre_post_rev, x='Protocol', y='Normalized MEPs', 
                      hue='Session', col='Condition', kind='point', join=False, 
                      estimator=trim_mean, ci=95, hue_order = ['pre','post'])
    
    down_y_points_post = np.mean([axs.axes_dict['down'].get_lines()[i].get_ydata() for i in range(6)], axis=1)
    down_y_points_pre = np.mean([axs.axes_dict['down'].get_lines()[i].get_ydata() for i in range(6,12)], axis=1)
    
    up_y_points_post = np.mean([axs.axes_dict['up'].get_lines()[i].get_ydata() for i in range(6)], axis=1)
    up_y_points_pre = np.mean([axs.axes_dict['up'].get_lines()[i].get_ydata() for i in range(6,12)], axis=1)
    
    sham_y_points_post = np.mean([axs.axes_dict['sham'].get_lines()[i].get_ydata() for i in range(6)], axis=1)
    sham_y_points_pre = np.mean([axs.axes_dict['sham'].get_lines()[i].get_ydata() for i in range(6,12)], axis=1)
    
    x_points = [0,1,2,3,4,5]
    x_data_ext = np.linspace(0,5,100)
    
    ## Plot fitted curves
    down_sessions = [down_y_points_pre,down_y_points_post, 'down']
    up_sessions = [up_y_points_pre,up_y_points_post, 'up']
    sham_sessions = [sham_y_points_pre,sham_y_points_post, 'sham']
    
    for session in (down_sessions, sham_sessions, up_sessions):
        p0 = [max(session[0]), trim_mean(x_points),1, min(session[0])] 
        popt0, _ = curve_fit(function_sigmoid, xdata=x_points, p0=p0,
                             ydata=session[0], maxfev=50000, method='lm')
        p1 = [max(session[1]), trim_mean(x_points),1, min(session[1])] 
        popt1, _ = curve_fit(function_sigmoid, xdata=x_points, p0=p1,
                             ydata=session[1], maxfev=50000, method='lm')
        
        axs.axes_dict[session[2]].plot(x_data_ext,function_sigmoid(x_data_ext, *popt1), linestyle='--', label = 'post')
        axs.axes_dict[session[2]].plot(x_data_ext,function_sigmoid(x_data_ext, *popt0), linestyle='--', label = 'pre')
    
        plt.tight_layout()
  
        plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/TMS/IO_curve_joint.jpg')
            
    #%%
    # =============================================================================
    # Subject level IO curve plots/analysis
    # TO-DO: REMOVE SUBJECTS WITH 2 OR LESS SESSIONS, DEAL WITH MISSING I/O PROTOCOL (~150 FOR SOME)
    # =============================================================================
    
    fac_idx = []
    param = []
    subject = list(dict.fromkeys(list(subject_pre_post.Subject)))
    for sub in subject:
        sub_res = subject_pre_post[subject_pre_post.Subject == sub]
        print(f'Computing faciliation index for subject: {sub}')
        # plt.figure()
        axs = sns.catplot(x='Protocol', y='Normalized MEPs', 
                          hue='Session', col='Condition', kind = 'point', join=False, 
                          estimator=trim_mean, ci=95, hue_order = ['pre','post'],
                          data=sub_res)
        #sub_res[sub_res.Condition.isin(['sham'])]
        
        if len(axs.axes_dict) < 3:
            continue             
        else:
            x_points = [0,1,2,3,4,5]
            x_data_ext = np.linspace(0,5,100)
            for item in axs.axes_dict:
                pre_s = np.mean([axs.axes_dict[item].get_lines()[i].get_ydata() for i in range(6)], axis=1)
                post_s = np.mean([axs.axes_dict[item].get_lines()[i].get_ydata() for i in range(6,12)], axis=1)
                # sessions = [pre_s, post_s, item]
                if all(~np.isnan([pre_s, post_s]).ravel()):
                    p0 = [max(pre_s), trim_mean(x_points),1, min(pre_s)]
                    popt0, _ = curve_fit(function_sigmoid, xdata=x_points, p0=p0,
                                         ydata=pre_s, maxfev=200000, method='lm')
                    p1 = [max(post_s), trim_mean(x_points),1, min(post_s)] 
                    popt1, _ = curve_fit(function_sigmoid, xdata=x_points, p0=p1,
                                         ydata=post_s, maxfev=200000, method='lm')
                
                    ## get line points 
                    y_points_pre = function_sigmoid(x_data_ext, *popt0)
                    y_points_post = function_sigmoid(x_data_ext, *popt1)
                    
                    ## plot curves
                    axs.axes_dict[item].plot(x_data_ext,y_points_pre, linestyle='--')
                    axs.axes_dict[item].plot(x_data_ext,y_points_post, linestyle='--')
                    
                    ## AUC calculation - compute difference post - pre x condition
                    AUC_pre = trapz(x = x_data_ext, y = y_points_pre, dx=x_data_ext[1] - x_data_ext[0]) / (len(x_data_ext) - 1)
                    AUC_post = trapz(x = x_data_ext, y = y_points_post, dx=x_data_ext[1] - x_data_ext[0]) / (len(x_data_ext) - 1)
        
                    # AUC based faciliation index
                    fac_idx.append([sub, item, AUC_post / AUC_pre])
                    # append further parameters (s50, k, MEP_max)
                    # param.append()
                else:
                    fac_idx.append([sub, item, np.nan])
        
        plt.tight_layout()
        plt.savefig(f'/media/administrator/data/Study_1_data/Statistics/TMS/IO_curve_{sub}.jpg')
        plt.close('all')

    fac_idx_df = pd.DataFrame(fac_idx, columns=['Subject','Condition','Facilitation Index (AUC Post/AUC Pre)'])
    axs_fac_idx = sns.violinplot(data = fac_idx_df, x='Condition', y='Facilitation Index (AUC Post/AUC Pre)', 
                                 estimator=np.median, ci=95)
    
    
    ### 
    
    # save ouput
    
    ###
    
    # annotate significance 
    pairs=[("down", "up"), ("down", "sham"), ("up", "sham")]
    annotator = Annotator(ax = axs_fac_idx, pairs=pairs, data=fac_idx_df, 
                          x='Condition', y='Facilitation Index (AUC Post/AUC Pre)')
    annotator.configure(test='t-test_welch', text_format='star', loc='outside')
    annotator.apply_and_annotate()
    plt.tight_layout()
    
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    fac_idx_df = pd.DataFrame(fac_idx, columns=['Subject','Condition','Facilitation_Idx'])
    md = smf.mixedlm("Facilitation_Idx ~ Condition", fac_idx_df[fac_idx_df.Facilitation_Idx>0.001], 
                     groups=fac_idx_df[fac_idx_df.Facilitation_Idx>0.001]["Subject"])
    mdf = md.fit()
    print(mdf.summary())
     
    ## rmANOVA
    rmanova = pg.rm_anova(dv='Facilitation_Idx', within='Condition', subject='Subject', 
                          detailed = True, data=fac_idx_df[fac_idx_df.Facilitation_Idx>0.001])
    # Pretty printing of ANOVA summary
    pg.print_table(rmanova)
    # Post hoc analysis
    posthocs = pg.pairwise_ttests(dv='Facilitation_Idx', within='Condition',
                                  subject='Subject', data=fac_idx_df[fac_idx_df.Facilitation_Idx>0.001])
    pg.print_table(posthocs)