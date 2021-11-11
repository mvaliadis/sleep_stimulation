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

#%%
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

def CMC(signal1, signal2, sf, l_foi=2, h_foi=40, plot=True): 
    if plot:
        plt.figure()
        coh, f = plt.cohere(signal1, signal2, NFFT=int((2/l_foi)*sf), Fs=sf)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Coherence')
        plt.title('CMC between C3 and EDC_R')
        plt.xlim(l_foi, h_foi)
        plt.show()
    else:
        f, coh = signal.coherence(signal1, signal2, fs=sf, nperseg=(2/l_foi)*sf, 
                                  detrend='constant')
        
    return f[l_foi:h_foi], coh[l_foi:h_foi]   

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
        return [], np.ones(rep)*np.nan, sf, [], intensity, np.ones(rep)*np.nan
    
    if eego.time_series.shape[0]/sf < 20 or stim < stim_min:
        return [], np.ones(rep)*np.nan, sf, [], intensity, np.ones(rep)*np.nan

    ## label data
    edcix = eego.channel_labels.index("chan_7")
    c3ix = eego.channel_labels.index("C3")
    
    ## match clocks
    eegoTMStimes = [np.argmin(np.abs(eego.time_stamps - ts)) for ts in TMStimes]
        
    ## get EMG/C3 data
    edcdat = eego.time_series[:,edcix]*1e6
    # edcdat = mne.filter.notch_filter(edcdat.astype('float64'),sf,freqs=(50,100,150,200)) 
    c3dat = eego.time_series[:,c3ix]*1e6
    # c3dat = mne.filter.notch_filter(c3dat.astype('float64'),sf,freqs=(50,100,150,200))

    ## CMC between EDC_R & C3
    cmc = CMC(edcdat, c3dat, sf=sf, l_foi=10, h_foi=30, plot=False)
    
    # ## Plot TMS artifact 
    # plt.figure()
    # plt.plot(eego.time_stamps,c3dat)
    # plt.plot(TMStimes, [0]*len(TMStimes), 'or')
    
    edcepochs = np.nan*np.zeros((len(eegoTMStimes), 200))
    for i, ts in enumerate(eegoTMStimes):
        c3ep = np.abs(c3dat[ts-100:ts+100])
        # use TKEO to determine peak coil artifact location in EEG
        tartifact = np.argmax(tkeo(c3ep, plot=False))
        ts_corrected = (ts-100)+tartifact
        # if any timestamp doesn't last 200 time points, correct for this and add nans to trial
        last_ts = int((eego.time_stamps - eego.time_stamps[0])[-1] * sf)
        if ts_corrected+100 > last_ts:
            add_nans = np.nan*np.zeros(ts_corrected+100 - last_ts)
            edcepochs[i,:] = np.concatenate([edcdat[ts_corrected-100:last_ts] - np.median(edcdat[ts_corrected-100:last_ts]), add_nans], axis=0)
        else:
            edcepochs[i,:] = edcdat[ts_corrected-100:ts_corrected+100] - np.median(edcdat[ts_corrected-100:ts_corrected+100])
            
    # if less than 20 trials, add nans to remaining VPPs
    if min(edcepochs.shape) < 21 and paired == False:
        add_nans = np.nan*np.zeros(21 - min(edcepochs.shape))
        Vpp = np.concatenate([np.ptp(edcepochs[:, 115:160], axis = 1), add_nans], axis=0)
    elif min(edcepochs.shape) < 12 and paired == True:
        add_nans = np.nan*np.zeros(12 - min(edcepochs.shape))
        Vpp = np.concatenate([np.ptp(edcepochs[:, 125:160], axis = 1), add_nans], axis=0)
    else:
        if paired:
            Vpp = np.ptp(edcepochs[:, 125:160], axis = 1)
        else:
            Vpp = np.ptp(edcepochs[:, 115:160], axis = 1)
      
    # flag MEPs that don't exceed 3 std of pre-stimulus signal
    if paired:
        bool_mep = 3*np.std(abs(edcepochs[:, 0:90]), axis=1) < Vpp[~np.isnan(Vpp)]
    else:
        bool_mep = 3*np.std(abs(edcepochs[:, 0:100]), axis=1) < Vpp[~np.isnan(Vpp)]
    if True in np.unique(np.isnan(Vpp)):
        if paired:
            bool_mep.resize((12), refcheck=False)
        else:
            bool_mep.resize((21), refcheck=False)
    
    # ## Plot EDC response
    # plt.figure()
    # plt.title(" ".join(file.split('/')[-2::]))
    # plt.plot(np.arange(-max(edcepochs.shape)/2, max(edcepochs.shape)/2)/sf, np.mean(edcepochs,0))
    
    return edcepochs, Vpp, sf, cmc, intensity, bool_mep 

def boltzmann_sigmoid(x, Amplitude, Bias, Slope, Threshold):
    y = ( Amplitude * (Bias + ( (1-Bias) / (1 + np.exp(-Slope*(x-Threshold))))) )
    return y  

def boltzmann_sigmoid_mep(mep_50, mep_x, m):
    y = 1/ (1 + np.exp(-m*(mep_50 - mep_x)))
    return y
 
def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return y
       
def io_curve(norm_mep, optimize = False, plot = True):
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
        edcepochs, Vpp, sf, cmc, intensity, bool_mep = process_rawXDF(file, paired=paired)
        results["Vpp"].extend(Vpp)
        results["Vpp_True"].extend(bool_mep)
        results["logVpp"].extend(np.log(Vpp))
        results["Intensity"].extend([intensity]*len(Vpp))
        results["Subject"].extend([subjname]*len(Vpp))
        results["Night"].extend([night]*len(Vpp))
        results["Condition"].extend([condition]*len(Vpp))
        results["Protocol"].extend([protocol]*len(Vpp))
        results["Session"].extend([session]*len(Vpp))
        results["File"].extend([rec]*len(Vpp))
        # results["CMC"].extend([cmc]*len(Vpp))
    
        # plt.figure()
        # plt.title(" ".join(file.split('/')[-2::]))
        # plt.plot(np.arange(-max(edcepochs.shape)/2, max(edcepochs.shape)/2)/sf, np.mean(edcepochs,0))
        # if paired:
        #     entry = 0.0125
        # else:
        #     entry = 0.0115
        # plt.vlines(entry, ymin=np.mean(edcepochs,0).min(), ymax=np.mean(edcepochs,0).max())
        # plt.vlines(0.060, ymin=np.mean(edcepochs,0).min(), ymax=np.mean(edcepochs,0).max())
        
    res_df = pd.DataFrame(results)
    
    # drop icf & sici
    ind_drop = res_df[res_df['Protocol'].apply(lambda x: x.startswith('sici') or x.startswith('icf'))].index
    io_df = res_df.drop(ind_drop)
    
    # group 90 - 150 I/O MEP responses 
    io_df['Protocol'] = io_df['Protocol'].str.replace('cse', '', regex=True).astype(int)
    group_pre_post = io_df[io_df['Vpp_True']==True].groupby(['Subject','Condition','Session','Protocol']).mean()
    
    # normalize meps by subject trial 
    def mep_normalization(mep_values : list):
        all_norm_meps = [(mep_values[i] - np.nanmin(mep_values[i])) / 
                         ((np.nanmax(mep_values[i])) - np.nanmin(mep_values[i])) for i in range(len(mep_values))]
        
        return all_norm_meps
 
    ##### 
    groupies, labels = [], []
    for idx, group in enumerate(group_pre_post.groupby(['Subject','Condition','Session'])):
        # print(group[1]['Vpp'])
        groupies.append(group[1]['Vpp'].to_numpy())
        labels.append(list(group[1].index[0:]))
    norm_mep = np.concatenate(mep_normalization(groupies))
    # labels = np.concatenate(labels)
    
    group_pre_post['Normalized MEPs'] = norm_mep

    final_res = group_pre_post.groupby(['Condition','Session','Protocol']).mean()['Normalized MEPs'].reset_index()
    
    ##### Plot I/O curves 
    sns.set_theme(color_codes=True)
    plt.figure()
    
    for idx, condition in enumerate(zip((['up_pre','up_post'],['down_pre','down_post'],['sham_pre','sham_post']))):
        print(idx, condition)
        sns.regplot(x='Protocol', y="Normalized MEPs", 
                    data=group_pre_post[group_pre_post['Condition']==condition[0][0].split('_')[0]][group_pre_post['Session']==condition[0][0].split('_')[1]],
                    scatter_kws={"s": 80}, order=4, ci=95, x_estimator=np.mean, label=condition[0][0])
        sns.regplot(x='Protocol', y="Normalized MEPs", 
                    data=group_pre_post[group_pre_post['Condition']==condition[0][1].split('_')[0]][group_pre_post['Session']==condition[0][1].split('_')[0]],
                    scatter_kws={"s": 80}, order=4, ci=95, x_estimator=np.mean, label=condition[0][1])
        plt.legend()
    
    ## Sigmoidal plotting needs work, above is a linear solution with 4th degree polynomial fit
    # ax1 = io_curve(norm_mep, optimize=True, plot=True)
    
    return final_res


#%%
# maindir = '/media/administrator/data/Study_1_data/Pre_post_data/'
if __name__ == '__main__':
    TMS_results(maindir)
    
    