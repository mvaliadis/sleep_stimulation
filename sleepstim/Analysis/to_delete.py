#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:45:36 2022

@author: administrator
"""

def save_edfs():
    path = '/media/administrator/data/Study_1_data/Pre-processed_data/Experimental_auditory_validation/'
    maindir = sorted([os.path.join(folder,i) for folder, subdirs, files in os.walk(path) for i in files if 'av.p' in i])
    for i, files in enumerate(tqdm(maindir)):
        cond = subject_cond_parser(files, study_phase = 'sleep')
        subject = files.split("/")[-1].split('_')[0]
        night = files.split("/")[-1].split('_')[1]
        hypno_path = '/media/administrator/data/Study_1_data/Hypnograms/Experimental/' + subject + '_' + files.split('/')[-1].split('_')[1]
        if os.path.exists(hypno_path + '_hypno.txt'):
            print(f'Self-Scored Hypnogram Utilized for : {files.split("/")[-1]}')
        else:
            # load data
            Data = load_preprocessed_data(files)[0]
            # take first (adjusted) pinknoise bursts as center point
            burst_range = Data.pinknoise_times_sync[0:-1]
            if len(burst_range) != 0:
                print(f'YASA-Classfier Hypnogram Utilized for : {files.split("/")[-1]}')
                # load bad channel info before epoching
                data_sheet = pd.read_csv('/media/administrator/data/Study_1_data/Data_tracking/epoch_reports.csv')
                mask = np.logical_and(data_sheet['Subject']==subject, 
                                      data_sheet['Condition']==cond)
                bad_items = data_sheet[mask]['Bad Channels'].to_numpy()
                if len(bad_items) > 0:
                    Data.bad_chans = bad_items[0].split("'")[1::2]
                else:
                    Data.bad_chans = []
                ## Data length based
                # create mne epoch object
                info = mne.create_info(ch_names=Data.chans, sfreq=Data.sfreq, ch_types=Data.chtypes)  
                info['bads'] = Data.bad_chans        
                # create epochs with online reference
                epochs = mne.io.RawArray(Data.data.T/1e6, info)

#%%
bandpower = yasa.bandpower(epochs.copy().pick('eeg')) #hypno = hypno_pred
fig = yasa.topoplot(bandpower['Delta'])
