#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:08:02 2024

@author: administrator
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import yasa
mne.set_log_level('ERROR')

def group_files_by_subject_night(files):
    """
    Group files by subject and night.
    """
    file_groups = {}
    for file in files:
        parts = file.split('/')[-1].split('-')[0].split('_')
        subject_night = '_'.join(parts[:2])  # Assuming first two parts are subject and night
        if subject_night not in file_groups:
            file_groups[subject_night] = []
        file_groups[subject_night].append(file)
    return file_groups

def postprocess_and_combine_raw_files(file_group, save_path, 
                                      hypno_path, fig_path, save=True):
    """
    Load Raw files for a group of files belonging to the same subject and night, and combine them.
    """
    raw_list = []
    for file in file_group:
        print(f"Loading Raw file: {file}")
        raw = mne.io.read_raw_fif(file, preload=True)
        # append data
        raw_list.append(raw)
    
    # Combine the processed raws
    subject, night = file_group[0].split('/')[-1].split('_')[:2]
    save_path = save_path + f'{subject}_{night}_csd-raw.fif'
    hypno_path = hypno_path + f'{subject}_{night}' 
    if save:   
        if len(raw_list) > 1:
            raw = mne.concatenate_raws(raw_list)
        else:
            raw = raw
                    
        # apply sleep stage classification 
        sls = yasa.SleepStaging(raw, eeg_name="C3", eog_name="EOG", 
                                #emg_name="EMG_L"
                                )
        # get the predicted sleep stages
        hypno = yasa.Hypnogram(sls.predict())
        # hypnogram = hypno.upsample_to_data(raw)
        
        # artifact detection
        roi = ['F4','Fz','F3','C4','Cz','C3','P4','Pz','P3']
        art_epochs, _ = yasa.art_detect(raw.copy().pick(roi), 
                                        hypno=hypno.upsample_to_data(raw),
                                        window=5, method="covar")
        art_up = yasa.hypno_upsample_to_data(art_epochs, sf_hypno=1/5, data=raw)
        # Add -1 to hypnogram where artifacts were detected
        hypno_with_art = hypno.upsample_to_data(raw).copy()
        hypno_with_art[art_up] = -1
        
        # save hypnogram, no annotations in raw; too heavy  
        np.save(hypno_path + '_hypno.npy', hypno_with_art)
        np.save(hypno_path + '_hypno30s.npy', hypno.as_int().to_numpy())
        
        # raw.annotations.append(onset=new_annotations['onset'],
        #                        duration=new_annotations['duration'],
        #                        description=new_annotations['description'])
        
        # apply CSD to data 
        raw = mne.preprocessing.compute_current_source_density(raw)
        
        # 2. Load and plot hypnogram (with spectrogram of ROI average)
        fig = yasa.plot_spectrogram(data=raw.get_data(picks=roi).mean(0)*1e3, 
                                    sf=raw.info['sfreq'], hypno=hypno_with_art, 
                                    **dict(lw=1))  
        # Add title based on filename
        plt.suptitle(f'Subject: {subject}, Night: {night} hypnogram', fontsize=16)
        
        # Save fig and close
        fig.savefig(fig_path + f'{subject}_{night}_spectro_hypno.png')
        plt.close('all')
        
        # save data 
        print(f"Saving processed raw to: {save_path}")
        raw.save(save_path, overwrite=True)
        
        return raw
   
def graveyard_code(raw):
    import pickle
    rf = pickle.load(open('/home/administrator/sleep_stimulation-development/rf_model_5_cfs.p','rb'))

    _, epochs_class = yasa.sliding_window(raw.get_data(['C3','C4'], units='uV'), 
                                          sf=raw.info['sfreq'], window=30)
    
    from sleepstim.sleep_funs import bandpower
    bp = bandpower(epochs_class, fs=raw.info['sfreq'], 
                   bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                          (8, 12, 'Alpha'), (12, 16, 'Sigma'), 
                          (16, 30, 'Beta'), (49, 51, 'Line noise')],
                   relative=True)
       
    # reshape data for classifier, must be (epochs x (nchans*bands))
    bp = np.swapaxes(bp, 0, 1)
    nepochs, nbands, nchans = np.shape(bp)
    bp = bp.reshape(nepochs, nchans*nbands, order='F')
    
    C3 = bp[:, 0:5]
    C4 = bp[:, 6:11]
    
    X_test = np.concatenate([C3, C4], axis=1)
    y_pred = rf.predict(X_test)
    
    #fig, axes = plt.subplots(nrows=2, figsize=(6, 4), constrained_layout=True)
    hypno = yasa.Hypnogram(yasa.hypno_int_to_str(y_pred), n_stages=5, freq='30s')
    # ax1 = hypno.plot_hypnogram(lw=1, fill_color="whitesmoke", highlight=None, ax=axes[0])
    yasa.plot_spectrogram(data=raw.get_data(picks='Cz',units='uV').squeeze(), 
                          sf=raw.info['sfreq'], 
                          hypno=hypno.upsample_to_data(raw, sf=raw.info["sfreq"]),
                          **dict(lw=1))
                                                                             
    
    
    # Initialize the sleep staging instance
    sls = yasa.SleepStaging(raw, eeg_name='C3', 
                            eog_name='EOG', emg_name='EMG_L')
    
    # Get the predicted sleep stages
    hypno2 = yasa.Hypnogram(sls.predict(), n_stages=5, freq='30s')
    # ax2 = hypno2.plot_hypnogram(lw=1, fill_color="whitesmoke", highlight=None, ax=axes[1])
    yasa.plot_spectrogram(data=raw.get_data(picks='Cz',units='uV').squeeze(), 
                          sf=raw.info['sfreq'], 
                          hypno=hypno2.upsample_to_data(raw, sf=raw.info["sfreq"]),
                          **dict(lw=1))  
      
#%%
if __name__ == '__main__':
    path = '/media/administrator/Sleep_Data/Processed/Sleep/Intermediate/*-raw.fif'
    save_path = '/media/administrator/Sleep_Data/Processed/Sleep/Final/Raw/'
    hypno_path = '/media/administrator/Sleep_Data/Processed/Hypnograms/'
    fig_path = '/media/administrator/Sleep_Data/Processed/Figures/Sleep/'
    file_groups = group_files_by_subject_night(glob.glob(path))
    for subject_night, file_group in tqdm(file_groups.items()):
        print(f"Combining recordings, if necessary: {subject_night}")
        postprocess_and_combine_raw_files(file_group, save_path, 
                                          hypno_path, fig_path, save=True)
