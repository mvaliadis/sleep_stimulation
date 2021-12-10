#!/usr/bin/env python3
"""
Created on Fri Oct 15 14:46:16 2021

@author: administrator

Topographical Analyses with MNE
===============================

Parameterizing neural power spectra with MNE, doing a topographical analysis.

This tutorial requires that you have `MNE <https://mne-tools.github.io/>`_
installed.

If you don't already have MNE, you can follow instructions to get it
`here <https://mne-tools.github.io/stable/getting_started.html>`_.

For this example, we will explore how to parameterize power spectra using data loaded
and managed with MNE, and how to plot topographies of resulting model parameters.
"""

# General imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colorbar

# Import MNE, as well as the MNE sample dataset
import mne
from mne import io
from mne.datasets import sample
from mne.viz import plot_topomap
from mne.time_frequency import psd_multitaper

# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

#%%
###################################################################################################
# Dealing with NaN Values
# -----------------------
#
# One thing to keep in mind when parameterizing power spectra, and extracting bands of
# interest, is that there is no guarantee that the model will detect peaks in any given range.
#
# We consider this a pro, since power spectrum model is able to adjudicate whether there is
# evidence of oscillatory power within a given band, but it does also mean that sometimes
# results for a given band can be NaN, which doesn't always work very well with further
# analyses that we may want to do.
#
# To be able to deal with nan-values, we will define a helper function to
# check for NaN values and apply a specified policy for how to deal with them.
#

###################################################################################################

def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data

###################################################################################################
# Calculating Power Spectra
# -------------------------
#
# To fit power spectrum models, we need to convert the time-series data we have loaded in
# frequency representations - meaning we have to calculate power spectra.
#
# To do so, we will leverage the time frequency tools available with MNE,
# in the `time_frequency` module. In particular, we can use the ``psd_welch``
# function, that takes in MNE data objects and calculates and returns power spectra.
#

###################################################################################################

#%% 

def oscillatory_plot_psd_map(epochs, foi=(0.5, 30), tmin=-3, tmax=0):
    
    # Calculate power spectra across the the continuous data
    spectra, freqs = psd_multitaper(epochs, fmin=foi[0], fmax=foi[1],
                                    tmin=tmin, tmax=tmax)
    
    # Fitting Power Spectrum Models
    # -----------------------------
    # Now that we have power spectra, we can fit some power spectrum models.
    # Since we have multiple power spectra, we will use the :class:`~fooof.FOOOFGroup` object.
    
    # Initialize a FOOOFGroup object, with desired settings
    fg = FOOOFGroup()
    # fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
    #                 peak_threshold=2., max_n_peaks=6, verbose=False)
    
    # Define the frequency range to fit
    freq_range = foi
    
    ###################################################################################################
    
    # Fit the power spectrum model across all channels
    fg.fit(freqs, spectra.mean(0), foi)
    
    ###################################################################################################
    
    # Check the overall results of the group fits
    # fg.plot()
    
    ###################################################################################################
    # Plotting Topographies
    # ---------------------
    #
    # Now that we have our power spectrum models calculated across all channels,
    # let's start by plotting topographies of some of the resulting model parameters.
    #
    # To do so, we can leverage the fact that both MNE and FOOOF objects preserve data order.
    # So, when we calculated power spectra, our output spectra kept the channel order
    # that is described in the MNE data object, and so did our :class:`~fooof.FOOOFGroup`
    # object.
    #
    # That means that to plot our topography, we can use the MNE ``plot_topomap``
    # function, passing in extracted data for power spectrum parameters per channel, and
    # using the MNE object to define the corresponding channel locations.
    #
    
    ###################################################################################################
    # Plotting Periodic Topographies
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Lets start start by plotting some periodic model parameters.
    #
    # To do so, we will use to :obj:`~.Bands` object to manage some band
    # definitions, and some analysis utilities to extracts peaks from bands of interest.
    #
    
    ###################################################################################################
    
    # Define frequency bands of interest
    bands = Bands({'delta': [0.5, 4],
                   'theta': [4, 8],
                   'alpha': [8, 12],
                   'sigma': [12, 16],
                   'beta': [16, 30]})
    
    def oscillatory_psd(fg, band_def):
        band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])
        return band_power
    
    # Plot the topographies across different frequency bands
    band_powers = []
    fig, axes = plt.subplots(1, 5, figsize=(2 * 5, 1.5))
    for ind, (label, band_def) in enumerate(bands):
    
        # Get the power values across channels for the current band
        band_power = oscillatory_psd(fg, band_def)
        band_powers.append(band_power) 
        
        # Create a topomap for the current oscillation band
        im, cn = mne.viz.plot_topomap(band_power, epochs.info, cmap='Spectral_r', contours=0,
                                      axes=axes[ind], show=False);
        
        # add color bar
        mne.viz.topomap._add_colorbar(axes[ind], im, cmap = 'Spectral_r', side='right', pad=0.05, 
                                      title=None, format=None, size='5%')
    
        # Set the plot title
        axes[ind].set_title(label + ' power')
        
        # tighten layout
        plt.tight_layout()
    
    ###################################################################################################
    #
    # You might notice that the topographies of some of the bands look a little 'patchy'.
    # This is because we are setting any channels for which we did not find a peak as zero
    # with our `check_nan` approach. Note this is also a single subject analysis.
    #
    
    ###################################################################################################
    #
    # Since we have the power spectrum models for each of our channels, we can also explore
    # what these peaks look like in the underlying power spectra.
    #
    # Next, lets check the power spectra for the largest detected peaks within each band.
    #
    
    ###################################################################################################
    
    # fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    # for ind, (label, band_def) in enumerate(bands):
    
    #     # Get the power values across channels for the current band
    #     band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])
    
    #     # Extracted and plot the power spectrum model with the most band power
    #     fg.get_fooof(np.argmax(band_power)).plot(ax=axes[ind], add_legend=False)
    
    #     # Set some plot aesthetics & plot title
    #     axes[ind].yaxis.set_ticklabels([])
    #     axes[ind].set_title('biggest ' + label + ' peak', {'fontsize' : 16})
    
    ###################################################################################################
    # Plotting Aperiodic Topographies
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Next up, let's plot the topography of the aperiodic exponent.
    #
    # To do so, we can simply extract the aperiodic parameters from our power spectrum models,
    # and plot them.
    #
    
    ###################################################################################################
    
    # Extract aperiodic exponent values
    # exps = fg.get_params('aperiodic_params', 'exponent')
    
    ###################################################################################################
    
    # Plot the topography of aperiodic exponents
    # plot_topomap(exps, epochs.info, cmap=cm.Spectral_r, contours=0)

    ## Return bandpower values for topomaps  
    
    return band_powers

