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

def oscillatory_plot_psd_map(epochs, foi=(0.5, 30), tmin=-3, tmax=0, session='sleep', plot = False):
    
    # Calculate power spectra across the the continuous data
    spectra, freqs = psd_multitaper(epochs, fmin=foi[0], fmax=foi[1],
                                    tmin=tmin, tmax=tmax)
    
    # Fitting Power Spectrum Models
    # -----------------------------
    # Now that we have power spectra, we can fit some power spectrum models.
    # Since we have multiple power spectra, we will use the :class:`~fooof.FOOOFGroup` object.
    
    # Initialize a FOOOFGroup object, with desired settings
    fg = FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
                    peak_threshold=2., max_n_peaks=6, verbose=False)
    
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
    if session=='sleep':
        bands = Bands({'delta': [0.5, 4],
                       'theta': [4, 8],
                       'alpha': [8, 12],
                       'sigma': [12, 16],
                       'beta': [16, 30]})
    else:
        bands = Bands({'delta': [1, 4],
                       'theta': [4, 8],
                       'alpha': [8, 12],
                       'beta': [12, 30],
                       'gamma': [30, 40]})
    
    def oscillatory_psd(fg, band_def):
        band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])
        return band_power
    
    # Plot the topographies across different frequency bands
    band_powers = []
    if plot:
        fig, axes = plt.subplots(1, 5, figsize=(2 * 5, 1.5))
        
    for ind, (label, band_def) in enumerate(bands):
    
        # Get the power values across channels for the current band
        band_power = oscillatory_psd(fg, band_def)
        band_powers.append(band_power) 
        
        if plot:
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

def periodic_fit(epochs, foi=(1, 40), tmin=0, tmax=2):
    ################################################################################################
    # Algorithmic Description
    # -----------------------
    #
    # In this tutorial we will step through how the power spectrum model is fit.
    #
    # Note that this notebook is for demonstrative purposes, and does not represent
    # recommended usage of how to fit power spectrum models.
    #
    # Broadly, the steps in the algorithm are:
    #
    # - 1) An initial fit of the aperiodic component is computed from the power spectrum
    # - 2) This aperiodic fit is subtracted from the power spectrum, creating a flattened spectrum
    # - 3) An iterative process identifies peaks in this flattened spectrum
    # - 4) A full peak fit is re-fit from all of the identified peak candidates
    # - 5) The peak fit is subtracted from the original power spectrum,
    #      creating a peak-removed power spectrum
    # - 6) A final fit of the aperiodic component is taken of the peak-removed power spectrum
    # - 7) The full model is reconstructed from the combination of the aperiodic and peak fits,
    #      and goodness of fit metrics are calculated.
    #
    
    ###################################################################################################
    
    # sphinx_gallery_thumbnail_number = 4
    
    # General imports
    import matplotlib.pyplot as plt
    
    # Import the FOOOF object
    from fooof import FOOOF, FOOOFGroup
    # Import utilities for working with FOOOF objects
    from fooof.objs import fit_fooof_3d, combine_fooofs
    
    # Import some internal functions
    #   These are used here to demonstrate the algorithm
    #   You do not need to import these functions for standard usage of the module
    from fooof.sim.gen import gen_aperiodic
    from fooof.plts.spectra import plot_spectrum
    from fooof.plts.annotate import plot_annotated_peak_search
    
    # Import a utility to download and load example data
    from fooof.utils.download import load_fooof_data
    from fooof.objs.utils import average_fg
    
    ###################################################################################################
    
    # Set whether to plot in log-log space
    plt_log = False
    
    ###################################################################################################
    
    # Calculate power spectra across the the continuous data
    spectrum, freqs = psd_multitaper(epochs, fmin=foi[0], fmax=foi[1],
                                     tmin=tmin, tmax=tmax)
    
    # spectrum = spectrum.mean(0).mean(0)
    
    fms = []
    for model in range(spectrum.shape[0]):

        ###################################################################################################
        
        # Initialize a FOOOF object, with some settings
        #   These settings will be more fully described later in the tutorials
        fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.15)
        # fms = fit_fooof_3d(fm, freqs, spectrum, freq_range=foi, n_jobs=-1)  
        
        ###################################################################################################
        #
        # Note that data can be added to a FOOOF object independent of fitting the model, using the
        # :meth:`~fooof.FOOOF.add_data` method. FOOOF objects can also be used to plot data,
        # prior to fitting any models.
        #
        
        ###################################################################################################
        
        # Add data to the object
        fm.add_data(freqs, spectrum[model,:,:].mean(0), foi)
        
        ###################################################################################################
        
        # Plot the power spectrum
        # fm.plot(plt_log)
        
        ###################################################################################################
        #
        # The FOOOF object stores most of the intermediate steps internally.
        #
        # For this notebook, we will first fit the full model, as normal, but then step through,
        # and visualize each step the algorithm took to come to that final fit.
        #
        
        ###################################################################################################
        
        # Fit the power spectrum model
        fm.fit(freqs, spectrum[model,:,:].mean(0), foi)
        
        ###################################################################################################
        # Step 1: Initial Aperiodic Fit
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #
        # We start by taking an initial aperiodic fit. This goal of this fit is to get an initial
        # fit that is good enough to get started with the fitting process.
        #
        
        ###################################################################################################
        
        # Do an initial aperiodic fit - a robust fit, that excludes outliers
        #   This recreates an initial fit that isn't ultimately stored in the FOOOF object
        init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
        
        # Plot the initial aperiodic fit
        # _, ax = plt.subplots(figsize=(12, 10))
        # plot_spectrum(fm.freqs, fm.power_spectrum, plt_log,
        #               label='Original Power Spectrum', color='black', ax=ax)
        # plot_spectrum(fm.freqs, init_ap_fit, plt_log, label='Initial Aperiodic Fit',
        #               color='blue', alpha=0.5, linestyle='dashed', ax=ax)
        
        ###################################################################################################
        # Step 2: Flatten the Spectrum
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #
        # The initial fit is then used to create a flattened spectrum.
        #
        # The initial aperiodic fit is subtracted out from the original data, leaving a flattened
        # version of the data which no longer contains the aperiodic component.
        #
        
        ###################################################################################################
        
        # Recompute the flattened spectrum using the initial aperiodic fit
        init_flat_spec = fm.power_spectrum - init_ap_fit
        
        # Plot the flattened the power spectrum
        # plot_spectrum(fm.freqs, init_flat_spec, plt_log,
        #               label='Flattened Spectrum', color='black')
        
        ###################################################################################################
        # Step 3: Detect Peaks
        # ^^^^^^^^^^^^^^^^^^^^
        #
        # The flattened spectrum is then used to detect peaks. We can better isolate
        # peaks in the data, as the aperiodic activity has been removed.
        #
        # The fitting algorithm uses an iterative procedure to find peaks in the flattened spectrum.
        #
        # For each iteration:
        #
        # - The maximum point of the flattened spectrum is found
        #
        #   - If this point fails to pass the relative or absolute height threshold,
        #     the procedure halts
        # - A Gaussian is fit around this maximum point
        # - This 'guess' Gaussian is then subtracted from the flatted spectrum
        # - The procedure continues to a new iteration with the new version of the flattened spectrum,
        #   unless `max_n_peaks` has been reached
        #
        
        ###################################################################################################
        
        # Plot the iterative approach to finding peaks from the flattened spectrum
        # plot_annotated_peak_search(fm)
        
        ###################################################################################################
        # Step 4: Create Full Peak Fit
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #
        # Once the iterative procedure has halted and the peaks have been identified in the
        # flattened spectrum, the set of identified 'guess' peaks, are then re-fit, all together.
        # This creates the full peak fit of the data.
        #
        
        ###################################################################################################
        
        # Plot the peak fit: created by re-fitting all of the candidate peaks together
        # plot_spectrum(fm.freqs, fm._peak_fit, plt_log, color='green', label='Final Periodic Fit')
        
        ###################################################################################################
        # Step 5: Create a Peak-Removed Spectrum
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #
        # Now that the peak component of the fit is completed and available, this fit is then
        # used in order to try and isolate a better aperiodic fit.
        #
        # To do so, the peak fit is removed from the original power spectrum,
        # leaving an 'aperiodic-only' spectrum for re-fitting.
        #
        
        ###################################################################################################
        
        # Plot the peak removed power spectrum, created by removing peak fit from original spectrum
        # plot_spectrum(fm.freqs, fm._spectrum_peak_rm, plt_log,
        #               label='Peak Removed Spectrum', color='black')
        
        ###################################################################################################
        # Step 6: Re-fit the Aperiodic Component
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #
        # The initial aperiodic component fit we made was a robust fit approach that was
        # used to get the fitting process started.
        #
        # With the peak-removed spectrum, we can now re-fit the aperiodic component, to
        # re-estimate a better fit, without the peaks getting in the way.
        #
        
        ###################################################################################################
        
        # Plot the final aperiodic fit, calculated on the peak removed power spectrum
        # _, ax = plt.subplots(figsize=(12, 10))
        # plot_spectrum(fm.freqs, fm._spectrum_peak_rm, plt_log,
        #               label='Peak Removed Spectrum', color='black', ax=ax)
        # plot_spectrum(fm.freqs, fm._ap_fit, plt_log, label='Final Aperiodic Fit',
        #               color='blue', alpha=0.5, linestyle='dashed', ax=ax)
        
        ###################################################################################################
        # Step 7: Combine the Full Model Fit
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #
        # Now that we have the final aperiodic fit, we can combine the aperiodic components
        # to create the full model fit.
        #
        # With this full model fit, we can also calculate the goodness of fit metrics,
        # including the error of the fit and the R-squared of the fit, by comparing the
        # full model fit to the original data.
        #
        
        ###################################################################################################
        
        # Plot full model, created by combining the peak and aperiodic fits
        # plot_spectrum(fm.freqs, fm.fooofed_spectrum_, plt_log,
        #               label='Full Model', color='red')
        
        ##################################################################################################
        #
        # The last stage is to calculate the goodness of fit metrics, meaning the fit error & R^2.
        #
        # At the end of the fitting process, the model object also organizes parameters, such as
        # updating gaussian parameters to be peak parameters,
        #
        # These results are part of what are stored, and printed, as the model results.
        #
        
        ###################################################################################################
        
        # Print out the model results
        # fm.print_results()
        
        ###################################################################################################
        #
        # Altogether, the full model fit is now available, and can be plotted.
        #
        
        ###################################################################################################
        
        # Plot the full model fit of the power spectrum
        #  The final fit (red), and aperiodic fit (blue), are the same as we plotted above
        # fm.plot(plt_log)
        
        ###################################################################################################
        # Addendum: Data & Model Component Attributes
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #
        # As you may have noticed through this tutorial, the :class:`~fooof.FOOOF` object keeps
        # track of some versions of the original data as well as individual model components fits,
        # as well as the final model fit, the ultimate outcome of the fitting procedure.
        #
        # These attributes in the FOOOF object are kept at the end of the fitting procedure.
        # Though they are primarily computed for internal use (hence being considered 'private'
        # attributes, with the leading underscore), they are accessible and potentially
        # useful for some analyses, and so are briefly described here.
        #
        # Stored model components:
        #
        # - Aperiodic Component: ``_ap_fit``
        #
        #   - This is the aperiodic-only fit of the data.
        #   - It is computed by generating a reconstruction of the measured aperiodic parameters
        #
        # - Periodic Component: ``_peak_fit``
        #
        #   - This is the periodic-only (or peak) fit of the data.
        #   - It is computed by generating a reconstruction of the measured periodic (peak) parameters
        #
        # Stored data attributes:
        #
        # - Flattened Spectrum: ``_spectrum_flat``
        #
        #   - The original data, with the aperiodic component removed
        #   - This is computed as ``power_spectrum`` - ``_ap_fit``
        #
        # - Peak Removed Spectrum: ``_spectrum_peak_rm``
        #
        #   - The original data, with the periodic component (peaks) removed
        #   - This is computed as ``power_spectrum`` - ``_peak_fit``
        #
        
        ###################################################################################################
        # Conclusion
        # ----------
        #
        # In this tutorial we have stepped through the parameterization algorithm for fitting
        # power spectrum models.
        #
        # Next, we will continue to explore the FOOOF object by properly introducing and more
        # fully describing the settings for the algorithm.
        #
        
        fms.append(fm)
    
    return fms


