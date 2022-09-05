#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:31:54 2022

@author: administrator
"""

###################################################################################################
# SLOW-WAVE ACTIVITY EXPONENTIAL DECAY
# see: https://github.com/raphaelvallat/yasa/blob/swa_decay/notebooks/feature_swa_decay.ipynb
# and: https://github.com/raphaelvallat/yasa/blob/swa_decay/yasa/spectral.py
###################################################################################################

import yasa 
import mne
import logging
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import simps
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit, OptimizeWarning
from yasa.io import set_log_level

logger = logging.getLogger('yasa')


__all__ = ['swa_decay']

def find_runs(x):
    """Find runs of consecutive items in an array.
    From https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """
    n = x.shape[0]
    # Find run starts
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]
    # Find run values
    run_values = x[loc_run_start]
    # Find run lengths
    run_lengths = np.diff(np.append(run_starts, n))
    return pd.DataFrame({'values': run_values, 'start': run_starts, 
                         'length': run_lengths})


def _decay_func(t, asym, intercept, tau):
    """Exponential decay equation.
    S(t) = (S_sleeponset - lower_asym) * exp(-t / tau) + lower_asym
    Parameters
    ----------
    t : array_like
        Time
    asym : float
        Asymptote. Must be between 0 and 1 if evaluating relative power. Expressed in units of
        SWA relative power.
    intercept : float
        Intercept. Must be between 0 and 1 if evaluating relative power. Expressed in units of
        SWA relative power.
    tau : float
        Time constant (in hours). Must be between 0 to 4 hours to stay within
        physiological range. Higher values = slower homeostasis decay (i.e. closer to a
        flat line).
    Notes
    -----
    The initial guess for the parameters are 0.5 for the asymptote, 0.8 for the intercept
    and 1 hour for the time constant.
    """
    return (intercept - asym) * np.exp(- t / tau) + asym


def swa_decay(data, hypno, *, sf=None, ch_names=None, include=(2, 3), freq_swa=(0.5, 4),
              freq_broad=(0.5, 30), epoch_length="5min", win_sec=4, bandpass=True,
              kwargs_welch=dict(average='median', window='hamming'), verbose=True):
    r"""
    Calculate the exponential decline of process S across the night using NREM sleep EEG slow-wave
    activity (SWA).
    .. versionadded:: 0.6.1
    Parameters
    ----------
    data : array_like or :py:class:`mne.io.BaseRaw`
        1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which case ``data``,
        ``sf``, and ``ch_names`` will be automatically extracted, and ``data`` will also be
        converted from Volts (MNE default) to micro-Volts (YASA).
    hypno : array_like
        Sleep stage (hypnogram). The hypnogram must have the exact same number of samples as
        ``data``. To upsample your hypnogram, please refer to
        :py:func:`yasa.hypno_upsample_to_data`.
        .. note::
            The default hypnogram format in YASA is a 1D integer vector where:
            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    sf : float
        The sampling frequency of data AND the hypnogram. Can be omitted if ``data`` is a
        :py:class:`mne.io.BaseRaw`. The sampling frequency must be a whole number.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None, channels will be labelled
        ['CHAN000', 'CHAN001', ...]. Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    include : tuple, list or int
        Values in ``hypno`` that will be included in the mask. The default is (2, 3), meaning that
        the SWA exponential decline will be calculated on a mask consisting of N2 and N3 sleep.
    freq_swa : list of tuples
        Frequency range of slow-wave activity (SWA). Default is 0.5 to 4 Hz.
    freq_broad : list of tuples
        Frequency range of broadband signal. Default is 0.5 to 30 Hz.
    epoch_length : string
        A string representing the minimum duration of the NREM epochs that will be included in the
        SWA calculation. Default is "5min", i.e. at least 5 minutes of consecutive NREM.
    win_sec : int or float
        The length of the sliding window, in seconds, used for the Welch PSD calculation.
        Ideally, this should be at least two times the inverse of the lower frequency of
        interest (e.g. for a lower frequency of interest of 0.5 Hz, the window length should
        be at least 2 * 1 / 0.5 = 4 seconds).
    bandpass : boolean
        If True (default), apply a standard FIR bandpass filter in the ``freq_broad`` band.
        For more details, refer to :py:func:`mne.filter.filter_data`.
    kwargs_welch : dict
        Optional keywords arguments that are passed to the :py:func:`scipy.signal.welch` function.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``).
    Returns
    -------
    swa_decay : :py:class:`pandas.DataFrame`
        Exponential SWA decay values for each channel.
        * ``'Intercept'``: Estimated relative SWA power at sleep onset.
        * ``'Asym'``: Estimated relative SWA power at sleep offset.
        * ``'Tau'`` : Time constant :math:`\tau` of the exponential decline, in hours.
        * ``'Decay'``: Exponential slope (= 1 / :math:`\tau`). Larger values indicate a more rapid
          decay of SWA across the night.
        * ``'MAE'``: Mean absolute error of the exponential fit. Lower is better.
    Notes
    -----
    The homeostatic decay of Process S during sleep is modeled using the following equation [1]_:
    .. math:: S(t) = (I - A) * \exp \left ( -\frac{t}{\tau} \right ) + A
    where,
    * :math:`I` is the intercept, i.e. SWA level at sleep onset, defined as the first epoch of
      N2 or N3 sleep
    * :math:`A` is the lower asymptote
    * :math:`\tau` is the time constant of the decreasing exponential function during sleep.
    :math:`I` and :math:`A` are expressed in relative SWA power [2]_ and thus are bounded between
    0 and 1. The time constant :math:`\tau` is expressed in hours and bounded within the
    physiological range of 0 to 4 hours.
    Parameter estimation is performed using non-linear least squares curve fitting
    (see :py:func:`scipy.optimize.curve_fit`).
    References
    ----------
    .. [1] Rusterholz, Dürr & Achermann (2010). Inter-individual differences in the dynamics of
       sleep homeostasis. Sleep. https://doi.org/10.1093/sleep/33.4.491
    .. [2] Robillard et al (2010). Topography of homeostatic sleep pressure dissipation across the
       night in young and middle-aged men and women. Journal of sleep research.
       https://pubmed.ncbi.nlm.nih.gov/20408933/
    Examples
    --------
    Simulated exponential decline with different :math:`\tau` parameters. The asymptote :math:`A`
    and intercept :math:`I` are fixed at 0.5 and 0.9 relative SWA power, respectively.
    .. plot::
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(font_scale=1.25)
        # Define exponential function
        def func(t, asym, intercept, tau):
            return (intercept - asym) * np.exp(- t / tau) + asym
        # Define x-data (one point every hour)
        x = np.arange(0, 9, 1)
        # Plot
        pal = sns.color_palette("Blues_d", n_colors=4)
        plt.figure(figsize=(5, 5))
        plt.plot(x, func(x, *[0.5, 0.9, 4]), marker="o", color=pal[0], label="$\\tau = 4$");
        plt.plot(x, func(x, *[0.5, 0.9, 2]), marker="o", color=pal[1], label="$\\tau = 2$");
        plt.plot(x, func(x, *[0.5, 0.9, 1]), marker="o", color=pal[2], label="$\\tau = 1$");
        plt.plot(x, func(x, *[0.5, 0.9, 0.5]), marker="o", color=pal[3], label="$\\tau = 0.5$");
        plt.ylim(0, 1)
        plt.xlim(0, None)
        plt.xlabel("Time (hours)")
        plt.ylabel("Relative SWA (0.5-4 Hz) power")
        plt.title("Simulated exponential decline", fontweight="bold")
        plt.legend(frameon=True, loc="lower right")
        plt.tight_layout()
    """
    ###############################################################################################
    # PREPROCESSING
    ###############################################################################################

    set_log_level(verbose)
    assert isinstance(freq_swa, (tuple, list)), 'band must be a list or a tuple'
    assert isinstance(freq_broad, (tuple, list)), 'band must be a list or a tuple'
    assert isinstance(bandpass, bool), 'bandpass must be a boolean'
    assert isinstance(epoch_length, str), "epoch_length must be a string."

    # Check if input data is a MNE Raw object
    if isinstance(data, mne.io.BaseRaw):
        sf = data.info['sfreq']  # Extract sampling frequency
        ch_names = data.ch_names  # Extract channel names
        data = data.get_data() * 1e6  # Convert from V to uV
        _, npts = data.shape
    else:
        # Safety checks
        assert isinstance(data, np.ndarray), 'Data must be a numpy array.'
        data = np.atleast_2d(data)
        assert data.ndim == 2, 'Data must be of shape (nchan, n_samples).'
        nchan, npts = data.shape
        # assert nchan < npts, 'Data must be of shape (nchan, n_samples).'
        assert sf is not None, 'sf must be specified if passing a numpy array.'
        assert isinstance(sf, (int, float))
        if ch_names is None:
            ch_names = ['CHAN' + str(i).zfill(3) for i in range(nchan)]
        else:
            ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
            assert ch_names.ndim == 1, 'ch_names must be 1D.'
            assert len(ch_names) == nchan, 'ch_names must match data.shape[0].'

    if bandpass:
        # Apply FIR bandpass filter
        fmin, fmax = min(freq_broad), max(freq_broad)
        data = mne.filter.filter_data(data, sf, fmin, fmax, verbose=0)

    # More safety checks for hypno, include and sf
    hypno = np.asarray(hypno)
    assert include is not None, 'include cannot be None if hypno is given'
    include = np.atleast_1d(np.asarray(include))
    assert hypno.ndim == 1, 'Hypno must be a 1D array.'
    assert hypno.size == npts, 'Hypno must have same size as data.shape[1]'
    assert include.size >= 1, '`include` must have at least one element.'
    #assert hypno.dtype.kind == include.dtype.kind, 'hypno and include must have same dtype'
    assert np.in1d(hypno, include).any(), (
        'None of the stages specified in `include` are present in hypno.')
    assert float(sf).is_integer(), "Sampling frequency must be an integer."
    mask = np.in1d(hypno, include).astype(int)

    ###############################################################################################
    # FIND "CLEAN" NREM PERIODS, i.e. at least X min of consecutive NREM
    ###############################################################################################

    # IMPORTANT: Sleep onset is defined as the first epoch of N2 or N3 sleep
    # TODO: Should we use the first two consecutive N2 or N3 sleep epoch?
    idx_onset = np.nonzero(np.in1d(hypno, (2, 3)))[0][0]

    # How many samples is 5 /10 / X min?
    n_samples_epoch = int(pd.Timedelta(epoch_length).seconds * sf)  # noqa

    # METHOD 1: variable-length epochs
    # This show the onset and duration (in samples) of all the NREM epochs that are > 5 min
    epochs = find_runs(mask).query(
        "values == 1 and length > @n_samples_epoch").reset_index(drop=True)

    # METHOD 2: equal-length epochs
    # n_windows = int(np.floor(mask[idx_onset:].size / n_samples_epoch))
    # epochs = {'start': [], 'length': []}
    # for i in range(n_windows):
    #     start = idx_onset + n_samples_epoch * i
    #     end = start + n_samples_epoch
    #     if mask[start:end].all():
    #         epochs['start'].append(start)
    #         epochs['length'].append(end - start)
    # epochs = pd.DataFrame(epochs)

    # Check that we have enough epochs
    n_epochs = epochs.shape[0]
    logger.info(
        f"{n_epochs} NREM epochs longer than {epoch_length} were found in hypno.")
    if n_epochs < 4:
        print(f"Less than 4 NREM epochs > {epoch_length} were found in hypno")
    #     raise ValueError(
    #         f"Less than 4 NREM epochs > {epoch_length} were found in hypno. SWA decay cannot be "
    #         f"calculated. Please decrease {epoch_length}.")

    # Calculate the onset time (relative to sleep onset) of each NREM period
    epochs['time_onset_hrs'] = (epochs['start'] - idx_onset) / sf / 3600
    epochs['length_hrs'] = epochs['length'] / 2 / sf / 3600
    epochs['time_mid_hrs'] = epochs['time_onset_hrs'] + epochs['length_hrs']

    # Calculate continuous mask with unique value for each epoch
    mask_epoch = np.zeros_like(mask, dtype=int)
    for i, row in epochs.iterrows():
        start = int(row['start'])
        end = int(row['start'] + row['length'])
        mask_epoch[start:end] = i + 1

    ###############################################################################################
    # EXPONENTIAL DECLINE IN SWA, for each channel separately
    ###############################################################################################

    # Calculate SWA absolute power in each period, for each channel
    bp = yasa.bandpower(
        data=data, sf=sf, ch_names=ch_names, hypno=mask_epoch,
        include=list(range(1, max(mask_epoch) + 1)),
        bands=[(freq_swa[0], freq_swa[1], "SWA"), (freq_broad[0], freq_broad[1], "Broad")],
        win_sec=win_sec, relative=True, bandpass=False, kwargs_welch=kwargs_welch)

    # Initialize output
    df_decay = {"Intercept": [], "Asym": [], "Tau": [], "Decay": [], "MAE": [],
                "X-data": [], "Y-data": []}

    # Calculate exponential decline, for each channel
    # Note that we use the midpoint of each epoch as the xdata, and not the onset, to account for
    # different epoch durations.
    xdata = epochs['time_mid_hrs'].to_numpy()
    for chan in ch_names:
        ydata = bp.xs(chan, level=-1)["SWA"].to_numpy()
        try:
            # See docstring of "_decay_func" for an explanation of the bound.
            popt, _ = curve_fit(
                _decay_func, xdata, ydata, p0=(0.5, 0.8, 1), bounds=((0, 0, 0), (1, 1, 4)))
            mae = np.mean(np.abs(ydata - _decay_func(xdata, *popt)))
        except (ValueError, RuntimeError, OptimizeWarning) as e:
            logger.error(f"Exponential fit failed. Returning NaN for channel {chan}\nError: {e}")
            popt = np.array([np.nan, np.nan, np.nan])
            mae = np.nan

        # Append to dict
        df_decay["Asym"].append(popt[0])
        df_decay["Intercept"].append(popt[1])
        df_decay["Tau"].append(popt[2])
        df_decay["Decay"].append(1 / popt[2])
        df_decay["MAE"].append(mae)
        df_decay["X-data"].append(xdata)
        df_decay["Y-data"].append(ydata)

    # Convert to dataframe
    return pd.DataFrame(df_decay, index=ch_names)