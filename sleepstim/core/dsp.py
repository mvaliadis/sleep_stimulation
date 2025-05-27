# -*- coding: utf-8 -*-
"""Core Digital Signal Processing (DSP) functions for filtering, referencing, and spectral analysis in the sleepstim project."""

import numpy as np
from scipy import signal, special
import math
import mne # For bfr_butter_filt if method='mne', and potentially for types if not careful
import yasa # For bandpower

def surface_laplacian(data, coordinates, sf, reject=None):
    """
    This function attempts to compute the surface laplacian transform parsed and 
    filtered EEG data Perrin et al. (1989)
    
    INPUTS are:
        - data: channels x epochs x timepoints
        - leg_order: maximum order of the Legendre polynomial
        - m: smothness parameter for G and H
        - smoothing: smothness parameter for the diagonal of G
        - montage: montage to reconstruct the transformed Epochs object (same as in raw data import)
        
    OUTPUTS are:
        - before: sham reconstruction of the original Epochs object
        - after: surface laplacian transform of the original Epochs object
        
    References:
        - Perrin, F., Pernier, J., Bertrand, O. & Echallier, J.F. (1989). Spherical splines for scalp 
          potential and current density mapping. Electroencephalography and clinical Neurophysiology, 72, 
          184-187.
    """
    
    coordinates = coordinates.T

    # remove rejected channel
    if reject:
        coordinates = np.delete(coordinates, reject, axis=1)
        
    #sfreq = sf # sampling rate # sf is passed but not used in this specific function body
    
    # define functions
    leg_order = 5 # Max order of legendre polynomials
    m = 6 # Determines the smoothness of the G matrix
    smoothing = 1e-3 # Smoothing parameter for the diagonal of G
  
    # if data.shape[0] != 64: # This check might be too specific if num_channels varies
    #     raise ValueError("Please format data as channels x epochs x timepoints")
    
    # get electrodes positions
    
    x = coordinates[0,:]
    y = coordinates[1,:]
    z = coordinates[2,:]
    orig_data_size = np.squeeze(data.shape)
    numelectrodes = data.shape[0]
    
    # normalize cartesian coordenates to sphere unit
    def cart2sph(x_coord, y_coord, z_coord): # Renamed to avoid conflict
        hxy = np.hypot(x_coord, y_coord)
        r = np.hypot(hxy, z_coord)
        el = np.arctan2(z_coord, hxy)
        az = np.arctan2(y_coord, x_coord)
        return az, el, r
    
    junk1, junk2, spherical_radii = cart2sph(x,y,z)
    maxrad = np.max(spherical_radii)
    x = x/maxrad
    y = y/maxrad
    z = z/maxrad
    
    # compute cousine distance between all pairs of electrodes
    cosdist = np.zeros((numelectrodes, numelectrodes))
    for i in range(numelectrodes):
        for j in range(i+1,numelectrodes):
            cosdist[i,j] = 1 - (((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)/2)
    
    cosdist = cosdist + cosdist.T + np.identity(numelectrodes)
    
    # get legendre polynomials
    legpoly = np.zeros((leg_order, numelectrodes, numelectrodes))
    for ni in range(leg_order):
        for i in range(numelectrodes):
            for j in range(i+1, numelectrodes):
                #temp = special.lpn(8,cosdist[0,1])[0][8]
                legpoly[ni,i,j] = special.lpn(ni+1,cosdist[i,j])[0][ni+1]
    
    legpoly = legpoly + np.transpose(legpoly,(0,2,1))
    
    for i in range(leg_order):
        legpoly[i,:,:] = legpoly[i,:,:] + np.identity(numelectrodes)
    
    # compute G and H matrixes
    twoN1 = np.multiply(2, range(1, leg_order+1))+1
    gdenom = np.power(np.multiply(range(1, leg_order+1), range(2, leg_order+2)), m, dtype=float)
    hdenom = np.power(np.multiply(range(1, leg_order+1), range(2, leg_order+2)), m-1, dtype=float)
    
    G = np.zeros((numelectrodes, numelectrodes))
    H = np.zeros((numelectrodes, numelectrodes))
    
    for i in range(numelectrodes):
        for j in range(i, numelectrodes):
    
            g_val = 0 # Renamed to avoid conflict with G matrix
            h_val = 0 # Renamed to avoid conflict with H matrix
    
            for ni in range(leg_order):
                g_val = g_val + (twoN1[ni] * legpoly[ni,i,j]) / gdenom[ni]
                h_val = h_val - (twoN1[ni] * legpoly[ni,i,j]) / hdenom[ni]
    
            G[i,j] = g_val / (4*math.pi)
            H[i,j] = -h_val / (4*math.pi)
    
    G = G + G.T
    H = H + H.T
    
    G = G - np.identity(numelectrodes) * G[1,1] / 2
    H = H - np.identity(numelectrodes) * H[1,1] / 2
    
    if np.any(orig_data_size==1): # This condition might need review based on expected data shape
        data_proc = data[:] # Renamed to avoid conflict
    else:
        data_proc = np.reshape(data, (orig_data_size[0], np.prod(orig_data_size[1:])))
    
    # compute C matrix
    Gs = G + np.identity(numelectrodes) * smoothing
    GsinvS = np.sum(np.linalg.inv(Gs), 0) # np.linalg.inv might be slow for large matrices
    dataGs = np.dot(data_proc.T, np.linalg.inv(Gs))
    C = dataGs - np.dot(np.atleast_2d(np.sum(dataGs, 1)/np.sum(GsinvS)).T, np.atleast_2d(GsinvS))
    
    # apply transform
    surf_lap = np.reshape(np.transpose(np.dot(C,np.transpose(H))), orig_data_size)
    
    return H, surf_lap

def bfr_butter_filt(data, fs, order = 4, lfreq = 0.5, hfreq = 35, btype='pass', method='none'):
    """PSG data buffer extraction butterworth filter.
    This function takes a numpy array of the data & filters it based on given
    parameters. The function primarily wraps scipy butterworth filter construction
    and zero-phase filtfilt implementation. 
    Parameters
    ----------
    data : numpy array of [n_samples x n_chans]
        The data.
    fs : int/float
        Sampling rate. 
    order : int 
        Order of the filter
    lfreq : int
        High-pass filter frequency
    hfreq : int
        Low-pass filter frequency
    btype : str {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, optional
        The type of filter. Default is ‘bandpass’
    method : str {'none', 'mne'}
        If data will be later used with mne, please select 'mne'. 
    Returns
    -------
    data : numpy array of shape [n_samples x n_chans]
        The filtered data
    """
    ## Safety check - check is necessary only when later utilizing MNE, 
    #  which requires float64 type data 
    if method == 'mne':
        if data.dtype != np.float64:
            data = np.asarray(data, dtype=np.float64) # Corrected: assign back to data
    ## Construct butter filter (scipy)
    if btype == 'pass':
        filtparams = signal.butter(order, (lfreq, hfreq), btype=btype, fs = fs)
    elif btype=='lowpass':
        filtparams = signal.butter(order, (hfreq), btype=btype, fs = fs)
    elif btype=='highpass':
        filtparams = signal.butter(order, (lfreq), btype=btype, fs = fs)
    # run zero phase digitial filter with butterworth parameters, 
    # but first confirm order of dims
    dpnts, chans = data.shape
    if chans < dpnts: # if samples are rows and channels are columns
        data = signal.filtfilt(*filtparams, data, axis=0, padtype='odd')
    else: # if channels are rows and samples are columns
        data = signal.filtfilt(*filtparams, data, axis=-1, padtype='odd')
    return data

def re_reference(data, ch_names, trans_csd=None, sf=512, reference='common average'): # Added ch_names and trans_csd, sf
    """EEG data re_referencing function.
    This function takes a numpy array of the data & rereferences the data based on
    the new selected reference. 
    Parameters
    ----------
    data : numpy array of [n_samples x n_chans]
        The data. Include only EEG channels for this argument!
    reference : str {‘common average’, ‘mastoids’, 'csd'}
        The new reference; default is ‘common average’.
    ch_names: list
        List of channel names. Required for 'mastoids' and 'csd'.
    trans_csd: numpy array
        Precomputed CSD transform matrix. Required for 'csd'.
    sf: int
        Sampling frequency. Required for 'csd' if calling surface_laplacian_rt.
    Returns
    -------
    data : numpy array of shape [n_samples x n_chans]
        The re-referenced data
    """
    dpnts, nchan = data.shape # Assuming data is samples x channels
    if nchan > dpnts: # if channels x samples
        data = np.transpose(data)
        dpnts, nchan = data.shape

    if reference == 'common average':
        ref_data = data.mean(axis=1, keepdims=True)
        data -= ref_data
    elif reference == 'mastoids':
        if ch_names is None or not all(m_ch in ch_names for m_ch in ['M1', 'M2']):
            raise ValueError("Channel names including 'M1' and 'M2' must be provided for mastoid reference.")
        m1_idx = ch_names.index('M1')
        m2_idx = ch_names.index('M2')
        ref_data = data[:, [m1_idx, m2_idx]].mean(axis=1, keepdims=True)
        data -= ref_data
    elif reference == 'csd':
        if trans_csd is None:
            raise ValueError("trans_csd matrix must be provided for CSD reference.")
        # Assuming surface_laplacian_rt is defined elsewhere and accessible
        # And that it expects data as (channels, samples)
        data = surface_laplacian_rt(data=data.T, trans_csd=trans_csd).T 
    else:
        raise ValueError(f"Unknown reference type: {reference}. Choose from 'common average', 'mastoids', 'csd'.")
    
    return data

def surface_laplacian_rt(data, trans_csd):
    #epochs = inst._data # inst is not defined here, this was from the original context
    # Assuming data is (channels, samples) as per the re_reference function's call
    # and trans_csd is (n_channels_output, n_channels_input)
    # The original function had np.expand_dims(data.T, 0) which implies data was (samples, channels)
    # If data is (channels, samples) as suggested by the placeholder:
    # csd_data = np.dot(trans_csd, data)
    # If data is (samples, channels) as re_reference passes it (after data.T):
    # The original surface_laplacian_rt takes data as (n_channels, n_samples)
    # data.T in the original call means (n_samples, n_channels) -> .T -> (n_channels, n_samples)
    # So, if re_reference calls it with data.T, then 'data' arg here is (n_channels, n_samples)
    
    # The original function from csd.py:
    # epochs = np.expand_dims(data.T, 0) # data here is (n_chans, n_samples)
    # for epo in epochs: # This loop is over a single epoch if data is 2D
    #    csd_data = np.dot(trans_csd, epo)
    # This implies data should be (n_channels, n_samples)
    # and the loop is a bit redundant if it's not epoched data (3D)
    
    # Corrected based on original function's intent for 2D data (channels, samples)
    if data.ndim == 2:
        csd_data = np.dot(trans_csd, data)
    elif data.ndim == 3: # if it's already epoched (n_epochs, n_channels, n_samples)
        csd_data = np.zeros_like(data)
        for i, epo_data in enumerate(data): # iterate over epochs
             csd_data[i] = np.dot(trans_csd, epo_data)
    else:
        raise ValueError("Data must be 2D (channels, samples) or 3D (epochs, channels, samples)")
        
    return csd_data

def downsample_scaled(data, old_sf, new_sf, nint_method='resample_poly'):
    """Downsample function 
    The following function allows the user to downsample the data, so long 
    as the new sampling rate is a multiple of 100 or 128. The function can 
    now also downsample on non-integer scales with scipy fft resampling and
    polyphase resampling, although the latter is preferable with respect to
    computation time. 
    Parameters
    ----------
    data : np.array of shape [n_epochs, n_samples] or [n_samples]
          The epoched data or continuous data.
    old_sf : int
            Initial sampling rate.
    new_sf : int
            Requested sampling rate.
    nint_method : str 'none' (default), 'resample_fft', 'resample_poly'
                Non-integer resampling method from scipy.signal.
    Returns
    -------
    data: np.array
        Downsampled data.
    """
    if old_sf == new_sf:
        return data

    if old_sf < new_sf:
        raise ValueError("New sampling rate must be lower than old sampling rate.")

    decim = old_sf / new_sf
    
    # Preserve original shape type (epochs or continuous)
    original_ndim = data.ndim
    if original_ndim == 1: # Continuous data (n_samples,)
        data = data.reshape(1, -1) # Treat as a single epoch for processing
    
    # Assuming data is (n_epochs, n_samples)
    # If data is (n_samples, n_epochs) after potential transpose in sleep_funs...
    # This needs to be robust. Let's assume n_samples is the last dimension for mne/yasa consistency.
    
    # Check if downsampling factor is integer
    if decim.is_integer():
        data_downsampled = data[:, ::int(decim)]
        print(f'Downsampled data by an integer factor of {int(decim)}')
    else:
        if nint_method == 'resample_poly':
            # resample_poly works along axis -1 by default
            num = new_sf
            den = old_sf
            # Simplify num and den by dividing by their greatest common divisor
            common = math.gcd(num, den)
            num //= common
            den //= common
            data_downsampled = signal.resample_poly(data, num, den, axis=-1)
            print(f'Downsampled data by a non-integer factor of {decim} using resample_poly.')
        elif nint_method == 'resample_fft':
            new_n_samples = int(data.shape[-1] / decim)
            data_downsampled = signal.resample(data, new_n_samples, axis=-1)
            print(f'Downsampled data by a non-integer factor of {decim} using FFT resample.')
        else:
            raise ValueError('Invalid non-integer resampling method. Choose "resample_poly" or "resample_fft".')

    if original_ndim == 1:
        return data_downsampled.squeeze()
    return data_downsampled

def bandpower(epochs, fs, bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), 
                                 (8, 12, 'Alpha'),(12, 16, 'Sigma'), 
                                 (16, 30, 'Beta'), (49, 51, 'Line noise')], relative=True):
    """PSG absolute/relative power band feature extraction.
    This function takes a numpy array of the data & creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn. The function first computes the spectral density based on Welch's 
    method then by wrapping the bandpower_from_psd_ndarray from yasa, a package 
    written by Raphael Vallat, computes the absolute or relative power per frequency
    band of interest. 
    Parameters
    ----------
    epochs : numpy array --> [n_epochs, n_chans, n_samples] or [n_chans, n_samples]
        The epoched data.
    fs : int
        Sampling rate. 
    bands : Frequency bands of interest.
        Feed in bands of interest up to and limited by the Nyquist limit (half of the sampling rate), 
        beyond the default above. Line noise PSD is used for channel failure detection. 
    relative : bool 
        Default is 'True', which takes relative PSD values per epoch of time, otherwise
        select 'False' for absolute PSD values.
    see 'yasa.bandpower_from_psd_ndarray' for further documentation related to bandpower extraction.
    https://raphaelvallat.com/yasa/build/html/generated/yasa.bandpower_from_psd_ndarray.html
    Returns
    -------
    bp : numpy array of shape --> [n_epochs, n_bands, n_chans] or [n_bands, n_chans] 
        relative/absolute power of data.
    """
    # Define window length sufficiently long encompassing at least two 2 cycles of the lowest frequency of interest
    nperseg = (2 / bands[0][0]) * fs # Ensure bands[0][0] is not zero

    # Compute the modified periodogram (Welch)
    # Ensure epochs is 2D or 3D and psd_array_welch handles it correctly.
    # Welch in scipy.signal expects data along the last axis by default.
    # If epochs is (n_epochs, n_chans, n_samples), this should be fine.
    # If epochs is (n_chans, n_samples), also fine.
    freqs, psd = signal.welch(epochs, fs, nperseg=nperseg, average='median', axis=-1)
    
    # extract relative or absolute spectral density values for frequency bands of interest
    bp = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative)
    return bp
