#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import median_filter, generic_filter


def baseline_estimation_medfilt(y, k=3):
    """
    Use the generic filter function to implement a median filter that handles masked
    data. Data at the edges are reflected so that we do not encounter dropouts or other
    edge effects.

    Parameters
    ----------
    y: np.ndarray
        The data to be filtered
    k: int
        The window size over which to compute the moving median value. Must be odd.

    Returns
    -------
    z: np.ndarray
        The median filtered result, effectively representing the baseline.

    """
    z = generic_filter(y, function=np.ma.median, size=k, mode="reflect")
    return z


def baseline_estimation_alss(y, lam=1000, p=0.5, niter=10):
    """
    Using the Asymmetric Least-Squares Smoothing technique (Eilers & Boelens, 2005) to
    robustly estimate a smoothed baseline from noisy data with arbitrary shape. Can be
    thrown off by large gaps in data.

    Parameters
    ----------
    y: np.ndarray
        The potentially noisy data from which to estimate a smooth baseline.
    lam: float
        Smoothing parameter, nominally between 100 and 1e9 are sensible depending on
        the data.
    p: float
        The asymmetry parameter defines how to weight the least-squares analysis. A
        value <0.5 downweights positive peaks (i.e. total baseline level is shifted
        lower), and a value >0.5 downweights dropouts (i.e. total baseline level is
        shifted higher). Setting p=0.5 equally weights both. Must be between in the
        range (0, 1).
    niter: int
        Maximum number of iterations to run before halting.

    Returns
    -------
    z: np.ndarray
        The smoothed baseline (same shape as y).
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    Dlam = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + Dlam
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


def powerlaw_gaussian_noise(
    exponent=2.0, size=1024, sample_rate=1.0, fmin=0.0, seed=1234
):
    """
    Gaussian (1/f)**beta noise, normalised to unit variance.

    Pulled directly from
    https://github.com/felixpatzelt/colorednoise/blob/master/colorednoise.py
    (with only minor style tweaks)

    Based on the algorithm in:
    Timmer, J. and Koenig, M., "On generating power law noise",
    Astron. Astrophys. 300, 707-710 (1995)

    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    size : int
        The output has the given size
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.

    Returns
    -------
    out : array
        The power-law distributed samples.
    """

    np.random.seed(seed)
    # Make sure size is a list so we can iterate it and assign to it.
    if not isinstance(size, list):
        if isinstance(size, int):
            size = [size]
        else:
            size = list(size)

    # The number of samples in each time series
    nsamples = size[-1]

    # Calculate Frequencies
    # Use fft functions for real output (-> Hermitian spectrum)
    f = np.fft.rfftfreq(nsamples, d=sample_rate)

    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1.0 / nsamples)  # Low frequency cutoff
    ix = np.sum(s_scale < fmin)  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (nsamples % 2)) / 2.0  # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w**2)) / nsamples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(np.newaxis,) * dims_to_add + (...,)]

    # Generate scaled random power + phase
    sr = np.random.normal(scale=s_scale, size=size)
    si = np.random.normal(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (nsamples % 2):
        si[..., -1] = 0

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0

    # Combine power + corrected phase to Fourier components
    s = sr + 1.0j * si

    # Transform to real time series & scale to unit variance
    y = np.fft.irfft(s, n=nsamples, axis=-1) / sigma

    return y
