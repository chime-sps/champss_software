#!/usr/bin/env python3

import logging
import warnings
from itertools import count, groupby
from typing import Tuple, Union

import numpy as np
from scipy.optimize import newton
from scipy.special import gamma, gammainc
from sps_common.constants import L1_NCHAN

log = logging.getLogger(__name__)


def known_bad_channels(nchan: int = 16384) -> list:
    """
    Given a nominal set of bad channels evaluated at the native L1 frequency.

    resolution, this function will downsample the bad frequency mask and
    return a list of channel indices in the range (0, nchan - 1) inclusive
    to mask.

    Parameters
    ----------
    nchan: int
        Number of frequency channels of the desired channel mask

    Returns
    -------
    downsampled_badchans: list
        A list of unique bad channel indices at the given frequency resolution
    """
    # TODO: This bad channel mask needs to be re-evaluated,
    #  and the ordering needs to be checked
    badchans_16k = np.concatenate(
        (
            range(0, 64),
            range(1804, 2224),
            range(2254, 2865),
            range(4331, 4353),
            [4429],
            range(8843, 9600),
            range(10012, 10576),
            range(10815, 11073),
            range(11325, 11556),
            range(12042, 12288),
            range(12559, 12785),
            range(12924, 12980),
            range(13534, 13554),
            range(16319, 16383),
            range(1728, 1792),
            range(2896, 2960),
            range(7504, 7728)
        )
    )
    badchans_16k = sorted(badchans_16k)

    if nchan == L1_NCHAN:
        return badchans_16k
    else:
        log.info("Bad channel mask will be downsampled to match data shape")
        ffact = L1_NCHAN // nchan
        log.info(f"  downsampling factor = {ffact}")

        split_badchans = [
            np.array(list(g))
            for k, g in groupby(badchans_16k, key=lambda i, j=count(): i - next(j))
        ]
        downsampled_badchans = []

        for sublist in split_badchans:
            # be conservative, take the floor/ceiling of the downsampled values
            submask_min = np.floor(np.min(sublist / ffact)).astype(int)
            submask_max = np.ceil(np.max(sublist / ffact)).astype(int)

            # include the submask maximum in the mask range, but ensure it is
            # always < nchan
            # NOTE: since range(a, b) returns an iterator with items
            # [a, a + 1, ..., b - 1] if b > a, or an empty list if a == b, we
            # need to increase the upper bound by 1 so that the final submask
            # includes at least the floor we calculated above and, if
            # applicable, also the ceiling calculated above. This is true
            # (even if a == b) EXCEPT when b == nchan, in which case the +1
            # operation isn't needed.
            if submask_max < nchan:
                submask_max += 1

            # the submask minimum should always be >= 0
            if submask_min < 0:
                submask_min = 0

            downsampled_badchans = np.concatenate(
                (downsampled_badchans, range(submask_min, submask_max))
            )

        return list(set(downsampled_badchans.astype(int)))


def median_absolute_deviation(x: np.array, k: float = 1.4826):
    """
    Calculate the median and median absolute deviation (MAD) in an attempt to robustly
    estimate the standard deviation of the data.

    NOTE: should use scipy's median_abs_deviation (v1.5+) as it more correctly deals
    with the scale factor k
    """
    med = np.ma.median(x)
    med_abs_dev = np.ma.median(np.abs(x - med))

    return med, k * med_abs_dev


def combine_cleaner_masks(masks: np.ndarray) -> Tuple[np.ndarray, float]:
    """Combine any number of boolean masks into a final, signal binary mask."""
    reduced_mask = np.logical_or.reduce(masks, axis=0)
    reduced_mask_frac = reduced_mask.sum() / reduced_mask.size
    return reduced_mask, reduced_mask_frac


def generalised_spectral_kurtosis(
    spec: np.ndarray, m: int, delta: Union[float, np.ndarray] = 1.0
) -> np.ndarray:
    """
    Compute the Generalised Spectral Kurtosis estimator base on (Nita, Gary & Hellbourg
    2017, eq.

    5), where the delta = d * N parameter encodes the true shape of the data
    distribution
    """
    # assuming that the spectrum is of shape (nchan, ntime) then perform a summation
    # over the power measurements and their square over the M time samples
    s1_sq = np.nansum(spec, axis=1) ** 2
    s2 = np.nansum(spec**2, axis=1)
    warnings.filterwarnings("ignore")
    gsk = ((m * delta + 1) / float(m - 1)) * (m * (s2 / s1_sq) - 1)

    return gsk


def rescale_gsk(
    gsk: np.ndarray, m: int, new_delta: Union[float, np.ndarray] = 1.0
) -> np.ndarray:
    """
    Rather than totally recomputing the GSK estimator, we can instead rescale the old
    values, since we can solve for x in the following equation:

        x * (m + 1) / (m - 1) * (m * (s2 / s1**2) - 1) =
                                        (m * d + 1)/(m - 1) * (m * (s2 / s1**2) - 1)

    where d is the required delta scaling factor. In this case, we find that

    x = (d * m + 1) / (m + 1)    assuming m + 1 != 0 and m != 1

    This should notionally be computationally more efficient.
    """
    rescaling_factor = (new_delta * m + 1) / (m + 1)
    return gsk * rescaling_factor


def gsk_upper_root_from_moments(x: float, mom2: float, mom3: float, p: float) -> float:
    """
    Compute the upper root of equation 20 in https://arxiv.org/pdf/1005.4371.pdf
    """
    beta = 4 * (mom2**3) / (mom3**2)
    argument = (-(mom3 - 2 * mom2**2) / mom3 + x) / (mom3 / 2.0 / mom2)
    root = abs(1 - gammainc(beta, argument) - p)

    return root


def gsk_lower_root_from_moments(x: float, mom2: float, mom3: float, p: float) -> float:
    """
    Compute the lower root of equation 20 in https://arxiv.org/pdf/1005.4371.pdf
    """
    beta = 4 * (mom2**3) / (mom3**2)
    argument = (-(mom3 - 2 * mom2**2) / mom3 + x) / (mom3 / 2.0 / mom2)
    root = abs(gammainc(beta, argument) - p)

    return root


def gsk_moments(m: int, n: float, d: float) -> Tuple[float, float]:
    """
    Based on the input SK parameter, compute the second and third central moments. By
    definition, the first central moment is 1.

    :param m: Number of off-board accumulations before S1 and S2 are computed
    :type m: int
    :param n: Number of on-board accumulations before S1 and S2 are computed.
    :type n: float
    :param d: The assumed shape parameter of the Gamma probability distribution
        describing the input data.
    :type d: float
    :return: The second and third central moments
    :rtype: Tuple[float, float]
    """
    # compute often-used quantities up-front
    dn = d * n
    mdn = m * dn

    # compute the second and third central moments
    m2_1 = 2 * (m**2) * dn * (1 + dn)
    m2_2 = (-1 + m) * (6 + 5 * m * dn + (m**2) * (dn**2))
    m2 = m2_1 / m2_2

    m3_1 = 8 * (m**3) * dn * (1 + dn) * (-2 + dn * (-5 + m * (4 + dn)))
    m3_2 = ((-1 + m) ** 2) * (2 + mdn) * (3 + mdn) * (4 + mdn) * (5 + mdn)
    m3 = m3_1 / m3_2

    return m2, m3


def transform_using_type6(
    m2: float, beta1: float, alpha1: float, ret_alpha_beta: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, float, float]]:
    """
    Compute the transformation corrections needed when data is in the Pearson Type VI
    error distribution regime.

    :param m2: The SK second central moment
    :type m2: float
    :param beta1: First pre-computed error distribution evaluation parameter, the ratio
        of third and second central moments (see get_pdf_transforms)
    :type beta1: float
    :param alpha1: Second pre-computed error distribution evaluation parameter (see
        get_pdf_transforms)
    :type alpha1: float
    :param ret_alpha_beta: Whether to return intermediate quantities (only useful if
        debugging), defaults to False
    :type ret_alpha_beta: bool, optional
    :return: The transformation parameters (scale and location), and optionally
        intermediate values
    :rtype: Union[Tuple[float, float], Tuple[float, float, float, float]]
    """
    # compute the transformation variable required for a Pearson Type VI distribution
    h = 4 + np.sqrt(beta1 * (4 + 1 / m2) + 16)
    alpha = 8 * m2 / alpha1 - 1
    alpha = (1 / alpha1) * alpha + 1
    alpha = h * alpha + 4
    alpha = m2 * alpha + 1
    alpha = (1 / alpha1) * alpha - 1
    beta = 3 + 2 * h / beta1

    scl = 1
    loc = 1 - alpha / (beta - 1)
    if ret_alpha_beta:
        return scl, loc, alpha, beta
    else:
        return scl, loc


def get_pdf_transforms(
    m: int, n: float, d: float, ret_ab=False
) -> Union[Tuple[float, float], Tuple[float, float, float, float]]:
    """
    Based on the input SK paramteres, determine which kind of Error distribution applies
    to our data and reeturn the correct transformation parameters.

    :param m: Number of off-board accumulations before S1 and S2 are computed
    :type m: int
    :param n: Number of on-board accumulations before S1 and S2 are computed.
    :type n: float
    :param d: The assumed shape parameter of the Gamma probability distribution
        describing the input data.
    :type d: float
    :param ret_ab: Whether to return intermediate quantities (only useful if debugging),
        defaults to False
    :type ret_ab: bool, optional
    :return: The transformation parameters (scale and location), and optionally
        intermediate values
    :rtype: Union[Tuple[float, float], Tuple[float, float, float, float]]
    """
    # compute often-used quantities up-front
    dn = d * n
    mdn = m * dn

    # compute the second and third central moments, and their ratio
    m2, m3 = gsk_moments(m, n, d)
    alpha1 = m3 / m2

    # quantities used to decide which Error Type distribution is appropriate
    beta1_n = 8 * (mdn + 2) * (mdn + 3) * (mdn * (dn + 4) - 5 * dn - 2) ** 2
    beta1_d = (m - 1) * (mdn + 4) ** 2 * (mdn + 5) ** 2 * dn * (dn + 1)
    beta1 = beta1_n / beta1_d

    beta2_1 = 3 * (mdn + 2) * (mdn + 3)
    beta2_2 = (m - 1) * (mdn + 4) * (mdn + 5) * (mdn + 6) * (mdn + 7)
    beta2_3 = beta2_1 / beta2_2
    beta2_4 = (
        mdn**3 * (dn + 1)
        + mdn**2 * (3 * dn**2 + 68 * dn + 125)
        - mdn * (93 * dn**2 + 245 * dn + 32)
        + 12 * (7 * dn**2 + 4 * dn + 2)
    )
    beta2_5 = 1 / (dn * (dn + 1))
    beta2 = beta2_3 * beta2_4 * beta2_5

    kappa_1 = beta1 * (beta2 + 3) ** 2
    kappa_2 = 4 * (4 * beta2 - 3 * beta1) * (2 * beta2 - 3 * beta1 - 6)
    kappa = kappa_1 / kappa_2

    # for SPS purposes, we will generally be in the Type VI regime,
    # since we are combining a minimum of 96 amplitudes together per
    # time sample
    if kappa <= 0:
        log.debug("error dist. type ~ I")
        raise NotImplementedError(
            "SPS data should never be in this regime (Pearson Type I)"
        )
    elif 0 < kappa < 1:
        log.debug("error dist. type ~ IV")
        raise NotImplementedError(
            "SPS data should never be in this regime (Pearson Type IV)"
        )
    elif np.isfinite(kappa) and kappa > 1:
        log.debug("error dist. type ~ VI")
        if ret_ab:
            return transform_using_type6(m2, beta1, alpha1, ret_alpha_beta=True)
        else:
            return transform_using_type6(m2, beta1, alpha1)
    else:
        if kappa == 1:
            raise NotImplementedError(
                "GSK cannot realistically be in this regime (Pearson Type V)"
            )

        if not np.isfinite(kappa):
            raise NotImplementedError(
                "GSK cannot realistically be in this regime (Pearson Type III)"
            )


def get_gaussian_gsk_bounds(
    m: int, n: float, d: float, eta: float = 3
) -> Tuple[float, float]:
    """
    Based on the input SK paramteres, determine the theoretical Gaussian RFI threshold
    values.

    :param m: Number of off-board accumulations before S1 and S2 are computed
    :type m: int
    :param n: Number of on-board accumulations before S1 and S2 are computed.
    :type n: float
    :param d: The assumed shape parameter of the Gamma probability distribution
        describing the input data.
    :type d: float
    :param eta: Equivalent Gaussian sigma cut-off threshold, defaults to 3
    :type eta: float, optional
    :return: The lower and upper SK RFI thresholds.
    :rtype: Tuple[float, float]
    """
    m2, _ = gsk_moments(m, n, d)
    scl, loc = get_pdf_transforms(m, n, d)
    # We can approximate an eta-sigma Gaussian limit
    approx_lower = (1 - loc - eta * np.sqrt(m2)) / scl
    approx_upper = (1 - loc + eta * np.sqrt(m2)) / scl
    true_lower = scl * approx_lower + loc
    true_upper = scl * approx_upper + loc
    return true_lower, true_upper


def theoretical_generalised_sk_bounds(
    m: int,
    n: int = 1,
    d: Union[float, np.ndarray] = 1.0,
    pfa: float = 0.0013499,
    maxiters=10000,
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    This function is derived from that presented in Appendix A of
    https://arxiv.org/pdf/1702.05391.pdf

    Equation numbers refer to https://arxiv.org/pdf/1005.4371.pdf

    Parameters
    ----------
    m: int
        Number of off-board accumulations before S1 and S2 are computed
    n: int
        Number of on-board accumulations before S1 and S2 are computed.
        Default is 1.
    d: float
        The shape parameter of the Gamma probability distribution
        corresponding to the PSD estimates. Default is 1 (PSD obtained by
        FFT means)
    pfa: float
        False alarm probability level. Default is 0.0013499, corresponding
        to a 3-sigma normal PDF false alarm rate.
    maxiters: int
        Maximum number of iterations allowed for newton numerical solver

    Returns
    -------
    (lower, upper): (float, float)
        Normalised upper and lower GSK thresholds
    """

    m2, m3 = gsk_moments(m, n, d)

    # Compute the thresholds according to equation 20, solving numerically
    if isinstance(d, float):
        # log.debug(f"dn={dn} mdn={mdn} m2={m2} m3={m3}")
        try:
            lower_thresh = newton(
                gsk_lower_root_from_moments,
                1.0,
                args=(m2, m3, pfa),
                maxiter=maxiters,
                rtol=1.0e-8,
            )
            upper_thresh = newton(
                gsk_upper_root_from_moments,
                1.0,
                args=(m2, m3, pfa),
                maxiter=maxiters,
                rtol=1.0e-8,
            )
        except RuntimeError as e:
            log.warning("Failed to converge when calculating RFI bounds")
            log.debug(e)
            lower_thresh = 0.99
            upper_thresh = 1.01
            log.debug(f"setting rfi bounds to ({lower_thresh}, {upper_thresh})")

    elif isinstance(d, np.ndarray):
        log.debug("d is an array, computing bounds for each element")
        try:
            lower_thresh = np.array(
                [
                    newton(
                        gsk_lower_root_from_moments,
                        1.0,
                        args=(i_m2, i_m3, pfa),
                        maxiter=maxiters,
                        rtol=1.0e-8,
                    )
                    for (i_m2, i_m3) in zip(m2, m3)
                ]
            )
            upper_thresh = np.array(
                [
                    newton(
                        gsk_upper_root_from_moments,
                        1.0,
                        args=(i_m2, i_m3, pfa),
                        maxiter=maxiters,
                        rtol=1.0e-8,
                    )
                    for (i_m2, i_m3) in zip(m2, m3)
                ]
            )
        except RuntimeError as e:
            log.warning("Failed to converge when calculating RFI bounds")
            log.debug(e)
            lower_thresh = 0.99
            upper_thresh = 1.01
            log.debug(f"setting rfi bounds to ({lower_thresh}, {upper_thresh})")

    else:
        lower_thresh = 0.99
        upper_thresh = 1.01
        log.debug(f"setting rfi bounds to ({lower_thresh}, {upper_thresh})")

    return lower_thresh, upper_thresh


def generalised_sk_shape_from_data(
    spectrum: np.ndarray, chan_start: int, chan_end: int
) -> float:
    """
    Following the method laid out in
    https://ieeexplore.ieee.org/abstract/document/7833535
    whereby the underlying Gamma distribution that actually describes real data is used
    to inform the GSK calculation (by providing a more realistic estimate of the shape)
    and to provide realistic upper and lower thresholds for RFI excision.

    Parameters
    ----------
    spectrum: array-like, 2D
        A dynamic spectrum (2D, nchan-by-ntime) from which to approximate the
        underlying Gamma distribution parameters.
    chan_start: int
        Starting channel index representing a nominally "clean" part of the spectrum
    chan_end: int
        Final channel index representing a nominally "clean" part of the spectrum

    Returns
    -------
    delta: float
        The more realistic shape parameters
    """
    # get valid data range by looking at centre 80% of samples (mitigates the
    # occasional RFI signal in the nominal "good" range)
    p_u = np.percentile(spectrum[chan_start:chan_end, :], 90)
    p_l = np.percentile(spectrum[chan_start:chan_end, :], 10)
    spec_view = spectrum[chan_start:chan_end, :]
    good_range_mask = np.where((spec_view <= p_u) & (spec_view >= p_l))
    clean_variance = np.nanvar(spec_view[good_range_mask])
    clean_mean = np.nanmean(spec_view[good_range_mask])

    real_location_param = clean_variance / clean_mean
    real_shape_param = clean_mean**2 / clean_variance
    log.debug(f"Gamma location paramter, lambda = {real_location_param}")
    log.debug(f"Gamma shape paramter, d = {real_shape_param}")

    return real_shape_param


# define some plotting functions used for debugging/diagnostic purposes
def plot_diagnostic_baseline(flagged_spec_sk, baseline, scale, idx):
    """Plot the GSK spectrum with basic flagging applied, and the corresponding median
    baseline.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=plt.figaspect(0.3))
    plt.plot(flagged_spec_sk, label="masked raw SK")
    plt.plot(baseline, label="median filtered baseline")
    plt.xlabel("Channel number")
    plt.ylabel("SK value")
    plt.legend()
    plt.title(f"M={scale}, delta=1")
    plt.savefig(f"chunk_sk_baseline_{idx}.png")
    plt.close()


def plot_diagnostic_delta(delta, idx):
    """Plot the frequency-dependent data distribution shape spectrum."""
    import matplotlib.pyplot as plt

    plt.plot(delta)
    plt.xlabel("Channel number")
    plt.ylabel(r"$\delta$ (data distribution shape)")
    plt.title(f"Frequency-dependent shape parameter")
    plt.savefig(f"chunk_sk_delta_{idx}.png")
    plt.close()


def plot_diagnostic_masked_sk(spec_sk, mask, llim, ulim, scale, mean_delta, idx):
    """Plot the "final" masked GSK spectrum, effectively a before/after snapshot."""
    import matplotlib.pyplot as plt

    masked = np.ma.masked_array(spec_sk, mask=mask)
    plt.plot(spec_sk, alpha=0.4, label="corrected SK")
    plt.plot(masked, label="masked SK")
    plt.fill_between(
        np.arange(len(spec_sk)),
        llim,
        ulim,
        alpha=0.4,
        color="C0",
        label="RFI bounds",
    )
    plt.axhline(1, color="k", ls=":")
    if isinstance(llim, float) and isinstance(ulim, float):
        ylims = (0.8 * llim, 2 * ulim)
    elif len(llim) > 1 and len(ulim) > 1:
        ylims = (0.8 * llim.mean(), 2 * ulim.mean())
    else:
        ylims = (0, 2)
    plt.ylim(*ylims)
    plt.xlabel("Channel number")
    plt.ylabel("SK value")
    plt.legend()
    plt.title(f"M={scale}, av. delta={mean_delta:g}")
    plt.savefig(f"chunk_sk_corr_{idx}.png")
    plt.close()


def plot_diagnostic_masked_spectrum(spec, mask, idx):
    """Plot the masked 2D spectrum for the given chunk of data."""
    import matplotlib.pyplot as plt

    full_mask = np.logical_or(spec.mask, mask[:, np.newaxis])
    masked_spec = np.ma.masked_array(spec.data, mask=full_mask)
    plt.imshow(masked_spec, origin="lower", aspect="auto", interpolation="none")
    plt.xlabel("Time sample")
    plt.ylabel("Channel number")
    plt.title(f"Data chunk {idx}")
    plt.savefig(f"chunk_waterfall_corr_{idx}.png")
    plt.clf()

    plt.plot(masked_spec.data.mean(1), color="k", alpha=0.3)
    plt.plot(masked_spec.mean(1), color="C0", alpha=0.5)
    plt.ylim(0, masked_spec.mean(1).max())
    plt.savefig(f"chunk_spec_corr_{idx}.png")
    plt.close()


def plot_diagnostic_power_spectrum(freqs, pspec, chan, thresh=20, nchan=16384):
    """Plot the FFT power spectrum for a given channel's data."""
    import matplotlib.pyplot as plt

    plt.plot(freqs, pspec)
    plt.axhline(thresh, color="k", ls=":")
    plt.xlabel("FFT freq. (Hz)")
    plt.ylabel("Norm. power")
    plt.title(f"Channel {chan}/{nchan} power spectrum")
    plt.savefig(f"power_spec_chan{chan}.png")
    plt.close()
