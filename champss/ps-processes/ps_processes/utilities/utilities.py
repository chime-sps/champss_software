"""Utility functions for ps_processes."""

import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import chi2, kstwo


def rednoise_normalise(power_spectrum, b0=50, bmax=100000, get_medians=True):
    """
    Script to normalise power spectrum while removing rednoise. Based on presto's method
    of rednoise removal, in which a logarithmically increasing window is used at low
    frequencies to compute the local median. The median value to divide for each
    frequency bin is identified by a linear fit between adjacent local medians.

    Parameters
    =======
    power_spectrum: np.ndarray
        An ndarray of the power spectrum to remove rednoise

    b0: int
        The size of the first window to normalise the power spectrum. Default = 50

    bmax: int
        The maximum size of the largest window to normalise the power spectrum. Default = 1000

    get_medians: bool
        Whether to get the medians out of the rednoise normalisation process. Default = False

    Returns
    =======
    normalised_power_sepctrum: np.ndarray
        An ndarray of the normalised power spectrum with rednoise removed
    """
    ps_len = len(power_spectrum)
    scale = []
    for n in range(0, ps_len):
        # create the log range for normalisation
        new_window = np.exp(1 + n / 3) * b0 / np.exp(1)
        if new_window > bmax:
            pass
        else:
            window = int(new_window)
        scale.append(window)
        if np.sum(scale) > ps_len:
            scale[-1] = ps_len - np.sum(scale[:-1])
            break
    # check if sum of scale is equal to ps_len
    if np.sum(scale) < ps_len:
        scale[-1] += ps_len - np.sum(scale)
    start = 0
    old_mid_bin = 0
    old_median = 1
    normalised_power_spectrum = np.zeros(shape=np.shape(power_spectrum))
    if get_medians:
        medians = []
    for bins in scale:
        mid_bin = int(start + bins / 2)
        new_median = np.nanmedian(power_spectrum[start : start + bins])
        if get_medians:
            medians.append(new_median)
        i = 0
        while np.isnan(new_median):
            i += 1
            new_median = np.nanmedian(
                power_spectrum[start + (i * bins) : start + ((i + 1) * bins)]
            )
            if not np.isnan(new_median):
                if start == 0:
                    new_median = new_median * (2**i)
                else:
                    computed_median = new_median + (
                        (old_median - new_median) * (2**i / (2 ** (i + 1) - 1))
                    )
                    if computed_median - new_median < 0:
                        new_median = old_median
                    else:
                        new_median = computed_median
        if start == 0:
            normalised_power_spectrum[start : start + mid_bin] = power_spectrum[
                start : start + mid_bin
            ] / (new_median / np.log(2))
        elif start + bins >= ps_len:
            median_slope = np.linspace(old_median, new_median, num=ps_len - old_mid_bin)
            normalised_power_spectrum[old_mid_bin:] = power_spectrum[old_mid_bin:] / (
                median_slope / np.log(2)
            )
        else:
            # compute slope of the power spectra
            median_slope = np.linspace(
                old_median, new_median, num=mid_bin - old_mid_bin
            )
            normalised_power_spectrum[old_mid_bin:mid_bin] = power_spectrum[
                old_mid_bin:mid_bin
            ] / (median_slope / np.log(2))

        start += bins
        old_mid_bin = mid_bin
        old_median = new_median
    if get_medians:
        return normalised_power_spectrum, medians, scale
    return normalised_power_spectrum


def rednoise_normalise_runmed(power_spectrum, w0=10, wmax=1000, bmax=3000):
    """
    Script to normalise power spectrum while removing rednoise using a running median
    window at lower frequencies. The window increases in size from w0 to wmax
    logarithmically from bin 0 to bin bmax.

    Parameters
    =======
    power_spectrum: np.ndarray
        An ndarray of the power spectrum to remove rednoise

    w0: int
        The size of the first window to normalise the power spectrum

    wmax: int
        The maximum size of the largest window to normalise the power spectrum

    bmax: int
        The largest frequency bin where the running median is computed

    Returns
    =======
    normalised_power_sepctrum: np.ndarray
        An ndarray of the normalised power spectrum with rednoise removed
    """
    runmed = np.ones(bmax)
    exp_fac = bmax / np.log(wmax / w0)
    normalised_power_spectrum = np.zeros(shape=np.shape(power_spectrum))
    for n in range(0, bmax):
        # compute local median for each bin with a log-increasing window size
        window_size = int(np.exp(1 + n / exp_fac) * w0 / np.exp(1))
        if n - window_size / 2 < 0:
            runmed[n] = np.median(power_spectrum[0:window_size])
        else:
            runmed[n] = np.median(
                power_spectrum[n - int(window_size / 2) : n + int(window_size / 2)]
            )
        # normalise data with the running median up to bmax
        normalised_power_spectrum[n] = power_spectrum[n] / (runmed[n] / np.log(2))
    # normalise rest of the data with just a single window over wmax bins
    for n in np.arange(bmax, len(power_spectrum), wmax):
        if n + wmax > len(power_spectrum):
            normalised_power_spectrum[n:] = power_spectrum[n:] / (
                np.median(power_spectrum[n:]) / np.log(2)
            )
        else:
            normalised_power_spectrum[n : n + wmax] = power_spectrum[n : n + wmax] / (
                np.median(power_spectrum[n : n + wmax]) / np.log(2)
            )

    return normalised_power_spectrum


def analytical_chi2_pdf(x, k):
    """Compute the chi2 probability density function."""
    a = (x ** (k / 2 - 1)) * np.exp(-x / 2)
    b = gamma(k / 2) * (2 ** (k / 2))
    pdf_val = a / b
    pdf_val[x <= 0] = 0
    return pdf_val


def analytical_chi2_cdf(x, k, llim=0, ulim=np.inf):
    """
    Compute the chi2 cumulative distribution function. This implementation allows for a
    finite lower integral bound, which allows us to nominally compare a truncated
    distribution to our data. The default returns the standard value, where the lower
    integration bound is -inf.

    Integrating the chi2 PDF over the range a -> b yields:

        CDF(a, b; k) = (lgammainc(k/2, b/2) - lgammainc(k/2, a/2)) / gamma(k/2)

    where lgammainc is the lower incomplete gamma function, and gamma
    is the standard gamma function.

    To make our lives easier with implementation, let us define the function

        g(x, y) = lgammainc(x, y) / gamma(x),

    which is also known as the "regularised lower incomplete gamma function".
    Then, we may re-write the analytical CDF as

        CDF(a, b; k) = g(k/2, b/2) - g(k/2, a/2)

    where the functional form of g(x, y) is implemented as scipy.special.gammainc.
    This CDF is normalised to reach 1 in the integration limits.
    """

    # All cases may not be necessary because gammainc(k, 0) and
    # gammainc(k, np.inf) can be computed
    if llim != 0:
        cdf_val = gammainc(k / 2, x / 2) - gammainc(k / 2, llim / 2)

        if ulim != np.inf:
            cdf_val /= gammainc(k / 2, ulim / 2) - gammainc(k / 2, llim / 2)
        else:
            cdf_val /= 1 - gammainc(k / 2, llim / 2)

    else:
        cdf_val = chi2.cdf(x, k)
        if ulim != np.inf:
            cdf_val /= gammainc(k / 2, ulim / 2)

    return cdf_val


def get_ks_distance(data, llim=0, ulim=np.inf, dof=2, pval=0.05):
    """Compute the KS distance for each data value and report the corresponding KS
    distance for the provided p-value.
    """
    sort_idx = np.argsort(data)
    if llim != 0 or ulim != np.inf:
        cdf_vals = analytical_chi2_cdf(data[sort_idx], dof, llim, ulim)
    else:
        cdf_vals = chi2.cdf(data[sort_idx], dof)
    n = len(data)
    _d_plus_list = np.arange(1.0, n + 1) / n - cdf_vals
    _d_minus_list = cdf_vals - np.arange(0.0, n) / n
    _d = [max(dp, dm) for dp, dm in zip(_d_plus_list, _d_minus_list)]
    d = np.zeros_like(data)
    for stat, si in zip(_d, sort_idx):
        d[si] = stat
    thresh_stat_val = kstwo.isf(pval, n)

    return d, thresh_stat_val


def check_in_range(test_values, llim, ulim):
    """
    Check if the test values are in a given range.

    Parameters
    ==========
    test_values: list(float) or float
        Values that are tested whether they are fully in the given range

    llim: float
        Lower limit of the tested range

    ulim: float
        Upper limit of the tested range

    Returns
    =======
    result: bool
        Whether test_values are fully contained in the given range
    """

    test_values = np.asarray(test_values)
    result = ((llim < test_values).all()) & ((test_values < ulim).all())

    return result


def grab_metric_history(obs_list, test, metric):
    """
    Grab the history for a quality metric from a list obsevrations.

    Parameters
    ==========
    obs_list: list[sps_database.models.Observation]
        List of observations from which the history is derived

    test: str
        Name of the quality test from which the metric is derived

    metric: str
        Name of the quality metric in the given quality test

    Returns
    =======
    all_metric: list[float]
        All values for the given metric in the observations
    """
    all_tests = []
    for obs in obs_list:
        # For obs properties these are also grabbed even if they are not saved in
        # the qc_test
        if test == "obs_properties":
            current_metric = getattr(obs, metric, None)
            if current_metric is not None:
                all_tests.append(current_metric)
        else:
            current_qc = getattr(obs, "qc_test", None)
            if isinstance(current_qc, dict):
                current_test = current_qc.get(test, None)
                if isinstance(current_test, dict):
                    current_metric = current_test.get(metric, None)
                    if current_metric is not None:
                        all_tests.append(current_metric)
    return all_tests
