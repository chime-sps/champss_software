"""Utility functions for ps_processes."""

from functools import lru_cache

import bottleneck as bn
import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import chi2, kstwo


LOG2 = np.log(2)


def _median_of(values, k_lo, k_hi):
    """Median from the order statistics k_lo and k_hi of values, which is modified."""
    if k_lo == k_hi:
        values.partition(k_lo)
        return values[k_lo]
    values.partition([k_lo, k_hi])
    return np.mean(values[k_lo : k_hi + 1])


def fast_nanmedian(x, sample_size=4096, min_size=8192):
    """
    Compute the median of a 1D array while ignoring nans.

    Gives the same result as np.nanmedian, but avoids partitioning the full array for
    large inputs. The median is bracketed using a strided sample of the data, so that
    only the values within the bracket need to be partitioned. If the median falls
    outside the bracket the full array is partitioned instead.

    Parameters
    =======
    x: np.ndarray
        1D array of which the median is computed. Is not modified.

    sample_size: int
        Approximate size of the sample used to bracket the median. Default = 4096

    min_size: int
        Number of non-nan values below which the full array is partitioned.
        Default = 8192

    Returns
    =======
    median: np.floating
        The median of x, nan if x only contains nans
    """
    nan_mask = np.isnan(x)
    n_nan = np.count_nonzero(nan_mask)
    n = x.size - n_nan
    if n == 0:
        return x.dtype.type(np.nan)
    k_lo, k_hi = (n - 1) // 2, n // 2
    if n < min_size:
        return _median_of(x[~nan_mask] if n_nan else x.copy(), k_lo, k_hi)
    sample = x[:: max(1, x.size // sample_size)].copy()
    m = sample.size - np.count_nonzero(np.isnan(sample))
    # bracket the median by +-4 sigma of its rank in the sample
    half_width = int(2 * np.sqrt(m)) + 1
    j_lo, j_hi = max(0, m // 2 - half_width), min(m - 1, m // 2 + half_width)
    # nans are sorted to the end
    sample.partition([j_lo, j_hi])
    lower, upper = sample[j_lo], sample[j_hi]
    below = x < lower
    n_below = np.count_nonzero(below)
    inside = (x <= upper) ^ below
    if n_below <= k_lo and n_below + np.count_nonzero(inside) > k_hi:
        return _median_of(x[inside], k_lo - n_below, k_hi - n_below)
    return _median_of(x[~nan_mask], k_lo, k_hi)


@lru_cache(maxsize=16)
def rednoise_window_sizes(ps_len, b0=50, bmax=100000):
    """
    Compute the window sizes used in rednoise_normalise.

    The windows only depend on the length of the power spectrum and the window
    configuration, so they are cached. In the pipeline they are identical for
    every DM trial searched by a worker process.

    Parameters
    =======
    ps_len: int
        Length of the power spectrum

    b0: int
        The size of the first window. Default = 50

    bmax: int
        The maximum size of the largest window. Default = 100000

    Returns
    =======
    scale: tuple
        The window sizes, which sum up to ps_len
    """
    scale = []
    total = 0
    shortcut = True
    for n in range(0, ps_len):
        # create the log range for normalisation
        new_window = np.exp(1 + n / 3) * b0 / np.exp(1)
        if new_window <= bmax:
            window = int(new_window)
        elif shortcut and window > 0:
            # All remaining windows have the same size, followed by the remainder
            n_full = (ps_len - total) // window
            if n_full + 1 <= ps_len - n:
                scale.extend([window] * n_full)
                total += n_full * window
                scale.append(ps_len - total)
                return tuple(scale)
            shortcut = False
        scale.append(window)
        total += window
        if total > ps_len:
            scale[-1] = ps_len - (total - window)
            total = ps_len
            break
    # check if sum of scale is equal to ps_len
    if total < ps_len:
        scale[-1] += ps_len - total
    return tuple(scale)


# Equally sized windows below this size have their medians computed together
MAX_BATCH_WINDOW = 8192
# Interpolate the medians in a single pass when there are at least this many segments
MIN_VECTORISED_SEGMENTS = 256


@lru_cache(maxsize=16)
def _rednoise_layout(ps_len, b0, bmax):
    """
    Precompute the window and interpolation layout used in rednoise_normalise.

    Returns
    =======
    layout: dict
        scale: tuple of the window sizes
        starts: list of the first bin of each window
        mids: list of the middle bin of each window
        runs: list of (first window, number of windows, window size) for runs of
            equally sized windows whose medians are computed together
        single: list of the windows whose medians are computed one at a time
        seg_lo: first bin of each segment. Segment k is the linear
            interpolation between the medians of windows k and k + 1
        num: number of bins in each segment
        Only with at least MIN_VECTORISED_SEGMENTS segments:
        seg_id: segment of each bin from mids[0] onwards
        offset: position of each bin within its segment
        seg_end: index of the last bin of each segment with more than one bin
    """
    scale = rednoise_window_sizes(ps_len, b0, bmax)
    n_win = len(scale)
    starts = [0]
    for bins in scale[:-1]:
        starts.append(starts[-1] + bins)
    mids = [int(start + bins / 2) for start, bins in zip(starts, scale)]

    runs = []
    single = []
    k = 0
    while k < n_win:
        k_end = k + 1
        while k_end < n_win and scale[k_end] == scale[k]:
            k_end += 1
        if k_end - k > 1 and scale[k] < MAX_BATCH_WINDOW:
            runs.append((k, k_end - k, scale[k]))
        else:
            single.extend(range(k, k_end))
        k = k_end

    layout = dict(scale=scale, starts=starts, mids=mids, runs=runs, single=single)
    if n_win > 1:
        # Segment k - 1 covers the bins from the middle of window k - 1 to the
        # middle of window k, or to the end for windows reaching the end.
        # A trailing window of size 0 makes the last two segments overlap,
        # the later one overwriting the earlier one.
        lo = np.array(mids[:-1])
        hi = np.array(
            [
                ps_len if start + bins >= ps_len else mid
                for start, bins, mid in zip(starts[1:], scale[1:], mids[1:])
            ]
        )
        num = hi - lo
        for arr in (lo, num):
            arr.setflags(write=False)
        layout.update(seg_lo=lo, num=num)
    if n_win - 1 >= MIN_VECTORISED_SEGMENTS and scale[-1] > 0:
        seg_id = np.repeat(np.arange(n_win - 1, dtype=np.int32), num)
        offset = np.arange(ps_len - lo[0], dtype=np.float64) - np.repeat(
            lo - lo[0], num
        )
        seg_end = (np.cumsum(num) - 1)[num > 1]
        for arr in (seg_id, offset, seg_end):
            arr.setflags(write=False)
        layout.update(seg_id=seg_id, offset=offset, seg_end=seg_end)
    return layout


def _interpolate_medians(medians, num):
    """
    Compute start, stop, delta and step of the linear interpolation between
    adjacent medians.

    Replicates the arithmetic of np.linspace for every segment, so that the
    interpolated medians are bit-identical to calling np.linspace per segment.
    np.linspace converts start and stop to float64 before computing delta and step.
    """
    medians = np.array(medians, dtype=np.float64)
    seg_start = medians[:-1]
    seg_stop = medians[1:]
    delta = seg_stop - seg_start
    with np.errstate(divide="ignore", invalid="ignore"):
        step = delta / (num - 1)
    # np.linspace multiplies with delta for num = 1
    step[num == 1] = 0
    return seg_start, seg_stop, delta, step


def rednoise_normalise(
    power_spectrum,
    b0=50,
    bmax=100000,
    get_medians=True,
    ignore_zeros=False,
    out=None,
):
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
        The maximum size of the largest window to normalise the power spectrum. Default = 100000

    get_medians: bool
        Whether to get the medians out of the rednoise normalisation process. Default = True

    ignore_zeros: bool
        Whether to ignore bins with a value of 0 when computing the medians.
        Default = False

    out: np.ndarray or None
        Array of the same length as power_spectrum to write the result to. May be
        power_spectrum itself to normalise in place, which avoids allocating a new
        array for every DM trial. If None a new float64 array is created. Default = None

    Returns
    =======
    normalised_power_sepctrum: np.ndarray
        An ndarray of the normalised power spectrum with rednoise removed
    """
    ps_len = len(power_spectrum)
    layout = _rednoise_layout(ps_len, b0, bmax)
    scale = layout["scale"]
    starts = layout["starts"]
    if out is None:
        out = np.zeros(ps_len)
    elif len(out) != ps_len:
        raise ValueError(
            f"out has length {len(out)}, but power spectrum has length {ps_len}"
        )

    def window_median(window):
        if ignore_zeros:
            window = window[window != 0]
        return fast_nanmedian(window)

    # Compute all local medians before writing anything, so that out may share
    # memory with power_spectrum
    medians = [None] * len(scale)
    for k, count, bins in layout["runs"]:
        block = power_spectrum[starts[k] : starts[k] + count * bins]
        if ignore_zeros:
            block = np.where(block == 0, np.nan, block)
        medians[k : k + count] = list(bn.nanmedian(block.reshape(count, bins), axis=1))
    for k in layout["single"]:
        medians[k] = window_median(power_spectrum[starts[k] : starts[k] + scale[k]])

    used_medians = list(medians)
    for k, new_median in enumerate(medians):
        if not np.isnan(new_median):
            continue
        start = starts[k]
        bins = scale[k]
        old_median = used_medians[k - 1] if k else 1
        i = 0
        while np.isnan(new_median):
            i += 1
            if start + i * bins >= ps_len:
                # all remaining bins are nan
                new_median = old_median
                break
            new_median = window_median(
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
        used_medians[k] = new_median

    mid_bin = layout["mids"][0]
    np.divide(power_spectrum[:mid_bin], used_medians[0] / LOG2, out=out[:mid_bin])
    if len(scale) == 1:
        out[mid_bin:] = 0
    else:
        num = layout["num"]
        if "seg_id" not in layout:
            # compute slope of the power spectra per segment
            for k, (seg_lo, seg_num) in enumerate(zip(layout["seg_lo"], num)):
                median_slope = np.linspace(
                    used_medians[k], used_medians[k + 1], num=seg_num
                )
                median_slope /= LOG2
                np.divide(
                    power_spectrum[seg_lo : seg_lo + seg_num],
                    median_slope,
                    out=out[seg_lo : seg_lo + seg_num],
                )
        else:
            seg_start, seg_stop, delta, step = _interpolate_medians(used_medians, num)
            seg_id = layout["seg_id"]
            offset = layout["offset"]
            # linear interpolation between adjacent medians in a single pass
            median_slope = offset * step[seg_id]
            median_slope += seg_start[seg_id]
            median_slope[layout["seg_end"]] = seg_stop[num > 1]
            # np.linspace uses a different order of operations when the step is 0
            for k in np.flatnonzero((step == 0) & (delta != 0) & (num > 1)):
                seg_lo = int(np.sum(num[:k]))
                median_slope[seg_lo : seg_lo + num[k]] = np.linspace(
                    used_medians[k], used_medians[k + 1], num=num[k]
                )
            median_slope /= LOG2
            np.divide(power_spectrum[mid_bin:], median_slope, out=out[mid_bin:])

    if get_medians:
        return out, medians, list(scale)
    else:
        return out


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
