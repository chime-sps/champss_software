"""
Red-noise normalisation of float32 power spectra in one compiled pass.

rednoise_normalise_into gives exactly the values that rednoise_normalise does, assigned
back into a float32 array, and writes them straight into the output row. It follows
numpy's arithmetic step by step:

- Each window's median is that of its non-NaN values, in float32; for an even count it
  is the float32 sum of the two middle values divided by 2, as np.nanmedian computes it.
- Bins before the middle of the first window are divided, in float32, by the median
  divided by ln 2 and rounded to float32 (numpy casts that scalar to the array's dtype).
- Later bins are divided, in float64, by np.linspace between neighbouring medians
  (computed in float64 exactly as numpy 1.x does) divided by ln 2, then rounded to float32.

A window whose bins are all NaN makes rednoise_normalise look further along the
spectrum; that rare case is not reproduced here, and the caller should fall back to
rednoise_normalise when this returns False.
"""

import numpy as np
from numba import njit
from sps_common.fast_select import _kth_and_below

LN2 = np.log(2)


@njit(cache=True)
def _nanmedian32(x, start, stop, buf, work):
    """np.nanmedian(x[start:stop]) for float32 x; NaN if there are no finite values."""
    n = 0
    for i in range(start, stop):
        v = x[i]
        buf[n] = v
        n += np.int64(not np.isnan(v))
    if n == 0:
        return np.float32(np.nan)
    even = n % 2 == 0
    upper, lower = _kth_and_below(buf, n, n // 2, even, work)
    if not even:
        return upper
    return np.float32(lower + upper) / np.float32(2)


@njit(cache=True)
def _divide_by_slope(x, out, lo, hi, first, last, ln2):
    """out[lo:hi] = x[lo:hi] / (np.linspace(first, last, hi - lo) / ln2), as numpy does."""
    num = hi - lo
    div = num - 1
    start = np.float64(first)
    delta = np.float64(last) - start
    step = delta / div if div > 0 else np.nan
    for i in range(num):
        if num > 1 and i == num - 1:
            y = np.float64(last)
        elif div > 0 and step == 0:
            y = (i / div) * delta + start
        elif div > 0:
            y = i * step + start
        else:
            y = i * delta + start
        out[lo + i] = np.float32(np.float64(x[lo + i]) / (y / ln2))


@njit(cache=True)
def _rednoise_into(x, out, scale, medians, buf, work, ln2):
    ps_len = x.shape[0]
    start = 0
    old_mid = 0
    old_median = np.float32(1)
    covered = 0  # bins written so far; rednoise_normalise leaves the rest at zero
    for w in range(scale.shape[0]):
        bins = scale[w]
        mid = int(start + bins / 2)
        median = _nanmedian32(x, start, min(start + bins, ps_len), buf, work)
        medians[w] = median
        if np.isnan(median):
            return False
        if start == 0:
            d = np.float32(np.float64(median) / ln2)
            covered = min(mid, ps_len)
            for i in range(covered):
                out[i] = x[i] / d
        elif start + bins >= ps_len:
            _divide_by_slope(x, out, old_mid, ps_len, old_median, median, ln2)
            covered = ps_len
        else:
            _divide_by_slope(x, out, old_mid, mid, old_median, median, ln2)
            covered = max(covered, mid)
        start += bins
        old_mid = mid
        old_median = median
    for i in range(covered, ps_len):
        out[i] = 0
    return True


def rednoise_normalise_into(power_spectrum, out, scale, buf):
    """
    Red-noise normalise a float32 spectrum into out, as rednoise_normalise would.

    Parameters
    ==========
    power_spectrum: np.ndarray
        float32 spectrum to normalise, with NaN in masked bins. Not modified.
    out: np.ndarray
        float32 array of the same length for the result; may be shared memory.
    scale: np.ndarray
        int64 window widths, from rednoise_scale(len(power_spectrum), b0, bmax).
    buf: np.ndarray
        float32 work space at least twice as long as the widest window.

    Returns
    =======
    medians: np.ndarray or None
        float32 median of each window, or None if a window has no finite values, in
        which case out is incomplete and rednoise_normalise must be used instead.
    """
    medians = np.empty(len(scale), dtype=np.float32)
    half = len(buf) // 2
    ok = _rednoise_into(power_spectrum, out, scale, medians, buf[:half], buf[half:], LN2)
    return medians if ok else None
