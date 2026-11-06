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

LN2 = np.log(2)


@njit(cache=True)
def _select(buf, n, k):
    """Partially sort buf[:n] so buf[k] is its k-th smallest value, smaller ones before."""
    lo, hi = 0, n - 1
    while hi > lo:
        # Median of three as pivot, then a Hoare partition
        mid = (lo + hi) // 2
        a, b, c = buf[lo], buf[mid], buf[hi]
        if a < b:
            pivot = b if b < c else (c if a < c else a)
        else:
            pivot = a if a < c else (c if b < c else b)
        i, j = lo, hi
        while i <= j:
            while buf[i] < pivot:
                i += 1
            while buf[j] > pivot:
                j -= 1
            if i <= j:
                buf[i], buf[j] = buf[j], buf[i]
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return
    return


# Windows longer than this find their median through a sorted sample (below)
SAMPLE_FROM = 4096
SAMPLE_SIZE = 1024
SAMPLE_MARGIN = 64


@njit(cache=True)
def _kth_and_below(buf, n, k, need_below, work):
    """
    The k-th smallest of buf[:n] and, if need_below, the (k-1)-th, both exactly.

    Long inputs are bracketed between two values of a sorted, evenly strided sample;
    one pass keeps only the values inside the bracket (copied to work) and counts those
    below it, and the selection runs on that small set. If the bracket misses k, the
    whole input is selected instead. buf[:n] is left unchanged unless that happens.
    """
    if n > SAMPLE_FROM:
        sample = np.empty(SAMPLE_SIZE, dtype=buf.dtype)
        for j in range(SAMPLE_SIZE):
            sample[j] = buf[(j * n) // SAMPLE_SIZE]
        sample.sort()
        pos = (k * SAMPLE_SIZE) // n
        lo_i = pos - SAMPLE_MARGIN
        hi_i = pos + SAMPLE_MARGIN
        lo = sample[lo_i] if lo_i > 0 else -np.inf
        hi = sample[hi_i] if hi_i < SAMPLE_SIZE - 1 else np.inf
        # Branch-free on purpose: the comparisons are unpredictable, and this loop
        # is most of the cost
        below = 0
        m = 0
        for i in range(n):
            v = buf[i]
            lt = np.int64(v < lo)
            work[m] = v
            m += np.int64(v <= hi) - lt
            below += lt
        kk = k - below
        if 0 <= kk < m and (not need_below or kk > 0 or below > 0):
            _select(work, m, kk)
            kth = work[kk]
            if not need_below:
                return kth, kth
            if kk == 0:
                # The value just below k is the largest one under the bracket
                max_below = -np.inf
                for i in range(n):
                    v = buf[i]
                    if v < lo and v > max_below:
                        max_below = v
                return kth, buf.dtype.type(max_below)
            lower = work[0]
            for i in range(1, kk):
                if work[i] > lower:
                    lower = work[i]
            return kth, lower
    _select(buf, n, k)
    kth = buf[k]
    lower = kth
    if need_below:
        lower = buf[0]
        for i in range(1, k):
            if buf[i] > lower:
                lower = buf[i]
    return kth, lower


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
