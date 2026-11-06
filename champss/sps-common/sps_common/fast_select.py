"""
Exact order statistics of float arrays, compiled.

_kth_and_below gives the k-th smallest value of an array, and optionally the (k-1)-th,
exactly as a full sort would, without sorting: long inputs are bracketed with a sorted
sample, only the values in the bracket are selected, and anything the bracket misses
falls back to selecting the whole input. Used for medians that must match numpy's to the
bit (red-noise normalisation, beamformer fill).
"""

import numpy as np
from numba import njit


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
def _masked_median32(x, mask, buf, work):
    """Median of x where mask is False; ok is False if one of those is not finite."""
    n = 0
    finite = True
    for i in range(x.shape[0]):
        v = x[i]
        keep = not mask[i]
        buf[n] = v
        n += np.int64(keep)
        if keep and not np.isfinite(v):
            finite = False
    if n == 0 or not finite:
        return np.float32(np.nan), False
    even = n % 2 == 0
    upper, lower = _kth_and_below(buf, n, n // 2, even, work)
    if not even:
        return upper, True
    return np.float32(lower + upper) / np.float32(2), True


def masked_median(x, mask):
    """
    np.ma.median(np.ma.masked_array(x, mask)) of float32 data, exactly, without sorting.

    For a flattened median, numpy takes the middle unmasked value, or for an even count
    the float32 sum of the two middle values divided by 2. Returns None instead where
    that is not the whole story: no unmasked values, an unmasked NaN or infinity (numpy
    then returns NaN, or its result depends on how ties with the fill value sort), or a
    zero median (whose sign could depend on the sort order). Callers then use numpy.
    The value is returned as float32; numpy gives float64 for an even count, but the
    same value, as halving is exact.
    """
    x = np.ascontiguousarray(x, dtype=np.float32).ravel()
    mask = np.ascontiguousarray(mask, dtype=np.bool_).ravel()
    buf = np.empty(2 * x.size, dtype=np.float32)
    median, ok = _masked_median32(x, mask, buf[: x.size], buf[x.size :])
    if not ok or median == 0:
        return None
    return np.float32(median)
