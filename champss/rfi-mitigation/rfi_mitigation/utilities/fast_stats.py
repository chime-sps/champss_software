"""
Compiled statistics that reproduce numpy's float32 results to the bit.

numpy sums a contiguous float32 run pairwise: runs of up to 128 values with eight
interleaved partial sums, combined as ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)), and longer
runs split in two at a multiple of 8 below the middle. Following the same order gives
the same rounding, so these kernels give exactly what the numpy expressions they replace
give, without the temporary arrays.
"""

import multiprocessing

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _block_sum(a, start, n):
    """numpy's pairwise sum of a run of at most 128 values."""
    if n < 8:
        res = a.dtype.type(0)
        for i in range(n):
            res += a[start + i]
        return res
    r0 = a[start]
    r1 = a[start + 1]
    r2 = a[start + 2]
    r3 = a[start + 3]
    r4 = a[start + 4]
    r5 = a[start + 5]
    r6 = a[start + 6]
    r7 = a[start + 7]
    i = 8
    end = n - n % 8
    while i < end:
        r0 += a[start + i]
        r1 += a[start + i + 1]
        r2 += a[start + i + 2]
        r3 += a[start + i + 3]
        r4 += a[start + i + 4]
        r5 += a[start + i + 5]
        r6 += a[start + i + 6]
        r7 += a[start + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[start + i]
        i += 1
    return res


@njit(cache=True)
def _pairwise_sum(a, start, n):
    """
    numpy's pairwise sum of a[start:start + n], in a's dtype.

    Runs of more than 128 values are the sum of their two halves (split at a multiple of
    8 below the middle), left plus right. This walks that same tree with an explicit
    stack rather than by recursion, as numba cannot reload a recursive function from its
    cache.
    """
    if n <= 128:
        return _block_sum(a, start, n)
    starts = np.empty(64, dtype=np.int64)
    lengths = np.empty(64, dtype=np.int64)
    states = np.empty(64, dtype=np.int64)  # 1: summing the left half, 2: the right
    lefts = np.empty(64, dtype=a.dtype)
    top = 0
    starts[0] = start
    lengths[0] = n
    states[0] = 0
    while True:
        m = lengths[top]
        if m > 128:
            # Sum the left half first
            half = m // 2
            half -= half % 8
            states[top] = 1
            top += 1
            starts[top] = starts[top - 1]
            lengths[top] = half
            continue
        value = _block_sum(a, starts[top], m)
        top -= 1
        # Hand the value up until a frame still needs its right half
        while True:
            if top < 0:
                return value
            if states[top] == 1:
                lefts[top] = value
                states[top] = 2
                half = lengths[top] // 2
                half -= half % 8
                top += 1
                starts[top] = starts[top - 1] + half
                lengths[top] = lengths[top - 1] - half
                break
            value = lefts[top] + value
            top -= 1


@njit(cache=True, parallel=True)
def _binned_means_zero_as_nan(spectra, nbins, time_bin):
    nchan = spectra.shape[0]
    out = np.empty((nchan, nbins), dtype=spectra.dtype)
    count = spectra.dtype.type(time_bin)
    for c in prange(nchan):
        row = spectra[c]
        for k in range(nbins):
            start = k * time_bin
            has_zero = False
            for i in range(start, start + time_bin):
                if row[i] == 0:
                    has_zero = True
                    break
            if has_zero:
                out[c, k] = np.nan
            else:
                out[c, k] = _pairwise_sum(row, start, time_bin) / count
    return out


@njit(cache=True)
def _binned_means_zero_as_nan_serial(spectra, nbins, time_bin):
    """
    _binned_means_zero_as_nan without threads, for pool workers: each bin is computed on
    its own either way, so the values are identical, and a worker starts no thread pool.
    A function of its own, not the parallel one recompiled: numba names cache files after
    the source function, so the two would share a cache entry and could load each other.
    """
    nchan = spectra.shape[0]
    out = np.empty((nchan, nbins), dtype=spectra.dtype)
    count = spectra.dtype.type(time_bin)
    for c in range(nchan):
        row = spectra[c]
        for k in range(nbins):
            start = k * time_bin
            has_zero = False
            for i in range(start, start + time_bin):
                if row[i] == 0:
                    has_zero = True
                    break
            if has_zero:
                out[c, k] = np.nan
            else:
                out[c, k] = _pairwise_sum(row, start, time_bin) / count
    return out


def binned_means_zero_as_nan(spectra, time_bin):
    """
    Means of consecutive time_bin samples of each channel, NaN where a bin holds a zero.

    The same values as

        work = spectra.copy(); work[work == 0] = np.nan
        nbins = spectra.shape[1] // time_bin
        work[:, :nbins * time_bin].reshape(len(spectra), -1, time_bin).mean(axis=-1)

    for C-contiguous float32 spectra, without copying them. Channels run in parallel in the
    main process (NUMBA_NUM_THREADS threads; see sps_common.cpus); in a pool worker, and for
    a single row, serially. The parallel kernel starts numba's OpenMP threads, after which
    numba stops any process this one forks, so it is used only where nothing forks later.
    """
    if spectra.dtype != np.float32 or not spectra.flags.c_contiguous:
        raise ValueError("binned_means_zero_as_nan needs C-contiguous float32 spectra")
    serial = spectra.shape[0] == 1 or multiprocessing.parent_process() is not None
    kernel = _binned_means_zero_as_nan_serial if serial else _binned_means_zero_as_nan
    return kernel(spectra, spectra.shape[1] // time_bin, time_bin)


@njit(cache=True)
def _nansum_rows_and_squares(spec):
    nrow, n = spec.shape
    s1 = np.empty(nrow, dtype=spec.dtype)
    s2 = np.empty(nrow, dtype=spec.dtype)
    values = np.empty(n, dtype=spec.dtype)
    squares = np.empty(n, dtype=spec.dtype)
    zero = spec.dtype.type(0)
    for r in range(nrow):
        for i in range(n):
            v = spec[r, i]
            if np.isnan(v):
                values[i] = zero
                squares[i] = zero
            else:
                values[i] = v
                squares[i] = v * v
        s1[r] = _pairwise_sum(values, 0, n)
        s2[r] = _pairwise_sum(squares, 0, n)
    return s1, s2


def nansum_rows_and_squares(spec):
    """
    np.nansum(spec, axis=1) and np.nansum(spec**2, axis=1) of a 2D float32 array, exactly,
    in one pass and without copying it (numpy replaces NaN in a copy, then sums each
    row pairwise).
    """
    if spec.dtype != np.float32 or spec.ndim != 2:
        raise ValueError("nansum_rows_and_squares needs a 2D float32 array")
    return _nansum_rows_and_squares(spec)
