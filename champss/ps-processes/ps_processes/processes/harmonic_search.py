"""
Harmonic-sum search kernel that computes harmonic bins instead of reading them.

The search sums up to 32 harmonics for every bin j of a power spectrum. Harmonic row h
of the table built by sps_common.interfaces.utilities.harmonic_sum is
round(j * num_h / 32), rounded half to even like np.round, with num_h fixed per row.
Filters (known sources, birdies, high mean sigma) redirect harmonics that land on an
excluded bin to bin 0. This module computes those bins on the fly from num_h and a
per-bin exclusion flag, so the 32 x N index table is never read during the search, and
only returns the bins above the detection threshold.

Sums are accumulated in the dtype of the spectrum and in the same harmonic order as the
table-based kernel, and compared to the same float64 cutoffs, so detections and their
summed powers are identical. prepare_fast_search checks that once per search and returns
None when the table cannot be reproduced, in which case the caller keeps the old kernel.
"""

import numpy as np
from numba import njit

ALLOWED_HARMONICS = np.array([1, 2, 4, 8, 16, 32])

# Numerators over 32 of the fractional harmonic rows, in table order:
# 1, 1/2, 1/4, 3/4, odd/8, odd/16, odd/32
HARMONIC_NUM32 = np.array(
    [32, 16, 8, 24]
    + [4 * i for i in (1, 3, 5, 7)]
    + [2 * i for i in range(1, 16, 2)]
    + list(range(1, 32, 2)),
    dtype=np.int64,
)

BLOCK = 4096


def formula_bins(num_harm, length):
    """The harmonic bin table, as harmonic_sum builds it, from the row fractions."""
    j = np.arange(length, dtype=np.int64)
    p = j[None, :] * HARMONIC_NUM32[:num_harm, None]
    q, r = p >> 5, p & 31
    return (q + ((r > 16) | ((r == 16) & ((q & 1) == 1)))).astype(np.int64)


def prepare_fast_search(
    harm_bins,
    num_harm,
    nsum_per_harmonic,
    power_cutoff_per_harmonic,
    sigma_min,
    spectra_dtype,
):
    """
    Derive the inputs of harmonic_search_rows and check that they reproduce the table.

    Parameters
    ==========
    harm_bins: np.ndarray
        The (num_harm, N) harmonic bin table used by the search, after filtering.
    num_harm: int
        Number of harmonics searched.
    nsum_per_harmonic: np.ndarray
        Summed day counts per harmonic level, (6, N) or (6,).
    power_cutoff_per_harmonic: np.ndarray
        Detection cutoff per harmonic level, (6, N) or (6,).
    sigma_min: float
        Detection threshold, used to tabulate the cutoff per summed day count.
    spectra_dtype: np.dtype
        dtype of the power spectra; only float32 and float64 are supported.

    Returns
    =======
    fast: dict or None
        excl (uint8, N), nsum (uint16, (levels, N)) or None, lut (float64) and
        level_cutoff (float64, levels); None if the table cannot be reproduced.
    """
    from sps_common.interfaces.utilities import powersum_at_sigma

    if np.dtype(spectra_dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        return None
    harm_bins = np.asarray(harm_bins)
    if harm_bins.ndim != 2 or harm_bins.shape[0] != num_harm or num_harm > 32:
        return None
    length = harm_bins.shape[1]
    formula = formula_bins(num_harm, length)
    zeroed = (harm_bins == 0) & (formula != 0)
    excl = np.zeros(length, dtype=np.uint8)
    excl[formula[zeroed]] = 1
    if not np.array_equal(harm_bins, np.where(excl[formula] == 1, 0, formula)):
        return None
    levels = int((ALLOWED_HARMONICS <= num_harm).sum())
    fast = dict(excl=excl, nsum=None, lut=np.zeros(1), level_cutoff=np.zeros(levels))
    nsum = np.asarray(nsum_per_harmonic)
    cutoff = np.asarray(power_cutoff_per_harmonic)
    if nsum.ndim == 1:
        # The table-based search compares a row of sums with one cutoff scalar, which
        # numpy 1.x does in the dtype of the sums; follow whatever numpy does here.
        compare_dtype = np.result_type(np.empty(1, spectra_dtype), cutoff[0])
        fast["level_cutoff"] = cutoff[:levels].astype(compare_dtype)
        return fast
    nsum = nsum[:levels]
    if nsum.min() < 0 or nsum.max() >= 2**16:
        return None
    lut = np.asarray(
        powersum_at_sigma(sigma_min, np.arange(nsum.max() + 1)), dtype=np.float64
    )
    lut[0] = np.inf
    if not np.array_equal(lut[nsum], cutoff[:levels]):
        return None
    fast["nsum"] = nsum.astype(np.uint16)
    fast["lut"] = lut
    return fast


@njit(cache=True)
def harmonic_search_rows(
    spectra,
    rows,
    num32,
    num_harm,
    excl,
    nsum,
    use_nsum,
    lut,
    level_cutoff,
    out_row,
    out_level,
    out_bin,
    out_power,
):
    """
    Harmonic sums of the given rows of spectra, keeping the bins above the cutoff.

    Parameters
    ==========
    spectra: 2D array
        Power spectra, (ndm, N), already cut to the searched length.
    rows: int64 array
        The rows of spectra to search.
    num32, num_harm, excl, nsum, use_nsum, lut, level_cutoff:
        From prepare_fast_search; nsum is ignored unless use_nsum.
    out_row, out_level, out_bin, out_power: arrays
        Filled with (position in rows, harmonic level, bin, summed power) per detection.

    Returns
    =======
    count: int
        Number of detections. If larger than the output arrays, only the first ones are
        stored and the caller should call again with larger arrays.
    """
    length = spectra.shape[1]
    cap = out_bin.shape[0]
    count = 0
    acc = np.empty(BLOCK, dtype=spectra.dtype)
    # For j = 32 a + t, bin = a * num + offset[parity of a * num, t]: the pattern repeats
    # every 32 bins, except that ties round to even, which depends on that parity.
    offset = np.empty((num_harm, 2, 32), dtype=np.int64)
    for h in range(num_harm):
        for parity in range(2):
            for t in range(32):
                p = t * num32[h]
                q = p >> 5
                r = p & 31
                up = r > 16 or (r == 16 and ((parity + q) & 1) == 1)
                offset[h, parity, t] = q + (1 if up else 0)
    any_excl = False
    for b in range(length):
        if excl[b]:
            any_excl = True
            break
    work = np.empty(length, dtype=spectra.dtype)
    for ri in range(rows.shape[0]):
        spec = spectra[rows[ri]]
        if any_excl:
            # Excluded harmonics take the value of bin 0, as in the filtered table
            for b in range(length):
                work[b] = spec[0] if excl[b] else spec[b]
            spec = work
        for j0 in range(0, length, BLOCK):
            m = min(BLOCK, length - j0)
            for k in range(m):
                acc[k] = 0
            level = 0
            for h in range(num_harm):
                num = num32[h]
                full = m - m % 32
                for a0 in range(0, full, 32):
                    base = ((j0 + a0) >> 5) * num
                    off = offset[h, base & 1]
                    for t in range(32):
                        acc[a0 + t] += spec[base + off[t]]
                for k in range(full, m):
                    p = (j0 + k) * num
                    q = p >> 5
                    r = p & 31
                    b = q + (1 if (r > 16 or (r == 16 and (q & 1) == 1)) else 0)
                    acc[k] += spec[b]
                if h + 1 == (1 << level):
                    for k in range(m):
                        if use_nsum:
                            cut = lut[nsum[level, j0 + k]]
                        else:
                            cut = level_cutoff[level]
                        if acc[k] > cut:
                            if count < cap:
                                out_row[count] = ri
                                out_level[count] = level
                                out_bin[count] = j0 + k
                                out_power[count] = acc[k]
                            count += 1
                    level += 1
    return count


def search_rows(spectra, rows, num_harm, fast, capacity=1 << 16):
    """
    Run harmonic_search_rows, growing the output if needed, sorted by row, level, bin.

    Returns
    =======
    row, level, bin, power: np.ndarray
        One entry per detection; row is the position in rows.
    """
    rows = np.asarray(rows, dtype=np.int64)
    use_nsum = fast["nsum"] is not None
    nsum = fast["nsum"] if use_nsum else np.zeros((1, 1), dtype=np.uint16)
    while True:
        out_row = np.empty(capacity, dtype=np.int64)
        out_level = np.empty(capacity, dtype=np.int64)
        out_bin = np.empty(capacity, dtype=np.int64)
        out_power = np.empty(capacity, dtype=spectra.dtype)
        count = harmonic_search_rows(
            spectra,
            rows,
            HARMONIC_NUM32,
            num_harm,
            fast["excl"],
            nsum,
            use_nsum,
            fast["lut"],
            fast["level_cutoff"],
            out_row,
            out_level,
            out_bin,
            out_power,
        )
        if count <= capacity:
            break
        capacity = count
    order = np.lexsort((out_bin[:count], out_level[:count], out_row[:count]))
    return (
        out_row[:count][order],
        out_level[:count][order],
        out_bin[:count][order],
        out_power[:count][order],
    )
