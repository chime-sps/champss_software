import numpy as np
import pytest
from ps_processes.processes.harmonic_search import (
    formula_bins,
    prepare_fast_search,
    search_rows,
)
from ps_processes.processes.ps_search import calc_harmonic_sum
from sps_common.interfaces.utilities import harmonic_sum, powersum_at_sigma

LEVELS = np.array([1, 2, 4, 8, 16, 32])


def table(num_harm, length):
    return np.vstack(
        (np.arange(length), harmonic_sum(num_harm, np.zeros(length))[1])
    ).astype(np.int32)


def exclude(bins, bad, length):
    # As PowerSpectraSearch.search does for known sources, birdies and mean sigma
    dummy = np.zeros(length)
    dummy[bad] = 1
    bins[dummy[bins] != 0] = 0
    return bins


def cutoffs(bins, days, sigma):
    nsum = np.zeros((6, bins.shape[1]), dtype=int)
    cut = np.zeros((6, bins.shape[1]))
    for i, harm in enumerate(LEVELS):
        n = days[bins[:harm]].sum(0)
        nsum[i] = n
        cut[i] = powersum_at_sigma(sigma, n)
        cut[i, n == 0] = np.inf
    return nsum, cut


def table_search(spectra, bins, cut):
    out = []
    for r, spec in enumerate(spectra):
        sums = calc_harmonic_sum(spec, bins)
        for i in range(6):
            idx = np.where(sums[i] > cut[i])[0]
            out += [(r, i, j, sums[i][j]) for j in idx]
    return out


def fast_search(spectra, num_harm, fast):
    row, level, b, p = search_rows(spectra, np.arange(len(spectra)), num_harm, fast)
    return list(zip(row, level, b, p))


def noisy_spectra(rows, length, seed):
    rng = np.random.default_rng(seed)
    spectra = rng.exponential(1, (rows, length)).astype(np.float32)
    spectra[:, 0] = 0
    # A few strong periodic signals so every harmonic level has detections
    for r in range(rows):
        f = rng.integers(200, length // 40)
        spectra[r, f * np.arange(1, 32)] += rng.uniform(5, 30)
    return spectra


def test_formula_reproduces_harmonic_sum_table():
    length = 32 * 1000 + 17
    np.testing.assert_array_equal(formula_bins(32, length), table(32, length))


def test_per_bin_cutoffs_match_table_search():
    length, sigma = 32 * 2048, 4.0
    rng = np.random.default_rng(1)
    spectra = noisy_spectra(6, length, 2)
    bins = exclude(table(32, length), rng.choice(length, 3000, replace=False), length)
    days = rng.integers(0, 6, length).astype(float)
    days[0] = 0
    nsum, cut = cutoffs(bins, days, sigma)
    fast = prepare_fast_search(bins, 32, nsum, cut, sigma, spectra.dtype)
    assert fast is not None and fast["nsum"] is not None
    expected = table_search(spectra, bins, cut)
    assert len(expected) > 100
    assert fast_search(spectra, 32, fast) == expected


def test_fixed_cutoffs_match_table_search():
    length, sigma = 32 * 2048, 4.0
    spectra = noisy_spectra(4, length, 3)
    bins = exclude(table(32, length), np.arange(5000, 5100), length)
    nsum = LEVELS * 3
    cut = powersum_at_sigma(sigma, nsum)
    fast = prepare_fast_search(bins, 32, nsum, cut, sigma, spectra.dtype)
    assert fast is not None and fast["nsum"] is None
    assert fast_search(spectra, 32, fast) == table_search(spectra, bins, cut)


def test_detections_beyond_the_output_capacity_are_kept():
    length = 32 * 512
    spectra = noisy_spectra(2, length, 4)
    bins = table(32, length)
    cut = powersum_at_sigma(1.0, LEVELS)
    fast = prepare_fast_search(bins, 32, LEVELS, cut, 1.0, spectra.dtype)
    full = search_rows(spectra, [0, 1], 32, fast)
    small = search_rows(spectra, [0, 1], 32, fast, capacity=8)
    for a, b in zip(full, small):
        np.testing.assert_array_equal(a, b)


def test_fewer_harmonics_sum_only_those_rows():
    length, num_harm = 32 * 256, 8
    spectra = noisy_spectra(2, length, 5)
    bins = table(num_harm, length)
    cut = powersum_at_sigma(2.0, LEVELS)
    fast = prepare_fast_search(bins, num_harm, LEVELS, cut, 2.0, spectra.dtype)
    row, level, b, p = search_rows(spectra, [0, 1], num_harm, fast)
    assert level.max() == 3
    for r, lv, j, power in zip(row, level, b, p):
        assert power == np.cumsum(spectra[r][bins[: LEVELS[lv], j]], dtype=np.float32)[-1]


def test_unreproducible_table_falls_back():
    length = 32 * 64
    bins = table(32, length)
    bins[5, 100] += 1
    cut = powersum_at_sigma(6.0, LEVELS)
    assert prepare_fast_search(bins, 32, LEVELS, cut, 6.0, np.float32) is None
    assert prepare_fast_search(table(32, length), 32, LEVELS, cut, 6.0, np.float16) is None
