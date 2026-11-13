"""The faster RFI code paths against the numpy code they replace, which they must match exactly."""

import warnings
from multiprocessing import shared_memory

import numpy as np
import pytest
from rfi_mitigation.cleaners import cleaners
from rfi_mitigation.pipeline import merge_cleaner_mask
from rfi_mitigation.utilities.cleaner_utils import (
    _known_bad_channels,
    combine_cleaner_masks,
    known_bad_channels,
)
from rfi_mitigation.utilities.fast_stats import binned_means_zero_as_nan


def numpy_binned_means(spectra, time_bin):
    work = spectra.copy()
    work[work == 0] = np.nan
    nbins = spectra.shape[1] // time_bin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return work[:, : nbins * time_bin].reshape(len(spectra), -1, time_bin).mean(axis=-1)


@pytest.mark.parametrize(
    "shape, time_bin",
    [((16, 100000), 1024), ((8, 5000), 37), ((8, 5000), 5), ((4, 3000), 128),
     ((3, 1000), 2000), ((5, 130), 129)],
)
@pytest.mark.parametrize("scale", [1.0, 1e5])
def test_binned_means_match_numpy(shape, time_bin, scale):
    rng = np.random.default_rng(shape[1] + time_bin)
    spectra = (rng.normal(1, 3, shape) * scale).astype(np.float32)
    spectra[rng.random(shape) < 0.001] = 0
    spectra[0, 3], spectra[1, 4] = -0.0, np.inf
    spectra[2 % shape[0], 5] = np.nan
    expected = numpy_binned_means(spectra, time_bin)
    got = binned_means_zero_as_nan(spectra, time_bin)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got.view(np.int32), expected.view(np.int32))


def run_stddev(spectra, mask, monkeypatch, fast):
    """The cleaner on copies in shared memory; with fast=False its binned means come from
    the numpy expression the kernel replaced (the rest of the cleaner is unchanged)."""
    if not fast:
        monkeypatch.setattr(cleaners, "binned_means_zero_as_nan", numpy_binned_means)
    shms = [shared_memory.SharedMemory(create=True, size=a.nbytes) for a in (spectra, mask)]
    try:
        s = np.ndarray(spectra.shape, np.float32, buffer=shms[0].buf)
        m = np.ndarray(mask.shape, bool, buffer=shms[1].buf)
        s[:], m[:] = spectra, mask
        cleaner = cleaners.StdDevChannelCleaner(spectra.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaner.clean(shms[0].name, shms[1].name, spectra.shape, np.float32)
        return m.copy(), cleaner.get_mask().copy(), cleaner.bad_channels
    finally:
        monkeypatch.undo()
        for shm in shms:
            shm.close()
            shm.unlink()


def test_stddev_cleaner_flags_the_same_channels(monkeypatch):
    rng = np.random.default_rng(3)
    nchan, ntime = 64, 40 * 1024 + 100
    spectra = rng.normal(10, 1, (nchan, ntime)).astype(np.float32)
    spectra[5] *= 3  # a noisy channel
    spectra[9] = 10  # a flat one
    spectra[12, : 5 * 1024] = 0  # zero-filled stretch
    mask = np.zeros((nchan, ntime), bool)
    mask[40:52] = True  # heavily flagged channels
    mask[:, 100:200] |= rng.random((nchan, 100)) < 0.3
    fast = run_stddev(spectra, mask, monkeypatch, fast=True)
    slow = run_stddev(spectra, mask, monkeypatch, fast=False)
    np.testing.assert_array_equal(fast[2], slow[2])
    np.testing.assert_array_equal(fast[0], slow[0])
    assert fast[2].any()
    # get_mask gives the flagged channels as a full mask
    np.testing.assert_array_equal(fast[1], np.repeat(fast[2][:, None], ntime, axis=1))


@pytest.mark.parametrize("broadcast", [False, True])
def test_merge_matches_combine(broadcast):
    rng = np.random.default_rng(4)
    full = rng.random((32, 500)) < 0.2
    view = full[:, 100:300]  # a slice of the shared mask, as the pipeline merges into
    other = rng.random((32, 1)) < 0.3 if broadcast else rng.random((32, 200)) < 0.1
    expected, _ = combine_cleaner_masks(np.array([view, np.broadcast_to(other, view.shape)]))
    merge_cleaner_mask(view, other)
    np.testing.assert_array_equal(view, expected)


@pytest.mark.parametrize("nchan", [1024, 2048, 4096, 16384])
def test_cached_bad_channels_match(nchan):
    expected = [int(c) for c in _known_bad_channels(nchan)]
    assert known_bad_channels(nchan) == expected
    assert known_bad_channels(nchan) == expected  # from the cache


@pytest.mark.parametrize("cols", [slice(0, 1024), slice(1000, 2024), slice(5, 12), slice(3000, 3997)])
def test_sk_sums_and_estimator_match_numpy(cols):
    from rfi_mitigation.utilities.cleaner_utils import generalised_spectral_kurtosis
    from rfi_mitigation.utilities.fast_stats import nansum_rows_and_squares

    rng = np.random.default_rng(cols.start)
    full = (rng.exponential(1, (256, 4000)) * 1e3).astype(np.float32)
    full[rng.random(full.shape) < 0.05] = np.nan
    full[3, cols.start], full[4, cols.start + 1] = np.inf, 0
    full[7] = np.nan
    spec = full[:, cols]  # a non-contiguous view, as the cleaner passes
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        s1, s2 = np.nansum(spec, axis=1), np.nansum(spec**2, axis=1)
        m, delta = spec.shape[1], 1536.0
        expected = ((m * delta + 1) / float(m - 1)) * (m * (s2 / s1**2) - 1)
        got = generalised_spectral_kurtosis(spec, m=m, delta=delta)
    g1, g2 = nansum_rows_and_squares(spec)
    np.testing.assert_array_equal(g1.view(np.int32), s1.view(np.int32))
    np.testing.assert_array_equal(g2.view(np.int32), s2.view(np.int32))
    np.testing.assert_array_equal(got.view(np.int32), expected.view(np.int32))


def test_binned_means_serial_equals_parallel():
    from rfi_mitigation.utilities import fast_stats

    rng = np.random.default_rng(4)
    spectra = rng.random((37, 4096)).astype(np.float32)
    spectra[3, 100] = 0
    spectra[5] = 0
    a = fast_stats._binned_means_zero_as_nan(spectra, 4, 1024)
    b = fast_stats._binned_means_zero_as_nan_serial(spectra, 4, 1024)
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    np.testing.assert_array_equal(
        fast_stats.binned_means_zero_as_nan(spectra[2:3], 1024).view(np.uint32),
        a[2:3].view(np.uint32),
    )
