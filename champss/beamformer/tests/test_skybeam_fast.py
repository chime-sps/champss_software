"""The compiled and streamlined beamformer steps against the code they replace."""

from multiprocessing import shared_memory

import numpy as np
import pytest
import beamformer.skybeam as skybeam
from beamformer.skybeam import (
    SkyBeamFormer,
    _scale_chunk_into,
    detrend_masked_row,
    detrend_masked_row_fast,
)


def channel(n, seed, masked):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = 100 + 5 * np.sin(t / 7000) + 0.0003 * t + rng.normal(0, 1, n)
    mask = rng.random(n) < masked
    mask[n // 3 : n // 3 + 2000] = True  # a masked stretch
    return x.astype(np.float32), mask


@pytest.mark.parametrize("n, step", [(143360, 32768), (1000, 300), (2 * 32768 + 5, 32768)])
@pytest.mark.parametrize("masked", [0, 0.3, 0.97])
def test_detrend_matches(n, step, masked):
    x, mask = channel(n, n + int(10 * masked), masked)
    mask[:step] = True  # one segment with nothing to fit
    expected = detrend_masked_row(
        x[np.newaxis], mask=~mask[np.newaxis], axis=1, type="linear",
        bp=np.arange(0, n, step),
    )[0]
    got = detrend_masked_row_fast(x, ~mask, step)
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)


def test_chunk_scaling_matches_numpy():
    rng = np.random.default_rng(1)
    nf, nt = 64, 1024
    data = rng.normal(0, 1, (nf, nt)).astype(np.float32)
    data[3, 5], data[4, 6], data[5, 7] = np.nan, np.inf, -50
    variance = rng.uniform(0.5, 4, nf).astype(np.float32)
    means = rng.uniform(-1, 10, nf).astype(np.float32)
    means[9] = 0
    good = rng.random((nf, nt)) > 0.1
    cols = slice(100, 900)

    expected_spectra = np.zeros((nf, 2000), np.float32)
    expected_mask = np.ones((nf, 2000), bool)
    chunk = data[:, cols].copy()
    chunk *= np.sqrt(variance)[:, np.newaxis]
    chunk += means[:, np.newaxis]
    expected_spectra[:, 500:1300] = chunk
    expected_mask[:, 500:1300] = (chunk <= 0) | ~np.isfinite(chunk) | ~good[:, cols]

    spectra = np.zeros((nf, 2000), np.float32)
    mask = np.ones((nf, 2000), bool)
    _scale_chunk_into(
        data[:, cols], np.sqrt(variance), means, good[:, cols], spectra, mask, 500,
        np.arange(nf),
    )
    np.testing.assert_array_equal(spectra, expected_spectra)
    np.testing.assert_array_equal(mask, expected_mask)

    # Only some channels held: row k takes channel rows[k]
    rows = np.array([0, 3, 4, 9, 40, 63])
    held_spectra = np.zeros((len(rows), 2000), np.float32)
    held_mask = np.ones((len(rows), 2000), bool)
    _scale_chunk_into(
        data[:, cols], np.sqrt(variance), means, good[:, cols], held_spectra, held_mask,
        500, rows,
    )
    np.testing.assert_array_equal(held_spectra, expected_spectra[rows])
    np.testing.assert_array_equal(held_mask, expected_mask[rows])


def run_fill(spectra, mask, monkeypatch, fast):
    """fill_and_norm_shared_spectra on every (channel, half) of copies in shared memory."""
    if not fast:
        monkeypatch.setattr(skybeam, "masked_median", lambda x, m: None)
    shms = [shared_memory.SharedMemory(create=True, size=a.nbytes) for a in (spectra, mask)]
    try:
        s = np.ndarray(spectra.shape, spectra.dtype, buffer=shms[0].buf)
        m = np.ndarray(mask.shape, bool, buffer=shms[1].buf)
        s[:], m[:] = spectra, mask
        half = spectra.shape[1] // 2
        for c in range(spectra.shape[0]):
            for t in (slice(0, half), slice(half, spectra.shape[1])):
                SkyBeamFormer.fill_and_norm_shared_spectra(
                    None, shms[0].name, shms[1].name, spectra.shape, "float32",
                    512000, slice(c, c + 1), t,
                    detrend_data=True, detrend_nsamp=32768, add_local_median=True,
                )
        return s.copy(), m.copy()
    finally:
        monkeypatch.undo()
        for shm in shms:
            shm.close()
            shm.unlink()


def test_fill_fast_path_matches_general_path(monkeypatch):
    rows, masks = zip(*(channel(2 * 70000, c, f) for c, f in enumerate([0, 0.2, 0.6, 0.95])))
    spectra, mask = np.array(rows), np.array(masks)
    mask[2, :70000] = True  # fully masked half
    spectra[3, mask[3]] = np.nan  # NaN only under the mask
    fast = run_fill(spectra, mask, monkeypatch, fast=True)
    general = run_fill(spectra, mask, monkeypatch, fast=False)
    np.testing.assert_array_equal(fast[0], general[0])
    np.testing.assert_array_equal(fast[1], general[1])
