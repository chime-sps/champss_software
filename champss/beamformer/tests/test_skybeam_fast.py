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


def reference_zero_replace(spectra, rfi_mask):
    """zero_replace_shared_spectra with flatten_bandpass, as numpy code (the original)."""
    if np.count_nonzero(rfi_mask) / rfi_mask.size == 1.0:
        return
    _contains_nan = np.isnan(np.min(spectra))
    if rfi_mask.mean() > 0.75 or _contains_nan or np.max(spectra) == 0:
        spectra[:] = 0
        rfi_mask[:] = True
        return
    zero = spectra == 0
    if zero.sum() > 0:
        spectra[~zero] = (spectra[~zero] - spectra[~zero].mean()) / spectra[~zero].std()
        spectra[zero] = 0
    else:
        spectra[:] = (spectra - spectra.mean()) / spectra.std()


def zero_replace_row(row, mask):
    shms = [shared_memory.SharedMemory(create=True, size=max(a.nbytes, 1)) for a in (row, mask)]
    try:
        s = np.ndarray(row.shape, np.float32, buffer=shms[0].buf)
        m = np.ndarray(mask.shape, bool, buffer=shms[1].buf)
        s[:], m[:] = row, mask
        SkyBeamFormer.zero_replace_shared_spectra(
            None, shms[0].name, shms[1].name, row.shape, "float32", slice(0, 1),
            flatten_bandpass=True,
        )
        return s.copy(), m.copy()
    finally:
        del s, m
        for shm in shms:
            shm.close()
            shm.unlink()


def zero_replace_case(kind, n, seed):
    rng = np.random.default_rng(seed)
    row = rng.normal(5, 2, n).astype(np.float32) * rng.choice([1, 1e3, 1e-3])
    mask = rng.random(n) < 0.1
    if kind == "zeros":
        row[n // 4 : n // 2] = 0  # a beam with no data in this channel
        row[min(7, n - 1)] = -0.0
    elif kind == "nan":
        row[n // 3] = np.nan
    elif kind == "mostly_masked":
        mask[: int(0.8 * n)] = True
    elif kind == "all_masked":
        mask[:] = True
    elif kind == "max_zero":
        row = -np.abs(row)
        row[3] = 0
    elif kind == "constant":
        row[:] = 2.5
    elif kind == "negative":
        row -= 6
    return row.reshape(1, -1), mask.reshape(1, -1)


@pytest.mark.parametrize("n", [5, 100, 8191, 8192, 8193, 70001, 573440])
@pytest.mark.parametrize(
    "kind",
    ["noise", "zeros", "nan", "mostly_masked", "all_masked", "max_zero", "constant", "negative"],
)
def test_zero_replace_matches_numpy(n, kind):
    row, mask = zero_replace_case(kind, n, seed=n)
    expected_row, expected_mask = row.copy(), mask.copy()
    with np.errstate(all="ignore"):
        reference_zero_replace(expected_row, expected_mask)
        got_row, got_mask = zero_replace_row(row, mask)
    np.testing.assert_array_equal(got_row.view(np.int32), expected_row.view(np.int32))
    np.testing.assert_array_equal(got_mask, expected_mask)


def test_zero_replace_follows_numpys_buffer_size():
    row, mask = zero_replace_case("zeros", 70001, seed=3)
    old = np.getbufsize()
    try:
        np.setbufsize(4096)
        expected_row, expected_mask = row.copy(), mask.copy()
        reference_zero_replace(expected_row, expected_mask)
        got_row, _ = zero_replace_row(row, mask)
    finally:
        np.setbufsize(old)
    np.testing.assert_array_equal(got_row.view(np.int32), expected_row.view(np.int32))


def zero_replace_rows(rows, mask, **kwargs):
    """zero_replace_shared_spectra on every row of copies in shared memory; returns the
    arrays and the summed (masked, fully masked) counts."""
    shms = [shared_memory.SharedMemory(create=True, size=max(a.nbytes, 1)) for a in (rows, mask)]
    try:
        s = np.ndarray(rows.shape, np.float32, buffer=shms[0].buf)
        m = np.ndarray(mask.shape, bool, buffer=shms[1].buf)
        s[:], m[:] = rows, mask
        per = kwargs.pop("per_slice", 1)
        totals = [0, 0]
        for start in range(0, rows.shape[0], per):
            masked, full = SkyBeamFormer.zero_replace_shared_spectra(
                None, shms[0].name, shms[1].name, rows.shape, "float32",
                slice(start, start + per), **kwargs,
            )
            totals[0] += masked
            totals[1] += full
        return s.copy(), m.copy(), totals
    finally:
        del s, m
        for shm in shms:
            shm.close()
            shm.unlink()


@pytest.mark.parametrize("per_slice", [1, 3])  # one channel (kernel) and several (numpy)
def test_zeroing_masked_first_matches_copyto_then_zero_replacement(per_slice):
    kinds = ["noise", "zeros", "nan", "mostly_masked", "all_masked", "max_zero", "constant", "negative"]
    rows, masks = zip(*(zero_replace_case(k, 20000, seed=i) for i, k in enumerate(kinds)))
    rows, mask = np.concatenate(rows), np.concatenate(masks)
    rows = np.concatenate([rows, rows[:1] + 1])  # 9 rows: per_slice 3 covers them all
    mask = np.concatenate([mask, mask[:1]])
    expected_rows, expected_mask = rows.copy(), mask.copy()
    np.copyto(expected_rows, 0, where=expected_mask)
    with np.errstate(all="ignore"):
        for start in range(0, len(rows), per_slice):
            sl = slice(start, start + per_slice)
            if per_slice == 1:
                reference_zero_replace(expected_rows[sl], expected_mask[sl])
        got_rows, got_mask, (masked, full) = zero_replace_rows(
            rows, mask, per_slice=per_slice, flatten_bandpass=True, zero_masked_first=True
        )
        if per_slice > 1:
            # The numpy path over several channels, after the main process's copyto
            expected_rows, expected_mask, _ = zero_replace_rows(
                np.where(mask, np.float32(0), rows), mask, per_slice=per_slice,
                flatten_bandpass=True,
            )
    np.testing.assert_array_equal(got_rows.view(np.int32), expected_rows.view(np.int32))
    np.testing.assert_array_equal(got_mask, expected_mask)
    assert masked == np.count_nonzero(got_mask)
    assert full == np.count_nonzero(got_mask.all(axis=1))
