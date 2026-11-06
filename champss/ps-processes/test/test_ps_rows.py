"""The per-row power spectrum steps against the operations they replace."""

from types import SimpleNamespace

import numpy as np
import pyfftw
import pytest
from ps_processes.processes.ps import (
    PowerSpectraCreation,
    _RowTransformer,
    fourier_barycentre_index,
    share_time_series,
)
from scipy.interpolate import interp1d
from sps_common.sm_utils import empty_shared_array

N = 2**14


def reference(dts, normalise, beta, barycentring_mode, nbit=32):
    """The steps as transform_data did them before the per-process plan."""
    n_to_pad = N - dts.size
    if normalise:
        dts = np.pad(dts, (0, n_to_pad), mode="constant", constant_values=0)
    else:
        dts = np.pad(dts, (0, n_to_pad), mode="median", stat_length=4096)
    spectrum = pyfftw.interfaces.numpy_fft.rfft(dts)
    spectrum[0] = 0
    spectrum = spectrum[:-1]
    power = (np.abs(spectrum) ** 2).astype(f"float{nbit}")
    if barycentring_mode == "Fourier":
        target = np.arange(len(power))
        power = interp1d(
            target * (1 - beta), power, kind="nearest", bounds_error=False, fill_value=0
        )(target)
    return power


@pytest.mark.parametrize("beta", [-1e-4, 0.0, 3e-4])
@pytest.mark.parametrize("normalise", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_row_transformer_is_identical(beta, normalise, dtype):
    rng = np.random.default_rng(0)
    transformer = _RowTransformer(N, np.dtype(dtype).str, 32, beta, "Fourier")
    for size in (N - 3000, N - 1, N):
        dts = (50 + rng.standard_normal(size)).astype(dtype)
        expected = reference(dts, normalise, beta, "Fourier")
        got = transformer.power_spectrum(dts, normalise)
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


def test_topocentric_and_nbit64():
    dts = np.random.default_rng(1).standard_normal(N - 100).astype(np.float32)
    transformer = _RowTransformer(N, "<f4", 64, 1e-4, "Topocentric")
    expected = reference(dts, False, 1e-4, "Topocentric", nbit=64)
    np.testing.assert_array_equal(transformer.power_spectrum(dts, False), expected)


@pytest.mark.parametrize("beta", [-2e-4, 2e-4])
def test_barycentre_index_marks_bins_outside_the_spectrum(beta):
    index = fourier_barycentre_index(N // 2, beta)
    assert (index < 0).any() == (beta > 0)
    assert index.max() < N // 2


def test_time_series_already_in_shared_memory_is_not_copied():
    ts, shm = empty_shared_array((3, 10), np.float32)
    try:
        dts = SimpleNamespace(dedisp_ts=ts, shared_memory=shm)
        new, info = share_time_series(dts)
        assert new is None and info["name"] == shm.name
        new, info = share_time_series(SimpleNamespace(dedisp_ts=ts[:, ::2]))
        assert new is not None
        copy = np.ndarray(info["shape"], dtype=info["dtype"], buffer=new.buf)
        np.testing.assert_array_equal(copy, ts[:, ::2])
        del copy
        new.close()
        new.unlink()
    finally:
        del ts
        shm.close()
        shm.unlink()


def run_transform_rows(ts_data, release, chunk=7, padded=2**14):
    """transform_rows over all rows in chunks, as the pool would, from a shared time series."""
    import os
    from multiprocessing import shared_memory

    ts, ts_shm = empty_shared_array(ts_data.shape, ts_data.dtype)
    ts[:] = ts_data
    nrow = ts_data.shape[0]
    target = shared_memory.SharedMemory(create=True, size=nrow * padded // 2 * 4)
    try:
        path = f"/dev/shm/{ts_shm.name.lstrip('/')}"
        before = os.stat(path).st_blocks
        for start in range(0, nrow, chunk):
            PowerSpectraCreation.transform_rows(
                start,
                min(start + chunk, nrow),
                dict(name=ts_shm.name, shape=ts.shape, dtype=ts.dtype.str),
                release_input=release,
                normalise=True,
                barycentring_mode="Fourier",
                beta=-6.3e-5,
                padded_length=padded,
                nbit=32,
                remove_rednoise=True,
                clean_rfi=True,
                rednoise_config={"b0": 10, "bmax": 1000},
                zero_replace=True,
                bad_freq_indices=[5, 6, 7, 400],
                shared_target_name=target.name,
                target_shape=(nrow, padded // 2),
            )
        after = os.stat(path).st_blocks  # before anything reads the freed rows
        spectra = np.ndarray((nrow, padded // 2), np.float32, buffer=target.buf).copy()
        return spectra, before, after
    finally:
        del ts
        ts_shm.close()
        ts_shm.unlink()
        target.close()
        target.unlink()


def test_releasing_the_time_series_gives_the_same_spectra_and_frees_it():
    rng = np.random.default_rng(5)
    # Rows of 12000 samples, not a whole number of pages
    ts_data = (100 + rng.normal(0, 1, (40, 12000))).astype(np.float32)
    kept, before, after_kept = run_transform_rows(ts_data, release=False)
    released, _, after_released = run_transform_rows(ts_data, release=True)
    np.testing.assert_array_equal(released, kept)
    assert after_kept == before
    assert after_released < 0.2 * before
