"""The per-row power spectrum steps against the operations they replace."""

from types import SimpleNamespace

import numpy as np
import pyfftw
import pytest
from ps_processes.processes.ps import (
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
