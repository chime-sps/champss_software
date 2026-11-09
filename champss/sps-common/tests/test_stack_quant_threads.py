"""Threaded zstd (de)compression of quantized stacks gives the same files and values as
the HDF5 filters."""

import h5py
import numpy as np
import pytest
from sps_common import stack_quant
from sps_common.stack_quant import QuantizedStack, read_quantized, round_power, write_quantized

pytest.importorskip("zstandard")

STEP = 0.25


def stack(days, shape, seed=1, bright=True):
    rng = np.random.default_rng(seed)
    spectra = sum(round_power(rng.exponential(1, shape), STEP) for _ in range(days))
    spectra = spectra.astype(np.float32)
    if bright:
        # Values far from the noise level, which go to the overflow list
        spectra[::7, ::97] += np.float32(5000)
        spectra[min(3, shape[0] - 1), :50] = 0
    return spectra


def write(path, spectra, days, threads):
    with h5py.File(path, "w") as f:
        write_quantized(f, spectra, STEP, days, threads=threads)


def chunks(path):
    with h5py.File(path) as f:
        d = f["power spectra"]
        return [d.id.read_direct_chunk((i, 0)) for i in range(d.shape[0])]


@pytest.mark.parametrize("nrow", [1, 7, 8, 9, 33, 70])
@pytest.mark.parametrize("threads", [1, 3, 8])
def test_same_file_as_filters(tmp_path, nrow, threads):
    days = 12
    spectra = stack(days, (nrow, 4099))
    write(tmp_path / "f.h5", spectra, days, threads=0)
    write(tmp_path / "t.h5", spectra, days, threads=threads)
    assert chunks(tmp_path / "f.h5") == chunks(tmp_path / "t.h5")
    with h5py.File(tmp_path / "f.h5") as a, h5py.File(tmp_path / "t.h5") as b:
        assert set(a) == set(b)
        for k in a:
            np.testing.assert_array_equal(a[k][()], b[k][()])
        assert dict(a.attrs) == dict(b.attrs)
    # (The files' own layout differs: HDF5 places directly written chunks differently)


def test_noncontiguous_and_float64_input(tmp_path):
    days = 3
    base = stack(days, (20, 3001)).astype(np.float64)
    spectra = base[::2, 1:]  # strided rows, float64
    write(tmp_path / "f.h5", spectra, days, threads=0)
    write(tmp_path / "t.h5", spectra, days, threads=4)
    assert chunks(tmp_path / "f.h5") == chunks(tmp_path / "t.h5")


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("span", [(0, None), (5, 41), (31, 33), (69, 70)])
def test_rows_match_filters(tmp_path, threads, raw, span):
    days = 30
    spectra = stack(days, (70, 2049))
    write(tmp_path / "s.h5", spectra, days, threads=0)
    with h5py.File(tmp_path / "s.h5") as f:
        ref = read_quantized(f, *span, raw=raw, threads=0)
        new = read_quantized(f, *span, raw=raw, threads=threads)
    np.testing.assert_array_equal(new.view(np.uint32), ref.view(np.uint32))


@pytest.mark.parametrize("threads", [1, 4])
def test_add_rounded_matches_filters(tmp_path, threads):
    days = 30
    spectra = stack(days, (70, 2049))
    write(tmp_path / "s.h5", spectra, days, threads=0)
    day = np.random.default_rng(5).exponential(1, (70, 2049)).astype(np.float32)
    day[10, :20] += 9000
    a, b = day.copy(), day.copy()
    with h5py.File(tmp_path / "s.h5") as f:
        QuantizedStack(f, threads=0).add_rounded(a, STEP)
        QuantizedStack(f, threads=threads).add_rounded(b, STEP)
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


def test_too_large_raises(tmp_path):
    spectra = stack(1, (20, 100), bright=False)
    spectra[13, 5] = 1e12
    with pytest.raises(ValueError, match="row 13"):
        write(tmp_path / "t.h5", spectra, 1, threads=4)


def test_other_layouts_use_filters(tmp_path):
    """A stack whose chunks are not shuffled zstd rows is read through the filters."""
    days = 2
    spectra = stack(days, (10, 500))
    with h5py.File(tmp_path / "s.h5", "w") as f:
        write_quantized(f, spectra, STEP, days, threads=0)
        data = f["power spectra"][()]
        del f["power spectra"]
        f.create_dataset("power spectra", data=data, chunks=(2, 500), compression="gzip")
    with h5py.File(tmp_path / "s.h5") as f:
        q = QuantizedStack(f, threads=4)
        assert q.threads == 0
        np.testing.assert_array_equal(q.rows(raw=True), spectra)


def test_without_zstandard(tmp_path, monkeypatch):
    monkeypatch.setattr(stack_quant, "zstandard", None)
    days = 4
    spectra = stack(days, (9, 300))
    write(tmp_path / "t.h5", spectra, days, threads=8)
    with h5py.File(tmp_path / "t.h5") as f:
        q = QuantizedStack(f)
        assert q.threads == 0
        np.testing.assert_array_equal(q.rows(raw=True), spectra)
