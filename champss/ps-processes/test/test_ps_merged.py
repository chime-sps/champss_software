"""The one-pass square/gather/NaN kernel and the red-noise birdie zeroing give exactly
the rows of the numpy steps they replace."""

import numpy as np
import pytest
from ps_processes.processes import ps
from ps_processes.processes.ps import PowerSpectraCreation, _RowTransformer
from sps_common.sm_utils import shared_memory

NTS, PADDED = 5000, 8192
NBIN = PADDED // 2


def run_rows(ts, bad, **kwargs):
    """transform_data on every row of ts into a fresh shared array; returns rows, medians."""
    shm = shared_memory.SharedMemory(create=True, size=ts.shape[0] * NBIN * 4)
    try:
        out = np.ndarray((ts.shape[0], NBIN), np.float32, buffer=shm.buf)
        out[:] = -7  # nothing may be left unwritten
        _, medians, scales = PowerSpectraCreation.transform_data(
            list(range(ts.shape[0])),
            list(ts),
            padded_length=PADDED,
            bad_freq_indices=bad,
            shared_target_name=shm.name,
            target_shape=out.shape,
            **kwargs,
        )
        return out.copy(), medians, scales
    finally:
        shm.close()
        shm.unlink()


def unsorted(bad):
    """The same birdies in an order that makes transform_data use the numpy steps."""
    bad = list(bad)
    return bad[::-1] + bad[:1] if bad else bad


def same(a, b):
    np.testing.assert_array_equal(np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32))


@pytest.fixture
def ts():
    rng = np.random.default_rng(3)
    t = (100 + rng.standard_normal((6, NTS))).astype(np.float32)
    t[1] += np.float32(5) * np.sin(np.arange(NTS) * 0.3).astype(np.float32)  # a strong line
    t[2] = 0  # all-zero row
    t[3] *= np.float32(1e-20)  # tiny values
    t[4] *= np.float32(1e14)  # large values (finite squares: an overflow to inf with an
    # all-masked red-noise window makes rednoise_normalise loop forever)
    return t


BADS = {
    "none": [],
    "random": sorted(np.random.default_rng(1).choice(NBIN, 300, replace=False).tolist()),
    "ends": [0, 1, 2, 100, NBIN - 2, NBIN - 1],
    "block": list(range(50, 400)),
}


@pytest.mark.parametrize("beta", [0.0, -6.3e-5, 1e-4, -2e-3])
@pytest.mark.parametrize("bad", list(BADS))
@pytest.mark.parametrize(
    "opts",
    [
        {},
        {"zero_replace": True, "rednoise_config": {"b0": 50, "bmax": 100000}},
        {"normalise": True},
        {"remove_rednoise": False},
        {"barycentring_mode": "Topocentric"},
    ],
    ids=["default", "b0_50", "normalise", "no_rednoise", "topocentric"],
)
def test_same_as_numpy_steps(ts, beta, bad, opts):
    kwargs = dict(beta=beta, **opts)
    kwargs.setdefault("rednoise_config", {"b0": 10, "bmax": 100000})
    if opts.get("normalise"):
        # Normalising an all-zero row gives an all-NaN spectrum, on which
        # rednoise_normalise loops forever
        ts[2] = ts[0][::-1]
    new = run_rows(ts, BADS[bad], **kwargs)
    old = run_rows(ts, unsorted(BADS[bad]), **kwargs)
    same(new[0], old[0])
    assert len(new[1]) == len(old[1])
    for a, b in zip(new[1], old[1]):
        same(a, b)
    assert new[2] == old[2]


def test_fallback_window_all_masked(ts):
    """A red-noise window with every bin masked makes the kernel give up mid-row."""
    bad = list(range(1, 200))
    kw = dict(beta=-6.3e-5, rednoise_config={"b0": 10, "bmax": 100000})
    new = run_rows(ts, bad, **kw)
    old = run_rows(ts, unsorted(bad), **kw)
    same(new[0], old[0])


def test_kernel_matches_steps():
    rng = np.random.default_rng(0)
    for beta in (-6.3e-5, 3e-4):
        t = _RowTransformer(PADDED, "<f4", 32, beta, "Fourier")
        assert t.offsets is not None
        dts = rng.standard_normal(NTS).astype(np.float32)
        bad = np.unique(rng.integers(0, NBIN, 200)).astype(np.int64)
        expected = t.power_spectrum(dts, False).copy()
        expected[bad] = np.nan
        same(t.power_spectrum_masked(dts, False, bad), expected)
        # An invalid list goes the numpy way and still matches
        same(t.power_spectrum_masked(dts, False, bad[::-1].copy()), expected)


def test_valid_bins():
    v = ps.valid_bins
    assert v(np.array([], np.int64), 5)
    assert v(np.array([0, 4], np.int64), 5)
    assert not v(np.array([0, 5], np.int64), 5)
    assert not v(np.array([-1, 2], np.int64), 5)
    assert not v(np.array([2, 2], np.int64), 5)
    assert not v(np.array([3, 2], np.int64), 5)
    assert not v(np.array([1, 2], np.int32), 5)
    assert not v([1, 2], 5)
