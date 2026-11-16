"""Daily power spectra as 16-bit codes (PowerSpectraCreation.quant_step) against float32."""

import numpy as np
import pytest
from ps_processes.processes.harmonic_search import prepare_fast_search, search_rows
from ps_processes.processes.ps import PowerSpectraCreation
from sps_common.interfaces.ps_processes import PowerSpectra
from sps_common.sm_utils import shared_memory
from sps_common.stack_quant import DAILY_CODE_MAX, encode_daily, round_power

STEP = 0.25
NTS, PADDED = 5000, 8192
NBIN = PADDED // 2


def test_encode_daily_rounds_as_the_stack_and_saturates():
    rng = np.random.default_rng(0)
    power = rng.exponential(1, 10000).astype(np.float32)
    power[[5, 17, 900]] = [1e5, 16383.8, 3e4]  # 16383.8: the largest code, 65535
    power[[1, 2]] = [-1, np.nan]
    codes = np.empty(power.size, np.uint16)
    over_bin = np.empty(power.size, np.int64)
    over_power = np.empty(power.size, np.float32)
    n = encode_daily(power, np.float32(1 / STEP), codes, over_bin, over_power)
    assert list(over_bin[:n]) == [5, 900] and list(over_power[:n]) == [1e5, 3e4]
    assert codes[5] == codes[900] == DAILY_CODE_MAX and codes[1] == codes[2] == 0
    ok = np.ones(power.size, bool)
    ok[[1, 2, 5, 900]] = False
    # Exactly the stack's rounding
    np.testing.assert_array_equal(codes[ok] * np.float32(STEP), round_power(power[ok], STEP))


def make_pspec(codes, overflow=None, float_rows=None):
    ndm, nbin = codes.shape
    return PowerSpectra(
        power_spectra=codes, dms=np.arange(ndm, dtype=float), freq_labels=np.arange(nbin) * 0.1,
        ra=1.0, dec=2.0, datetimes=[], num_days=1, beta=0.0, bad_freq_indices=[[]],
        obs_id=["x"], quant_step=STEP, overflow=overflow, float_rows=float_rows,
    )  # fmt: skip


def test_powers_and_decoded_use_overflow_and_float_rows():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 50, (4, 100)).astype(np.uint16)
    codes[2, 7] = DAILY_CODE_MAX
    overflow = dict(row=np.array([2]), bin=np.array([7]), power=np.float32([1e5]))
    exact0 = rng.exponential(1, 100).astype(np.float32)
    pspec = make_pspec(codes, overflow, {0: exact0})
    full = pspec.decoded()
    assert full.dtype == np.float32 and full[2, 7] == 1e5 and full[1, 3] == codes[1, 3] * STEP
    np.testing.assert_array_equal(pspec.powers(0), exact0)  # the exact row
    np.testing.assert_array_equal(pspec.powers(2), full[2])
    np.testing.assert_array_equal(pspec.powers(2, [7, 3]), full[2, [7, 3]])
    assert pspec.code_scale() == STEP and pspec.is_coded


def run_rows(ts, **kwargs):
    """transform_data on every row; returns the shared target array and the results."""
    coded = kwargs.get("quant_step") is not None
    dtype = np.uint16 if coded else np.float32
    shm = shared_memory.SharedMemory(create=True, size=ts.shape[0] * NBIN * 4)
    fshm = shared_memory.SharedMemory(create=True, size=2 * NBIN * 4)
    try:
        out = np.ndarray((ts.shape[0], NBIN), dtype, buffer=shm.buf)
        res = PowerSpectraCreation.transform_data(
            list(range(ts.shape[0])), list(ts), padded_length=PADDED,
            bad_freq_indices=[3, 40, 41], shared_target_name=shm.name,
            target_shape=out.shape, float_target_name=fshm.name if coded else None,
            **kwargs,
        )  # fmt: skip
        floats = np.ndarray((2, NBIN), np.float32, buffer=fshm.buf).copy()
        result = out.copy(), res, floats
        del out
        return result
    finally:
        for s in (shm, fshm):
            s.close()
            s.unlink()


def test_creation_codes_are_the_rounded_float_spectra():
    rng = np.random.default_rng(3)
    ts = (100 + rng.standard_normal((5, NTS))).astype(np.float32)
    ts[1] += np.float32(40) * np.sin(np.arange(NTS) * 0.3).astype(np.float32)  # a line
    # With red-noise removal (the pipeline's path) the line stays in range; without it
    # (one median for the whole spectrum) it saturates
    for kwargs in (dict(), dict(remove_rednoise=False)):
        check_codes(ts, kwargs, saturates=bool(kwargs))


def check_codes(ts, kwargs, saturates):
    floats, *_ = run_rows(ts, **kwargs)
    codes, res, held = run_rows(ts, quant_step=STEP, float_rows=[0, 3], **kwargs)
    rows, bins, powers = res[3]
    expected = np.rint(floats / STEP)
    sat = expected > DAILY_CODE_MAX
    assert sat.any() == saturates
    np.testing.assert_array_equal(codes[~sat], expected[~sat])
    assert (codes[sat] == DAILY_CODE_MAX).all()
    np.testing.assert_array_equal(np.sort(rows * NBIN + bins), np.flatnonzero(sat))
    np.testing.assert_array_equal(powers, floats[rows, bins])
    # Rows kept as float are the float spectra exactly
    np.testing.assert_array_equal(held, floats[[0, 3]])


@pytest.mark.parametrize("num_harm", [1, 8, 32])
def test_search_on_codes_matches_float_search_of_decoded_power(num_harm):
    from ps_processes.processes.harmonic_search import formula_bins
    from sps_common.interfaces.utilities import powersum_at_sigma

    rng = np.random.default_rng(4)
    nbin = 20000
    power = rng.exponential(1, (6, nbin)).astype(np.float32)
    power[2, 4000:4005] += 30
    codes = np.rint(power / STEP).astype(np.uint16)
    decoded = codes.astype(np.float32) * np.float32(STEP)
    harm_bins = formula_bins(num_harm, nbin)
    levels = int((np.array([1, 2, 4, 8, 16, 32]) <= num_harm).sum())
    nsum = np.array([1, 2, 4, 8, 16, 32])[:levels]
    cutoff = np.asarray(powersum_at_sigma(5.0, nsum), dtype=np.float64)
    fast_f = prepare_fast_search(harm_bins, num_harm, nsum, cutoff, 5.0, np.float32)
    fast_c = prepare_fast_search(harm_bins, num_harm, nsum, cutoff, 5.0, np.uint16)
    rows = np.arange(6)
    a = search_rows(decoded, rows, num_harm, fast_f)
    b = search_rows(codes, rows, num_harm, fast_c, step=STEP)
    assert len(a[0]) > 0
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_decoded_with_release_and_decoded_rows():
    from sps_common.sm_utils import empty_shared_array

    rng = np.random.default_rng(6)
    data = rng.integers(0, 400, (70, 4096)).astype(np.uint16)
    data[3, 9] = DAILY_CODE_MAX
    overflow = dict(row=np.array([3]), bin=np.array([9]), power=np.float32([2e5]))
    codes, shm = empty_shared_array(data.shape, np.uint16)
    try:
        codes[:] = data
        pspec = make_pspec(codes, overflow)
        rows = pspec.decoded_rows()
        assert len(rows) == 70 and rows.shape == data.shape
        np.testing.assert_array_equal(rows[3], pspec.powers(3))
        expected = data.astype(np.float32) * np.float32(STEP)
        expected[3, 9] = 2e5
        np.testing.assert_array_equal(pspec.decoded(release=True, block_rows=16), expected)
        # The codes' memory has been handed back (they read as zero)
        assert not codes.any()
    finally:
        del codes, pspec
        shm.close()
        shm.unlink()
