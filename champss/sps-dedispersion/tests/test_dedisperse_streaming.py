from types import SimpleNamespace

import numpy as np
import pytest
from sps_dedispersion.dedisperse import dedisperse_streaming

dmtlib = pytest.importorskip("dmtlib")

FBOT, FTOP, TSAMP = 400.1953125, 800.1953125, 0.00098304
NCHAN, NTIME, MAX_DT = 64, 1500, 200


def make_skybeam(spectra):
    return SimpleNamespace(spectra=spectra, ra=10.0, dec=20.0, utc_start=1.7e9, obs_id="x")


def make_fdmt(chunk_size, dt_trials):
    return dmtlib.FDMT(
        FBOT, FTOP, NCHAN, chunk_size, TSAMP, dt_grid=dt_trials, use_box_smearing=False
    )


@pytest.fixture
def noise():
    # Highest frequency first, as in a SkyBeam
    return np.random.default_rng(0).normal(5, 1, (NCHAN, NTIME)).astype(np.float32)


def run(spectra, chunk_size, dt_trials):
    fdmt = make_fdmt(chunk_size, dt_trials)
    return dedisperse_streaming(fdmt, make_skybeam(spectra), chunk_size, MAX_DT, dt_trials)


@pytest.mark.parametrize("chunk_size", [64, 128, 700, NTIME])
def test_output_does_not_depend_on_chunking(noise, chunk_size):
    dt_trials = np.arange(0, MAX_DT, 2)
    whole = make_fdmt(NTIME, dt_trials).execute(noise[::-1])[:, MAX_DT:]
    dts = run(noise, chunk_size, dt_trials)
    assert dts.dedisp_ts.shape == (len(dt_trials), NTIME - MAX_DT)
    np.testing.assert_array_equal(dts.dedisp_ts, whole)


def test_chunk_longer_than_data_is_padded(noise):
    dt_trials = np.arange(0, MAX_DT, 2)
    expected = run(noise, NTIME, dt_trials).dedisp_ts
    np.testing.assert_array_equal(run(noise, 2 * NTIME, dt_trials).dedisp_ts, expected)


def test_sparse_grid_matches_rows_of_dense_grid(noise):
    dense = np.arange(0, MAX_DT)
    sparse = np.array([0, 1, 3, 4, 7, 20, 55, 120, 199])
    a = run(noise, 128, dense).dedisp_ts
    b = run(noise, 128, sparse)
    np.testing.assert_array_equal(b.dedisp_ts, a[sparse])
    assert b.dms[0] == 0
    assert len(b.dms) == len(sparse)


def test_dispersed_pulse_peaks_at_its_trial(noise):
    true_dt, t_bottom = 100, 900
    freqs = FBOT + (np.arange(NCHAN) + 0.5) * (FTOP - FBOT) / NCHAN
    delays = (FBOT**-2 - freqs**-2) / (FBOT**-2 - FTOP**-2) * true_dt
    spectra = np.zeros((NCHAN, NTIME), np.float32)
    # The bottom channel arrives last: t_bottom is its arrival time
    spectra[np.arange(NCHAN), np.rint(t_bottom - delays).astype(int)] = 1
    dt_trials = np.arange(0, MAX_DT)
    out = run(spectra[::-1], 128, dt_trials).dedisp_ts
    row, sample = np.unravel_index(out.argmax(), out.shape)
    assert abs(row - true_dt) <= 2
    assert abs(sample + MAX_DT - t_bottom) <= 2
    # Channels this wide smear over a few samples at this delay, so check a small window
    assert out[row, sample - 2 : sample + 3].sum() > 0.9 * NCHAN


def test_rejects_trials_beyond_the_dropped_samples(noise):
    dt_trials = np.arange(0, MAX_DT + 50)
    with pytest.raises(ValueError):
        run(noise, 128, dt_trials)


def test_shared_output_matches_and_is_unlinked_when_freed(noise):
    import gc
    import os

    dt_trials = np.arange(0, MAX_DT, 2)
    expected = run(noise, 128, dt_trials).dedisp_ts
    fdmt = make_fdmt(128, dt_trials)
    dts = dedisperse_streaming(
        fdmt, make_skybeam(noise), 128, MAX_DT, dt_trials, shared=True
    )
    name = dts.shared_memory.name.lstrip("/")
    np.testing.assert_array_equal(dts.dedisp_ts, expected)
    assert os.path.exists(f"/dev/shm/{name}")
    del dts
    gc.collect()
    assert not os.path.exists(f"/dev/shm/{name}")


def test_releasing_the_input_gives_the_same_output_and_frees_it():
    import os

    from sps_common.sm_utils import empty_shared_array

    # Long enough that each row spans many pages, so blocks free real memory
    nchan, ntime, chunk = NCHAN, 40000, 4096
    rng = np.random.default_rng(1)
    data = rng.normal(5, 1, (nchan, ntime)).astype(np.float32)
    dt_trials = np.arange(0, MAX_DT, 2)
    fdmt = dmtlib.FDMT(FBOT, FTOP, nchan, chunk, TSAMP, dt_grid=dt_trials, use_box_smearing=False)
    expected = dedisperse_streaming(fdmt, make_skybeam(data.copy()), chunk, MAX_DT, dt_trials)

    spectra, shm = empty_shared_array(data.shape, data.dtype)
    try:
        spectra[:] = data
        path = f"/dev/shm/{shm.name.lstrip('/')}"
        before = os.stat(path).st_blocks
        fdmt = dmtlib.FDMT(FBOT, FTOP, nchan, chunk, TSAMP, dt_grid=dt_trials, use_box_smearing=False)
        got = dedisperse_streaming(
            fdmt, make_skybeam(spectra), chunk, MAX_DT, dt_trials, release_input=True
        )
        np.testing.assert_array_equal(got.dedisp_ts, expected.dedisp_ts)
        # Almost all of the input's memory has been returned
        assert os.stat(path).st_blocks < 0.05 * before
    finally:
        del spectra
        shm.close()
        shm.unlink()
