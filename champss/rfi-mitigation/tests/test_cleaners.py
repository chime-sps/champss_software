#!/usr/bin/env python3

import numpy as np
from rfi_mitigation.cleaners.cleaners import DummyCleaner
from sps_common.constants import TSAMP

###########################################
#  SETUP MOCK DATA SET WITH KNOWN INPUTS  #
###########################################

np.random.seed(2020)
# chunk size (1 second)
NTIME = 1024
# average number of expected frequency channels
NFREQ = 2048
# random noise with non-zero baseline
TEST_DATA = np.random.normal(size=(NFREQ, NTIME)).astype(np.float32) + 100
t = np.linspace(0, NTIME * TSAMP, NTIME)

# add power-law distributed noise to different channels
TEST_DATA[516, :] += 10 * np.random.pareto(1.4, size=NTIME)
TEST_DATA[1567, :] += 5 * np.random.pareto(1.2, size=NTIME)

# add strong periodic signals and a handful of harmonics
for harm in range(1, 5):
    # nominal mains power signal
    TEST_DATA[:, :] += 0.5 * np.sin(2 * np.pi * (harm * 60.0) * t)
    # cellular network resynchronisation signal
    TEST_DATA[:, :] += 0.7 * np.sin(2 * np.pi * (harm * 3.125) * t)

# add some random periodic signals
TEST_DATA[351, :] += np.sin(2 * np.pi * 255.60 * t)
TEST_DATA[128, :] += np.sin(2 * np.pi * 476.78 * t)

# simulate certain channel drops as in L0 packet misses/GPU node failures
channels_dropped = [
    10,
    11,
    12,
    13,
    600,
    605,
    987,
    1345,
    1347,
    1780,
    1781,
    1782,
    1783,
    2000,
    2003,
]
TEST_DATA[channels_dropped, :] = 0


##################################################
#  TEST THAT CLEANERS ACTUALLY DETECT SOMETHING  #
#  (since there's quite a bit of junk added...)  #
##################################################
def test_dummy_cleaner():
    # by definition, the input should be identical to the output
    c = DummyCleaner(TEST_DATA)
    c.clean()
    print(c.summary())

    # mask should all be zeroes (i.e. not masked)
    np.testing.assert_equal(c.get_mask(), np.zeros_like(TEST_DATA))
    np.testing.assert_equal(c.get_masked_fraction(), 0)


def _shared_spectra(data):
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    spectra = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    spectra[:] = data
    return shm, spectra


def _norm_power(series, k):
    power = np.abs(np.fft.rfft(series - series.mean())) ** 2
    return power[k] / (np.median(power[1:]) / np.log(2))


def test_fourier_zap_cleaner():
    from rfi_mitigation.cleaners.cleaners import FourierZapCleaner

    nchan, ntime = 32, 2**15
    rng = np.random.default_rng(1)
    data = rng.normal(size=(nchan, ntime)).astype(np.float32) + 10
    t = np.arange(ntime)
    # broadband periodic signal, below the channel threshold but strong at zero DM
    k_broad = 3001
    data += 0.05 * np.sin(2 * np.pi * k_broad * t / ntime).astype(np.float32)
    # narrowband periodic signal in a single channel
    k_narrow, narrow_chan = 7000, 5
    data[narrow_chan] += 0.2 * np.sin(2 * np.pi * k_narrow * t / ntime)
    # weak broadband signal below both thresholds, which has to be kept
    k_weak = 11000
    data += 0.008 * np.sin(2 * np.pi * k_weak * t / ntime).astype(np.float32)
    # strong signal below min_freq, which has to be kept
    k_low = 2
    data += 0.5 * np.sin(2 * np.pi * k_low * t / ntime).astype(np.float32)
    # a completely masked channel
    data[20] = 0

    results = []
    for num_threads in [1, 2]:
        shm, spectra = _shared_spectra(data)
        try:
            cleaner = FourierZapCleaner(data.shape, chan_block=8)
            cleaner.clean(shm.name, data.shape, data.dtype, num_threads=num_threads)
            results.append(spectra.copy())
        finally:
            shm.close()
            shm.unlink()
    cleaned = results[0]
    assert np.array_equal(results[0], results[1])
    assert cleaner.cleaned and cleaner.nzapped > 0
    assert cleaner.zero_dm_zapped_freqs.size > 0

    zero_dm_before, zero_dm_after = data.sum(0), cleaned.sum(0)
    assert _norm_power(zero_dm_before, k_broad) > 100
    assert _norm_power(zero_dm_after, k_broad) < 15
    assert _norm_power(data[narrow_chan], k_narrow) > 100
    assert _norm_power(cleaned[narrow_chan], k_narrow) < 15
    # the narrowband signal is only replaced in its own channel
    other = [c for c in range(nchan) if c not in (narrow_chan, 20)]
    before = np.abs(np.fft.rfft(data[other], axis=1)[:, k_narrow])
    after = np.abs(np.fft.rfft(cleaned[other], axis=1)[:, k_narrow])
    assert np.allclose(before, after, rtol=1e-3, atol=1e-2)
    # weak and low frequency signals are kept
    weak_ratio = _norm_power(zero_dm_after, k_weak) / _norm_power(
        zero_dm_before, k_weak
    )
    assert 0.9 < weak_ratio < 1.1
    low_ratio = _norm_power(zero_dm_after, k_low) / _norm_power(zero_dm_before, k_low)
    assert 0.9 < low_ratio < 1.1
    # channel levels and noise are preserved, the masked channel is untouched
    assert np.allclose(cleaned.mean(1), data.mean(1), rtol=1e-5, atol=1e-5)
    assert np.allclose(cleaned[other].std(1), data[other].std(1), rtol=0.01)
    assert np.array_equal(cleaned[20], data[20])
