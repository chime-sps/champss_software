"""rednoise_normalise_into against rednoise_normalise, which it must match exactly."""

import numpy as np
import pytest
from ps_processes.utilities.rednoise_fast import rednoise_normalise_into
from ps_processes.utilities.utilities import rednoise_normalise, rednoise_scale


def spectrum(length, kind, masked, seed):
    rng = np.random.default_rng(seed)
    x = rng.exponential(1, length) * (1 + 50 / (1 + np.arange(length) / 200))
    x = x.astype(np.float32)
    if kind == "ties":
        x = np.round(x * 4) / 4
    elif kind == "sorted":
        x = np.sort(x)[::-1].copy()
    elif kind == "step":
        x[length // 2 :] *= 1000
    x[rng.random(length) < masked] = np.nan
    return x


def fast(x, b0, bmax):
    scale = np.asarray(rednoise_scale(len(x), b0, bmax), dtype=np.int64)
    out = np.full(len(x), -1, dtype=np.float32)
    buf = np.empty(2 * int(scale.max()), dtype=np.float32)
    before = x.copy()
    medians = rednoise_normalise_into(x, out, scale, buf)
    np.testing.assert_array_equal(x, before)
    return out, medians


@pytest.mark.parametrize(
    "length, b0, bmax",
    [(524287, 10, 100000), (524287, 50, 100000), (100001, 10, 1000), (4095, 10, 100000),
     (30, 10, 100000), (1000, 3, 50), (300000, 10, 100000)],
)
@pytest.mark.parametrize("kind", ["noise", "ties", "sorted", "step"])
@pytest.mark.parametrize("masked", [0, 0.05, 0.4])
def test_matches_rednoise_normalise(length, b0, bmax, kind, masked):
    x = spectrum(length, kind, masked, seed=length + b0)
    expected, medians, _ = rednoise_normalise(x.copy(), b0=b0, bmax=bmax)
    out, fast_medians = fast(x, b0, bmax)
    assert fast_medians is not None
    np.testing.assert_array_equal(out, expected.astype(np.float32))
    np.testing.assert_array_equal(fast_medians, np.asarray(medians, dtype=np.float32))


def test_all_masked_window_asks_for_the_fallback():
    x = spectrum(4095, "noise", 0, seed=1)
    x[10:23] = np.nan  # the whole second window
    out, medians = fast(x, 10, 100000)
    assert medians is None
