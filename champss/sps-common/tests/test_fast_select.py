"""masked_median against np.ma.median, which it must match to the bit."""

import numpy as np
import pytest
from sps_common.fast_select import masked_median


def reference(x, mask):
    return np.ma.median(np.ma.masked_array(x.reshape(1, -1), mask=mask.reshape(1, -1)))


@pytest.mark.parametrize("n", [1, 2, 3, 10, 4095, 4096, 4097, 50000, 143360])
@pytest.mark.parametrize("kind", ["noise", "ties", "sorted", "nan_under_mask"])
@pytest.mark.parametrize("masked", [0, 0.1, 0.6, 0.99])
def test_matches_numpy(n, kind, masked):
    rng = np.random.default_rng(n + int(100 * masked))
    x = (rng.exponential(1, n) * 300).astype(np.float32)
    if kind == "ties":
        x = (np.round(x / 8) * 8).astype(np.float32)
    elif kind == "sorted":
        x = np.sort(x)
    mask = rng.random(n) < masked
    if kind == "nan_under_mask":
        x[mask] = np.nan
    got = masked_median(x, mask)
    if not (~mask).any():
        assert got is None
        return
    assert got is not None
    assert got.dtype == np.float32
    assert np.float32(reference(x, mask)).tobytes() == got.tobytes()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_unmasked_non_finite_values_ask_for_numpy(bad):
    x = np.arange(1, 101, dtype=np.float32)
    mask = np.zeros(100, bool)
    x[7] = bad
    assert masked_median(x, mask) is None


def test_zero_median_asks_for_numpy():
    x = np.array([-1, 0, 0, 1], np.float32)
    assert masked_median(x, np.zeros(4, bool)) is None
