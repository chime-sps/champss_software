"""powersum_at_sigma_counts against powersum_at_sigma, which it must match exactly."""

import numpy as np
import pytest
from sps_common.interfaces.utilities import powersum_at_sigma, powersum_at_sigma_counts


def day_counts(ndays, harm, seed):
    """Summed bin weights as the search makes them: floats holding whole day counts."""
    rng = np.random.default_rng(seed)
    weights = np.full(200000, float(ndays))
    weights[rng.random(weights.size) < 0.05] -= rng.integers(1, ndays + 1, 1)
    weights[0] = 0
    bins = rng.integers(0, weights.size, (harm, weights.size))
    return weights[bins].sum(0)


@pytest.mark.parametrize("ndays", [1, 5, 30])
@pytest.mark.parametrize("harm", [1, 2, 8, 32])
@pytest.mark.parametrize("sigma", [3.0, 5.5, 8.0])
def test_matches_for_search_counts(ndays, harm, sigma):
    nsum = day_counts(ndays, harm, seed=ndays * harm)
    np.testing.assert_array_equal(
        powersum_at_sigma_counts(sigma, nsum), powersum_at_sigma(sigma, nsum)
    )


@pytest.mark.parametrize(
    "nsum",
    [
        np.array([0, 1, 7, 3]),  # integer dtype
        np.array([0.5, 2.0]),  # fractional
        np.array([-1.0, 2.0]),  # negative
        np.array([np.nan, 2.0]),  # not finite
        np.array([], dtype=float),
        np.array([[1.0, 2.0], [3.0, 4.0]]),  # two dimensional
    ],
)
def test_other_inputs_match(nsum):
    with np.errstate(invalid="ignore"):
        np.testing.assert_array_equal(
            powersum_at_sigma_counts(5.0, nsum), powersum_at_sigma(5.0, nsum)
        )


def test_array_sigma_matches():
    sigma, nsum = np.array([4.0, 6.0]), np.array([2.0, 3.0])
    np.testing.assert_array_equal(
        powersum_at_sigma_counts(sigma, nsum), powersum_at_sigma(sigma, nsum)
    )
