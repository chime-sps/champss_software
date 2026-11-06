"""chi2_ks_distances against compare_ps_to_chisqr_kstest, whose statistic it must match exactly."""

import numpy as np
import pytest
from ps_processes.utilities.qc import (
    chi2_ks_distances,
    compare_ps_to_chisqr_kstest,
    ks_statistic_sorted,
)
from ps_processes.utilities.utilities import analytical_chi2_cdf
from scipy.stats import kstest


def spectrum(n, ndays, kind, seed):
    """A power spectrum normalised to mean 1, as the pipeline makes, with masked zeros."""
    rng = np.random.default_rng(seed)
    x = rng.chisquare(2 * ndays, n) / 2 / ndays * ndays
    if kind == "excess":
        x[rng.random(n) < 0.002] *= 30  # strong outliers, as signals and RFI give
    elif kind == "quantised":
        x = np.round(x * 4) / 4  # many ties, as in the quantised stack
    elif kind == "scaled":
        x *= 1.03  # a slightly wrong normalisation, so the statistic is large
    x = x.astype(np.float32)
    x[rng.random(n) < 0.05] = 0
    return x


DAYS = list(np.linspace(0.975, 1.025, 11))


@pytest.mark.parametrize("n", [100, 5000, 500000])
@pytest.mark.parametrize("ndays", [1, 5])
@pytest.mark.parametrize("kind", ["noise", "excess", "quantised", "scaled"])
@pytest.mark.parametrize(
    "cuts",
    [
        {},
        {"lower_percentile_cut": 0.9},
        {"lower_percentile_cut": 0.01, "upper_percentile_cut": 0.99},
        {"upper_percentile_cut": 0.999, "dof_factor": 0.986},
    ],
)
def test_matches_compare_ps_to_chisqr_kstest(n, ndays, kind, cuts):
    ps = spectrum(n, ndays, kind, seed=n + ndays)
    days = [d * ndays for d in DAYS]
    fast = chi2_ks_distances(ps, days, red_noise_nbins=0, **cuts)
    slow = [
        compare_ps_to_chisqr_kstest(ps, ndays=d, red_noise_nbins=0, **cuts)[0]
        for d in days
    ]
    assert fast == slow


@pytest.mark.parametrize("block", [1, 7, 32, 1000])
def test_any_block_size_gives_kstest_statistic(block):
    x = np.sort(np.random.default_rng(3).chisquare(2.1, 20000))
    expected = kstest(x, analytical_chi2_cdf, args=(1.97, 0, np.inf)).statistic
    assert ks_statistic_sorted(x, analytical_chi2_cdf, (1.97, 0, np.inf), block) == expected
