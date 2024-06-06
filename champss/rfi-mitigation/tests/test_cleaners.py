#!/usr/bin/env python3

import numpy as np
import pytest
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
