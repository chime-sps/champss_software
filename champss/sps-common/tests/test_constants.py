#!/usr/bin/env python

from sps_common import constants


def test_freq_constants():
    assert type(constants.FREQ_BOTTOM) == float
    assert type(constants.FREQ_TOP) == float
    assert type(constants.TSAMP) == float
