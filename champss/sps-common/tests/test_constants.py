#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sps_common import constants


def test_freq_constants():
    assert type(constants.FREQ_BOTTOM) == float
    assert type(constants.FREQ_TOP) == float
    assert type(constants.TSAMP) == float
