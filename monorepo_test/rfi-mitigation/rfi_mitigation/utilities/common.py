#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    from sps_common import constants

    SAMPLE_TIME_IN_SECONDS = constants.SAMPLE_TIME_IN_SECONDS
    NATIVE_NTIME_PER_MSGPACK = constants.NATIVE_NTIME_PER_MSGPACK
    NATIVE_L0_NCHAN = constants.NATIVE_L0_NCHAN
    NATIVE_L1_NCHAN = constants.NATIVE_L1_NCHAN
    FREQ_TOP_MHZ = constants.FREQ_TOP_MHZ
    FREQ_BOTTOM_MHZ = constants.FREQ_BOTTOM_MHZ
    no_sps_common = False
except ImportError:
    print("sps-common package is not installed, using default constants")
    no_sps_common = True

if no_sps_common:
    SAMPLE_TIME_IN_SECONDS = 0.00098304
    NATIVE_NTIME_PER_MSGPACK = 1024
    NATIVE_L0_NCHAN = 1024
    NATIVE_L1_NCHAN = 16384
    FREQ_TOP_MHZ = 800.1953125
    FREQ_BOTTOM_MHZ = 400.1953125
