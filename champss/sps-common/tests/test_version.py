#!/usr/bin/env python

import sps_common


def test_sps_version():
    assert type(sps_common.__version__) == str
