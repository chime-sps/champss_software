#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sps_common


def test_sps_version():
    assert type(sps_common.__version__) == str
