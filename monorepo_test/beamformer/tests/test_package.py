#! /usr/bin/env python

import beamformer


def test_version():
    version = beamformer.__version__
    assert type(version) == str
