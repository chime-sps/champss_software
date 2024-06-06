#! /usr/bin/env python


from random import random

import numpy as np
from beamformer.utilities import dm

dm_map = dm.DMMap()


def test_ne_2001():
    dm = dm_map.get_dm_ne2001(random() * 90, random() * 360)
    assert type(dm) == np.ndarray
    assert dm > 0


def test_ymw16():
    dm = dm_map.get_dm_ymw16(random() * 90, random() * 360)
    assert type(dm) == np.ndarray
    assert dm > 0
