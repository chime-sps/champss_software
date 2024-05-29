#! /usr/bin/env python

from random import random

import numpy as np
from beamformer.strategist import mapper


def test_pointing_mapper():
    beams = np.asarray([0, 32])
    pointing_map = mapper.PointingMapper(beams=beams)
    pointing = pointing_map.get_pointing_map()
    pointing_map.plot_dm_nchan_map(pointing, save=False)
    random_idx = int(random() * len(pointing))
    assert type(pointing) == list
    assert type(pointing[random_idx]) == dict
