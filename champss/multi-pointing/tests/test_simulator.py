"""
Run tests on the simulator
"""

import glob
import os


def test_simulate_output(simulator):
    """Checks that the expected output files of the simulation are present"""
    assert len(glob.glob("*_sim_ps_candidates.npz")) == 30
    assert len(glob.glob("*_sim_pulsars.npy")) == 30
    assert os.path.exists("injected_known_sources.npz")
