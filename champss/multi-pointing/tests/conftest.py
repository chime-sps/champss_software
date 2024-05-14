import glob
import os
import pytest
from sps_multi_pointing.simulator import PointingGrid


@pytest.fixture(scope="session")
def simulator():
    p = PointingGrid(num_rows=5, num_cols=6)
    p.create()
    yield p

    # cleanup
    for pointing in glob.glob("*_sim_ps_candidates.npz"):
        os.remove(pointing)
    for ptg_pulsars in glob.glob("*_sim_pulsars.npy"):
        os.remove(ptg_pulsars)
    os.remove("injected_known_sources.npz")
