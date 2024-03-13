import glob
import os
from sps_common.interfaces import MultiPointingCandidate


def read_multi_pointing_candidates(filepath):
    """
    Read a set of multi pointing candidates file from a directory and returns a list of MultiPointingCandidate class.

    Parameters
    ----------
    filepath: str
        Path to where all the multi pointing candidates .npz files are located.

    Returns
    -------
    mp_cands: list(MultiPointingCandidate)
        A list of multi pointing candidates as a MultiPointingCandidate class object.
    """
    files = glob.glob(os.path.join(filepath, "*.npz"))
    mp_cands = []
    for f in files:
        mp_cands.append(MultiPointingCandidate.read(f))
    return mp_cands
