"""Data reading functions."""

import glob
import logging
import os

from easydict import EasyDict
from sps_common.interfaces import MultiPointingCandidate
from sps_common.interfaces.single_pointing import SinglePointingCandidateCollection

log = logging.getLogger(__name__)


def read_multi_pointing_candidates(filepath):
    """
    Read multiple multi pointing candidates.

    Read a set of multi pointing candidates file from a directory and returns a list of
    MultiPointingCandidate class.

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


def read_cands_summaries(file, sigma_threshold=0):
    """
    Read candidate file and return summaries.

    Read a single pointing candidate file and return the summaries with some added
    flags.

    Parameters
    ----------
    file: str
        Path to the SinglePointingCandidateCollection

    Returns
    -------
    all_cands: list(EasyDict)
        The summaries of all candidates
    """
    all_cands = []
    try:
        spcc = SinglePointingCandidateCollection.read(file, verbose=False)

        # datetimes may not be included in the candidates already
        datetimes = spcc.candidates[0].datetimes
        """
        This is commented out for now because reading the database from a
        multiprocessing call seems to be troublesome.

        if not datetimes:
            try:
                datetimes = db_api.get_dates(spcc.candidates[0].obs_id)
            except Exception as e:
                log.error(f"Could not grab datetimes. May not have access to db. {e}")
                datetimes = []
        """
        for cand_index, candidate in enumerate(spcc.candidates):
            if candidate.sigma < sigma_threshold:
                continue
            cand_summary = EasyDict(candidate.summary)
            cand_summary["file_name"] = file
            cand_summary["cand_index"] = cand_index
            cand_summary["features"] = candidate.features
            cand_summary["datetimes"] = datetimes
            cand_summary["nharm"] = candidate.nharm
            cand_summary["best_harmonic_sum"] = candidate.best_harmonic_sum

            all_cands.append(cand_summary)
        return all_cands
    except Exception as e:
        log.error(f"Can't process file {file} because of {e}.")
