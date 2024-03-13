import numpy as np
import numpy.lib.recfunctions as rfn
import glob, os
from sps_common.interfaces import (
    SearchAlgorithm,
    SinglePointingCandidate,
    SinglePointingCandidateCollection,
    MultiPointingCandidate,
    CandidateClassification,
    CandidateClassificationLabel,
    KnownSourceLabel,
)
from typing import List


def make_single_pointing_candidate_collection(
    ra: float,
    dec: float,
    group_summary,
    groups,
) -> SinglePointingCandidateCollection:
    """Slimmed-down clone of `candidate_processor.harmonic_filter.make_single_pointing_candidate_collection`"""

    spcs: List[SinglePointingCandidate] = []
    for cluster_id, cluster in group_summary.items():
        main_cluster = groups[cluster_id]
        freq_arr, dm_arr = get_diagnostic_from_cluster(
            main_cluster, cluster["f"], cluster["dm"]
        )
        # not implemented attributes, these allow the tests not to fail
        raw_harmonic_powers_array_dummy = {
            "powers": np.zeros((5, 32, 1)),
            "dms": np.arange(5),
            "freqs": np.tile(np.arange(32), (5, 1))[:, :, np.newaxis],
            "freq_bins": np.tile(np.arange(32), (5, 1))[:, :, np.newaxis],
        }
        harmonics_info_dummy = np.array([1.0, 0.5, 2])
        spc = SinglePointingCandidate(
            freq=cluster["f"],
            freq_arr=freq_arr,
            dm=cluster["dm"],
            dm_arr=dm_arr,
            dc=cluster["dc"],
            sigma=cluster["sigma"],
            ra=ra,
            dec=dec,
            features=np.array((), dtype=[]),
            rfi=cluster["rfi"],
            obs_id=["1"],
            detection_statistic=SearchAlgorithm.power_spec,
            harmonics_info=harmonics_info_dummy,
            raw_harmonic_powers_array=raw_harmonic_powers_array_dummy,
        )
        spcs.append(spc)
    return SinglePointingCandidateCollection(
        candidates=spcs,
    )


def get_diagnostic_from_cluster(cluster, freq, dm):
    """
    Extract the frequency vs sigma array at the input DM, and DM vs sigma array at the input frequency

    Clone of `candidate_processor.harmonic_filter.get_diagnostic_from_cluster`
    """
    cluster = rfn.structured_to_unstructured(cluster)
    freq_arr = cluster[np.where(np.isclose(cluster[:, 1], dm))][:, [0, -1]]
    dm_arr = cluster[np.where(np.isclose(cluster[:, 0], freq))][:, [1, -1]]
    return freq_arr, dm_arr


def relabel_simulated_candidates(
    candidates_path,
    sim_pulsars_path,
    freq_diff=3.5 * 9.70127682e-04,
    dm_diff=1.0,
    delete_old_candidates=True,
):
    """
    Function to read MultiPointingCandidates from simulated data and relabel them based on matches to simulated
    pulsars in the dataset and save them in separate file with 'relabelled' in its file name.

    Parameters
    ----------
    candidates_path: str
        Path to the MultiPointingCandidates .npz files

    sim_pulsars_path: str
        Path to the simulated pulsars' .npy files

    freq_diff: float
        The max difference in frequency that a source is consider a match to a pulsar. Default = 3.5 * 9.70127682e-04.

    dm_diff: float
        The max difference in DM that a source is consider a match to a pulsar. Default = 1.0.

    delete_old_candidates: bool
        Whether to delete the original MultiPointingCandidates .npz files. Default = True.
    """
    candidate_files = glob.glob(
        os.path.join(candidates_path, "Multi_Pointing_Groups_f_*.npz")
    )
    candidates = []
    for cand_file in candidate_files:
        candidates.append(MultiPointingCandidate.read(cand_file))
    psr_list = []
    psr_files = glob.glob(os.path.join(sim_pulsars_path, "*_sim_pulsars.npy"))
    for psr_file in psr_files:
        psr = np.load(psr_file, allow_pickle=True)
        psr_list.extend(psr.tolist())
    for i, cand in enumerate(candidates):
        label = 0
        if cand.known_source.label == KnownSourceLabel.Known:
            print("This candidate is labelled as known")
            label = 1
        else:
            for psr in psr_list:
                if (
                    np.abs(cand.best_freq - psr["freq"]) < freq_diff
                    and np.abs(cand.best_dm - psr["dm"]) < dm_diff
                ):
                    print("This is a pulsar")
                    label = 1
                    break
        if label == 0:
            cand.classification = CandidateClassification(
                label=CandidateClassificationLabel.RFI, grade=0.0
            )
        if delete_old_candidates:
            os.remove(candidate_files[i])
        cand.write(os.path.join(candidates_path, "Multi_Pointing_Groups_relabelled"))
