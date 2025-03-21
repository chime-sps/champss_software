"""Utility function for processing."""

import logging

from sps_common.constants import MIN_SEARCH_FREQ
from sps_common.interfaces.multi_pointing import KnownSourceLabel

log = logging.getLogger(__name__)


def process_mp_candidate(
    cand_classifier,
    kss,
    csv,
    plot_cands,
    plot_threshold,
    plot_dm_threshold,
    plot_all_pulsars,
    out_folder,
    cand,
):
    """
    Process an mp Candidate.

    This performs:
        - classification
        - known source identification
        - writing to disk
        - plotting candidate
        - exctracting information for summary csv

    Parameters
    ----------
    cand_classifier: classifier.CandidateClassifier
        Used classifier
    kss: KnownSourceSifter
        Used known source sifter
    csv: bool
        Extract csv info
    plot_cands: bool
        Perform plotting
    plot_threshold: float
        Plotting sigma threshold
    plot_dm_threshold: float
        Plotting DM threshold
    plot_all_pulsars: bool
        Plot known pulsar
    out_folder: str
        Output folder
    cand: MultiPointingCandidate
        The candidate to be processed

    Returns
    -------
    cand: MultiPointingCandidate
        The updated candidate
    cand_dict: dict or None
        The dictionary containing infromation for the summary csv
    """
    # Run Classifier
    cand = cand_classifier.classify(cand)

    # Run KSS
    if kss is not None:
        cand = kss.classify(cand, pos_filter=True)
    if cand.known_source.label == KnownSourceLabel.Known:
        log.info(
            "Candidate: %s -- (%.2f, %.2f) f=%.3f, DM=%.1f, sigma=%.3f",
            ", ".join(cand.known_source.matches["source_name"]),
            cand.ra,
            cand.dec,
            cand.best_freq,
            cand.best_dm,
            cand.best_sigma,
        )
    # Save the candidate to file
    # `Multi_Pointing_Groups_f_<<F>>_DM_<<DM>>_class_<<Astro|RFI|Ambiguous>>.npz`
    mp_cand_file_base = f"{out_folder}/candidates/Multi_Pointing_Groups"
    mp_cand_file_name = cand.write(mp_cand_file_base)
    plot_path = ""
    if plot_cands:
        if plot_all_pulsars and cand.known_source.label.value:
            plot_path = cand.plot_candidate(path=f"{out_folder}/plots/")
        else:
            if cand.best_freq > (MIN_SEARCH_FREQ * 2):
                if cand.best_sigma > plot_threshold:
                    if cand.best_dm >= plot_dm_threshold:
                        plot_path = cand.plot_candidate(path=f"{out_folder}/plots/")
    # if plot_path != "":
    #    plot_path = os.path.join(os.getcwd(), plot_path)

    if csv:
        cand_dict = {
            "mean_freq": cand.mean_freq,
            "mean_dm": cand.mean_dm,
            "sigma": cand.best_sigma,
            "ra": cand.ra,
            "dec": cand.dec,
            "best_ra": cand.summary["ra"],
            "best_dec": cand.summary["dec"],
            "ncands": len(cand.all_dms),
            "std_ra": cand.position_features["std_ra"],
            "std_dec": cand.position_features["std_dec"],
            "delta_ra": cand.position_features["delta_ra"],
            "delta_dec": cand.position_features["delta_dec"],
            "file_name": mp_cand_file_name,
            "plot_path": plot_path,
            "known_source_label": cand.known_source.label.value,
        }
        if cand.known_source.label.value:
            # Extracting only the first entry of the known sources
            cand_ks_dict = {
                "known_source_likelihood": cand.known_source.matches["likelihood"][0],
                "known_source_name": cand.known_source.matches["source_name"][0],
                "known_source_p0": cand.known_source.matches["spin_period_s"][0],
                "known_source_dm": cand.known_source.matches["dm"][0],
                "known_source_ra": cand.known_source.matches["pos_ra_deg"][0],
                "known_source_dec": cand.known_source.matches["pos_dec_deg"][0],
            }
            cand_dict = {**cand_dict, **cand_ks_dict}
            # previously this used an operation which only worked after Python 3.9
            # cand_dict = cand_dict | cand_ks_dict

        return cand, cand_dict
    else:
        return cand, None
