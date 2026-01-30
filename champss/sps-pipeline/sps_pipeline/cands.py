"""Executes the pipeline component for single-pointing candidate processing."""

import logging
import os
from os import path

import numpy as np
from candidate_processor.feature_generator import Features
from omegaconf import OmegaConf
from prometheus_client import Summary
from sps_common.interfaces import PowerSpectraDetectionClusters
from sps_databases import db_api
from sps_pipeline import utils

log = logging.getLogger(__package__)

min_group_size = 5
threshold = 5.5

cand_hhat_processing_time = Summary(
    "cand_hhat_pointing_processing_seconds",
    "Duration of processing Hhat for a pointing",
    ("pointing", "beam_row"),
)
cand_ps_processing_time = Summary(
    "cand_ps_pointing_processing_seconds",
    "Duration of processing power spectra for a pointing",
    ("pointing", "beam_row"),
)


def run(
    pointing,
    cands_processor,
    psdc=None,
    power_spectra=None,
    plot=False,
    plot_threshold=0,
    basepath="./",
    write_hrc=False,
    injection_path=None,
    injection_idx=[],
    only_injections=False,
):
    """
    Search a `pointing` for periodicity candidates. Used for daily stacks.

    Parameters
    ----------
    pointing: sps_common.interfaces.beamformer.ActivePointing
        Sky pointing to process.
    cands_processor: Wrapper
        Wrapper object consisting of the config used and the candidates processing pipeline generated from an
        OmegaConf instance via cands.initialise.
    psdc: sps_common.interfaces.PowerSpectraDetectionClusters
        The PowerSpectraDetectionClusters interface containing all clusters of detections from the power spectra stack.
    power_spectra: sps_common.interfaces.ps_rocesses.PowerSpectra
        The power spectra of the poinitng. Needed to fully create all candidate properties.
    plot: bool
        Whether to create candidate plots. Default: False
    plot_threshold: float
        Sigma threshold for created candidate plots. Default: 0
    basepath: str
        Folder which is used to store data. Default: "./"
    write_hrc: bool
        Whether to write the harmonically related clusters. Default: False
    injection_path: str
        Path to injection file or string describing default injection type
    injection_idx: list
        Indices of injection file entries that are injected
    only_injections: bool
        Whether non-injections are filtered out. Default: False
    """
    date = utils.transit_time(pointing).date()
    log.info(
        f"Candidate Processor ({pointing.ra:.2f} {pointing.dec:.2f}) @ {date:%Y-%m-%d}"
    )
    if only_injections:
        file_path = path.join(
            basepath,
            "injections",
            date.strftime("%Y/%m/%d"),
        )
        os.makedirs(
            file_path,
            exist_ok=True,
        )
    else:
        file_path = path.join(
            basepath,
            date.strftime("%Y/%m/%d"),
            f"{pointing.ra:.02f}_{pointing.dec:.02f}",
        )
    if not psdc:
        ps_detection_clusters = f"{file_path}/{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}_power_spectra_detection_clusters.hdf5"
        psdc = PowerSpectraDetectionClusters.read(ps_detection_clusters)
    with cand_ps_processing_time.labels(pointing.pointing_id, pointing.beam_row).time():
        spcc = cands_processor.fg.make_single_pointing_candidate_collection(
            psdc, power_spectra
        )
        if only_injections:
            period_string = "_".join(
                [f"{inj['frequency']:.2f}" for inj in spcc.injections]
            )
            ps_candidates = (
                f"{file_path}/candidates/{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}_"
                f"{injection_path.split('/')[-1]}_{str(injection_idx).replace(' ', '')}_{period_string}_injection_candidates.npz"
            )
        else:
            ps_candidates = f"{file_path}/{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}_power_spectra_candidates.npz"
        spcc.write(ps_candidates)
        if injection_path:
            injection_performance = spcc.test_injection_performance()
        payload = {"path_candidate_file": path.abspath(ps_candidates)}
        db_api.update_observation(pointing.obs_id, payload)
        if plot:
            log.info("Creating candidates plots.")
            if only_injections:
                plot_folder = f"{file_path}/plots_injections/"
            else:
                plot_folder = f"{file_path}/plots/"
            candidate_plots = spcc.plot_candidates(
                sigma_threshold=plot_threshold, folder=plot_folder
            )
            log.info(f"Plotted {len(candidate_plots)} candidate plots.")
    log.info(f"{len(spcc.candidates)} candidates.")
    if len(spcc.candidates):
        # Currently candidates are not sorted
        all_sigmas = spcc.return_attribute_array("sigma")
        best_cand_pos = np.argmax(all_sigmas)
        log.info(
            f"Best candidate: F0: {spcc.candidates[best_cand_pos].freq:.3f} DM:"
            f" {spcc.candidates[best_cand_pos].dm:.2f} Sigma:"
            f" {spcc.candidates[best_cand_pos].sigma:.2f}"
        )


def run_interface(
    ps_detection_clusters,
    stack_path,
    cand_path,
    cands_processor,
    power_spectra=None,
    plot=False,
    plot_threshold=0,
    write_hrc=False,
    update_db=False,
    injection_path=None,
    injection_idx=[],
    only_injections=False,
    closest_pointing_id=None,
):
    """
    Search a `pointing` for periodicity candidates. Candidates will be written out in a
    candidates.npz file. Used for the cumulative stack.

    Parameters
    ----------
    ps_detection_clusters: sps_common.interfaces.PowerSpectraDetectionClusters
        The PowerSpectraDetectionClusters interface containing all clusters of detections from a cumulative stack.
    stack_path: str
        Path to the location of the cumulative power spectra stack.
    cands_processor: Wrapper
        Wrapper object consisting of the config used and the candidates processing pipeline generated from an
        OmegaConf instance via cands.initialise.
    power_spectra: sps_common.interfaces.ps_rocesses.PowerSpectra
        The power spectra of the pointing. Needed to fully create all candidate properties.
    plot: bool
        Whether to create candidate plots. Default: False
    plot_threshold: float
        Sigma threshold for created candidate plots. Default: 0
    write_hrc: bool
        Whether to write the harmonically related clusters. Default: False
    update_db: bool
        Whether to update the update the db with the candiate file. Default: False
    injection_path: str
        Path to injection file or string describing default injection type
    injection_idx: list
        Indices of injection file entries that are injected
    only_injections: bool
        Whether non-injections are filtered out. Default: False
    """

    log.info(
        "Candidate Processor of stack"
        f" ({ps_detection_clusters.ra:.2f} {ps_detection_clusters.dec:.2f})"
    )
    if "/stack/" in stack_path:
        stack_root_folder = stack_path.rsplit("/stack/")[0]
    else:
        stack_root_folder = path.abspath(stack_path).rsplit("/", 1)[0]
    if not cand_path:
        if only_injections:
            candidate_folder = f"{stack_root_folder}/injections/"
        else:
            candidate_folder = f"{stack_root_folder}/candidates"
    else:
        if only_injections:
            candidate_folder = f"{cand_path}/injections/"
        else:
            candidate_folder = f"{cand_path}/candidates"
    if not only_injections:
        if "cumulative" in stack_path.rsplit("/", 1)[-1]:
            candidate_folder += "_cumul/"
        else:
            candidate_folder += "_monthly/"
    os.makedirs(candidate_folder, exist_ok=True)
    base_name = path.basename(stack_path)
    if only_injections:
        # Brackets in file name may cause ls to show the file name in quotes
        ps_candidates = (
            f"{candidate_folder}{base_name.rstrip('.hdf5')}_"
            f"{power_spectra.datetimes[-1].strftime('%Y%m%d')}"
            f"_{len(power_spectra.datetimes)}_"
            f"{injection_path.split('/')[-1]}_{str(injection_idx).replace(' ', '')}_candidates.npz"
        )
    else:
        ps_candidates = (
            f"{candidate_folder}{base_name.rstrip('.hdf5')}_"
            f"{power_spectra.datetimes[-1].strftime('%Y%m%d')}"
            f"_{len(power_spectra.datetimes)}_candidates.npz"
        )

    psdc = ps_detection_clusters
    spcc = cands_processor.fg.make_single_pointing_candidate_collection(
        psdc, power_spectra
    )
    spcc.write(ps_candidates)
    if injection_path:
        injection_performance = spcc.test_injection_performance()
    if update_db and closest_pointing_id:
        # For now don't write to db by default, I'll think later about how to turn this on and off
        # Turned it on again
        payload = {
            "path_candidate_file": path.abspath(ps_candidates),
            "num_total_candidates": len(spcc.candidates),
        }
        db_api.append_ps_stack(closest_pointing_id, payload)
    if plot:
        if not cand_path:
            plot_folder = f"{stack_root_folder}/plots"
        else:
            plot_folder = f"{cand_path}/plots"
        if only_injections:
            plot_folder += "_injections/"
        else:
            if "cumulative" in stack_path.rsplit("/stack/", 1)[-1]:
                plot_folder += "_cumul/"
            else:
                plot_folder += "_monthly/"
        log.info(f"Creating candidates plots in {plot_folder}.")
        candidate_plots = spcc.plot_candidates(
            sigma_threshold=plot_threshold, folder=plot_folder
        )
        log.info(f"Plotted {len(candidate_plots)} candidate plots.")

    log.info(f"{len(spcc.candidates)} candidates.")
    if len(spcc.candidates):
        # Currently candidates are not sorted
        all_sigmas = spcc.return_attribute_array("sigma")
        best_cand_pos = np.argmax(all_sigmas)
        log.info(
            f"Best candidate: F0: {spcc.candidates[best_cand_pos].freq:.3f} DM:"
            f" {spcc.candidates[best_cand_pos].dm:.2f} Sigma:"
            f" {spcc.candidates[best_cand_pos].sigma:.2f}"
        )


def initialise(configuration, num_threads=4):
    class Wrapper:
        def __init__(self, config):
            self.config = config
            self.fg = Features.from_config(
                OmegaConf.to_container(self.config.cands.features),
                OmegaConf.to_container(self.config.cands.arrays),
                num_threads=num_threads,
            )

    return Wrapper(configuration)
