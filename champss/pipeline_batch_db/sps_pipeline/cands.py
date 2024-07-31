"""Executes the pipeline component for single-pointing candidate processing."""
import logging
import os
from os import path

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
    """
    date = utils.transit_time(pointing).date()
    log.info(
        f"Candidate Processor ({pointing.ra :.2f} {pointing.dec :.2f}) @"
        f" { date :%Y-%m-%d}"
    )

    file_path = path.join(
        basepath,
        date.strftime("%Y/%m/%d"),
        f"{ pointing.ra :.02f}_{ pointing.dec :.02f}",
    )
    if not psdc:
        ps_detection_clusters = (
            f"{ file_path }/{ pointing.ra :.02f}_{ pointing.dec :.02f}_{pointing.sub_pointing}_power_spectra_detection_clusters.hdf5"
        )
        psdc = PowerSpectraDetectionClusters.read(ps_detection_clusters)
    ps_candidates = (
        f"{file_path}/{pointing.ra :.02f}_{pointing.dec :.02f}_{pointing.sub_pointing}_power_spectra_candidates.npz"
    )
    with cand_ps_processing_time.labels(pointing.pointing_id, pointing.beam_row).time():
        spcc = cands_processor.fg.make_single_pointing_candidate_collection(
            psdc, power_spectra
        )
        spcc.write(ps_candidates)
        payload = {"path_candidate_file": path.abspath(ps_candidates)}
        db_api.update_observation(pointing.obs_id, payload)
        if plot:
            log.info("Creating candidates plots.")
            plot_folder = f"{ file_path }/plots/"
            candidate_plots = spcc.plot_candidates(
                sigma_threshold=plot_threshold, folder=plot_folder
            )
            log.info(f"Plotted {len(candidate_plots)} candidate plots.")
    log.info(f"{len(spcc.candidates)} candidates.")


def run_interface(
    ps_detection_clusters,
    pointing,
    stack_path,
    cand_path,
    cands_processor,
    power_spectra=None,
    plot=False,
    plot_threshold=0,
    write_hrc=False,
):
    """
    Search a `pointing` for periodicity candidates. Candidates will be written out in a
    candidates.npz file. Used for the cumulative stack.

    Parameters
    ----------
    ps_detection_clusters: sps_common.interfaces.PowerSpectraDetectionClusters
        The PowerSpectraDetectionClusters interface containing all clusters of detections from a cumulative stack.
    pointing: sps_common.interfaces.beamformer.ActivePointing
        Sky pointing to process.
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
    """
    log.info(f"Candidate Processor of stack ({pointing.ra :.2f} {pointing.dec :.2f})")
    stack_root_folder = stack_path.rsplit("/stack/")[0]
    if cand_path is None:
        candidate_folder = f"{stack_root_folder}/candidates"
    else:
        candidate_folder = f"{cand_path}/candidates"
    if "cumulative" in stack_path.rsplit("/stack/")[1]:
        candidate_folder += "_cumul/"
    else:
        candidate_folder += "_monthly/"
    os.makedirs(candidate_folder, exist_ok=True)
    base_name = path.basename(stack_path)
    ps_candidates = (
        f"{candidate_folder}{base_name.rstrip('.hdf5')}_"
        f"{power_spectra.datetimes[-1].strftime('%Y%m%d')}"
        f"_{len(power_spectra.datetimes)}_candidates.npz"
    )

    with cand_ps_processing_time.labels(pointing._id, pointing.beam_row).time():
        psdc = ps_detection_clusters
        spcc = cands_processor.fg.make_single_pointing_candidate_collection(
            psdc, power_spectra
        )
        spcc.write(ps_candidates)
        payload = {
            "path_candidate_file": path.abspath(ps_candidates),
            "num_total_candidates": len(spcc.candidates),
        }
        db_api.append_ps_stack(pointing._id, payload)
        if plot:
            if cand_path is None:
                plot_folder = f"{stack_root_folder}/plots"
            else:
                plot_folder = f"{cand_path}/plots"
            if "cumulative" in stack_path.rsplit("/stack/")[1]:
                plot_folder += "_cumul/"
            else:
                plot_folder += "_monthly/"
            log.info(f"Creating candidates plots in {plot_folder}.")
            candidate_plots = spcc.plot_candidates(
                sigma_threshold=plot_threshold, folder=plot_folder
            )
            log.info(f"Plotted {len(candidate_plots)} candidate plots.")
    log.info(f"{len(spcc.candidates)} candidates.")


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
