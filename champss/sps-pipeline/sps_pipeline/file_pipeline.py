import your
import numpy as np
from omegaconf import OmegaConf
from astropy.time import Time
import multiprocessing
import click
import logging
import os
from astropy.coordinates import EarthLocation
import astropy.units as u


log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from sps_pipeline.config_utils import load_config
from sps_common.constants import DM_CONSTANT
from sps_dedispersion.dedisperse import dedisperse
from sps_common.interfaces.beamformer import SkyBeam
from dmt.libdmt import FDMT
from ps_processes.processes.ps import PowerSpectraCreation
from ps_processes.processes.ps_search import PowerSpectraSearch
from candidate_processor.feature_generator import Features
from ps_processes.processes.ps_stack import PowerSpectraStack


def apply_logging_config(config, log_file="./logs/default.log"):
    """
    Applies logging settings from the given configuration.

    Logging settings are under the 'logging' key, and include:
    - format: string for the `logging.formatter`
    - level: logging level for the root logger
    - modules: a dictionary of submodule names and logging level to be
               applied to that submodule's logger
    """
    log_stream.setFormatter(
        logging.Formatter(fmt=config.logging.format, datefmt="%b %d %H:%M:%S")
    )

    if config.logging.get("file_logging", False):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(fmt=config.logging.format, datefmt="%b %d %H:%M:%S")
        )
        logging.root.addHandler(file_handler)
    logging.root.setLevel(config.logging.level.upper())
    log.debug("Set default level to: %s", config.logging.level)

    if "modules" in config.logging:
        for module_name, level in config.logging["modules"].items():
            logging.getLogger(module_name).setLevel(level.upper())
            log.debug("Set %s level to: %s", module_name, level)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--obs-file",
    type=str,
    help="File containing the observation. Needs to be readable using your.",
    required=True,
)
@click.option(
    "--stack-file", type=str, help="File where power spectra are saved.", required=True
)
@click.option(
    "--config-file",
    default="sps_config.yml",
    type=str,
    help="Name of used config. Default: sps_config.yml",
)
@click.option(
    "--config-options",
    default="{}",
    type=str,
    help=(
        "Additional options that overwrite the config options. Provide a string that would define a python dictionary"
        """For example use --config-options '{"beamform": {"max_mask_frac": 0.1}}' """
    ),
)
@click.option(
    "--num-threads",
    default=4,
    type=int,
    help=("Number of multiprocessing threads to use."),
)
@click.option(
    "--maxdm",
    default=200.0,
    type=float,
    help="Maximum DM. Config value will be ignored.",
)
def run_file_pipeline(
    obs_file, stack_file, config_file, config_options, num_threads, maxdm
):
    config = load_config(config_file, config_options)
    apply_logging_config(config)
    multiprocessing.set_start_method("forkserver", force=True)

    observation = your.Your(obs_file)
    obs_header = observation.your_header
    log.info(obs_header)

    telescope_location = EarthLocation(
        x=float(observation.header["ANT_X"]) * u.m,
        y=float(observation.header["ANT_Y"]) * u.m,
        z=float(observation.header["ANT_Z"]) * u.m,
    )
    obs_length = obs_header.nspectra
    obs_spectrum = observation.get_data(0, obs_length).T

    # maxdm = config.dedisp.maxdm
    FREQ_BOTTOM = obs_header.fch1
    FREQ_TOP = obs_header.fch1 + obs_header.bw
    if FREQ_BOTTOM > FREQ_TOP:
        FREQ_BOTTOM, FREQ_TOP = FREQ_TOP, FREQ_BOTTOM
        obs_spectrum = obs_spectrum[::-1, :]
    log.info(f"Loaded data with shape {obs_spectrum.shape}.")
    TSAMP = obs_header.tsamp
    fdmt_config = config.dedisp.fdmt
    num_dms_fac = fdmt_config.num_dms_fac

    num_dms = (
        int(
            DM_CONSTANT
            * maxdm
            * (1 / FREQ_BOTTOM**2 - 1 / FREQ_TOP**2)
            / TSAMP
            // num_dms_fac
        )
        + 1
    ) * num_dms_fac

    # restrict num_dms to number of channels for now
    # num_dms = min(num_dms, obs_spectrum.shape[0])
    nsamps = fdmt_config.chunk_size + num_dms
    dm_step = fdmt_config.dm_step
    fdmt = FDMT(
        obs_header.fch1,
        obs_header.fch1 + obs_header.bw,
        obs_spectrum.shape[0],
        nsamps,
        obs_header.tsamp,
        dt_max=num_dms,
        dt_min=0,
        dt_step=dm_step,
    )
    fdmt.set_num_threads(num_threads)

    skybeam = SkyBeam(
        spectra=obs_spectrum,
        ra=obs_header.ra_deg,
        dec=obs_header.dec_deg,
        nchan=obs_spectrum.shape[0],
        ntime=obs_spectrum.shape[1],
        maxdm=maxdm,
        beam_row=0,
        utc_start=Time(obs_header.tstart, format="mjd").unix,
        nbits=obs_spectrum.dtype.itemsize * 8,
    )
    dts = dedisperse(
        fdmt,
        skybeam,
        fdmt_config.chunk_size,
        num_dms,
        dm_step,
        True,
        freq_bottom=FREQ_BOTTOM,
        freq_top=FREQ_TOP,
        tsamp=TSAMP,
    )

    log.info(f"Created {len(dts.dms)} DM trials up to a DM of {max(dts.dms)}")

    padded_length = int(
        np.power(2, np.ceil(np.log(dts.dedisp_ts.shape[1]) / np.log(2)))
    )
    spectra_nbit = config.ps_creation.nbit
    psc_pipeline = PowerSpectraCreation(
        num_threads=num_threads,
        use_db=False,
        update_db=False,
        tsamp=obs_header.tsamp,
        find_common_birdies=False,
        padded_length=padded_length,
        clean_rfi=False,
        run_static_filter=False,
        run_dynamic_filter=False,
        nbit=spectra_nbit,
        telescope_location=telescope_location,
    )

    power_spectra = psc_pipeline.transform(dts)

    pss_pipeline = PowerSpectraSearch(
        cluster_config=dict(config.ps.ps_search_config.cluster_config),
        padded_length=padded_length,
        num_threads=num_threads,
        update_db=False,
    )

    (
        power_spectra_detection_clusters,
        power_spectra_detections,
    ) = pss_pipeline.search(
        power_spectra,
        cutoff_frequency=100,
    )
    fg = Features.from_config(
        OmegaConf.to_container(config.cands.features),
        OmegaConf.to_container(config.cands.arrays),
        num_threads=num_threads,
    )

    spcc = fg.make_single_pointing_candidate_collection(
        power_spectra_detection_clusters, power_spectra
    )

    plotted_cands = spcc.plot_candidates(
        folder="./single_observation_plots/", sigma_threshold=0, dm_threshold=-1
    )
    log.info(f"Plotted {len(plotted_cands)} candidates.")
    cand_folder = "./single_observation_candidates/"
    os.makedirs(cand_folder, exist_ok=True)
    cand_path = (
        cand_folder
        + os.path.splitext(os.path.basename(obs_file))[0]
        + "_candidates.npz"
    )
    spcc.write(cand_path)
    log.info(f"Wrote candidates to {cand_path}.")

    stack_nbit = config.ps.ps_stack_config.stack_nbit

    stack_exists = os.path.isfile(stack_file)
    if not stack_exists:
        log.info(f"Stack does not exist. Writing new stack to {stack_file}.")
        power_spectra.write(stack_file, nbit=stack_nbit)
    else:
        stack_config = OmegaConf.to_container(config.ps.ps_stack_config)
        stack_config["qc"] = False
        stack_config["qc_config"] = {}
        ps_stack_pipeline = PowerSpectraStack(
            mode="month",
            update_db=False,
            spectra_nbit=stack_nbit,
            **stack_config,
        )
        power_spectra_stacked = ps_stack_pipeline.stack_power_spectra_readout(
            power_spectra, stack_file
        )
        ps_stack_pipeline.replace_spectra(
            power_spectra_stacked, os.path.abspath(stack_file)
        )

        (
            power_spectra_detection_clusters_stacked,
            power_spectra_detections_stacked,
        ) = pss_pipeline.search(
            power_spectra_stacked,
            cutoff_frequency=100,
        )

        spcc_stacked = fg.make_single_pointing_candidate_collection(
            power_spectra_detection_clusters_stacked, power_spectra_stacked
        )

        spcc_stacked.plot_candidates(
            folder="./stack_plots/", sigma_threshold=0, dm_threshold=-1
        )

        plotted_cands = spcc.plot_candidates(sigma_threshold=0, dm_threshold=-1)
        log.info(f"Plotted {len(plotted_cands)} candidates.")
        cand_folder = "./stack_candidates/"
        os.makedirs(cand_folder, exist_ok=True)
        cand_path = (
            cand_folder
            + os.path.splitext(os.path.basename(stack_file))[0]
            + "_candidates.npz"
        )
        spcc.write(cand_path)
        log.info(f"Wrote candidates to {cand_path}.")
