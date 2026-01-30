"""Runner script for the Slow Pulsar Search prototype pipeline v0."""

__version__ = "2021.4a0"

import datetime as dt
import gc
import logging
import multiprocessing
import os
import sys
import time
from contextlib import nullcontext
from glob import glob
import ast

import click
import numpy as np
import pyroscope
import pytz
from omegaconf import OmegaConf
from prometheus_api_client import PrometheusConnect
from pymongo.errors import ServerSelectionTimeoutError

# Careful disabling HDF5 file locking: can lead to stack corruption
# if two processes write to the same file concurrently (as
# is the case with same pointings, different dates)
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# set these up before importing any SPS packages
log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from beamformer import NoSuchPointingError
from beamformer.strategist.strategist import PointingStrategist
from beamformer.utilities.common import find_closest_pointing
from ps_processes.processes.ps import PowerSpectraCreation
from ps_processes.ps_pipeline import PowerSpectraPipeline
from scheduler.utils import convert_date_to_datetime
from sps_common.interfaces import DedispersedTimeSeries
from sps_databases import db_api, db_utils, models
from sps_pipeline import (  # ps,
    beamform,
    cands,
    cleanup,
    ffa,
    ps_cumul_stack,
    utils,
)

default_datpath = "/data/chime/sps/raw"


def load_config(config_file="sps_config.yml", cli_config_string="{}"):
    """
    Combines default/user-specified config settings and applies them to loggers.

    User-specified settings are given as a YAML file in the current directory.

    The format of the file is (all sections optional):
    ```
    logging:
      format: string for the `logging.formatter`
      level: logging level for the root logger
      modules:
        module_name: logging level for the submodule `module_name`
        module_name2: etc.
    ```
    Paramaters
    ----------
    config_files: str
        Name of config file. Default: "sps_config.yml"

    Returns
    -------
    The `omegaconf` configuration object merging all the default configuration
    with the (optional) user-specified overrides.
    """
    base_config_path = os.path.dirname(__file__) + "/" + config_file
    if os.path.isfile(base_config_path):
        base_config = OmegaConf.load(base_config_path)
    else:
        log.info(
            "Base config does not exist. Will only use config from execution folder."
        )
        base_config = OmegaConf.create()
    if os.path.exists("./" + config_file):
        user_config = OmegaConf.load("./" + config_file)
    else:
        user_config = OmegaConf.create()
    cli_config = ast.literal_eval(cli_config_string)
    return OmegaConf.merge(base_config, user_config, cli_config)


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
        # Remove possibly pre-existing file handler
        handlers_to_be_removed = []
        for log_handler in log.root.handlers:
            if isinstance(log_handler, logging.FileHandler):
                handlers_to_be_removed.append(log_handler)
        for handler in handlers_to_be_removed:
            log.root.removeHandler(handler)
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


def dbexcepthook(type, value, tb):
    """
    Exception hook that logs uncaught exception to processes db.

    Based on: https://stackoverflow.com/a/20829384
    """
    # Also print default output
    sys.__excepthook__(type, value, tb)

    import traceback

    log.error("Exception hook activated. Will try to log error to process db.")
    tbtext = "".join(traceback.format_exception(type, value, tb))
    pipeline_end_time = time.time()
    if "active_process" in globals():
        try:
            db_api.update_process(
                active_process.id,
                {
                    "status": 3,
                    "error_message": tbtext,
                    "process_time": pipeline_end_time - pipeline_start_time,
                },
            )
        except Exception as e:
            log.error(
                f"Could not update process with id {active_process.id}, error {e}"
            )
    try:
        power_spectra.unlink_shared_memory()
        log.info("Unlinked shared memory.")
    except Exception:
        log.debug("No shared memory detected.")
    log.info(
        "Pipeline execution time:"
        f" {((pipeline_end_time - pipeline_start_time) / 60):.2f} minutes"
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=False,
    help="Date of data to process. Default = Today in UTC",
)
@click.option(
    "--stack/--no-stack",
    "-s",
    default=False,
    help="Whether to stack the power spectra",
)
@click.option(
    "--fdmt/--no-fdmt",
    default=True,
    help="Whether to use fdmt for dedispersion",
)
@click.option(
    "--plot/--no-plot",
    default=False,
    help="Whether to create candidate plots",
)
@click.option(
    "--plot-threshold",
    default=0.0,
    type=float,
    help="Sigma threshold above which the candidate plots are created",
)
@click.argument("ra", type=click.FloatRange(-180, 360))
@click.argument("dec", type=click.FloatRange(-90, 90))
@click.argument(
    "components",
    type=click.Choice(
        ["all", "rfi", "beamform", "dedisp", "ps", "ffa", "search", "cleanup"]
    ),
    nargs=-1,
)
@click.option(
    "--num-threads",
    default=None,
    type=int,
    help=(
        "Number of multiprocessing threads to use. If no value is given, "
        "the calculated config's value will be used."
    ),
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver1",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--basepath",
    default="./",
    type=str,
    help="Path for created files. Default './'",
)
@click.option(
    "--stackpath",
    default=None,
    type=str,
    help=(
        "Path for the cumulative stack. As default the basepath from the config is"
        " used."
    ),
)
@click.option(
    "--using-pyroscope/--not-using-pyroscope",
    default=False,
    help="Whether to profile this function using Pyroscope or not",
)
@click.option(
    "--using-docker/--not-using-docker",
    default=False,
    help="Whether this run is being used with Workflow + Docker Swarm",
)
@click.option(
    "--known-source-threshold",
    "--kst",
    default=np.inf,
    type=float,
    help=(
        "Threshold under which known sources are filtered based on the previously"
        " strongest detection of that source in that pointing."
    ),
)
@click.option(
    "--config-file",
    default="sps_config.yml",
    type=str,
    help="Name of used config. Default: sps_config.yml",
)
@click.option(
    "--injection-path",
    default=None,
    type=str,
    help="Path to yml file containing injection",
)
@click.option(
    "--injection-idx",
    "--ii",
    default=[],
    type=int,
    multiple=True,
    help="Index of pulse to inject from yml file",
)
@click.option(
    "--only-injections/--no-only-injections",
    default=False,
    help="Only process clusters containing injections.",
)
@click.option(
    "--scale-injections/--not-scale-injections",
    default=False,
    help="Scale injection so that input sigma should be detected sigma.",
)
@click.option(
    "--datpath",
    default=default_datpath,
    type=str,
    help="Path to the raw data folder.",
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
def main(
    date,
    stack,
    fdmt,
    plot,
    plot_threshold,
    ra,
    dec,
    components,
    num_threads,
    db_port,
    db_host,
    db_name,
    basepath,
    stackpath,
    using_pyroscope,
    using_docker,
    known_source_threshold,
    config_file,
    injection_path,
    injection_idx,
    only_injections,
    scale_injections,
    datpath,
    config_options,
):
    """
    Runner script for the Slow Pulsar Search prototype pipeline v0.

    The normal way to run the script to process a pointing for a
    given day is run-pipeline --date YYYYMMDD <RA> <DEC>

    Subcommands:
    - all: run all components (default)

    - beamform: run only the beamformer

    - dedisp: run only the dedisperser

    - ps: run only power spectrum computation

    - search: run only the search

    - cleanup: run only the cleanup operation

    You can also run a combination of several processes together.
    E.g. If you want to produce filterbank files only from scratch,
    you can do run-pipeline --date 20200701 317.86 20.96 rfi beamform
    """
    log.info("START")
    # Logging in multiprocessing child processes with Linux default
    # "fork" leads to unexpected behaviour
    multiprocessing.set_start_method("forkserver", force=True)

    date = convert_date_to_datetime(date)

    date_string = date.strftime("%Y/%m/%d")

    if using_pyroscope:
        pyroscope.configure(
            application_name="pipeline",
            server_address="http://sps-archiver1.chime:4040",
            detect_subprocesses=True,  # Include multiprocessing pools
            sample_rate=100,  # In milliseconds
            oncpu=False,  # Include idle CPU time
            gil_only=False,  # Include threads not managed by Python's GIL
            enable_logging=True,
        )

    with (
        pyroscope.tag_wrapper({"pointing": f"{ra}-{dec}-{date_string}"})
        if using_pyroscope
        else nullcontext()
    ):
        sys.excepthook = dbexcepthook

        db_utils.connect(host=db_host, port=db_port, name=db_name)

        global pipeline_start_time
        pipeline_start_time = time.time()

        config = load_config(config_file, config_options)
        now = dt.datetime.utcnow()
        processing_failed = False

        if not date:
            date = dt.datetime(year=now.year, month=now.month, day=now.day)
        if stackpath:
            config.ps.ps_stack_config.basepath = stackpath
        basepath = os.path.abspath(basepath)

        log_path = basepath + f"/logs/{date.strftime('%Y/%m/%d')}/"

        log_name = (
            f"run_pipeline_{date.strftime('%Y-%m-%d')}_{ra:.02f}_"
            f"{dec:.02f}_{now.strftime('%Y-%m-%dT%H-%M-%S')}"
        )
        if using_docker:
            # So that we can trace Grafana container metrics to a log file
            # and see which node the container was spawned on
            log_name = (
                log_name
                + f"_{os.environ.get('CONTAINER_NAME')}"
                + f"_{os.environ.get('NODE_NAME')}"
            )
        log_name = log_name + ".log"

        log_file = log_path + log_name
        apply_logging_config(config, log_file)

        log.info(f"Used config: {config_file}")
        if config_options != "{}":
            log.info(f"Additional config options {config_options}")

        date = date.replace(tzinfo=pytz.UTC)
        log.info(date.strftime("%Y-%m-%d"))
        # First just look up the pointing without having to create an Observation
        try:
            closest_pointing = find_closest_pointing(ra, dec)
        except NoSuchPointingError:
            log.error("No observation within half a degree: %.2d, %.2d", ra, dec)
            exit(1)

        # Then check if an observation has already been created
        obs_folder = os.path.join(
            basepath,
            date.strftime("%Y/%m/%d"),
            f"{closest_pointing.ra:.02f}_{closest_pointing.dec:.02f}",
        )
        obs_id_files = sorted(
            glob(
                os.path.join(
                    obs_folder,
                    (
                        f"{closest_pointing.ra:.02f}_{closest_pointing.dec:.02f}_*_obs_id.txt"
                    ),
                )
            )
        )
        if len(obs_id_files) > 0:
            data_list = []
            # Just updating an existing observation
            strat = PointingStrategist(
                create_db=False, split_long_pointing=True, basepath=basepath
            )
            active_pointings = strat.active_pointing_from_pointing(
                closest_pointing, date
            )
            for obs_id_file in obs_id_files:
                with open(obs_id_file) as infile:
                    sub_pointing = int(obs_id_file.split("obs")[0].split("_")[-2])
                    for active_pointing in active_pointings:
                        if active_pointing.sub_pointing == sub_pointing:
                            current_obs_id = infile.read()
                            try:
                                old_obs_entry = db_api.get_observation(current_obs_id)
                            except models.DatabaseError as db_error:
                                log.error(db_error)
                                log.error(
                                    f"Observation with obs id {current_obs_id} from"
                                    f" file {obs_id_file} does not exist in used"
                                    " database. It may have been created using a"
                                    " different database or the entry was deleted."
                                )
                                exit(1)
                            active_pointing.obs_id = current_obs_id
        else:
            # New observation
            os.makedirs(
                obs_folder,
                exist_ok=True,
            )
            strat = PointingStrategist(split_long_pointing=True, basepath=basepath)
            active_pointings = strat.active_pointing_from_pointing(
                closest_pointing, date
            )
            # Record the obs id
            for active_pointing in active_pointings:
                obs_id_file = os.path.join(
                    basepath,
                    date.strftime("%Y/%m/%d"),
                    f"{active_pointing.ra:.02f}_{active_pointing.dec:.02f}",
                    (
                        f"{active_pointing.ra:.02f}_{active_pointing.dec:.02f}"
                        f"_{active_pointing.sub_pointing}_obs_id.txt"
                    ),
                )
                with open(obs_id_file, "w") as outfile:
                    outfile.write(str(active_pointing.obs_id))

        log.info("Observation: %s", active_pointings)
        assert date.date() == utils.transit_time(active_pointings[0]).date()

        global active_process
        N_ap = len(active_pointings)
        if N_ap > 1:
            log.info(
                "Pointing > maximum length specified in PointingStrategist, splitting"
                f" into {N_ap} active_pointings"
            )
            config.beamform.beam_to_normalise = None
        padded_length = config.ps_creation.padded_length

        for i_ap, active_pointing in enumerate(active_pointings):
            log.info(f"Processing active_pointing {i_ap + 1} of {N_ap}")
            active_process = db_api.get_process_from_active_pointing(
                active_pointings[0]
            )
            db_api.update_process(active_process.id, {"status": 4})
            db_api.update_observation(
                active_pointing.obs_id, {"status": 4, "log_file": log_file}
            )
            if not components or "all" in components:
                components = set(components) | {
                    "rfi",
                    "beamform",
                    "dedisp",
                    "ps",
                    "search",
                    "cleanup",
                }

            dedisp_ts = None
            ps_detections = None
            prefix = (
                f"{active_pointing.ra:.02f}_{active_pointing.dec:.02f}"
                f"_{active_pointing.sub_pointing}"
            )

            # Compute number of threads required.
            # Currently based on the number of channels of the input data

            ntime_factor = int(
                2 ** np.ceil(np.log2(active_pointing.ntime / 2**20))
            )  # round up to closest power of 2
            ntime_factor = max(ntime_factor, 1)  # minimum of 1
            nchan_factor = active_pointing.nchan // 1024
            if num_threads is None:
                num_threads = int(config.threads.thread_per_1024_chan * nchan_factor)

            log.info(f"Using {num_threads} threads to run the parallel codes")
            if stack and "ps" not in components:
                log.warning(
                    "The `--stack` option has no effect if power spectrum is not being"
                    " calculated."
                )
            if fdmt and "beamform" not in components:
                log.warning(
                    "Cannot run FDMT without beamforming, using presto dedispersion"
                    " instead"
                )
                fdmt = False
            if "beamform" in components:
                beamformer = beamform.initialise(
                    config, "rfi" in components, basepath, datpath
                )
                skybeam, spectra_shared = beamform.run(
                    active_pointing, beamformer, fdmt, num_threads, basepath
                )
                if skybeam is None:
                    spectra_shared.close()
                    spectra_shared.unlink()
                    components = ["cleanup"]
                    # processing_failed = True
                    insufficient_data = True
                else:
                    insufficient_data = False
            if "dedisp" in components:
                if fdmt:
                    from sps_pipeline import dedisp

                    dedisp_ts = dedisp.run_fdmt(
                        active_pointing, skybeam, config, num_threads
                    )
                    # remove skybeam from memory
                    spectra_shared.close()
                    spectra_shared.unlink()
                    del skybeam
                    gc.collect()
                else:
                    dedisp.run(active_pointing, config, basepath)
            elif "ps" in components or "ffa" in components:
                dedisp_ts = DedispersedTimeSeries.from_presto_datfiles(
                    obs_folder, active_pointing.obs_id, prefix=prefix
                )
            else:
                if "beamform" in components:
                    spectra_shared.close()
                    spectra_shared.unlink()

            # Run FFA search if requested
            if "ffa" in components and config.get("ffa", {}).get(
                "run_ffa_search", False
            ):
                log.info(
                    "FFA Search"
                    f" ({active_pointing.ra:.2f} {active_pointing.dec:.2f}) @"
                    f" {date:%Y-%m-%d}"
                )
                ffa_processor = ffa.initialise(config, num_threads)
                ffa.run(
                    active_pointing,
                    dedisp_ts,
                    ffa_processor,
                    basepath,
                    date,
                )
                # Note: We don't delete dedisp_ts here as PS might need it

            if "ps" in components:
                # splitting the FFT for power spectra and search/stack process, so
                # that we can delete dedispersed time series from memory first
                # before stacking.
                config.ps_creation.padded_length = ntime_factor * padded_length
                psc_pipeline = PowerSpectraCreation(
                    **OmegaConf.to_container(config.ps_creation),
                    num_threads=num_threads,
                )
                ps_pipeline = PowerSpectraPipeline(
                    **OmegaConf.to_container(config.ps),
                    run_ps_stack=stack,
                    num_threads=num_threads,
                    known_source_threshold=known_source_threshold,
                )
                # Use global to allow unlinking memory in exception hook
                global power_spectra
                log.info(
                    "Power Spectrum"
                    f" ({active_pointing.ra:.2f} {active_pointing.dec:.2f}) @"
                    f" {date:%Y-%m-%d}"
                )
                power_spectra = psc_pipeline.transform(dedisp_ts)
                # remove dedispersed time series from memory
                del dedisp_ts
                gc.collect()
                if "search" in components:
                    ps_detections = ps_pipeline.power_spectra_search(
                        power_spectra,
                        injection_path,
                        injection_idx,
                        only_injections,
                        scale_injections,
                        obs_folder,
                        prefix,
                    )
                    if ps_detections is None:
                        power_spectra.unlink_shared_memory()
                        del power_spectra
                        log.warning("No detections. Will proceed with cleanup.")
                        components = ["cleanup"]
                    else:
                        cands_processor = cands.initialise(config, num_threads)
                        cands.run(
                            active_pointing,
                            cands_processor,
                            ps_detections,
                            power_spectra,
                            plot,
                            plot_threshold,
                            basepath,
                            config.cands.get(
                                "write_harmonically_related_clusters", False
                            ),
                            injection_path,
                            injection_idx,
                            only_injections,
                        )
                    gc.collect()
                if stack:
                    # Depending on the stacking method this may change power_spectra,
                    # not entirely sure
                    ps_pipeline.power_spectra_stack(power_spectra)
                power_spectra.unlink_shared_memory()
                del power_spectra
            else:
                power_spectra = None
                # If FFA was run without PS, clean up dedisp_ts
                if "ffa" in components and "dedisp_ts" in locals():
                    del dedisp_ts
                    gc.collect()

            if "cleanup" in components:
                clean_up = cleanup.CleanUp(**OmegaConf.to_container(config.cleanup))
                clean_up.remove_files(active_pointing)
            # finishing the observation -- update its status to completed
            if insufficient_data:
                final_status = models.ObservationStatus.insufficient_data.value
            else:
                if processing_failed:
                    final_status = models.ObservationStatus.failed.value
                else:
                    final_status = models.ObservationStatus.complete.value
            obs_final_dict = db_api.update_observation(
                active_pointing.obs_id,
                {"status": final_status},
            )
            obs_final = models.Observation.from_db(obs_final_dict)
            is_in_stack = db_api.obs_in_stack(obs_final)

            new_process = {
                "status": 2,
                "obs_status": final_status,
                "quality_label": obs_final.add_to_stack,
                "is_in_stack": is_in_stack,
            }

            peak_memory = None
            peak_cpu = None

            if using_docker:
                try:

                    def get_peak_usage(values):
                        peak_usage = 0
                        for timestamp, usage in values:
                            usage = float(usage)
                            if usage > peak_usage:
                                peak_usage = usage
                        return peak_usage

                    prometheus_url = f"http://{db_host}:9090"
                    prometheus_client = PrometheusConnect(url=prometheus_url)
                    container_name = os.environ.get("CONTAINER_NAME")
                    pipeline_execution_time = (time.time() - pipeline_start_time) / 60
                    end_time = dt.datetime.now()
                    start_time = end_time - dt.timedelta(
                        minutes=pipeline_execution_time
                    )
                    # Use higher step size on testbed for benchmarking,
                    # but not for CHIME, to avoid Prometheus overload
                    # during processing at the telescope
                    step_size = 10
                    memory_query = (
                        f"container_memory_usage_bytes{{name='{container_name}'}}"
                    )
                    cpu_query = (
                        f"rate(container_cpu_user_seconds_total{{name='{container_name}'}}[120s])"
                        " * 100"
                    )
                    http_params = {"timeout": 3}  # seconds
                    memory_response = prometheus_client.custom_query_range(
                        memory_query,
                        start_time,
                        end_time,
                        step_size,
                        params=http_params,
                    )[0]
                    cpu_response = prometheus_client.custom_query_range(
                        cpu_query, start_time, end_time, step_size, params=http_params
                    )[0]
                    peak_memory = get_peak_usage(memory_response["values"]) / 1e9
                    peak_cpu = get_peak_usage(cpu_response["values"])
                    new_process["max_memory_usage"] = peak_memory
                    new_process["max_cpu_usage"] = peak_cpu
                    log.info(f"Max memory usage: {peak_memory}")
                    log.info(f"Max cpu usage: {peak_cpu}")
                except Exception as e:
                    log.info(
                        f"Cannot connect to Prometheus (Error: {e}). Will not "
                        "update resource usage for current process MongoDB item."
                    )

            pipeline_execution_time = time.time() - pipeline_start_time

            new_process["process_time"] = pipeline_execution_time

            db_api.update_process(active_process.id, new_process)

        log.info(
            f"Pipeline execution time: {(pipeline_execution_time / 60):.2f} minutes"
        )

        # Silence Workflow errors, requires results, products, plots
        return (
            {
                "container_name": os.environ.get("CONTAINER_NAME"),
                "node_name": os.environ.get("NODE_NAME"),
                "max_memory_usage": peak_memory,
                "max_cpu_usage": peak_cpu,
                "process_time": pipeline_execution_time / 60,
                "log_file": log_file,
            },
            [],
            [],
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--plot/--no-plot",
    default=False,
    help="Whether to create candidate plots",
)
@click.option(
    "--plot-threshold",
    default=0.0,
    type=float,
    help="Sigma threshold above which the candidate plots are created",
)
@click.option(
    "--file",
    default=None,
    type=str,
    help=(
        "Path to stack file to be processed. Allows processing without database access."
        "Only works with search-monthly. Overrules ra, dec option."
    ),
)
@click.argument("ra", type=click.FloatRange(-180, 360))
@click.argument("dec", type=click.FloatRange(-90, 90))
@click.argument(
    "components",
    type=click.Choice(["all", "search-monthly", "stack", "search", "cands"]),
    nargs=-1,
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver1",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--path-cumul-stack",
    default=None,
    type=str,
    help=(
        "Path for the cumulative stack. As default the basepath from the config is"
        " used."
    ),
)
@click.option(
    "--cand-path",
    default="",
    type=str,
    help="Path where the candidates are created",
)
@click.option(
    "--known-source-threshold",
    "--kst",
    default=np.inf,
    type=float,
    help=(
        "Threshold under which known sources are filtered based on the previously"
        " strongest detection of that source in that pointing."
    ),
)
@click.option(
    "--injection-path",
    default=None,
    help="Path to yml file containing injection",
)
@click.option(
    "--injection-idx",
    "--ii",
    default=[],
    type=int,
    multiple=True,
    help="Index of pulse to inject from yml file",
)
@click.option(
    "--only-injections/--no-only-injections",
    default=False,
    help="Only process clusters containing injections.",
)
@click.option(
    "--scale-injections/--not-scale-injections",
    default=False,
    help="Scale injection so that input sigma should be detected sigma.",
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
    help="Additional options that overwrite the config options. Provide a string that would define a python dictionary",
)
def stack_and_search(
    plot,
    plot_threshold,
    file,
    ra,
    dec,
    components,
    db_port,
    db_host,
    db_name,
    path_cumul_stack,
    cand_path,
    known_source_threshold,
    injection_path,
    injection_idx,
    only_injections,
    scale_injections,
    config_file,
    config_options,
):
    """
    Runner script to stack monthly PS into cumulative PS and search the eventual stack.

    Subcommands:
    - all: run all components (default)
    - stack: run stacking of monthly stack to cumulative stack
    - search: run the searching of the cumulative stack
    - search-monthly: run the searching of the monthly stack
    """

    multiprocessing.set_start_method("forkserver", force=True)
    sys.excepthook = dbexcepthook
    global pipeline_start_time
    pipeline_start_time = time.time()
    config = load_config(config_file, config_options)
    now = dt.datetime.now()
    log_path = str(cand_path) + f"./stack_logs/{now.strftime('%Y/%m/%d')}/"
    if not file:
        log_name = (
            f"run_stack_search_{ra:.02f}_"
            f"{dec:.02f}_{now.strftime('%Y-%m-%dT%H-%M-%S')}.log"
        )
    else:
        log_name = (
            f"run_stack_search_{os.path.basename(file)}_"
            f"{now.strftime('%Y-%m-%dT%H-%M-%S')}.log"
        )
    log_file = log_path + log_name
    apply_logging_config(config, log_file)
    if path_cumul_stack:
        config.ps_cumul_stack.ps_stack_config.basepath = path_cumul_stack
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    # First just look up the pointing without having to create an Observation
    try:
        db.command("ping")
        db_connection = True
    except ServerSelectionTimeoutError:
        log.info("Connection to database not possible.")
        db_connection = False
    if db_connection:
        try:
            closest_pointing = find_closest_pointing(ra, dec)
            closest_pointing_id = closest_pointing._id
        except NoSuchPointingError:
            log.error("No observation within half a degree: %.2d, %.2d", ra, dec)
            exit(1)
    else:
        closest_pointing_id = None
        config.ps_cumul_stack.ps_search_config.update_db = False
    if not components or "all" in components:
        components = set(components) | {"search-monthly", "stack", "search"}
    to_search_monthly = False
    to_stack = False
    to_search = False
    if "search-monthly" in components:
        to_search_monthly = True
    if "stack" in components:
        to_stack = True
    if "search" in components:
        to_search = True
    if "search-monthly" in components or "search" in components:
        cands_processor = cands.initialise(config)
    ps_cumul_stack_processor = ps_cumul_stack.initialise(
        config,
        to_stack,
        to_search,
        to_search_monthly,
        known_source_threshold=known_source_threshold,
    )
    global power_spectra
    if to_search_monthly:
        global power_spectra_monthly
        if ps_cumul_stack_processor.pipeline.run_ps_search_monthly:
            if not file:
                log.info(f"Performing searching on monthly stack {ra:.2f} {dec:.2f}")
            else:
                log.info(f"Performing searching on {file}")
        (
            ps_detections_monthly,
            power_spectra_monthly,
        ) = ps_cumul_stack_processor.pipeline.load_and_search_monthly(
            closest_pointing_id,
            injection_path,
            injection_idx,
            only_injections,
            scale_injections,
            file=file,
        )
        if db_connection:
            ps_stack = db_api.get_ps_stack(closest_pointing_id)
            stack_path = ps_stack.datapath_month
        else:
            stack_path = file
        if ps_detections_monthly:
            cands.run_interface(
                ps_detections_monthly,
                stack_path,
                cand_path,
                cands_processor,
                power_spectra_monthly,
                plot,
                plot_threshold,
                config.cands.get("write_harmonically_related_clusters", False),
                True,
                injection_path,
                injection_idx,
                only_injections,
                closest_pointing_id,
            )
    else:
        power_spectra_monthly = None

    if to_stack or to_search:
        ps_detections, power_spectra = ps_cumul_stack.run(
            closest_pointing,
            ps_cumul_stack_processor,
            power_spectra_monthly,
            injection_path,
            injection_idx,
            only_injections,
            scale_injections,
        )
        if to_search:
            if not ps_detections:
                log.error("No detections produced. Will not run candidate processing.")
            else:
                if power_spectra is None:
                    log.error(
                        "Candidate creation requires access to the power spectra."
                    )
                else:
                    ps_stack = db_api.get_ps_stack(closest_pointing._id)
                    cands.run_interface(
                        ps_detections,
                        ps_stack.datapath_cumul,
                        cand_path,
                        cands_processor,
                        power_spectra,
                        plot,
                        plot_threshold,
                        config.cands.get("write_harmonically_related_clusters", False),
                        False,
                        injection_path,
                        injection_idx,
                        only_injections,
                        closest_pointing_id,
                    )
    try:
        power_spectra_monthly.unlink_shared_memory()
        log.info("Unlinked shared memory for monthly stack.")
    except Exception:
        log.debug("No shared memory for monthly stack detected.")
    try:
        power_spectra.unlink_shared_memory()
        log.info("Unlinked shared memory for cumulative stack.")
    except Exception:
        log.debug("No shared memory detected for cumulative stack.")
    pipeline_end_time = time.time()
    log.info(
        "Pipeline execution time:"
        f" {((pipeline_end_time - pipeline_start_time) / 60):.2f} minutes"
    )

    return {}, [], []


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--full-transit/--no-full-transit",
    "-f",
    default=False,
    help=(
        "Only process pointings whose full transit fall within the specified utc range"
    ),
)
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=False,
    help="Date of data to process. Default = Today in UTC",
)
@click.option(
    "--beam",
    type=click.IntRange(0, 255),
    required=False,
    help=(
        "Beam row to check for existing data. Can only input a single beam row. Default"
        " = All rows from 0 to 224"
    ),
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver1",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--datpath",
    default=default_datpath,
    type=str,
    help="Path to the raw data folder.",
)
def find_pointing_with_data(
    date, beam, full_transit, db_port, db_host, db_name, datpath
):
    """
    Script to determine the pointings with data to process in a given day.

    How to use it : pointing-with-data --date YYYYMMDD
    or YYYY-MM-DD
    or YYYY/MM/DD
    --beam <beam_no>
    """
    db_utils.connect(host=db_host, port=db_port, name=db_name)

    if not date:
        now = dt.datetime.utcnow()
        date = dt.datetime(year=now.year, month=now.month, day=now.day)
    date = date.replace(tzinfo=pytz.UTC)
    print(date.strftime("%Y-%m-%d"))
    strat = PointingStrategist(create_db=False)
    istheredata = False
    if not beam:
        beam = np.arange(0, 224)
    else:
        beam = np.asarray([beam])
    for b in beam:
        datlist = sorted(
            glob(
                os.path.join(
                    datpath, date.strftime("%Y/%m/%d"), str(b).zfill(4), "*.dat"
                )
            )
        )
        start_times, end_times = utils.get_pointings_from_list(datlist)
        for i in range(len(start_times)):
            istheredata = True
            active_pointings = strat.get_pointings(
                start_times[i], end_times[i], np.asarray([b])
            )
            if full_transit:
                new_active_pointings = []
                for ap in active_pointings:
                    if (
                        ap.max_beams[0]["utc_start"] >= start_times[i]
                        and ap.max_beams[-1]["utc_end"] <= end_times[i]
                    ):
                        new_active_pointings.append(ap)
                active_pointings = new_active_pointings
            print(
                "List of pointings at beam row {} with data to process between utc {}"
                " and {}".format(b, start_times[i], end_times[i])
            )
            for ap in active_pointings:
                print(f"{ap.ra:.2f} {ap.dec:.2f}")
    if not istheredata:
        print("There are no pointings with data to process for the given beam rows")


if __name__ == "__main__":
    main()
