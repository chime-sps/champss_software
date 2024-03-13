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

import click
import docker
import numpy as np
import pyroscope
import pytz
from omegaconf import OmegaConf
from prometheus_api_client import PrometheusConnect
from slack_sdk import WebClient

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
from beamformer.utilities.common import find_closest_pointing, get_data_list
from chime_frb_api.workflow import Work
from folding.schedule_workflow import find_and_run_all_folding_processes
from ps_processes.processes.ps import PowerSpectraCreation
from ps_processes.ps_pipeline import PowerSpectraPipeline
from sps_common.interfaces import DedispersedTimeSeries
from sps_databases import db_api, db_utils, models

from sps_pipeline import (  # ps,
    beamform,
    cands,
    cleanup,
    dedisp,
    hhat,
    ps_cumul_stack,
    rfi,
    utils,
)

datpath = "/data/chime/sps/raw"


def load_config():
    """
    Combines default/user-specified config settings and applies them to loggers.

    User-specified settings can be given in two forms: as a YAML file in the
    current directory named "sps_config.yml", or as command-line arguments.

    The format of the file is (all sections optional):
    ```
    logging:
      format: string for the `logging.formatter`
      level: logging level for the root logger
      modules:
        module_name: logging level for the submodule `module_name`
        module_name2: etc.
    ```

    Returns
    -------
    The `omegaconf` configuration object merging all the default configuration
    with the (optional) user-specified overrides.
    """
    base_config = OmegaConf.load(os.path.dirname(__file__) + "/sps_config.yml")
    if os.path.exists("./sps_config.yml"):
        user_config = OmegaConf.load("./sps_config.yml")
    else:
        user_config = OmegaConf.create()

    return OmegaConf.merge(base_config, user_config)


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


def message_slack(
    slack_message,
    slack_channel="#slow-pulsar-alerts",
    slack_token="xoxb-194910630096-6273790557189-FKbg9w1HwrJYqGmBRY8DF0te",
):
    slack_client = WebClient(token=slack_token)
    slack_request = slack_client.chat_postMessage(
        channel=slack_channel,
        text=slack_message,
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
    "--rfi-beamform/--no-rfi-beamform",
    default=True,
    help="Whether to run rfi mitigation during beamforming instead of separately",
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
        ["all", "quant", "rfi", "beamform", "dedisp", "ps", "hhat", "search", "cleanup"]
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
    default="localhost",
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
    "--use-pyroscope",
    default=False,
    help=(
        "Whether to send this run's profiling to the Pyroscope service to view on"
        " its UI"
    ),
)
@click.option(
    "--using-docker/--not-using-docker",
    default=False,
    help="Whether this run is being used with Workflow + Docker Swarm",
)
def main(
    date,
    stack,
    fdmt,
    rfi_beamform,
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
    use_pyroscope,
    using_docker,
):
    """
    Runner script for the Slow Pulsar Search prototype pipeline v0.

    The normal way to run the script to process a pointing for a
    given day is run-pipeline --date YYYYMMDD <RA> <DEC>

    Subcommands:
    - all: run all components (default)

    - rfi: run only RFI excision

    - beamform: run only the beamformer

    - dedisp: run only the dedisperser

    - ps: run only power spectrum computation

    - hhat: run only Hhat computation

    - search: run only the search

    - cleanup: run only the cleanup operation

    You can also run a combination of several processes together.
    E.g. If you want to produce filterbank files only from scratch,
    you can do run-pipeline --date 20200701 317.86 20.96 rfi beamform
    """
    # Logging in multiprocessing child processes with Linux default
    # "fork" leads to unexpected behaviour
    multiprocessing.set_start_method("forkserver", force=True)

    if isinstance(date, str):
        for date_format in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
            try:
                date = dt.datetime.strptime(date, date_format)
                break
            except ValueError:
                continue

    if use_pyroscope:
        pyroscope.configure(
            application_name=f"pipeline-pyroscope-{ra}-{dec}-{date}",
            server_address="http://sps-archiver.chime:4040",
            detect_subprocesses=True,
            tags={"pointing": f"{ra}-{dec}-{date.strftime('%Y/%m/%d')}"},
        )
    with (
        pyroscope.tag_wrapper({"run1": "pyroscope test"})
        if use_pyroscope
        else nullcontext()
    ):
        sys.excepthook = dbexcepthook
        db_utils.connect(host=db_host, port=db_port, name=db_name)

        global pipeline_start_time
        pipeline_start_time = time.time()

        config = load_config()
        now = dt.datetime.utcnow()

        if not date:
            date = dt.datetime(year=now.year, month=now.month, day=now.day)
        if stackpath:
            config.ps.ps_stack_config.basepath = stackpath

        log_path = basepath + "/logs/"

        log_name = (
            f"run_pipeline_{date.strftime('%Y-%m-%d')}_{round(ra, 2)}_"
            f"{round(dec, 2)}_{now.strftime('%Y-%m-%dT%H-%M-%S')}"
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
            f"{closest_pointing.ra :.02f}_{closest_pointing.dec :.02f}",
        )
        obs_id_files = sorted(
            glob(
                os.path.join(
                    obs_folder,
                    (
                        f"{closest_pointing.ra :.02f}_{closest_pointing.dec :.02f}_*_obs_id.txt"
                    ),
                )
            )
        )
        if len(obs_id_files) > 0:
            data_list = []
            # Just updating an existing observation
            strat = PointingStrategist(create_db=False, split_long_pointing=True)
            active_pointings = strat.active_pointing_from_pointing(
                closest_pointing, date
            )
            for active_pointing in active_pointings:
                data_list.extend(
                    get_data_list(
                        active_pointing.max_beams, basepath=datpath, extn="dat"
                    )
                )
            if not data_list:
                log.error(
                    "No data found for the pointing {:.2f} {:.2f}".format(
                        active_pointings[0].ra, active_pointings[0].dec
                    )
                )
                sys.exit()
            for obs_id_file in obs_id_files:
                with open(obs_id_file) as infile:
                    sub_pointing = int(obs_id_file.split("obs")[0].split("_")[-2])
                    for active_pointing in active_pointings:
                        if active_pointing.sub_pointing == sub_pointing:
                            active_pointing.obs_id = infile.read()
        else:
            # New observation
            os.makedirs(
                obs_folder,
                exist_ok=True,
            )
            strat = PointingStrategist(split_long_pointing=True)
            active_pointings = strat.active_pointing_from_pointing(
                closest_pointing, date
            )
            data_list = []
            for active_pointing in active_pointings:
                data_list.extend(
                    get_data_list(
                        active_pointing.max_beams, basepath=datpath, extn="dat"
                    )
                )
            if not data_list:
                log.error(
                    "No data found for the pointing {:.2f} {:.2f}".format(
                        active_pointings[0].ra, active_pointings[0].dec
                    )
                )
                sys.exit()
            # Record the obs id
            for active_pointing in active_pointings:
                obs_id_file = os.path.join(
                    basepath,
                    date.strftime("%Y/%m/%d"),
                    f"{active_pointing.ra :.02f}_{active_pointing.dec :.02f}",
                    (
                        f"{active_pointing.ra :.02f}_{active_pointing.dec :.02f}"
                        f"_{active_pointing.sub_pointing}_obs_id.txt"
                    ),
                )
                with open(obs_id_file, "w") as outfile:
                    outfile.write(str(active_pointing.obs_id))

        log.info("Observation: %s", active_pointings)
        assert date.date() == utils.transit_time(active_pointings[0]).date()

        global active_process
        # RFI clean the data first
        if "quant" in components:
            nchan = max([p.nchan for p in active_pointings])
            beams_start_end = rfi.get_data_to_clean(active_pointings)
            run_quant(beams_start_end, nchan)
        if "rfi" in components and not rfi_beamform:
            beams_start_end = rfi.get_data_to_clean(active_pointings)
            rfi.run(beams_start_end, config, basepath)
        N_ap = len(active_pointings)
        if N_ap > 1:
            log.info(
                "Pointing > maximum length specified in PointingStrategist, splitting"
                f" into {N_ap} active_pointings"
            )
            config.beamform.beam_to_normalise = None
        padded_length = config.ps_creation.padded_length

        for i_ap, active_pointing in enumerate(active_pointings):
            log.info(f"Processing active_pointing {i_ap+1} of {N_ap}")
            active_process = db_api.get_process_from_active_pointing(
                active_pointings[0]
            )
            db_api.update_process(active_process.id, {"status": 4})
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
            prefix = f"{active_pointing.ra :.02f}_{active_pointing.dec :.02f}_{active_pointing.sub_pointing}"

            # Compute number of threads required. Currently based on the number of channels of the input data

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
                beamformer = beamform.initialise(config, rfi_beamform, basepath)
                skybeam = beamform.run(
                    active_pointing, beamformer, fdmt, num_threads, basepath
                )
                if db_api.get_observation(active_pointing.obs_id).mask_fraction == 1.0:
                    log.warning(
                        "Beamformed spectra are completely masked. Will proceed with"
                        " cleanup."
                    )
                    components = ["cleanup"]
            if "dedisp" in components:
                if fdmt:
                    dedisp_ts = dedisp.run_fdmt(
                        active_pointing, skybeam, config, num_threads
                    )
                    # remove skybeam from memory
                    del skybeam
                    gc.collect()
                else:
                    dedisp.run(active_pointing, config, basepath)
            elif "ps" in components:
                dedisp_ts = DedispersedTimeSeries.from_presto_datfiles(
                    obs_folder, active_pointing.obs_id, prefix=prefix
                )
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
                )
                # Use global to allow unlinking memory in exception hook
                global power_spectra
                log.info(
                    "Power Spectrum"
                    f" ({active_pointing.ra :.2f} {active_pointing.dec :.2f}) @"
                    f" { date :%Y-%m-%d}"
                )
                power_spectra = psc_pipeline.transform(dedisp_ts)
                # remove dedispersed time series from memory
                del dedisp_ts
                gc.collect()
                if "search" in components:
                    ps_detections = ps_pipeline.power_spectra_search(
                        power_spectra, obs_folder, prefix
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
            if "hhat" in components:
                hhat.run(active_pointing)
            if "cleanup" in components:
                clean_up = cleanup.CleanUp(**OmegaConf.to_container(config.cleanup))
                clean_up.remove_files(active_pointing)
                if config.cleanup_rfi:
                    cleanup.cleanup_rfi(
                        active_pointing.max_beams,
                    )
            # finishing the observation -- update its status to completed
            obs_final_dict = db_api.update_observation(
                active_pointing.obs_id,
                {"status": models.ObservationStatus.complete.value},
            )
            obs_final = models.Observation.from_db(obs_final_dict)
            is_in_stack = db_api.obs_in_stack(obs_final)

            new_process = {
                "status": 2,
                "obs_status": 2,
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
                    step_size = 30
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
    default="localhost",
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
def stack_and_search(
    plot,
    plot_threshold,
    ra,
    dec,
    components,
    db_port,
    db_host,
    db_name,
    path_cumul_stack,
):
    """
    Runner script to stack monthly PS into cumulative PS and search the eventual stack.

    Subcommands:
    - all: run all components (default)
    - stack: run stacking of monthly stack to cumulative stack
    - search: run the searching of the cumulative stack
    - search-monthly: run the searching of the monthly stack
    """
    multiprocessing.set_start_method("forkserver")
    sys.excepthook = dbexcepthook
    global pipeline_start_time
    pipeline_start_time = time.time()
    config = load_config()
    apply_logging_config(config)
    if path_cumul_stack:
        config.ps_cumul_stack.ps_stack_config.basepath = path_cumul_stack
    db_utils.connect(host=db_host, port=db_port, name=db_name)
    # First just look up the pointing without having to create an Observation
    try:
        closest_pointing = find_closest_pointing(ra, dec)
    except NoSuchPointingError:
        log.error("No observation within half a degree: %.2d, %.2d", ra, dec)
        exit(1)
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
        config, to_stack, to_search, to_search_monthly
    )
    global power_spectra
    if to_search_monthly:
        global power_spectra_monthly
        if ps_cumul_stack_processor.pipeline.run_ps_search_monthly:
            log.info(
                "Performing searching on monthly stack"
                f" {closest_pointing.ra:.2f} {closest_pointing.dec:.2f}"
            )
        (
            ps_detections_monthly,
            power_spectra_monthly,
        ) = ps_cumul_stack_processor.pipeline.load_and_search_monthly(
            closest_pointing.pointing_id
        )
        ps_stack = db_api.get_ps_stack(closest_pointing.pointing_id)
        if ps_detections_monthly:
            cands.run_interface(
                ps_detections_monthly,
                closest_pointing,
                ps_stack.datapath_month,
                cands_processor,
                power_spectra_monthly,
                plot,
                plot_threshold,
                config.cands.get("write_harmonically_related_clusters", False),
            )
    else:
        power_spectra_monthly = None

    if to_stack or to_search:
        ps_detections, power_spectra = ps_cumul_stack.run(
            closest_pointing, ps_cumul_stack_processor, power_spectra_monthly
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
                    ps_stack = db_api.get_ps_stack(closest_pointing.pointing_id)
                    cands.run_interface(
                        ps_detections,
                        closest_pointing,
                        ps_stack.datapath_cumul,
                        cands_processor,
                        power_spectra,
                        plot,
                        plot_threshold,
                        config.cands.get("write_harmonically_related_clusters", False),
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
    default="localhost",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
)
def find_pointing_with_data(date, beam, full_transit, db_port, db_host, db_name):
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


def run_quant(utc_start, utc_end, beam_row, nchan):
    """Quantize L1 data for the `pointing`."""
    try:
        from sps_pipeline import quant
    except ImportError:
        log.error("`ch_frb_l1` is a required dependency for quantization this step.")
        sys.exit(1)
    quant.run(utc_start, utc_end, beam_row, nchan)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--db-host",
    default="sps-archiver",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps-processing",
    type=str,
    help="Namespace used for the mongodb database.",
)
@click.option(
    "--start-date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="First date to start continuous processing on.",
)
@click.option(
    "--number-of-days",
    default=-1,
    type=int,
    help="Number of days to perform continuous processing on. -1 (default) for forever",
)
@click.option(
    "--basepath",
    default="/data/chime/sps/sps_processing",
    type=str,
    help="Path for created files. Default './'",
)
@click.option(
    "--min-ra",
    default=0,
    type=float,
    help="Minimum ra to process.",
)
@click.option(
    "--max-ra",
    default=360,
    type=float,
    help="Maximum ra to process.",
)
@click.option(
    "--min-dec",
    default=-30,
    type=float,
    help="Minimum dec to process.",
)
@click.option(
    "--max-dec",
    default=90,
    type=float,
    help="Maximum dec to process.",
)
@click.option(
    "--workflow-name",
    default="sps-processing",
    type=str,
    help="Which Worklow DB to create/use.",
)
def start_continuous_daily_processing(
    db_host,
    db_port,
    db_name,
    start_date,
    number_of_days,
    basepath,
    min_ra,
    max_ra,
    min_dec,
    max_dec,
    workflow_name,
):
    log.setLevel(logging.INFO)

    db = db_utils.connect(host=db_host, port=db_port, name=db_name)

    start_date = start_date.replace(tzinfo=pytz.UTC)

    date_to_process = start_date

    client = docker.from_env()

    service_tiers = [
        "tiny",
        "small",
        "medium",
        "large",
        "huge",
    ]

    def wait_for_no_running_tasks():
        is_a_task_running = True
        while is_a_task_running is True:
            is_a_task_running = False
            # Re-fetch Docker Swarm Service states
            services_all_tiers = [
                service
                for service in client.services.list()
                if service.name.split("_")[1] in service_tiers
            ]
            for service in services_all_tiers:
                for task in service.tasks():
                    if task["Status"]["State"] == "running":
                        is_a_task_running = True
                        log.info("A task is still running. Will check again.")
                        break
                if is_a_task_running is True:
                    break

    number_of_days_processed = 0

    def loop_condition():
        if number_of_days != -1:
            return number_of_days_processed < number_of_days
        else:
            return True

    while loop_condition():
        try:
            present_date = dt.datetime.now(dt.timezone.utc)
            yesterday_date = present_date - dt.timedelta(days=1)

            if date_to_process <= yesterday_date:
                log.info(
                    f"{date_to_process} should be done recording data (24 hours have"
                    " passed)."
                )
            else:
                time_left = date_to_process - yesterday_date
                seconds_left = time_left.total_seconds()
                hours_left = seconds_left / 3600

                log.info(
                    f"{date_to_process} is not at least 24 hours before"
                    " the present date. Data may not be ready. Sleeping for"
                    f" {hours_left} (that's the hours left until its ready)..."
                )

                time.sleep(seconds_left)

            find_all_available_processes(
                [
                    "--db-host",
                    db_host,
                    "--db-port",
                    db_port,
                    "--db-name",
                    db_name,
                    "--date",
                    date_to_process,
                ],
                standalone_mode=False,
            )

            date_string = date_to_process.strftime("%Y/%m/%d")
            total_processes = db["processes"].count_documents({"date": date_string})

            present_date = dt.datetime.now(dt.timezone.utc)

            time_passed = present_date - (date_to_process + dt.timedelta(days=1))
            seconds_passed = time_passed.total_seconds()
            hours_passed = seconds_passed / 3600

            beginning_message = (
                f"Starting processing for {date_string} \n"
                f"{total_processes} total processes\n"
                f"{hours_passed:.2f} hours passed since data recording was complete"
            )
            message_slack(beginning_message)

            start_time_of_processing = time.time()

            process_all_processes(
                [
                    "--db-host",
                    db_host,
                    "--db-port",
                    db_port,
                    "--db-name",
                    db_name,
                    "--date",
                    date_to_process,
                    "--min-ra",
                    min_ra,
                    "--max-ra",
                    max_ra,
                    "--min-dec",
                    min_dec,
                    "--max-dec",
                    max_dec,
                    "--basepath",
                    basepath,
                    "--stackpath",
                    basepath,
                    "--workflow-name",
                    workflow_name,
                ],
                standalone_mode=False,
            )

            wait_for_no_running_tasks()

            end_time_of_processing = time.time()

            overall_time_of_processing = (
                end_time_of_processing - start_time_of_processing
            ) / 60
            average_time_of_processing = (
                overall_time_of_processing / total_processes
            ) * 60

            completed_processes = db["processes"].count_documents(
                {
                    "date": date_string,
                    "status": 2,
                }
            )
            rfi_processeses = db["processes"].count_documents(
                {"date": date_string, "status": 2, "is_in_stack": False}
            )

            observations = list(
                db["observations"].find(
                    {
                        "datetime": {
                            "$gte": date_to_process,
                            "$lte": date_to_process + dt.timedelta(days=1),
                        }
                    },
                    {"num_detections": 1},
                )
            )
            observations = np.array(
                [
                    obs["num_detections"]
                    for obs in observations
                    if obs["num_detections"] is not None
                ]
            )

            if len(observations) > 0:
                mean_detections = round(np.mean(observations))
            else:
                mean_detections = 0

            processes = list(
                db["processes"].find(
                    {
                        "datetime": {
                            "$gte": date_to_process,
                            "$lte": date_to_process + dt.timedelta(days=1),
                        }
                    },
                    {"process_time": 1, "nchan": 1},
                )
            )

            time_per_nchan = {
                1024: {"total": 0, "count": 0},
                2048: {"total": 0, "count": 0},
                4096: {"total": 0, "count": 0},
                8192: {"total": 0, "count": 0},
                16384: {"total": 0, "count": 0},
            }

            for process in processes:
                nchan = process["nchan"]
                process_time = process["process_time"]
                if process_time is not None and nchan in time_per_nchan:
                    time_per_nchan[nchan]["total"] += process_time
                    time_per_nchan[nchan]["count"] += 1

            slack_message = (
                f"For {date_string}:\n"
                f"{completed_processes} / {total_processes} finished successfully\n"
                f"Of those, {rfi_processeses} were rejected by quality metrics\n"
                f"Mean number of detections: {mean_detections}\n"
                f"Overall processing time: {overall_time_of_processing:.2f} minutes ({average_time_of_processing:.2f} seconds per process average)"
            )

            for nchan in time_per_nchan.keys():
                time_of_nchan = time_per_nchan[nchan]
                if time_of_nchan["count"] > 0:
                    average_time = round(
                        time_of_nchan["total"] / time_of_nchan["count"]
                    )
                    slack_message += f"\nMean process time for nchan {nchan}: {average_time} seconds ({average_time / 60:.2f} minutes)"

            message_slack(slack_message)

            find_and_run_all_folding_processes(
                date=date_to_process,
                db_host=db_host,
                db_port=db_port,
                db_name=db_name,
                workflow_name=workflow_name,
            )

            number_of_days_processed = number_of_days_processed + 1

            wait_for_no_running_tasks()

            date_to_process = date_to_process + dt.timedelta(days=1)

        except Exception as error:
            log.error(error)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--full-transit/--no-full-transit",
    default=True,
    help=(
        "Only process pointings whose full transit fall within the specified utc range"
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
    default="localhost",
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
    "--complete/--no-complete",
    default=False,
    help=(
        "Rerun creation of all processes. Alternatively will start with last day where"
        " processes exist. NOT IMPLEMENTED YET"
    ),
)
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=False,
    help=(
        "First date for which processes are created. Default = All days. All days will"
        " take quite a long time."
    ),
)
@click.option(
    "--ndays",
    default=1,
    type=int,
    help=(
        "Number of days for which processes are created when --date is used for the"
        " first day."
    ),
)
def find_all_available_processes(
    full_transit, db_port, db_host, db_name, complete, date, ndays
):
    """Find all available processes and add them to the database."""
    log.setLevel(logging.INFO)
    db_utils.connect(host=db_host, port=db_port, name=db_name)
    strat = PointingStrategist(create_db=False)
    if not date:
        all_days = glob(os.path.join(datpath, "*/*/*"))
    else:
        all_days = []
        for day in range(ndays):
            all_days.extend(
                glob(
                    os.path.join(
                        datpath, (date + dt.timedelta(days=day)).strftime("%Y/%m/%d")
                    )
                )
            )
    log.info(f"Number of days: {len(all_days)}")
    total_processes = 0
    for day in all_days:
        log.info(f"Creating processes for {day}.")
        beam = np.arange(0, 224)
        total_processes_day = 0
        try:
            first_coordinates = None
            last_coordinates = None
            for b in beam:
                datlist = sorted(glob(os.path.join(day, str(b).zfill(4), "*.dat")))[:]
                start_times, end_times = utils.get_pointings_from_list(datlist)
                for i in range(len(start_times)):
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
                    if len(active_pointings) >= 1:
                        first_coordinates = (
                            active_pointings[0].ra,
                            active_pointings[0].dec,
                        )
                        last_coordinates = (
                            active_pointings[-1].ra,
                            active_pointings[-1].dec,
                        )
                        for ap in active_pointings:
                            db_api.get_process_from_active_pointing(ap)
                            total_processes_day += 1
            total_processes += total_processes_day
            log.info(f"{total_processes_day} available processes created for {day}.")
            log.info(f"First coordinates : {first_coordinates}")
            log.info(f"Last coordinates : {last_coordinates}")
        except Exception as error:
            log.error(error)
            log.info(f"Can't create processes for {day}")
    log.info(f"{total_processes} available processes in total.")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="localhost",
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
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=False,
    help="First date of data to process. Default = All days.",
)
@click.option(
    "--ndays",
    default=1,
    type=int,
    help="Number of days to process when --date is used for the first day.",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of parallel processes that are executed.",
)
@click.option(
    "--min-ra",
    default=0,
    type=float,
    help="Minimum ra to process.",
)
@click.option(
    "--max-ra",
    default=360,
    type=float,
    help="Maximum ra to process.",
)
@click.option(
    "--min-dec",
    default=-30,
    type=float,
    help="Minimum dec to process.",
)
@click.option(
    "--max-dec",
    default=90,
    type=float,
    help="Maximum dec to process.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Only print commands without actually running processes.",
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
    help="Path for the monthly stack. As default the basepath from the config is used.",
)
@click.option(
    "--workflow-name",
    default="",
    type=str,
    help="Which Worklow DB to create/use.",
)
def process_all_processes(
    db_port,
    db_host,
    db_name,
    date,
    ndays,
    workers,
    min_ra,
    max_ra,
    min_dec,
    max_dec,
    dry_run,
    basepath,
    stackpath,
    workflow_name,
):
    """Process all unprocessed processes in the database for a given range."""
    log.setLevel(logging.INFO)
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    query = {
        "ra": {"$gte": min_ra, "$lte": max_ra},
        "dec": {"$gte": min_dec, "$lte": max_dec},
        #    "status": 1,   For now grad all processes which are not
        #                   in stack and also not are considered RFI
        "$and": [{"is_in_stack": False}, {"quality_label": {"$ne": False}}],
        "nchan": {"$lte": 10000},  # Temporarily filter 16k nchan proc
    }
    if date:
        query["datetime"] = {"$gte": date, "$lte": date + dt.timedelta(days=ndays)}
    all_processes = list(db.processes.find(query))
    if dry_run:
        log.info("Will only print out processes commands without running them.")
    log.info(f"{len(all_processes)} process found.")

    all_processes = sorted(
        all_processes,
        key=lambda process: (process["date"], process["ra"], process["dec"]),
    )

    for process_index, process_dict in enumerate(all_processes):
        process = models.Process.from_db(process_dict)
        log.info(
            f"Will process pointing ({process.ra}, {process.dec}) for date"
            f" {process.date}"
        )
        try:
            cmd_string_list = (
                f"--date {process.date} --db-port {db_port} --db-host {db_host} --stack"
                " --fdmt --rfi-beamform".split(" ")
            )
            if basepath:
                cmd_string_list.extend(["--basepath", f"{basepath}"])
            if stackpath:
                cmd_string_list.extend(["--stackpath", f"{stackpath}"])
            cmd_string_list.extend(
                [
                    f" {process.ra}",
                    f" {process.dec}",
                    "all",
                ]
            )

            log.info(f"Running command: run-pipeline {' '.join(cmd_string_list)}")
            if not dry_run:
                if workflow_name:
                    log.info(
                        f"Scheduling Workflow job for {process} "
                        f"on {db_host}:{db_port}:{db_name} MongoDB to "
                        f"{workflow_name} Workflow DB"
                    )
                    schedule_workflow_job(
                        db_host=db_host,
                        db_port=db_port,
                        db_name=db_name,
                        basepath=basepath,
                        stackpath=stackpath,
                        date=process.date,
                        ra=process.ra,
                        dec=process.dec,
                        nchan=process.nchan,
                        ntime=process.ntime,
                        workflow_name=workflow_name,
                    )
                else:
                    main(cmd_string_list, standalone_mode=False)

                    log.info(
                        f"Finished processing pointing ({process.ra}, {process.dec})"
                        f" for date {process.date}"
                    )
        except Exception as error:
            log.error(error)
            if not dry_run:
                db_api.update_process(process.id, {"status": 3})
    log.info(f"{len(all_processes)} process found.")


def schedule_workflow_job(
    db_port,
    db_host,
    db_name,
    basepath,
    stackpath,
    date,
    ra,
    dec,
    nchan,
    ntime,
    workflow_name,
):
    """Deposit Work and scale Docker Service, as node resources are free."""
    client = docker.from_env()

    service_tiers = [
        "tiny",
        "small",
        "medium",
        "large",
        "huge",
    ]

    if ntime <= 2**20:
        ntime_factor = 1
    elif ntime > 2**20 and ntime <= 2**21:
        ntime_factor = 2
    else:
        # capped, since longer pointings broken into chunks of 2^22
        ntime_factor = 4
    i_tier = int(np.log2(nchan // 1024 * ntime_factor))

    tags = ["pipeline"]
    if i_tier < len(service_tiers):
        tags.append(service_tiers[i_tier])
    else:
        log.error(
            f"nchan {nchan} ntime {ntime} does not correspond into an existing Docker"
            " Swarm Service"
        )
        return

    service_attrs = [
        service
        for service in client.services.list()
        if service.name == f"pipeline_{tags[1]}"
    ][0].attrs

    def get_resource(attrs, cap_type, resource_type):
        return attrs["Spec"]["TaskTemplate"]["Resources"][cap_type][resource_type] / 1e9

    # Available CPU cores is approx. 1/2 of available GB of RAM on-site, and
    # max_overall_threads == total_cpu_cores is optimal performance unless
    # mutliprocesses are often blocked by sync events (I/O, network, etc) which
    # in our case are not. Although n threads does not equal n cores being 100% used,
    # so more can be used if shown to be beneficial.
    threads_needed = int(
        get_resource(service_attrs, "Reservations", "MemoryBytes") // 2
    )

    work = Work(pipeline=f"{workflow_name}", site="chime", user="SPAWG")
    work.function = "sps_pipeline.pipeline.main"
    work.parameters = {
        "stack": True,
        "fdmt": True,
        "rfi_beamform": True,
        "plot": True,
        "plot_threshold": 8.0,
        "components": ["all"],
        "num_threads": threads_needed,
        "ra": ra,
        "dec": dec,
        "date": date,
        "db_port": db_port,
        "db_host": db_host,
        "db_name": db_name,
        "basepath": basepath,
        "stackpath": stackpath,
        "use_pyroscope": False,
        "using_docker": True,
    }
    work.tags = tags
    work.config.archive.results = True
    work.config.archive.plots = "pass"
    work.config.archive.products = "pass"
    work.retries = 0

    docker_swarm_pending_states = [
        "new",
        "pending",
        "assigned",
        "accepted",
        "ready",
        "preparing",
        "starting",
    ]
    docker_swarm_running_states = ["running"]
    docker_swarm_finished_states = [
        "complete",
        "failed",
        "shutdown",
        "rejected",
        "orphaned",
        "remove",
    ]

    # 15 minutes from now timeout. Sometimes a Docker Swarm task
    # gets stuck in pending state indefinitely for unknown reasons...
    timeout = time.time() + (60 * 15)  # 15 minutes from now
    is_timeout_reached = False

    # Docker Swarm can be freeze if too many tasks are in pending state concurrently.
    # Additionally, Docker Swarm deqeues pending jobs in random order, so we need to
    # only have one pending job at a time, to maintain ordering
    is_a_task_pending = True
    while is_a_task_pending is True and is_timeout_reached is False:
        is_a_task_pending = False
        # Re-fetch Docker Swarm Service states
        services_all_tiers = [
            service
            for service in client.services.list()
            if service.name.split("_")[1] in service_tiers
        ]
        for service in services_all_tiers:
            if time.time() > timeout:
                is_timeout_reached = True
                break

            for task in service.tasks():
                task_state = task["Status"]["State"]
                task_id = task["ID"]
                if (task_state not in docker_swarm_finished_states) and (
                    task_state not in docker_swarm_running_states
                ):
                    log.info(
                        f"Unfinished, non-running task {task_id} is in state {task_state}"
                    )
                    if task_state in docker_swarm_pending_states:
                        is_a_task_pending = True
                        break
            if is_a_task_pending is True:
                break

    if is_timeout_reached is True:
        log.info(
            (
                "A task is stuck in pending state, timeout reached."
                "Scaling down Docker Swarm Services..."
            )
        )
        services_all_tiers = [
            service
            for service in client.services.list()
            if service.name.split("_")[1] in service_tiers
        ]
        for service in services_all_tiers:
            service.scale(0)

    log.info("Depositing Workflow Work object...")

    work.deposit()

    service_this_tier_name = f"pipeline_{tags[1]}"
    service_this_tier = [
        service
        for service in client.services.list()
        if service.name == service_this_tier_name
    ][0]

    log.info("Scaling Docker Swarm Service Workflow runner...")

    service_this_tier.scale(
        service_this_tier.attrs["Spec"]["Mode"]["Replicated"]["Replicas"] + 1
    )

    # Wait a second before querying Docker Swarm for next task
    time.sleep(1)


if __name__ == "__main__":
    main()
