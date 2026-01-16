"""Functions to enable automatic processing."""

import datetime as dt
import logging
import os
import sys
import time
import traceback
from functools import partial
from glob import glob
from multiprocessing import Pool
import pymongo
import atexit
import pandas as pd
from bson.objectid import ObjectId

import click
import docker
import numpy as np
import pytz
from beamformer.strategist.strategist import PointingStrategist
from folding.filter_mpcandidates import filter_mp_df, write_df_to_fsdb
from scheduler.utils import convert_date_to_datetime
from scheduler.workflow import (  # docker_swarm_pending_states,
    clear_workflow_buckets,
    docker_swarm_running_states,
    get_work_from_results,
    message_slack,
    schedule_workflow_job,
    wait_for_no_tasks_in_states,
    remove_finished_service,
)
from sps_pipeline.create_quality_report import create_report_pdf
from sps_common.interfaces import MultiPointingCandidate
from workflow.definitions.work import Work
from workflow.http.context import HTTPContext
from sps_databases import db_api, db_utils, models
from sps_pipeline.pipeline import default_datpath
from sps_pipeline.utils import get_pointings_from_list, merge_images
from sps_pipeline.candidate_viewer import CandidateViewerRegistrar
from sps_pipeline.stack_scheduling import find_monthly_search_commands


log = logging.getLogger()

max_work_duration = 60 * 60  # in seconds


def scale_down_service(service_name):
    try:
        docker_client = docker.from_env()
        docker_client.services.get(service_name).scale(0)
        log.info(f"Scaled down service {service_name}")
    except Exception as error:
        pass


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    help="Date to find folding processes for.",
)
@click.option(
    "--db-host",
    default="sps-archiver1",
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
    help="Name used for the mongodb database.",
)
def find_all_folding_processes(date, db_host, db_port, db_name):
    """Find all available folding processes for a given date."""
    log.setLevel(logging.INFO)

    db = db_utils.connect(host=db_host, port=db_port, name=db_name)

    log.info(f"Filtering candidates for {date}")

    date = convert_date_to_datetime(date)
    daily_run = db_api.get_daily_run(date)
    csv_input_name = daily_run.classification_result["output_file"]
    candidate_df = pd.read_csv(csv_input_name, index_col=0)

    filtered_df = filter_mp_df(candidate_df, sigma_min=6.5, class_min=0.9)
    filtered_df = write_df_to_fsdb(filtered_df, date)
    # output_file = csv.rsplit("_", 1)[0] + "_folded.csv"

    # filtered_df.to_csv(output_file)

    # Filter(
    #     cand_obs_date=date,
    #     db_host=db_host,
    #     db_port=db_port,
    #     db_name=db_name,
    #     write_to_db=True,
    #     basepath=basepath,
    #     foldpath=foldpath,
    # )
    # Filter(
    #     cand_obs_date=date,
    #     db_host=db_host,
    #     db_port=db_port,
    #     db_name=db_name,
    #     write_to_db=True,
    #     basepath=basepath,
    #     foldpath=foldpath,
    #     class_threshold=0.5,
    # )

    log.info("Candidate filtering complete")

    # Update FollowUpSources database, then get all ids where active is True
    IDs = []
    ras = []
    decs = []
    dms = []

    info = []

    for source in db.followup_sources.find({"active": True}):
        IDs.append(source["_id"])
        ras.append(source["ra"])
        decs.append(source["dec"])
        dms.append(source["dm"])

        info.append(
            {
                "fs_id": source["_id"],
                "ra": source["ra"],
                "dec": source["dec"],
                "dm": source["dm"],
            }
        )
    # Outputting the df will probably break, if this function is run by workflow.
    return {"info": info, "df": filtered_df}, [], []


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    help="Date to find folding processes for.",
)
@click.option(
    "--db-host",
    default="sps-archiver1",
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
    help="Name used for the mongodb database.",
)
@click.option(
    "--foldpath",
    default="/data/chime/sps/archives",
    type=str,
    help="Path for created files during fold step.",
)
@click.option(
    "--datpath",
    default=default_datpath,
    type=str,
    help="Path to the raw data folder.",
)
@click.option(
    "--processes",
    default=[],
    type=list,
    help="A list of dictionaries representing each process' information needed to run.",
)
@click.option(
    "--workflow-buckets-name",
    default="champss-fold",
    type=str,
    help="Which Worklow DB to create/use.",
)
@click.option(
    "--docker-image-name",
    default="sps-archiver1.chime:5000/champss_software:latest",
    type=str,
    help="Which Docker Image name to use.",
)
@click.option(
    "--docker-service-name-prefix",
    default="fold",
    type=str,
    help="What prefix to apply to the Docker Service name",
)
def run_all_folding_processes(
    date,
    db_host,
    db_port,
    db_name,
    foldpath,
    datpath,
    processes,
    workflow_buckets_name,
    docker_image_name,
    docker_service_name_prefix,
):
    """Run all available folding processes for a given date."""
    date = convert_date_to_datetime(date)

    date_string = date.strftime("%Y/%m/%d")

    message_slack(f"Folding {len(processes)} candidates for {date_string}")

    work_ids = []
    for process in processes:
        fs_id = process["fs_id"]
        ra = process["ra"]
        dec = process["dec"]
        dm = process["dm"]

        nchan_tier = int(np.ceil(np.log2(dm // 212.5 + 1)))
        nchan = 1024 * (2**nchan_tier)

        formatted_ra = f"{ra:.02f}"
        formatted_dec = f"{dec:.02f}"
        formatted_date = date_string
        formatted_dm = f"{dm:.02f}"
        formatted_id = str(fs_id)

        docker_name = (
            f"{docker_service_name_prefix}-{formatted_ra}-"
            f"{formatted_dec}-{formatted_date}-{formatted_id}"
        )
        docker_memory_reservation = (nchan / 1024) * 8
        docker_image = docker_image_name
        docker_mounts = [
            f"{datpath}:{datpath}",
            # f"{basepath}:{basepath}",
            f"{foldpath}:{foldpath}",
        ]

        workflow_function = "folding.fold_candidate.main"
        workflow_params = {
            "date": formatted_date,
            "fs_id": formatted_id,
            "db_host": db_host,
            "db_port": db_port,
            "db_name": db_name,
            "foldpath": foldpath,
            "datpath": datpath,
            "write_to_db": True,
            "using_workflow": True,
        }
        workflow_tags = [
            "fold",
            formatted_id,
            formatted_dm,
            formatted_ra,
            formatted_dec,
            formatted_date,
        ]
        work_ids.append(
            schedule_workflow_job(
                docker_image,
                docker_mounts,
                docker_name,
                docker_memory_reservation,
                workflow_buckets_name,
                workflow_function,
                workflow_params,
                workflow_tags,
            )
        )
    return work_ids


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
    default="sps-archiver1",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps-processing",
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
@click.option(
    "--datpath",
    default=default_datpath,
    type=str,
    help="Path to the raw data folder.",
)
@click.option(
    "--alert-slack/--no-slack-alert", default=False, help="Alert slack about results."
)
def find_all_pipeline_processes(
    full_transit, db_port, db_host, db_name, complete, date, ndays, datpath, alert_slack
):
    """Find all available processes and add them to the database."""
    date = convert_date_to_datetime(date)

    log.setLevel(logging.INFO)
    import multiprocessing

    multiprocessing.set_start_method("forkserver", force=True)
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
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

    all_processes = []
    for day in all_days:
        log.info(f"Creating processes for {day}.")
        beams = np.arange(0, 224)
        total_processes_day = 0
        split_day = day.split("/")
        date = dt.datetime(int(split_day[-3]), int(split_day[-2]), int(split_day[-1]))
        end_date = date + dt.timedelta(days=1)
        old_proc = list(
            db.processes.find({"datetime": {"$gte": date, "$lte": end_date}})
        )
        proc_dict = {}
        for proc in old_proc:
            proc_dict[str(proc["pointing_id"])] = proc
        try:
            first_coordinates = None
            last_coordinates = None
            with Pool(32) as pool:
                active_pointings_list = pool.map(
                    partial(
                        find_active_pointings,
                        day=day,
                        strat=strat,
                        full_transit=full_transit,
                        db_name=db_name,
                        db_host=db_host,
                        db_port=db_port,
                    ),
                    beams,
                )

            active_pointings = [
                ap for ap_list in active_pointings_list for ap in ap_list
            ]

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
                existing_proc = proc_dict.get(ap.pointing_id, {})
                if not existing_proc:
                    process = db_api.get_process_from_active_pointing(
                        ap, assume_new=True
                    )
                else:
                    process = models.Process.from_db(existing_proc)
                total_processes_day += 1
                all_processes.append(process)

            total_processes += total_processes_day
            log.info(f"{total_processes_day} available processes created for {day}.")
            log.info(f"First coordinates : {first_coordinates}")
            log.info(f"Last coordinates : {last_coordinates}")
        except Exception as error:
            log.error(error)
            log.info(f"Can't create processes for {day}")
    finished_procs = [proc for proc in all_processes if proc.status.value == 2]
    unfinished_procs = [proc for proc in all_processes if proc.status.value != 2]
    log.info(
        f"{total_processes} available processes in total. {len(finished_procs)} have already finished."
    )
    if alert_slack:
        message_slack(
            f"For folders {all_days} found {total_processes} available processes in total. \n{len(finished_procs)} of those have already finished."
        )
    return (
        {
            "unfinished_processes": unfinished_procs,
            "finished_processes": finished_procs,
        },
        [],
        [],
    )


def find_active_pointings(beam, day, strat, full_transit, db_name, db_host, db_port):
    """Helper function to look for active pointings for a single beam."""
    db_utils.connect(host=db_host, port=db_port, name=db_name)
    datlist = sorted(glob(os.path.join(day, str(beam).zfill(4), "*.dat")))[:]
    start_times, end_times = get_pointings_from_list(datlist)
    all_active_pointings = []
    if not start_times:
        return []
    for i in range(len(start_times)):
        active_pointings = strat.get_pointings(
            start_times[i], end_times[i], np.asarray([beam])
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
        all_active_pointings.extend(active_pointings)
    return all_active_pointings


def deposit_pipeline_work(
    workflow_params, workflow_buckets_name, httpcontext, process_dict
):
    process = models.Process.from_db(process_dict)
    formatted_ra = f"{process.ra:.02f}"
    formatted_dec = f"{process.dec:.02f}"
    formatted_maxdm = f"{process.maxdm:.02f}"
    formatted_date = process.date

    ram_requirement = process.ram_requirement

    tier_name, tier_limit = process.tier
    threads_reserved = int(tier_limit / 3)

    workflow_params.update(
        {
            "date": process.date,
            "ra": process.ra,
            "dec": f" {process.dec}",
            "num_threads": threads_reserved,
        }
    )
    work = Work(
        pipeline=workflow_buckets_name, site="chime", user="CHAMPSS", http=httpcontext
    )

    work.function = "sps_pipeline.pipeline.main"
    work.parameters = workflow_params
    work.tags = [
        "pipeline",
        formatted_ra,
        formatted_dec,
        formatted_maxdm,
        formatted_date,
        tier_name,
        str(process.id),
        str(process.pointing_id),
        str(process.obs_id),
    ]
    work.config.archive.results = True
    work.config.archive.plots = "bypass"
    work.config.archive.products = "bypass"
    work.retries = 1
    work.timeout = max_work_duration

    work_id = work.deposit(return_ids=True)
    return work_id


def deposit_stack_work(workflow_buckets_name, httpcontext, stack_parameters):
    work = Work(
        pipeline=workflow_buckets_name, site="chime", user="CHAMPSS", http=httpcontext
    )
    work.function = "sps_pipeline.pipeline.stack_and_search"
    work.parameters = stack_parameters["arguments"]
    work.tags = [
        "stack-search",
        stack_parameters["tier_name"],
    ]
    work.config.archive.results = True
    work.config.archive.plots = "bypass"
    work.config.archive.products = "bypass"
    work.retries = 1
    work.timeout = max_work_duration

    work_id = work.deposit(return_ids=True)
    return work_id


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--db-host",
    default="sps-archiver1",
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
    help="Path for created files during pipeline step.",
)
@click.option(
    "--stackpath",
    default=None,
    type=str,
    help="Path for the monthly stack. As default the basepath from the config is used.",
)
@click.option(
    "--datpath",
    default=default_datpath,
    type=str,
    help="Path to the raw data folder.",
)
@click.option(
    "--workflow-buckets-name",
    default="champss-pipeline",
    type=str,
    help="Which Worklow DB to create/use.",
)
@click.option(
    "--docker-image-name",
    default="sps-archiver1.chime:5000/champss_software:latest",
    type=str,
    help="Which Docker Image name to use.",
)
@click.option(
    "--docker-service-name-prefix",
    default="pipeline",
    type=str,
    help="What prefix to apply to the Docker Service name",
)
@click.option(
    "--run-stacking",
    default=True,
    type=bool,
    help="To run the stacking part of the pipeline or not.",
)
@click.option(
    "--pipeline-arguments",
    default="",
    type=str,
    help=("Additional pipeline arguments."),
)
@click.option(
    "--pipeline-config-options",
    default="{}",
    type=str,
    help=(
        "Options passed to --config-options of the pipeline. Use single quotes inside the string!"
    ),
)
@click.option(
    "--alert-slack/--no-slack-alert", default=False, help="Alert slack about results."
)
@click.option(
    "--mode",
    default="pipeline",
    type=str,
    help="Which mode to use. Available: [pipeline, stack]",
)
def run_all_pipeline_processes(
    db_host,
    db_port,
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
    datpath,
    workflow_buckets_name,
    docker_image_name,
    docker_service_name_prefix,
    run_stacking,
    pipeline_arguments,
    pipeline_config_options,
    alert_slack,
    mode,
):
    """Process all unprocessed processes in the database for a given range."""
    date = convert_date_to_datetime(date)

    log.setLevel(logging.INFO)
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    http = HTTPContext()
    if mode == "pipeline":
        if run_stacking:
            query = {
                "ra": {"$gte": min_ra, "$lte": max_ra},
                "dec": {"$gte": min_dec, "$lte": max_dec},
                #    "status": 1,   For now grad all processes which are not
                #                   in stack and also not are considered RFI
                "$and": [{"is_in_stack": False}, {"quality_label": {"$ne": False}}],
                # "nchan": {"$lte": 10000},  # Temporarily filter 16k nchan proc
            }
        else:
            query = {
                "ra": {"$gte": min_ra, "$lte": max_ra},
                "dec": {"$gte": min_dec, "$lte": max_dec},
                "status": {"$ne": 2},
            }
        if date:
            query["datetime"] = {"$gte": date, "$lte": date + dt.timedelta(days=ndays)}
        all_processes = list(db.processes.find(query))
        if dry_run:
            log.info("Will only print out processes commands without running them.")
        log.info(f"{len(all_processes)} process found.")

        if alert_slack:
            time_passed = dt.datetime.now(dt.timezone.utc) - (
                date + dt.timedelta(days=1)
            )
            hours_passed = time_passed.total_seconds() / 3600
            if len(all_processes):
                message_slack(
                    "Starting processing for"
                    f" {date.strftime('%Y/%m/%d')}\n{len(all_processes)} processes will run."
                    f" \n{hours_passed:.2f} hours passed"
                    " since data recording was complete."
                )
            else:
                message_slack(f"No process found that fit the query {query}")

        all_processes = sorted(
            all_processes,
            key=lambda process: (process["date"], process["ra"], process["dec"]),
        )

        workflow_params = {
            "stack": run_stacking,
            "fdmt": True,
            "rfi_beamform": True,
            "plot": False,
            "plot_threshold": 8.0,
            "components": ["all"],
            "db_port": db_port,
            "db_host": db_host,
            "db_name": db_name,
            "basepath": basepath,
            "stackpath": stackpath,
            "datpath": datpath,
            "using_pyroscope": False,
            "using_docker": True,
            "config_options": pipeline_config_options,
            "known_source_threshold": 10,
        }
        if pipeline_arguments != "":
            split_args = pipeline_arguments.split("--")
            for arg_string in split_args:
                arg_string = arg_string.strip()
                if arg_string != "":
                    arg_count = len(arg_string.split(" "))
                    if arg_count > 2:
                        argument, value = arg_string.split(" ", 1)
                        workflow_params[argument] = (value,)
                    elif arg_count == 1:
                        workflow_params[argument] = value
                    else:
                        log.error(
                            "Flags not implimented yet. Reformated your option to --option_python_name True"
                        )
        process_ids = [process["_id"] for process in all_processes]
        pool = Pool(4)
        # First deposit all process in workflow bucket
        # Imap will perform these jobs in the background, a single process can deposit ~4 jobs per second
        workflow_buckets_name = "champss-pipeline"
        work_ids = pool.imap(
            partial(
                deposit_pipeline_work, workflow_params, workflow_buckets_name, http
            ),
            all_processes,
        )
    elif mode == "stack":
        all_processes = find_monthly_search_commands(
            db_port, db_host, db_name, basepath, 10
        )
        pool = Pool(4)
        # First deposit all process in workflow bucket
        # Imap will perform these jobs in the background, a single process can deposit ~4 jobs per second
        workflow_buckets_name = "champss-stack-search"
        work_ids = pool.imap(
            partial(
                deposit_stack_work,
                workflow_buckets_name,
                http,
            ),
            all_processes,
        )
    else:
        log.error("Unrecognized mode.")
        sys.exit()

    docker_client = docker.from_env()
    services = []
    docker_mounts = [
        f"{datpath}:{datpath}",
        f"{basepath}:{basepath}",
    ]
    docker_volumes = [
        docker.types.Mount(
            # Bind mount the Docker socket to allow Docker-in-Docker (Workflow-in-Workflow) usage
            target="/var/run/docker.sock",
            source="/var/run/docker.sock",
            type="bind",
        ),
        docker.types.Mount(
            # Only way I know of to add custom shared memory size allocations with Docker Swarm
            target="/dev/shm",
            source="",  # Source value must be empty for tmpfs mounts
            type="tmpfs",
            tmpfs_size=int(
                100 * 1e9
            ),  # Just give it 100GB of a shared memory as an upper-limit
        ),
    ]
    for mount_path in docker_mounts:
        mount_paths = mount_path.split(":")
        mount_source = mount_paths[0]
        mount_target = mount_paths[1]
        docker_volumes.append(
            docker.types.Mount(target=mount_target, source=mount_source, type="bind")
        )

    # Create Processing services
    processing_tier_limits = models.processing_tier_limits
    processing_tier_names = models.processing_tier_names
    for tier_name, tier_limit in zip(processing_tier_names, processing_tier_limits):
        service_name = f"processing-{tier_name}"
        try:
            docker_client.services.get(service_name).remove()
        except docker.errors.NotFound:
            pass
        docker_service = {
            "image": docker_image_name,
            # Can't have dots or slashes in Docker Service names
            # All Docker Services made with this function will be prefixed with "processing-"
            "name": service_name,
            "command": (
                "workflow run"
                f" {workflow_buckets_name} --site"
                f" chime --lives -1 --sleep 1"
                f" --tag {tier_name}"
            ),
            # Using template Docker variables as in-container environment variables
            # that allow us this access out-of-container information
            "env": ["CONTAINER_NAME={{.Task.Name}}", "NODE_NAME={{.Node.Hostname}}"],
            # This is neccessary to allow Pyroscope (py-spy) to work in Docker
            # 'cap_add': ['SYS_PTRACE'],
            "mode": docker.types.ServiceMode("replicated", replicas=0),
            "restart_policy": docker.types.RestartPolicy(
                condition="none", max_attempts=0
            ),
            # Labels allow for easy filtering with Docker CLI
            "labels": {"type": "processing"},
            # The labels on the Docker Nodes are pre-empetively set beforehand
            "constraints": ["node.labels.compute == true"],
            # Must be in bytes
            "resources": docker.types.Resources(mem_reservation=int(tier_limit * 1e9)),
            # Will throw an error if you give two of the same bind mount paths
            # e.g. avoid double-mounting basepath and stackpath when they are the same
            "mounts": docker_volumes,
            # An externally created Docker Network that allows these spawned containers
            # to communicate with other containers (MongoDB, Prometheus, etc) that are
            # also manually added to this network
            "networks": ["pipeline-network"],
            "stop_grace_period": int(max_work_duration * 1e9),
        }

        log.info(f"Creating Docker Service: \n{docker_service}")

        service = docker_client.services.create(**docker_service)
        services.append(service.attrs["ID"])
        atexit.register(scale_down_service, service.attrs["ID"])

    requested_containers = 100
    update_time = 60
    surplus_replicas = 20
    # This checks if enough work objects have been deposited. More work objects are scheduled in the background
    for work_index, work in enumerate(work_ids):
        if work_index > requested_containers:
            break

    buckets_db = pymongo.MongoClient(host="sps-archiver1", port=27018).work.buckets
    # Get the next work objects. These should be processed in order by workflow
    all_works = list(
        buckets_db.find({"pipeline": workflow_buckets_name})
        .sort("creation", pymongo.ASCENDING)
        .limit(requested_containers)
    )
    first_loop = True
    running_tiers = {}
    while len(all_works) > 0:
        # Count how many services should be created
        upcoming_tags = {tier: 0 for tier in processing_tier_names}
        for i, work in enumerate(all_works[:]):
            current_tag = set(work["tags"]).intersection(set(processing_tier_names))
            current_tag = [tag for tag in current_tag][0]
            upcoming_tags[current_tag] += 1
        if first_loop:
            log.info(
                f"Requested distribution with {requested_containers} containers: {upcoming_tags}."
            )

        # Scale services
        for i, tier in enumerate(processing_tier_names):
            # Sometime old services get stuck.
            # Momentarily scale to running jobs to fix that
            if tier in running_tiers.keys():
                docker_client.services.get(services[i]).scale(running_tiers[tier])
            docker_client.services.get(services[i]).scale(upcoming_tags[tier])
        time.sleep(update_time)
        # Check how many services are running
        # DO not update during the first loop since the image may still need to be distributed
        if not first_loop:
            running_tasks = 0
            running_tiers = {tier: 0 for tier in processing_tier_names}
            running_tiers_string = {}
            for i, tier in enumerate(processing_tier_names):
                service_tasks = docker_client.services.get(services[i]).tasks()
                running_tasks_per_tier = sum(
                    1 for task in service_tasks if task["Status"]["State"] == "running"
                )
                running_tasks += running_tasks_per_tier
                running_tiers_string[tier] = (
                    f"{running_tasks_per_tier}/{upcoming_tags[tier]}"
                )
                running_tiers[tier] = running_tasks_per_tier
            log.info(
                f"Currently running distribution with {running_tasks} containers: {running_tiers_string}"
            )
            requested_containers = max(
                running_tasks + surplus_replicas, 2 * surplus_replicas
            )
        else:
            first_loop = False
        all_works = list(
            buckets_db.find({"pipeline": workflow_buckets_name})
            .sort("creation", pymongo.ASCENDING)
            .limit(requested_containers)
        )
    log.info(
        "No work scheduled anymore, will scale all services down and remove them once they are finished."
    )
    for i, tier in enumerate(processing_tier_names):
        docker_client.services.get(services[i]).scale(0)
    log.info("Scaled down services.")

    for i, tier in enumerate(processing_tier_names):
        running_tasks_per_tier = 1
        while running_tasks_per_tier:
            service_tasks = docker_client.services.get(services[i]).tasks()
            running_tasks_per_tier = sum(
                1 for task in service_tasks if task["Status"]["State"] == "running"
            )
            if running_tasks_per_tier == 0:
                try:
                    docker_client.services.get(services[i]).remove()
                    log.info(f"Removed service {services[i]}.")
                except docker.errors.NotFound:
                    log.info("Could not remove processing service {services[i]}.")
            else:
                time.sleep(update_time)
    log.info("Finished processing and removed all services.")

    return process_ids


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--db-host",
    default="sps-archiver1",
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
    "--ndays",
    default=-1,
    type=int,
    help="Number of days to perform continuous processing on. -1 (default) for forever",
)
@click.option(
    "--basepath",
    default="/data/chime/sps/sps_processing",
    type=str,
    help="Path for created files during pipeline step.",
)
@click.option(
    "--foldpath",
    default="/data/chime/sps/archives",
    type=str,
    help="Path for created files during fold step.",
)
@click.option(
    "--datpath",
    default=default_datpath,
    type=str,
    help="Path to the raw data folder.",
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
    "--workflow-buckets-name-prefix",
    default="champss",
    type=str,
    help="What prefix to include for the Worklow DB to create/use.",
)
@click.option(
    "--docker-image-name",
    default="sps-archiver1.chime:5000/champss_software:latest",
    type=str,
    help="Which Docker Image name to use.",
)
@click.option(
    "--run-pipeline",
    default=True,
    type=bool,
    help="To run the pipeline phase of processing or not.",
)
@click.option(
    "--run-multipointing",
    default=True,
    type=bool,
    help="To run the multipointing phase of processing or not.",
)
@click.option(
    "--run-classification",
    default=True,
    type=bool,
    help="To run the classification phase of processing or not.",
)
@click.option(
    "--run-folding",
    default=True,
    type=bool,
    help="To run the folding phase of processing or not.",
)
@click.option(
    "--run-stacking",
    default=True,
    type=bool,
    help="To run the stacking part of the pipeline or not.",
)
@click.option(
    "--pipeline-arguments",
    default="",
    type=str,
    help=(
        "Additional pipeline arguments. Not usable for --config-options."
        "Use -- to separate options but use the python names of the variables (with underscored instead of dashes) instead of the cli names."
        """For example: "--only_injections True --kst 10" """
    ),
)
@click.option(
    "--pipeline-config-options",
    default="{}",
    type=str,
    help=(
        """Options passed to --config-options of the pipeline. Example: "{'beamform': {'max_mask_frac': 0.9}}" """
    ),
)
@click.option(
    "--skip-finding-processes/--no-skip-finding-processes",
    default=False,
    type=bool,
    help="Skip finding processes when this is not necessary anymore (during debugging for example).",
)
@click.option(
    "--run-stack-search/--no-run-stack-search",
    default=False,
    type=bool,
    help="Run stack search only. Will later implement to run stack every n days. INCOMPLETE: WILL ADD BETTER CONTROL LATER.",
)
def start_processing_manager(
    db_host,
    db_port,
    db_name,
    start_date,
    number_of_days,
    basepath,
    foldpath,
    datpath,
    min_ra,
    max_ra,
    min_dec,
    max_dec,
    workflow_buckets_name_prefix,
    docker_image_name,
    run_pipeline,
    run_multipointing,
    run_classification,
    run_folding,
    run_stacking,
    pipeline_arguments,
    pipeline_config_options,
    skip_finding_processes,
    run_stack_search,
):
    """Manager function containing the multiple processing steps."""
    # atexit.register(remove_processing_services, None, None)
    # signal.signal(signal.SIGINT, remove_processing_services)
    # signal.signal(signal.SIGQUIT, remove_processing_services)
    # signal.signal(signal.SIGABRT, remove_processing_services)
    # signal.signal(signal.SIGTERM, remove_processing_services)

    # Ugly way of removing superfluous handler that comes from somehwere
    try:
        log.removeHandler(log.handlers[1])
    except Exception as error:
        pass

    start_date = convert_date_to_datetime(start_date)

    log.setLevel(logging.INFO)

    db = db_utils.connect(host=db_host, port=db_port, name=db_name)

    start_date = start_date.replace(tzinfo=pytz.UTC)

    date_to_process = start_date

    number_of_days_processed = 0

    def loop_condition():
        # For now just enale running h stack search once.
        if run_stack_search:
            return number_of_days_processed < 1
        elif number_of_days != -1:
            # If number_of_days is not -1, then we want to run for a specific number of days
            return number_of_days_processed < number_of_days
        else:
            # If number_of_days is -1 (default if none given), then we want to run forever
            return True

    while loop_condition():
        try:
            present_date = dt.datetime.now(dt.timezone.utc)
            present_date_string = present_date.strftime("%Y/%m/%d")
            yesterday_date = present_date - dt.timedelta(days=1)

            date_string = date_to_process.strftime("%Y/%m/%d")
            if run_stack_search:
                mode = "stack-search"
            else:
                mode = "pipeline"

            if date_to_process <= yesterday_date:
                log.info(
                    f"{date_string} should be done recording data (24 hours have"
                    " passed)."
                )
            else:
                time_left = date_to_process - yesterday_date
                seconds_left = time_left.total_seconds()
                hours_left = seconds_left / 3600

                log.info(
                    f"{date_string} is not at least 24 hours before"
                    " the present date. Data may not be ready. Sleeping for"
                    f" {hours_left} (that's the hours left until its ready)..."
                )

                time.sleep(seconds_left)

            # Start of pipeline phase
            if run_pipeline:
                if not skip_finding_processes and not (mode == "stack-search"):
                    processes, [], [] = find_all_pipeline_processes.main(
                        args=[
                            "--db-host",
                            db_host,
                            "--db-port",
                            db_port,
                            "--db-name",
                            db_name,
                            "--date",
                            date_to_process,
                            "--datpath",
                            datpath,
                            "--alert-slack",
                        ],
                        standalone_mode=False,
                    )
                    daily_run = db_api.update_daily_run(
                        date_to_process,
                        {
                            "schedule_result": {
                                key: len(val) for key, val in processes.items()
                            }
                        },
                    )
                else:
                    processes = {
                        "unfinished_processes": ["dummy"],
                        "finished_processes": [],
                    }

                if len(processes["unfinished_processes"]) == 0:
                    if len(processes["finished_processes"]) == 0:
                        message_slack(
                            f"No processes found for {date_string}. Will progress to"
                            " next day"
                        )
                        number_of_days_processed = number_of_days_processed + 1
                        date_to_process = date_to_process + dt.timedelta(days=1)
                        continue
                    else:
                        message_slack(
                            f"No unfinished processes found for {date_string}. Will progress to"
                            " next step"
                        )
                else:
                    present_date = dt.datetime.now(dt.timezone.utc)

                    time_passed = present_date - (
                        date_to_process + dt.timedelta(days=1)
                    )

                    start_time_of_processing = time.time()

                    if run_stack_search:
                        docker_service_name_prefix = "stack-search"
                    else:
                        docker_service_name_prefix = "pipeline"

                    workflow_buckets_name = (
                        f"{workflow_buckets_name_prefix}-{docker_service_name_prefix}"
                    )
                    db_work = pymongo.MongoClient(
                        host="sps-archiver1", port=27018
                    ).work.buckets
                    db_work.delete_many({"pipeline": workflow_buckets_name})
                    pipeline_args = [
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
                        "--datpath",
                        datpath,
                        "--workflow-buckets-name",
                        workflow_buckets_name,
                        "--docker-image-name",
                        docker_image_name,
                        "--docker-service-name-prefix",
                        docker_service_name_prefix,
                        "--run-stacking",
                        run_stacking,
                        "--pipeline-arguments",
                        pipeline_arguments,
                        "--pipeline-config-options",
                        pipeline_config_options,
                        "--alert-slack",
                    ]
                    if run_stack_search:
                        pipeline_args.extend(["--mode", "stack"])
                    else:
                        pipeline_args.extend(["--mode", "pipeline"])
                    process_ids = run_all_pipeline_processes.main(
                        args=pipeline_args,
                        standalone_mode=False,
                    )

                    # wait_for_no_tasks_in_states(
                    #     docker_swarm_running_states, docker_service_name_prefix
                    # )
                    if not run_stack_search:
                        end_time_of_processing = time.time()
                        processed = list(
                            db.processes.find({"_id": {"$in": process_ids}})
                        )
                        if not len(processed):
                            number_of_days_processed = number_of_days_processed + 1
                            date_to_process = date_to_process + dt.timedelta(days=1)
                            continue

                        overall_time_of_processing = (
                            end_time_of_processing - start_time_of_processing
                        ) / 60
                        average_time_of_processing = (
                            overall_time_of_processing / len(processed)
                        ) * 60

                        completed_processes = [
                            proc for proc in processed if proc["status"] == 2
                        ]
                        rfi_processes = [
                            proc
                            for proc in completed_processes
                            if proc["quality_label"] is False
                        ]
                        db["processes"].count_documents(
                            {"date": date_string, "status": 2, "is_in_stack": False}
                        )

                        obs_ids = [
                            ObjectId(proc["obs_id"]) for proc in completed_processes
                        ]
                        observations = list(
                            db.observations.find({"_id": {"$in": obs_ids}})
                        )
                        mean_detections = np.nanmean(
                            [
                                obs["num_detections"]
                                for obs in observations
                                if obs["num_detections"] is not None
                            ]
                        )

                        # Get the execution time and nchan per process for the day
                        time_per_nchan = {
                            "1024": {"total": 0, "count": 0},
                            "2048": {"total": 0, "count": 0},
                            "4096": {"total": 0, "count": 0},
                            "8192": {"total": 0, "count": 0},
                            "16384": {"total": 0, "count": 0},
                        }

                        for process in processed:
                            nchan = str(process["nchan"])
                            execution_time = process["process_time"]

                            if execution_time is not None and nchan in time_per_nchan:
                                time_per_nchan[nchan]["total"] += execution_time
                                time_per_nchan[nchan]["count"] += 1

                        slack_message = (
                            f"For {date_string}:\n{len(completed_processes)} /"
                            f" {len(processed)} finished successfully for the day\nOf those,"
                            f" {len(rfi_processes)} were rejected by quality metrics\nMean number"
                            f" of detections: {mean_detections}\nOverall processing time for"
                            f" current run: {overall_time_of_processing:.2f} minutes\n(/"
                            f" {len(processed)} jobs run ="
                            f" {average_time_of_processing:.2f} seconds)"
                        )

                        for nchan in time_per_nchan.keys():
                            time_of_nchan = time_per_nchan[nchan]

                            if time_of_nchan["count"] > 0:
                                mean_time_of_nchan = round(
                                    time_of_nchan["total"] / time_of_nchan["count"]
                                )

                                slack_message += (
                                    f"\nMean process time for nchan {float(nchan)}:"
                                    f" {mean_time_of_nchan} seconds"
                                    f" ({mean_time_of_nchan / 60:.2f} minutes)"
                                )

                        message_slack(slack_message)
                        pipeline_result = {
                            "completed_processes": len(completed_processes),
                            "overall_time_of_processing": overall_time_of_processing,
                            "time_per_nchan": time_per_nchan,
                            "rfi_processes": len(rfi_processes),
                            "average_time_of_processing": average_time_of_processing,
                        }
                        daily_run = db_api.update_daily_run(
                            date_to_process, {"pipeline_result": pipeline_result}
                        )
            # End of pipeline phase

            # Start of multi-pointing phase
            if run_multipointing:
                message_slack(f"Running multi-pointing for {date_string}")

                docker_service_name_prefix = "mp"

                workflow_buckets_name = (
                    f"{workflow_buckets_name_prefix}-{docker_service_name_prefix}"
                )
                clear_workflow_buckets.main(
                    args=["--workflow-buckets-name", workflow_buckets_name],
                    standalone_mode=False,
                )
                mp_timeout = 60 * 60 * 6
                if mode == "pipeline":
                    workflow_params = {
                        "output": basepath,
                        "file_path": None,
                        "get_from_db": True,
                        "date": date_string,
                        "ndays": 1,
                        "plot": True,
                        "plot_cands": True,
                        "plot_all_pulsars": True,
                        "db": True,
                        "csv": True,
                        "plot_threshold": 8,
                        "plot_dm_threshold": 3,
                        "db_port": db_port,
                        "db_host": db_host,
                        "db_name": db_name,
                        "num_threads": 64,
                        "run_name": f"daily_{date_string}",
                    }
                    docker_name = f"{docker_service_name_prefix}-{date_string}"
                    workflow_tags = ["mp", date_string]
                else:
                    date_string
                    workflow_params = {
                        "output": basepath,
                        "file_path": None,
                        "get_from_db": True,
                        "date": date_string,
                        "plot": True,
                        "plot_cands": True,
                        "plot_all_pulsars": True,
                        "db": True,
                        "csv": True,
                        "plot_threshold": 8,
                        "plot_dm_threshold": 3,
                        "db_port": db_port,
                        "db_host": db_host,
                        "db_name": db_name,
                        "num_threads": 64,
                        "run_name": "stack_1",
                        "use_stacks": True,
                    }
                    docker_name = f"{docker_service_name_prefix}-{date_string}"
                    workflow_tags = ["mp", "stack", present_date_string]
                work_id, mp_service_id = schedule_workflow_job(
                    docker_image=docker_image_name,
                    docker_mounts=[
                        f"{datpath}:{datpath}",
                        f"{basepath}:{basepath}",
                    ],
                    docker_name=docker_name,
                    docker_memory_reservation=200,
                    workflow_buckets_name=workflow_buckets_name,
                    workflow_function="sps_multi_pointing.mp_pipeline.cli",
                    workflow_params=workflow_params,
                    workflow_tags=workflow_tags,
                    timeout=mp_timeout,
                    return_service_id=True,
                )

                # wait_for_no_tasks_in_states(
                #     docker_swarm_running_states,
                #     docker_service_name_prefix,
                #     timeout=mp_timeout,
                # )
                remove_finished_service(mp_service_id)

                # Need to wait a few seconds for results to propogate to Workflow
                time.sleep(5)

                work_result = get_work_from_results(
                    workflow_results_name=workflow_buckets_name,
                    work_id=work_id,
                    failover_to_buckets=True,
                )["results"]

                # See if result exists, and has the keys (by checking one of them)
                if work_result and "num_files" in work_result:
                    message_slack(
                        "Results from multi-pointing: \nNumber of files:"
                        f" {work_result['num_files']}\nNumber of single-pointing"
                        f" candidates: {work_result['num_sp_cands']}\nNumber of"
                        " multi-pointing candidates:"
                        f" {work_result['num_mp_cands']}\nNumber of known pulsar"
                        f" detections: {work_result['known_pulsar_detections']}"
                    )
                else:
                    message_slack(
                        "No results from multi-pointing. See Workfklow Web for errors,"
                        f" with filter: ALL_tags=['mp', '{date_string}']"
                    )
                if mode == "pipeline":
                    daily_run = db_api.update_daily_run(
                        date_to_process, {"multipointing_result": work_result}
                    )
            # End of multi-pointing phase

            # Start of classification phase
            if run_classification:
                if mode == "pipeline":
                    daily_run = db_api.get_daily_run(date_to_process)
                message_slack(f"Running classfication for {date_string}")

                docker_service_name_prefix = "class"

                workflow_buckets_name = (
                    f"{workflow_buckets_name_prefix}-{docker_service_name_prefix}"
                )
                clear_workflow_buckets.main(
                    args=["--workflow-buckets-name", workflow_buckets_name],
                    standalone_mode=False,
                )
                class_timeout = 60 * 60 * 5
                if mode == "pipeline":
                    input_csv = daily_run.multipointing_result["csv_file"]
                else:
                    input_csv = f"{basepath}/mp_runs/stack_1/all_mp_cands.csv"
                work_id, class_service_id = schedule_workflow_job(
                    docker_image="sps-archiver1.chime:5000/champss_classification:test",
                    docker_mounts=[
                        f"{datpath}:{datpath}",
                        f"{basepath}:{basepath}",
                    ],
                    docker_name=f"{docker_service_name_prefix}-{date_string}",
                    docker_memory_reservation=300,
                    workflow_buckets_name=workflow_buckets_name,
                    workflow_function="champss_classification.pytorch_model.classify_lazy.load_model_and_classify_mp_csv_lazy",
                    workflow_params={
                        "csv": input_csv,
                    },
                    workflow_tags=["class", date_string],
                    timeout=class_timeout,
                    return_service_id=True,
                )
                remove_finished_service(class_service_id)
                time.sleep(5)
                # wait_for_no_tasks_in_states(
                #     docker_swarm_running_states,
                #     docker_service_name_prefix,
                #     timeout=class_timeout,
                # )
                work_result = get_work_from_results(
                    workflow_results_name=workflow_buckets_name,
                    work_id=work_id,
                    failover_to_buckets=True,
                )["results"]
                if mode == "pipeline":
                    daily_run = db_api.update_daily_run(
                        date_to_process,
                        {"classification_result": work_result},
                    )
            # End of classification phase

            # Start of folding phase
            if run_folding:
                if mode == "pipeline":
                    daily_run = db_api.get_daily_run(date_to_process)
                    df_folded_name = (
                        daily_run.classification_result["output_file"].rsplit("_", 1)[0]
                        + "_folded.csv"
                    )
                else:
                    df_folded_name = f"{basepath}/stack_{present_date_string}/all_mp_cands_folded.csv"
                fold_schedule_output, [], [] = find_all_folding_processes.main(
                    args=[
                        "--date",
                        date_to_process,
                        "--db-host",
                        db_host,
                        "--db-port",
                        db_port,
                        "--db-name",
                        db_name,
                    ],
                    standalone_mode=False,
                )
                df_mp = fold_schedule_output["df"]
                try:
                    df_mp.to_csv(df_folded_name)
                except Exception as error:
                    # Might fail due to permission
                    pass
                processes = fold_schedule_output["info"]

                docker_service_name_prefix = "fold"

                workflow_buckets_name = (
                    f"{workflow_buckets_name_prefix}-{docker_service_name_prefix}"
                )
                clear_workflow_buckets.main(
                    args=["--workflow-buckets-name", workflow_buckets_name],
                    standalone_mode=False,
                )

                work_ids = run_all_folding_processes.main(
                    args=[
                        "--date",
                        date_to_process,
                        "--db-host",
                        db_host,
                        "--db-port",
                        db_port,
                        "--db-name",
                        db_name,
                        "--foldpath",
                        foldpath,
                        "--processes",
                        processes,
                        "--workflow-buckets-name",
                        workflow_buckets_name,
                        "--docker-image-name",
                        docker_image_name,
                        "--docker-service-name-prefix",
                        docker_service_name_prefix,
                        "--datpath",
                        datpath,
                    ],
                    standalone_mode=False,
                )

                wait_for_no_tasks_in_states(
                    docker_swarm_running_states, docker_service_name_prefix
                )

                # Merge candidates
                # breakpoint()
                log.info("Creating merged candidate plots.")
                merged_candidate_path = (
                    basepath + "/combined_candidates/" + date_string + "/"
                )
                replotted_mp_path = basepath + "/mp_candidates/" + date_string + "/"

                os.makedirs(merged_candidate_path, exist_ok=True)
                os.makedirs(replotted_mp_path, exist_ok=True)

                for index, row in df_mp.iterrows():
                    plot_path = row["plot_path"]
                    fs_db_entry = db_api.get_followup_source(row["fs_id"])
                    fold_history = fs_db_entry.folding_history
                    if len(fs_db_entry.folding_history) == 0:
                        continue
                    last_fold = fs_db_entry.folding_history[-1]
                    df_mp.at[index, "fold_plot"] = last_fold["path_to_plot"]
                    df_mp.at[index, "fs_sigma"] = last_fold["SN"]
                    df_mp.at[index, "fs_file"] = last_fold["archive_fname"]
                    if not os.path.exists(last_fold["path_to_plot"]):
                        continue
                    if type(row["plot_path"]) is not str:
                        if not os.path.exists(row["file_name"]):
                            continue
                        mp_cand = MultiPointingCandidate.read(row["file_name"])
                        plot_path = mp_cand.plot_candidate(path=replotted_mp_path)
                        df_mp.at[index, "plot_path"] = plot_path

                    output_path = (
                        merged_candidate_path
                        + plot_path.rsplit("/", 1)[1].rsplit(".", 1)[0]
                        + "_combined.png"
                    )
                    merge_images(
                        [plot_path, df_mp.at[index, "fold_plot"]],
                        output_path=output_path,
                    )
                    df_mp.at[index, "combined_plot_path"] = output_path
                # Could get work results, alternatively can query fs db
                try:
                    df_mp.to_csv(df_folded_name)
                except Exception as error:
                    log.error("Could not write out csv containing combined candidates.")
                    # Might fail due to permission
                    pass
                # Add to candidate viewer site
                db_config = {
                    "user": "automation",
                    "password": "",  # no password for automation user
                    "host": "sps-archiver1",
                    "database": "champss",
                    "port": 3306,
                }
                min_sigma_folded = 9
                try:
                    with (
                        CandidateViewerRegistrar(
                            survey="dailycands",  # the project name under top-right corner of the website
                            folder=date_to_process.strftime(
                                "%Y-%m-%d"
                            ),  # the folder name on the website
                            db_config=db_config,
                            survey_dir="/data/candidate_viewer/champss_candidate_viewer/surveys",  # path to the directory containing survey (project) config files
                        ) as sd
                    ):
                        df_mp_filtered = df_mp[df_mp["fs_sigma"] > min_sigma_folded]
                        sd.add_candidates(
                            df_mp_filtered
                        )  # add candidates from dataframe
                        sd.commit()  # commit to database and update survey config

                        message_slack(f"Candidate folding for {date_string} complete")
                        daily_run = db_api.update_daily_run(
                            date_to_process,
                            {"folding_result": {"fold_count": len(processes)}},
                        )
                except Exception as error:
                    log.error(f"Could not update daily candidates due to {error}")
                    log.error(traceback.format_exc())
            # End of folding phase
            create_report_pdf(date_to_process, db_host, db_port, db_name, basepath)

            number_of_days_processed = number_of_days_processed + 1
            date_to_process = date_to_process + dt.timedelta(days=1)

        except Exception as error:
            message_slack(
                "Error encountered during"
                f" processing:\n{traceback.format_exc() or error}\nWill progress to"
                " next day."
            )
            number_of_days_processed = number_of_days_processed + 1
            date_to_process = date_to_process + dt.timedelta(days=1)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--db-host",
    default="sps-archiver1",
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
    type=str,
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
    help="Path for created files during pipeline step.",
)
@click.option(
    "--foldpath",
    default="/data/chime/sps/archives",
    type=str,
    help="Path for created files during fold step.",
)
@click.option(
    "--datpath",
    default=default_datpath,
    type=str,
    help="Path to the raw data folder.",
)
@click.option(
    "--workflow-buckets-name-prefix",
    default="champss",
    type=str,
    help="What prefix to include for the Worklow DB to create/use.",
)
@click.option(
    "--manager-docker-image-name",
    default="sps-archiver1.chime:5000/champss_software:latest",
    type=str,
    help="Name of Docker Image to use for the processing manager.",
)
@click.option(
    "--pipeline-docker-image-name",
    default="sps-archiver1.chime:5000/champss_software:latest",
    type=str,
    help="Name of Docker Image to use for the processing pipeline.",
)
@click.option(
    "--run-pipeline",
    default=True,
    type=bool,
    help="To run the pipeline phase of processing or not.",
)
@click.option(
    "--run-multipointing",
    default=True,
    type=bool,
    help="To run the multipointing phase of processing or not.",
)
@click.option(
    "--run-classification",
    default=True,
    type=bool,
    help="To run the classification phase of processing or not.",
)
@click.option(
    "--run-folding",
    default=True,
    type=bool,
    help="To run the folding phase of processing or not.",
)
@click.option(
    "--run-stacking",
    default=True,
    type=bool,
    help="To run the stacking part of the pipeline or not.",
)
@click.option(
    "--pipeline-arguments",
    default="",
    type=str,
    help=(
        "Additional pipeline arguments. Not usable for --config-options."
        "Use -- to separate options but use the python names of the variables (with underscored instead of dashes) instead of the cli names."
        """For example: "--only_injections True --kst 10" """
    ),
)
@click.option(
    "--pipeline-config-options",
    default="{}",
    type=str,
    help=(
        """Options passed to --config-options of the pipeline. Example: "{'beamform': {'max_mask_frac': 0.9}}" """
    ),
)
def start_processing_services(
    db_host,
    db_port,
    db_name,
    start_date,
    number_of_days,
    basepath,
    foldpath,
    datpath,
    workflow_buckets_name_prefix,
    manager_docker_image_name,
    pipeline_docker_image_name,
    run_pipeline,
    run_multipointing,
    run_classification,
    run_folding,
    run_stacking,
    pipeline_arguments,
    pipeline_config_options,
):
    """Start the processing manager and the cleanup service."""
    # Please run "docker login" in your CLI to allow retrieval of the images
    log.setLevel(logging.INFO)

    docker_client = docker.from_env()

    docker_service_manager = {
        "image": manager_docker_image_name,
        "name": "processing-manager",
        "command": (
            f"start-processing-manager --db-host {db_host} --db-port"
            f" {db_port} --db-name {db_name} --start-date {start_date} --number-of-days"
            f" {number_of_days} --basepath {basepath} --foldpath"
            f" {foldpath} --workflow-buckets-name-prefix"
            f" {workflow_buckets_name_prefix} --docker-image-name"
            f" {pipeline_docker_image_name} --run-pipeline"
            f" {run_pipeline} --run-multipointing {run_multipointing} --run-folding"
            f" {run_folding} --run-stacking {run_stacking} --datpath {datpath}"
            f" --run-classification {run_classification}"
            f' --pipeline-arguments "{pipeline_arguments}"'
            f' --pipeline-config-options "{pipeline_config_options}"'
        ),
        "mode": docker.types.ServiceMode("replicated", replicas=1),
        "restart_policy": docker.types.RestartPolicy(condition="none", max_attempts=0),
        # Labels allow for easy filtering with Docker CLI
        "labels": {"type": "processing"},
        "constraints": ["node.role == manager"],
        # Will throw an error if you give two of the same bind mount paths
        # e.g. avoid double-mounting basepath and stackpath when they are the same
        "mounts": [
            f"{datpath}:{datpath}",
            "/data/chime/sps/logs:/data/chime/sps/logs",
            f"{basepath}:{basepath}",
            f"{foldpath}:{foldpath}",
            # Need this mount so container can access host machine's Docker Client
            "/var/run/docker.sock:/var/run/docker.sock",
            "/data/candidate_viewer/champss_candidate_viewer/:/data/candidate_viewer/champss_candidate_viewer/",
        ],
        # An externally created Docker Network that allows these spawned containers
        # to communicate with other containers (MongoDB, Prometheus, etc) that have
        # also been manually added to this network
        "networks": ["pipeline-network"],
    }

    docker_service_pipeline_image_clenaup = {
        "image": manager_docker_image_name,
        "name": "processing-cleanup",
        "command": "start-processing-cleanup",
        "mode": docker.types.ServiceMode("global"),
        # Labels allow for easy filtering with Docker CLI
        "labels": {"type": "processing"},
        "restart_policy": docker.types.RestartPolicy(condition="none", max_attempts=0),
        "mounts": [
            # Need this mount so container can access host machine's Docker Client
            "/var/run/docker.sock:/var/run/docker.sock"
        ],
    }

    log.info(f"Creating Docker Service: \n{docker_service_manager}")
    docker_client.services.create(**docker_service_manager)

    # log.info(f"Creating Docker Service: \n{docker_service_pipeline_image_clenaup}")
    # docker_client.services.create(**docker_service_pipeline_image_clenaup)


def remove_processing_services(signal, frame):
    """Remove all processing services."""
    log.setLevel(logging.INFO)

    docker_client = docker.from_env()

    log.info(
        f"Processing manager received signal {signal}. Once all tasks are exited, will"
        " remove all outstanding processing services"
    )

    wait_for_no_tasks_in_states(docker_swarm_running_states)

    processing_services = [
        service
        for service in docker_client.services.list()
        if "processing" in service.name and service.name != "processing-manager"
    ]

    log.info(
        "Removing processing services:"
        f" \n{[service.name for service in processing_services]}"
    )

    for service in processing_services:
        try:
            service.remove()
        except Exception as error:
            message_slack(f"Error during cleanup of processing services: {error}")

    sys.exit(0)


def start_processing_cleanup():
    """Cleanup by removing old images."""
    log.setLevel(logging.INFO)

    docker_client = docker.from_env()

    while True:
        images = {}

        for image in docker_client.images.list():
            # Only delete duplicate images that are not the most recent of them
            try:
                # If the image was pulled from a repository and not locally built:
                image_name = image.attrs["RepoDigests"][0].split("@")[0]
            except Exception:
                try:
                    # If the image was locally built:
                    image_name = image.attrs["RepoTags"][0].split(":")[0]
                except Exception as error:
                    log.info(f"Error getting image name: {error}")
                    continue

            if image_name not in images:
                images[image_name] = []

            try:
                image_id = image.attrs["Id"]
                image_timestamp_string = (
                    image.attrs["Created"].split("Z", 1)[0].split(".", 1)[0]
                )
                image_timestamp = dt.datetime.fromisoformat(image_timestamp_string)

                images[image_name].append(
                    {"name": image_name, "id": image_id, "timestamp": image_timestamp}
                )
            except Exception as error:
                log.info(f"Error getting image id or created timestamp: {error}")
                continue

        for image_name in images.keys():
            # Sort by most recently created image to oldest image
            try:
                sorted_images = sorted(
                    images[image_name],
                    key=lambda image: image["timestamp"],
                    reverse=True,
                )
            except Exception as error:
                log.info(f"Error sorting images by created timestamp: {error}")
                continue

            for image in sorted_images[1:]:
                image_name = image["name"]
                image_id = image["id"]

                try:
                    # No force=True; only delete if they are not in use
                    docker_client.images.remove(image_id)
                    log.info(f"Successfully removed old image: {image_name}:{image_id}")
                except Exception as error:
                    log.info(f"Error removing old image: {error}")
                    continue

        # Wait 10 minutes before checking to delete old images again
        time.sleep(600)


if __name__ == "__main__":
    start_processing_manager()
