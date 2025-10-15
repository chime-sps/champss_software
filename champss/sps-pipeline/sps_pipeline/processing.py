"""Functions to enable automatic processing."""

import atexit
import datetime as dt
import logging
import os
import signal
import sys
import time
import traceback
from bson.objectid import ObjectId
from functools import partial
from glob import glob
from multiprocessing import Pool

import click
import docker
import numpy as np
import pytz
from beamformer.strategist.strategist import PointingStrategist
from folding.filter_mpcandidates import Filter
from scheduler.utils import convert_date_to_datetime
from scheduler.workflow import (  # docker_swarm_pending_states,
    clear_workflow_buckets,
    docker_swarm_running_states,
    get_work_from_results,
    message_slack,
    schedule_workflow_job,
    wait_for_no_tasks_in_states,
)
from sps_databases import db_api, db_utils, models
from sps_pipeline.pipeline import default_datpath, main
from sps_pipeline.utils import get_pointings_from_list

log = logging.getLogger()


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
def find_all_folding_processes(date, db_host, db_port, db_name, basepath, foldpath):
    """Find all available folding processes for a given date."""
    log.setLevel(logging.INFO)

    db = db_utils.connect(host=db_host, port=db_port, name=db_name)

    log.info(f"Filtering candidates for {date}")

    date = convert_date_to_datetime(date)

    Filter(
        cand_obs_date=date,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        write_to_db=True,
        basepath=basepath,
        foldpath=foldpath,
    )

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

    return {"info": info}, [], []


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
    "--basepath",
    default="./",
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
    basepath,
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
            f"{basepath}:{basepath}",
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
):
    """Process all unprocessed processes in the database for a given range."""
    date = convert_date_to_datetime(date)

    log.setLevel(logging.INFO)
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
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
        time_passed = dt.datetime.now(dt.timezone.utc) - (date + dt.timedelta(days=1))
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

    process_ids = []
    for process_index, process_dict in enumerate(all_processes):
        process = models.Process.from_db(process_dict)
        process_ids.append(ObjectId(process._id))
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
                    process.ra,
                    f" {process.dec}",
                    "all",
                ]
            )

            if workflow_buckets_name == "":
                log.info(f"Running command: run-pipeline {' '.join(cmd_string_list)}")
            if not dry_run:
                if workflow_buckets_name:
                    formatted_ra = f"{process.ra:.02f}"
                    formatted_dec = f"{process.dec:.02f}"
                    formatted_maxdm = f"{process.maxdm:.02f}"
                    formatted_date = process.date

                    docker_memory_reservation = process.ram_requirement
                    docker_threads_needed = int(docker_memory_reservation / 3)
                    docker_image = docker_image_name
                    docker_mounts = [
                        f"{datpath}:{datpath}",
                        f"{basepath}:{basepath}",
                    ]
                    docker_name = (
                        f"{docker_service_name_prefix}-{formatted_ra}-"
                        f"{formatted_dec}-{formatted_maxdm}-{formatted_date}"
                    )

                    workflow_function = "sps_pipeline.pipeline.main"
                    workflow_params = {
                        "date": process.date,
                        "stack": run_stacking,
                        "fdmt": True,
                        "rfi_beamform": True,
                        "plot": True,
                        "plot_threshold": 8.0,
                        "ra": process.ra,
                        "dec": f" {process.dec}",
                        "components": ["all"],
                        "num_threads": docker_threads_needed,
                        "db_port": db_port,
                        "db_host": db_host,
                        "db_name": db_name,
                        "basepath": basepath,
                        "stackpath": stackpath,
                        "datpath": datpath,
                        # Run Pyroscope profiling every 100th job
                        # "using_pyroscope": True if process_index % 100 == 0 else False,
                        "using_pyroscope": False,
                        "using_docker": True,
                        "config_options": pipeline_config_options,
                    }
                    if pipeline_arguments != "":
                        split_args = pipeline_arguments.split("--")
                        for arg_string in split_args:
                            arg_string = arg_string.strip()
                            if arg_string != "":
                                arg_count = len(arg_string.split(" "))
                                if arg_count > 1:
                                    argument, value = arg_string.split(" ", 1)
                                    workflow_params[argument] = (value,)
                                else:
                                    log.error(
                                        "Flags not implimented yet. Reformated your option to --option_python_name True"
                                    )

                    workflow_tags = [
                        "pipeline",
                        formatted_ra,
                        formatted_dec,
                        formatted_maxdm,
                        formatted_date,
                    ]

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
                else:
                    main(cmd_string_list, standalone_mode=False)

                    log.info(
                        f"Finished processing pointing ({process.ra}, {process.dec})"
                        f" for date {process.date}"
                    )
        except Exception as error:
            traceback.print_exc()
            log.error(error)
            if not dry_run:
                db_api.update_process(process.id, {"status": 3})

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
    "--run-classfication",
    default=True,
    type=bool,
    help="To run the classfication phase of processing or not.",
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
):
    """Manager function containing the multiple processing steps."""
    atexit.register(remove_processing_services, None, None)
    signal.signal(signal.SIGINT, remove_processing_services)
    signal.signal(signal.SIGQUIT, remove_processing_services)
    signal.signal(signal.SIGABRT, remove_processing_services)
    signal.signal(signal.SIGTERM, remove_processing_services)

    start_date = convert_date_to_datetime(start_date)

    log.setLevel(logging.INFO)

    db = db_utils.connect(host=db_host, port=db_port, name=db_name)

    start_date = start_date.replace(tzinfo=pytz.UTC)

    date_to_process = start_date

    number_of_days_processed = 0

    def loop_condition():
        if number_of_days != -1:
            # If number_of_days is not -1, then we want to run for a specific number of days
            return number_of_days_processed < number_of_days
        else:
            # If number_of_days is -1 (default if none given), then we want to run forever
            return True

    while loop_condition():
        try:
            present_date = dt.datetime.now(dt.timezone.utc)
            yesterday_date = present_date - dt.timedelta(days=1)

            date_string = date_to_process.strftime("%Y/%m/%d")

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

                if len(processes["unfinished_processes"]) == 0:
                    message_slack(
                        f"No unfinished processes found for {date_string}. Will progress to"
                        " next day"
                    )
                    number_of_days_processed = number_of_days_processed + 1
                    date_to_process = date_to_process + dt.timedelta(days=1)
                    continue

                present_date = dt.datetime.now(dt.timezone.utc)

                time_passed = present_date - (date_to_process + dt.timedelta(days=1))

                start_time_of_processing = time.time()

                docker_service_name_prefix = "pipeline"

                workflow_buckets_name = (
                    f"{workflow_buckets_name_prefix}-{docker_service_name_prefix}"
                )
                clear_workflow_buckets.main(
                    args=["--workflow-buckets-name", workflow_buckets_name],
                    standalone_mode=False,
                )
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
                process_ids = run_all_pipeline_processes.main(
                    args=pipeline_args,
                    standalone_mode=False,
                )

                wait_for_no_tasks_in_states(
                    docker_swarm_running_states, docker_service_name_prefix
                )

                end_time_of_processing = time.time()
                processed = list(db.processes.find({"_id": {"$in": process_ids}}))
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
                rfi_processeses = [
                    proc
                    for proc in completed_processes
                    if proc["quality_label"] is False
                ]
                db["processes"].count_documents(
                    {"date": date_string, "status": 2, "is_in_stack": False}
                )

                obs_ids = [proc["obs_id"] for proc in completed_processes]
                observations = list(db.observations.find({"_id": {"$in": obs_ids}}))
                mean_detections = np.nanmean(
                    [obs["num_detections"] for obs in observations]
                )

                # Get the execution time and nchan per process for the day
                time_per_nchan = {
                    1024: {"total": 0, "count": 0},
                    2048: {"total": 0, "count": 0},
                    4096: {"total": 0, "count": 0},
                    8192: {"total": 0, "count": 0},
                    16384: {"total": 0, "count": 0},
                }

                for process in processed:
                    nchan = process["nchan"]
                    execution_time = process["process_time"]

                    if execution_time is not None and nchan in time_per_nchan:
                        time_per_nchan[nchan]["total"] += execution_time
                        time_per_nchan[nchan]["count"] += 1

                slack_message = (
                    f"For {date_string}:\n{len(completed_processes)} /"
                    f" {len(processed)} finished successfully for the day\nOf those,"
                    f" {len(rfi_processeses)} were rejected by quality metrics\nMean number"
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
                            f"\nMean process time for nchan {nchan}:"
                            f" {mean_time_of_nchan} seconds"
                            f" ({mean_time_of_nchan / 60:.2f} minutes)"
                        )

                message_slack(slack_message)
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
                work_id = schedule_workflow_job(
                    docker_image=docker_image_name,
                    docker_mounts=[
                        f"{datpath}:{datpath}",
                        f"{basepath}:{basepath}",
                    ],
                    docker_name=f"{docker_service_name_prefix}-{date_string}",
                    docker_memory_reservation=100,
                    workflow_buckets_name=workflow_buckets_name,
                    workflow_function="sps_multi_pointing.mp_pipeline.cli",
                    workflow_params={
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
                    },
                    workflow_tags=["mp", date_string],
                    timeout=mp_timeout,
                )

                wait_for_no_tasks_in_states(
                    docker_swarm_running_states,
                    docker_service_name_prefix,
                    timeout=mp_timeout,
                )

                # Need to wait a few seconds for results to propogate to Workflow
                time.sleep(5)

                work_result = get_work_from_results(
                    workflow_results_name=workflow_buckets_name,
                    work_id=work_id,
                    failover_to_buckets=True,
                )

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
            # End of multi-pointing phase

            # Start of classification phase
            if run_classification:
                message_slack(f"Running classfication for {date_string}")

                docker_service_name_prefix = "class"

                workflow_buckets_name = (
                    f"{workflow_buckets_name_prefix}-{docker_service_name_prefix}"
                )
                clear_workflow_buckets.main(
                    args=["--workflow-buckets-name", workflow_buckets_name],
                    standalone_mode=False,
                )
                class_timeout = 60 * 60 * 60
                work_id = schedule_workflow_job(
                    docker_image="sps-archiver1.chime:5000/champss_classification:stable",
                    docker_mounts=[
                        f"{datpath}:{datpath}",
                        f"{basepath}:{basepath}",
                    ],
                    docker_name=f"{docker_service_name_prefix}-{date_string}",
                    docker_memory_reservation=100,
                    workflow_buckets_name=workflow_buckets_name,
                    workflow_function="champss_classification.pytorch_model.classify.load_model_and_classify_mp_csv",
                    workflow_params={
                        "csv": work_result["csv_file"],
                    },
                    workflow_tags=["class", date_string],
                    timeout=class_timeout,
                )

                wait_for_no_tasks_in_states(
                    docker_swarm_running_states,
                    docker_service_name_prefix,
                    timeout=class_timeout,
                )
            # End of classification phase

            # Start of folding phase
            if run_folding:
                processes, [], [] = find_all_folding_processes.main(
                    args=[
                        "--date",
                        date_to_process,
                        "--db-host",
                        db_host,
                        "--db-port",
                        db_port,
                        "--db-name",
                        db_name,
                        "--basepath",
                        basepath,
                        "--foldpath",
                        foldpath,
                    ],
                    standalone_mode=False,
                )

                docker_service_name_prefix = "fold"

                workflow_buckets_name = (
                    f"{workflow_buckets_name_prefix}-{docker_service_name_prefix}"
                )
                clear_workflow_buckets.main(
                    args=["--workflow-buckets-name", workflow_buckets_name],
                    standalone_mode=False,
                )

                run_all_folding_processes.main(
                    args=[
                        "--date",
                        date_to_process,
                        "--db-host",
                        db_host,
                        "--db-port",
                        db_port,
                        "--db-name",
                        db_name,
                        "--basepath",
                        basepath,
                        "--foldpath",
                        foldpath,
                        "--processes",
                        processes["info"],
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

                message_slack(f"Candidate folding for {date_string} complete")
            # End of folding phase

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

    log.info(f"Creating Docker Service: \n{docker_service_pipeline_image_clenaup}")
    docker_client.services.create(**docker_service_pipeline_image_clenaup)


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
