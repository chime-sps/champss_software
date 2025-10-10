import datetime as dt

import click
import multiday_search.confirm_cand as confirm_cand
import multiday_search.fold_multiday as fold_multiday
from foldutils.database_utils import add_mdcand_from_candpath, add_mdcand_from_psrname
from scheduler.workflow import (
    clear_workflow_buckets,
    docker_swarm_running_states,
    schedule_workflow_job,
    wait_for_no_tasks_in_states,
)
from sps_databases import db_utils
from sps_pipeline.pipeline import default_datpath

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--candpath",
    type=str,
    default="",
    help="Path to candidate file",
)
@click.option(
    "--psr",
    type=str,
    default="",
    help="Pulsar Name",
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
    "--nday",
    default=0,
    type=int,
    help="Number of days to fold and search. Default will fold and search all available days.",
)
@click.option(
    "--start-date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=False,
    help="Start date of data to process. Default = Today in UTC",
)
@click.option(
    "--overwrite-folding",
    is_flag=True,
    help="Re-run folding even if already folded on this date.",
)
@click.option(
    "--use-workflow",
    is_flag=True,
    help="Queue folding jobs in parallel into Workflow, otherwise run locally.",
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
def main(
    candpath,
    psr,
    foldpath,
    datpath,
    db_port,
    db_host,
    db_name,
    nday,
    start_date,
    overwrite_folding,
    use_workflow,
    workflow_buckets_name_prefix,
    docker_image_name,
):

    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    if psr != "":
        fs_id = str(add_mdcand_from_psrname(psr, dt.datetime.now()))
    elif candpath != "":
        fs_id = str(add_mdcand_from_candpath(candpath, dt.datetime.now()))
    else:
        raise ValueError("Must provide either a candidate path or pulsar name")
    
    args = [
        "--fs_id", fs_id,
        "--foldpath", foldpath,
        "--datpath", datpath,
        "--db-port", str(db_port),
        "--db-name", db_name,
        "--db-host", db_host,
        "--nday", str(nday)
    ]
    if start_date:
        args += ["--start-date", start_date.strftime("%Y-%m-%d")]
    if overwrite_folding:
        args += ["--overwrite-folding"]

    if use_workflow:
        docker_service_name_prefix = "fold-multiday"

        workflow_buckets_name = (
            f"{workflow_buckets_name_prefix}-{docker_service_name_prefix}"
        )
        clear_workflow_buckets.main(
            args=["--workflow-buckets-name", workflow_buckets_name],
            standalone_mode=False,
        )

        args.append("--use-workflow")
        args += ["--docker-image-name", docker_image_name]
        args += ["--docker-service-name-prefix", docker_service_name_prefix]
        args += ["--workflow-buckets-name", workflow_buckets_name]

        fold_multiday.main(
            args=args,
            standalone_mode=False,
        )

        wait_for_no_tasks_in_states(
            docker_swarm_running_states, docker_service_name_prefix
        )

        print("Finished multiday folding, beginning the coherent search")

        docker_service_name_prefix = "multiday-confirm"
        docker_name = f"{docker_service_name_prefix}-{fs_id}"
        docker_memory_reservation = 64
        docker_mounts = [
            f"{datpath}:{datpath}",
            f"{foldpath}:{foldpath}",
        ]

        workflow_buckets_name = (
            f"{workflow_buckets_name_prefix}-{docker_service_name_prefix}"
        )
        clear_workflow_buckets.main(
            args=["--workflow-buckets-name", workflow_buckets_name],
            standalone_mode=False,
        )

        workflow_function = "multiday_search.confirm_cand.main"
        workflow_params = {
            "fs_id": fs_id,
            "db_host": db_host,
            "db_port": db_port,
            "db_name": db_name,
            "nday": nday,
            "write_to_db": True,
            "foldpath": foldpath,
        }
        workflow_tags = [
            "multiday",
            "confirm",
            fs_id,
        ]
        work_id = schedule_workflow_job(
            docker_image_name,
            docker_mounts,
            docker_name,
            docker_memory_reservation,
            workflow_buckets_name,
            workflow_function,
            workflow_params,
            workflow_tags,
        )

        wait_for_no_tasks_in_states(
            docker_swarm_running_states, docker_service_name_prefix
        )

        # Can add Slack alerts here
        print("Finished multiday search")
        foldresults_dict = {"coherentsearch_work_id": work_id}
        return foldresults_dict, [], []
    else:
        fold_multiday.main(
            args=args,
            standalone_mode=False,
        )

        print("Finished multiday folding, beginning the coherent search")
        confirm_cand.main(
            args=[
                "--fs_id",
                fs_id,
                "--db-port",
                db_port,
                "--db-name",
                db_name,
                "--db-host",
                db_host,
                "--nday",
                nday,
                "--write-to-db",
                "--foldpath",
                foldpath,
            ],
            standalone_mode=False,
        )

        # Can add Slack alerts here
        print("Finished multiday search")


if __name__ == "__main__":
    main()
