import datetime as dt
import logging

import click
import multiday_search.confirm_cand as confirm_cand
import multiday_search.fold_multiday as fold_multiday
from foldutils.database_utils import add_mdcand_from_candpath
from sps_databases import db_api, db_utils, models
from sps_pipeline.workflow import (
    docker_swarm_pending_states,
    docker_swarm_running_states,
    schedule_workflow_job,
    wait_for_no_tasks_in_states,
)

log = logging.getLogger()
log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--candpath",
    type=str,
    default="",
    help="Path to candidate file",
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver",
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
    "--docker-image-name",
    default="chimefrb/champss_software:test",
    type=str,
    help="Which Docker Image name to use.",
)
@click.option(
    "--docker-password",
    default="",
    type=str,
    help="chimefrb DockerHub private registry password",
)
@click.option(
    "--workflow-name",
    default="champss-processing",
    type=str,
    help="Which Worklow DB to create/use.",
)
def main(
    candpath,
    db_port,
    db_host,
    db_name,
    docker_image_name,
    docker_password,
    workflow_name,
):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    fs_id = str(add_mdcand_from_candpath(candpath, dt.datetime.now()))
    print(fs_id)

    docker_name_prefix = "fold-multiday"
    fold_multiday.main(
        [
            "--fs_id",
            fs_id,
            "--db-port",
            db_port,
            "--db-name",
            db_name,
            "--db-host",
            db_host,
            "--docker-image-name",
            docker_image_name,
            "--docker-password",
            docker_password,
            "--docker-name-prefix",
            docker_name_prefix,
            "--workflow-name",
            workflow_name,
        ],
        standalone_mode=False,
    )

    # To Do (Chris): add docker_name_prefix attribute to wait_for_no_tasks_in_states
    # wait_for_no_tasks_in_states(docker_swarm_pending_states, docker_name_prefix)
    # wait_for_no_tasks_in_states(docker_swarm_running_states, docker_name_prefix)
    wait_for_no_tasks_in_states(docker_swarm_pending_states)
    wait_for_no_tasks_in_states(docker_swarm_running_states)

    print("Finished multiday folding, beginning coherent search")

    docker_name_prefix = "multiday"
    docker_name = f"{docker_name_prefix}-{fs_id}"
    docker_memory_reservation = 64
    docker_mounts = [
        "/data/chime/sps/raw:/data/chime/sps/raw",
        "/data/chime/sps/archives:/data/chime/sps/archives",
    ]

    workflow_function = "multiday_search.confirm_cand.main"
    workflow_params = {
        "fs_id": fs_id,
        "db_host": db_host,
        "db_port": db_port,
        "db_name": db_name,
        "write_to_db": True,
    }
    workflow_tags = [
        "multiday",
        fs_id,
    ]
    schedule_workflow_job(
        docker_image_name,
        docker_mounts,
        docker_name,
        docker_memory_reservation,
        docker_password,
        workflow_name,
        workflow_function,
        workflow_params,
        workflow_tags,
    )

    wait_for_no_tasks_in_states(docker_swarm_pending_states)
    wait_for_no_tasks_in_states(docker_swarm_running_states)

    # Can add Slack alerts here


if __name__ == "__main__":
    main()
