import json

import click
from sps_databases import db_api, db_utils, models
from sps_pipeline.workflow import (
    docker_swarm_pending_states,
    docker_swarm_running_states,
    wait_for_no_tasks_in_states,
    message_slack,
    schedule_workflow_job,
    clear_workflow_buckets,
    get_work_from_results
)

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
    "--day-threshold",
    default=20,
    type=int,
    help="Number of days that are used as threshold.",
)
@click.option(
    "--options",
    default=None,
    type=str,
    help="Options used for search. Write a string in the form of a python dict",
)
@click.option(
    "--run-name",
    default="test_run",
    type=str,
    help="Name of the run",
)
def find_monthly_search_commands(db_port, db_host, db_name, day_threshold, options, run_name
):

    db = db_utils.connect(port=db_port, host=db_host, name=db_name)
    all_stacks = list(db.ps_stacks.find({"num_days_month": {"$gte": day_threshold}}))

    all_commands = []
    options_dict = json.loads(options)
    for stack in all_stacks:
        stack_obj = models.PsStack.from_db(stack)
        pointing = stack_obj.pointing(db)
        arguments = {"ra": pointing.ra, "dec": pointing.dec, "components": "search-monthly",  "db_port": db_port, "db_name": db_name, "db_host": db_host, **options_dict}
        all_commands.append({"arguments": arguments, "scheduled" : False, "maxdm": pointing.maxdm})

    with open(f'{run_name}.json', 'w') as file:
        file.write(json.dumps(all_commands, indent=4))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--docker-image-name",
    default="chimefrb/champss_software:latest",
    type=str,
    help="Which Docker Image name to use.",
)
@click.option(
    "--docker-password",
    type=str,
    help="Password to login to chimefrb DockerHub (hint: frbadmin's common password).",
)
@click.option(
    "--command-file",
    type=str,
    help="File from which the files are executed.",
)
def execute_monthly_search_commands( docker_image_name,
    docker_password, command_file,
):

    with open(command_file, 'r') as file:
        all_commands = json.load(file)

    docker_mounts = [
                        "/data/chime/sps/raw:/data/chime/sps/raw",
                        "/data/chime/sps/sps_processing:/data/chime/sps/sps_processing",
                    ]
    workflow_function = "sps_pipeline.pipeline.stack_and_search"
    workflow_name = "champss-stack-search"
    clear_workflow_buckets(
                [
                    "--workflow-buckets-name",
                    workflow_name
                ],
                standalone_mode=False
            )

    for command_dict in all_commands:
        if command_dict["scheduled"] is False:
            formatted_ra = f'{command_dict["arguments"]["ra"]:.02f}'.replace(".", "_")
            formatted_dec = f'{command_dict["arguments"]["dec"]:.02f}'.replace(".", "_")
            docker_name = f"stack_search_{formatted_ra}_{formatted_dec}"
            docker_memory_reservation = (70 + ((command_dict["maxdm"] / 100) * 4))
            workflow_params = command_dict["arguments"]
            workflow_tags = ["stack_search", "monthly_search", formatted_ra, formatted_dec]
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
            command_dict["scheduled"] = True
            with open(command_file, 'w') as file:
                file.write(json.dumps(all_commands, indent=4))
