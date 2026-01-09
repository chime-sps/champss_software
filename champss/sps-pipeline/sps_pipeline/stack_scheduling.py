import json
import os

import click
from scheduler.workflow import (
    clear_workflow_buckets,
    schedule_workflow_job,
)
from sps_databases import db_utils, models


# @click.command(context_settings={"help_option_names": ["-h", "--help"]})
# @click.option(
#     "--db-port",
#     default=27017,
#     type=int,
#     help="Port used for the mongodb database.",
# )
# @click.option(
#     "--db-host",
#     default="sps-archiver1",
#     type=str,
#     help="Host used for the mongodb database.",
# )
# @click.option(
#     "--db-name",
#     default="sps-processing",
#     type=str,
#     help="Name used for the mongodb database.",
# )
# @click.option(
#     "--day-threshold",
#     default=20,
#     type=int,
#     help="Number of days that are used as threshold.",
# )
# @click.option(
#     "--options",
#     default=None,
#     type=str,
#     help="Options used for search. Write a string in the form of a python dict",
# )
# @click.option(
#     "--run-name",
#     default="test_run",
#     type=str,
#     help="Name of the run",
# )
def find_monthly_search_commands(db_port, db_host, db_name, basepath, day_threshold=10):
    db = db_utils.connect(port=db_port, host=db_host, name=db_name)
    all_stacks = list(db.ps_stacks.find({"num_days_month": {"$gte": day_threshold}}))

    all_commands = []
    # options_dict = json.loads(options)
    for stack in all_stacks:
        stack_obj = models.PsStack.from_db(stack)
        pointing = stack_obj.pointing(db)
        ram_requirement = pointing.ram_requirement
        tier_name, tier_limit = pointing.tier
        threads_reserved = int(tier_limit / 3)
        arguments = {
            "ra": pointing.ra,
            "dec": f" {pointing.dec}",
            "components": "search-monthly",
            "db_port": db_port,
            "db_name": db_name,
            "db_host": db_host,
            "num_threads": threads_reserved,
            "components": ["search-monthly"],
            "known_source_threshold": 10,
            "cand_path": basepath,
        }
        all_commands.append(
            {
                "arguments": arguments,
                "maxdm": pointing.maxdm,
                "stackpath": stack_obj.datapath_month,
                "length": pointing.length,
                "ram_requirement": ram_requirement,
                "tier_name": tier_name,
            }
        )

    return all_commands


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--docker-image-name",
    default="sps-archiver1.chime:5000/champss_software:latest",
    type=str,
    help="Which Docker Image name to use.",
)
@click.option(
    "--command-file",
    type=str,
    help="File from which the files are executed.",
)
def execute_monthly_search_commands(
    docker_image_name,
    command_file,
):
    with open(command_file) as file:
        all_commands = json.load(file)
    cand_path = all_commands[0]["arguments"]["cand_path"]
    all_stack_folders = [
        os.path.dirname(command["stackpath"]) for command in all_commands
    ]
    stack_folders = set(all_stack_folders)

    docker_mounts = [
        f"{cand_path}:{cand_path}",
    ]
    for folder in stack_folders:
        docker_mounts.append(f"{folder}:{folder}")
    workflow_function = "sps_pipeline.pipeline.stack_and_search"
    workflow_buckets_name = "champss-stack-search"
    clear_workflow_buckets.main(
        args=["--workflow-buckets-name", workflow_buckets_name],
        standalone_mode=False,
    )
    for command_dict in all_commands:
        if command_dict["scheduled"] is False:
            formatted_ra = f"{command_dict['arguments']['ra']:.02f}"
            formatted_dec = f"{command_dict['arguments']['dec']:.02f}"
            docker_name = f"stack_search_{formatted_ra}_{formatted_dec}"
            docker_memory_reservation = 40 + (
                (command_dict["maxdm"] / 100) * 4
            ) * 2 ** (command_dict["length"] // 2**20)
            workflow_params = command_dict["arguments"]
            workflow_tags = [
                "stack_search",
                "monthly_search",
                formatted_ra,
                formatted_dec,
            ]
            schedule_workflow_job(
                docker_image_name,
                docker_mounts,
                docker_name,
                docker_memory_reservation,
                workflow_buckets_name,
                workflow_function,
                workflow_params,
                workflow_tags,
            )
            command_dict["scheduled"] = True
            with open(command_file, "w") as file:
                file.write(json.dumps(all_commands, indent=4))
