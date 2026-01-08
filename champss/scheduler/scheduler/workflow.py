import datetime as dt
import json
import logging
import os
import re
import time
import threading

import click
import docker
from slack_sdk import WebClient
from workflow.definitions.work import Work
from workflow.http.context import HTTPContext
from bson.objectid import ObjectId

log = logging.getLogger()

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

perpetual_processing_services = ["processing-manager", "processing-cleanup"]

# Sometimes a Docker Swarm task gets stuck in pending/running state
# indefinitely for unknown reasons...
task_timeout_seconds = 60 * 120  # 120 minutes


def message_slack(
    slack_message,
    slack_channel="#slow-pulsar-alerts",
    slack_token="",
):
    log.setLevel(logging.INFO)
    log.info(f"Sending to Slack: \n{slack_message}")

    if not slack_token:
        slack_token = os.getenv("SLACK_APP_TOKEN", "")

    slack_client = WebClient(token=slack_token)
    try:
        slack_request = slack_client.chat_postMessage(
            channel=slack_channel,
            text=slack_message,
        )
    except Exception as error:
        log.info(error)


def save_container_logs(service):
    """
    Save the logs of a given container service to a file.

    This function retrieves the logs from the specified container service and writes them to a log file.
    The log file is saved in the directory "/data/chime/sps/logs/services/" with the name of the service.

    Parameters:
        service: The container service object from which to retrieve logs. The service object must have a `logs` method
                 that returns a generator of log chunks.

    Raises:
        Exception: If there is an error while writing the logs to the file, it logs the error and skips gracefully.
    """

    try:
        docker_client = docker.from_env()

        clean_pattern = re.compile(r"^\S+\s+")

        log_text = ""
        log_generator = service.logs(
            details=True,
            stdout=True,
            stderr=True,
            follow=False,
        )
        for log_chunk in log_generator:
            log_line = log_chunk.decode("utf-8").strip()
            clean_line = clean_pattern.sub("", log_line)
            log_text += clean_line + "\n"

        path = f"/data/chime/sps/logs/services/{service.name}.log"

        with open(path, "w") as file:
            file.write(log_text)

        for task in service.tasks():
            task_id = task["ID"]
            task_inspection = docker_client.api.inspect_task(task_id)
            task_inspection_json = json.dumps(task_inspection, indent=4)

        with open(path, "a") as file:
            file.write(task_inspection_json)

    except Exception as error:
        log.info(f"Error dumping logs at {path}: {error} (will skip gracefully).")


def get_service_created_at_datetime(service):
    """
    Extracts and parses the 'CreatedAt' attribute from a service object to a datetime
    object.

    Parameters:
        service: An object that contains a dictionary attribute 'attrs' with a key 'CreatedAt'.
                 The 'CreatedAt' value is expected to be a string in the format "%Y-%m-%dT%H:%M:%S".

    Returns:
        datetime: A datetime object representing the parsed 'CreatedAt' value.
                  Returns None if there is an error during parsing.
    """

    try:
        datetime = dt.datetime.strptime(
            service.attrs["CreatedAt"].split(".")[0], "%Y-%m-%dT%H:%M:%S"
        )
        return datetime
    except Exception as error:
        log.info(
            f"Error parsing CreatedAt for service {service}: {error} (will skip"
            " gracefully)."
        )
        return None


def wait_for_no_tasks_in_states(
    states_to_wait_for_none, docker_service_name_prefix="", timeout=task_timeout_seconds
):
    """
    Waits for no tasks to be in the specified states within Docker Swarm services.

    This function monitors Docker Services and waits until there are no tasks
    in the specified states (`states_to_wait_for_none`). It handles services with
    names containing the `docker_service_name_prefix` and excludes perpetual processing
    services. The function logs relevant information and errors, and removes services
    that are finished or have been running/pending for too long.

    Parameters:
        states_to_wait_for_none (list): List of task states to wait for none to be in.
        docker_service_name_prefix (str, optional): Prefix to filter Docker service names. Defaults to "".

    Returns:
        None
    """

    log.setLevel(logging.INFO)

    docker_client = docker.from_env()

    is_task_in_state = True

    # For pending:
    # Docker Swarm can be freeze if too many tasks are in pending state concurrently.
    # Additionally, Docker Swarm dequeues pending jobs in random order, so we need to
    # only have one pending job at a time, to maintain ordering
    # For running:
    # It's helpful for Slack messages to wait for all running processes to finish
    # so they have all the final process information
    while is_task_in_state is True:
        is_task_in_state = False

        # Re-fetch Docker Swarm Service states
        try:
            processing_services = sorted(
                [
                    service
                    for service in docker_client.services.list()
                    if "processing" in service.name
                    and docker_service_name_prefix in service.name
                    and service.name not in perpetual_processing_services
                ],
                # Sort from oldest to newest to find finished services to remove
                key=get_service_created_at_datetime,
            )
            processing_services = [
                service for service in processing_services if service is not None
            ]
        except Exception as error:
            log.info(
                f"Error fetching Docker Swarm services: {error}."
                f" Will stop waiting for tasks in states {states_to_wait_for_none}."
            )
            return

        for service in processing_services:
            service_created_at = get_service_created_at_datetime(service)

            if service_created_at is None:
                continue

            try:
                for task in service.tasks():
                    task_state = task["Status"]["State"]
                    task_id = task["ID"]

                    if task_state in docker_swarm_finished_states:
                        log.info(
                            f"Removing finished service {service.name} in state"
                            f" {task_state}."
                        )

                        try:
                            save_container_logs(service)
                            service.remove()
                        except Exception as error:
                            log.info(
                                f"Error removing service {service.name}: {error} (will"
                                " skip gracefully)."
                            )
                    elif task_state in docker_swarm_running_states:
                        if (dt.datetime.now() - service_created_at).total_seconds() > (
                            timeout * 2
                        ):
                            log.info(
                                f"Service {service.name} has been ruuning for more than"
                                f" {(timeout * 2) / 60} minutes in state"
                                f" {task_state}, implying frozen task on 1st or 2nd"
                                " final Workflow runner attempt. Will remove service."
                            )

                            try:
                                save_container_logs(service)
                                service.remove()
                            except Exception as error:
                                log.info(
                                    f"Error removing service {service.name}:"
                                    f" {error} (will skip gracefully)."
                                )
                        # Task in state running, and we want to wait for no running states
                        # Loop will continue
                        elif states_to_wait_for_none == docker_swarm_running_states:
                            is_task_in_state = True
                            break
                    elif task_state in docker_swarm_pending_states:
                        if (dt.datetime.now() - service_created_at).total_seconds() > (
                            timeout * 2
                        ):
                            log.info(
                                f"Service {service.name} has been pending for more than"
                                f" {(timeout * 2) / 60} minutes in state"
                                f" {task_state}, implying failed Docker task"
                                " scheduling. Will remove service."
                            )

                            try:
                                save_container_logs(service)
                                service.remove()
                            except Exception as error:
                                log.info(
                                    f"Error removing service {service.name}:"
                                    f" {error} (will skip gracefully)."
                                )
                        # Task in state pending, and we want to wait for no pending states
                        # Loop will continue
                        elif states_to_wait_for_none == docker_swarm_pending_states:
                            is_task_in_state = True
                            break
            except Exception as error:
                log.info(
                    f"Error checking tasks for service {service}: {error} (will"
                    " skip gracefully)."
                )

            if is_task_in_state is True:
                break


def wait_until_service_not_pending(service_id, timeout=0.5):
    docker_client = docker.from_env()
    pending = True
    while pending:
        try:
            service_tasks = docker_client.services.get(service_id).tasks()
            pending = service_tasks[0]["Status"]["State"] in docker_swarm_pending_states
            if pending:
                time.sleep(timeout)
        except (docker.errors.NotFound, IndexError) as e:
            return None
    return service_tasks[0]["Status"]["State"]


def remove_finished_service(service_id, timeout=10):
    docker_client = docker.from_env()
    finished = False
    while not finished:
        time.sleep(timeout)
        try:
            service_tasks = docker_client.services.get(service_id).tasks()
            finished = (
                service_tasks[0]["Status"]["State"] in docker_swarm_finished_states
            )
            if finished:
                try:
                    docker_client.services.get(service_id).remove()
                except:
                    pass
        except (docker.errors.NotFound, IndexError) as e:
            return None
    return service_tasks[0]["Status"]["State"]


def schedule_workflow_job(
    docker_image,
    docker_mounts,
    docker_name,
    docker_memory_reservation,
    workflow_buckets_name,
    workflow_function,
    workflow_params,
    workflow_tags,
    workflow_user="CHAMPSS",
    timeout=task_timeout_seconds,
    return_service_id=False,
):
    """
    Deposit Work and scale Docker Service, as node resources are free.

    Additionally calls other functions to handle queuing, cleanup finished Docker,
    Services, and output logs to /data/chime/sps/logs/services/<service_name>.log.

    Parameters:
    docker_image (str): Custom name of your Docker Image, in form sps-archiver1.chime:5000/<repo>:<branch>
    docker_mounts (list): List of mount paths for your Docker Service, in form source:target
    docker_name (str): Custom name for your Docker service.
    docker_memory_reservation (float): Memory reservation for your Docker Service in GB.
    workflow_buckets_name (str): Custom name your the Workflow job to be deposited in.
    workflow_function (str): Path to your function, in form "module.function".
    workflow_params (dict): Parameters to your function, in form {"param1": "value1", "param2": "value2"}.
    workflow_tags (list): Custom  tags for your Workflow job to be filtered by.
    workflow_user (str): Name of the user who shall own the Workflow job.
    timeout (int): Timeout of the workflow task
    return_service_id (bool): Also return service id and not only work id

    Returns:
    str: The ID of the deposited Workflow job if successful, otherwise an empty string.
    """
    log.setLevel(logging.INFO)

    docker_client = docker.from_env()

    workflow_site = "chime"

    tag_id = ObjectId().__str__()
    workflow_tags.append(tag_id)

    try:
        work = Work(
            pipeline=workflow_buckets_name, site=workflow_site, user=workflow_user
        )

        work.function = workflow_function
        work.parameters = workflow_params
        work.tags = workflow_tags
        work.config.archive.results = True
        work.config.archive.plots = "bypass"
        work.config.archive.products = "bypass"
        work.retries = 1
        work.timeout = timeout

        work_id = work.deposit(return_ids=True)

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
                docker.types.Mount(
                    target=mount_target, source=mount_source, type="bind"
                )
            )
        service_name = f"processing-{docker_name.replace('.', '_').replace('/', '')}"
        docker_service = {
            "image": docker_image,
            # Can't have dots or slashes in Docker Service names
            # All Docker Services made with this function will be prefixed with "processing-"
            "name": service_name,
            # Use one-shot Workflow runners since we need a new container per process for unique memory reservations
            # (we currently only use Workflow as a wrapper for its additional features, e.g. frontend)
            "command": (
                "workflow run"
                f" {workflow_buckets_name} --tag {tag_id} --site"
                f" {workflow_site} --lives 1 --sleep 1"
            ),
            # Using template Docker variables as in-container environment variables
            # that allow us this access out-of-container information
            "env": ["CONTAINER_NAME={{.Task.Name}}", "NODE_NAME={{.Node.Hostname}}"],
            # This is neccessary to allow Pyroscope (py-spy) to work in Docker
            # 'cap_add': ['SYS_PTRACE'],
            # Again, using one-shot Docker Service tasks too
            "mode": docker.types.ServiceMode("replicated", replicas=1),
            "restart_policy": docker.types.RestartPolicy(
                condition="none", max_attempts=0
            ),
            # Labels allow for easy filtering with Docker CLI
            "labels": {"type": "processing"},
            # The labels on the Docker Nodes are pre-empetively set beforehand
            "constraints": ["node.labels.compute == true"],
            # Must be in bytes
            "resources": docker.types.Resources(
                mem_reservation=int(docker_memory_reservation * 1e9)
            ),
            # Will throw an error if you give two of the same bind mount paths
            # e.g. avoid double-mounting basepath and stackpath when they are the same
            "mounts": docker_volumes,
            # An externally created Docker Network that allows these spawned containers
            # to communicate with other containers (MongoDB, Prometheus, etc) that are
            # also manually added to this network
            "networks": ["pipeline-network"],
        }

        log.info(f"Creating Docker Service: \n{docker_service}")

        service = docker_client.services.create(**docker_service)
        service_id = service.attrs["ID"]
        status = wait_until_service_not_pending(service_id)
        remove_service_thread = threading.Thread(
            target=remove_finished_service, args=(service_id,)
        )
        remove_service_thread.start()
        if return_service_id:
            return work_id[0], service_id
        else:
            return work_id[0]
    except Exception as error:
        log.info(
            f"Failed to deposit Work or create Docker Service: {error}. "
            "Will not schedule this task."
        )

        try:
            work.delete()
        except Exception as error:
            log.info(f"Failed to delete dangling Work: {error}.")

        if return_service_id:
            return "", ""
        else:
            return ""


@click.command()
@click.option(
    "--workflow-buckets-name",
    type=str,
    required=True,
    help="Name of the Workflow Buckets collection to delete",
)
def clear_workflow_buckets(workflow_buckets_name):
    """Function to empty given SPS Buckets collection on-site."""
    try:
        http_context = HTTPContext()
        buckets_api = http_context.buckets
        # Bucket API only allows 100 deletes per request
        buckets_list = buckets_api.view(
            query={"pipeline": workflow_buckets_name},
            limit=100,
            projection={"id": True},
        )
        log.info(f"Initial buckets entries: {buckets_list}")
        while len(buckets_list) != 0:
            buckets_list = buckets_api.view(
                query={"pipeline": workflow_buckets_name},
                limit=100,
                projection={"id": True},
            )
            bucket_ids_to_delete = [bucket["id"] for bucket in buckets_list]
            log.info(f"Will delete buckets entries with ids: {bucket_ids_to_delete}")
            buckets_api.delete_ids(ids=bucket_ids_to_delete)
            buckets_list = buckets_api.view(
                query={"pipeline": workflow_buckets_name},
                limit=100,
                projection={"id": True},
            )
        log.info(f"Final buckets entries: {buckets_list}")
    except Exception as error:
        print(error)


@click.command()
@click.option(
    "--workflow-results-name",
    type=str,
    required=True,
    help="Name of the Workflow Results collection to delete",
)
def clear_workflow_results(workflow_results_name):
    """Function to empty given SPS Results collection on-site."""
    log.setLevel(logging.INFO)

    try:
        http_context = HTTPContext()
        results_api = http_context.results
        # Results API only allows 10 deletes per request
        results_list = results_api.view(
            query={}, pipeline=workflow_results_name, limit=10, projection={"id": 1}
        )
        log.info(f"Initial results entries: {results_list}")
        while len(results_list) != 0:
            result_ids_to_delete = [result["id"] for result in results_list]
            log.info(f"Will delete results entries with ids: {result_ids_to_delete}")
            results_api.delete_ids(
                pipeline=workflow_results_name, ids=result_ids_to_delete
            )
            results_list = results_api.view(
                query={}, pipeline=workflow_results_name, limit=10, projection={"id": 1}
            )
        log.info(f"Final results entries: {results_list}")
    except Exception:
        pass


def get_work_from_buckets(workflow_buckets_name, work_id, failover_to_results):
    log.setLevel(logging.INFO)

    http_context = HTTPContext()
    workflow_buckets_api = http_context.buckets
    workflow_buckets_list = workflow_buckets_api.view(
        query={"pipeline": workflow_buckets_name, "id": work_id},
        limit=1,
        # projection={"results": 1},
    )

    log.info(f"Workflow Buckets for Work ID {work_id}: \n{workflow_buckets_list}")

    if len(workflow_buckets_list) > 0:
        if (
            type(workflow_buckets_list[0]) == dict
            and "results" in workflow_buckets_list[0]
        ):
            workflow_buckets_dict = workflow_buckets_list[0]  # ["results"]
            return workflow_buckets_dict

    if failover_to_results:
        work_bucket = get_work_from_results(
            workflow_results_name=workflow_buckets_name,
            work_id=work_id,
            failover_to_buckets=False,
        )
        if work_bucket:
            return work_bucket
        else:
            # Maybe finally appeared in Buckets in meantime, try one more time
            return get_work_from_buckets(
                workflow_buckets_name == workflow_buckets_name,
                work_id=work_id,
                failover_to_results=False,
            )

    return None


def get_work_from_results(workflow_results_name, work_id, failover_to_buckets):
    log.setLevel(logging.INFO)

    http_context = HTTPContext()
    workflow_results_api = http_context.results
    workflow_results_list = workflow_results_api.view(
        query={"id": work_id},
        pipeline=workflow_results_name,
        limit=1,
        # projection={"results": 1},
    )

    log.info(f"Workflow Results for Work ID {work_id}: \n{workflow_results_list}")

    if len(workflow_results_list) > 0:
        if (
            type(workflow_results_list[0]) == dict
            and "results" in workflow_results_list[0]
        ):
            workflow_results_dict = workflow_results_list[0]  # ["results"]
            return workflow_results_dict

    if failover_to_buckets:
        work_result = get_work_from_buckets(
            workflow_buckets_name=workflow_results_name,
            work_id=work_id,
            failover_to_results=False,
        )
        if work_result:
            return work_result
        else:
            # Maybe moved to Results in meantime, try one more time
            return get_work_from_results(
                workflow_results_name=workflow_results_name,
                work_id=work_id,
                failover_to_buckets=False,
            )

    return None
