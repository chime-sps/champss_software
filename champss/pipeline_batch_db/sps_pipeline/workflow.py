import datetime as dt
import logging
import os
import time

import click
import docker
from slack_sdk import WebClient

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
task_timeout_seconds = 60 * 40  # 40 minutes


def message_slack(
    slack_message,
    slack_channel="#slow-pulsar-alerts",
    slack_token="xoxb-194910630096-6273790557189-FKbg9w1HwrJYqGmBRY8DF0te",
):
    log.setLevel(logging.INFO)
    log.info(f"Sending to Slack: \n{slack_message}")
    slack_client = WebClient(token=slack_token)
    try:
        slack_request = slack_client.chat_postMessage(
            channel=slack_channel,
            text=slack_message,
        )
    except Exception as error:
        log.info(error)


def get_service_created_at_datetime(service):
    try:
        datetime = dt.datetime.strptime(
            service.attrs["CreatedAt"].split(".")[0], "%Y-%m-%dT%H:%M:%S"
        )
        return datetime
    except Exception as error:
        log.info(
            f"Error parsing CreatedAt for service {service}: {error} (will skip gracefully)."
        )
        return None


def wait_for_no_tasks_in_states(states_to_wait_for_none):
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
                        # Dump logs of multiprocessing container before removing it
                        if "processing-mp" in service.name:
                            try:
                                date = service.name.split("-")[-1]

                                log_text = ""
                                log_generator = service.logs(
                                    details=True,
                                    stdout=True,
                                    stderr=True,
                                    follow=False,
                                )
                                for log_chunk in log_generator:
                                    log_text += log_chunk.decode("utf-8")

                                path = f"/data/chime/sps/sps_processing/mp_runs/daily_{date}/container.log"
                                directory = os.path.dirname(path)

                                if not os.path.exists(directory):
                                    os.makedirs(directory)
                                    log.info(f"Created directory: {directory}")

                                with open(
                                    path,
                                    "w",
                                ) as file:
                                    file.write(log_text)
                            except Exception as error:
                                log.info(
                                    "Error dumping logs for service"
                                    f" {service.name}: {error} (will skip"
                                    " gracefully)."
                                )

                        try:
                            service.remove()
                        except Exception as error:
                            log.info(
                                f"Error removing service {service.name}: {error} (will"
                                " skip gracefully)."
                            )
                    elif task_state in docker_swarm_running_states:
                        if (dt.datetime.now() - service_created_at).total_seconds() > (
                            task_timeout_seconds * 2
                        ):
                            log.info(
                                f"Service {service.name} has been ruuning for more than"
                                f" {(task_timeout_seconds  * 2) / 60} minutes in state"
                                f" {task_state}, implying frozen task on 1st or 2nd final"
                                f" Workflow runner attempt. Will remove service."
                            )

                            try:
                                service.remove()
                            except Exception as error:
                                log.info(
                                    f"Error removing service {service.name}: {error} (will"
                                    " skip gracefully)."
                                )
                        # Task in state running, and we want to wait for no running states
                        # Loop will continue
                        elif states_to_wait_for_none == docker_swarm_running_states:
                            is_task_in_state = True
                            break
                    elif task_state in docker_swarm_pending_states:
                        if (dt.datetime.now() - service_created_at).total_seconds() > (
                            task_timeout_seconds * 2
                        ):
                            log.info(
                                f"Service {service.name} has been pending for more than"
                                f" {(task_timeout_seconds  * 2) / 60} minutes in state"
                                f" {task_state}, implying failed Docker task scheduling. Will"
                                " remove service."
                            )

                            try:
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


def schedule_workflow_job(
    docker_image,
    docker_mounts,
    docker_name,
    docker_memory_reservation,
    docker_password,
    workflow_name,
    workflow_function,
    workflow_params,
    workflow_tags,
):
    """Deposit Work and scale Docker Service, as node resources are free."""
    log.setLevel(logging.INFO)

    docker_client = docker.from_env()

    try:
        docker_client.login(username="chimefrb", password=docker_password)
    except Exception as error:
        log.info(
            f"Failed to login to DockerHub to schedule {docker_name}: {error}."
            " Will not schedule this task."
        )
        return ""

    workflow_site = "chime"
    workflow_user = "CHAMPSS"

    try:
        work = Work(pipeline=workflow_name, site=workflow_site, user=workflow_user)
        work.function = workflow_function
        work.parameters = workflow_params
        work.tags = workflow_tags
        work.config.archive.results = True
        work.config.archive.plots = "pass"
        work.config.archive.products = "pass"
        work.retries = 1
        work.timeout = task_timeout_seconds

        wait_for_no_tasks_in_states(docker_swarm_pending_states)

        work_id = work.deposit(return_ids=True)

        docker_volumes = [
            docker.types.Mount(
                # Only way I know of to add custom shared memory size allocations with Docker Swarm
                target="/dev/shm",
                source="",  # Source value must be empty for tmpfs mounts
                type="tmpfs",
                tmpfs_size=int(
                    100 * 1e9
                ),  # Just give it 100GB of a shared memory upper-limit
            )
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

        docker_service = {
            "image": docker_image,
            # Can't have dots or slashes in Docker Service names
            "name": f"processing-{docker_name.replace('.', '_').replace('/', '')}",
            # Use one-shot Workflow runners since we need a new container per process for unique memory reservations
            # (we currently only use Workflow as a wrapper for its additional features, e.g. frontend)
            "command": (
                "workflow run"
                f" {workflow_name} {' '.join([f'--tag {tag}' for tag in workflow_tags])} --site"
                f" {workflow_site} --lifetime 1 --sleep-time 0"
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

        # Wait a few seconds because Work might still not have propogated to Buckets
        # and Workflow runner can pickup nothing and quietly exit
        time.sleep(2)

        docker_client.services.create(**docker_service)

        # Wait a few seconds before querying Docker Swarm again
        time.sleep(2)

        return work_id[0]
    except Exception as error:
        log.info(
            f"Failed to deposit Work and create Docker Service for {docker_name}: {error}."
            " Will not schedule this task."
        )
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
        buckets_api = Buckets()
        # Bucket API only allows 100 deletes per request
        buckets_list = buckets_api.view(
            query={"pipeline": workflow_buckets_name},
            limit=100,
            projection={"id": True},
        )
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
    except Exception as error:
        pass


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
        results_api = Results()
        # Results API only allows 10 deletes per request
        results_list = results_api.view(
            query={}, pipeline=workflow_results_name, limit=10, projection={"id": 1}
        )
        while len(results_list) != 0:
            result_ids_to_delete = [result["id"] for result in results_list]
            log.info(f"Will delete results entries with ids: {result_ids_to_delete}")
            results_api.delete_ids(
                pipeline=workflow_results_name, ids=result_ids_to_delete
            )
            results_list = results_api.view(
                query={}, pipeline=workflow_results_name, limit=10, projection={"id": 1}
            )
    except Exception as error:
        pass


def get_work_from_buckets(workflow_buckets_name, work_id, failover_to_results):
    log.setLevel(logging.INFO)

    workflow_buckets_api = Buckets()
    workflow_buckets_list = workflow_buckets_api.view(
        query={"pipeline": workflow_buckets_name, "id": work_id},
        limit=1,
        projection={"results": 1},
    )

    log.info(f"Workflow Buckets for Work ID {work_id}: \n{workflow_buckets_list}")

    if len(workflow_buckets_list) > 0:
        if (
            type(workflow_buckets_list[0]) == dict
            and "results" in workflow_buckets_list[0]
        ):
            workflow_buckets_dict = workflow_buckets_list[0]["results"]
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

    workflow_results_api = Results()
    workflow_results_list = workflow_results_api.view(
        query={"id": work_id},
        pipeline=workflow_results_name,
        limit=1,
        projection={"results": 1},
    )

    log.info(f"Workflow Results for Work ID {work_id}: \n{workflow_results_list}")

    if len(workflow_results_list) > 0:
        if (
            type(workflow_results_list[0]) == dict
            and "results" in workflow_results_list[0]
        ):
            workflow_results_dict = workflow_results_list[0]["results"]
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
