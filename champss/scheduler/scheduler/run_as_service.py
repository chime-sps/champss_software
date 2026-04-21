import docker
from bson.objectid import ObjectId
import logging
import threading
import click
import getpass


from scheduler.workflow import wait_until_service_not_pending, remove_finished_service

log = logging.getLogger()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "command",
    required=True,
    type=str,
)
@click.option(
    "--image",
    default="sps-archiver1.chime:5000/champss_software:latest",
    type=str,
    help="The used docker image.",
)
@click.option(
    "--memory",
    default=20,
    type=int,
    help="The memory reservation in GB.",
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    help="Whether to remove the service after finishing or not. With cleanup enabled the script will only finish after cleanup.",
)
@click.option(
    "--manager/--no-manager",
    default=False,
    help="Only run job on manager node.",
)
def run_as_service_cli(command, image, memory, cleanup, manager):
    run_as_service(
        command, image=image, memory=memory, cleanup=cleanup, manager=manager
    )


def run_as_service(
    command,
    image="sps-archiver1.chime:5000/champss_software:latest",
    memory=50,
    cleanup=True,
    manager=False,
):
    docker_client = docker.from_env()
    command_start = command.split(" ")[0]
    id = ObjectId().__str__()
    docker_volumes = [
        docker.types.Mount(
            # Bind mount the Docker socket to allow Docker-in-Docker (Workflow-in-Workflow) usage
            target="/var/run/docker.sock",
            source="/var/run/docker.sock",
            type="bind",
        ),
        docker.types.Mount(
            target="/dev/shm",
            source="",  # Source value must be empty for tmpfs mounts
            type="tmpfs",
            tmpfs_size=int(
                100 * 1e9  # Use 100 GB by default since it will crash when more is used
            ),
        ),
        docker.types.Mount(
            target="/mnt/beegfs-client/", source="/mnt/beegfs-client/", type="bind"
        ),
        docker.types.Mount(target="/data/", source="/data/", type="bind"),
    ]
    service_name = f"{command_start.replace('.', '_').replace('/', '')}-{id}"
    user = getpass.getuser()

    if manager:
        constraints = ["node.role == manager"]
    else:
        constraints = ["node.labels.compute == true"]
    docker_service = {
        "image": image,
        "name": service_name,
        "command": command,
        "env": ["CONTAINER_NAME={{.Task.Name}}", "NODE_NAME={{.Node.Hostname}}"],
        "mode": docker.types.ServiceMode("replicated", replicas=1),
        "restart_policy": docker.types.RestartPolicy(condition="none", max_attempts=0),
        "labels": {"type": "run-as-service", "user": user},
        "constraints": constraints,
        # Must be in bytes
        "resources": docker.types.Resources(mem_reservation=int(memory * 1e9)),
        "mounts": docker_volumes,
        "networks": ["pipeline-network"],
    }

    # log.info(f"Creating Docker Service: \n{docker_service}")
    log.info(f"Creating Docker Service for command '{command}'")

    service = docker_client.services.create(**docker_service)

    service_id = service.attrs["ID"]
    status = wait_until_service_not_pending(service_id)
    log.info(f"Service {service.name} started with id {service_id}")
    if cleanup:
        remove_service_thread = threading.Thread(
            target=remove_finished_service, args=(service_id,)
        )
        remove_service_thread.start()
        log.info("Cleanup thread started. Will remove service once finished.")
        return service_id, remove_service_thread
    else:
        log.info("Started without cleanup.")
        log.info(
            f"Consider cleaning up after finish with: \n docker service ls --filter 'label=user={user}' --quiet | xargs docker service rm"
        )
        return service_id
