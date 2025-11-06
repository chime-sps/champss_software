# Code to benchmark overhead of processing
import logging
import tqdm
import time
import docker

log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from sps_databases import db_utils
from scheduler.workflow import (
    schedule_workflow_job,
    clear_workflow_buckets,
)


def run_dummy_processing():
    clear_workflow_buckets.main(
                    args=["--workflow-buckets-name", "dummy-schedule"],
                    standalone_mode=False,
                )
    db = db_utils.connect()
    docker_client = docker.from_env()
    all_pointings = list(db.pointings.find())
    for index in range(len(all_pointings)):
        current_pointing = all_pointings[index]
        all_pointings[index]["ram_requirement"] = ram_requirement(current_pointing)
    log.info("Start Scheduling")
    a= time.time()
    for pointing in tqdm.tqdm(all_pointings):
        docker_mounts = [
            f"{'/mnt/beegfs-client/'}:{'/mnt/beegfs-client/'}",
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
                docker.types.Mount(
                    target=mount_target, source=mount_source, type="bind"
                )
            )
        docker_service = {
            "image": "sps-archiver1.chime:5000/champss_software:dummy_scheduling",
            # Can't have dots or slashes in Docker Service names
            # All Docker Services made with this function will be prefixed with "processing-"
            "name": f"processing-{pointing['_id'].__str__()}",
            # Use one-shot Workflow runners since we need a new container per process for unique memory reservations
            # (we currently only use Workflow as a wrapper for its additional features, e.g. frontend)
            "command": "run-dummy-task",
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
                mem_reservation=int( pointing["ram_requirement"] * 1e9)
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


        docker_client.services.create(**docker_service)
    b = time.time()
    log.info("Finish Scheduling")
    log.info(b-a)


def ram_requirement(pointing_dict):
    # As the memory requirement is a property of the process rather than the pointing, I copy pasted the current formula here
    return min(
        100,
        int(
            4
            + (pointing_dict["maxdm"] * 0.04 + pointing_dict["length"] * 6e-6)
            * 2 ** (pointing_dict["length"] // 2**20)
        ),
    )


if __name__ == "__main__":
    run_dummy_processing()
