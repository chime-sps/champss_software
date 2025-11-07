# Code to benchmark overhead of processing
import logging
import tqdm
import time
import docker
from multiprocessing import Pool

log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from workflow.definitions.work import Work
from sps_databases import db_utils
from scheduler.workflow import (
    schedule_workflow_job,
    clear_workflow_buckets,
    wait_for_no_tasks_in_states
)

docker_swarm_pending_states = [
    "new",
    "pending",
    "assigned",
    "accepted",
    "ready",
    "preparing",
    "starting",
]
docker_swarm_finished_states = [
    "complete",
    "failed",
    "shutdown",
    "rejected",
    "orphaned",
    "remove",
]
tiers = ["small", "large"]
mem_ranges = [[0,10], [10,1000]]

def get_tier(mem_req):
    tier = -1
    for i, mem_range in enumerate(mem_ranges):
        if mem_req > mem_range[0] and mem_req < mem_range[1]:
            tier = i
            break
    return tier

    

def deposit_dummy_work(pointing):
    work = Work(
        pipeline="dummy-schedule", site="chime", user="CHAMPSS"
    )

    work.function = "scheduler.utils.dummy_workflow_task"
    mem_req =ram_requirement(pointing)
    work.parameters = {"wait_time": mem_req *1/100}
    tier = get_tier(mem_req)
    work.tags = [pointing['_id'].__str__(), tiers[tier]]
    work.config.archive.results = True
    work.config.archive.plots = "bypass"
    work.config.archive.products = "bypass"
    work.retries = 1
    # work.timeout = timeout

    # wait_for_no_tasks_in_states(docker_swarm_pending_states)

    work_id = work.deposit(return_ids=True)
    return work_id


def run_dummy_processing():
    clear_workflow_buckets.main(
                    args=["--workflow-buckets-name", "dummy-schedule"],
                    standalone_mode=False,
                )
    db = db_utils.connect()
    docker_client = docker.from_env()
    all_pointings = list(db.pointings.find())
    all_pointings = all_pointings[:100]
    for index in range(len(all_pointings)):
        current_pointing = all_pointings[index]
        all_pointings[index]["ram_requirement"] = ram_requirement(current_pointing)
    log.info("Start Scheduling")

    # for pointing_index, pointing in tqdm.tqdm(enumerate(all_pointings), total=len(all_pointings)):
    #     work = Work(
    #         pipeline="dummy-schedule", site="chime", user="CHAMPSS"
    #     )

    #     work.function = "run-dummy-task 0"
    #     work.parameters = {}
    #     work.tags = [pointing['_id'].__str__()]
    #     work.config.archive.results = True
    #     work.config.archive.plots = "bypass"
    #     work.config.archive.products = "bypass"
    #     work.retries = 1
    #     # work.timeout = timeout

    #     # wait_for_no_tasks_in_states(docker_swarm_pending_states)

    #     work_id = work.deposit(return_ids=True)
    with Pool(16) as pool:
        work_ids = list(tqdm.tqdm(pool.imap(deposit_dummy_work, all_pointings), total=len(all_pointings)))

    pointing = all_pointings[0]

    log.info("Filled the bucket")
    # breakpoint()
    sim_wait = 1
    cleanup_interval = 30
    waiting_services = []
    non_finished_services = []

    a= time.time()
    work_is_not_done = False
    # for pointing_index, pointing in tqdm.tqdm(enumerate(all_pointings), total=len(all_pointings)):
    for i in range(5):
        # while len(waiting_services) > sim_wait:
        # while len(waiting_services) > sim_wait:
        #     new_waiting_services = []
        #     for service_index, service in enumerate(waiting_services):
        #         service.reload()
        #         service_state = service.tasks()[0]["Status"]["State"]
        #         if service_state in docker_swarm_pending_states:
        #             new_waiting_services.append(service)
        #         else:
        #             log.info("Removed finished service")
        #     waiting_services = new_waiting_services
        # if pointing_index % cleanup_interval == 1:
        #     finished_indices = []
        #     for service_index, service in enumerate(non_finished_services):
        #         service.reload()
        #         service_state = service.tasks()[0]["Status"]["State"]
        #         if service_state in docker_swarm_finished_states:
        #             service.remove()
        #             finished_indices.append(service_index)
        #     for service_index in finished_indices[::-1]:
        #         del non_finished_services[service_index]
        

                

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
            "name": f"processing-{i}",
            # Use one-shot Workflow runners since we need a new container per process for unique memory reservations
            # (we currently only use Workflow as a wrapper for its additional features, e.g. frontend)
            "command": (
                "workflow run"
                f" dummy-schedule --site"
                f" chime --lives 100 --sleep 1"
                f" --tag {tiers[i%2]}"
            ),
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


        service = docker_client.services.create(**docker_service)
        waiting_services.append(service)
        non_finished_services.append(service)
        # breakpoint()
        # wait_for_no_tasks_in_states(docker_swarm_pending_states, docker_service_name_prefix=f"processing-{pointing['_id'].__str__()}")
    b = time.time()
    log.info("Finish Scheduling")
    log.info(b-a)
    all_states = ["started"]
    while not set(all_states).issubset(docker_swarm_finished_states):
        all_states = []
        for service in non_finished_services:
            all_states.append(service.tasks()[0]["Status"]["State"])
    c = time.time()
    log.info("Prcessing finished")
    log.info(c-a)
    log.info((c-a)/len(all_pointings))


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
