# Code to benchmark overhead of processing
import logging
import tqdm
import time

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
    all_pointings = list(db.pointings.find())
    for index in range(len(all_pointings)):
        current_pointing = all_pointings[index]
        all_pointings[index]["ram_requirement"] = ram_requirement(current_pointing)
    log.info("Start Scheduling")
    a= time.time()
    for pointing in tqdm.tqdm(all_pointings):
        workflow_function = "scheduler.utils.dummy_workflow_task"
        workflow_params = {"input_dict": f"{pointing['ra']:.2f}_{pointing['dec']:.2f}"}
        workflow_tags = [pointing["_id"].__str__(),]
        docker_mounts = [
            f"{'/mnt/beegfs-client/'}:{'/mnt/beegfs-client/'}",
        ]

        schedule_workflow_job(
            "sps-archiver1.chime:5000/champss_software:dummy_scheduling",
            docker_mounts,
            pointing["_id"].__str__(),
            pointing["ram_requirement"],
            "dummy-schedule",
            workflow_function,
            workflow_params,
            workflow_tags,
        )
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
