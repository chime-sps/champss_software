# Code to benchmark overhead of processing
import logging

log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from sps_databases import db_utils
from scheduler.workflow import (
    schedule_workflow_job,
)


def run_dummy_processing():
    db = db_utils.connect()
    all_pointings = list(db.pointings.get())
    for index in len(all_pointings):
        current_pointing = all_pointings[index]
        all_pointings[index]["ram_requirement"] = ram_requirement(current_pointing)
    log.info("Start Scheduling")
    for pointing in all_pointings:
        workflow_function = "scheduler.utils.dummy_workflow_task"
        workflow_params = {"input_dict": pointing}
        workflow_tags = [pointing["ra"], pointing["dec"]]
        docker_mounts = [
            f"{'/mnt/beegfs-client/'}:{'/mnt/beegfs-client/'}",
        ]

        schedule_workflow_job(
            "sps-archiver1.chime:5000/champss_software:dummy_scheduler",
            docker_mounts,
            pointing["_id"].__str__(),
            pointing["ram_requirement"],
            "dummy-schedule",
            workflow_function,
            workflow_params,
            workflow_tags,
        )
    log.info("Finish Scheduling")


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
