import docker
import time
from chime_frb_api.workflow import Work
from folding.Candidate_Filter import Filter
from sps_databases import db_api, db_utils
import datetime as dt
from beamformer.strategist.strategist import PointingStrategist
import numpy as np

from slack_sdk import WebClient

def message_slack( slack_message, slack_channel="#slow-pulsar-alerts",
                  slack_token="xoxb-194910630096-6273790557189-FKbg9w1HwrJYqGmBRY8DF0te"):
    slack_client = WebClient(token=slack_token)
    slack_request = slack_client.chat_postMessage(
        channel=slack_channel,
        text=slack_message,
    )

def find_and_run_all_folding_processes(date, db_port, db_host, db_name, workflow_name):

    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    print("Polulating processes database with folding processes")
    pipe_processes = list(
         db["processes"].find(
             {
                 "datetime": {
                    "$gte": date,
                    "$lte": date + dt.timedelta(days=1),
                },
                "status": 2,
                "folded_status": False
            }
        )
    )

    if len(pipe_processes) == 0:
        print(f"No folding processes found for {date} /n"
               "either the pipeline has not been run, or the candidates have already been folded")
        return

    foldprocesses = Filter(date)
    number_of_foldprocesses = len(foldprocesses["ras"])

    date_string = date.strftime("%Y/%m/%d")
    print(f"{number_of_foldprocesses} folding processes found for {date_string}")

    beginning_message = f"Folding {number_of_foldprocesses} candidates for {date_string}"
    message_slack(beginning_message)

    for i in range(len(pipe_processes)):
        db_api.update_process(pipe_processes[0]["_id"], {"folded_status": True})

    pst = PointingStrategist(create_db=False)
    for i in range(number_of_foldprocesses):
        ra = foldprocesses["ras"][i]
        dec = foldprocesses["decs"][i]
        active_pointing = pst.get_single_pointing(ra, dec, date)
        active_process = db_api.get_process_from_active_pointing(active_pointing[0])
        db_api.update_process(active_process.id, {"folded_status": False})

    for i in range(number_of_foldprocesses):
        sigma = foldprocesses["sigmas"][i]
        dm = foldprocesses["dms"][i]
        f0 = foldprocesses["f0s"][i]
        ra = foldprocesses["ras"][i]
        dec = foldprocesses["decs"][i]
        known = foldprocesses["known"][i]

        print(
            (
                f"Scheduling Workflow job for {date} {sigma} {dm} {f0} {ra} {dec} {known} "
                f"on {db_host}:{db_port}:{db_name} MongoDB to {workflow_name} Workflow DB"
            )
        )

        schedule_workflow_job(
            date,
            sigma,
            dm,
            f0,
            ra,
            dec,
            known,
            db_port,
            db_host,
            db_name,
            workflow_name,
        )
    ending_message = f"Candidate folding for {date_string} complete"
    message_slack(ending_message)


def schedule_workflow_job(
    date,
    sigma,
    dm,
    f0,
    ra,
    dec,
    known,
    db_port,
    db_host,
    db_name,
    workflow_name,
):
    """Deposit Work and scale Docker Service, as node resources are free."""
    client = docker.from_env()

    service_tiers = [
        "tiny",
        "small",
        "medium",
        "large",
        "huge",
    ]

    nchan_tier = int(np.ceil(np.log2(dm // 212.5 + 1)))
    nchan = 1024 * (2**nchan_tier)

    tags = ["folding"]
    if nchan == 1024:
        tags.append(service_tiers[0])
    elif nchan == 2048:
        tags.append(service_tiers[1])
    elif nchan == 4096:
        tags.append(service_tiers[2])
    elif nchan == 8192:
        tags.append(service_tiers[3])
    elif nchan == 16384:
        tags.append(service_tiers[4])
    else:
        print(
            f"nchan {nchan} does not correspond into an existing "
            "Docker Swarm Service"
        )
        return

    work = Work(pipeline=workflow_name, site="chime", user="SPAWG")
    work.function = "folding.fold_candidate.main"
    work.parameters = {
        "date": date.strftime("%Y-%m-%d"),
        "sigma": sigma,
        "dm": dm,
        "f0": f0,
        "ra": ra,
        "dec": dec,
        "known": known,
        "db_host": db_host,
        "db_port": db_port,
        "db_name": db_name,
        "write_to_db": True,
        "using_workflow": True,
    }
    work.tags = tags
    work.config.archive.results = True
    work.config.archive.plots = "pass"
    work.config.archive.products = "pass"
    work.retries = 0

    print("Depositing Workflow Work object...")

    work.deposit()

    print(f"{work}")

    # Wait a bit for Work object to propogate
    # to Workflow DB
    time.sleep(3)

    # Need to slowly queue tasks into Docker Swarm
    # otherwise it freezes
    is_a_task_pending = True
    while is_a_task_pending is True:
        is_a_task_pending = False
        # Re-fetch Docker Swarm Service states
        services_all_tiers = [
            service
            for service in client.services.list()
            if service.name.split("_")[1] in service_tiers
        ]
        for service in services_all_tiers:
            for task in service.tasks():
                if task["Status"]["State"] == "pending":
                    is_a_task_pending = True
                    print(
                        "A task is pending. Will wait for no pending tasks before"
                        " scaling runners."
                    )
                    break
            if is_a_task_pending is True:
                # Wait a bit before checking again for no pending tasks
                time.sleep(3)
                break

    service_this_tier_name = f"pipeline_{tags[1]}"
    service_this_tier = [
        service
        for service in client.services.list()
        if service.name == service_this_tier_name
    ][0]

    print("Scaling Docker Swarm Service Workflow runner...")

    service_this_tier.scale(
        service_this_tier.attrs["Spec"]["Mode"]["Replicated"]["Replicas"] + 1
    )

    print(f"{service_this_tier}")

    # Wait a bit for Workflow runner to grab this Work object
    # before depositing next one
    time.sleep(3)
