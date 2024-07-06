import datetime as dt
import logging
import os
from glob import glob

import click
import folding.fold_candidate as fold_candidate
import numpy as np
from beamformer.strategist.strategist import PointingStrategist
from beamformer.utilities.common import find_closest_pointing, get_data_list
from sps_databases import db_api, db_utils, models
from sps_pipeline.workflow import schedule_workflow_job
from sps_pipeline.utils import convert_date_to_datetime

log = logging.getLogger()


def find_all_dates_with_data(ra, dec, basepath, Nday=10):
    log.setLevel(logging.INFO)

    filepaths = np.sort(glob(f"{basepath}/*/*/*"))
    os.chdir(f"{basepath}")
    pst = PointingStrategist(create_db=False)

    dates_with_data = []

    for filepath in filepaths:
        year = int(filepath.split("/")[-3])
        month = int(filepath.split("/")[-2])
        day = int(filepath.split("/")[-1])

        date = dt.datetime(year, month, day)

        datelow = dt.datetime(2024, 1, 31)
        datehigh = dt.datetime(2024, 12, 31)
        if (date > datelow) and (date < datehigh):
            active_pointing = pst.get_single_pointing(ra, dec, date)

            files = get_data_list(
                active_pointing[0].max_beams, basepath=basepath, extn="dat"
            )
            if len(files) > 0:
                print(filepath, len(files))
                dates_with_data.append(date.strftime("%Y%m%d"))

            if len(dates_with_data) >= Nday:
                return dates_with_data

    return dates_with_data


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--fs_id",
    type=str,
    default="",
    help="FollowUpSource ID, to fold from database values",
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver",
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
    "--basepath",
    type=str,
    default="/data/chime/sps/raw/",
    help="Base directory for raw data",
)
@click.option(
    "--foldpath",
    default="/data/chime/sps/archives",
    type=str,
    help="Path for created files during fold step.",
)
@click.option(
    "--workflow-buckets-name",
    default="champss-processing",
    type=str,
    help="Which Worklow DB to create/use.",
)
@click.option(
    "--docker-image-name",
    default="chimefrb/champss_software:latest",
    type=str,
    help="Which Docker Image name to use.",
)
@click.option(
    "--docker-service-name-prefix",
    default="fold-multiday",
    type=str,
    help="What prefix to apply to the Docker Service name",
)
@click.option(
    "--docker-password",
    prompt=True,
    confirmation_prompt=False,
    hide_input=True,
    required=True,
    type=str,
    help="Password to login to chimefrb DockerHub (hint: frbadmin's common password).",
)
def main(
    fs_id,
    db_port,
    db_host,
    db_name,
    basepath,
    foldpath,
    workflow_buckets_name,
    docker_image_name,
    docker_service_name_prefix,
    docker_password,
):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    source = db_api.get_followup_source(fs_id)
    ra = source.ra
    dec = source.dec
    dm = source.dm
    nchan_tier = int(np.ceil(np.log2(dm // 212.5 + 1)))
    nchan = 1024 * (2**nchan_tier)
    dates_with_data = find_all_dates_with_data(ra, dec, basepath, Nday=10)
    log.info(f"Folding {len(dates_with_data)} days of data: {dates_with_data}")
    for date in dates_with_data:
        docker_name = f"{docker_service_name_prefix}-{date}-{fs_id}"
        docker_memory_reservation = (nchan / 1024) * 8
        docker_mounts = [
            "/data/chime/sps/raw:/data/chime/sps/raw",
            f"{foldpath}:{foldpath}",
        ]

        workflow_function = "folding.fold_candidate.main"
        workflow_params = {
            "date": date,
            "fs_id": fs_id,
            "db_host": db_host,
            "db_port": db_port,
            "db_name": db_name,
            "write_to_db": True,
            "using_workflow": True,
        }
        workflow_tags = [
            "fold",
            "multiday",
            date,
            fs_id,
        ]

        schedule_workflow_job(
            docker_image_name,
            docker_mounts,
            docker_name,
            docker_memory_reservation,
            docker_password,
            workflow_buckets_name,
            workflow_function,
            workflow_params,
            workflow_tags,
        )


if __name__ == "__main__":
    main()
