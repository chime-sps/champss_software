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
def main(fs_id, db_port, db_host, db_name, basepath):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    source = db_api.get_followup_source(fs_id)
    ra = source.ra
    dec = source.dec
    dates_with_data = find_all_dates_with_data(ra, dec, basepath, Nday=10)
    log.info(f"Folding {len(dates_with_data)} days of data: {dates_with_data}")
    for date in dates_with_data:
        fold_candidate.main(
            [
                "--date",
                date,
                "--db-port",
                db_port,
                "--db-name",
                db_name,
                "--db-host",
                db_host,
                "--fs_id",
                fs_id,
                "--write-to-db",
            ],
            standalone_mode=False,
        )
    # Silence Workflow errors, requires results, products, plots
    return {}, [], []


if __name__ == "__main__":
    main()
