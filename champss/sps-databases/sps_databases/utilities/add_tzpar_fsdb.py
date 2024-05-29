import glob
import os

import click
import numpy as np
from beamformer.utilities.dm import DMMap
from sps_databases import db_api, db_utils
from sps_multi_pointing.known_source_sifter.add_tzpar_sources import (
    ra_dec_from_ecliptic,
)


def add_source_to_database(payload):
    """
    Add a source into the follow-up source database, given a dictionary including all
    the properties required by the database.

    Arguments
    ---------
    payload: dict
        A dictionary including all the properties expected by the follow-up source database. It should have :
        ['source_type', 'source_name', 'pos_ra_deg', 'pos_dec_deg', 'pos_error_semimajor_deg',
        'pos_error_semiminor_deg', 'pos_error_theta_deg', 'dm', 'dm_error', 'spin_period_s',
        'spin_period_s_error', 'dm_galactic_ne_2001_max', 'dm_galactic_ymw_2016_max', 'spin_period_derivative',
        'spin_period_derivative_error', 'spin_period_epoch']
    """
    db = db_utils.connect()
    fs = db.followup_sources.find_one({"source_name": payload["source_name"]})
    if not fs:
        print(f"Adding {payload['source_name']} to the follow-up source database.")
        db_api.create_followup_source(payload)
        return
    if (
        "pepoch" not in fs.keys()
        or payload["pepoch"] > fs["pepoch"]
        or np.isnan(fs["pepoch"])
    ):
        print(f"Updating {payload['source_name']} in the follow-up source database.")
        db_api.update_followup_source(fs["_id"], payload)
    else:
        print(
            f"Parfile for {payload['source_name']} is older than the entry in the"
            " follow-up source database. Not adding it to the database"
        )
    return


@click.command()
@click.argument("path", type=str)
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
def main(path, db_port, db_host, db_name):
    """
    The script loops through the parfiles in a directory to extract the relevant values
    to be added to the known source database.

    The attributes extracted are :
    ['source_type', 'source_name', 'pos_ra_deg', 'pos_dec_deg', 'pos_error_semimajor_deg',
    'pos_error_semiminor_deg', 'pos_error_theta_deg', 'dm', 'dm_error', 'spin_period_s',
    'spin_period_s_error', 'dm_galactic_ne_2001_max', 'dm_galactic_ymw_2016_max', 'spin_period_derivative',
    'spin_period_derivative_error', 'spin_period_epoch']
    """
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    tzpar_path = path
    dmm = DMMap()
    for f in sorted(glob.glob(f"{tzpar_path}/*.par")):
        payload = {}
        with open(f) as infile:
            elong = np.nan
            elat = np.nan
            for line in infile:
                if len(line.split()) == 0:
                    continue
                if "PSR" in line.split()[0]:
                    payload["source_name"] = line.split()[1]
                    payload["source_type"] = "known_source"
                if line.split()[0] == "RAJ":
                    ra_string = line.split()[1].split(":")
                    ra_hour = float(ra_string[0])
                    ra_min = float(ra_string[1])
                    ra_sec = float(ra_string[2])
                    ra = ra_hour * 15 + ra_min * 15 / 60 + ra_sec * 15 / 3600
                    payload["ra"] = ra
                if line.split()[0] == "DECJ":
                    dec_string = line.split()[1].split(":")
                    dec_deg = float(dec_string[0])
                    dec_min = float(dec_string[1])
                    dec_sec = float(dec_string[2])
                    dec = dec_deg + dec_min / 60 + dec_sec / 3600
                    payload["dec"] = dec
                if line.split()[0] in ("ELAT", "BETA"):
                    elat = float(line.split()[1])
                if line.split()[0] in ("ELONG", "LAMBDA"):
                    elong = float(line.split()[1])
                if line.split()[0] == "F0":
                    payload["f0"] = float(line.split()[1].replace("D", "e"))
                if line.split()[0] == "DM":
                    payload["dm"] = line.split()[1]
                if line.split()[0] == "PEPOCH":
                    payload["pepoch"] = float(line.split()[1])
        # Add if statement to catch sources without ra, dec
        if elat is not np.nan and elong is not np.nan:
            ra, dec, ra_err, dec_err = ra_dec_from_ecliptic(elong, elat, 0, 0)
            payload["ra"] = ra
            payload["dec"] = dec
        payload["dm_galactic_ne_2001_max"] = float(
            dmm.get_dm_ne2001(payload["dec"], payload["ra"])
        )
        payload["dm_galactic_ymw_2016_max"] = float(
            dmm.get_dm_ymw16(payload["dec"], payload["ra"])
        )
        payload["followup_duration"] = 10000
        payload["active"] = True
        ephpath = os.path.abspath(f)
        payload["path_to_ephemeris"] = ephpath
        try:
            add_source_to_database(payload)
        except Exception as e:
            print(f"Failed to add {payload['source_name']} to the database.")
            print(e)


if __name__ == "__main__":
    main()
