import argparse
import subprocess

import numpy as np
import pulsarsurveyscraper
from add_tzpar_sources import add_source_to_database
from beamformer.utilities.dm import DMMap
from sps_databases import db_utils

if __name__ == "__main__":
    """
    This script adds all pulsars in the pulsarsurveyscraper into the known_source
    database.

    https://github.com/dlakaplan/pulsarsurveyscraper
    """
    parser = argparse.ArgumentParser(description="Adding known sources to the database")
    parser.add_argument(
        "--db-port",
        type=int,
        default=27017,
        help="The port of the database.",
    )
    parser.add_argument(
        "--db-host",
        default="localhost",
        type=str,
        help="Host used for the mongodb database.",
    )
    parser.add_argument(
        "--db-name",
        default="sps",
        type=str,
        help="Name used for the mongodb database.",
    )
    parser.add_argument(
        "--cachedir",
        default="/data/chime/sps/pulsarscraper_cache/",
        type=str,
        help="the path to directory with the surver scraper hdf5 file cache",
    )
    args = parser.parse_args()

    # command from pulsarsurveyscraper, keeps pulsar cache up to date
    subprocess.run(["cache_survey", "-s", "all", "-o", args.cachedir])

    pulsar_table = pulsarsurveyscraper.PulsarTable(directory=f"{args.cachedir}")
    data = pulsar_table.data

    db = db_utils.connect(host=args.db_host, port=args.db_port, name=args.db_name)
    dmm = DMMap()

    for i in range(len(data["PSR"])):
        psrname = data["PSR"][i]
        ra = data["RA"][i]
        dec = data["Dec"][i]
        P0 = data["P"][i] / 1000.0
        DM = data["DM"][i]
        try:
            P0 = float(P0)
            P0err = 10 ** -(len(str(P0).split(".")[-1]))
        except:
            P0 = nan
            P0err = nan
        try:
            DM = float(DM)
            DMerr = 10 ** -(len(str(DM).split(".")[-1]))
        except:
            DM = nan
            DMerr = nan
        print(P0, P0err, DM, DMerr)

        survey = data["survey"][i]
        if dec > -20:
            payload = {
                "source_type": 1,
                "source_name": psrname,
                "pos_ra_deg": ra,
                "pos_dec_deg": dec,
                "pos_error_semimajor_deg": 0.068,
                "pos_error_semiminor_deg": 0.068,
                "pos_error_theta_deg": 0.0,
                "dm": DM,
                "dm_error": DMerr,
                "spin_period_s": P0,
                "spin_period_s_error": P0err,
                "dm_galactic_ne_2001_max": float(dmm.get_dm_ne2001(dec, ra)),
                "dm_galactic_ymw_2016_max": float(dmm.get_dm_ymw16(dec, ra)),
                "spin_period_derivative": 0.0,
                "spin_period_derivative_error": 0.0,
                "spin_period_epoch": 45000.0,  # placeholder, older than psrcat so as not to overwrite
            }
            print(psrname, ra, dec, P0, DM)
            add_source_to_database(payload)
