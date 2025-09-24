import argparse
import subprocess

import numpy as np
import pulsarsurveyscraper
from add_tzpar_sources import add_source_to_database
from beamformer.utilities.dm import DMMap
from sps_databases import db_api, db_utils
from sps_databases.db_api import get_nearby_known_sources, delete_known_source

def check_if_duplicate(psrname, ra, dec, P0, dm, survey):
    sources = get_nearby_known_sources(ra, dec, 0.5)  
    for source in sources: 
        if survey != source.survey: # Needed in case same exact source is already in db
            del_dm = np.abs(source.dm - dm)
            del_spin_period = np.abs(source.spin_period_s - P0)
            if del_dm < dm/10 and del_spin_period < P0/1000:
                # print(f'{psrname} is a duplicate of {source.source_name}')
                # print(source.source_name, source.pos_ra_deg, source.pos_dec_deg, source.spin_period_s, source.dm, source.survey)
                # print(psrname, ra, dec, P0, dm, survey)
                return True
    return False

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
        default="sps-archiver1",
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
    parser.add_argument(
        "--update",
        help="Update the known source database instead of adding new sources",
        action="store_true",
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

        survey = data["survey"][i]
        if dec > -20 and survey != 'ATNF': # ATNF sources are redundant with the write_psrcat script
            duplicate = check_if_duplicate(psrname, ra, dec, P0, DM, [survey.lower(), "psr_scraper"])
            if duplicate: 
                ks = db.known_sources.find_one({"source_name": psrname, "survey" : [survey.lower(), "psr_scraper"]})
                if ks: 
                    print(f'Removing {psrname} from database \n')
                    ks_id = ks["_id"]
                    delete_known_source(ks_id)
            else:
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
                    "spin_period_epoch": 39000.0,  # placeholder, older than oldest ephemeris in psrcat so as to not overwrite
                    "survey": [survey.lower(), "psr_scraper"],
                }
                
                if not args.update:
                    add_source_to_database(payload)
                else:
                    print("Updating known source database using psrscraper")
                    ks = db.known_sources.find_one({"source_name": psrname, "survey" : [survey.lower(), "psr_scraper"]})
                    if ks is None:
                        add_source_to_database(payload)
                    else: 
                        ks_id = ks["_id"]
                        db_api.update_known_source(ks_id, payload)
