"""
Database utilities for folding operations.

This module contains functions for interacting with the followup_sources database.
"""

import os

import numpy as np
import pymongo
from bson.objectid import ObjectId
from sps_databases import db_api, db_utils


def get_pulsar_coords_from_timing_db(psr, server_url="mongodb://sps-archiver1:27017/", db_name="timing_ops"):
    """
    Get RA and Dec for a pulsar from the timing_ops database.

    Parameters
    ----------
    psr : str
        Pulsar name (psr_id)
    server_url : str
        MongoDB server URL
    db_name : str
        Database name

    Returns
    -------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    """
    client = pymongo.MongoClient(server_url)
    database = client[db_name]
    collection = database["sources"]

    source = collection.find_one({'psr_id': psr})
    if not source:
        raise ValueError(f"Pulsar {psr} not found in timing_ops database")

    return source['ra'], source['dec']


def update_folding_history(id, payload):
    """
    Updates a followup_source with the given attribute and values as a dict.

    Parameters
    ----------
    id: str or ObjectId
        The id of the followup source to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    followup_source: dict
        The dict of the updated followup source
    """
    db = db_utils.connect()
    if isinstance(id, str):
        id = ObjectId(id)
    return db.followup_sources.find_one_and_update(
        {"_id": id},
        {"$push": payload},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def scrape_ephemeris(ephem_path):
    """
    Parse ephemeris file and extract source parameters.

    Parameters
    ----------
    ephem_path : str
        Path to ephemeris file

    Returns
    -------
    payload : dict
        Dictionary containing source parameters
    """
    from beamformer.utilities.dm import DMMap

    dmm = DMMap()
    payload = {}
    print(ephem_path)
    with open(ephem_path) as infile:
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
    ephem_path_absolute = os.path.abspath(ephem_path)
    payload["path_to_ephemeris"] = ephem_path_absolute
    return payload


def add_knownsource_to_fsdb(ephem_path):
    """
    Create a payload for a known source to be input in the followup sources database.

    Parameters
    ----------
    ephem_path : str
        Path to ephemeris file
    """
    db = db_utils.connect()
    payload = scrape_ephemeris(ephem_path)

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

    print(f"Added {name} to the follow-up source database")
    return


def add_candidate_to_fsdb(
    date_str,
    ra,
    dec,
    f0,
    dm,
    sigma,
    cand_path="",
    source_type="sd_candidate",
    path_to_ephemeris=None,
):
    """
    Create a payload for a candidate to be input in the followup sources database.

    Parameters
    ----------
    date_str : str
        Date string in YYYYMMDD format
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    f0 : float
        Spin frequency in Hz
    dm : float
        Dispersion measure in pc/cm^3
    sigma : float
        Detection sigma
    cand_path : str, optional
        Path to candidate file
    source_type : str, optional
        Type of source ("sd_candidate" or "md_candidate")
    path_to_ephemeris : str, optional
        Path to ephemeris file

    Returns
    -------
    fs : dict
        Followup source document
    """
    from beamformer.utilities.dm import DMMap

    dmm = DMMap()
    db = db_utils.connect()

    ra_name = np.round(float(ra), 2)
    dec_name = np.round(float(dec), 2)
    f0_name = np.round(float(f0), 6)
    dm_name = np.round(float(dm), 2)

    if source_type == "md_candidate":
        duration = 30
        name = f"{source_type[:2]}_{ra_name}_{dec_name}_{f0_name}_{dm_name}"
    elif source_type == "sd_candidate":
        duration = 1
        name = f"{date_str}_{source_type[:2]}_{ra_name}_{dec_name}_{f0_name}_{dm_name}"
    else:
        print("must be either md or sd candidate")
        return

    payload = {
        "source_type": source_type,
        "source_name": name,
        "ra": ra,
        "dec": dec,
        "f0": f0,
        "dm": dm,
        "candidate_sigma": sigma,
        "followup_duration": duration,
        "active": True,
        "path_to_candidates": [cand_path],
        "path_to_ephemeris": path_to_ephemeris,
    }

    payload["dm_galactic_ne_2001_max"] = float(
        dmm.get_dm_ne2001(payload["dec"], payload["ra"])
    )
    payload["dm_galactic_ymw_2016_max"] = float(
        dmm.get_dm_ymw16(payload["dec"], payload["ra"])
    )

    fs = db.followup_sources.find_one({"source_name": payload["source_name"]})
    if not fs:
        print(f"Adding {payload['source_name']} to the follow-up source database.")
        fs_id = db_api.create_followup_source(payload)
        fs = db.followup_sources.find_one({"source_name": payload["source_name"]})
        return fs
    else:
        print(
            f"Source {payload['source_name']} already in the follow-up source database."
        )
        return fs


def add_mdcand_from_candpath(candpath, date):
    """
    Add a multi-day candidate to followup sources database from candidate file.

    Parameters
    ----------
    candpath : str
        Path to candidate file
    date : datetime
        Observation date

    Returns
    -------
    fs_id : ObjectId
        Followup source ID
    """
    from sps_common.interfaces import MultiPointingCandidate

    date_str = date.strftime("%Y%m%d")
    mpc = MultiPointingCandidate.read(candpath)
    ra = mpc.ra
    dec = mpc.dec
    f0 = mpc.best_freq
    dm = mpc.best_dm
    sigma = mpc.best_sigma

    followup_source = add_candidate_to_fsdb(
        date_str, ra, dec, f0, dm, sigma, candpath, source_type="md_candidate"
    )
    fs_id = followup_source["_id"]
    return fs_id


def add_mdcand_from_psrname(psrname, date):
    """
    Add a multi-day candidate to followup sources database from pulsar name.

    Parameters
    ----------
    psrname : str
        Pulsar name
    date : datetime
        Observation date

    Returns
    -------
    fs_id : ObjectId
        Followup source ID
    """
    source = db_api.get_known_source_by_names(psrname)[0]

    f0 = 1 / source.spin_period_s
    ra = source.pos_ra_deg
    dec = source.pos_dec_deg
    dm = source.dm
    sigma = 0.0

    date_str = date.strftime("%Y%m%d")

    followup_source = add_candidate_to_fsdb(
        date_str, ra, dec, f0, dm, sigma, source_type="md_candidate"
    )
    fs_id = followup_source["_id"]
    return fs_id
