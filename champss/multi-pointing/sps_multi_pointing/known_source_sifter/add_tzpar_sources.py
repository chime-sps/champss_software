import argparse
import glob

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from beamformer.utilities.dm import DMMap
from sps_databases import db_api, db_utils

obliquity = 84381.448 / 3600


def ra_dec_from_ecliptic(elong, elat, elong_err=np.nan, elat_err=np.nan):
    """
    Compute the ra and dec of a source, including its uncertainties, given the ecliptic
    position of the source and its uncertainties.

    Arguments
    ---------
    elong: float
        The ecliptic longitude of the source.

    elat: float
        The ecliptic latitude of the source.

    elong_err: float
        The uncertainty in the ecliptic longitude of the source. Default = np.nan

    elat_err: float
        The uncertainty in the ecliptic latitude of the source. Default = np.nan

    Returns:
    ra: float
        The Right Ascension of the source.

    dec: float
        The Declination of the source.

    ra_err: float
        The uncertainty in the Right Ascension of the source.

    dec_err: float
        The uncertainty in the Declination of the source.
    """
    sc = SkyCoord(elong * u.degree, elat * u.degree, frame="barycentricmeanecliptic")
    ra = sc.icrs.ra.value
    dec = sc.icrs.dec.value
    if elong_err is not np.nan and elat_err is not np.nan:
        # dec = arcsin(cos(e)sin(b) + sin(e)cos(b)sin(l))
        # ra = arctan((cos(e)sin(l) - sin(e)tan(b))/cos(l))
        # where e is obliquity of ecliptic, b is elat and l is elong
        # now we want to propagate error in b and l into dec and ra
        # equation for error is s(x)**2 = s(a)**2(dx/da)**2 + s(b)**2(dx/db)**2 + ...
        decx = np.cos(np.radians(obliquity)) * np.sin(np.radians(elat)) + np.sin(
            np.radians(obliquity)
        ) * np.cos(np.radians(elat)) * np.sin(np.radians(elong))
        ddecxdb = np.cos(np.radians(obliquity)) * np.cos(np.radians(elat)) - np.sin(
            np.radians(obliquity)
        ) * np.sin(np.radians(elat)) * np.sin(np.radians(elong))
        ddecxdl = (
            np.sin(np.radians(obliquity))
            * np.cos(np.radians(elat))
            * np.cos(np.radians(elong))
        )
        dec_err = np.sqrt(
            (1 / (1 - decx**2))
            * ((ddecxdb * elat_err) ** 2 + (ddecxdl * elong_err) ** 2)
        )
        rax = (
            np.cos(np.radians(obliquity)) * np.sin(np.radians(elong))
            - np.sin(np.radians(obliquity)) * np.tan(np.radians(elat))
        ) / np.cos(np.radians(elong))
        draxdb = -np.sin(np.radians(obliquity)) / (
            np.cos(np.radians(elat))
            * np.cos(np.radians(elat))
            * np.cos(np.radians(elong))
        )
        draxdl = np.cos(np.radians(obliquity)) / (
            np.cos(np.radians(elong)) * np.cos(np.radians(elong))
        ) - np.sin(np.radians(obliquity)) * np.tan(np.radians(elat)) * np.tan(
            np.radians(elong)
        ) / np.cos(
            np.radians(elong)
        )
        ra_err = (1 / (1 + rax**2)) * np.sqrt(
            (draxdb * elat_err) ** 2 + (draxdl * elong_err) ** 2
        )
    else:
        ra_err = 0.068
        dec_err = 0.068
    return ra, dec, ra_err, dec_err


def add_source_to_database(payload, db_port=27017, db_host="sps-archiver1", db_name="sps"):
    """
    Add a source into the known source database, given a dictionary including all the
    properties required by the database.

    Arguments
    ---------
    payload: dict
        A dictionary including all the properties expected by the known source database. It should have :
        ['source_type', 'source_name', 'pos_ra_deg', 'pos_dec_deg', 'pos_error_semimajor_deg',
        'pos_error_semiminor_deg', 'pos_error_theta_deg', 'dm', 'dm_error', 'spin_period_s',
        'spin_period_s_error', 'dm_galactic_ne_2001_max', 'dm_galactic_ymw_2016_max', 'spin_period_derivative',
        'spin_period_derivative_error', 'spin_period_epoch', 'survey']
    """
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    ks = db.known_sources.find_one({"source_name": payload["source_name"]})
    if not ks:
        print(f"Adding {payload['source_name']} to the known source database.")
        db_api.create_known_source(payload)
        return
    if (
        "spin_period_epoch" not in ks.keys()
        or payload["spin_period_epoch"] > ks["spin_period_epoch"]
        or np.isnan(ks["spin_period_epoch"])
    ):
        print(f"Updating {payload['source_name']} in the known source database.")
        db_api.update_known_source(ks["_id"], payload)
    else:
        print(
            f"Parfile for {payload['source_name']} is older than the entry in the known"
            " source database. Not adding it to the database"
        )
    return


if __name__ == "__main__":
    """
    The script loops through the parfiles in a directory to extract the relevant values
    to be added to the known source database.

    The attributes extracted are :
    ['source_type', 'source_name', 'pos_ra_deg', 'pos_dec_deg', 'pos_error_semimajor_deg',
    'pos_error_semiminor_deg', 'pos_error_theta_deg', 'dm', 'dm_error', 'spin_period_s',
    'spin_period_s_error', 'dm_galactic_ne_2001_max', 'dm_galactic_ymw_2016_max', 'spin_period_derivative',
    'spin_period_derivative_error', 'spin_period_epoch', 'survey']
    """
    parser = argparse.ArgumentParser(
        description=(
            "converting a directory of parfiles into entries for the known source"
            " database"
        )
    )
    parser.add_argument(
        "path",
        metavar="path",
        type=str,
        help="the path to the directory containing the parfiles",
    )
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
        "--survey",
        nargs="+",                    # one or more arguments as a list
        default=["champss"],          # default list
        type=str,
        help="Survey(s) for the new source",
    )
    args = parser.parse_args()
    tzpar_path = args.path
    print(tzpar_path)
    db_port = args.db_port
    db_host = args.db_host
    db_name = args.db_name
    survey = args.survey
    dmm = DMMap()
    for f in sorted(glob.glob(f"{tzpar_path}/*.par")):
        payload = {}
        elat = np.nan
        elat_err = np.nan
        elong = np.nan
        elong_err = np.nan
        fake_parfile = False
        with open(f) as infile:
            for line in infile:
                if len(line.split()) == 0:
                    continue
                if "PSR" in line.split()[0]:
                    payload["source_name"] = line.split()[1]
                    payload["source_type"] = 1
                if line.split()[0] == "RAJ":
                    if len(line.split()[1].split(":")) < 3:
                        ra_sec = 0.0
                    else:
                        ra_sec = float(line.split()[1].split(":")[2])
                    ra = (
                        float(line.split()[1].split(":")[0]) * 15
                        + float(line.split()[1].split(":")[1]) * 15 / 60
                        + ra_sec * 15 / 3600
                    )
                    if len(line.split()) > 2:
                        ra_err = float(line.split()[-1]) * 15 / 3600
                    else:
                        ra_err = 0.068
                    payload["pos_ra_deg"] = ra
                    payload["pos_error_semimajor_deg"] = ra_err
                if line.split()[0] in ("ELAT", "BETA"):
                    elat = float(line.split()[1])
                    if len(line.split()) > 2:
                        elat_err = float(line.split()[-1])
                if line.split()[0] in ("ELONG", "LAMBDA"):
                    elong = float(line.split()[1])
                    if len(line.split()) > 2:
                        elong_err = float(line.split()[-1])
                if line.split()[0] == "DECJ":
                    if len(line.split()[1].split(":")) < 3:
                        dec_sec = 0.0
                    else:
                        dec_sec = float(line.split()[1].split(":")[2])
                    if len(line.split()[1].split(":")) < 2:
                        dec_min = 0.0
                    else:
                        dec_min = float(line.split()[1].split(":")[1])
                    dec = (
                        float(line.split()[1].split(":")[0])
                        + dec_min / 60
                        + dec_sec / 3600
                    )
                    if len(line.split()) > 2:
                        dec_err = float(line.split()[-1]) / 3600
                    else:
                        dec_err = 0.068
                    payload["pos_dec_deg"] = dec
                    payload["pos_error_semiminor_deg"] = dec_err
                if line.split()[0] == "F0":
                    payload["spin_period_s"] = 1 / float(
                        line.split()[1].replace("D", "e")
                    )
                    if len(line.split()) > 2:
                        p_err = (1 / (float(line.split()[1]) ** 2)) * float(
                            line.split()[-1]
                        )
                    else:
                        p_err = payload["spin_period_s"] * 1e-4
                    payload["spin_period_s_error"] = p_err
                if line.split()[0] == "F1":
                    payload["spin_period_derivative"] = -(
                        payload["spin_period_s"] ** 2
                    ) * float(line.split()[1].replace("D", "e"))
                    if len(line.split()) > 2:
                        pd_err = np.abs(
                            np.sqrt(
                                (
                                    (
                                        2
                                        * payload["spin_period_s_error"]
                                        / payload["spin_period_s"]
                                    )
                                    ** 2
                                )
                                + (
                                    float(line.split()[-1].replace("D", "e"))
                                    / float(line.split()[1].replace("D", "e"))
                                )
                                ** 2
                            )
                            * payload["spin_period_s"] ** 2
                            * float(line.split()[1].replace("D", "e"))
                        )
                    else:
                        pd_err = 0.05 * payload["spin_period_derivative"]
                    payload["spin_period_derivative_error"] = pd_err
                if line.split()[0] == "P0":
                    payload["spin_period_s"] = float(line.split()[1])
                    if len(line.split()) > 2:
                        p_err = float(line.split()[-1])
                    else:
                        p_err = payload["spin_period_s"] * 1e-4
                    payload["spin_period_s_error"] = p_err
                if line.split()[0] == "DM":
                    payload["dm"] = line.split()[1]
                    if len(line.split()) > 2:
                        dm_err = line.split()[-1]
                    else:
                        dm_err = 0.1
                    payload["dm_error"] = dm_err
                if line.split()[0] == "PEPOCH":
                    payload["spin_period_epoch"] = float(line.split()[1])
                if "Fake" in line.split() or "test" in line.split():
                    print(f"{payload['source_name']} is a fake parfile")
                    fake_parfile = True
        if fake_parfile:
            continue
        if elat is not np.nan and elong is not np.nan:
            (
                payload["pos_ra_deg"],
                payload["pos_dec_deg"],
                payload["pos_error_semimajor_deg"],
                payload["pos_error_semiminor_deg"],
            ) = ra_dec_from_ecliptic(elong, elat, elong_err, elat_err)
        payload["dm_galactic_ne_2001_max"] = float(
            dmm.get_dm_ne2001(payload["pos_dec_deg"], payload["pos_ra_deg"])
        )
        payload["dm_galactic_ymw_2016_max"] = float(
            dmm.get_dm_ymw16(payload["pos_dec_deg"], payload["pos_ra_deg"])
        )
        payload["pos_error_theta_deg"] = 0.0
        if "spin_period_derivative" not in payload.keys():
            payload["spin_period_derivative"] = 0.0
            payload["spin_period_derivative_error"] = 0.0
        if survey != None:
            payload["survey"] = survey
        if len(payload.keys()) < 16:
            print(f"{payload['source_name']} is not complete")
            continue
        add_source_to_database(payload, db_port, db_host, db_name)
