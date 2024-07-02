import numpy as np
from sps_databases import db_api, db_utils


def scrape_ephemeris(ephem_path):
    from beamformer.utilities.dm import DMMap

    dmm = DMMap()

    payload = {}
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
    ephem_path_absolute = os.path.abspath(f)
    payload["path_to_ephemeris"] = ephem_path_absolute
    return payload


def add_knownsource_to_fsdb(ephem_path):
    """Create a payload for a known source to be input in the followup sources
    database.
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
    """Create a payload for a candidate to be input in the followup sources database."""
    from beamformer.utilities.dm import DMMap

    dmm = DMMap()
    db = db_utils.connect()

    if source_type == "md_candidate":
        duration = 30
    elif source_type == "sd_candidate":
        duration = 1
    else:
        print("must be either md or sd candidate")
        return

    ra_name = np.round(float(ra), 2)
    dec_name = np.round(float(dec), 2)
    name = f"{date_str}_{source_type[:2]}_{ra_name}_{dec_name}_{f0}_{dm}"

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
        followup_source = db_api.create_followup_source(payload)
        return followup_source
    else:
        print(
            f"Source {payload['source_name']} already in the follow-up source database."
        )
        return fs


def add_mdcand_from_candpath(candpath, date):
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
