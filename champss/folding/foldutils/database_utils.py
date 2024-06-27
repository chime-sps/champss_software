from sps_databases import db_api, db_utils


def add_candidate_to_fsdb(
    date_str, ra, dec, f0, dm, sigma, cand_path="", source_type="sd_candidate"
):
    """Create a payload for a candidate to be input in the followup sources database."""
    from beamformer.utilities.dm import DMMap

    dmm = DMMap()
    db_utils.connect()

    if source_type == "md_candidate":
        duration = 30
    elif source_type == "sd_candidate":
        duration = 1
    else:
        print("must be either md or sd candidate")
        return

    name = f"{date_str}_{source_type[:2]}_{ra}_{dec}_{f0}_{dm}"

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
    }

    payload["dm_galactic_ne_2001_max"] = float(
        dmm.get_dm_ne2001(payload["dec"], payload["ra"])
    )
    payload["dm_galactic_ymw_2016_max"] = float(
        dmm.get_dm_ymw16(payload["dec"], payload["ra"])
    )

    db_api.create_followup_source(payload)
    print(f"Added {name} to the follow-up source database")
