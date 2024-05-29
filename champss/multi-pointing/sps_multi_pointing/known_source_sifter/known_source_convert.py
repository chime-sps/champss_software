"""
Add the things in the .npy file into the mongodb.

This file can be run once
to populate the initial `KnownSources` collection.
"""

import bson
import numpy as np
import pymongo

if __name__ == "__main__":
    client = pymongo.MongoClient()
    known_source_db = client.sps.known_sources

    ks_np_db = np.load("ks_database_210203_1541.npy")
    ks_db_dicts = [dict(zip(ks_np_db.dtype.names, x)) for x in ks_np_db]

    for entry in ks_db_dicts:
        # Use existing id as ObjectID in hex
        entry["_id"] = bson.ObjectId("{:0>24x}".format(entry["id"]))

        # Convert numpy types into bson-compatible base python types
        entry["source_type"] = int(entry["source_type"])
        entry["pos_ra_deg"] = float(entry["pos_ra_deg"])
        entry["pos_dec_deg"] = float(entry["pos_dec_deg"])
        entry["pos_error_semimajor_deg"] = float(entry["pos_error_semimajor_deg"])
        entry["pos_error_semiminor_deg"] = float(entry["pos_error_semiminor_deg"])
        entry["pos_error_theta_deg"] = float(entry["pos_error_theta_deg"])
        entry["dm"] = float(entry["dm"])
        entry["dm_error"] = float(entry["dm_error"])
        entry["spin_period_s"] = float(entry["spin_period_s"])
        entry["spin_period_s_error"] = float(entry["spin_period_s_error"])
        entry["dm_galactic_ne_2001_max"] = float(entry["dm_galactic_ne_2001_max"])
        entry["dm_galactic_ymw_2016_max"] = float(entry["dm_galactic_ymw_2016_max"])

        # Delete remaining two superfluous entries
        del entry["id"]  # (using _id)
        # this and the field below are 0 for all entries
        del entry["spectral_index"]
        del entry["chime_frb_peak_flux_density_jy"]

    # Don't add these known sources if any are already in the database
    found_result = known_source_db.find_one(
        {
            "$or": [{"_id": entry["_id"]} for entry in ks_db_dicts]
            + [{"source_name": entry["source_name"]} for entry in ks_db_dicts]
        }
    )
    if found_result is not None:
        raise ValueError(
            "Couldn't put the known source catalogue into the "
            "database because at least one entry already exists:\n"
            "{}".format(found_result)
        )

    print(ks_np_db.dtype)

    # Was everything inserted propertly?
    insert_result = known_source_db.insert_many(ks_db_dicts)
    if set(insert_result.inserted_ids) == {entry["_id"] for entry in ks_db_dicts}:
        print("Inserted all the entries. Done.")
    else:
        print("Didn't insert all the entries. Double-check the database.")
