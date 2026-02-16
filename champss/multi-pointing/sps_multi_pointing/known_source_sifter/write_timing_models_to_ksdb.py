import argparse
import glob

from sps_databases import db_api, db_utils
from folding.utilities.archives import read_par

if __name__ == "__main__":
    """
    Update known source database with timing model.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Write content of folders containing par files to known source database."
        )
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
        "--folder",
        default="/data/timing/timing_sources/",
        type=str,
        help="Folder which contains the sub folder containing the par files",
    )
    args = parser.parse_args()
    db = db_utils.connect(host=args.db_host, port=args.db_port, name=args.db_name)
    par_files = glob.glob(args.folder + "/*/pulsar.par")
    for par_file in par_files:
        psr = par_file.split("/")[-2]
        print(par_file)
        try:
            timing_model = read_par(par_file)
        except:
            print(par_file, "could not be loaded.")
            continue
        print(f"Read {par_file} with {len(timing_model)} parameters.")
        try:
            known_source_entry = db_api.get_known_source_by_names(psr)[0]
        except:
            print(f"Could not retrieve pulsar {psr} from database")
        if len(known_source_entry):
            payload = {"timing_model": timing_model}
            db_api.update_known_source(known_source_entry.id, payload)
