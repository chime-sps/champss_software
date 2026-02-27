import glob
import os
import subprocess

import click
import numpy as np
from folding.utilities.archives import read_par
from folding.utilities.database import add_knownsource_to_fsdb, scrape_ephemeris
from sps_common.interfaces import MultiPointingCandidate
from sps_databases import db_api, db_utils


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--fs_id",
    type=str,
    default="",
    help="FollowUpSource ID, containing search results and candidate path.",
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver1",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps-processing",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--timingpath",
    default="/data/chime/sps/archives/known_sources/folded_profiles",
    type=str,
    help="Path for created files during fold step.",
)
def main(
    fs_id,
    db_port,
    db_host,
    db_name,
    timingpath,
):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    source = db_api.get_followup_source(fs_id)

    if source is None:
        print(f"FollowUpSource {fs_id} not found.")
        return

    parfile = source.path_to_ephemeris

    if source.folding_history:
        fold_dates = [entry["date"].date() for entry in source.folding_history]
        fold_SN = [entry["SN"] for entry in source.folding_history]
        archives = [entry["archive_fname"] for entry in source.folding_history]
    else:
        print(f"Source {fs_id} has no folding history in db, exiting...")
        return

    par_vals = read_par(parfile)
    name = par_vals["PSRJ"]

    base_dir = os.path.dirname(archives[0])
    source_dir = f"{timingpath}/{name}"
    print(f"Setting up for source: ", name)
    print(f"Copying archives to timing directory: {source_dir}")
    print(f"{archives}")
    if not os.path.exists(source_dir):
        os.mkdir(source_dir)
    # Copy parfile to timing directory
    subprocess.run(["rsync", "-aP", parfile, source_dir])

    # Copy archives to timing directory
    for archive in archives:
        subprocess.run(["rsync", "-aP", archive, source_dir])
        archiveFT = archive.replace(".ar", ".FT")
        subprocess.run(["rsync", "-aP", archiveFT, source_dir])

    # copy best-fit parfile to timing directory
    optimal_par = glob.glob(f"{base_dir}/*optimal.par")[0]
    new_par = f"{source_dir}/{name}.par"
    subprocess.run(["rsync", "-aP", optimal_par, new_par])

    # Add to FollowUpSource as timing / known_source
    print("Adding to FollowUpSource as timing / known_source")
    try:
        add_knownsource_to_fsdb(new_par)
    except Exception as e:
        print(f"Failed to add {name} to the database.")
        print(e)

    print("Applying best-fit ephemeris")
    os.chdir(source_dir)
    subprocess.run(["pam", "-E", optimal_par, "-m", "*.ar"])
    subprocess.run(["pam", "-E", optimal_par, "-m", "*.FT"])
    subprocess.run(["pam", "-T", "-e", ".T", "*.ar"])
    subprocess.run(["pam", "-F", "-T", "-e", ".FT", "*.ar"])
    subprocess.run(["psradd", "*.T", "-o", "added.T"])
    subprocess.run(["psradd", "*.FT", "-o", "added.FT"])


if __name__ == "__main__":
    main()
