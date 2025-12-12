import glob

import click
from folding.utilities.database import add_knownsource_to_fsdb, scrape_ephemeris
from sps_databases import db_api, db_utils


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
    default="sps-archiver1",
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
    for f in sorted(glob.glob(f"{tzpar_path}/*.par")):
        payload = scrape_ephemeris(f)
        try:
            add_knownsource_to_fsdb(payload)
        except Exception as e:
            print(f"Failed to add {payload['source_name']} to the database.")
            print(e)


if __name__ == "__main__":
    main()
