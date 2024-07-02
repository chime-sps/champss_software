import datetime as dt
import logging

import click
import multiday_search.confirm_cand as confirm_cand
import multiday_search.fold_multiday as fold_multiday
from foldutils.database_utils import add_mdcand_from_candpath
from sps_databases import db_api, db_utils, models

log = logging.getLogger()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--candpath",
    type=str,
    default="",
    help="Path to candidate file",
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
)
def main(candpath, db_port, db_host, db_name):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    fs_id = add_mdcand_from_candpath(candpath, dt.datetime.now())
    print(fs_id)
    fold_multiday.main(
        [
            "--fs_id",
            fs_id,
            "--db-port",
            db_port,
            "--db-name",
            db_name,
            "--db-host",
            db_host,
        ],
        standalone_mode=False,
    )
    print("outside of fold_multiday")
    confirm_cand.main(
        [
            "--fs_id",
            fs_id,
            "--db-port",
            db_port,
            "--db-name",
            db_name,
            "--db-host",
            db_host,
        ],
        standalone_mode=False,
    )

    # Silence Workflow errors, requires results, products, plots
    return {}, [], []


if __name__ == "__main__":
    main()
