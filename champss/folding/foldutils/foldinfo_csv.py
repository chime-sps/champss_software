import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sps_databases import db_api, db_utils


@click.command()
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Date of data to process. Default = Today in UTC",
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
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--basepath",
    default="/data/chime/sps/sps_processing",
    type=str,
    help="Path for created files during pipeline step.",
)
def foldinfo_csv(date, db_host, db_port, db_name, basepath):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    date_str = date.strftime("%Y%m%d")
    fname = f"{basepath}/mp_runs/daily_{date_str}/all_mp_cands.csv"
    df = pd.read_csv(fname)

    mp_cands = df["file_name"].values

    date = fname.split("daily_")[1][:8]
    folded = np.zeros_like(mp_cands)
    foldplots = [None] * len(mp_cands)
    foldsigmas = np.zeros_like(mp_cands)

    fs = list(db.followup_sources.find({"source_type": "sd_candidate"}))
    for i, fscand in enumerate(fs):
        candpath = fs[i]["path_to_candidates"][0]
        try:
            if candpath in mp_cands:
                j = np.argwhere(mp_cands == candpath).squeeze()
                plotpath = fs[i]["folding_history"][0]["path_to_plot"]
                folded[j] = 1
                foldplots[j] = plotpath
                foldsigmas[j] = fs[i]["folding_history"][0]["SN"]
        except Exception as e:
            print(candpath, e)
    df = df.assign(folded=folded)
    df = df.assign(fold_plotpath=foldplots)
    df = df.assign(fold_sigma=foldsigmas)
    fname = fname.replace(".csv", "_foldinfo.csv")
    # df.to_csv(fname)
    df.to_csv(f"/data/ramain/candidates/mpcands_{date}_foldinfo.csv")


if __name__ == "__main__":
    foldinfo_csv()
