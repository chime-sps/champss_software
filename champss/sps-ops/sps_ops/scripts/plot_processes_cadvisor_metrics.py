"""Script to query cAdvisor metrics from Promtheus DB as seen on Grafana."""

import click
import matplotlib.pyplot as plotter
import pandas as pd
from pymongo import MongoClient

key_options = [
    "max_memory_usage",
    "max_cpu_usage",
    "date",
    "ra",
    "dec",
    "nchan",
    "ntime",
    "maxdm",
]


@click.command()
@click.option(
    "--start-date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Start date to query containers from",
)
@click.option(
    "--end-date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="End date to query containers to",
)
@click.option(
    "--x-axis-key",
    type=click.Choice(key_options),
    required=False,
    default="maxdm",
    help="What key to use for the X axis of the plot.",
)
@click.option(
    "--y-axis-key",
    type=click.Choice(key_options),
    required=False,
    default="max_memory_usage",
    help="What key to use for the Y axis of the plot.",
)
@click.option(
    "--colouring-key",
    type=click.Choice(key_options),
    required=False,
    default="nchan",
    help="What key to use for the colouring of the plot.",
)
def main(start_date, end_date, x_axis_key, y_axis_key, colouring_key):
    """Entry function to query and plot Processes MongoDB collection for cAdvisor
    metrics.
    """
    client = MongoClient(host="sps-archiver", port=27017)
    db = client["sps-processing"]
    collection = db["processes"]

    processes = list(
        collection.find(
            {"datetime": {"$gte": start_date, "$lte": end_date}, "status": {"$eq": 2}}
        )
    )

    plot_data = pd.DataFrame(processes)

    plot_title = f"{x_axis_key}_vs_{y_axis_key}_by{colouring_key}"

    plot = plot_data.plot.scatter(
        x=x_axis_key,
        y=y_axis_key,
        c=plot_data[colouring_key],
        colormap="viridis",
        title=plot_title,
    )

    plotter.savefig(f"{plot_title}.png")
