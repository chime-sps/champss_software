import os
import shutil
import time
from datetime import datetime, timedelta, timezone

import click
import pytz
from slack_sdk import WebClient
from sps_databases import db_utils


def delete_raw_data_folder(date, raw_data_path, dry_run):
    raw_data_folder = f"{raw_data_path}/{date.year}/{date.month:02d}/{date.day:02d}"

    print(f"\tAttempting to delete {raw_data_folder}...")

    if os.path.isdir(raw_data_folder) == False:
        print(f"\tRaw data folder is already deleted.")
        return

    slack_token = "xoxb-194910630096-6273790557189-FKbg9w1HwrJYqGmBRY8DF0te"
    slack_client = WebClient(token=slack_token)
    slack_channel = "#slow-pulsar-alerts"
    slack_warning = slack_client.chat_postMessage(
        channel=slack_channel,
        text=(
            f"{raw_data_folder} will be deleted in 1 hour. Reply 'skip' to skip this"
            " day's raw data deletion."
        ),
    )

    time.sleep(3600)

    slack_replies = slack_client.conversations_replies(
        channel=slack_warning["channel"],
        ts=slack_warning["ts"],
        inclusive=True,
        limit=10,
    )

    for message in slack_replies["messages"]:
        if message["text"] == "skip":
            slack_client.chat_postMessage(
                channel=slack_channel, text="Skip received. Will not delete folder."
            )
            return

    slack_client.chat_postMessage(
        channel=slack_channel, text="No skip was received. Will delete folder."
    )

    try:
        if not dry_run:
            shutil.rmtree(raw_data_folder, ignore_errors=True)
        print(f"\tRaw data folder is now deleted.")
    except Exception as error:
        print(f"\tRaw data folder could not be deleted: {error}")


@click.command()
@click.option(
    "--db-host",
    default="sps-archiver",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps-processing",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--start-date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Only delete raw data for date's after this date.",
)
@click.option(
    "--days-threshold",
    default=7,
    type=int,
    help="Number of days since date to allow deletion of date's raw data",
)
@click.option(
    "--completed-threshold",
    default=0.9,
    type=float,
    help=(
        "Percentage of date's available processes that were processed "
        "successfully to stack to allow deletion of date's raw data"
    ),
)
@click.option(
    "--raw-data-path",
    default="/data/chime/sps/raw",
    type=str,
    help="Path to raw data for deletion",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Only print what it would do, not actually delete the folders.",
)
def clear_raw_data(
    db_host,
    db_port,
    db_name,
    start_date,
    days_threshold,
    completed_threshold,
    raw_data_path,
    dry_run,
):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)

    start_date = start_date.replace(tzinfo=pytz.UTC)

    print(f"Using start date {start_date}\n")

    while True:
        processes = db["processes"]
        dates = sorted(processes.distinct("date"))

        print(f"All dates to consider for raw data deletion: {dates}\n")

        for date_string in dates:
            for date_format in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
                try:
                    date = datetime.strptime(date_string, date_format).replace(
                        tzinfo=pytz.UTC
                    )
                except ValueError:
                    continue
            print(f"Checking if {date.date()}'s raw data is ready for deletion.")

            # Check if at least "days_threshold" days have passed since this date's
            # last processed "last_changed" value
            processes_for_date = processes.find({"date": date_string})

            # Sort by the 'last_changed' attribute in descending order and get
            # the most recent process
            last_process_for_date = processes_for_date.sort("last_changed", -1)[0]
            last_process_for_date_timestamp = last_process_for_date.get("last_changed")

            print(
                f"Most recent process for {date} has last timestamp of"
                f" {last_process_for_date_timestamp}"
            )

            if (
                datetime.now(timezone.utc) - last_process_for_date_timestamp
            ) < timedelta(days=days_threshold):
                print(
                    f"\tLess than {days_threshold} days have passed since this "
                    "date's last processed timestamp, skipping processes completion "
                    "check, will not delete date's raw data."
                )
                continue

            # Check if at least "completed_threshold" % of processes for this date
            # have been added to the stack
            completed_processes = processes.count_documents(
                {"date": date_string, "status": 2, "folded_status": True}
            )
            total_processes = processes.count_documents({"date": date_string})
            percentage_processes = completed_processes / total_processes

            if percentage_processes >= completed_threshold:
                print(
                    f"\tMore than {completed_threshold * 100}% of date's processed are"
                    f" in stack (has {percentage_processes * 100}%), will delete date's"
                    " raw data."
                )
                delete_raw_data_folder(date, raw_data_path, dry_run)
            else:
                print(
                    f"\tLess than {completed_threshold * 100}% of date's processes are"
                    f" in stack (only {percentage_processes * 100}%), will not delete"
                    " date's raw data."
                )
                # This date might just have a lot of failures, so continue checking
                # next dates (do not break out of this loop)

        # Wait an hour between checking all dates again
        time.sleep(3600)
