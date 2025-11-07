import datetime as dt
import time
import click


def convert_date_to_datetime(date):
    if isinstance(date, str) or isinstance(date, int):
        for date_format in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
            try:
                date = dt.datetime.strptime(str(date), date_format)
                break
            except ValueError:
                continue
    return date


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("wait_time", type=float)
def dummy_workflow_task(wait_time=None):
    time.sleep(wait_time)
    print(f"Task completed after {wait_time} seconds.")
    return {"wait_time": wait_time}, [], []