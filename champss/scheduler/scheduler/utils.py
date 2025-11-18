import datetime as dt
import time
import click
import atexit
import signal

import logging

log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)
file_handler = logging.FileHandler("/data/lkuenkel/dummy_lof.txt")
# file_handler.setFormatter(
#     logging.Formatter(fmt=config.logging.format, datefmt="%b %d %H:%M:%S")
# )
logging.root.addHandler(file_handler)
logging.root.setLevel("INFO")


def continue_running(signum, frame):
    log.info("Received Interrupt.")

def signal_handler(sig, frame):
    log.info(f"Received signal: {sig} ({signal.Signals(sig).name})")
    # Add any other desired actions here

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
@click.option("--wait_time", type=float, default=10)
def dummy_workflow_task(wait_time):
    # signal.signal(signal.SIGTERM, continue_running)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    # signal.signal(signal.SIGINT, signal.SIG_IGN)
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    # signal.signal(signal.SIGKILL, signal_handler)
    # signal.signal(signal.SIGKILL, signal.SIG_IGN)
    log.info(f"Task started. Will wait for {wait_time} seconds.")
    time.sleep(wait_time)
    log.info(f"Task completed after {wait_time} seconds.")
    return {"wait_time": wait_time}, [], []
