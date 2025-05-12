"""Executes the Hhat pipeline component."""

import logging
import os
from os import path

import docker
from prometheus_client import Summary
from sps_pipeline import utils

log = logging.getLogger(__package__)

hhat_processing_time = Summary(
    "hhat_pointing_processing_seconds",
    "Duration of creating a Hhat for a pointing",
    ("pointing", "beam_row"),
)


def run(pointing):
    """
    Search for periodicity on a `pointing` using Hhat.

    Parameters
    ----------
    pointing: sps_common.interfaces.beamformer.ActivePointing
        Sky pointing to process.
    """
    date = utils.transit_time(pointing).date()
    log.info(f"Hhat ({pointing.ra:.2f} {pointing.dec:.2f}) @ {date:%Y-%m-%d}")

    client = docker.from_env()

    file_path = path.join(
        "/data",
        date.strftime("%Y/%m/%d"),
        f"{pointing.beam_row:03d}",
    )
    out_path = path.join(file_path, f"{pointing.ra:.02f}_{pointing.dec:.02f}")

    with hhat_processing_time.labels(pointing.pointing_id, pointing.beam_row).time():
        output = client.containers.run(
            "chime-sps/proton-hhat:v20201207",
            remove=True,
            detach=False,
            stream=True,
            volumes={os.getcwd(): {"bind": "/data", "mode": "rw"}},
            command=[
                "hhat_from_presto_dat",
                "--dat",
                file_path,
                "--out",
                out_path,
                "--rednoise",
            ],
        )
    for line in output:
        log.info(line.decode().rstrip(os.linesep))
