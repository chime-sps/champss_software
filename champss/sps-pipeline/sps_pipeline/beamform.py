"""Executes the beamforming pipeline component."""

import logging
import os
from os import path

from beamformer.skybeam import SkyBeamFormer
from omegaconf import OmegaConf
from prometheus_client import Summary
from sps_pipeline import utils

log = logging.getLogger(__package__)

beamform_processing_time = Summary(
    "beamform_pointing_processing_seconds",
    "Duration of running the beamformer on a pointing",
    ("pointing", "beam_row"),
)


def run(pointing, beamformer, fdmt, num_threads, basepath):
    """
    Execute the beamforming step on a `pointing`

    Parameters
    ----------
    pointing: sps_common.interfaces.beamformer.ActivePointing
        Sky pointing to process.
    beamformer: Wrapper
        Wrapper class containing the SkyBeamFormer class.
    fdmt: bool
        Whether to return the skybeam object for fdmt instead of writing into a filterbank file.
    num_threads: int
        Number of threads to use for the parallel processing in SkyBeamFormer.
    basepath: str
        Folder which is used to store data.
    """
    tt = utils.transit_time(pointing)
    date = tt.date()
    log.info(f"Beamform ({pointing.ra:.2f} {pointing.dec:.2f}) @ {date:%Y-%m-%d}")
    log.info("Transit at: %s", tt.strftime("%Y-%m-%d %H:%M:%S %Z"))

    with beamform_processing_time.labels(
        pointing.pointing_id, pointing.beam_row
    ).time():
        # Beamform on the data
        skybeam = beamformer.bf.form_skybeam(pointing, num_threads)

        # Save data to disk for hhat
        file_path = path.join(
            basepath,
            date.strftime("%Y/%m/%d"),
            f"{pointing.ra:.02f}_{pointing.dec:.02f}",
        )
        os.makedirs(file_path, exist_ok=True)
        spectra_file = path.join(
            file_path,
            f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}.fil",
        )
        if fdmt:
            return skybeam
        log.info("Beamformed file: %s", spectra_file)
        skybeam.write(spectra_file)


def initialise(configuration, rfi_beamform, basepath, datpath):
    """"""
    if rfi_beamform:
        extn = "dat"
        basepath = datpath
    else:
        extn = "hdf5"
        basepath = basepath

    class Wrapper:
        def __init__(self, config):
            self.config = config
            # Get global_masking_dict from config if it exists, otherwise use empty dict
            global_masking_dict = (
                OmegaConf.to_container(config.rfi_global)
                if "rfi_global" in config
                else {}
            )
            self.bf = SkyBeamFormer(
                **OmegaConf.to_container(config.beamform),
                basepath=basepath,
                extn=extn,
                run_rfi_mitigation=rfi_beamform,
                masking_dict=OmegaConf.to_container(config.rfi),
                global_masking_dict=global_masking_dict,
            )

    return Wrapper(configuration)
