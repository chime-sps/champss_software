"""Executes the dedispersion pipeline component."""

import logging
import os

import docker
import numpy as np

# from sps_dedispersion.fdmt.cpu_fdmt import FDMT
from prometheus_client import Summary
from sps_common.constants import DM_CONSTANT, FREQ_BOTTOM, FREQ_TOP, TSAMP
from sps_dedispersion.dedisperse import dedisperse
from sps_pipeline import utils

log = logging.getLogger(__package__)

dedisp_processing_time = Summary(
    "dedisperse_pointing_processing_seconds",
    "Duration of running dedispersion on a pointing",
    ("pointing", "beam_row"),
)


def run_fdmt(pointing, skybeam, config, num_threads=1):
    """
    Execute the dedispersion step with fdmt on a `pointing`

    Parameters
    ----------
    pointing: sps_common.interfaces.beamformer.ActivePointing
        Sky pointing to process.
    skybeam: sps_common.interface.beamformer.SkyBeam
        SkyBeam object containing the spectra to process.
    config:
        OmegaConf instance with the configuration
    """
    date = utils.transit_time(pointing).date()
    log.info(
        f"FDMT Dedispersion ({pointing.ra:.2f} {pointing.dec:.2f}) @ {date:%Y-%m-%d}"
    )
    fdmt_config = config.dedisp.fdmt
    num_dms_fac = fdmt_config.num_dms_fac

    # check config for maximum dm to process
    if config.dedisp.maxdm:
        maxdm = np.min([pointing.maxdm, config.dedisp.maxdm])
    else:
        maxdm = pointing.maxdm
    num_dms = (
        int(
            DM_CONSTANT
            * maxdm
            * (1 / FREQ_BOTTOM**2 - 1 / FREQ_TOP**2)
            / TSAMP
            // num_dms_fac
        )
        + 1
    ) * num_dms_fac
    nsamps = fdmt_config.chunk_size + num_dms

    if fdmt_config.cpp == True:
        log.info("Using C++ FDMT")
        from dmt.libdmt import FDMT

        fdmt = FDMT(
            FREQ_BOTTOM,
            FREQ_TOP,
            skybeam.nchan,
            nsamps,
            TSAMP,
            dt_max=num_dms,
            dt_min=0,
            dt_step=1,
        )
        fdmt_threads = np.min([num_threads, 8])
        #fdmt.set_num_threads(num_threads)  # may want to cap at 8 threads
        fdmt.set_num_threads(fdmt_threads)
    else:
        log.info("Using Python FDMT")
        from sps_dedispersion.fdmt.cpu_fdmt import FDMT

        fdmt = FDMT(
            fmin=FREQ_BOTTOM,
            fmax=FREQ_TOP,
            nchan=skybeam.nchan,
            maxDT=num_dms,
            num_threads=num_threads,
        )

    dts = dedisperse(
        fdmt,
        skybeam,
        fdmt_config.chunk_size,
        num_dms,
        fdmt_config.dm_step,
        fdmt_config.cpp,
    )

    log.info(
        f"FDMT Dedispersion ({pointing.ra:.2f} {pointing.dec:.2f}) completed with"
        f" {len(dts.dedisp_ts)} DM steps"
    )
    return dts


def run(pointing, config, basepath):
    """
    Execute the dedispersion step on a `pointing`

    Parameters
    ----------
    pointing: sps_common.interfaces.beamformer.ActivePointing
        Sky pointing to process.
    config:
        OmegaConf instance with the configuration
    """
    date = utils.transit_time(pointing).date()
    log.info(
        f"Presto Dedispersion ({pointing.ra:.2f} {pointing.dec:.2f}) @ {date:%Y-%m-%d}"
    )

    file_path = os.path.join(
        "/data",
        basepath,
        date.strftime("%Y/%m/%d"),
        f"{pointing.ra:.02f}_{pointing.dec:.02f}",
        f"{pointing.ra:.02f}_{pointing.dec:.02f}_{pointing.sub_pointing}",
    )
    outfile_base = file_path + "_timeseries"
    infile = file_path + ".fil"
    # check config for maximum dm to process
    if config.dedisp.maxdm:
        maxdm = np.min([pointing.maxdm, config.dedisp.maxdm])
    else:
        maxdm = pointing.maxdm
    with dedisp_processing_time.labels(pointing.pointing_id, pointing.beam_row).time():
        for i in np.arange(0, maxdm / 0.125, config.dedisp.numdms):
            presto_dedisperse(
                lodm=i * 0.125,
                numdms=config.dedisp.numdms,
                nsub=config.dedisp.nsub,
                outfile=outfile_base,
                infile=infile,
            )


def presto_dedisperse(
    lodm=0,
    numdms=800,
    dmstep=0.125,
    dmprec=3,
    nsub=64,
    outfile="/frb-archiver/SPS/presto_test/test",
    infile="/frb-archiver/SPS/presto_test/test.fil",
):
    client = docker.from_env()
    log.info(f"Starting Presto: {infile}")
    output = client.containers.run(
        "chimefrb/presto",
        remove=True,
        detach=False,
        stream=True,
        volumes={os.getcwd(): {"bind": "/data", "mode": "rw"}},
        command=[
            "prepsubband",
            "-lodm",
            str(lodm),
            "-numdms",
            str(numdms),
            "-dmstep",
            str(dmstep),
            "-noclip",
            "-nobary",
            "-dmprec",
            str(dmprec),
            "-nsub",
            str(nsub),
            "-o",
            outfile,
            infile,
        ],
    )
    log.info("Presto completed: { infile }")
    for line in output:
        log.info(line.decode().rstrip(os.linesep))
