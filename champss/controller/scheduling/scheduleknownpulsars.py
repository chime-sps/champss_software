import datetime
import logging
import os
import signal
import subprocess  # nosec
import time
import atexit
from dataclasses import dataclass
from typing import Any

import astropy.units as u
import click
import pymongo
from astropy.time import Time
from beamformer.strategist.strategist import PointingStrategist
from sps_databases import db_api, db_utils

# Folder mtime age (seconds) below which a beam is considered "actively
# recording" by is_beam_recording(). Also used as the grace period during
# which a beam we ourselves just stopped is still remembered as "ours", so a
# second pulsar transiting the same beam shortly after isn't mistaken for an
# externally-controlled recording.
BEAM_RECORDING_GRACE_PERIOD = 600  # 10 minutes


@dataclass
class PulsarSchedule:
    """
    Per-pulsar scheduling state: identity, current pointing, whether an
    acquisition is currently active for it, and the process/marker recording
    its beam.
    """

    psr: str
    pointing: Any  # the ap value returned by PointingStrategist
    active: bool = False
    # One of: 0 (acquisition not started), subprocess.Popen (we own the
    # process), "external" (beam is controlled by something else), or an int
    # beamrow (handoff placeholder: another pulsar in this schedule holds the
    # actual process/marker for this shared beam).
    process: Any = 0


@dataclass
class BeamState:
    """
    Per-beam scheduling state: how many currently-active pulsars share this
    beam, and when we last stopped a process we owned on it (if ever).
    """

    active_count: int = 0
    last_stopped: float | None = None


def setup_logger(logfile="schedknownpsrlog.txt"):
    """
    Set up logger with both console and file output.

    Args:
        logfile: Path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("scheduleknownpulsars")
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(logfile, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_champss_fm_sources(
    server_url="mongodb://sps-archiver1:27017/", db_name="timing_ops"
):
    # Initialize connection and cursor
    client = pymongo.MongoClient(server_url)

    # Create database if it does not exist
    database = client[db_name]

    # Setup
    collection = database["sources"]

    # Get sources
    return list(collection.find({"champss_foldmode": True}))


def get_pulsar_radec(psr):
    """
    Return ra and dec for a pulsar from the known_source database
    psr: string, B name if it exists following our DB convention
    """
    source = db_api.get_known_source_by_name(psr)[0]
    ra = source.pos_ra_deg
    dec = source.pos_dec_deg
    return ra, dec


def update_psr_list(schedule, pst, logger):
    """
    Update the pulsar list by querying the timing_ops database. 

    Compares current pulsar list with database, adds new pulsars with
    champss_foldmode=True, and removes pulsars that no longer have it enabled.

    Args:
        schedule: Current list of PulsarSchedule entries
        pst: PointingStrategist object
        logger: Logger instance

    Returns:
        Updated list of PulsarSchedule entries
    """
    logger.debug("Checking database for pulsar list updates...")

    # Query database for current pulsar list
    new_pulsar_entries = get_champss_fm_sources()
    new_psr_ids = {entry["psr_id"] for entry in new_pulsar_entries}
    current_psr_ids = {entry.psr for entry in schedule}

    # Find new pulsars to add
    pulsars_to_add = new_psr_ids - current_psr_ids
    if pulsars_to_add:
        logger.info(
            f"Adding {len(pulsars_to_add)} new pulsar(s): {sorted(pulsars_to_add)}"
        )

        for entry in new_pulsar_entries:
            psr = entry["psr_id"]
            if psr in pulsars_to_add:
                ra = entry["ra"]
                dec = entry["dec"]
                Dnow_update = datetime.datetime.now()
                ap = pst.get_single_pointing(ra, dec, Dnow_update, use_grid=False)
                beamrow = ap[0].max_beams[0]["beam"]
                schedule.append(PulsarSchedule(psr=psr, pointing=ap))
                logger.info(f"Added {psr} (beam {beamrow})")
                logger.info(f"Updated pulsar count: {len(schedule)}")

    # Find pulsars to remove (champss_foldmode=False in DB)
    pulsars_to_remove = current_psr_ids - new_psr_ids
    if pulsars_to_remove:
        logger.info(
            f"Removing {len(pulsars_to_remove)} pulsar(s): {sorted(pulsars_to_remove)}"
        )

        schedule = [entry for entry in schedule if entry.psr not in pulsars_to_remove]
        logger.info(f"Updated pulsar count: {len(schedule)}")

    if not pulsars_to_add and not pulsars_to_remove:
        logger.debug("No changes to pulsar list")

    return schedule


def is_beam_recording(beam, basepath, source="champss"):
    """
    Check if a beam is currently recording by checking folder modification time.

    This prevents interference with externally-controlled beams (e.g., spsctl) by
    checking the modification time of the beam's data folder.

    Args:
        beam: Beam row number to check
        basepath: Base path for data storage (e.g., "/sps-archiver2/raw/")
        source: Writer source ("champss" or "slow")

    Returns:
        bool: True if beam is actively recording, False otherwise
    """
    logger = logging.getLogger("scheduleknownpulsars")

    # Convert between names between basepath (L1) to local mount path
    local_path = (
        basepath.replace("/sps-archiver1/", "/data/")
        .replace("/sps-archiver2/", "/mnt/beegfs-client/")
        .replace("/sps-archiver3/", "/mnt/beegfs-client/")
        .replace("/sps-archiver4/", "/mnt/beegfs-client/")
        .replace("/sps-archiver5/", "/mnt/beegfs-client/")
    )

    # Today's
    data_folder = (
        local_path
        + datetime.datetime.utcnow().strftime("/%Y/%m/%d/")
        + f"{str(beam).zfill(4)}"
    )

    max_folder_age = BEAM_RECORDING_GRACE_PERIOD

    try:
        now = datetime.datetime.utcnow().timestamp()
        folder_age = now - os.path.getmtime(data_folder)

        # If folder was modified recently, beam is actively recording
        if folder_age < max_folder_age:
            return True

    except FileNotFoundError:
        # Folder doesn't exist, check previous day's folder
        previous_data_folder = (
            local_path
            + (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime(
                "/%Y/%m/%d/"
            )
            + f"{str(beam).zfill(4)}"
        )
        try:
            now = datetime.datetime.utcnow().timestamp()
            folder_age = now - os.path.getmtime(previous_data_folder)
            if folder_age < max_folder_age:
                return True
        except (FileNotFoundError, OSError):
            pass
    except OSError as e:
        logger.warning(f"Could not check folder age for beam {beam}: {e}")

    # No indicators of active recording found
    return False


# def recordexcepthook(type, value, tb):
def stop_processes():
    """
    Exception hook that stops all unstopped recordings on uncaught exception.
    """
    # Also print default output
    logger = globals()["logger"]
    logger.error("Will try to stop all recording jobs after brief pause.")
    # Brief pause to no interfere with other possible stops
    time.sleep(5.0)
    schedule = globals()["schedule"]
    for entry in schedule:
        if isinstance(entry.process, subprocess.Popen):
            logger.info("Stopping acq for one process.")
            try:
                entry.process.send_signal(signal.SIGINT)
            except Exception as e:
                logger.error(f"Could not stop process due to {e}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--psrfile",
    type=click.File("r"),
    default=None,
    help="Text file with list of pulsar names. If not provided, loads from timing_ops database.",
)
@click.option(
    "--logfile",
    type=str,
    default="schedknownpsrlog.txt",
    help="Log file for acquisition messages and console output.",
)
@click.option(
    "--basepath",
    type=str,
    default="/sps-archiver2/raw/",
    help="Path on L1 cf nodes to a CHAMPSS mount.",
)
@click.option(
    "--source",
    type=str,
    default="champss",
    help=(
        "The chime_slow_pulsar_writer object to use on L1, must be either 'champss' or 'slow'. "
        "Do not use 'slow' before consulting with the Slow team"
    ),
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
def main(psrfile, logfile, basepath, source, db_port, db_host, db_name):
    """
    Record CHAMPSS data when known pulsars are transitting.

    By default, loads pulsars from timing_ops database (champss_foldmode=True).
    Optionally, can load from a text file using --psrfile.

    For each pulsar, spsctl is run on the transitting beam when a source
    transits (+- transit_buffer), then an interrupt sent to the process.
    It checks every minute to start/stop acquisition, with logic in place to
    continue recording as long as there is still one pulsar in the beamrow.

    Periodically checks pulsars to add/remove from the timing_ops DB

    Prevents interference with externally-controlled beams by checking folder
    modification times before starting/stopping acquisitions.
    """
    # Setup logger for both console and file output
    global logger
    logger = setup_logger(logfile)

    logger.info("Starting scheduleknownpulsars controller")
    logger.info(f"Logfile: {logfile}")
    logger.info(f"L1 Basepath: {basepath}")
    logger.info(f"Source: {source}")
    logger.info(f"Database: {db_name} at {db_host}:{db_port}")

    db_utils.connect(host=db_host, port=db_port, name=db_name)
    pst = PointingStrategist(create_db=False)

    Dnow = datetime.datetime.now()
    global schedule
    schedule = []
    # beamrow -> BeamState (active pulsar count + last time we stopped a
    # process we owned on that beam), created on demand as beams are seen.
    beam_state = {}
    transit_buffer = 3 * u.min
    db_check_interval = 600

    # Track last database check time (for periodic updates in database mode)
    last_db_check = Time.now()
    use_database_mode = psrfile is None

    # Load pulsar list from database or text file
    if psrfile is None:
        # Load from timing_ops database
        logger.info("Loading pulsars from timing_ops database (champss_foldmode=True)")
        logger.info(
            f"Database check interval: {db_check_interval} seconds ({db_check_interval / 60:.1f} minutes)"
        )
        pulsar_entries = get_champss_fm_sources()
        logger.info(f"Found {len(pulsar_entries)} pulsars in database")

        logger.info("Acquiring pointings for all pulsars in database")
        seen_psrs = set()
        for entry in pulsar_entries:
            psr = entry["psr_id"]
            ra = entry["ra"]
            dec = entry["dec"]

            # avoid duplicates
            if psr not in seen_psrs:
                ap = pst.get_single_pointing(ra, dec, Dnow, use_grid=False)
                beamrow = ap[0].max_beams[0]["beam"]
                schedule.append(PulsarSchedule(psr=psr, pointing=ap))
                seen_psrs.add(psr)
                logger.info(f"{psr} (beam {beamrow})")
            else:
                logger.info(f"{psr} duplicated in database")
    else:
        # Load from text file
        logger.info("Loading pulsars from text file")
        logger.info("Acquiring pointings for all pulsars in list")
        seen_psrs = set()
        for psr in psrfile:
            psr = psr.strip()
            # avoid duplicates
            if psr not in seen_psrs:
                ra, dec = get_pulsar_radec(psr)
                ap = pst.get_single_pointing(ra, dec, Dnow, use_grid=False)
                beamrow = ap[0].max_beams[0]["beam"]
                schedule.append(PulsarSchedule(psr=psr, pointing=ap))
                seen_psrs.add(psr)
                logger.info(f"{psr} (beam {beamrow})")
            else:
                logger.info(f"{psr} duplicated in list")

    logger.info("Pointings loaded, running dynamic scheduler")
    # Stop recording on exit
    atexit.register(stop_processes)

    while True:
        Tnow = Time.now()

        # Periodically check database for pulsar list updates (database mode only)
        if use_database_mode:
            time_since_last_check = (Tnow - last_db_check).to(u.s).value
            if time_since_last_check >= db_check_interval:
                schedule = update_psr_list(schedule, pst, logger)
                last_db_check = Tnow

        for entry in schedule:
            ap = entry.pointing
            activeacq = entry.active
            Tend = Time(ap[0].max_beams[3]["utc_end"], format="unix")
            Tstart = Time(ap[0].max_beams[0]["utc_start"], format="unix")
            Tend = Tend + transit_buffer
            Tstart = Tstart - transit_buffer
            transit_duration = (Tend - Tstart).to(u.s).value
            beamrow = ap[0].max_beams[0]["beam"]
            time_to_transit = Tend.unix - Tnow.unix
            state = beam_state.setdefault(beamrow, BeamState())

            if (time_to_transit < transit_duration) and (time_to_transit > 0):
                if not activeacq:
                    if state.active_count == 0:
                        recently_stopped_by_us = (
                            state.last_stopped is not None
                            and (Tnow.unix - state.last_stopped)
                            < BEAM_RECORDING_GRACE_PERIOD
                        )
                        # If we ourselves stopped this beam a moment ago, any
                        # residual folder activity is a leftover of our own
                        # process shutting down, not external control - skip
                        # straight to starting a fresh acquisition. Otherwise,
                        # check if beam is already recording (e.g., by spsctl
                        # or another controller).
                        if recently_stopped_by_us or not is_beam_recording(
                            beamrow, basepath, source
                        ):
                            logger.info(
                                f"Starting acq, {entry.psr} transitting row {beamrow}"
                            )
                            processi = subprocess.Popen(
                                [
                                    "spsctl",
                                    f"{beamrow}",
                                    "--basepath",
                                    f"{basepath}",
                                    "--source",
                                    f"{source}",
                                ],
                                shell=False,
                            )  # nosec
                            entry.process = processi
                        else:
                            logger.info(
                                f"{entry.psr} transitting row {beamrow}, beam already "
                                f"recording, will not interfere"
                            )
                            # Mark as externally controlled - don't start spsctl
                            entry.process = "external"
                    else:
                        logger.info(
                            f"{entry.psr} transitting row {beamrow}, continuing acq"
                        )
                        # logic to include process id for beams with 2+ pulsars
                        entry.process = beamrow
                    state.active_count += 1
                    entry.active = True
            elif time_to_transit < 0:
                if (state.active_count == 1) and activeacq:
                    processi = entry.process
                    # Only stop if we control this beam (not externally controlled)
                    if isinstance(processi, subprocess.Popen):
                        logger.info(f"Stopping acq, {entry.psr} row {beamrow}")
                        processi.send_signal(signal.SIGINT)
                        state.last_stopped = Tnow.unix
                    elif processi == "external":
                        logger.info(
                            f"Not stopping {entry.psr} row {beamrow}, "
                            f"beam controlled externally"
                        )
                    state.active_count -= 1
                    entry.active = False
                elif (state.active_count > 1) and activeacq:
                    logger.info(f"Continuing acq, removing {entry.psr} row {beamrow}")
                    state.active_count -= 1
                    entry.active = False
                    # passing process id to next pulsar in same beamrow
                    handoff_entry = next(
                        e for e in schedule if e.process == beamrow
                    )
                    handoff_entry.process = entry.process
                    entry.process = 0

                # update pointing to current time, plan next transit in ~24 hours
                Dnow = datetime.datetime.now()
                # ra, dec = get_pulsar_radec(psr)
                ra = ap[0].ra
                dec = ap[0].dec
                entry.pointing = pst.get_single_pointing(ra, dec, Dnow, use_grid=False)
        time.sleep(60.0)


if __name__ == "__main__":
    main()
