import datetime
import logging
import os
import signal
import subprocess  # nosec
import time

import astropy.units as u
import click
import pymongo
from astropy.time import Time
from beamformer.strategist.strategist import PointingStrategist
from sps_databases import db_api, db_utils

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
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(logfile, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_champss_fm_sources(server_url="mongodb://localhost:27017/", db_name="timing_ops"):
    # Initialize connection and cursor
    client = pymongo.MongoClient(server_url)

    # Create database if it does not exist
    database = client[db_name]

    # Setup
    collection = database["sources"]

    # Get sources
    return list(collection.find({'champss_foldmode': True}))

def get_pulsar_radec(psr):
    """
    Return ra and dec for a pulsar from the known_source database
    psr: string, B name if it exists following our DB convention                                                                                                                                                                                                                          
    """
    source = db_api.get_known_source_by_name(psr)[0]
    ra = source.pos_ra_deg
    dec = source.pos_dec_deg
    return ra, dec


def update_psr_list(psrs, pointings, current_acq, processes, pst, logger):
    """
    Update the pulsar list by querying the timing_ops database.

    Compares current pulsar list with database, adds new pulsars with
    champss_foldmode=True, and removes pulsars that no longer have it enabled.

    Args:
        psrs: Current list of pulsar IDs
        pointings: Current list of pointing objects
        current_acq: Current acquisition status list
        processes: Current process list
        pst: PointingStrategist object
        logger: Logger instance

    Returns:
        Tuple of (psrs, pointings, current_acq, processes) with updates applied
    """
    logger.debug("Checking database for pulsar list updates...")

    # Query database for current pulsar list
    new_pulsar_entries = get_champss_fm_sources()
    new_psr_ids = {entry['psr_id'] for entry in new_pulsar_entries}
    current_psr_ids = set(psrs)

    # Find new pulsars to add
    pulsars_to_add = new_psr_ids - current_psr_ids
    if pulsars_to_add:
        logger.info(f"Adding {len(pulsars_to_add)} new pulsar(s): {sorted(pulsars_to_add)}")

        for entry in new_pulsar_entries:
            psr = entry['psr_id']
            if psr in pulsars_to_add:
                ra = entry['ra']
                dec = entry['dec']
                Dnow_update = datetime.datetime.now()
                ap = pst.get_single_pointing(ra, dec, Dnow_update)
                beamrow = ap[0].max_beams[0]["beam"]
                pointings.append(ap)
                current_acq.append(0)
                processes.append(0)
                psrs.append(psr)
                logger.info(f"Added {psr} (beam {beamrow})")
                logger.info(f"Updated pulsar count: {len(psrs)}")
                
    # Find pulsars to remove (champss_foldmode=False in DB)
    pulsars_to_remove = current_psr_ids - new_psr_ids
    if pulsars_to_remove:
        logger.info(f"Removing {len(pulsars_to_remove)} pulsar(s): {sorted(pulsars_to_remove)}")

        # Remove pulsars by rebuilding all lists without removed pulsars
        indices_to_keep = [i for i, psr in enumerate(psrs) if psr not in pulsars_to_remove]
        psrs = [psrs[i] for i in indices_to_keep]
        pointings = [pointings[i] for i in indices_to_keep]
        current_acq = [current_acq[i] for i in indices_to_keep]
        processes = [processes[i] for i in indices_to_keep]
        logger.info(f"Updated pulsar count: {len(psrs)}")

    if not pulsars_to_add and not pulsars_to_remove:
        logger.debug("No changes to pulsar list")

    return psrs, pointings, current_acq, processes


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
    local_path = (basepath
                  .replace("/sps-archiver1/", "/data/")
                  .replace("/sps-archiver2/", "/mnt/beegfs-client/")
                  .replace("/sps-archiver3/", "/mnt/beegfs-client/")
                  .replace("/sps-archiver4/", "/mnt/beegfs-client/")
                  .replace("/sps-archiver5/", "/mnt/beegfs-client/"))

    # Today's
    data_folder = (
        local_path
        + datetime.datetime.utcnow().strftime("/%Y/%m/%d/")
        + f"{str(beam).zfill(4)}"
    )

    max_folder_age = 600  # 10 minutes - threshold for active recording

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
    help=("The chime_slow_pulsar_writer object to use on L1, must be either 'champss' or 'slow'. "
          "Do not use 'slow' before consulting with the Slow team"),
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
    logger = setup_logger(logfile)

    logger.info("Starting scheduleknownpulsars controller")
    logger.info(f"Logfile: {logfile}")
    logger.info(f"L1 Basepath: {basepath}")
    logger.info(f"Source: {source}")
    logger.info(f"Database: {db_name} at {db_host}:{db_port}")

    db_utils.connect(host=db_host, port=db_port, name=db_name)
    pst = PointingStrategist(create_db=False)

    Dnow = datetime.datetime.now()
    pointings = []
    current_acq = []
    active_beams = []
    psrs = []
    processes = []
    transit_buffer = 3 * u.min
    db_check_interval = 600
    
    # Track last database check time (for periodic updates in database mode)
    last_db_check = Time.now()
    use_database_mode = (psrfile is None)

    # Load pulsar list from database or text file
    if psrfile is None:
        # Load from timing_ops database
        logger.info("Loading pulsars from timing_ops database (champss_foldmode=True)")
        logger.info(f"Database check interval: {db_check_interval} seconds ({db_check_interval/60:.1f} minutes)")
        pulsar_entries = get_champss_fm_sources()
        logger.info(f"Found {len(pulsar_entries)} pulsars in database")

        logger.info("Acquiring pointings for all pulsars in database")
        for entry in pulsar_entries:
            psr = entry['psr_id']
            ra = entry['ra']
            dec = entry['dec']

            # avoid duplicates
            if psr not in psrs:
                ap = pst.get_single_pointing(ra, dec, Dnow)
                beamrow = ap[0].max_beams[0]["beam"]
                pointings.append(ap)
                current_acq.append(0)
                processes.append(0)
                psrs.append(psr)
                logger.info(f"{psr} (beam {beamrow})")
            else:
                logger.info(f"{psr} duplicated in database")
    else:
        # Load from text file
        logger.info("Loading pulsars from text file")
        logger.info("Acquiring pointings for all pulsars in list")
        for psr in psrfile:
            psr = psr.strip()
            # avoid duplicates
            if psr not in psrs:
                ra, dec = get_pulsar_radec(psr)
                ap = pst.get_single_pointing(ra, dec, Dnow)
                beamrow = ap[0].max_beams[0]["beam"]
                pointings.append(ap)
                current_acq.append(0)
                processes.append(0)
                psrs.append(psr)
                logger.info(f"{psr} (beam {beamrow})")
            else:
                logger.info(f"{psr} duplicated in list")

    logger.info("Pointings loaded, running dynamic scheduler")
    while True:
        Tnow = Time.now()

        # Periodically check database for pulsar list updates (database mode only)
        if use_database_mode:
            time_since_last_check = (Tnow - last_db_check).to(u.s).value
            if time_since_last_check >= db_check_interval:
                psrs, pointings, current_acq, processes = update_psr_list(
                    psrs, pointings, current_acq, processes, pst, logger
                )
                last_db_check = Tnow

        i = 0
        for psr in psrs:
            ap = pointings[i]
            activeacq = current_acq[i]
            Tend = Time(ap[0].max_beams[3]["utc_end"], format="unix")
            Tstart = Time(ap[0].max_beams[0]["utc_start"], format="unix")
            Tend = Tend + transit_buffer
            Tstart = Tstart - transit_buffer
            transit_duration = (Tend - Tstart).to(u.s).value
            beamrow = ap[0].max_beams[0]["beam"]
            time_to_transit = Tend.unix - Tnow.unix

            if (time_to_transit < transit_duration) and (time_to_transit > 0):
                if not activeacq:
                    if beamrow not in active_beams:
                        # Check if beam is already recording (e.g., by spsctl or another controller)
                        if is_beam_recording(beamrow, basepath, source):
                            logger.info(
                                f"{psr} transitting row {beamrow}, beam already "
                                f"recording, will not interfere"
                            )
                            # Mark as externally controlled - don't start spsctl
                            processes[i] = "external"
                        else:
                            # Beam is not recording, safe to start it
                            logger.info(
                                f"Starting acq, {psr} transitting row {beamrow}"
                            )
                            processi = subprocess.Popen(
                                ["spsctl", f"{beamrow}", "--basepath", f"{basepath}", "--source", f"{source}"], shell=False
                            )  # nosec
                            processes[i] = processi
                    else:
                        logger.info(
                            f"{psr} transitting row {beamrow}, continuing acq"
                        )
                        # logic to include process id for beams with 2+ pulsars
                        processes[i] = beamrow
                    active_beams.append(beamrow)
                    current_acq[i] = 1
            elif time_to_transit < 0:
                activecount = active_beams.count(beamrow)
                if (activecount == 1) and (activeacq):
                    processi = processes[i]
                    # Only stop if we control this beam (not externally controlled)
                    if isinstance(processi, subprocess.Popen):
                        logger.info(f"Stopping acq, {psr} row {beamrow}")
                        processi.send_signal(signal.SIGINT)
                    elif processi == "external":
                        logger.info(
                            f"Not stopping {psr} row {beamrow}, "
                            f"beam controlled externally"
                        )
                    active_beams.remove(beamrow)
                    current_acq[i] = 0
                elif (activecount > 1) and (activeacq):
                    logger.info(
                        f"Continuing acq, removing {psr} row {beamrow}"
                    )
                    active_beams.remove(beamrow)
                    current_acq[i] = 0
                    # passing process id to next pulsar in same beamrow
                    k = processes.index(beamrow)
                    processi = processes[i]
                    processes[k] = processi
                    processes[i] = 0

                # update pointing to current time, plan next transit in ~24 hours
                Dnow = datetime.datetime.now()
                ra, dec = get_pulsar_radec(psr)
                ap_updated = pst.get_single_pointing(ra, dec, Dnow)
                pointings[i] = ap_updated
            i += 1
        time.sleep(60.0)
        
if __name__ == "__main__":
    main()
