import datetime
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

def get_champss_fm_sources(server_url="mongodb://localhost:27017/", db_name="timing_ops"):
    # Initialize connection and cursor
    client = pymongo.MongoClient(server_url)

    # Create database if it does not exist
    database = client[db_name]

    # Setup
    collection = database["sources"]

    # Get sources
    return list(collection.find({'champss_foldmode': True}))

def get_folding_pars(psr):
    """
    Return ra and dec for a pulsar from the known_source database
    psr: string, pulsar B name                                                                                                                                                                                                                                   df: Panda of psrqpy query
    """
    source = db_api.get_known_source_by_name(psr)[0]
    ra = source.pos_ra_deg
    dec = source.pos_dec_deg
    return ra, dec


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
    # Convert basepath to local mount path
    local_path = basepath.replace("/sps-archiver2/", "/mnt/beegfs-client/").replace(
        "/sps-archiver1/", "/data/"
    )

    # Construct data folder path for today
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
        print(f"Warning: Could not check folder age for beam {beam}: {e}")

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
    "--outfile",
    type=click.File("a"),
    default="schedknownpsrlog.txt",
    help="Log file for acquisition messages.",
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
def main(psrfile, outfile, basepath, source, db_port, db_host, db_name):
    """
    Record SPS data when known pulsars are transitting.

    By default, loads pulsars from timing_ops database (champss_foldmode=True).
    Optionally, can load from a text file using --psrfile.

    For each pulsar, spsctl is run on the transitting beam when a source
    transits (+- transit_buffer), then an interrupt sent to the process.
    It checks every minute to start/stop acquisition, with logic in place to
    continue recording as long as there is still one pulsar in the beamrow.

    Prevents interference with externally-controlled beams by checking folder
    modification times before starting/stopping acquisitions.
    """
    print("Starting scheduleknownpulsars controller")
    print(f"Basepath: {basepath}")
    print(f"Source: {source}")
    print(f"Database: {db_name} at {db_host}:{db_port}")

    db_utils.connect(host=db_host, port=db_port, name=db_name)
    pst = PointingStrategist(create_db=False)

    Dnow = datetime.datetime.now()
    pointings = []
    current_acq = []
    active_beams = []
    psrs = []
    processes = []
    transit_buffer = 3 * u.min

    # Load pulsar list from database or text file
    if psrfile is None:
        # Load from timing_ops database
        print("Loading pulsars from timing_ops database (champss_foldmode=True)")
        pulsar_entries = get_champss_fm_sources()
        print(f"Found {len(pulsar_entries)} pulsars in database\n")

        print("Acquiring pointings for all pulsars in database \n")
        for entry in pulsar_entries:
            psr = entry['psr_id']
            # Database stores ra in hours, convert to degrees
            ra = entry['ra'] * 15.0
            dec = entry['dec']

            # avoid duplicates
            if psr not in psrs:
                ap = pst.get_single_pointing(ra, dec, Dnow)
                beamrow = ap[0].max_beams[0]["beam"]
                pointings.append(ap)
                current_acq.append(0)
                processes.append(0)
                psrs.append(psr)
                print(f"{psr} (beam {beamrow})")
            else:
                print(f"{psr} duplicated in database")
    else:
        # Load from text file
        print("Loading pulsars from text file")
        print("Acquiring pointings for all pulsars in list \n")
        for psr in psrfile:
            psr = psr.strip()
            # avoid duplicates
            if psr not in psrs:
                ra, dec = get_folding_pars(psr)
                ap = pst.get_single_pointing(ra, dec, Dnow)
                beamrow = ap[0].max_beams[0]["beam"]
                pointings.append(ap)
                current_acq.append(0)
                processes.append(0)
                psrs.append(psr)
                print(f"{psr} (beam {beamrow})")
            else:
                print(f"{psr} duplicated in list")

    print("Pointings loaded, running dynamic scheduler \n")
    while True:
        Tnow = Time.now()
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
                            outfile.write(
                                f"{Tnow.isot} {psr} transitting row {beamrow}, beam already "
                                f"recording (possibly spsctl), will not interfere\n"
                            )
                            outfile.flush()
                            # Mark as externally controlled - don't start spsctl
                            processes[i] = "external"
                        else:
                            # Beam is not recording, safe to start it
                            outfile.write(
                                f"{Tnow.isot} Starting acq, {psr} transitting row"
                                f" {beamrow} \n"
                            )
                            outfile.flush()
                            processi = subprocess.Popen(
                                ["spsctl", f"{beamrow}", "--basepath", f"{basepath}", "--source", f"{source}"], shell=False
                            )  # nosec
                            processes[i] = processi
                    else:
                        outfile.write(
                            f"{Tnow.isot} {psr} transitting row {beamrow}, continuing"
                            " acq \n"
                        )
                        outfile.flush()
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
                        outfile.write(f"{Tnow.isot} Stopping acq, {psr} row {beamrow} \n")
                        outfile.flush()
                        processi.send_signal(signal.SIGINT)
                    elif processi == "external":
                        outfile.write(
                            f"{Tnow.isot} Not stopping {psr} row {beamrow}, "
                            f"beam controlled externally (possibly spsctl)\n"
                        )
                        outfile.flush()
                    active_beams.remove(beamrow)
                    current_acq[i] = 0
                elif (activecount > 1) and (activeacq):
                    outfile.write(
                        f"{Tnow.isot} continuing acq, removing {psr} row {beamrow} \n"
                    )
                    outfile.flush()
                    active_beams.remove(beamrow)
                    current_acq[i] = 0
                    # passing process id to next pulsar in same beamrow
                    k = processes.index(beamrow)
                    processi = processes[i]
                    processes[k] = processi
                    processes[i] = 0

                # update pointing to current time, plan next transit in ~24 hours
                Dnow = datetime.datetime.now()
                ra, dec = get_folding_pars(psr)
                ap_updated = pst.get_single_pointing(ra, dec, Dnow)
                pointings[i] = ap_updated
            i += 1
        time.sleep(60.0)
        
if __name__ == "__main__":
    main()
