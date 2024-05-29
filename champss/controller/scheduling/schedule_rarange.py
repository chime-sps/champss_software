import signal
import subprocess
import time

import astropy.units as u
import click
from astropy.coordinates import EarthLocation
from astropy.time import Time


def schedule_acq(beams, ra_range):
    """
    Create acq schedule for a list of beams and a range of RA.

    beams: list of beams
    rarange: list of [ra_start, ra_end]
    """

    Tnow = Time.now()
    observing_location = EarthLocation(lat="49d19m14.52s", lon="-119d37m25.25s")
    observing_time = Time(
        Tnow.datetime.utcnow(), scale="utc", location=observing_location
    )

    LST = observing_time.sidereal_time("mean")

    ra_start = ra_range[0]
    if ra_range[-1] < ra_range[0]:
        ra_end = ra_range[-1] + 24.0
    else:
        ra_end = ra_range[-1]

    dt_start = (ra_start - LST.deg * 24 / 360.0) * u.hour
    dt_end = (ra_end - LST.deg * 24 / 360.0) * u.hour
    Tstart = Tnow + dt_start
    Tend = Tnow + dt_end

    print(f"Current time of {Tnow.isot}")

    if (Tnow > Tstart) and (Tnow < Tend):
        print("Transit is already in progress, starting acq")
    elif Tstart < Tnow:
        print("Start time in the past, adding 1 day to acq")
        Tstart = Tstart + 1 * u.day
        Tend = Tend + 1 * u.day
    else:
        dt = (Tstart - Tnow).to(u.min)
        print(f"Start time in the future, waiting {dt} for transit")

    print(f"Scheduled acq of beams {beams} from {Tstart.isot} to {Tend.isot}")

    acq = {
        "beams": beams,
        "ra_start": ra_start,
        "ra_end": ra_end,
        "Tstart": Tstart,
        "Tend": Tend,
    }
    return acq


@click.command()
@click.option(
    "--beams", "-b", type=str, default="121,122", help="Comma-separated list of beams"
)
@click.option(
    "--ra_range", "-r", type=str, default="18,24", help='RA range in format "start,end"'
)
def main(beams, ra_range):
    beams = [int(b) for b in beams.split(",")]
    ra_range = [float(r) for r in ra_range.split(",")]

    print("Setting up acq schedule \n")
    acq = schedule_acq(beams, ra_range)

    outfile = open("acq_log.txt", "a")
    current_acq = 0
    processes = [None] * len(beams)

    print("Running dynamic scheduler \n")
    while True:
        Tnow = Time.now()

        if (Tnow.mjd > acq["Tstart"].mjd) and (Tnow.mjd < acq["Tend"].mjd):
            if not current_acq:
                for i, beamrow in enumerate(acq["beams"]):
                    outfile.write(f"{Tnow.isot} Starting acq, row {beamrow} \n")

                    outfile.flush()
                    processi = subprocess.Popen(
                        ["spsctl", f"{beamrow}"], shell=False
                    )  # nosec
                    processes[i] = processi
                current_acq = 1

        else:
            if current_acq:
                for i, beamrow in enumerate(acq["beams"]):
                    outfile.write(f"{Tnow.isot} Stopping acq, row {beamrow} \n")
                    outfile.flush()
                    processi = processes[i]
                    processi.send_signal(signal.SIGINT)
                    time.sleep(10.0)
                current_acq = 0
                print("acq finished, generating new schedule \n")
                acq = schedule_acq(beams, ra_range)

        time.sleep(60.0)
