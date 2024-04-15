import datetime
import signal
import subprocess  # nosec
import time

import astropy.units as u
import click
from astropy.time import Time
from beamformer.strategist.strategist import PointingStrategist
from psrqpy import QueryATNF


def get_folding_pars(psr, df):
    """
    Return ra and dec for a pulsar from psrcat database
    psr: string, pulsar B name
    df: Panda of psrqpy query
    """
    if "B" in psr:
        psrcolumn = df[df["PSRB"].values == psr]
    elif "J" in psr:
        psrcolumn = df[df["PSRJ"].values == psr]
    else:
        print(f"psrname {psr} not in query")
    ra = psrcolumn["RAJD"].values[0]
    dec = psrcolumn["DECJD"].values[0]

    return ra, dec


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("psrfile", type=click.File("r"), required="True")
@click.argument("outfile", type=click.File("a"), required="True")
def main(psrfile, outfile):
    """
    Record SPS data when known pulsars are transitting.

    For a list of pulsars, spsctl is run on the transitting beam, when a
    source transits (+- transit_buffer), then an interrupt sent to the process.
    It checks every minute to start/stop acquisition, with logic in place to
    continue recording as long as there is still one pulsar in the beamrow

    psrfile: a txt file with a list of pulsar names.
    Per SPS convention, the 'B' name must be used if it exists

    outfile: a txt file to record acquisition log messages.
    """
    pst = PointingStrategist(create_db=False)

    Dnow = datetime.datetime.now()
    pointings = []
    current_acq = []
    active_beams = []
    psrs = []
    processes = []
    transit_buffer = 3 * u.min

    query = QueryATNF()
    psrcat_df = query.pandas

    print("Acquiring pointings for all pulsars in list \n")
    # Append all pointings into one list
    for psr in psrfile:
        psr = psr.strip()
        # avoid duplicates
        if psr not in psrs:
            ra, dec = get_folding_pars(psr, psrcat_df)
            ap = pst.get_single_pointing(ra, dec, Dnow)
            pointings.append(ap)
            current_acq.append(0)
            processes.append(0)
            psrs.append(psr)
            print(psr)
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
                        outfile.write(
                            f"{Tnow.isot} Starting acq, {psr} transitting row"
                            f" {beamrow} \n"
                        )
                        outfile.flush()
                        processi = subprocess.Popen(
                            ["spsctl", f"{beamrow}"], shell=False
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
                    outfile.write(f"{Tnow.isot} Stopping acq, {psr} row {beamrow} \n")
                    outfile.flush()
                    processi = processes[i]
                    processi.send_signal(signal.SIGINT)
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
