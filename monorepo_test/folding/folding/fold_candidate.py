import os
import sys
import subprocess
import numpy as np
import datetime as dt
from astropy.time import Time
from astropy.coordinates import SkyCoord
import click

from beamformer.utilities.common import find_closest_pointing
from beamformer.strategist.strategist import PointingStrategist
from beamformer.skybeam import SkyBeamFormer
from folding.Candidate_Plotter import *
import sys
from sps_databases import db_api, db_utils 


def candidate_name(ra_deg, dec_deg, j2000=True):
    ra_hhmmss = ra_deg * 24 / 360
    dec_ddmmss = abs(dec_deg)
    ra_str = "{:02d}{:02d}".format(int(ra_hhmmss), int((ra_hhmmss * 60) % 60))
    dec_sign = "+" if dec_deg >= 0 else "-"
    dec_str = "{:02d}{:02d}".format(int(dec_ddmmss), int((dec_ddmmss * 60) % 60))
    candidate_name = "J" + ra_str + dec_sign + dec_str
    return candidate_name


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Date of data to process. Default = Today in UTC",
)
@click.option("--sigma", type=float, help="Pipeline sigma of candidate")
@click.option("--dm", type=float, help="DM")
@click.option("--f0", type=float, help="F0")
@click.option("--ra", type=float, help="RA")
@click.option("--dec", type=float, help="DEC")
@click.option("--known", type=str, default=" ", help="Name of known pulsar, otherwise empty string")
@click.option(
    "--psr", type=str, default="", help="Fold on known pulsar, using ephemeris"
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--write-to-db",
    is_flag=True,
    help="Set folded_status to True in the processes database.",
)
@click.option(
        "--basepath", type=str, default='/data/chime/sps/raw/',
        help="Base directory for raw data",
)
def main(
    date,
    sigma,
    dm,
    f0,
    ra,
    dec,
    known,
    psr,
    db_port,
    db_host,
    db_name,
    basepath,
    write_to_db=False,
    using_workflow=False,
):
    if using_workflow:
        if isinstance(date, str):
            for date_format in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
                try:
                    date = dt.datetime.strptime(date, date_format)
                    break
                except ValueError:
                    continue

    db_utils.connect(host=db_host, port=db_port, name=db_name)
    pst = PointingStrategist()

    if psr:
        source = db_api.get_known_source_by_name(psr)[0]
        ra = source.pos_ra_deg
        dec = source.pos_dec_deg
        dm = source.dm
        name = psr
        dir_suffix = "known_sources"
        f0 = 1 / source.spin_period_s
        known = psr
    elif ra and dec:
        coords = find_closest_pointing(ra, dec)
        ra = coords.ra
        dec = coords.dec
        dir_suffix = "candidates"
    else:
        print("Must provide either a pulsar name or candidate RA and DEC")

    # set number of turns, roughly equalling 10s
    turns = int(np.ceil(10 * f0))
    if turns <= 2:
        intflag = "-turns"
    else:
        intflag = "-L"
        turns = 10

    directory_path = f"/data/chime/sps/archives/{dir_suffix}"

    year = date.year
    month = date.month
    day = date.day

    cand_pos = SkyCoord(ra, dec, unit="deg")
    raj = f"{cand_pos.ra.hms.h:02.0f}:{cand_pos.ra.hms.m:02.0f}:{cand_pos.ra.hms.s:.6f}"
    decj = f"{cand_pos.dec.dms.d:02.0f}:{abs(cand_pos.dec.dms.m):02.0f}:{abs(cand_pos.dec.dms.s):.6f}"

    if not psr:
        name = candidate_name(round(ra, 2), round(dec, 2))
        print("Setting up pointing for {0} {1}...".format(round(ra, 2), round(dec, 2)))
        coord_path = f"{directory_path}/{round(ra, 2)}_{round(dec, 2)}"
        ephem_path = f"{coord_path}/cand_{round(dm, 2)}_{round(f0, 2)}_{year}-{month:02}-{day:02}.par"
        archive_fname = f"{coord_path}/cand_{round(dm, 2)}_{round(f0, 2)}_{year}-{month:02}-{day:02}"
    else:
        print(f"Setting up pointing for {psr}...")
        coord_path = f"{directory_path}/folded_profiles/{psr}"
        ephem_path = f"{directory_path}/ephemerides/{psr}.par"
        archive_fname = f"{coord_path}/{psr}_{year}-{month:02}-{day:02}"
    if not os.path.exists(coord_path):
        os.makedirs(coord_path)
    else:
        print(f"Directory '{coord_path}' already exists.")

    outdir = coord_path
    fname = f"/{year}-{month:02}-{day:02}.fil"
    fil = outdir + fname

    obs_date = dt.datetime(year, month, day)

    pst = PointingStrategist()
    ap = pst.get_single_pointing(ra, dec, obs_date)
    active_process = db_api.get_process_from_active_pointing(ap[0])

    nchan_tier = int(np.ceil(np.log2(dm // 212.5 + 1)))
    nchan = 1024 * (2**nchan_tier)
    if nchan < ap[0].nchan:
        print(
            f"only need nchan = {nchan} for dm = {dm}, beamforming with {nchan} channels"
        )
        ap[0].nchan = nchan
    num_threads = nchan // 1024

    if not psr:
        pepoch = Time(obs_date).mjd
        print("Making new candidate ephemeris...")
        ephem = [
            ["PSRJ", name],
            ["RAJD", str(ra)],
            ["DECJD", str(dec)],
            ["RAJ", str(raj)],
            ["DECJ", str(decj)],
            ["DM", str(dm)],
            ["PEPOCH", str(pepoch)],
            ["F0", str(f0)],
            ["DMEPOCH", str(pepoch)],
            ["EPHVER", "2"],
            ["UNITS", "TDB"],
        ]

        with open(ephem_path, "w") as file:
            for row in ephem:
                line = "\t".join(row)
                file.write(line + "\n")

    if not os.path.isfile(fil):
        print("Beamforming..., basepath {0}".format(basepath))
        sbf = SkyBeamFormer(
            extn="dat",
            update_db=False,
            min_data_frac=0.5,
            basepath=basepath,
            add_local_median=True,
            detrend_data=True,
            detrend_nsamp=32768,
            masking_timescale=512000,
            # flatten_bandpass=False,
            run_rfi_mitigation=True,
            masking_dict=dict(
                weights=True,
                l1=True,
                badchan=True,
                kurtosis=False,
                mad=False,
                sk=True,
                powspec=False,
                dummy=False,
            ),
            beam_to_normalise=1,
        )
        sb = sbf.form_skybeam(ap[0], num_threads=num_threads)

        print(f"Writing to {fil}")
        sb.write(fil)
        del sb

    if not os.path.isfile(
        f"{archive_fname}.ar"
    ):
        print("Folding...")
        subprocess.run(
            [
                "dspsr",
                "-t",
                f"{num_threads}",
                f"{intflag}",
                f"{turns}",
                "-A",
                "-k",
                "chime",
                "-E",
                f"{ephem_path}",
                "-O",
                f"{archive_fname}",
                f"{fil}",
            ]
        )
        print(f"Finished, deleting {fil}")
        os.remove(fil)

    archive_fname = archive_fname + ".ar"
    create_FT = f"pam -T -F {archive_fname} -e FT"
    subprocess.run(create_FT, shell=True, capture_output=True, text=True)

    SNprofs, SN_arr = plot_foldspec(
        archive_fname,
        sigma,
        dm,
        f0,
        ra,
        dec,
        coord_path,
        known,
    )

    print("SN of folded profile: ", SN_arr)

    plt.close()

    if write_to_db:
        print("Setting folded_status = True in process db")
        db_api.update_process(active_process.id, {"folded_status": True})

    # Silence Workflow errors, requires results, products, plots
    return {}, [], []


if __name__ == "__main__":
    main()
    # main(year, month, day, sigma, dm, f0, ra, dec, known)
