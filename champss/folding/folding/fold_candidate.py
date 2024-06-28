import datetime as dt
import logging
import os
import subprocess

import click
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

# set these up before importing any SPS packages
log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from beamformer.skybeam import SkyBeamFormer
from beamformer.strategist.strategist import PointingStrategist
from beamformer.utilities.common import find_closest_pointing, get_data_list
from folding.plot_candidate import plot_candidate_archive
from sps_databases import db_api, db_utils


def apply_logging_config(level):
    """
    Applies logging settings from the given configuration.

    Logging settings are under the 'logging' key, and include:
    - format: string for the `logging.formatter`
    - level: logging level for the root logger
    - modules: a dictionary of submodule names and logging level to be applied to that submodule's logger
    """
    log_stream.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s >> %(message)s", datefmt="%b %d %H:%M:%S"
        )
    )

    logging.root.setLevel(level)
    log.debug("Set default level to: %s", level)


apply_logging_config(logging.INFO)


def candidate_name(ra_deg, dec_deg, j2000=True):
    ra_hhmmss = ra_deg * 24 / 360
    dec_ddmmss = abs(dec_deg)
    ra_str = f"{int(ra_hhmmss):02d}{int((ra_hhmmss * 60) % 60):02d}"
    dec_sign = "+" if dec_deg >= 0 else "-"
    dec_str = f"{int(dec_ddmmss):02d}{int((dec_ddmmss * 60) % 60):02d}"
    candidate_name = "J" + ra_str + dec_sign + dec_str
    return candidate_name


def create_ephemeris(name, ra, dec, dm, obs_date, f0, ephem_path, fs_id=False):
    cand_pos = SkyCoord(ra, dec, unit="deg")
    raj = f"{cand_pos.ra.hms.h:02.0f}:{cand_pos.ra.hms.m:02.0f}:{cand_pos.ra.hms.s:.6f}"
    decj = f"{cand_pos.dec.dms.d:02.0f}:{abs(cand_pos.dec.dms.m):02.0f}:{abs(cand_pos.dec.dms.s):.6f}"
    pepoch = Time(obs_date).mjd
    log.info("Making new candidate ephemeris...")
    ephem = [
        ["PSRJ", name],
        ["RAJ", str(raj)],
        ["DECJ", str(decj)],
        ["DM", str(dm)],
        ["PEPOCH", str(pepoch)],
        ["F0", str(f0)],
        ["DMEPOCH", str(pepoch)],
        ["RAJD", str(ra)],
        ["DECJD", str(dec)],
        ["EPHVER", "2"],
        ["UNITS", "TDB"],
    ]

    with open(ephem_path, "w") as file:
        for row in ephem:
            line = "\t".join(row)
            line.expandtabs(8)
            file.write(line + "\n")

    if fs_id:
        db_api.update_followup_source(fs_id, {"path_to_ephemeris": ephem_path})


def main(
    date,
    sigma,
    dm,
    f0,
    ra,
    dec,
    known,
    psr,
    fs_id,
    db_port,
    db_host,
    db_name,
    basepath,
    candpath="",
    write_to_db=False,
    using_workflow=False,
    overwrite_folding=False,
):
    """
    Perform the main processing steps for folding a candidate or known source.  It can
    be called for a set of ra, dec, f0, dm, from a pulsar name using the known_source
    database, or from a FollowUpSource ID which uses the ephemeris in the database.

    The main automated processing will use FollowUpSource ID, and update the database with the folding history.

    Args:
        date (str or datetime.datetime): The date of the observation.
        sigma (float): The significance threshold for folding.
        dm (float): The dispersion measure.
        f0 (float): The spin frequency.
        ra (float): The right ascension of the source.
        dec (float): The declination of the source.
        known (str): The name of the known source.
        psr (str): The name of the pulsar.
        fs_id (int): The ID of the FollowUpSource.
        db_port (int): The port number for the database connection.
        db_host (str): The hostname of the database.
        db_name (str): The name of the database.
        basepath (str): The base path for the data.
        write_to_db (bool, optional): Whether to write the results to the database. Defaults to False.
        using_workflow (bool, optional): Whether the function is being called from a workflow. Defaults to False.

    Returns:
        tuple: A tuple containing an empty dictionary, an empty list, and an empty list.
    """

    if using_workflow:
        if isinstance(date, str):
            for date_format in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
                try:
                    date = dt.datetime.strptime(str(date), date_format)
                    break
                except ValueError:
                    continue

    db_utils.connect(host=db_host, port=db_port, name=db_name)
    pst = PointingStrategist(create_db=False)

    # fs_id known_source, md_candidate, sd_candidate, ra+dec, psr
    ephem_path = None
    if fs_id:
        source = db_api.get_followup_source(fs_id)
        source_type = source.source_type
        if source.folding_history:
            fold_dates = [entry["date"].date() for entry in source.folding_history]
            if not overwrite_folding and date.date() in fold_dates:
                log.info(f"Already folded on {date.date()}, skipping...")
                return

        f0 = source.f0
        ra = source.ra
        dec = source.dec
        dm = source.dm
        sigma = source.candidate_sigma
        if source_type == "known_source":
            psr = source.source_name
            dir_suffix = "known_sources"
            known = psr
        elif source_type == "md_candidate" or source_type == "sd_candidate":
            dir_suffix = "candidates"
            name = candidate_name(ra, dec)
        ephem_path = source.path_to_ephemeris
    elif psr:
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
        name = candidate_name(ra, dec)
    elif candpath and write_to_db:
        from folding.filter_mpcandidates import add_candidate_to_fsdb
        from sps_common.interfaces import MultiPointingCandidate

        date_str = date.strftime("%Y%m%d")
        mpc = MultiPointingCandidate.read(candpath)
        ra = mpc.ra
        dec = mpc.dec
        f0 = mpc.best_freq
        dm = mpc.best_dm
        sigma = mpc.best_sigma
        dir_suffix = "candidates"
        name = candidate_name(ra, dec)

        followup_source = add_candidate_to_fsdb(
            date_str, ra, dec, f0, dm, sigma, candpath
        )
        print(followup_source)
        fs_id = followup_source._id
    else:
        log.error(
            "Must provide either a pulsar name, FollowUpSource ID, candidate path, or"
            " candidate RA and DEC"
        )
        return

    directory_path = f"/data/chime/sps/archives/{dir_suffix}"

    year = date.year
    month = date.month
    day = date.day

    if dir_suffix == "candidates":
        log.info(f"Setting up pointing for {round(ra, 2)} {round(dec, 2)}...")
        coord_path = f"{directory_path}/{round(ra, 2)}_{round(dec, 2)}"
        archive_fname = f"{coord_path}/cand_{round(dm, 2)}_{round(f0, 2)}_{year}-{month:02}-{day:02}"
        if not os.path.exists(coord_path):
            os.makedirs(coord_path)
        else:
            log.info(f"Directory '{coord_path}' already exists.")
        if not ephem_path:
            ephem_path = f"{coord_path}/cand_{round(dm, 2)}_{round(f0, 2)}_{year}-{month:02}-{day:02}.par"
            create_ephemeris(name, ra, dec, dm, date, f0, ephem_path, fs_id)
    elif dir_suffix == "known_sources":
        log.info(f"Setting up pointing for {psr}...")
        coord_path = f"{directory_path}/folded_profiles/{psr}"
        archive_fname = f"{coord_path}/{psr}_{year}-{month:02}-{day:02}"
        if not os.path.exists(coord_path):
            os.makedirs(coord_path)
        else:
            log.info(f"Directory '{coord_path}' already exists.")
        if not ephem_path:
            ephem_path = f"{directory_path}/ephemerides/{psr}.par"

    if not os.path.exists(ephem_path):
        log.error(f"Ephemeris file {ephem_path} not found")
        return

    outdir = coord_path
    fname = f"/{year}-{month:02}-{day:02}.fil"
    fil = outdir + fname

    pst = PointingStrategist(create_db=False)
    ap = pst.get_single_pointing(ra, dec, date)

    data_list = []
    for active_pointing in ap:
        data_list.extend(
            get_data_list(active_pointing.max_beams, basepath=basepath, extn="dat")
        )
    if not data_list:
        log.error(f"No data found for the pointing {ap[0].ra:.2f} {ap[0].dec:.2f}")
        return

    nchan_tier = int(np.ceil(np.log2(dm // 212.5 + 1)))
    nchan = 1024 * (2**nchan_tier)
    if nchan < ap[0].nchan:
        log.info(
            f"only need nchan = {nchan} for dm = {dm}, beamforming with"
            f" {nchan} channels"
        )
        ap[0].nchan = nchan
    num_threads = 4 * nchan // 1024
    log.info(f"using {num_threads} threads")

    # set number of turns, roughly equalling 10s
    turns = int(np.ceil(10 * f0))
    if turns <= 2:
        intflag = "-turns"
    else:
        intflag = "-L"
        turns = 10

    if not os.path.isfile(fil):
        log.info(f"Beamforming..., basepath {basepath}")
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
        skybeam, spectra_shared = sbf.form_skybeam(ap[0], num_threads=num_threads)

        log.info(f"Writing to {fil}")
        skybeam.write(fil)
        spectra_shared.close()
        spectra_shared.unlink()
        del skybeam

    if not os.path.isfile(f"{archive_fname}.ar"):
        log.info("Folding...")
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
        log.info(f"Finished, deleting {fil}")
        os.remove(fil)

    archive_fname = archive_fname + ".ar"
    create_FT = f"pam -T -F {archive_fname} -e FT"
    subprocess.run(create_FT, shell=True, capture_output=True, text=True)

    SNprof, SN_arr, plot_fname = plot_candidate_archive(
        archive_fname,
        sigma,
        dm,
        f0,
        ra,
        dec,
        coord_path,
        known,
    )

    log.info(f"SN of folded profile: {SN_arr}")

    if fs_id and write_to_db:
        log.info("Updating FollowUpSource with folding history")
        folding_history = source.folding_history
        fold_details = {
            "date": date,
            "archive_fname": archive_fname,
            "SN": float(SN_arr),
            "path_to_plot": plot_fname,
        }
        fold_dates = [entry["date"].date() for entry in folding_history]
        if date.date() in fold_dates:
            index = fold_dates.index(date.date())
            folding_history[index] = fold_details
        else:
            folding_history.append(fold_details)
        db_api.update_followup_source(fs_id, {"folding_history": folding_history})
        if len(folding_history) >= source.followup_duration:
            log.info(
                f"Finished follow-up duration of {source.followup_duration} days,"
                " setting active = False"
            )
            db_api.update_followup_source(fs_id, {"active": False})

    fold_details["date"] = fold_details["date"].strftime("%Y%m%d")
    # Silence Workflow errors, requires results, products, plots
    return fold_details, [], []


if __name__ == "__main__":
    main()
    # main(year, month, day, sigma, dm, f0, ra, dec, known)
