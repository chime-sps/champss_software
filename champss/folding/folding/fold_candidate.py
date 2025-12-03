import logging
import multiprocessing
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

from beamformer.strategist.strategist import PointingStrategist
from beamformer.utilities.common import find_closest_pointing, get_data_list
from folding.plot_aliases import plot_aliases
from folding.plot_candidate import plot_candidate_archive
from scheduler.utils import convert_date_to_datetime
from sps_databases import db_api, db_utils
from sps_pipeline import beamform
from sps_pipeline.pipeline import default_datpath, load_config, apply_logging_config


def update_folding_history(id, payload):
    """
    Updates a followup_source with the given attribute and values as a dict.

    Parameters
    ----------
    id: str or ObjectId
        The id of the followup source to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    followup_source: dict
        The dict of the updated followup source
    """
    import pymongo
    from bson.objectid import ObjectId

    db = db_utils.connect()
    if isinstance(id, str):
        id = ObjectId(id)
    return db.followup_sources.find_one_and_update(
        {"_id": id},
        {"$push": payload},
        return_document=pymongo.ReturnDocument.AFTER,
    )

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


def create_alias_ephemeris(base_ephem_path, alias_factor, output_path):
    """
    Create a modified ephemeris file with F0 scaled by the alias factor.

    Parameters
    ----------
    base_ephem_path : str
        Path to the base ephemeris file
    alias_factor : float
        Factor to multiply F0 by (e.g., 2 for 2nd harmonic, 0.5 for sub-harmonic)
    output_path : str
        Path to save the modified ephemeris file

    Returns
    -------
    output_path : str
        Path to the created ephemeris file
    """
    with open(base_ephem_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.strip().startswith('F0'):
            parts = line.split()
            original_f0 = float(parts[1])
            new_f0 = original_f0 * alias_factor
            modified_lines.append(f"F0\t{new_f0}\n")
        else:
            modified_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(modified_lines)

    return output_path


def get_alias_factors():
    """
    Return the list of alias factors to fold at.

    Returns
    -------
    alias_factors : list
        List of (factor, label) tuples for folding
    """
    factors = [
        (1/16, "1_16"),
        (1/8, "1_8"),
        (1/4, "1_4"),
        (1/3, "1_3"),
        (1/2, "1_2"),
        (1, "1"),
        (2, "2"),
        (3, "3"),
        (4, "4"),
        (8, "8"),
        (16, "16"),
    ]
    return factors


def fold_at_alias(fil, ephem_path, alias_dir, alias_factor, alias_label, num_threads):
    """
    Fold a filterbank at a specific alias factor.

    Parameters
    ----------
    fil : str
        Path to the filterbank file
    ephem_path : str
        Path to the base ephemeris file
    alias_dir : str
        Directory path for alias parfiles and archives
    alias_factor : float
        Factor to multiply F0 by
    alias_label : str
        Label for the alias (e.g., "1_2", "2")
    num_threads : int
        Number of threads for dspsr

    Returns
    -------
    archive_fname : str
        Path to the created archive file, or None if folding failed
    """
    # Create ephemeris with modified F0 in aliases directory
    alias_ephem_path = f"{alias_dir}/alias_{alias_label}.par"
    create_alias_ephemeris(ephem_path, alias_factor, alias_ephem_path)

    # Create archive filename in aliases directory
    archive_fname = f"{alias_dir}/alias_{alias_label}"

    log.info(f"Folding at alias {alias_label} (factor={alias_factor})...")
    result = subprocess.run(
        [
            "dspsr",
            "-t",
            f"{num_threads}",
            "-k",
            "chime",
            "-E",
            f"{alias_ephem_path}",
            "-O",
            f"{archive_fname}",
            f"{fil}",
        ],
        capture_output=True,
        text=True,
    )

    archive_fname_full = archive_fname + ".ar"
    if os.path.isfile(archive_fname_full):
        # Create frequency and time scrunched version
        create_FT = f"pam -T -F {archive_fname_full} -e FT"
        subprocess.run(create_FT, shell=True, capture_output=True, text=True)
        return archive_fname_full
    else:
        log.warning(f"Failed to create archive for alias {alias_label}")
        return None


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
@click.option(
    "--known",
    type=str,
    default=" ",
    help="Name of known pulsar, otherwise empty string",
)
@click.option(
    "--psr", type=str, default="", help="Fold on known pulsar, using ephemeris"
)
@click.option(
    "--fs_id",
    type=str,
    default="",
    help="FollowUpSource ID, to fold from database values",
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
@click.option(
    "--foldpath",
    default="/data/chime/sps/archives",
    type=str,
    help="Path for created files during fold step.",
)
@click.option(
    "--datpath",
    default=default_datpath,
    type=str,
    help="Path to the raw data folder.",
)
@click.option(
    "--candpath",
    type=str,
    default="",
    help="Path to candidate file",
)
@click.option(
    "--write-to-db",
    is_flag=True,
    help="Set folded_status to True in the processes database.",
)
@click.option(
    "--overwrite-folding",
    is_flag=True,
    help="Re-run folding even if already folded on this date.",
)
@click.option(
    "--filterbank-to-ram/--no-filterbank-to-ram",
    default=True,
    help="Use ramdisk for filterbank files, default True.",
)
@click.option(
    "--fold-aliases",
    is_flag=True,
    help="Fold at multiple frequency aliases (1/16, 1/8, 1/4, 1/3, 1/2, 1, 2, 3, 4, 8, 16).",
)
@click.option(
    "--exact-coords",
    is_flag=True,
    help="Use exact RA/Dec coordinates without snapping to pointing map grid (for known pulsars).",
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
    fs_id,
    db_port,
    db_host,
    db_name,
    foldpath,
    datpath,
    candpath="",
    write_to_db=False,
    using_workflow=False,
    overwrite_folding=False,
    filterbank_to_ram=True,
    fold_aliases=False,
    exact_coords=False,
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
        date = convert_date_to_datetime(date)

    multiprocessing.set_start_method("forkserver", force=True)
    db_utils.connect(host=db_host, port=db_port, name=db_name)

    # fs_id known_source, md_candidate, sd_candidate, ra+dec, psr
    ephem_path = None
    if fs_id:
        source = db_api.get_followup_source(fs_id)
        source_type = source.source_type
        if source.folding_history:
            fold_dates = [entry["date"].date() for entry in source.folding_history]
            if not overwrite_folding and date.date() in fold_dates:
                log.info(f"Already folded on {date.date()}, skipping...")
                return {}, [], []

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
        source = db_api.get_known_source_by_names(psr)[0]
        ra = source.pos_ra_deg
        dec = source.pos_dec_deg
        dm = source.dm
        name = psr
        dir_suffix = "known_sources"
        f0 = 1 / source.spin_period_s
        known = psr
    elif ra and dec:
        if not exact_coords:
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
        fs_id = followup_source._id
    else:
        log.error(
            "Must provide either a pulsar name, FollowUpSource ID, candidate path, or"
            " candidate RA and DEC"
        )
        return {}, [], []

    config = load_config()
    config['beamform']['update_db'] = False
    config['beamform']['flatten_bandpass'] = False
    log_path = foldpath + f"/logs/{date.strftime('%Y/%m/%d')}/"
    log_name = (
        f"fold_candidate{date.strftime('%Y-%m-%d')}_{ra:.02f}_"
        f"{dec:.02f}_{f0:.02f}_{dm:.02f}.log"
    )
    log_file = log_path + log_name
    apply_logging_config(config, log_file)

    directory_path = f"{foldpath}/{dir_suffix}"

    year = date.year
    month = date.month
    day = date.day

    if dir_suffix == "candidates":
        log.info(f"Setting up pointing for {ra:.02f} {dec:.02f}...")
        coord_path = f"{directory_path}/{ra:.02f}_{dec:.02f}"
        archive_fname = (
            f"{coord_path}/cand_{f0:.02f}_{dm:.02f}_{year}-{month:02}-{day:02}"
        )
        if not os.path.exists(coord_path):
            os.makedirs(coord_path)
        else:
            log.info(f"Directory '{coord_path}' already exists.")
        if not ephem_path:
            ephem_path = (
                f"{coord_path}/cand_{f0:.02f}_{dm:.02f}_{year}-{month:02}-{day:02}.par"
            )
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

    if os.path.isfile(f"{archive_fname}.ar") and not overwrite_folding:
        log.info(f"Archive file {archive_fname}.ar already exists, skipping folding...")
        return {}, [], []

    if not os.path.exists(ephem_path):
        log.error(f"Ephemeris file {ephem_path} not found")
        return {}, [], []

    if exact_coords:
        log.info(f"Using exact coordinates (RA={ra:.6f}, Dec={dec:.6f}) without grid snapping")
    else:
        log.info(f"Using coordinates from pointing map (RA={ra:.2f}, Dec={dec:.2f})")

    pst = PointingStrategist(create_db=False, split_long_pointing=True)
    ap = pst.get_single_pointing(ra, dec, date, use_grid=not exact_coords)

    # If multiple sub-pointings, force to disk (too large for RAM)
    if len(ap) > 1:
        log.info(f"Multiple sub-pointings ({len(ap)}), writing filterbank to disk")
        filterbank_to_ram = False
        config.beamform.beam_to_normalise = None

    fname = f"/{ra:.02f}_{dec:.02f}_{f0:.02f}_{dm:.02f}_{year}-{month:02}-{day:02}.fil"
    if filterbank_to_ram:
        log.info("Using ram for filterbank file")
        fildir = "/dev/shm"
    else:
        log.info("Using disk for filterbank file")
        fildir = coord_path
    fil = fildir + fname

    nchan_tier = int(np.ceil(np.log2(dm // 212.5 + 1)))
    nchan = 1024 * (2**nchan_tier)

    # set number of turns, roughly equalling 10s
    turns = int(np.ceil(10 * f0))
    if turns <= 2:
        intflag = "-turns"
    else:
        intflag = "-L"
        turns = 10

    if not os.path.isfile(fil):
        log.info(f"Beamforming {len(ap)} sub-pointing(s)...")
        fdmt = True
        beamformer = beamform.initialise(config, rfi_beamform=True,
                                         basepath=foldpath, datpath=datpath)

        # Loop through all active pointings and append them into one filterbank
        for i, active_pointing in enumerate(ap):
            # Adjust nchan if needed
            if nchan < active_pointing.nchan:
                log.info(
                    f"only need nchan = {nchan} for dm = {dm}, beamforming with"
                    f" {nchan} channels"
                )
                active_pointing.nchan = nchan

            num_threads = 4 * nchan // 1024
            log.info(f"Beamforming sub-pointing {i+1}/{len(ap)} with {num_threads} threads")

            skybeam, spectra_shared = beamform.run(
                active_pointing, beamformer, fdmt, num_threads, foldpath
            )

            if skybeam is None:
                log.warning(
                    f"Insufficient unmasked data to form skybeam for sub-pointing {i+1}, skipping"
                )
                spectra_shared.close()
                spectra_shared.unlink()
                continue

            # Write first sub-pointing to create the file, append subsequent ones
            if i == 0:
                log.info(f"Writing sub-pointing {i+1} to {fil}")
                skybeam.write(fil)
            else:
                log.info(f"Appending sub-pointing {i+1} to {fil}")
                skybeam.append(fil)

            spectra_shared.close()
            spectra_shared.unlink()
            del skybeam

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

    archive_fname = archive_fname + ".ar"
    create_FT = f"pam -T -F {archive_fname} -e FT"
    subprocess.run(create_FT, shell=True, capture_output=True, text=True)

    # Alias folding if requested
    alias_results = {}
    if fold_aliases:
        log.info("Folding at frequency aliases...")
        alias_dir = f"{coord_path}/aliases"
        if not os.path.exists(alias_dir):
            os.makedirs(alias_dir)

        alias_factors = get_alias_factors()
        for factor, label in alias_factors:
            alias_archive = fold_at_alias(
                fil, ephem_path, alias_dir, factor, label, num_threads
            )
            if alias_archive:
                alias_results[label] = alias_archive

        log.info(f"Completed alias folding: {len(alias_results)} of {len(alias_factors)} successful")

        # Plot alias results
        if alias_results:
            alias_plot_path = os.path.join(
                alias_dir, f"alias_plot_{f0:.02f}_{dm:.02f}_{year}-{month:02}-{day:02}.png"
            )
            plot_aliases(alias_results, output_path=alias_plot_path)

    log.info(f"Finished, deleting {fil}")
    if os.path.isfile(fil):
        os.remove(fil)

    cand_info = {
        'sigma': sigma,
        'known': known,
        'ap': ap,
    }
    SNprof, SN_arr, plot_fname = plot_candidate_archive(
        archive_fname,
        coord_path,
        cand_info=cand_info,
        foldpath=foldpath + "/plots/folded_candidate_plots/",
    )

    log.info(f"SN of folded profile: {SN_arr}")
    fold_details = {
        "date": date,
        "archive_fname": archive_fname,
        "SN": float(SN_arr),
        "path_to_plot": plot_fname,
    }

    if fs_id and write_to_db:
        log.info("Updating FollowUpSource with folding history")
        folding_history = source.folding_history
        fold_dates = [entry["date"].date() for entry in folding_history]
        if date.date() in fold_dates:
            index = fold_dates.index(date.date())
            folding_history[index] = fold_details
        else:
            folding_history.append(fold_details)
            update_folding_history(fs_id, {"folding_history": fold_details})
        if len(folding_history) >= source.followup_duration:
            log.info(
                f"Finished follow-up duration of {source.followup_duration} days,"
                " setting active = False"
            )
            db_api.update_followup_source(fs_id, {"active": False})

    fold_details["date"] = fold_details["date"].strftime("%Y%m%d")
    # Silence Workflow errors, requires results, products, plots
    return fold_details, [plot_fname], [plot_fname]


if __name__ == "__main__":
    main()
    # main(year, month, day, sigma, dm, f0, ra, dec, known)
