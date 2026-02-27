import logging
import multiprocessing
import os
import click

# set these up before importing any SPS packages
log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from beamformer.utilities.common import find_closest_pointing
from folding.fold import Fold
from folding.utilities.utils import candidate_name, create_ephemeris
from folding.plot_aliases import plot_aliases
from folding.utilities.database import update_folding_history
from scheduler.utils import convert_date_to_datetime
from sps_databases import db_api, db_utils
from sps_pipeline.pipeline import default_datpath, load_config, apply_logging_config


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Date of data to process. Default = Today in UTC",
)
@click.option(
    "--foldvalues",
    type=str,
    default="",
    help="Space-separated candidate parameters: 'ra dec f0 dm' (degrees, degrees, Hz, pc/cm^3)",
)
@click.option(
    "--parfile",
    "-par",
    type=str,
    default="",
    help="Path to ephemeris file for folding",
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
    help="Write folding results to database (only used with --fs_id).",
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
    help="Use exact RA/Dec coordinates without snapping to pointing map grid.",
)
def main(
    date,
    foldvalues,
    parfile,
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
    Perform the main processing steps for folding a candidate.

    The main automated processing will use FollowUpSource ID, and update the database with the folding history.

    Use cases:
    1. --fs_id: Read from database using FollowUpSource ID
    2. --foldvalues: Fold with explicit ra, dec, f0, dm values
    3. --parfile: Fold using an explicit ephemeris file
    4. --candpath: Fold from a candidate file

    Args:
        date (str or datetime.datetime): The date of the observation.
        foldvalues (str): Space-separated "ra dec f0 dm" values.
        parfile (str): Path to parfile.
        fs_id (str): FollowUpSource ID from the database.
        db_port (int): The port number for the database connection.
        db_host (str): The hostname of the database.
        db_name (str): The name of the database.
        foldpath (str): The basepath for fold outputs.
        datpath (str): The basepath for raw data.
        candpath (str): Path to candidate file.
        write_to_db (bool, optional): Whether to write the results to the database (only with --fs_id).
        using_workflow (bool, optional): Whether to run the function through Workflow.
        overwrite_folding (bool, optional): Re-run folding even if already done.
        filterbank_to_ram (bool, optional): Use ramdisk for filterbank files.
        fold_aliases (bool, optional): Fold at multiple frequency aliases.
        exact_coords (bool, optional): Use exact coordinates without grid snapping.

    Returns:
        tuple: A tuple containing fold_details dict, list of plot filenames, list of plot filenames.
    """

    if using_workflow:
        date = convert_date_to_datetime(date)

    multiprocessing.set_start_method("forkserver", force=True)
    db_utils.connect(host=db_host, port=db_port, name=db_name)

    # Parse input parameters
    ephem_path = None
    sigma = None

    if fs_id:
        # Main production use case: read from database
        source = db_api.get_followup_source(fs_id)

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
        dir_suffix = "candidates"
        name = candidate_name(ra, dec)
        ephem_path = source.path_to_ephemeris

    elif foldvalues:
        # Parse space-separated "ra dec f0 dm" values
        try:
            values = foldvalues.split()
            if len(values) != 4:
                log.error("--foldvalues must contain exactly 4 space-separated values: ra dec f0 dm")
                return {}, [], []
            ra, dec, f0, dm = map(float, values)
        except ValueError as e:
            log.error(f"Failed to parse --foldvalues: {e}")
            return {}, [], []

        if not exact_coords:
            coords = find_closest_pointing(ra, dec)
            ra = coords.ra
            dec = coords.dec
        dir_suffix = "candidates"
        name = candidate_name(ra, dec)

    elif parfile:
        # Explicit ephemeris file provided
        if not os.path.exists(parfile):
            log.error(f"Ephemeris file {parfile} not found")
            return {}, [], []
        ephem_path = parfile

        # Parse ephemeris to get ra, dec, f0, dm
        from folding.utilities.database import scrape_ephemeris
        payload = scrape_ephemeris(parfile)
        ra = payload['ra']
        dec = payload['dec']
        f0 = payload['f0']
        dm = float(payload['dm'])

        if not exact_coords:
            coords = find_closest_pointing(ra, dec)
            ra = coords.ra
            dec = coords.dec
        dir_suffix = "candidates"
        name = candidate_name(ra, dec)

    elif candpath:
        # Fold from candidate file
        from folding.utilities.database import add_candidate_to_fsdb
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

        # Only add to database if write_to_db is True and fs_id wasn't already provided
        if write_to_db:
            followup_source = add_candidate_to_fsdb(
                date_str, ra, dec, f0, dm, sigma, candpath
            )
            fs_id = followup_source._id
    else:
        log.error(
            "Must provide one of: --fs_id, --foldvalues, --parfile, or --candpath"
        )
        return {}, [], []

    config = load_config()
    log_path = foldpath + f"/logs/{date.strftime('%Y/%m/%d')}/"
    log_name = (
        f"fold_candidate{date.strftime('%Y-%m-%d')}_{ra:.02f}_"
        f"{dec:.02f}_{f0:.02f}_{dm:.02f}.log"
    )
    log_file = log_path + log_name
    apply_logging_config(config, log_file)

    directory_path = f"{foldpath}/{dir_suffix}"
    coord_path = f"{directory_path}/{ra:.02f}_{dec:.02f}"

    # Create ephemeris if not provided
    if not ephem_path:
        if not os.path.exists(coord_path):
            os.makedirs(coord_path)
        ephem_path = (
            f"{coord_path}/cand_{f0:.02f}_{dm:.02f}_{date.year}-{date.month:02}-{date.day:02}.par"
        )
        create_ephemeris(name, ra, dec, dm, date, f0, ephem_path, fs_id)

    # Check if archive already exists
    archive_fname = (
        f"{coord_path}/cand_{f0:.02f}_{dm:.02f}_{date.year}-{date.month:02}-{date.day:02}.ar"
    )
    if os.path.isfile(archive_fname) and not overwrite_folding:
        log.info(f"Archive file {archive_fname} already exists, skipping folding...")
        return {}, [], []

    if not os.path.exists(ephem_path):
        log.error(f"Ephemeris file {ephem_path} not found")
        return {}, [], []

    # Initialize Fold object
    folder = Fold(
        ra=ra,
        dec=dec,
        f0=f0,
        dm=dm,
        date=date,
        ephem_path=ephem_path,
        foldpath=foldpath,
        datpath=datpath,
        config=config,
        name=name,
        filterbank_to_ram=filterbank_to_ram,
        exact_coords=exact_coords,
        coord_path=coord_path,
    )

    # Setup paths
    folder.setup_paths(dir_suffix=dir_suffix)

    # Beamform
    if not folder.beamform():
        log.error("Beamforming failed")
        return {}, [], []

    # Fold
    if not folder.fold():
        log.error("Folding failed")
        folder.cleanup()
        return {}, [], []

    # Fold aliases if requested
    alias_results = {}
    if fold_aliases:
        alias_results = folder.fold_aliases()

        # Plot alias results
        if alias_results:
            alias_plot_path = os.path.join(
                folder.coord_path, "aliases", f"alias_plot_{f0:.02f}_{dm:.02f}_{date.year}-{date.month:02}-{date.day:02}.png"
            )
            plot_aliases(alias_results, output_path=alias_plot_path)

    # Plot
    SNprof, SN_arr, plot_fname = folder.plot(
        sigma=sigma,
        foldpath_plots=foldpath + "/plots/folded_candidate_plots/"
    )

    # Cleanup
    folder.cleanup()

    log.info(f"SN of folded profile: {SN_arr}")
    fold_details = {
        "date": date,
        "archive_fname": folder.archive_fname,
        "SN": float(SN_arr),
        "path_to_plot": plot_fname,
    }

    # Update database
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
