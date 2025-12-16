"""
This script is designed specifically for folding known pulsars, as part of 
the regular timing follow-up. It queries the timing_ops database for pulsar 
coordinates, folds at exact RA, using a user-specified parfile.
"""

import logging
import multiprocessing
import os

import click

# set these up before importing any SPS packages
log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from folding.fold import Fold
from sps_pipeline.pipeline import default_datpath, load_config, apply_logging_config
from folding.utilities.database import scrape_ephemeris, get_pulsar_coords_from_timing_db


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Date of data to process.",
)
@click.option(
    "--psr",
    type=str,
    required=True,
    help="Pulsar name (must exist in timing_ops database)",
)
@click.option(
    "--parfile",
    "-par",
    type=str,
    required=True,
    help="Path to ephemeris file for folding",
)
@click.option(
    "--foldpath",
    default="/data/foldmode_tmp/",
    type=str,
    help="Path for output archive files.",
)
@click.option(
    "--datpath",
    default=default_datpath,
    type=str,
    help="Path to the raw data folder.",
)
@click.option(
    "--filterbank-to-ram/--no-filterbank-to-ram",
    default=True,
    help="Use ramdisk for filterbank files, default True.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing archive file if it exists.",
)
@click.option(
    "--timing-db-url",
    default="mongodb://sps-archiver1:27017/",
    type=str,
    help="MongoDB URL for timing_ops database.",
)
@click.option(
    "--timing-db-name",
    default="timing_ops",
    type=str,
    help="Name of the timing_ops database.",
)
def main(
    date,
    psr,
    parfile,
    foldpath,
    datpath,
    filterbank_to_ram,
    overwrite,
    timing_db_url,
    timing_db_name,
):
    """
    Fold a known pulsar using an ephemeris file.

    This script is designed for folding known pulsars. It:
    1. Queries the timing_ops database for the pulsar's RA and Dec
    2. Beamforms at the exact coordinates (no grid snapping)
    3. Folds using dspsr with the provided ephemeris
    4. Saves the archive to the specified foldpath

    Note: This script does NOT create plots or update databases.
    """
    multiprocessing.set_start_method("forkserver", force=True)

    # Verify parfile exists
    if not os.path.exists(parfile):
        log.error(f"Ephemeris file {parfile} not found")
        return

    # Get pulsar coordinates from timing_ops database
    log.info(f"Querying timing_ops database for {psr}...")
    try:
        ra, dec = get_pulsar_coords_from_timing_db(psr, timing_db_url, timing_db_name)
        log.info(f"Found {psr} at RA={ra:.6f}, Dec={dec:.6f}")
    except ValueError as e:
        log.error(str(e))
        return

    # Parse ephemeris to get DM and F0 for beamforming parameters
    log.info(f"Reading ephemeris from {parfile}...")
    # Parse ephemeris to get ra, dec, f0, dm
    parvals = scrape_ephemeris(parfile)
    f0 = parvals['f0']
    dm = float(parvals['dm'])
    log.info(f"Ephemeris parameters: F0={f0:.6f} Hz, DM={dm:.2f} pc/cm^3")

    # Setup output paths
    output_dir = f"{foldpath}/{psr}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.info(f"Created directory: {output_dir}")
    else:
        log.info(f"Directory '{output_dir}' already exists.")

    archive_fname = f"{output_dir}/{psr}_{date.year}-{date.month:02}-{date.day:02}.ar"

    if os.path.isfile(archive_fname) and not overwrite:
        log.info(f"Archive file {archive_fname} already exists, skipping...")
        return

    # Setup logging
    config = load_config()
    log_path = foldpath + f"/logs/{date.strftime('%Y/%m/%d')}/"
    log_name = f"fold_pulsar_{psr}_{date.strftime('%Y-%m-%d')}.log"
    log_file = log_path + log_name
    apply_logging_config(config, log_file)

    # Initialize Fold object
    folder = Fold(
        ra=ra,
        dec=dec,
        f0=f0,
        dm=dm,
        date=date,
        ephem_path=parfile,
        foldpath=foldpath,
        datpath=datpath,
        config=config,
        name=psr,
        filterbank_to_ram=filterbank_to_ram,
        exact_coords=True,  # Always use exact coordinates for known pulsars
        coord_path=output_dir,
    )

    # Setup paths, use pulsar name instead of candidates/
    folder.coord_path = output_dir
    folder.setup_paths(dir_suffix="", archive_basename=psr)

    # Beamform
    folder.beamform()

    # Fold
    folder.fold()

    # Cleanup
    folder.cleanup()

    log.info(f"Successfully created archive: {folder.archive_fname}")


if __name__ == "__main__":
    main()
