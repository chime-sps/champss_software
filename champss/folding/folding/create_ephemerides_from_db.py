"""
Script to create ephemeris files for all known pulsars in the database.

For each pulsar in the known_sources database:
- If no parfile exists, create one using parameters from the database
- If parfile exists but has no PEPOCH, add a PEPOCH line
"""

import os

import click
from astropy.time import Time

from folding.fold_candidate import create_ephemeris
from sps_databases import db_api, db_utils


def create_ephemeris_from_db(psr_name, obs_date=None, ephem_path=None):
    """
    Create an ephemeris file for a known pulsar using parameters from the database.

    Parameters
    ----------
    psr_name : str
        Name of the pulsar to look up in the known_sources database
    obs_date : datetime or str, optional
        Observation date for PEPOCH. If None, uses current time.
    ephem_path : str, optional
        Path to save the ephemeris file. If None, uses './{psr_name}.par'

    Returns
    -------
    ephem_path : str
        Path to the created ephemeris file

    Raises
    ------
    ValueError
        If pulsar is not found in the database
    """
    # Query the database for the pulsar
    sources = db_api.get_known_source_by_names(psr_name)
    if not sources:
        raise ValueError(f"Pulsar '{psr_name}' not found in known_sources database")

    source = sources[0]

    # Extract parameters from database
    ra = source.pos_ra_deg
    dec = source.pos_dec_deg
    dm = source.dm
    f0 = 1 / source.spin_period_s

    # Set defaults
    if obs_date is None:
        obs_date = Time.now()
    if ephem_path is None:
        ephem_path = f"./{psr_name}.par"

    # Create the ephemeris
    create_ephemeris(psr_name, ra, dec, dm, obs_date, f0, ephem_path)
    print(f"Created ephemeris for {psr_name} at {ephem_path}")

    return ephem_path


def check_parfile_has_pepoch(parfile_path):
    """
    Check if a parfile has a PEPOCH entry.

    Parameters
    ----------
    parfile_path : str
        Path to the parfile

    Returns
    -------
    bool
        True if PEPOCH is present, False otherwise
    """
    try:
        with open(parfile_path, 'r') as f:
            for line in f:
                if line.strip().startswith('PEPOCH'):
                    return True
        return False
    except Exception as e:
        print(f"Warning: Error reading {parfile_path}: {e}")
        return False


def check_parfile_is_valid(parfile_path):
    """
    Check if a parfile is valid and contains all essential parameters.

    A valid parfile must have: PSRJ, RAJ, DECJ, DM, F0, PEPOCH

    Parameters
    ----------
    parfile_path : str
        Path to the parfile

    Returns
    -------
    bool
        True if parfile is valid, False if corrupted or missing parameters
    """
    required_params = {'PSRJ', 'RAJ', 'DECJ', 'DM', 'F0', 'PEPOCH'}
    found_params = set()

    try:
        with open(parfile_path, 'r') as f:
            for line in f:
                # Skip comments and warnings
                if line.startswith('#') or line.startswith('WARNING'):
                    continue

                # Check if line starts with any required parameter
                for param in required_params:
                    if line.strip().startswith(param):
                        found_params.add(param)
                        break

        # Check if all required parameters are present
        missing_params = required_params - found_params
        if missing_params:
            print(f"Parfile {parfile_path} is missing: {missing_params}")
            return False
        return True

    except Exception as e:
        print(f"Warning: Error reading {parfile_path}: {e}")
        return False


def add_pepoch_to_parfile(parfile_path, pepoch=None):
    """
    Add a PEPOCH line to an existing parfile.

    Parameters
    ----------
    parfile_path : str
        Path to the parfile
    pepoch : float, optional
        PEPOCH value in MJD. If None, uses current time.
    """
    if pepoch is None:
        pepoch = Time.now().mjd

    try:
        with open(parfile_path, 'r') as f:
            lines = f.readlines()

        # Find a good place to insert PEPOCH (after F0 if present, otherwise at end)
        insert_idx = len(lines)
        for i, line in enumerate(lines):
            if line.strip().startswith('F0'):
                insert_idx = i + 1
                break

        pepoch_line = f"PEPOCH\t{pepoch}\n"
        lines.insert(insert_idx, pepoch_line)

        with open(parfile_path, 'w') as f:
            f.writelines(lines)

        print(f"Added PEPOCH to {parfile_path}")
    except Exception as e:
        print(f"Error adding PEPOCH to {parfile_path}: {e}")


def get_all_known_sources():
    """
    Get all known sources from the database.

    Returns
    -------
    list
        List of known source objects
    """
    db = db_utils.connect()
    sources = list(db.known_sources.find({}))
    return sources


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--directory",
    "-d",
    required=True,
    type=click.Path(exists=False),
    help="Directory to store/check ephemeris files",
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
    "--pepoch",
    type=float,
    default=None,
    help="PEPOCH value in MJD to use for new ephemerides. Default: current time.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Only show what would be done, don't create/modify files.",
)
def main(directory, db_port, db_host, db_name, pepoch, dry_run):
    """
    Create ephemeris files for all known pulsars in the database.

    For each pulsar:
    - If no parfile exists in the directory, create one
    - If parfile exists but has no PEPOCH, add a PEPOCH line
    """
    # Connect to database
    db_utils.connect(host=db_host, port=db_port, name=db_name)

    # Create directory if it doesn't exist
    if not dry_run and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # Get all known sources
    sources = get_all_known_sources()
    print(f"Found {len(sources)} known sources in database")

    # Set default PEPOCH
    if pepoch is None:
        pepoch_mjd = Time.now().mjd
    else:
        pepoch_mjd = pepoch
    obs_date = Time(pepoch_mjd, format='mjd')

    # Track statistics
    created = 0
    updated = 0
    regenerated = 0
    skipped = 0
    errors = 0

    for source in sources:
        psr_name = source.get('source_name')
        if not psr_name:
            continue

        parfile_path = os.path.join(directory, f"{psr_name}.par")

        if os.path.exists(parfile_path):
            # First check if parfile is valid
            if not check_parfile_is_valid(parfile_path):
                # Regenerate corrupted parfile
                if dry_run:
                    print(f"[DRY RUN] Would regenerate corrupted ephemeris for {psr_name}")
                    regenerated += 1
                else:
                    try:
                        print(f"Regenerating corrupted ephemeris for {psr_name}")
                        create_ephemeris_from_db(psr_name, obs_date=obs_date, ephem_path=parfile_path)
                        regenerated += 1
                    except Exception as e:
                        print(f"Error regenerating ephemeris for {psr_name}: {e}")
                        errors += 1
            # Check if PEPOCH exists (for valid parfiles)
            elif check_parfile_has_pepoch(parfile_path):
                skipped += 1
            else:
                # Add PEPOCH to existing file
                if dry_run:
                    print(f"[DRY RUN] Would add PEPOCH to {parfile_path}")
                else:
                    add_pepoch_to_parfile(parfile_path, pepoch_mjd)
                updated += 1
        else:
            # Create new ephemeris
            if dry_run:
                print(f"[DRY RUN] Would create ephemeris for {psr_name}")
                created += 1
            else:
                try:
                    create_ephemeris_from_db(psr_name, obs_date=obs_date, ephem_path=parfile_path)
                    created += 1
                except Exception as e:
                    print(f"Error creating ephemeris for {psr_name}: {e}")
                    errors += 1

    # Print summary
    print("=" * 50)
    print("Summary:")
    print(f"  Created: {created}")
    print(f"  Regenerated (corrupted): {regenerated}")
    print(f"  Updated (added PEPOCH): {updated}")
    print(f"  Skipped (already complete): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total processed: {created + regenerated + updated + skipped + errors}")


if __name__ == "__main__":
    main()
