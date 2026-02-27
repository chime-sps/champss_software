"""
Helper utilities for folding operations.

This module contains common functions used by fold_candidate and fold_pulsar.
"""

import logging
import os
import subprocess

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

from sps_databases import db_api

log = logging.getLogger(__name__)


def candidate_name(ra_deg, dec_deg, j2000=True):
    """
    Create a J-name for a source from RA/Dec coordinates.

    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    j2000 : bool, optional
        Whether to use J2000 epoch (default: True)

    Returns
    -------
    name : str
        Source name in format JHHMM+DDMM or JHHMM-DDMM
    """
    ra_hhmmss = ra_deg * 24 / 360
    dec_ddmmss = abs(dec_deg)
    ra_str = f"{int(ra_hhmmss):02d}{int((ra_hhmmss * 60) % 60):02d}"
    dec_sign = "+" if dec_deg >= 0 else "-"
    dec_str = f"{int(dec_ddmmss):02d}{int((dec_ddmmss * 60) % 60):02d}"
    candidate_name = "J" + ra_str + dec_sign + dec_str
    return candidate_name


def create_ephemeris(name, ra, dec, dm, obs_date, f0, ephem_path, fs_id=False):
    """
    Create an ephemeris file for folding.

    Parameters
    ----------
    name : str
        Source name
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    dm : float
        Dispersion measure in pc/cm^3
    obs_date : datetime
        Observation date
    f0 : float
        Spin frequency in Hz
    ephem_path : str
        Path where ephemeris file will be written
    fs_id : str or ObjectId, optional
        If provided, update the followup_source in database with ephemeris path
    """
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
