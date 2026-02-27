"""
Utilities for handling frequency aliases in folding.

This module contains functions for folding at different frequency aliases,
to detect the true rotation perios of a new pulsar.
"""

import logging
import os
import subprocess

log = logging.getLogger(__name__)


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
