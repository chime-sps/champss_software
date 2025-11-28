import os

import astropy.units as u
import numpy as np
import astropy.constants as const
from astropy.coordinates import (
    BarycentricTrueEcliptic,
    EarthLocation,
    SkyCoord,
    get_body_barycentric,
)
from astropy.time import Time
from folding.archive_utils import *


def get_ssb_delay(raj, decj, times):
    """Get Römer delay to Solar System Barycentre (SSB)."""

    # Pulsar unit vector in ICRS
    coord = SkyCoord(raj, decj, frame="icrs", unit=(u.hourangle, u.deg))
    psr_xyz = coord.cartesian.xyz.to(u.one)  # dimensionless, unit vector

    # Earth position wrt SSB (with units!)
    earth_xyz = get_body_barycentric("earth", times).xyz  # in AU by default

    # Dot product -> length (still with units, e.g. AU)
    e_dot_p = np.einsum("ij,i->j", earth_xyz.to(u.m).value, psr_xyz.value) * u.m

    t_bary = e_dot_p / const.c
    # Divide by c -> time
    return t_bary.to(u.s)


def unwrap_profiles(profiles, dts, f0, f1):
    npbin = profiles.shape[-1]
    dphis = f0 * dts + 0.5 * f1 * dts**2
    i_phis = (dphis * npbin).astype("int")
    profs_shifted = np.zeros_like(profiles)
    for k, prof in enumerate(profiles):
        profs_shifted[k] = np.roll(prof, -i_phis[k], axis=-1)
    return profs_shifted


def load_profiles(archives, max_npbin=256):
    """
    Load profiles from archive files.

    Args:
        archives (list): List of archive file paths.
        max_npbin (int, optional): Maximum number of phase bins. Defaults to 256.

    Returns:
        dict: Dictionary containing the following parameters:
            - "archives": List of archive file paths.
            - "profiles": Array of loaded profiles.
            - "times": Array of time values.
            - "F0": Value of F0 parameter.
            - "DM": Value of DM parameter.
            - "RA": Value of RAJD parameter.
            - "DEC": Value of DECJD parameter.
            - "directory": Directory of the first archive file.
            - "npbin": Number of phase bins.
            - "T": Maximum time difference from reference.

    Raises:
        ValueError: If not all profiles reference the same PEPOCHs.
    """
    print("Loading in archive files...")
    profs = []
    times = []
    PEPOCHs = []
    print(*archives,sep='\n')
    for filename in sorted(archives):
        f = filename.replace(".ar", ".FT")
        if os.path.isfile(f) and f.endswith(".FT"):
            print(f)
            data_ar, params = readpsrarch(f)
            T = params["T"]
            data_ar = data_ar.squeeze()
            if len(data_ar.shape) > 1:
                prof = data_ar.sum(0)
                prof = prof.sum(0)
            else:
                prof = data_ar
            prof = prof - np.median(prof)
            profs.append(prof)
            times.append(T[0])
            PEPOCH = get_archive_parameter(f, "PEPOCH")
            PEPOCHs.append(PEPOCH)

    if np.unique(PEPOCHs).size > 1:
        print(
            "Not all profiles reference the same PEPOCHs, re-apply same ephemeris to"
            " all archives"
        )
        raise ValueError("Not all profiles reference the same PEPOCHs")

    T0 = Time(PEPOCH, format="mjd")
    print(f"Reference epoch: {T0}, {T0.isot}")
    times = Time(times, format="mjd")
    if (min(times) > T0) or (max(times) < T0):
        print("Warning: PEPOCH is outside range of observation epochs")
        T0 = Time(np.median(times.mjd), format="mjd")
        print(f"Changing reference epoch to central observation {T0}, {T0.isot}")


    npbin = len(profs[0])
    profs = np.array(profs)
    if npbin > max_npbin:
        print(f"Binning to {max_npbin} phase bins.")
        profs = profs.reshape(
            profs.shape[0], max_npbin, profs.shape[1] // max_npbin
        ).sum(2)
        npbin = max_npbin

    RA = get_archive_parameter(f, "RAJD")
    DEC = get_archive_parameter(f, "DECJD")
    F0 = get_archive_parameter(f, "F0")
    DM = get_archive_parameter(f, "DM")
    directory = os.path.dirname(archives[0])

    t_bary = get_ssb_delay(RA, DEC, times)

    dts = times + t_bary
    dts = dts - T0
    dts = dts.to_value("second")
    Tmax_from_reference = max(abs(dts))
    times = np.array(dts)

    param_dict = dict(
        {
            "archives": archives,
            "profiles": profs,
            "times": times,
            "F0": F0,
            "DM": DM,
            "RA": RA,
            "DEC": DEC,
            "directory": directory,
            "npbin": npbin,
            "PEPOCH": T0.mjd,
            "Tmax_from_reference": Tmax_from_reference,
        }
    )
    return param_dict


def load_unwrapped_archives(archives, optimal_parameters, max_npbin=256, max_nfbin=128):
    """
    Load and unwrap full archives.

    Args:
        archives (list): List of archive file paths.
        optimal_parameters (list): list containing optimal F0, F1
        max_npbin (int, optional): Maximum number of phase bins. Defaults to 256.
        max_nfbin (int, optional): Maximum number of frequency bins. Defaults to 128.

    Returns:
        tuple: A tuple containing the data arrays for time and frequency.
    """

    print("Loading and unwrapping full archives...")
    times = []
    PEPOCHs = []

    F0_incoherent = optimal_parameters[0]
    F1_incoherent = optimal_parameters[1]
    for i, f in enumerate(sorted(archives)):
        print(f)
        data_ar, params = readpsrarch(f)
        F, times = params["F"], params["T"]
        data_ar = data_ar.squeeze()
        data_ar = data_ar - np.median(data_ar, axis=-1, keepdims=True)

        RA = get_archive_parameter(f, "RAJD")
        DEC = get_archive_parameter(f, "DECJD")
        F0 = get_archive_parameter(f, "F0")
        PEPOCH = get_archive_parameter(f, "PEPOCH")
        PEPOCHs.append(PEPOCH)
        T0 = Time(PEPOCH, format="mjd")

        times = Time(times, format="mjd")
        t_bary = get_ssb_delay(RA, DEC, times)
        dts = times + t_bary - T0
        dts = dts.to_value("second")
        dF0 = F0_incoherent - F0
        dF1 = F1_incoherent

        data_unwrapped = unwrap_profiles(data_ar, dts, -dF0, -dF1)
        if i == 0:
            data_F = data_unwrapped.sum(0)
            data_T = data_unwrapped.sum(1)
        else:
            data_F += data_unwrapped.sum(0)
            # Sometimes the data_T shape mismatch, need to diagnose
            try:
                data_T += data_unwrapped.sum(1)
            except ValueError:
                print("Data_T shape mismatch, skipping")

    npbin = data_F.shape[-1]
    if npbin > max_npbin:
        print(f"Binning to {max_npbin} phase bins.")
        data_F = data_F.reshape(
            data_F.shape[0], max_npbin, data_F.shape[1] // max_npbin
        ).sum(2)
        data_T = data_T.reshape(
            data_T.shape[0], max_npbin, data_T.shape[1] // max_npbin
        ).sum(2)
        npbin = max_npbin
    nfbin = data_F.shape[0]
    if nfbin > max_nfbin:
        print(f"Binning to {max_nfbin} frequency bins.")
        data_F = data_F.reshape(max_nfbin, data_F.shape[0] // max_nfbin, -1).sum(1)
        nfbin = max_nfbin

    return data_T, data_F
