import os

import astropy.units as u
import numpy as np
from astropy.constants import au, c
from astropy.coordinates import (
    BarycentricTrueEcliptic,
    EarthLocation,
    SkyCoord,
    get_body_barycentric,
)
from astropy.time import Time
from folding.archive_utils import *


def get_ssb_delay(raj, decj, times):
    """Get Romer delay to Solar System Barycentre (SSB) for correction of site arrival
    times to barycentric.
    """

    coord = SkyCoord(
        raj, decj, frame=BarycentricTrueEcliptic, unit=(u.hourangle, u.deg)
    )
    psr_xyz = coord.cartesian.xyz.value
    earth_xyz = get_body_barycentric("earth", times).xyz.value
    t_bary = []
    for i in range(len(times)):
        e_dot_p = np.dot(earth_xyz[:, i], psr_xyz)
        t_bary.append(e_dot_p * au.value / c.value)
    return np.array(t_bary) * u.s


def load_profiles(archives, max_npbin=256):
    """
    Finds central observation and uses its ephemeris as the reference epoch for all
    other archives Apply this ephemeris to the archives, creating new archives with
    .newar extension Barycenters the squeezed archives (pulse profiles)

    Parameters
    ----------
    directory: string, location of archives
    DM, RA, DEC: float, from incoherent search
    load_only: Bool, set to True if only wanting to load the archive files

    Returns intensity data for each observation loaded
    """

    print("Loading in altered archive files...")
    profs = []
    times = []
    PEPOCHs = []
    print(archives)
    for filename in sorted(archives):
        f = filename.replace(".ar", ".FT")
        if os.path.isfile(f) and f.endswith(".FT"):
            print(f)
            data_ar, F, T, source, tel = readpsrarch(f)
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
        log.error(
            "Not all profiles reference the same PEPOCHs, re-apply same ephemeris to"
            " all archives"
        )
        return
    T0 = Time(PEPOCHs[0], format="mjd")

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

    times = Time(times, format="mjd")
    t_bary = get_ssb_delay(RA, DEC, times)
    dts = times + t_bary
    dts = dts - T0
    dts = dts.to_value("second")
    Tmax_from_reference = max(abs(dts))
    times = np.array(dts)

    param_dict = dict(
        {
            "profiles": profs,
            "times": times,
            "F0": F0,
            "DM": DM,
            "RA": RA,
            "DEC": DEC,
            "directory": directory,
            "npbin": npbin,
            "T": Tmax_from_reference,
        }
    )
    return param_dict
