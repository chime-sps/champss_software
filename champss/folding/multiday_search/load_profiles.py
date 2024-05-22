import datetime as dt
import os
import re
import subprocess

import astropy.units as u
import numpy as np
from astropy.constants import au, c
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord, get_body_barycentric
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


def find_central_obs(directory, F0, DM, message=True):
    print("Finding oldest observation...")
    dates = []
    for filename in sorted(os.listdir(directory)):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and filename.endswith(".ar"):
            if str(round(DM, 2)) in filename:
                date_txt = re.search(r"([0-9]{4}\-[0-9]{2}\-[0-9]{2})", f)[0]
                z = date_txt.split("-")
                date = dt.date(int(z[0]), int(z[1]), int(z[2]))
                dates.append(date)

    oldest_obs_date = min(dates)
    central_obs_date = dt.date.fromordinal(
        int(np.median([date.toordinal() for date in dates]))
    )
    if message:
        print(f"Oldest observation is {oldest_obs_date}")
        print(
            f"Center observation is {central_obs_date}, referencing ephemeris to this"
            " date"
        )
    # first_obs_par = f"{directory}/cand_{round(DM, 2)}_{round(F0, 2)}_" + oldest_obs_date.strftime('%Y-%m-%d') + ".par"
    central_obs_par = (
        f"{directory}/cand_{round(DM, 2)}_{round(F0, 2)}_"
        + central_obs_date.strftime("%Y-%m-%d")
        + ".par"
    )

    return central_obs_par


def load_profiles(directory, F0, DM, RA, DEC, load_only=False, max_npbin=256):
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

    if not load_only:
        central_obs_par = find_central_obs(directory, F0, DM)

        # Apply this ephemeris to each archive file, so that they have an absolute start time to reference
        print("Applying central observation ephemeris...")
        subprocess.run(
            [
                "pam",
                "-E",
                central_obs_par,
                "-e",
                ".newar.FT",
                f"cand*{round(DM,2)}*{round(F0,2)}*-??.FT",
            ],
            cwd=directory,
        )

    print("Loading in altered archive files...")
    profs = []
    times = []
    PEPOCHs = []
    for filename in sorted(os.listdir(directory)):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and filename.endswith(".newar.FT"):
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
        print(
            "WARNING: not all profiles references to the same PEPOCHs, re-apply same"
            " ephemeris to all archives"
        )
    T0 = Time(PEPOCHs[0], format="mjd")

    npbin = len(profs[0])
    profs = np.array(profs)
    if npbin > max_npbin:
        print(f"Binning to {max_npbin} phase bins.")
        profs = profs.reshape(
            profs.shape[0], max_npbin, profs.shape[1] // max_npbin
        ).sum(2)
        npbin = max_npbin

    raj = RA
    decj = DEC
    times = Time(times, format="mjd")
    t_bary = get_ssb_delay(raj, decj, times)
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
