import numpy as np
import matplotlib.pyplot as plt

import os
import glob
from scipy.ndimage import uniform_filter
from astropy.time import Time
from astropy.constants import au, c
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord, get_body_barycentric, EarthLocation
import astropy.units as u
import psrchive
import re
import datetime as dt
import subprocess

from folding.archive_utils import *

def get_ssb_delay(raj, decj, times):
    """
    Get Romer delay to Solar System Barycentre (SSB) for correction of site
    arrival times to barycentric.
    """
    
    coord = SkyCoord(raj, decj, frame=BarycentricTrueEcliptic,
                 unit=(u.hourangle, u.deg))
    psr_xyz = coord.cartesian.xyz.value
    earth_xyz = get_body_barycentric('earth', times).xyz.value
    t_bary = []
    for i in range(len(times)):
        e_dot_p = np.dot(earth_xyz[:,i], psr_xyz)
        t_bary.append(e_dot_p*au.value/c.value)
    return np.array(t_bary) * u.s  

def find_oldest_obs(directory, F0, DM):
        print("Finding oldest observation...")
        dates = []
        for filename in sorted(os.listdir(directory)):
            f = os.path.join(directory, filename)
            if os.path.isfile(f) and filename.endswith('.par'):
                if str(round(DM,2)) in filename:
                    date_txt = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", f)[0]
                    z = date_txt.split("-")
                    date = dt.date(int(z[0]), int(z[1]), int(z[2]))
                    dates.append(date)
                
        oldest_obs_date = min(dates)
        print(f"Oldest observation is {oldest_obs_date}")
        first_obs_par = f"{directory}/cand_{round(DM, 2)}_{round(F0, 2)}_" + oldest_obs_date.strftime('%Y-%m-%d') + ".par"

        return first_obs_par

def load_profiles(directory, F0, DM, RA, DEC, load_only=False):
    """
    Finds earliest observation and uses its ephemeris as the reference epoch for all other archives
    Apply this ephemeris to the archives, creating new archives with .newar extension
    Barycenters the squeezed archives (pulse profiles)
    
    Parameters
    ----------
    directory: string, location of archives
    DM, RA, DEC: float, from incoherent search
    load_only: Bool, set to True if only wanting to load the archive files
    
    Returns intensity data for each observation loaded
    """

    if not load_only: 
        first_obs_par = find_oldest_obs(directory, F0, DM)

        # Apply this ephemeris to each archive file, so that they have an absolute start time to reference
        print("Applying oldest observation ephemeris...")
        subprocess.run(["pam", "-E", first_obs_par, "-e", "newar", 
                        f"cand*{round(DM,2)}*{round(F0,2)}*.ar"], cwd=directory)

    print("Loading in altered archive files...")
    profs = []
    times = []
    for filename in sorted(os.listdir(directory)):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and filename.endswith('.newar'):
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
    raj = RA 
    decj = DEC  
    times = Time(times, format="mjd")
    t_bary = get_ssb_delay(raj, decj, times)
    dts = times + t_bary
    t0 = dts[0]
    dts = dts - t0
    dts = dts.to_value('second')
    data_time_array = [*zip(profs, dts)]  
    param_dict = dict({'data_time_array': data_time_array,'F0': F0,'DM': DM,'RA': RA,'DEC': DEC, 'directory': directory})
    return param_dict
