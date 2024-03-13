import numpy as np
import glob
import datetime as dt

from sps_databases import db_api

def Filter(cand_obs_date, min_sigma=10., return_candidates=True, save_candidates=True, fold_known_sources=False):
    """
    Read a days worth of candidates, retrieve a set of the most promising candidates to fold. 
    Filters based on a set of DM, F0, sigma, and spatial tests.

    Arguments: datetime, return_candidates (bool), save_candidates (bool), min_sigma (float)
    if return_candidates if True, returns the filtered candidates as a dictionary
    if save_candidates is True, saves the filtered candidates to a npz file

    Will be superseeded by / merged with the MultiPointingCandidates class.

    Creates an npz file to be used in fold_batch.py, with the following arrays:
    sigmas: array of sigmas
    dms: array of dms
    ras: array of ras
    decs: array of decs
    f0s: array of f0s
    known: array of known pulsars (True/False)
    
    """

    print(cand_obs_date)
    min_dm = 5

    print("Creating power spectra candidate file names...")
    cand_obs_year = cand_obs_date.year
    cand_obs_month = str(cand_obs_date.month).zfill(2)
    cand_obs_day = str(cand_obs_date.day).zfill(2)
    basepath = "/data/chime/sps/sps_processing"
    cand_dir = "{}/{}/{}/{}/*/*_power_spectra_candidates.npz".format(
        basepath,
        cand_obs_year, 
        cand_obs_month, 
        cand_obs_day)
    fnames_single_day = np.sort(glob.glob(cand_dir))
    Nfiles = len(fnames_single_day)

    # Save arrays for each candidate file 
    print(f"Loading in {Nfiles} power spectra candidate npz files...")

    ncands = []
    sigmas = []
    ras = []
    decs = []
    f0s = []
    dms = []
    for k,fname in enumerate(fnames_single_day): # For each observation
        cand_lib = np.load(fname, allow_pickle=True)
        cand_dicts = cand_lib['candidate_dicts']
        N = len(cand_dicts)
        if N >= 1:
            dm = np.zeros(N)
            sigma = np.zeros(N)
            f0 = np.zeros(N)
            ra = np.zeros(N)
            dec = np.zeros(N)

            for i in range(N):
                dm[i] = cand_dicts[i]['dm']
                sigma[i] = cand_dicts[i]['sigma']
                f0[i] = cand_dicts[i]['freq']
                ra[i] = cand_dicts[i]['ra']
                dec[i] = cand_dicts[i]['dec']
            threshold = np.where(sigma > min_sigma) 
            sigma = sigma[threshold]
            ra = ra[threshold]
            dec = dec[threshold]
            f0 = f0[threshold]
            dm = dm[threshold]
            
            # currently hardcoded, as many 60Hz candidates are coming through
            birdie_filter = np.where(np.logical_or(f0 > 60.5, f0 < 59.5))
            sigma = sigma[birdie_filter]
            ra = ra[birdie_filter]
            dec = dec[birdie_filter]
            f0 = f0[birdie_filter]
            dm = dm[birdie_filter]

            # taking the best candidate above max_sigma per pointing
            if len(sigma) > 0:
                ibest = np.argmax(sigma)
                sigmas.append(sigma[ibest])
                ras.append(ra[ibest])
                decs.append(dec[ibest])
                f0s.append(f0[ibest])
                dms.append(dm[ibest])

    sigmas = np.array(sigmas)
    ras = np.array(ras)
    decs = np.array(decs)
    f0s = np.array(f0s)
    dms = np.array(dms)

    counts = len(sigmas)
    print(f"Number of best candidates before filtering: {counts}")
    
    filter_idx = np.where(np.logical_and(sigmas >= min_sigma, dms > min_dm))
    sigmas = sigmas[filter_idx]
    dms = dms[filter_idx]
    f0s = f0s[filter_idx]
    ras = ras[filter_idx]
    decs = decs[filter_idx]
        
    counts = len(sigmas)
    print(f"Number of best candidates after filtering for DM and sigma: {counts}")
        
    # P_min = 0.01 sec = 10 ms, P_max = 100 sec 
    filter_idx = np.where(np.logical_and(f0s > 0.01, f0s < 100))
    sigmas = sigmas[filter_idx]
    dms = dms[filter_idx]
    f0s = f0s[filter_idx]
    ras = ras[filter_idx]
    decs = decs[filter_idx]
        
    counts = len(sigmas)
    print(f"Number of best candidates after filtering for frequency: {counts}")
        
    # Save filtered candidates to a dictionary
    filtered_data = {
        'ras': [],
        'decs': [],
        'f0s': [],
        'dms': [],
        'sigmas': [],
        'known': []
    }

    for i in range(len(decs)):

        sources = db_api.get_nearby_known_sources(ras[i], decs[i], radius=3)
        known_found = ' '
        append_cands = 1
        if len(sources) >=1:

            for source in sources:
                name = source.source_name
                Fs = 1. / source.spin_period_s
                ddm = np.abs(dms[i] - source.dm)
                dF0 = np.abs(f0s[i] - Fs)
                if (ddm < 1.) and (dF0 < Fs/100.):
                    known_found = name
                    if not fold_known_sources:
                        append_cands = 0
        if append_cands:
            filtered_data['ras'].append(ras[i])
            filtered_data['decs'].append(decs[i])
            filtered_data['f0s'].append(f0s[i])
            filtered_data['dms'].append(dms[i])
            filtered_data['sigmas'].append(sigmas[i])
            filtered_data['known'].append(known_found)

    filtered_data = {k: np.array(v) for k, v in filtered_data.items()}

    sigmas = filtered_data["sigmas"]
    dms = filtered_data["dms"]
    f0s = filtered_data["f0s"]
    ras = filtered_data["ras"]
    decs = filtered_data["decs"]
    known = filtered_data["known"]
    npz_path = "/data/chime/sps/archives/candidates/filtered_cands/cands-{}-{}-{}_filtered"
    if save_candidates:
        print("Saving filtered data...")
        np.savez(npz_path.format(cand_obs_year, cand_obs_month, cand_obs_day),sigmas=sigmas,dms=dms,ras=ras,decs=decs,f0s=f0s, known=known)
    if return_candidates:
        return filtered_data

import click
@click.command()
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Date of data to process. Default = Today in UTC",
)
@click.option(
    "--min_sigma",
    type=float,
    default=10.,
    help="Minimum sigma to filter candidates. Default = 10",
)
@click.option(
    "--return_candidates",
    type=bool,
    default=True,
    help="Return filtered candidate dictionary. Default = True",
)
@click.option(
    "--save_candidates",
    type=bool,
    default=True,
    help="Save filtered candidates to npz file. Default = True",
)
@click.option(
    "--fold_known_sources",
    type=bool,
    default=False,
    help="Fold candidates matching known sources. Default = False",
)
def main(date,min_sigma,return_candidates,save_candidates,fold_known_sources):
    Filter(date, min_sigma, return_candidates, save_candidates, fold_known_sources)

if __name__ == '__main__':
    main()
