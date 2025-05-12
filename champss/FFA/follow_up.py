#!/usr/bin/env python

import numpy as np
import datetime
import os
import click
from omegaconf import OmegaConf
import logging
import beamformer.skybeam as bs
from beamformer.strategist.strategist import PointingStrategist
from sps_databases import db_api
from sps_databases import db_utils
from importlib import reload
from sps_pipeline import dedisp
from riptide import TimeSeries, ffa_search, find_peaks
import riptide
import time
import multiprocessing
import matplotlib.pyplot as plt
from datetime import date
import json
import csv
from astropy.time import Time
import h5py
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
import sps_common.barycenter as barycenter
from scipy import interpolate
from sps_pipeline import utils
from sps_common.constants import TSAMP
#from sps_common.interfaces import SinglePointingCandidate_FFA, SearchAlgorithm_FFA
try:
    import importlib.resources as pkg_resources
except ImportError as e:
    # For Python <3.7 we would need to use the backported version
    import importlib_resources as pkg_resources







@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--psr", 
    type=str, 
    default=None,
    required=False,
    help="PSR of known pulsar. Default is None")
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Date of data to process.",
)
@click.option("--ra", type=click.FloatRange(-180, 360), default=None)
@click.option("--dec", type=click.FloatRange(-90, 90), default=None)
@click.option(
    "--dm_target", 
    type=float,
    default=0, 
    help="Only search for DM closest to this value."
)
@click.option(
    "--freq_target", 
    type=float,
    default=0, 
    help="Only search for frequency closest to this value."
)
@click.option("--num_threads", type=int, default=16, help="Number of threads to use")
def main(
    psr, 
    date, 
    ra, 
    dec, 
    dm_target,
    freq_target,
    num_threads
):
    apply_logging_config('INFO')
    db_utils.connect(host = 'sps-archiver1', port = 27017, name = 'sps-processing')

    datetime.date = utils.convert_date_to_datetime(date)
    date_string = date.strftime("%Y/%m/%d")
    mjd_time = Time(date).mjd
    
    config_file = '/home/ltarabout/FFA/sps_config_ffa.yml'
    config = OmegaConf.load(config_file)
    if psr is not None:
        ra, dec = get_folding_pars(psr)
    elif ra is None or dec is None:
        log.error("Please provide either a PSR or a position in (ra,dec)")
        return
    
    pst = PointingStrategist(create_db=False)
    ap = pst.get_single_pointing(ra, dec, date)
    ra = ap[0].ra
    dec = ap[0].dec
    obs_id = ap[0].obs_id
    #print(ap)
    
    sbf = bs.SkyBeamFormer(
        extn="dat",
        update_db=False,
        min_data_frac=0.5,
        basepath="/data/chime/sps/raw/",
        add_local_median=True,
        detrend_data=True,
        detrend_nsamp=32768,
        masking_timescale=512000,
        run_rfi_mitigation=True,
        masking_dict=dict(weights=True, l1=True, badchan=True, kurtosis=False, mad=False, sk=True, powspec=False, dummy=False),
        beam_to_normalise=1,
    )
    skybeam, spectra_shared = sbf.form_skybeam(ap[0], num_threads=num_threads)
    
    dedisp_ts = dedisp.run_fdmt(
        ap[0], skybeam, config, num_threads
    )

    period_target = 1/freq_target
    tobs = len(dedisp_ts.dedisp_ts[0])*TSAMP
    # Compute barycentric shift
    barycentric_beta = barycenter.get_mean_barycentric_correction(str(ra),str(dec),mjd_time,tobs)
    log.info(f"Tobs: {tobs}, beta: {barycentric_beta}")

    dm_index = np.argmin(abs(dedisp_ts.dms - dm_target))
    ts = riptide.TimeSeries.from_numpy_array(dedisp_ts.dedisp_ts[dm_index], TSAMP)

    bins = 256
    # Need to fold at the TOPOCENTRIC period not barycentric
    subints = ts.fold((1-barycentric_beta)/(freq_target), bins)

    plt.subplot(211)
    plt.title(f"Folded profile at p={np.round(period_target,4)}s, dm={dedisp_ts.dms[dm_index]}, {date_string}")
    plt.imshow(subints, cmap='Greys', aspect='auto')
    plt.xlim(0,255)
    plt.ylabel("Subints")
    plt.subplot(212)
    plt.plot(subints.sum(axis=0))
    plt.xlabel("Bins")
    plt.ylabel("Intensity")
    plt.xlim(0,255)

    file_path = f"follow_up_plots/FFA_profile_ra_{np.round(ra,2)}_dec_{np.round(dec,2)}_p_{np.round(period_target,6)}_dm_{dedisp_ts.dms[dm_index]}.png"
    plt.savefig(file_path)
    log.info(f"Saved best folded pulse profile at {file_path}")
    plt.clf()


    del dedisp_ts, ts
    del skybeam, spectra_shared, sbf, pst, ap

    return



def apply_logging_config(level):
    """
    Applies logging settings from the given configuration
    Logging settings are under the 'logging' key, and include:
    - format: string for the `logging.formatter`
    - level: logging level for the root logger
    - modules: a dictionary of submodule names and logging level to be applied to that submodule's logger
    """
    log_stream.setFormatter(
        logging.Formatter(fmt="%(asctime)s %(levelname)s >> %(message)s", datefmt="%b %d %H:%M:%S")
    )    
    logging.root.setLevel(level)
    log.debug("Set default level to: %s", level)












log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)# import in this way for easy reload

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)
    main()