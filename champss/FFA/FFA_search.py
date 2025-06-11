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

import matplotlib
matplotlib.use("Agg") #Avoids overhead of interactable plot GUI backends
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

from single_pointing_FFA import SinglePointingCandidate_FFA, SearchAlgorithm_FFA, detections_dtype
from clustering_FFA import Clusterer_FFA
from interfaces_FFA import Cluster_FFA

import yaml
from rfi_mitigation import data as rfi_mitigation_data
try:
    import importlib.resources as pkg_resources
except ImportError as e:
    # For Python <3.7 we would need to use the backported version
    import importlib_resources as pkg_resources


from line_profiler import profile


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
    
def get_folding_pars(psr):
    """
    Return ra and dec for a pulsar from SPS database    
    psr: string, pulsar B or J name    
    Returns: ra, dec
    """
    dbpsr = db_api.get_known_source_by_names(psr)[0]
    ra = dbpsr.pos_ra_deg
    dec = dbpsr.pos_dec_deg    
    return ra, dec

def gaussian_model(x, a, mu, sigma):
    """
    Define the Gaussian function for fitting noise distribution
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def lorentzian(x, x0, gamma, a):
    """
    Define the Lorentzian function for fitting periodogram peaks
    """
    return a * (gamma**2 / ((x - x0)**2 + gamma**2))


@profile
def remove_baseline(pgram, b0=50, bmax=1000):
    """
    Remove the rising baseline from a raw periodogram. To do this, split up the periodogram into
    logarithmically increasing sections, from b0 up to bmax
    """
    pgram_len = len(pgram.periods)
    sections = [0]
    section_end = 0
    for n in range(0, pgram_len):
        new_window = np.exp(1 + n / 3) * b0 / np.exp(1)
        if new_window > bmax:
            section_end += bmax
        else:
            section_end += int(new_window)
        sections.append(section_end)
        if section_end > pgram_len:
            sections[-1] = pgram_len
            break

    for iw, width in enumerate(pgram.widths):
        s = pgram.snrs[:, iw].astype(float)

        for i in range(len(sections)-1):
            s[sections[i]:sections[i+1]] /= np.median(s[sections[i]:sections[i+1]])
        pgram.snrs[:, iw] = s

    return pgram

# def remove_baseline(pgram, b0=50, bmax=2000):
#     """
#     Remove the rising baseline from a raw periodogram using a piecewise linear interpolation
#     between the median of logarithmically increasing sections.
#     """
#     pgram_len = len(pgram.periods)
#     sections = [0]
#     section_centers = []
#     section_end = 0
    
#     for n in range(0, pgram_len):
#         new_window = np.exp(1 + n / 3) * b0 / np.exp(1)
#         if new_window > bmax:
#             section_end += bmax
#         else:
#             section_end += int(new_window)
        
#         sections.append(section_end)
#         if section_end > pgram_len:
#             sections[-1] = pgram_len
#             break
    
#     # Compute section centers
#     section_centers = [(sections[i] + sections[i+1]) // 2 for i in range(len(sections) - 1)]

#     # Compute medians for each section
#     medians = []
#     for i in range(len(sections) - 1):
#         section_data = pgram.snrs.max(axis=1)[sections[i]:sections[i + 1]]
#         local_median = np.partition(section_data, len(section_data) // 2)[len(section_data) // 2]
#         medians.append(local_median)
    
#     # Create interpolation function
#     interp_values = np.interp(np.arange(pgram_len), section_centers, medians, left=medians[0], right=medians[-1])
    
#     for iw, width in enumerate(pgram.widths):
#         pgram.snrs[:, iw] /= interp_values
    
#     return pgram


@profile
def standardize_pgram(pgram, dm):
    """
    Standardize the noise distribution of a periodogram such that it is a Gaussian with mean=0, stdev=1
    """
    snr_distrib = np.histogram(pgram.snrs.max(axis=1), bins=500)
    try:
        popt, _ = curve_fit(
            gaussian_model, 
            snr_distrib[1][:-1], 
            snr_distrib[0],
            p0=[3000,1,0.1],
            bounds=([0,0,0],[np.inf,np.inf,np.inf]),
            method="trf"
        )
    except (RuntimeError, OptimizeWarning) as e:
        # Make a rough guess of the noise properties based off what we'd expect empirically
        log.error(f"Couldn't properly standardize the periodogram at DM {dm}")
        popt = [1.0,0.1]
    # Subtract the mean, divide by stdev
    pgram.snrs -= popt[1]
    pgram.snrs /= popt[2]
    
    return pgram

@profile
def barycentric_shift(pgram, shifted_freqs, shift_min, shift_max):
    """
    Shifts a standardized periodogram's snrs such that the frequency is barycentric rather than topocentric.
    Uses nearest interpolation to shift snrs to the corresponding shifted frequency range.
    """
    freqs_subset = pgram.freqs[shift_min:shift_max+1]  # Extract frequency range once

    for iw in range(len(pgram.widths)):
        s = pgram.snrs[:, iw].astype(float)
        
        # Use fast 'nearest' interpolation
        f = interpolate.interp1d(shifted_freqs, s, kind="nearest", bounds_error=False, fill_value=0)
        
        # Directly assign interpolated values
        pgram.snrs[shift_min:shift_max+1, iw] = f(freqs_subset)

    return pgram

@profile
def rogue_width_filter(peaks, snrs, widths):
    """
    Remove peaks whose trial width bin is much stronger than the others
    This is satisfied if:
    - The width bin of the peak is one of the largest 2 (for our normal width bins, this is width=6 and 9)
    - The maximum sigma is > 2*the sigma of all other width bins (should not happen for real signals as we don't expect them to be highly sinusoidal)
    """
    width_limit = widths[-2]
    filtered_peaks = [
        peak for peak in peaks
        if not (
            peak.width >= width_limit 
            and snrs[peak.ip][peak.iw] > 2 * (sum(snrs[peak.ip]) - snrs[peak.ip][peak.iw])
        )
    ]

    return filtered_peaks

@profile
def remove_duplicates(peaks):
    """
    Remove peaks with duplicate periods such that only the strongest sigma remains
    """
    unique_periods = set()
    filtered_peaks = []
    
    for peak in peaks:
        # Only add the first occurrence of each period (these are already sorted in riptide by decreasing sigma)
        if peak.period not in unique_periods:
            unique_periods.add(peak.period)
            filtered_peaks.append(peak)
    
    return filtered_peaks

##########################################################
#####################                  ###################
##################### PERIODOGRAM_FORM ###################
#####################                  ###################
##########################################################
    
@profile
def periodogram_form(
        tseries_np,
        dm,
        birdies, 
        shifted_freqs, 
        shift_min, 
        shift_max, 
        period_min, 
        period_max,
        min_rfi_sigma=5,
        min_detection_sigma=7,
        birdie_tolerance=0.005,
        bins_min=190, 
        bins_max=210, 
        rmed_width=4.0, 
        ducy_max=0.05,
        num_birdie_harmonics=1
    ):
    """
    Forms a periodogram from a dedispersed time series
    Input:
        tseries_np: Numpy ND array of a dedispersed time series, an element from dedisp_ts.dedisp_ts
        dm: Dispersion measure of the tseries
        birdies: 1D array containing all the periods of birdies to filter out
        shifted_freqs: Numpy 1D array, the frequency array in barycentric frame
        shift_min: the first index of pgram.freqs which is smaller than shifted_freqs
        shift_max: the last index of pgram.freqs which is larger than shifted_freqs
        period_min: minimum period to search for
        period_max: maximum period to search for
        bins_min: minimum number of period bins to fold at
        bins_max: maximum number of period bins to fold at
        rmed_width: running median width (see riptide docs)
        ducy_max: maximum duty cycle to search for
        num_birdie_harmonics: Number of harmonics of birdie to zap. Includes the fundamental! 
            (ie 1 only searches the fundamental, 0 will not remove birdies at all)

    Output:
        ts: a riptide.TimeSeries object, with the time series that was actually searched (after initial downsampling and red noise removal)
        pgram: a riptide.Periodogram object. The frequencies/periods will be shifted to barycentric frame, and the noise distribution will 
            be standardized such that mean=0 and stdev=1. Birdie frequencies are removed according to specified parameters
        peaks: an array of riptide.Peak objects, sorted by decreasing snr
        For more information of each of these, see https://riptide-ffa.readthedocs.io/en/latest/reference.html
    """
    tseries = riptide.TimeSeries.from_numpy_array(tseries_np, TSAMP)

    # See https://riptide-ffa.readthedocs.io/en/latest/reference.html for full explanation of these parameters
    ts, pgram = riptide.ffa_search(
        tseries, 
        period_min=period_min, 
        period_max=period_max, 
        bins_min=bins_min, 
        bins_max=bins_max, 
        rmed_width=rmed_width, 
        ducy_max=ducy_max
    )

    # For low dms, the baseline increases with trial period, so we need to remove it
    pgram = remove_baseline(pgram)

    # Normalize the noise distribution such that mean=0 and stddev=1
    pgram = standardize_pgram(pgram, dm)

    # Correct for barycentric frame. This involves shifting the entire periodogram by a certain factor and interpolating extra points
    pgram = barycentric_shift(pgram, shifted_freqs, shift_min, shift_max)

    # Find every point above sigma min_detection_sigma (default 7) and cluster them into Peak objects
    peaks, _ = riptide.find_peaks(pgram, smin = min_detection_sigma, nstd = 0)

    # Remove peaks with duplicate periods, only keep the strongest
    peaks = remove_duplicates(peaks)
   
    # Remove peaks at periods where one width is stronger than the others
    peaks = rogue_width_filter(peaks, pgram.snrs, pgram.widths)
    
    pgram_len = len(pgram.snrs)

    # Zap peaks at periods within the tolerance range of a known birdie or its first num_birdie_harmonics harmonics
    peaks_filter = []
    for peak in peaks:
        cleaned = False
        for birdie in birdies:
            for harmonic_order in range(1,num_birdie_harmonics+1):
                harmonic_period = birdie*harmonic_order
                if abs(peak.period - harmonic_period) < birdie_tolerance:
                    period_index = peak.ip
                    # Find the width of the RFI peak in indexes
                    peak_width = 0
                    while (pgram.snrs[max(0,period_index-peak_width)][peak.iw] > min_rfi_sigma) and (pgram.snrs[min(pgram_len-1,period_index+peak_width)][peak.iw] > min_rfi_sigma):
                        peak_width += 1
                    for index in range(max(0,period_index-peak_width), min(pgram_len,period_index+peak_width)):
                        pass
                        # pgram.snrs[index][peak.iw] = 0
                    cleaned = True
                    break
            if cleaned:
                break
        peaks_filter.append(not cleaned)
    # Remove all the peaks that got filtered
    peaks_filter = np.array(peaks_filter, dtype=bool)
    peaks = [peak for peak, keep in zip(peaks, peaks_filter) if keep]


    del tseries
                    
    return ts, pgram, peaks


def periodogram_form_dm_0(
        tseries_np,
        period_min, 
        period_max,
        min_rfi_sigma=5,
        bins_min=190, 
        bins_max=210, 
        rmed_width=4.0, 
        ducy_max=0.05,
    ):
    """
    Finds RFI peaks from a dedispersed time series at DM 0
    Input:
        tseries_np: Numpy ND array of a dedispersed time series, an element from dedisp_ts.dedisp_ts
        period_min: minimum period to search for
        period_max: maximum period to search for
        bins_min: minimum number of period bins to fold at
        bins_max: maximum number of period bins to fold at
        rmed_width: running median width (see riptide docs)
        ducy_max: maximum duty cycle to search for

    Output:
        peaks: an array of riptide.Peak objects, sorted by decreasing snr
        For more information of each of these, see https://riptide-ffa.readthedocs.io/en/latest/reference.html
    """
    tseries = riptide.TimeSeries.from_numpy_array(tseries_np, TSAMP)

    # See https://riptide-ffa.readthedocs.io/en/latest/reference.html for full explanation of these parameters
    ts, pgram = riptide.ffa_search(
        tseries, 
        period_min=period_min, 
        period_max=period_max, 
        bins_min=bins_min, 
        bins_max=bins_max, 
        rmed_width=rmed_width, 
        ducy_max=ducy_max
    )

    # For low dms, the baseline increases with trial period, so we need to remove it
    pgram = remove_baseline(pgram)

    # Find every point above sigma min_rfi_sigma (default 5) and cluster them into Peak objects
    peaks, _ = riptide.find_peaks(pgram, smin = min_rfi_sigma, nstd = 0)
    
    # Remove peaks with duplicate periods, only keep the strongest
    peaks = remove_duplicates(peaks)

    del tseries
                    
    return ts, pgram, peaks




def find_area_around_peak(sigmas, peak_index, smin):
    """
    Returns minimum and maximum index around the peak_index whose sigma is higher than smin
    """
    min_index = peak_index
    while min_index > 0 and sigmas[min_index-1] > smin:
        min_index -= 1
    
    max_index = peak_index
    while max_index < len(sigmas)-1 and sigmas[max_index+1] > smin:
        max_index += 1

    return min_index, max_index


def rate_periodogram(peaks, dm, birdies):
    """
    Gives a rating for a periodogram, taking into account the dm, the strength of the main peak and its harmonics
    Mainly used for finding the best candidate DM of an observation
    """
    if len(peaks) == 0 or dm < 2:
        return 0
    main_peak = peaks[0]

    tolerance = 0.01
    # Zap if main peak is one of the first 10 harmonics of a birdie
    for birdie in birdies:
        for harmonic_order in range(1,11):
            harmonic_period = birdie*harmonic_order
            if abs(main_peak.period - harmonic_period)/harmonic_period < tolerance:
                return 0
        
    
    # Add first 10 harmonics of the main peak with a weight
    tolerance = 0.01
    rating = main_peak.snr
    for harmonic_order in range(2,10):
        harmonic_period = main_peak.period * harmonic_order
        harmonic_peaks = list(filter(lambda peak: abs(peak.period-harmonic_period)/harmonic_period < tolerance, peaks))
        if harmonic_peaks:
            rating += harmonic_peaks[0].snr / 2
    return rating






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
@click.option(
    "--plot/--no-plot",
    default=False,
    help="Whether or not to generate periodogram plots and pulse profile at best DM"
)
@click.option(
    "--generate_candidates/--no_candidates",
    "-c",
    default=False, 
    help="Whether or not to generate candidates for the given observation")
@click.option(
    "--stack/--no-stack",
    "-s",
    default=False,
    help="Whether to stack the periodograms"
)
@click.option("--ra", type=click.FloatRange(-180, 360), default=None)
@click.option("--dec", type=click.FloatRange(-90, 90), default=None)
@click.option(
    "--dm_downsample", 
    type=int, 
    default=8, 
    help="The downsampling factor in DM trials. Must be an integer >= 1. Default is 8"
)
@click.option(
    "--dm_min", 
    default=0, 
    help="Only search for DMs above this value. Default is 0"
)
@click.option(
    "--dm_max", 
    type=float, 
    default=None, 
    help="Only search for DMs smaller than this value. Default is None"
)
@click.option("--num_threads", type=int, default=16, help="Number of threads to use")
def main(
    psr, 
    date, 
    plot, 
    generate_candidates, 
    stack, 
    ra, 
    dec, 
    dm_downsample, 
    dm_min, 
    dm_max, 
    num_threads
):
    """
    Simulates part of the pipeline in order to generate dedispersed time series. For testing purposes mostly
    Example:
    FFA_search.py --psr B0154+61 --date 20240612 --stack --generate_candidates
    Input: See click options
    Output: If specified, creates candidates, plots, and/or writes to the periodogram stacks. Nothing returned directly
    """
    
    apply_logging_config('INFO')
    db_utils.connect(host = 'sps-archiver1', port = 27017, name = 'sps-processing')

    datetime.date = utils.convert_date_to_datetime(date)
    date_string = date.strftime("%Y/%m/%d")
    
    config_file = '/home/ltarabout/FFA/sps_config_ffa.yml'
    config = OmegaConf.load(config_file)
    if psr is not None:
        ra, dec = get_folding_pars(psr)
    elif ra is None or dec is None:
        log.error("Please provide either a PSR or a position in (ra,dec)")
        return
    if plot:
        generate_candidates = True
    
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
        basepath = "/mnt/beegfs-client/raw/",
        # basepath="/data/chime/sps/raw/",
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

    FFA_search(
        dedisp_ts,
        obs_id,
        date,
        ra,
        dec,
        True,
        False,
        plot, 
        False,
        generate_candidates, 
        stack,
        dm_downsample, 
        dm_min, 
        dm_max, 
        2,
        5,
        7,
        0.005,
        num_threads,
        "./"
    )

    del dedisp_ts
    del skybeam, spectra_shared, sbf, pst, ap

    return


##########################################################
#####################                  ###################
#####################    FFA_SEARCH    ###################
#####################                  ###################
##########################################################

@profile
def FFA_search(
    dedisp_ts,
    obs_id,
    date,
    ra,
    dec,
    run_ffa_search=True,
    create_periodogram_plots=False,
    create_profile_plots=False,
    write_ffa_detections=False,
    write_ffa_candidates=True,
    write_ffa_stack=True,
    dm_downsample=8, 
    dm_min=0, 
    dm_max=None,
    p_min=2,
    min_rfi_sigma=5,
    min_detection_sigma=10,
    birdie_tolerance=0.005,
    num_threads=16,
    basepath="/data2/chime/sps/sps_processing"
):
    """
    Performs the FFA search on a set of dedispersed time series for a single pointing.
    Input:
        dedisp_ts: The array of dedispersed time series from the FDMT. Should include dedisp_ts.dedisp_ts AND dedisp_ts.dms
        obs_id: Observation id of the pointing
        date: datetime object corresponding to the observation
        ra: Right Ascension of the observation
        dec: Declination of the observation
        run_ffa_search: boolean. This is just part of the config file, should always be True in this function.
        create_periodogram_plots: boolean. Whether or not to plot the periodogram at the best DM trial. 
            These are saved at "/data2/chime/sps/sps_processing/{date.year}/{date.month}/{date.day}/{np.round(ra,2)}_{np.round(dec,2)}/plots/FFA_periodogram_ra_{np.round(ra,2)}_dec_{np.round(dec,2)}_snr_{np.round(best_peak.snr,2)}_dm_{np.round(best_dm,3)}.png"
        create_profile_plots: boolean. Whether or not to plot the folded pulse profile at the best DM and period trial.
            These are saved at "/data2/chime/sps/sps_processing/{date.year}/{date.month}/{date.day}/{np.round(ra,2)}_{np.round(dec,2)}/plots/FFA_profile_ra_{np.round(ra,2)}_dec_{np.round(dec,2)}_snr_{np.round(best_peak.snr,2)}_p_{np.round(best_peak.period,6)}_dm_{np.round(best_dm,3)}.png"
        write_ffa_detections: boolean. Whether or not to write the detections (peaks of the periodograms) in a separate file.
            These are saved at "/data2/chime/sps/sps_processing/detections/{date.year}/{date.month}/{date.day}/{np.round(ra,2)}_{np.round(dec,2)}_FFA_detections.npz"
        write_ffa_candidates: boolean. Whether or not to write the single pointing candidates in a separate file
            These are saved at "/data2/chime/sps/sps_processing/{date.year}/{date.month}/{date.day}/{np.round(ra,2)}_{np.round(dec,2)}/{np.round(ra,2)}_{np.round(dec,2)}_FFA_candidates.npz"
        write_ffa_stack: boolean. Whether or not to write to the stacks.
            These are saved at "/data2/chime/sps/sps_processing/stack_FFA/{np.round(ra,2)}_{np.round(dec,2)}_FFA_stack.hdf5"
        dm_downsample: The downsampling factor in DM trials. Must be an integer >= 1. Recommended value is 8
        dm_min: Only search for DMs above this value. Warning, currently the code assumes that the first DM trial is at DM=0
        dm_max: Only search for DMs below this value.
        min_rfi_sigma: float. Birdies are classified as any peak in the periodogram at DM 0 above min_rfi_sigma
        min_detection_sigma: float. Detections are classified as any peak in any periodogram above min_detection_sigma
        num_threads: Number of cores available. Recommended value is 16
    Output:
        None. All data products will be written to disk as described above.
    """
    total_start_time = time.time()
    
    # Takes one of every dm_downsample DM trials from initial array
    ts = dedisp_ts.dedisp_ts[::dm_downsample]
    dms = dedisp_ts.dms[::dm_downsample]
    if dm_max is None:
        dm_max = dms[-1]
    # Remove ts and dms outside the specified range
    mask = (dms >= dm_min) & (dms <= dm_max)
    ts = ts[mask]
    dms = dms[mask]
    n_dm_trials = len(dms)
    
    mjd_time = Time(date).mjd
    date_string = date.strftime("%Y/%m/%d")

    tobs = len(ts[0])*TSAMP
    # Compute barycentric shift
    barycentric_beta = barycenter.get_barycentric_correction(str(ra),str(dec),mjd_time,tobs)
    log.info(f"Tobs: {tobs}, beta: {barycentric_beta}")

    time_start = time.time()

    # Load birdies from the yaml file. Transform it to period birdies instead
    with pkg_resources.path(rfi_mitigation_data, "birdies.yaml") as p:
        # the variable `p` is a PosixPath object, so we still need to open it
        log.debug(f"Accessing known birdies list from {str(p)}")
        known_birdies = yaml.load(
            open(p),
            Loader=yaml.FullLoader,
        )
    birdies = []
    for k, bird in known_birdies["known_bad_frequencies"].items():
        if barycentric_beta != 0:
            birdies.append(1/barycenter.bary_from_topo_freq(bird["frequency"], barycentric_beta))
        else:
            birdies.append(1/bird["frequency"])

    # Compute pgram at DM 0 to get additional birdies. A birdie is any peak above min_rfi_sigma snr (default 5)
    ts_0, pgram_0, peaks_0 = periodogram_form_dm_0(ts[0], p_min, tobs/8, min_rfi_sigma)
    for peak in peaks_0:
        if barycentric_beta != 0:
            birdies.append(1/barycenter.bary_from_topo_freq(peak.freq, barycentric_beta))
        else:
            birdies.append(peak.period)
    log.info(f"Birdies: {birdies}")
            
    # Use radio convention to shift the frequencies/periods
    shifted_freqs = pgram_0.freqs * (1-barycentric_beta)
    shift_min = np.where(pgram_0.freqs <= shifted_freqs[0])[0][0]
    shift_max = np.where(pgram_0.freqs >= shifted_freqs[-1])[0][-1]
    
    # Call all other DM results using MP. This is the main processing loop of the FFA search
    args = [(
        ts[ts_index], 
        dms[ts_index], 
        birdies, 
        shifted_freqs, 
        shift_min, 
        shift_max,
        p_min,
        tobs/8,
        min_rfi_sigma,
        min_detection_sigma
    ) for ts_index in range(1,n_dm_trials)]

    ts_test, pgram_test, peaks_test = periodogram_form(ts[1],dms[1],birdies,shifted_freqs,shift_min, shift_max, p_min, tobs/8, min_rfi_sigma, min_detection_sigma)
    
    with multiprocessing.Pool(processes=num_threads) as pool:
            results = pool.starmap(periodogram_form,args)
    ts_array, pgram_array, peaks_array = zip(*results)
    dms = dms[1:]
    n_dm_trials -= 1

    # ts_array = np.concatenate(([ts_0], ts_array))
    # pgram_array = np.concatenate(([pgram_0], pgram_array))
    # peaks_array = [peaks_0] + list(peaks_array)

    if create_periodogram_plots or create_profile_plots:
        best_dm_trial = 0
        best_rating = 0
        # Arrays for storing the fit parameters and errors
        for dm_trial in range(n_dm_trials):
            # Rate each periodogram
            rating = rate_periodogram(peaks_array[dm_trial],dms[dm_trial],birdies)
            if rating > best_rating:
                best_rating = rating
                best_dm_trial = dm_trial
    
        # Find the best fit and error for the main peak, if there is one
        # if len(peaks_array[best_dm_trial]) > 0:
        #     main_peak = peaks_array[best_dm_trial][0]
        #     pgram = pgram_array[best_dm_trial]
        #     mask = (pgram.periods >= main_peak.period-0.01) & (pgram.periods <= main_peak.period+0.01)
        #     periods_section = pgram.periods[mask]
        #     snrs_section = pgram.snrs.max(axis=1)[mask]
            
        #     # Fit the Lorentzian curve to the selected data if possible
        #     try:
        #         popt, pcov = curve_fit(lorentzian, periods_section, snrs_section, p0=[main_peak.period, 0.01, main_peak.snr])
        #         perr = np.sqrt(np.diag(pcov))
        #     except (RuntimeError, OptimizeWarning) as e:
        #         popt = None
        #         perr = None
        # else:
        #     popt = None
        #     perr = None
        
        best_pgram = pgram_array[best_dm_trial]
        if len(peaks_array[best_dm_trial]) > 0:
            best_peak = peaks_array[best_dm_trial][0]
            best_ts = ts_array[best_dm_trial]
            best_dm = dms[best_dm_trial]
            log.info(
                f"Best DM: {best_dm}"
                f"Best peak: period: {best_peak.period : .2f}, freq: {best_peak.freq : .2f}"
                f"           snr: {best_peak.snr : .2f}, iw: {best_peak.iw}"
            )
        else:
            best_peak = None
            log.warning("Best dm trial has no peaks!")

    total_time = time.time()-time_start
    log.info("Finished making periodograms for all DM trials")
    log.info(f"Time: {total_time} s")

    detections = []
    candidates = []
    if write_ffa_candidates or write_ffa_detections:           
        detections = np.array([(
            dms[dm_trial],
            1/peak.period,
            peak.snr,
            peak.width
        ) for dm_trial in range(n_dm_trials) for peak in peaks_array[dm_trial]], dtype = detections_dtype)

        log.info(f"Periodogram resulted in {len(detections)} raw detections")

        if len(detections) > 0:
            print("Detections:")
            print(detections)
            clusterer = Clusterer_FFA()
            cluster_dm_spacing = dms[1]-dms[0]
            cluster_df_spacing = pgram_0.freqs[0]-pgram_0.freqs[1]
            print(f"cluster_dm_spacing: {cluster_dm_spacing}, cluster_df_spacing: {cluster_df_spacing}")
    
            clusterer.cluster(detections,cluster_dm_spacing,cluster_df_spacing)
    
            (
                clusters,
                summary,
                clustering_sigma_min,
                used_detections_len,
            ) = clusterer.make_clusters(
                detections,
                cluster_dm_spacing,
                cluster_df_spacing,
                plot_fname=""
            )
    
            for cluster in clusters.values():
                # The candidate is saved with a map showing sigma & width as a function of dm and frequency
                # This allows us to make the candidate plots later
                # The DM range looks at 10 units around the edges of the cluster
                min_dm_index = np.abs(dms - (min(cluster.unique_dms)-10)).argmin()
                max_dm_index = np.abs(dms - (max(cluster.unique_dms)+10)).argmin()
                dm_range = np.array(dms[min_dm_index:max_dm_index])
    
                # The frequency range looks at 0.001 Hz around the edges of the cluster
                max_p_index = np.abs(pgram_0.periods - 1/(min(cluster.unique_freqs)-0.001)).argmin()
                min_p_index = np.abs(pgram_0.periods - 1/(max(cluster.unique_freqs)+0.001)).argmin()
                freq_range = np.array([1/pgram_0.periods[period_index] for period_index in range(max_p_index,min_p_index,-1)])
    
                dm_freq_sigma = np.array([[
                    pgram_array[dm_index].snrs[period_index].max()
                        for period_index in range(max_p_index,min_p_index,-1)] 
                        for dm_index in range(min_dm_index,max_dm_index)]
                )
    
                dm_freq_widths = np.array([[
                    pgram_0.widths[np.argmax(pgram_array[dm_index].snrs[period_index])]
                        for period_index in range(max_p_index,min_p_index,-1)] 
                        for dm_index in range(min_dm_index,max_dm_index)]
                )
    
                # If sigma is high enough, make a small plot of the above maps
                if cluster.sigma > 10:
                    fig, axs = plt.subplots(1, 5, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.5]})
                    extent = [freq_range.min(), freq_range.max(), dm_range.min(), dm_range.max()]
                    
                    axs[0].imshow(dm_freq_sigma, aspect='auto', cmap='viridis', origin='lower',
                               extent=extent, interpolation="none")
                    
                    axs[0].set_xlabel('Frequency (Hz)')
                    axs[0].set_ylabel('DM')
                    axs[0].set_title('Sigma')
                
                    axs[1].imshow(dm_freq_widths, aspect='auto', cmap='Oranges', origin='lower',
                               extent=extent, interpolation="none")
                    
                    axs[1].set_xlabel('Frequency (Hz)')
                    axs[1].set_ylabel('DM')
                    axs[1].set_title('Best Width')
                
                    axs[2].plot(dm_range,dm_freq_sigma.max(axis=1),color="black")
                    axs[2].set_xlabel('DM')
                    axs[2].set_ylabel('Sigma')
                
                    axs[3].plot(freq_range,dm_freq_sigma.max(axis=0),color="black")
                    axs[3].set_xlabel('Frequency (Hz)')
                    axs[3].set_ylabel('Sigma')
                
                    text_info = f"ra:{ra}\ndec:{dec}\ndate:{date_string}\nsigma:{cluster.sigma}\nfreq:{cluster.freq}\ndm:{cluster.dm}"
                    axs[4].text(0, 0.5, text_info, fontsize=12, verticalalignment='center', horizontalalignment='left')
                    axs[4].axis('off')
                
                    os.makedirs(f"{basepath}/{date_string}/{np.round(ra,2)}_{np.round(dec,2)}/plots/", exist_ok=True)
                    plot_path = f"{basepath}/{date_string}/{np.round(ra,2)}_{np.round(dec,2)}/plots/FFA_cand_{np.round(ra,2)}_{np.round(dec,2)}_sigma_{np.round(cluster.sigma,2)}_freq_{np.round(cluster.freq,4)}_dm_{np.round(cluster.dm,3)}.png"
    
                    plt.rcParams["savefig.dpi"] = 100  # Lower resolution
                    plt.rcParams["savefig.pad_inches"] = 0.05  # Minimize extra space
                    fig.tight_layout()
                    plt.savefig(plot_path, bbox_inches=None, format="png", transparent=False)
                    plt.close(fig)
                else:
                    plot_path = ""
                
    
                candidates.append({
                    "freq": cluster.freq,
                    "dm": cluster.dm,
                    "sigma": cluster.sigma,
                    #"features": ,
                    "rfi": False,
                    "max_sig_det": cluster.max_sig_det,
                    "ndetections": len(cluster.detections),
                    "detections": cluster.detections,
                    "unique_freqs": cluster.unique_freqs,
                    "unique_dms": cluster.unique_dms,
                    "num_unique_freqs": cluster.num_unique_freqs,
                    "num_unique_dms": cluster.num_unique_dms,
                    "obs_id": obs_id,
                    "dm_freq_sigma": dm_freq_sigma,
                    "dm_freq_widths": dm_freq_widths,
                    "dm_range": dm_range,
                    "freq_range": freq_range,
                    "plot_path": plot_path,
                })

        # candidates = np.array([(
        #     {
        #         "freq": cluster.freq,
        #         "dm": cluster.dm,
        #         "sigma": cluster.sigma,
        #         #"features": ,
        #         "rfi": False,
        #         "max_sig_det": cluster.max_sig_det,
        #         "ndetections": len(cluster.detections),
        #         "detections": cluster.detections,
        #         "unique_freqs": cluster.unique_freqs,
        #         "unique_dms": cluster.unique_dms,
        #         "num_unique_freqs": cluster.num_unique_freqs,
        #         "num_unique_dms": cluster.num_unique_dms,
        #         "obs_id": obs_id,
        #         "dm_freq_sigma": [],
        #     }
        # ) for cluster in clusters.values()])

        log.info("Finished clustering detections into single pointing candidates")
                

    ### WRITE SECTION ###

    start_time = time.time()

    directory_name = basepath
    # directory_name = f"/scratch/ltarabout/ffa_pulsar_list/stack_{np.round(ra,2)}_{np.round(dec,2)}"
    os.makedirs(directory_name, exist_ok=True)
    
    if write_ffa_detections and len(detections) > 0:
        # Save the detections in the appropriate files
        detection_directory_name = os.path.join(directory_name, f"detections_FFA/{date.year}/{date.month}/{date.day}")
        os.makedirs(detection_directory_name, exist_ok=True)
        file_path = os.path.join(detection_directory_name, f"{np.round(ra,2)}_{np.round(dec,2)}_FFA_detections.npz")
        np.savez(file_path, detections=detections, allow_pickle=True)
            
        log.info(f"Saved a new detections file at {file_path}")

    if write_ffa_candidates and len(candidates) > 0:
        # Save the candidates in the appropriate files
        # In reality, full path will be /data/chime/sps/sps_processing/FFA/{year}/{month}/{day}/{ra}_{dec}_FFA_candidates.npz
        candidate_directory_name = os.path.join(directory_name, f"{date_string}/{np.round(ra,2)}_{np.round(dec,2)}")
        os.makedirs(candidate_directory_name, exist_ok=True)
        file_path = os.path.join(candidate_directory_name, f"{np.round(ra,2)}_{np.round(dec,2)}_FFA_candidates.npz")
        np.savez(file_path, candidates=candidates, allow_pickle=True)

        # Also append the candidate to the list of candidates for that day, for the Candidate Visualizer
        csv_candidates = [
            {
                "": i,
                "mean_freq": cand['freq'],
                "mean_dm": cand['dm'],
                "sigma": cand['sigma'],
                "ra": ra,
                "dec": dec,
                "best_ra": ra,
                "best_dec": dec,
                "ncands": cand['ndetections'],
                "std_ra": 0,
                "std_dec": 0,
                "delta_ra": 0,
                "delta_dec": 0,
                "file_name": file_path,
                "plot_path": cand['plot_path'],
                "known_source_label": 0,
                "known_source_likelihood": "",
                "known_source_name": "",
                "known_source_p0": "",
                "known_source_dm": "",
                "known_source_ra": "",
                "known_source_dec": ""
            } for i, cand in enumerate(candidates)
        ]

        fieldnames = ["","mean_freq","mean_dm","sigma","ra","dec","best_ra","best_dec","ncands","std_ra","std_dec","delta_ra","delta_dec","file_name","plot_path","known_source_label","known_source_likelihood","known_source_name","known_source_p0","known_source_dm","known_source_ra","known_source_dec"]

        csv_path = os.path.join(directory_name, f"{date_string}/all_sp_ffa_cands.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerows(csv_candidates)  # Write candidate data
        else:
            with open(csv_path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()  # Write column headers
                writer.writerows(csv_candidates)  # Write candidate data
            
        log.info(f"Saved a new single pointing candidates file at {file_path}")
    
    if create_periodogram_plots:
        plot_directory_name = os.path.join(directory_name, f"{date_string}/{np.round(ra,2)}_{np.round(dec,2)}/plots")
        os.makedirs(plot_directory_name, exist_ok=True)
        file_path = os.path.join(plot_directory_name, f"FFA_periodogram_ra_{np.round(ra,2)}_dec_{np.round(dec,2)}_snr_{np.round(best_peak.snr,2)}_dm_{np.round(best_dm,3)}.png")

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(best_pgram.periods, best_pgram.snrs.max(axis=1),linewidth=0.3, marker='o', markersize=0.7)
        plt.title("Best Sigma at any trial width")
        plt.xlim(min(best_pgram.periods),max(best_pgram.periods))
        plt.xlabel("Trial Period (s)")
        plt.ylabel("Sigma")
        plt.grid(True)
        
        plt.savefig(file_path)
        log.info(f"Saved best periodogram at {file_path}")
        plt.clf()
    
    if create_profile_plots:   
        plot_directory_name = os.path.join(directory_name, f"{date_string}/{np.round(ra,2)}_{np.round(dec,2)}/plots")
        os.makedirs(plot_directory_name, exist_ok=True)
        file_path = os.path.join(plot_directory_name, f"FFA_profile_ra_{np.round(ra,2)}_dec_{np.round(dec,2)}_snr_{np.round(best_peak.snr,2)}_p_{np.round(best_peak.period,6)}_dm_{np.round(best_dm,3)}.png")
        
        bins = 256
        subints = best_ts.fold((1-barycentric_beta)*best_peak.period, bins)
    
        plt.subplot(211)
        plt.title(f"Folded profile at p={np.round(best_peak.period,4)}s, dm={np.round(best_dm,3)}, {date_string}")
        plt.imshow(subints, cmap='Greys', aspect='auto')
        plt.xlim(0,255)
        plt.ylabel("Subints")
        plt.subplot(212)
        plt.plot(subints.sum(axis=0))
        plt.xlabel("Bins")
        plt.ylabel("Intensity")
        plt.xlim(0,255)

        plt.savefig(file_path)
        log.info(f"Saved best folded pulse profile at {file_path}")
        plt.clf()


    if write_ffa_stack:
        stack_directory_name = os.path.join(directory_name, "stack_FFA")
        os.makedirs(stack_directory_name, exist_ok=True)
        
        stack_name = f"{np.round(ra,2)}_{np.round(dec,2)}_FFA_stack.hdf5"
        file_path = os.path.join(stack_directory_name, stack_name)
        # Write/add to the stack files
        log.info("Writing to stack")
        
        if os.path.exists(file_path):
            with h5py.File(file_path, 'a') as hf:
                if np.array_equal(dms, hf['dms']) and np.array_equal(pgram_array[0].periods, hf['periods']):
                    stack_length = hf.attrs['stack_length']
                    
                    # Create an empty array to store all updated SNRs
                    snrs = np.empty_like(hf['snrs'])  # Match shape of original dataset
                    
                    for dm_index in range(n_dm_trials):
                        stack_snrs = hf['snrs'][dm_index][:]
        
                        # Rescale down then back up
                        stack_snrs = (stack_snrs + (pgram_array[dm_index].snrs / stack_length))
                        
                        # Standardize noise distribution
                        snr_distrib = np.histogram(stack_snrs.max(axis=1), bins=500)
                        popt, _ = curve_fit(
                            gaussian_model, 
                            snr_distrib[1][:-1], 
                            snr_distrib[0],
                            p0=[3000,1,0.1],
                            bounds=([0,0,0],[np.inf,np.inf,np.inf]),
                            method="trf"
                        )
                        stack_snrs -= popt[1]
                        stack_snrs /= popt[2]
        
                        # Assign updated data to `snrs`
                        snrs[dm_index] = stack_snrs
        
                    # Write updated SNRs back to the file
                    hf['snrs'][...] = snrs
                    hf.attrs['stack_length'] = stack_length + 1
                    log.info(f"Appended the stack file at {file_path}")
                else:
                    log.error("Cannot add data to stack as the DM range or period range is different!")
        
        else:
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('snrs', data=[pgram_array[dm_index].snrs for dm_index in range(n_dm_trials)])
                hf.attrs['stack_length'] = 1
                hf.attrs['ra'] = ra
                hf.attrs['dec'] = dec
                hf.create_dataset('dms', data=dms)
                hf.attrs["widths"] = pgram_array[0].widths
                hf.create_dataset("periods", data=pgram_array[0].periods)
                hf.create_dataset("foldbins", data=pgram_array[0].foldbins)
                log.info(f"Created a stack file at {file_path}")


    log.info(f"FFA Write time: {time.time()-start_time}")
    log.info(f"Total FFA time: {time.time()-total_start_time}")

    del pgram_array
    del peaks_array
    del ts_array
    del detections
    del args, results
    del ts, ts_0, pgram_0, peaks_0, shifted_freqs
    
    return

log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)# import in this way for easy reload

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)
    main()
    