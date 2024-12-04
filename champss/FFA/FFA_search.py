import numpy as np
from matplotlib import pyplot as plt
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
import datetime
import json
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
from single_pointing_FFA import SinglePointingCandidate_FFA, SearchAlgorithm_FFA

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

def remove_baseline(pgram, b0=50, bmax=1000):
    """
    Remove the rising baseline from a raw periodogram. To do this, split up the periodogram into
    logarithmically increasing sections, from b0 up to bmax
    """
    pgram_len = len(pgram.periods)
    sections = [0]
    section_end = 0
    for n in range(0, pgram_len):
        # create the log range for normalisation
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

    pgram = standardize_pgram(pgram)
    return pgram


def standardize_pgram(pgram):
    """
    Standardize the noise distribution of a periodogram such that it is a Gaussian with mean=0, stdev=1
    """
    snr_distrib = np.histogram(pgram.snrs.max(axis=1), bins=500)
    try:
        popt, _ = curve_fit(gaussian_model, snr_distrib[1][:-1], snr_distrib[0],p0=[3000,1,0.1],bounds=([0,0,0],[np.inf,np.inf,np.inf]))
    except (RuntimeError, OptimizeWarning) as e:
        # Make a rough guess of the noise properties based off what we'd expect
        popt = [1.0,0.1]
    # Subtract the mean, divide by stdev
    pgram.snrs -= popt[1]
    pgram.snrs /= popt[2]
    
    return pgram

def barycentric_shift(pgram, shifted_freqs, shift_min, shift_max):
    """
    Shifts a standardized periodogram's snrs such that the frequency is barycentric rather than topocentric
    """
    for iw, width in enumerate(pgram.widths):
        s = pgram.snrs[:, iw].astype(float)   
        f = interpolate.interp1d(shifted_freqs, s, kind="nearest")
        pgram.snrs[:, iw] = np.zeros(np.shape(pgram.snrs[:, iw]))
        pgram.snrs[:, iw][shift_min:shift_max+1] = f(pgram.freqs[shift_min:shift_max+1])
    return pgram



##########################################################
#####################                  ###################
##################### PERIODOGRAM_FORM ###################
#####################                  ###################
##########################################################

def periodogram_form(
        tseries_np,
        dm,
        birdies, 
        shifted_freqs, 
        shift_min, 
        shift_max, 
        period_min=2, 
        period_max=10, 
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
        dm: dm of the tseries
        birdies: 1D array containing all the frequencies of birdies to filter out
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
        pgram: a riptide.Periodogram object
        peaks: an array of riptide.Peak objects, sorted by decreasing snr
        For more information of each of these, see https://riptide-ffa.readthedocs.io/en/latest/reference.html
    """
    tseries = TimeSeries.from_numpy_array(tseries_np, TSAMP)

    # See https://riptide-ffa.readthedocs.io/en/latest/reference.html for full explanation of these parameters
    ts, pgram = ffa_search(
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
    # Correct for barycentric shift. This involves shifting the entire periodogram by a certain factor and interpolating extra points
    if shifted_freqs is not None:
        pgram = barycentric_shift(pgram, shifted_freqs, shift_min, shift_max)

    # Find every point above sigma 5 and cluster them into Peak objects
    peaks, _ = find_peaks(pgram, smin = 5, nstd = 0)
    
    # Remove peaks with duplicate periods
    # seen_periods = set()
    # peaks = [peak for peak in peaks if peak.period not in seen_periods and not seen_periods.add(peak.period)]

    # Currently we only remove birdies for DM trials below 5. This is semi-arbitrary, could possibly make this more rigorous?
    if dm < 5:
        # Tolerance range in seconds
        tolerance = 0.005
        pgram_len = len(pgram.snrs)

        # Zap peaks at periods within the tolerance range of a known birdie or its first num_birdie_harmonics harmonics
        for peak in peaks:
            cleaned = False
            for birdie in birdies:
                for harmonic_order in range(1,num_birdie_harmonics+1):
                    harmonic_period = birdie*harmonic_order
                    if abs(peak.period - harmonic_period) < tolerance:
                        period_index = peak.ip
                        # Find the width of the RFI peak in indexes
                        peak_width = 0
                        while (pgram.snrs[max(0,period_index-peak_width)][peak.iw] > 5) and (pgram.snrs[min(pgram_len,period_index+peak_width)][peak.iw] > 5):
                            peak_width += 1
                        for index in range(max(0,period_index-peak_width), min(pgram_len,period_index+peak_width)):
                            pgram.snrs[index][peak.iw] = 0
                        cleaned = True
                        break
                if cleaned:
                    break
                    
    return ts, pgram, peaks

def rate_periodogram(peaks, dm, birdies):
    """
    Gives a rating for a periodogram, taking into account the dm, the strength of the main peak and its harmonics
    Mainly used for finding the best candidate DM of an observation
    """
    
    if len(peaks) == 0 or dm < 2:
        return 0
    
    main_peak = peaks[0]

    tolerance = 0.005
    # Zap if main peak is one of the first 32 harmonics of a birdie
    for birdie in birdies:
        for harmonic_order in range(1,33):
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
    create_dedisp_ts --psr B0154+61 --date 20240610 --stack --candidates
    Input: See click options
    Output: If specified, creates candidates, plots, and writes to the periodogram stacks. Nothing returned directly
    """
    
    apply_logging_config('INFO')
    db_utils.connect(host = 'sps-archiver1', port = 27017, name = 'sps-processing')

    date = utils.convert_date_to_datetime(date)
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

    FFA_search(
        dedisp_ts,
        obs_id,
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
    )

    return


##########################################################
#####################                  ###################
#####################    FFA_SEARCH    ###################
#####################                  ###################
##########################################################


def FFA_search(
    dedisp_ts,
    obs_id,
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
    Performs the FFA search on a set of dedispersed time series for a single pointing.
    Input:
        dedisp_ts: The array of dedispersed time series from the FDMT. Should include dedisp_ts.dedisp_ts AND dedisp_ts.dms
        date: datetime object corresponding to the observation
        plot: boolean. Whether or not to generate periodogram plots and pulse profiles at best DM
        generate_candidates: boolean. Whether or not to generate candidates
        stack: boolean. Whether or not to write to the stacks
        ra: Right Ascension of the observation
        dec: Declination of the observation
        dm_downsample: The downsampling factor in DM trials. Must be an integer >= 1. Recommended value is 8
        dm_min: Only search for DMs above this value
        dm_max: Only search for DMs below this value
        num_threads: Number of cores available
    Output:
        If generate_candidates is set to true, will create SinglePointingCandidate_FFA files in the appropriate folder
        If plot is set to true, will create periodogram plots and pulse profiles in the appropriate folder
        If stack is set to true, will append periodograms to the stacks or create a new stack
    """
    
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
    barycentric_beta = barycenter.get_mean_barycentric_correction(str(ra),str(dec),mjd_time,tobs)
    log.info(f"Tobs: {tobs}, beta: {barycentric_beta}")

    time_start = time.time()

    # Compute pgram at DM 0 to get birdies. A birdie is any peak above 5 snr (maybe tweak this value?)
    birdies = []
    ts_0, pgram_0, peaks_0 = periodogram_form(ts[0], 0, birdies, None, None, None)
    for peak in peaks_0:
        if peak.snr > 5 and peak.period not in birdies:
            birdies.append(peak.period)
    log.info(f"Birdies: {birdies}")
            
    # Use radio convention to shift the frequencies/periods
    shifted_freqs = pgram_0.freqs * (1-barycentric_beta)
    shift_min = np.where(pgram_0.freqs <= shifted_freqs[0])[0][0]
    shift_max = np.where(pgram_0.freqs >= shifted_freqs[-1])[0][-1]
    
    # Call all other DM results using MP
    args = [(
        ts[ts_index], 
        dms[ts_index], 
        birdies, 
        shifted_freqs, 
        shift_min, 
        shift_max
    ) for ts_index in range(1,len(ts))]

    with multiprocessing.Pool(processes=num_threads) as pool:
            results = pool.starmap(periodogram_form,args)
    ts_array, pgram_array, peaks_array = zip(*results)

    ts_array = np.concatenate(([ts_0], ts_array))
    pgram_array = np.concatenate(([pgram_0], pgram_array))
    peaks_array = [peaks_0] + list(peaks_array)

    if generate_candidates:
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
        if len(peaks_array[best_dm_trial]) > 0:
            main_peak = peaks_array[best_dm_trial][0]
            pgram = pgram_array[best_dm_trial]
            mask = (pgram.periods >= main_peak.period-0.01) & (pgram.periods <= main_peak.period+0.01)
            periods_section = pgram.periods[mask]
            snrs_section = pgram.snrs.max(axis=1)[mask]
            
            # Fit the Lorentzian curve to the selected data if possible
            try:
                popt, pcov = curve_fit(lorentzian, periods_section, snrs_section, p0=[main_peak.period, 0.01, main_peak.snr])
                perr = np.sqrt(np.diag(pcov))
            except (RuntimeError, OptimizeWarning) as e:
                popt = None
                perr = None
        else:
            popt = None
            perr = None
        
        best_pgram = pgram_array[best_dm_trial]
        if len(peaks_array[best_dm_trial]) > 0:
            best_peak = peaks_array[best_dm_trial][0]
        else:
            best_peak = None
            log.warning("Best dm trial has no peaks!")
        best_ts = ts_array[best_dm_trial]
        best_dm = dms[best_dm_trial]
        log.info(
            f"Best DM: {best_dm}"
            f"Best peak: period: {best_peak.period : .2f}, freq: {best_peak.freq : .2f}"
            f"           snr: {best_peak.snr : .2f}, iw: {best_peak.iw}"
        )

    total_time = time.time()-time_start
    log.info("Finished making periodograms for all DM trials")
    log.info(f"Time: {total_time} s")

    candidates = np.array([])
    if generate_candidates and best_peak.snr > 10 and False:
        # Create Candidate object with useful info
        dm_range = dms
        freq_range = [1/pgram_array[0].periods[period_index] for period_index in range(best_peak.ip+200,best_peak.ip-200,-1)]
        
        dm_freq_sigma = {
            "sigmas": [[pgram_array[dm_index].snrs[period_index].max() 
                for period_index in range(best_peak.ip+200,best_peak.ip-200,-1)] 
                    for dm_index in range(n_dm_trials)],
            "dms": dm_range,
            "freqs": freq_range
        }
        
        raw_widths_array = {
            "widths": [[pgram_array[0].widths[np.argmax(pgram_array[dm_index].snrs[period_index])] 
                for period_index in range(best_peak.ip+200,best_peak.ip-200,-1)] 
                    for dm_index in range(max(best_dm_trial-30,0),min(best_dm_trial+30,n_dm_trials))],
            "dms": dm_range,
            "freqs": freq_range
        }
        
        min_dm_index, max_dm_index = find_area_around_peak(
            [pgram_array[i].snrs[best_peak.ip][best_peak.iw] for i in range(n_dm_trials)],
            best_dm_trial,
            5
        )
        unique_dms = dms[min_dm_index:max_dm_index]
        
        min_freq_index, max_freq_index = find_area_around_peak(
            pgram_array[best_dm_trial].snrs[: ,best_peak.iw],
            best_peak.ip,
            5
        )
        unique_freqs = best_pgram.periods[min_freq_index:max_freq_index]
        
        sigmas_per_width = pgram_array[best_dm_trial].snrs[best_peak.ip]

        max_sig_det_dtype = [
            ('dm', 'f8'),       # float64
            ('freq', 'f8'),     # float64
            ('sigma', 'f8'),    # float64
            ('width', 'i4'),    # integer
            ('injection', 'b1') # boolean
        ]
        max_sig_det = np.zeros(1,dtype=max_sig_det_dtype)
        max_sig_det['dm'] = dms[best_dm_trial]
        max_sig_det['freq'] = best_peak.freq
        max_sig_det['sigma'] = best_peak.snr
        max_sig_det['width'] = best_pgram.widths[best_peak.iw]
        max_sig_det['injection'] = False

        widths_info_dtype = [
            ('dm', 'f8'),       # float64
            ('freq', 'f8'),     # float64
            ('sigma', 'f8'),    # float64
            ('width', 'i4'),    # integer
        ]
        widths_info = np.array([(dms[best_dm_trial],best_peak.freq, best_pgram[best_peak.ip][iw], best_pgram.widths[iw]) for iw in range(len(best_pgram.widths))])
        # Sort by decreasing sigma
        widths_info = np.sort(widths_info, order='sigma')[::-1]


        candidate = SinglePointingCandidate_FFA(
            freq = best_peak.freq,
            dm = dms[best_dm_trial],
            sigma = best_peak.snr,
            ra = ra,
            dec = dec,
            features = np.array([]), #???
            pgram_freq_resolution = pgram_array[0].freqs[1]-pgram_array[0].freqs[0],
            obs_id = [obs_id],
            widths_info = widths_info, #???
            raw_widths_array = raw_widths_array,
            detection_statistic = SearchAlgorithm_FFA(1),
            dm_freq_sigma = dm_freq_sigma,
            sigmas_per_width = sigmas_per_width,
            datetimes = [date],
            mjd_time = mjd_time,
            unique_freqs = unique_freqs,
            unique_dms = unique_dms,

        )

    ### WRITE SECTION ###

    start_time = time.time()

    directory_name = f"/scratch/ltarabout/stack_{np.round(ra,2)}_{np.round(dec,2)}"
    os.makedirs(directory_name, exist_ok=True)

    # Save the periods and foldbins in a separate file, since they are the same for every dm trial
    periods_bins_name = f"periods_bins_{np.round(ra,2)}_{np.round(dec,2)}.json"
    file_path = os.path.join(directory_name, periods_bins_name)
    if not os.path.exists(file_path):
        periods_bins = {
            "widths": pgram_array[0].widths.tolist(),
            "periods": pgram_array[0].periods.tolist(),
            "foldbins": pgram_array[0].foldbins.tolist(),
            "metadata": {"tobs":tobs}
        }
        with open(file_path, 'w') as json_file:
            json.dump(periods_bins, json_file, indent=4)
    
    if generate_candidates and candidate is not None:
        # Save the candidate in the candidate file
        # In reality, full path will be /data/chime/sps/sps_processing/{year}/{month}/{day}/{ra}_{dec}_FFA_candidates.npz (????)
        os.makedirs(f"candidates/{date.year}/{date.month}/{date.day}", exist_ok=True)
        file_path = f"candidates/{date.year}/{date.month}/{date.day}/{np.round(ra,2)}_{np.round(dec,2)}_FFA_candidates.npz"
        riptide.save_json(file_path, candidate)
        log.info(f"Saved a new candidate at {file_path}")

    if plot:
        best_pgram.plot()
        os.makedirs(f"dailies/plots", exist_ok=True)
        file_path = f"dailies/plots/FFA_periodogram_{np.round(ra,2)}_{np.round(dec,2)}_{np.round(best_peak.snr,2)}_{np.round(best_peak.freq,3)}_{np.round(best_dm,3)}.png"
        plt.savefig(file_path)
        log.info(f"Saved best periodogram at {file_path}")
        plt.clf()
    
        try:
            bins = 256
            subints = best_ts.fold(best_peak.period, bins, subints=16)
        
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(subints, cmap='Greys', aspect='auto')
            ax2.plot(subints.sum(axis=0))
            ax2.set_xlim(0, bins)
            file_path = f"dailies/plots/FFA_profile_{np.round(ra,2)}_{np.round(dec,2)}_{date_string}_{np.round(best_peak.freq,3)}_{np.round(best_dm,3)}.png"
            plt.savefig(file_path)
            log.info(f"Saved best folded pulse profile at {file_path}")
            plt.close(fig)
        except:
            log.warning("Failed to generate pulse profile plot")
    
    # Write peak track files and snrs for each dm trial separately
    for dm_index in range(n_dm_trials):
        if generate_candidates and len(peaks_array[dm_index]) > 0:
            # Write to the peak track file
            peak_track_name = f"peak_track_{np.round(ra,2)}_{np.round(dec,2)}_{np.round(dms[dm_index],2)}.json"
            file_path = os.path.join(directory_name, peak_track_name)
            
            new_peak_track = {
                "mjd":mjd_time,
                "period":peaks_array[dm_index][0].period,
                "freq":peaks_array[dm_index][0].freq,
                "ra":ra,
                "dec":dec,
                "dm":dms[dm_index]
            }
            if dm_index == best_dm_trial:
                new_peak_track["fit params"] = popt
                new_peak_track["fit errors"] = perr

            # Add new best peak for that DM
            if os.path.exists(file_path):
                peak_track = riptide.load_json(file_path)
    
                # Add new best peak for that DM
                peak_track.append(new_peak_track)    
                # Sort the array by the mjd time
                peak_track = sorted(peak_track, key=lambda x: x["mjd"])
                
                riptide.save_json(file_path,peak_track)
            else:
                riptide.save_json(file_path,[new_peak_track])


    if stack:
        # Write/add to the stack files
        log.info("Writing to stack")
        for dm_index in range(n_dm_trials):
            stack_name = f"stack_{np.round(ra,2)}_{np.round(dec,2)}_{np.round(dms[dm_index],2)}.hdf5"
            file_path = os.path.join(directory_name, stack_name)

            # If file already exists, add each periodogram to the stack
            if os.path.exists(file_path):
                with h5py.File(file_path, 'a') as hf:
                    stack_snrs = hf['snrs'][:]
                    stack_length = hf.attrs['stack_length']

                    # Rescale down then back up to preserve similar sigmas across different days
                    stack_snrs = (stack_snrs + (pgram_array[dm_index].snrs / stack_length))
                    
                    snr_distrib = np.histogram(stack_snrs.max(axis=1), bins=500)
                    popt, pcov = curve_fit(gaussian_model, snr_distrib[1][:-1], snr_distrib[0],p0=[3000,1,3])
                    stack_snrs -= popt[1]
                    stack_snrs /= popt[2]
                    
                    hf['snrs'][:] = stack_snrs
                    hf.attrs['stack_length'] = stack_length+1
            else:
                with h5py.File(file_path, 'w') as hf:
                    hf.create_dataset('snrs', data=pgram_array[dm_index].snrs)
                    hf.attrs['stack_length'] = 1
                    hf.attrs['ra'] = ra
                    hf.attrs['dec'] = dec
                    hf.attrs['dm'] = dms[dm_index]


    log.info(f"Write time: {time.time()-start_time}")
    return

log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)# import in this way for easy reload

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)
    main()
    