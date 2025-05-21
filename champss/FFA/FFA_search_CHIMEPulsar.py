import numpy as np
import riptide
import datetime
import os
import time
import h5py

from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from scipy import interpolate
from astropy.time import Time

import sps_common.barycenter as barycenter
TSAMP = 0.00032768

def gaussian_model(x, a, mu, sigma):
    """
    Define the Gaussian function for fitting noise distribution
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def standardize_pgram(pgram):
    """
    Standardize the noise distribution of a periodogram such that it is a Gaussian with mean=0, stdev=1
    """
    # Standardize noise distribution
    snr_distrib = np.histogram(pgram.snrs.max(axis=1), bins=500)
    popt, _ = curve_fit(gaussian_model, snr_distrib[1][:-1], snr_distrib[0], p0=[3000, 1, 3])
    pgram.snrs -= popt[1]
    pgram.snrs /= abs(popt[2])

    peaks, _ = riptide.find_peaks(pgram, smin = 7, nstd = 0)
    
    return pgram

def barycentric_shift(pgram, shifted_freqs, shift_min, shift_max):
    """
    Shifts a standardized periodogram's snrs such that the frequency is barycentric rather than topocentric
    This is done by nearest interpolation of the snrs to the corresponding shifted frequency, calculated before the multiprocessing loop once
    """
    for iw, width in enumerate(pgram.widths):
        s = pgram.snrs[:, iw].astype(float)   
        f = interpolate.interp1d(shifted_freqs, s, kind="nearest")
        pgram.snrs[:, iw] = np.zeros(np.shape(pgram.snrs[:, iw]))
        pgram.snrs[:, iw][shift_min:shift_max+1] = f(pgram.freqs[shift_min:shift_max+1])
    return pgram

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
        period_min, 
        period_max,
        barycentric_beta,
        best_period_profile_bool,
        expected_period,
        bins_min=1900, 
        bins_max=2100, 
        rmed_width=4.0, 
        ducy_max=0.05,
        num_birdie_harmonics=1
    ):
    """
    Forms a periodogram from a dedispersed time series
    Input:
        tseries_np: Numpy ND array of a dedispersed time series sampled at TSAMP
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

    # Find every point above sigma 7 and cluster them into Peak objects
    peaks, _ = riptide.find_peaks(pgram, smin = 7, nstd = 0)

    if len(peaks) > 0:
        print("Topocentric", peaks[0])
    
    # Currently we only remove birdies for DM trials below 10. This is semi-arbitrary, could possibly make this more rigorous?
    # Ideally, this limit should depend on the maximum sigma of the DM 0 peaks
    if dm < 10:
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

    
    # Correct for barycentric frame. This involves shifting the entire periodogram by a certain factor and interpolating extra points
    shifted_freqs = pgram.freqs * (1-barycentric_beta)
    shift_min = np.where(pgram.freqs <= shifted_freqs[0])[0][0]
    shift_max = np.where(pgram.freqs >= shifted_freqs[-1])[0][-1]

    pgram = barycentric_shift(pgram, shifted_freqs, shift_min, shift_max)

    # Standardize the noise distribution such that mean=0, stdev=1
    pgram = standardize_pgram(pgram)

    # Unfortunately, we need to compute the peaks again in barycentric frame since the width/spacing could be different now
    peaks, _ = riptide.find_peaks(pgram, smin = 7, nstd = 0)
    if len(peaks) > 0:
        print("Barycentric", peaks[0])

    # desired_peak = [peak for peak in peaks if peak.period > 0.38 and peak.period < 0.41]
    # print(desired_peak)
    # print([peak.period for peak in peaks if peak.period > 0.3 and peak.period < 0.5])
    # folded_period = desired_peak[0].period
    if len(peaks) > 0:
        folded_period = peaks[0].period
    else:
        folded_period = expected_period

    subints = []  
    if best_period_profile_bool:
        bins = 256
        # try:
        subints = ts.fold(1, bins, subints=64)
        # except:
        #     print("Couldn't fold the profile at best period")
        #     pass
    
    # Remove peaks at periods where one width is much stronger than the others
    peaks = rogue_width_filter(peaks, pgram.snrs, pgram.widths)
    
    # Remove peaks with duplicate periods, only keep the strongest
    peaks = remove_duplicates(peaks)
                    
    return ts, pgram, peaks, subints, folded_period



def FFA_search_CHIMEPulsar(
    ts,
    dm,
    date, 
    ra, 
    dec,
    expected_period,
    birdies,
    stack_bool,
    best_period_profile_bool
):
    """
    Performs the FFA search on a set of dedispersed time series for a single pointing.
    Input:
        ts: Dedispersed time series (as np.ndarray) from the FDMT.
        date: datetime object corresponding to the observation
        ra: Right Ascension of the observation
        dec: Declination of the observation
        birdies: Array of birdie periods to filter out (in s)
        stack_bool: Boolean. Whether or not to write to the stack files
        best_period_profile_bool: Boolean. Whether or not to generate profile at best period
    Output:
        Appends periodograms to the stacks or creates a new stack for that pointing
    """
    tobs = len(ts)*TSAMP
    mjd_time = Time(date).mjd
    date_string = date.strftime("%Y%m%d")
    
    # Compute barycentric shift
    barycentric_beta = barycenter.get_mean_barycentric_correction(str(ra),str(dec),mjd_time,tobs)
    # log.info(f"Tobs: {tobs}, beta: {barycentric_beta}")

    time_start = time.time()

    ts, pgram, peaks, subints, folded_period = periodogram_form(
        ts, 
        dm, 
        birdies, 
        expected_period*0.9, 
        expected_period*1.1, 
        barycentric_beta, 
        best_period_profile_bool,
        expected_period
    )

    total_time = time.time()-time_start
    print("Finished making periodograms for all DM trials")
    print(f"Time: {total_time} s")


    ### WRITE SECTION ###

    start_time = time.time()

    directory_name = f"./FFA_search_CHIMEPulsar_test/stack_{np.round(ra,2)}_{np.round(dec,2)}"
    os.makedirs(directory_name, exist_ok=True)

    if stack_bool:
        stack_name = f"stack_{np.round(ra,2)}_{np.round(dec,2)}.hdf5"
        file_path = os.path.join(directory_name, stack_name)
        
        # Write/add to the stack files 
        if os.path.exists(file_path):
            with h5py.File(file_path, 'a') as hf:
                if dm == hf.attrs['dm'] and np.array_equal(pgram.periods, hf['periods']):
                    stack_length = hf.attrs['stack_length']                    
                    stack_snrs = hf['snrs'][:]
    
                    # Rescale down, add the new pgram, then rescale back up to conserve similar sigmas
                    stack_snrs = (stack_snrs + (pgram.snrs / stack_length))
                    
                    # Standardize noise distribution
                    snr_distrib = np.histogram(stack_snrs.max(axis=1), bins=500)
                    popt, _ = curve_fit(gaussian_model, snr_distrib[1][:-1], snr_distrib[0], p0=[3000, 1, 3])
                    stack_snrs -= popt[1]
                    stack_snrs /= abs(popt[2])
        
                    hf['snrs'][...] = stack_snrs
                    hf.attrs['stack_length'] = stack_length + 1
    
                else:
                    print("Cannot add data to stack as the DM or period range is different!")
            
        else:
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('snrs', data=pgram.snrs)
                hf.attrs['stack_length'] = 1
                hf.attrs['ra'] = ra
                hf.attrs['dec'] = dec
                hf.attrs['dm'] = dm
                hf.attrs['tobs'] = tobs
                hf.attrs['widths'] = pgram.widths
                hf.create_dataset('periods', data=pgram.periods)
                hf.create_dataset('foldbins', data=pgram.foldbins)
    
        print(f"Wrote stack file at: {file_path}")

    if best_period_profile_bool and subints != []:
        profile_name = f"profile_{np.round(ra,2)}_{np.round(dec,2)}_{date_string}.npz"
        file_path = os.path.join(directory_name, profile_name)
        
        np.savez(file_path, subints=subints, period=folded_period)
        print(f"Wrote profile from {date_string} at {file_path}")
        
    print(f"Write time: {time.time()-start_time}")
    return



    