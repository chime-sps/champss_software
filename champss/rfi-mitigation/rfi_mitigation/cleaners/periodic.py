#!/usr/bin/env python3

import logging
from typing import List

import numpy as np
import yaml

# import location for data files to exist so that importlib can correctly find them
from rfi_mitigation import data as rfi_mitigation_data
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sps_common.barycenter import bary_from_topo_freq, topo_from_bary_freq
from sps_databases import db_api

try:
    import importlib.resources as pkg_resources
except ImportError as e:
    # For Python <3.7 we would need to use the backported version
    import importlib_resources as pkg_resources


log = logging.getLogger(__name__)


class StaticPeriodicFilter:
    def __init__(self, birdies=None):
        self.known_birdies = birdies

        if self.known_birdies is None:
            # read the birdies from the YAML file
            with pkg_resources.path(rfi_mitigation_data, "birdies.yaml") as p:
                # the variable `p` is a PosixPath object, so we still need to open it
                log.debug(f"Accessing known birdies list from {str(p)}")
                self.known_birdies = yaml.load(
                    open(p),
                    Loader=yaml.FullLoader,
                )

    @classmethod
    def from_ks_list(cls, ks_list):
        ks_birdies = {}
        for source in ks_list:
            if not np.isnan(source.spin_period_s):
                source_dict = {
                    "frequency": 1 / source.spin_period_s,
                    "width": getattr(source, "filter_width", 0.01),
                    "int_nharm": getattr(source, "filter_int_nharm", None),
                    "frac_nharm": getattr(source, "filter_frac_nharm", 0),
                }
                ks_birdies[f"{source.source_name}"] = source_dict
        birdie_dict = {"known_bad_frequencies": ks_birdies}
        return cls(birdies=birdie_dict)

    def apply_static_mask(self, ps_freqs: np.ndarray, beta: float = 0) -> List[int]:
        """
        For a list of power spectrum frequencies, flag those indices that should be
        ignored due to known periodic signals. Freuqencies are assumed to be
        topocentric, thus a barycentric correction is made if `beta` is non-zero.

        Parameters
        ----------
        :param ps_freqs: An array of Fourier frequencies belonging to a power
        spectrum
        :type ps_freqs: numpy.ndarray

        :param beta: The barycentric correction applied, in units of the speed
        of light
        :type beta: float

        :return bad_indices: A list of integers representing the frequency bins
        to be flagged when searching power spectra.
        :rtype: List[int]
        """
        log.info(
            "Gathering known bad frequencies (birdies), barycentering, "
            "flagging integer and fractional harmonics"
        )
        df = ps_freqs[1] - ps_freqs[0]
        nfreqs = len(ps_freqs)
        bad_indices = []

        for k, bird in self.known_birdies["known_bad_frequencies"].items():
            if beta != 0:
                bf = bary_from_topo_freq(bird["frequency"], beta)
            else:
                bf = bird["frequency"]

            bin_width = int(bird["width"] / df)
            # at this point, the bin width could technically be 0, especially
            # if the only a "small" number of zeroes we padded onto the time
            # series
            if bin_width == 0:
                log.warning("Listed birdy width corresponds to <1 Fourier bin!")
                log.warning(
                    "This will be corrected later so that at least the "
                    "central and first adjacent bins (i.e. "
                    "[x-1, x, x+1] are flagged."
                )
                bin_width = 1

            # decide how many harmonics to compute
            if "int_nharm" in bird.keys() and bird["int_nharm"] is not None:
                nharm_integer = bird["int_nharm"]
            else:
                # maximum frequency is just over 508 Hz, number of integer
                # harmonics up to 512 Hz covers the entire sensitive space
                nharm_integer = int(np.ceil(512.0 / bf))

            if "frac_nharm" in bird.keys() and bird["frac_nharm"] is not None:
                nharm_fractional = bird["frac_nharm"]
            else:
                # much more than this and we'll be wiping out the lower
                # frequency end entirely, which isn't ideal
                nharm_fractional = 8

            log.info(f"Topo. freq. = {bird['frequency']} Hz Bary. freq. = {bf} Hz")
            log.info(f"Num. integer harmonics to flag = {nharm_integer}")
            log.info(f"Num. fractional harmonics to flag = {nharm_fractional}")

            bad_freqs = [n * bf for n in range(1, nharm_integer + 1)]
            bad_freqs.extend([(1.0 / n) * bf for n in range(2, nharm_fractional + 2)])

            # find the nearest index corresponding to the harmonics
            for b in bad_freqs:
                log.debug(f"checking {b} Hz")
                idx = np.argmin(np.abs(ps_freqs - b))
                log.debug(f"closest = {ps_freqs[idx]} Hz ({idx})")

                # in addition to the closest, also remove the adjacent bins
                # based on the given width as the peaks are usually >1 bin
                # wide (unless the Fourier interpolated resolution is small)
                # compute the edges, where we will flag bins in the range
                #   [x-left_edge, x+right_edge)
                # (i.e. exclude bins up to, but not including, the right edge)
                left_edge = idx - int(np.ceil(1.5 * bin_width))
                right_edge = idx + int(np.ceil(1.5 * bin_width)) + 1
                if right_edge >= nfreqs:
                    right_edge = nfreqs
                if left_edge < 0:
                    left_edge = 0

                bad_indices.extend(range(left_edge, right_edge))

        # sort things so that bin numbers are arranged from smallest to
        # largest, and ensure that the list is unique
        bad_indices = sorted(set(bad_indices))
        # Make sure that no indices that are too high exist
        bad_indices = [index for index in bad_indices if index < nfreqs]
        n_bad_indices = len(bad_indices)
        log.info(
            "Fraction of FFT bins removed ="
            f" {n_bad_indices / nfreqs:g} ({100 * n_bad_indices / nfreqs:g}%)"
        )
        return bad_indices


class DynamicPeriodicFilter:
    """
    Inspect the power spectrum values to determine obviously bad frequency bins and flag
    them for downstream processes via the function excise_by_identifying_outliers. The
    output of the bad indices via DynamicPeriodicFilter are TOPOCENTRIC.

    Parameters
    ----------
    :param peak_find_height: The height above the mean of the log of the
    power spectrum to flag the birdies in excise_by_identifying_outliers.
    Default = 2.0
    :type peak_find_height: float

    :param peak_find_distance: The distance between adjacent birdies of the
    power spectrum to consider them as separate in number of frequency bins
    in excise_by_identifying_outliers.
    Default = 5.0
    :type peak_find_height: float

    :param peak_find_width: The expected width of the birdies when looking
    for birdies in the power spectrum in excise_by_identifying_outliers.
    Default = 3.0
    type: peak_find_width: float

    :param peak_find_rel_height: The relative peak height between adjacent
    birdies in the power spectrum to consider them separate in
    excise_by_identifying_outliers.
    Default = 0.37
    type: peak_find_rel_height: float

    :param red_noise_fit_start_freq: The lower limit to red-noise removal
    in excise_by_identifying_outliers.
    Default = 0.0048 Hz
    :type red_noise_fit_start_freq: float

    :param red_noise_fit_upper_freq: The higher limit to rednoise removal
    in excise_by_identifying_outliers.
    Default = 3.0048 Hz
    :type red_noise_fit_upper_freq: float

    :param strong_birdie_threshold: The threshold above which birdies
    are flagged as strong birdies and always removed.
    Default = 5.
    :type strong_birdie_threshold: float

    :param update_db: Whether to update the SPS database post mitigation.
    Default = True
    :type update_db: bool
    """

    peak_find_height: float = 2.0
    peak_find_distance: float = 5.0
    peak_find_width: float = 3.0
    peak_find_rel_height: float = 0.37
    red_noise_fit_start_freq: float = 0.0048
    red_noise_fit_upper_freq: float = 3.0048
    strong_birdie_threshold: float = 5.0
    update_db: bool = True

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def excise_by_identifying_outliers(
        self,
        ps: np.ndarray,
        ps_freqs: np.ndarray,
        obs_id: str,
        red_noise_fit: bool = False,
        debug: bool = False,
        barycentric_cleaning: bool = False,
        beta: float = 0,
    ):
        """
        Inspect the raw power spectrum at DM=0 for different beams and list the common
        birdies at 3sigma. excise those birdies which appear in morethan one beams.

        First we obtain the power spectra log for the zero DM and then we remove the red_noise and fit
        for the red noise (using polynomial of fourth order) and subtract to bring the mean at zero.
        The same is applied at freq and obtain a normalised power spectra with mean 0.
        then we carry out the statistics to determine the birdies at 2 sigma.

        Parameters
        ----------
        :param ps: An array of the power spectrum to be excised
        :type ps: numpy.ndarray

        :param ps_freqs: An array of Fourier frequencies belonging to a power
        spectrum
        :type ps_freqs: numpy.ndarray

        :param obs_id: The Observation ID of the power spectrum being processed
        :type obs_id: str

        :param red_noise_fit: Whether to perform the red noise fit
        Default = False
        :type red_noise_fit: bool

        :param debug: Whether to make debugging plots of the mitigation process
        Default = False
        :type debug: bool

        :param barycentric_cleaning: Whether the cleaned spectrum is barycentric
        Default = False
        :type barycentric_cleaning: bool

        :param beta: Beta value with which the values are corrected in order to save topocentric values
        Default = 0
        :type beta: float

        Returns
        -------
        :return (birdies_inf, birdies_inf[strong_birdies_indices]): A tuple containing
        two numpy.recarray that contains birdies attributes for single pointing
        with the following dtype : [
            ("position", int),
            ("height", float),
            ("left", float),
            ("right", float),
        ], where position is the index of the peak of the birdies, height is the peak height of
        the birdies in log space, left and right are the leftmost and rightmost frequencies of the
        birdies. The first entry of the tuple contains the data for all found peaks,
        while the second only contains those with a height above strong_birdie_threshold.
        """

        log.debug(
            f"flagging freq bins with peak above {self.peak_find_height} sigma limit "
        )
        log.debug("Replace first bin in power spectra by mean value")
        ps[0] = np.median(ps)
        red_noise_fit_start_freq_bin = np.where(
            ps_freqs < self.red_noise_fit_start_freq
        )[0][-1]
        red_noise_fit_upper_freq_bin = np.where(
            ps_freqs > self.red_noise_fit_upper_freq
        )[0][0]
        red_noise_range = slice(
            red_noise_fit_start_freq_bin, red_noise_fit_upper_freq_bin
        )

        log.debug("taking the log of power spectrum")

        # replacing 0's wth 1's to avoid log function error

        np.place(ps, ps == 0, [1])
        log_ps = np.log10(ps)
        log.debug(
            "normallising the log of power spectra to zero mean and unit standard"
            " deviation"
        )

        # replace NaNs and infs with 0

        std = np.std(log_ps)
        if np.std(log_ps) == 0.0:
            std = 1.0
        log_ps = np.where(log_ps == np.inf, 0, log_ps)
        log_norm_ps = (log_ps - np.mean(log_ps)) / std
        log_norm_ps = np.nan_to_num(log_norm_ps)
        log_norm_ps = np.where(log_norm_ps == np.inf, 0, log_norm_ps)

        # normalising power spectrum with mean 0 and standard deviation 1

        if red_noise_fit:
            # fitting the polynomial for the red noise part odf the ps (upto 2Hz)
            pars, cov = curve_fit(
                f=fourth_order_polynomial,
                xdata=ps_freqs[red_noise_range],
                ydata=log_norm_ps[red_noise_range],
                p0=[-1, 1, 1, 1, 1],
                bounds=(-np.inf, np.inf),
            )

            log.debug(
                "the covariance values for fitting the part of power spetcra affected"
                f" by red noise {cov}"
            )

            # some type of warning message if the covariance values are abnormal
            # checking the goodness of fit for the red noise removal

            # subtracting the fitted fourth_order_polynomial from the power spectra
            log_norm_ps[red_noise_range] = log_norm_ps[red_noise_range] - (
                fourth_order_polynomial(ps_freqs[red_noise_range], *pars)
            )

            # TODO : to get the estimate of how much area red noise is covering for quality control

        # Find peaks or birdies with scipy module find_peaks
        peaks = find_peaks(
            log_norm_ps,
            height=self.peak_find_height,
            distance=self.peak_find_distance,
            width=self.peak_find_width,
            rel_height=self.peak_find_rel_height,
        )

        height = peaks[1]["peak_heights"]  # list of the heights of the peaks

        # saving topocentric frequencies in the database is better for comparing birdies from different days
        peaks_pos = peaks[0]  # peak frequency index of each birdies
        left_markers = (
            peaks[1]["left_ips"] * ps_freqs[1]
        )  # leftmost frequency of the peak
        right_markers = (
            peaks[1]["right_ips"] * ps_freqs[1]
        )  # rightmost frequency of the peak

        # with barycentric_cleaning the topocentric values need to be converted back to the topocentric frame
        # in order to write them to the database
        if barycentric_cleaning:
            peaks_pos = topo_from_bary_freq(peaks_pos, beta).round().astype(int)
            left_markers = topo_from_bary_freq(left_markers, beta)
            right_markers = topo_from_bary_freq(right_markers, beta)

        widths_of_birdies = right_markers - left_markers
        percentage_of_data_flagged = sum(widths_of_birdies) * 100 / ps_freqs[-1]

        log.info(f"The percentage of data flagged is {percentage_of_data_flagged}")

        if debug:
            log.info("enable debug mode to see Diagnostic Plots for troubleshooting")
            from matplotlib import pyplot as plt

            res = ps_freqs[1]
            figure = plt.figure(figsize=(100, 30))
            ax = figure.subplots()
            plt.rc("font", size=50)
            x = ps_freqs
            y = log_norm_ps

            ax.plot(x, y)

            ax.scatter(
                peaks_pos * res, height, color="r", s=50, marker="D", label="birdies"
            )

            # getting previous observation data for same pointing
            current_observation = db_api.get_observation(obs_id)
            surround_observations = db_api.get_observations_before_observation(
                current_observation, 0, 2
            )

            log.info(f"the number of surrounding pointngs {len(surround_observations)}")
            num_to_marker = {0: "o", 1: "+", 2: "x", 3: "1", 4: "2", 5: "3", 6: "4"}
            for num, observ in enumerate(surround_observations):
                if observ == None or observ.birdies_position == None:
                    plt.title("plotting_without_previous_observations")
                    continue
                # the number of pointings we limit to 7 (randomly) to see the periodic RFI sitatuation for the current pointing being processed and list to see
                # common birdies in different pointings to rule out masking any astrophysical signal or noise.
                if num < 6:
                    pointing = db_api.get_pointing(observ.pointing_id)
                    log.info(
                        f"length of birdies index array  {len(observ.birdies_position)}"
                    )
                    log.info(
                        f"length of birdies height array  {len(observ.birdies_height)}"
                    )
                    ax.scatter(
                        np.array(observ.birdies_position) * res,
                        observ.birdies_height,
                        color="black",
                        s=20,
                        marker=num_to_marker[num],
                        label=str(pointing.ra)[:-11] + "_" + str(pointing.dec)[:-12],
                    )

            plt.axhline(y=5.0, color="green", linestyle="--")
            plt.axhline(y=2.0, color="green", linestyle="--")
            plt.xlabel("frequency(Hz)", fontsize=50)
            plt.ylabel("Intensity(arb. units)", fontsize=50)
            plt.xticks(fontsize=40)
            plt.yticks(fontsize=40)
            ax.legend(fontsize=50)
            plt.savefig(obs_id + ".png")
            plt.clf()

        # Preparing a recursive array that will return the peaks positions heights and widths for zero DM power spectra
        # of the particular beam
        birdies_inf = np.recarray(
            (len(height),),
            dtype=[
                ("position", int),
                ("height", float),
                ("left", float),
                ("right", float),
            ],
        )
        # len(height) actually measures the number of birdies in the current beam
        birdies_inf.height = height
        birdies_inf.position = peaks_pos
        birdies_inf.right = right_markers
        birdies_inf.left = left_markers

        birdies_height = height.tolist()
        birdies = peaks_pos.tolist()
        birdies_left_freq = birdies_inf.left.tolist()
        birdies_right_freq = birdies_inf.right.tolist()

        # select strong birdies and add to periodic rfi mask
        strong_birdies_indices = np.where(
            birdies_inf.height > self.strong_birdie_threshold
        )
        length = [len(birdies_height), len(birdies)]

        log.info(
            f"Birdie count: {len(birdies_height)}; Strong birdies:"
            f" {len(strong_birdies_indices)}"
        )
        # Only take ps mean and std outside rednoise fitting bins
        if self.update_db:
            log.info("Updating birdie information in database.")
            self.update_database(
                obs_id,
                birdies_height,
                birdies,
                birdies_left_freq,
                birdies_right_freq,
            )

        return birdies_inf, birdies_inf[strong_birdies_indices]

    def update_database(
        self,
        obs_id,
        bad_freq_peak,
        bad_freqs,
        birdies_left_freq,
        birdies_right_freq,
    ):
        """
        Update sps-databases with the properties of the birdies computed from
        excise_by_identifying_outliers.

        Parameters
        ----------

        :param obs_id: Observation ID of the power spectrum processed.
        :type obs_id: str

        :param bad_freqs: List of the indices of the birdies.
        :type bad_freqs: List[int]

        :param bad_freq_peak: List of peak heights of the birdies in
        log space.
        :type bad_freq_peak: List[float]

        :param birdies_left_freq: List of leftmost frequency of the birdies.
        :type birdies_left_freq: List[float]

        :param birdies_right_freq: List of rightmost frequency of the birdies.
        :type birdies_right_freq: List[float]
        """
        payload = dict(
            birdies_position=bad_freqs,
            birdies_height=bad_freq_peak,
            birdies_left_freq=birdies_left_freq,
            birdies_right_freq=birdies_right_freq,
            birdie_file=None,
        )
        db_api.update_observation(obs_id, payload)


def fourth_order_polynomial(x, e, a, b, c, d):
    """Function to return value for a general fourth order polynomial."""

    return e * x**4 + a * x**3 + b * x**2 + c * x + d
