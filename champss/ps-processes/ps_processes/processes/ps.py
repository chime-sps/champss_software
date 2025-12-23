"""Class for creating power spectra."""

import logging
import os
import time
from functools import partial
from multiprocessing import Pool, shared_memory

import numpy as np
import pyfftw
import pytz
from astropy.time import Time
from attr import ib as attribute
from attr import s as attrs
from attr.validators import instance_of
from prometheus_client import Summary
from ps_processes.utilities.utilities import rednoise_normalise
from rfi_mitigation.cleaners.periodic import DynamicPeriodicFilter, StaticPeriodicFilter
from scipy.interpolate import interp1d
from scipy.fft import rfftfreq
from sps_common.barycenter import (
    bary_from_topo_freq,
    barycenter_timeseries,
    get_barycentric_correction,
)
from sps_common.constants import SEC_PER_DAY, TSAMP
from sps_common.conversion import convert_ra_dec
from sps_common.interfaces.ps_processes import PowerSpectra
from sps_databases import db_api

log = logging.getLogger(__name__)

ps_processing_time = Summary(
    "ps_pointing_processing_seconds",
    "Duration of running power spectrum creation and quality control for an pointing",
    ("pointing_id",),
)


@attrs(slots=True)
class PowerSpectraCreation:
    """
    Class to create a set of power spectra from a dedispersed time series.

    Parameters
    ----------
    tsamp: float
        Sampling time of the dedispersed time series.
        Default = native sampling time of sps intensity data

    normalise: bool
        Normalise the time series prior to power spectrum creation. Default = True

    barycenter: bool
        Apply the barycentric correction to the topocentric time series. Default = False

    barycentring_mode: str
        Mode which is used for the barycentring. Available: ["Fourier", "Time", "Topocentric"]
        Default: Fourier

    padded_length: int
        The desired padded length of the time series (default is
        2**20 = 1048576), but this will depend on declination

    clean_rfi: bool
        Run RFI cleaning routines to determine what Fourier bins are
        corrupted by periodic RFI. Default = True

    run_static_filter: bool
        Whether to run the static periodic RFI filter to remove known RFI instances.
        Default = True

    run_dynamic_filter: bool
        Whether to run the dynamic periodic RFI filter to remove unknown RFI instances.
        Default = True

    dynamic_filter_config: dict
        The dictonary containing the parameters to use to identify birdies in
        the dynamic periodic RFI filter. Default = {} (No changes to the default
        parameters determined in the dynamic periodic RFI filter class)

    barycentric_cleaning: bool
        Clean the barycentered power spectrum instead of the topocentric. Default = False

    find_common_birdies: bool
        Whether to look for common birdies from surrounding pointings of the observation.
        Default = False

    find_common_birdie_fraction: float
        The fraction of the surrounding observation in which a
        birdie must be found to be removed.
        Default = 0.9

    common_birdies_config: dict
        The configuration for identifying common birdies between adjacent pointings. Currently
        includes 'ra_range' and 'dec_range' that define the ra and dec range to look for common
        birdies of a given observation. Default: dict(ra_range=0.0, dec_range=0.0)

    zero_replace: bool
        Whether to explicitly replace corrupted Fourier bins with zeros.
        Otherwise replace them with random values in a
        chi2 distribution with dof=2. Default = True

    remove_rednoise: bool
        Whether to run rednoise removal on the power spectrum. Default = True

    rednoise_config: dict
        Configuration to run the rednoise removal on the power spectrum produced.
        Default = dict(b0=10, bmax=100000)

    update_db: bool
        Whether to update the sps-databases. Default = True

    nbit: int
        The number of bits to process the power spectra in. Default = 32

    num_threads: int
        The number of threads to run the parallel processing of the FFT process. Default = 8

    save_medians: bool
        Whether to save the rednoise information to the power spectrum.

    write_medians: bool
        Whether to write the rednoise information to a separate npz file.

    write_zero_dm_medians: bool
        Whether to write the zero DM rednoise information to an npz file.
    """

    tsamp = attribute(validator=instance_of(float), default=TSAMP)
    normalise = attribute(validator=instance_of(bool), default=True)
    barycentring_mode = attribute(validator=instance_of(str), default="Fourier")
    padded_length = attribute(validator=instance_of(int), default=1048576)
    clean_rfi = attribute(validator=instance_of(bool), default=True)
    run_static_filter = attribute(validator=instance_of(bool), default=True)
    run_dynamic_filter = attribute(validator=instance_of(bool), default=True)
    dynamic_filter_config = attribute(
        validator=instance_of(dict), type=dict, default={}
    )
    barycentric_cleaning = attribute(validator=instance_of(bool), default=False)
    find_common_birdies = attribute(validator=instance_of(bool), default=False)
    common_birdie_fraction = attribute(validator=instance_of(float), default=0.9)
    common_birdies_config = attribute(
        validator=instance_of(dict), default=dict(ra_range=0.0, dec_range=0.0)
    )
    zero_replace = attribute(validator=instance_of(bool), default=True)
    remove_rednoise = attribute(validator=instance_of(bool), default=True)
    rednoise_config = attribute(
        validator=instance_of(dict), default=dict(b0=10, bmax=100000)
    )
    update_db = attribute(validator=instance_of(bool), default=True)
    nbit = attribute(validator=instance_of(int), default=32)
    num_threads = attribute(validator=instance_of(int), default=8)
    mp_chunk_size: bool = attribute(default=10)
    save_medians: bool = attribute(default=True)
    write_medians: bool = attribute(default=False)
    write_zero_dm_medians: bool = attribute(default=False)
    static_filter = attribute(init=False)
    dynamic_filter = attribute(init=False)

    @common_birdies_config.validator
    def _validate_common_birdies_config(self, attribute, value):
        assert "ra_range" and "dec_range" in value, (
            "The keys 'ra_range' and/or 'dec_range' are not in common_birdies_config"
        )

    @nbit.validator
    def _validate_nbit(self, attribute, value):
        assert value in [
            16,
            32,
            64,
            128,
        ], "The nbit value must be 16, 32, 64 or 128"

    @barycentring_mode.validator
    def _validate_barycentring_mode(self, attribute, value):
        assert value in [
            "Topocentric",
            "Time",
            "Fourier",
        ], "The barycentring_mode  must be Topocentric, Time or Fourier"

    def __attrs_post_init__(self):
        """Setup cleaner and check verifu chosen barycentring method."""
        if self.clean_rfi:
            if self.run_static_filter:
                self.static_filter = StaticPeriodicFilter()
            if self.run_dynamic_filter:
                self.dynamic_filter = DynamicPeriodicFilter(
                    **self.dynamic_filter_config, update_db=self.update_db
                )
        if self.barycentring_mode == "Topocentric" and self.barycentric_cleaning:
            self.barycentric_cleaning = False

    def transform(self, dedisp_time_series, write_medians=False):
        """
        Function to perform the FFT and creating the normalised power spectrum.

        Paramaters
        ----------
        dedisp_time_series: DedispersedTimeSeries
            The input DedispersedTimeSeries class which have the data and
            the metadata to process
        write_medians: bool
            Whether to write the rednoise information to a npz file.

        Returns
        -------
        pspec: PowerSpectra
            The PowerSpectra class as defined in the interface
        """
        pool = Pool(self.num_threads)
        observation = db_api.get_observation(dedisp_time_series.obs_id)
        pointing_id = self.get_pointing_id_from_observation_id(
            dedisp_time_series.obs_id
        )

        with ps_processing_time.labels(pointing_id).time():
            dedisp_ts_len = dedisp_time_series.dedisp_ts.shape[-1]
            beta = self.get_barycentric_correction(
                dedisp_ts_len,
                dedisp_time_series.ra,
                dedisp_time_series.dec,
                dedisp_time_series.start_mjd,
            )
            bad_freq_indices = []
            compared_obs = []
            medians_path_dm0 = None
            freq_labels = self.get_freq_labels()
            if self.clean_rfi:
                log.info("Running Periodic RFI Cleaning")
                # Current we only masked specific frequencies so this works
                bad_freq_indices, compared_obs, birdies_inf, medians_path_dm0 = (
                    self.flag_periodic_rfi(
                        dedisp_time_series, freq_labels, beta, observation
                    )
                )
                nbins_flagged = len(bad_freq_indices)
                log.info(
                    "Number of FFT bins flagged ="
                    f" {nbins_flagged} ({nbins_flagged / len(freq_labels):.6f} of data)"
                )
            # Pool method to run parallel jobs on the FFT to form power spectra
            log.info(
                f"Number of threads used for power spectra creation: {self.num_threads}"
            )
            target_shape = (
                dedisp_time_series.dedisp_ts.shape[0],
                int(self.padded_length // 2),
            )
            buffer_size = int(target_shape[0] * target_shape[1] * self.nbit / 8)
            power_spectra_shared = shared_memory.SharedMemory(
                create=True, size=buffer_size
            )
            power_spectra = np.ndarray(
                target_shape, dtype=f"float{self.nbit}", buffer=power_spectra_shared.buf
            )

            if not self.mp_chunk_size:
                self.mp_chunk_size = (
                    dedisp_time_series.dedisp_ts.shape[0] // self.num_threads + 1
                )
            log.info(
                "Number of dm trials in single multi processing job:"
                f" {self.mp_chunk_size}"
            )

            dedisp_series_list = [
                dedisp_time_series.dedisp_ts[i : i + self.mp_chunk_size]
                for i in range(
                    0, dedisp_time_series.dedisp_ts.shape[0], self.mp_chunk_size
                )
            ]
            full_indices = np.arange(dedisp_time_series.dedisp_ts.shape[0])
            dm_indices = [
                full_indices[i : i + self.mp_chunk_size]
                for i in range(
                    0, dedisp_time_series.dedisp_ts.shape[0], self.mp_chunk_size
                )
            ]

            median_info = pool.starmap(
                partial(
                    self.transform_data,
                    normalise=self.normalise,
                    barycentring_mode=self.barycentring_mode,
                    beta=beta,
                    padded_length=self.padded_length,
                    nbit=self.nbit,
                    remove_rednoise=self.remove_rednoise,
                    clean_rfi=self.clean_rfi,
                    rednoise_config=self.rednoise_config,
                    zero_replace=self.zero_replace,
                    bad_freq_indices=bad_freq_indices,
                    shared_target_name=power_spectra_shared.name,
                    target_shape=power_spectra.shape,
                ),
                zip(dm_indices, dedisp_series_list),
            )
            pool.close()
            pool.join()
            log.info("Power spectra creation successful.")

            if self.save_medians or self.write_medians:
                median_dm_indices = []
                medians = []
                scales = []

                for info in median_info:
                    median_dm_indices.extend(info[0])
                    medians.extend(info[1])
                    scales.extend(info[2])

                medians = np.asarray(medians)
                median_dm_indices = np.asarray(median_dm_indices)
                scales = np.asarray(scales[0]) #same for each DM
                #this is the jankiest way of getting the freq_labels but I'm not sure how else to do it
                rn_all_freqs = rfftfreq(2 * (power_spectra.shape[1] - 1), d = TSAMP)
                rn_freq_labels = rn_all_freqs[np.cumsum(scales)]
            
            if self.save_medians:
                rn_medians = medians[np.newaxis, :]
                rn_dm_indices = median_dm_indices[np.newaxis, :]
                rn_scales = scales[np.newaxis, :]
            else:
                rn_medians = None
                rn_scales = None
                rn_dm_indices = None

            if self.write_medians:
                medians_path = f"{os.path.abspath(observation.datapath)}/medians.npz"
                log.info(f"Saving rednoise information to {medians_path}.")
                np.savez(
                    medians_path,
                    medians=medians[np.newaxis, :],
                    scales=scales[np.newaxis, :],
                    freq_labels=rn_freq_labels[np.newaxis, :],
                    dms=dedisp_time_series.dms,
                )
            else:
                medians_path = None
            # update the observation database
            if self.update_db:
                self.update_database(
                    observation,
                    power_spectra[0],
                    bad_freq_indices,
                    beta,
                    self.barycentring_mode,
                    self.barycentric_cleaning,
                    compared_obs,
                    birdies_inf,
                    medians_path,
                    medians_path_dm0,
                )
        datetimes = Time(dedisp_time_series.start_mjd, format="mjd").datetime.replace(
            tzinfo=pytz.utc
        )
        
        return PowerSpectra(
            power_spectra=power_spectra,
            dms=dedisp_time_series.dms,
            freq_labels=freq_labels,
            ra=dedisp_time_series.ra,
            dec=dedisp_time_series.dec,
            datetimes=[datetimes],
            num_days=1,
            beta=beta,
            bad_freq_indices=[bad_freq_indices],
            obs_id=[dedisp_time_series.obs_id],
            power_spectra_shared=power_spectra_shared,
            rn_medians=rn_medians,
            rn_scales=rn_scales,
            rn_dm_indices=rn_dm_indices,
        )

    @staticmethod
    def transform_data(
        dm_indices,
        dm_series_list,
        normalise=False,
        barycentring_mode="Fourier",
        beta=0,
        padded_length=1048576,
        nbit=32,
        remove_rednoise=True,
        clean_rfi=True,
        rednoise_config=dict(b0=10, bmax=100000),
        zero_replace=True,
        bad_freq_indices=[],
        shared_target_name=None,
        target_shape=None,
    ):
        """
        Create the power spectrum from a dedispersed time series.

        The static method to run the FFT to create the power spectra from dedispersed
        time series in a parallel loop. A static function that does not depend on any
        other function in the object is required to run a multiprocessing task. This is
        due to multiprocessing will pickle the function to spread out the jobs to
        different cores, and a function that relies on other functions in the class
        object will not be pickle-able.

        Parameters
        ----------
        dts: np.ndarray
            The numpy array containing the dedispersed time series

        normalise: bool
            Whether to normalise the time series prior to FFT. Default = False

        barycentring_mode: string
            Which barycentring mode to use.
            Available: ["Time", "Fourier", "Topocentric"]  Default = "Fourier"

        beta: float
            The barycentric velocity correction value. Default = 0

        padded_length: int
            The padded length of the dedispersed time series. Default = 1048576

        nbit: int
            The number of bits of the resultant power spectrum. Default = 32

        remove_rednoise: bool
            Whether to run rednoise removal on the power spectrum produced. Default = True

        rednoise_config: dict
            Configuration to run the rednoise removal on the power spectrum produced.
            Default = dict(b0=10, bmax=100000)

        clean_rfi: bool
            Whether to mask the bad frequencies found by the periodic RFI mitigation
            process from the power spectrum.
            Default = True

        zero_replace: bool
            Whether to mask the bad frequencies with zeroes. Default = True

        bad_freq_indices: List[int]
            The list of the frequency bins that are to be masked. Default = []

        Returns
        -------
        power_spectrum: np.ndarray
            The power spectrum form from the FFT of the dedispersed time series.
        """
        shared_spectra = shared_memory.SharedMemory(name=shared_target_name)
        power_spectra = np.ndarray(
            target_shape, dtype=f"float{nbit}", buffer=shared_spectra.buf
        )

        all_medians = []
        all_scales = []

        for dm_index, dts in zip(dm_indices, dm_series_list):
            if normalise:
                ts_mean = np.mean(dts)
                ts_std = np.std(dts)
                dts = (dts - ts_mean) / ts_std
            if barycentring_mode == "Time":
                # barycenter the time series before forming the power spectrum
                dts = barycenter_timeseries(dts, beta)
            n_to_pad = int(padded_length - dts.size)
            if normalise:
                dts = np.pad(dts, (0, n_to_pad), mode="constant", constant_values=0)
            else:
                dts = np.pad(dts, (0, n_to_pad), mode="median", stat_length=4096)

            log.debug("Computing power spectrum and corresponding frequency labels")
            spectrum = pyfftw.interfaces.numpy_fft.rfft(dts)
            spectrum[0] = 0
            spectrum = spectrum[:-1]
            power_spectrum = np.abs(spectrum) ** 2
            power_spectrum = power_spectrum.astype(f"float{nbit}")

            if barycentring_mode == "Fourier":
                # target labels are the frequencies we want our label to have
                # initial labels are the barycentric frequencies that are actually recorded
                # only the relation between the labels is used so we assume tsamp=1 here
                target_labels = np.arange(len(power_spectrum))
                initial_labels = target_labels * (1 - beta)
                power_spectrum = interp1d(
                    initial_labels,
                    power_spectrum,
                    kind="nearest",
                    bounds_error=False,
                    fill_value=0,
                )(target_labels)

            # normalise power spectrum
            if clean_rfi:
                power_spectrum[bad_freq_indices] = np.nan

            if remove_rednoise:
                log.debug("Normalising power spectrum with rednoise removal")
                power_spectrum[1:], medians, scale = rednoise_normalise(
                    power_spectrum[1:], **rednoise_config
                )
                all_medians.append(medians)
                all_scales.append(scale)

            else:
                log.debug("Normalising power spectrum")
                power_spectrum[1:] = power_spectrum[1:] / (
                    np.nanmedian(power_spectrum[1:]) / np.log(2)
                )
            if clean_rfi:
                if zero_replace:
                    power_spectrum[bad_freq_indices] = 0
                else:
                    power_spectrum[bad_freq_indices] = (
                        np.random.chisquare(2, len(bad_freq_indices)) / 2
                    )

            power_spectra[dm_index, :] = power_spectrum
        shared_spectra.close()

        return [dm_indices, all_medians, all_scales]

    def get_pointing_id_from_observation_id(self, obs_id):
        """Get the corresponding pointing ID from the observation ID."""
        try:
            pointing_id = db_api.get_observation(obs_id).pointing_id
        except Exception as e:
            log.warning(e)
            log.warning("Cannot obtain pointing id, setting it to None")
            pointing_id = None
        return pointing_id

    def normalise_ts(self, dedisp_ts):
        """Function to normalise the dedispersed time series."""
        norm_start = time.time()
        log.info(
            "Normalising time series before transform (0 mean, unit standard deviation)"
        )
        ts_mean = np.mean(dedisp_ts)
        ts_std = np.std(dedisp_ts)
        dedisp_ts = (dedisp_ts - ts_mean) / ts_std
        norm_end = time.time()
        log.debug(f"Took {norm_end - norm_start} seconds to normalize time series")
        return dedisp_ts

    def get_barycentric_correction(self, dedisp_ts_len, ra, dec, start_mjd):
        """
        Calculates the barycentric correction factor beta of the observation.

        Parameters
        ----------
        dedisp_ts_len: int
            The length of the dedipsersed time series

        ra: float
            The right ascension of the observation in degrees

        dec: float
            The declination of the observation in degrees

        start_mjd: float
            The start time of the observation

        Returns:
        =======
        beta: float
            The barycentric correction factor.
        """
        bary_corr_start = time.time()
        log.info("Calculating barycentric velocity correction")
        ras, decs = convert_ra_dec(ra, dec)
        ras = ras[0:2] + ":" + ras[2:4] + ":" + ras[4:]
        if dec < 0:
            decs = decs[0:3] + ":" + decs[3:5] + ":" + decs[5:]
        else:
            decs = decs[0:2] + ":" + decs[2:4] + ":" + decs[4:]
        beta = get_barycentric_correction(
            ras,
            decs,
            start_mjd + (float(dedisp_ts_len * self.tsamp / 2) / SEC_PER_DAY),
        )
        bary_corr_end = time.time()
        log.debug(
            f"Took {bary_corr_end - bary_corr_start} seconds to barycenter time series"
        )
        return beta

    def barycenter_ts(self, dedisp_ts, beta):
        """
        Apply the barycentric correction to the time series.

        Parameters
        ----------
        dedisp_ts: np.ndarray
            The dedispersed time series to be barycentered

        beta: float
            The barycentric correction factor.

        Returns:
        =======
        dedisp_ts: np.ndarray
            1-D numpy array of the barycentered time series
        """
        bary_start = time.time()
        log.info("Applying barycentric correction to time series")
        dedisp_ts = barycenter_timeseries(dedisp_ts, beta)
        bary_end = time.time()
        log.debug(f"Took {bary_end - bary_start} seconds to barycenter time series")
        return dedisp_ts

    def barycenter_ps(self, power_spectrum, beta):
        """
        Use interpolation to create a barycentered power spectrum.

        A nearest neighbour interpolation is performed to go from the
        topocentric FFT bins to the barycentric FFT bins.

        Parameters
        ----------
        power_spectrum: np.ndarray
            The topocentric power spectrum to be barycentered

        beta: float
            The barycentric correction factor.

        Returns:
        =======
        power_spectrum_interpolated: np.ndarray
            1-D numpy array of the barycentered power spectrum
        """
        # This function could be moved so sps_common to be used in transform_data

        target_labels = np.arange(len(power_spectrum))
        initial_labels = target_labels * (1 - beta)
        power_spectrum_interpolated = interp1d(
            initial_labels,
            power_spectrum,
            kind="nearest",
            bounds_error=False,
            fill_value=0,
        )(target_labels)
        return power_spectrum_interpolated

    def zeropad_ts(
        self, dedisp_ts, desired_size: int = 1048576, is_normalised: bool = True
    ):
        """
        Zero pad (the end) of a time series to the given desired time.

        This will allow for Fourier interpolation when computing the power spectrum.

        Parameters
        ----------
        desired_size: int
            The desired size of the padded time series (ideally a integer with
            only 2, 3 or 5 as factors). Default: 2**20 = 1048576

        is_normalised: bool
            Whether the input time series is normalised. Default = True
        """
        zeropad_start = time.time()
        log.info(f"Zero-padding time series to size = {desired_size}")
        n_to_pad = int(desired_size - dedisp_ts.size)
        if is_normalised:
            dedisp_ts = np.pad(
                dedisp_ts, (0, n_to_pad), mode="constant", constant_values=0
            )
        else:
            dedisp_ts = np.pad(
                dedisp_ts, (0, n_to_pad), mode="median", stat_length=4096
            )
        zeropad_end = time.time()
        log.debug(
            f"Took {zeropad_end - zeropad_start} seconds to zero-pad the time series"
        )
        return dedisp_ts

    def update_database(
        self,
        observation,
        power_spectrum,
        bad_freq_indices,
        beta,
        barycentring_mode,
        barycentric_cleaning,
        compared_obs,
        birdies_inf,
        medians_path,
        medians_path_dm0,
    ):
        """Prepare and deliver the payload to the observation database."""

        if self.run_dynamic_filter:
            birdie_path = f"{os.path.abspath(observation.datapath)}/birdie_info.npz"
            np.savez(
                birdie_path,
                birdies=bad_freq_indices,
                birdies_position=birdies_inf.position,
                birdies_height=birdies_inf.height,
                birdies_left_freq=birdies_inf.left,
                birdies_right_freq=birdies_inf.right,
            )
            log.info(f"Wrote out birdie info to {birdie_path}")
        else:
            birdie_path = None

        payload = dict(
            mean_power=float(power_spectrum.mean()),
            std_power=float(power_spectrum.std()),
            birdies=None,
            birdies_position=None,
            birdies_height=None,
            birdies_left_freq=None,
            birdies_right_freq=None,
            birdie_file=birdie_path,
            beta=beta,
            barycentring_mode=barycentring_mode,
            barycentric_cleaning=barycentric_cleaning,
            compared_obs=compared_obs,
            rn_file=medians_path,
            dm0_rn_file=medians_path_dm0,
        )
        db_api.update_observation(observation._id, payload)

    def run_rfft(self, dedisp_ts):
        """
        Function to run real fft on a dedispsersed time series.

        After calculating the FFT the zero-th
        bin of the fourier series is removed.

        Parameters
        =======
        dedisp_ts: np.ndarray
            1-D numpy array of a dedispersed time series

        Returns:
        =======
        spectrum: np.ndarray
            1-D numpy array of the fourier transform of the time series
        """
        spectrum = pyfftw.interfaces.numpy_fft.rfft(dedisp_ts)
        spectrum[0] = 0
        return spectrum[:-1]

    def get_freq_labels(self):
        """
        Get the topocentric frequency labels of the power spectra.

        Returns:
        =======
        freq_labels: np.ndarray
            1-D numpy array of the frequency labels of the topocentric power spectra
        """
        freq_labels = np.fft.rfftfreq(self.padded_length, d=self.tsamp)
        return freq_labels[:-1]

    def get_barycentric_freq_labels(self, beta):
        """
        Get the baryenctric frequency labels of the power spectra based on a given beta.

        Parameters
        =======
        beta: float
            The barycentric correction factor

        Returns:
        =======
        freq_labels: np.ndarray
            1-D numpy array of the frequency labels of the barycentric power spectra
        """
        freq_labels = np.fft.rfftfreq(self.padded_length, d=self.tsamp * (1 + beta))
        return freq_labels[:-1]

    def flag_periodic_rfi(self, dedisp_time_series, freq_labels, beta, observation):
        """
        Flag periodic RFI by comparing birdies with adjacent observations.

        Parameters
        =======
        dedisp_time_series: DedispersedTimeSeries
            The DedispersedTimeSeries interface storing the data and properties
            of the dedispersed time series to process.

        freq_labels: np.ndarray
            1-D numpy array of frequency labels for each power spectrum bin.

        beta: float
            The barycentric correction factor.

        Returns:
        =======
        bad_freq_indices: list
            List of power spectrum bins that correspond to known RFI signals
        compared_obs: List[str]
            The observation id's that are used for the comparison
        birdies: np.recarray
            Birdie infromation sa returned by the DynamicPeriodicFilter
        """
        bad_freq_indices = []
        if self.run_static_filter and self.barycentring_mode != "Topocentric":
            log.info("Running static periodic signal filter")
            bad_freq_indices = self.static_filter.apply_static_mask(freq_labels, beta)
        elif self.run_static_filter and self.barycentring_mode == "Topocentric":
            log.info("Running static periodic signal filter")
            bad_freq_indices = self.static_filter.apply_static_mask(freq_labels, 0)

        # Periodic RFI mitigation
        compared_obs = []
        medians_path = None
        if self.run_dynamic_filter:
            log.info("Running dynamic periodic signal filter")
            zero_dm_ts = dedisp_time_series.dedisp_ts[0]
            if self.normalise:
                zero_dm_ts = self.normalise_ts(zero_dm_ts)
            if self.barycentring_mode == "Time" and self.barycentric_cleaning:
                zero_dm_ts = barycenter_timeseries(zero_dm_ts, beta)
            zero_dm_ts = self.zeropad_ts(zero_dm_ts, self.padded_length, self.normalise)
            zero_dm_spectrum = self.run_rfft(zero_dm_ts)
            zero_dm_power_spectrum = np.abs(zero_dm_spectrum) ** 2
            zero_dm_power_spectrum[1:], medians, scales = rednoise_normalise(
                zero_dm_power_spectrum[1:], **self.rednoise_config
            )
            if self.barycentring_mode == "Fourier" and self.barycentric_cleaning:
                # Should this used before the rednoise normalisation?
                zero_dm_power_spectrum = self.barycenter_ps(
                    zero_dm_power_spectrum, beta
                )
            # calling periodic filter function to compute birdies for the current pointing
            (
                birdies,
                strong_birdies,
            ) = self.dynamic_filter.excise_by_identifying_outliers(
                zero_dm_power_spectrum,
                freq_labels,
                dedisp_time_series.obs_id,
                red_noise_fit=False,
                debug=False,
                barycentric_cleaning=self.barycentric_cleaning,
                beta=beta,
            )
            if self.write_zero_dm_medians:
                medians_path = (
                    f"{os.path.abspath(observation.datapath)}/zero_dm_medians.npz"
                )
                log.info(f"Saving zero-DM medians to {medians_path}.")
                np.savez(
                    medians_path,
                    medians=medians,
                    scales=scales,
                )
            # if self.update_db == False, birdies are not recorded into database
            if self.find_common_birdies and self.update_db:
                log.info(
                    "Retrieving common birdies for observation and its surroundings"
                    " from database"
                )
                (
                    left_freqs,
                    right_freqs,
                    compared_obs,
                ) = self.common_birdies(observation, freq_labels[1], birdies)
            else:
                log.info(
                    "Using the dynamic birdies list generated for the observation on"
                    " its own"
                )
                left_freqs, right_freqs = (
                    birdies["left"],
                    birdies["right"],
                )
                compared_obs = []
            if self.barycentring_mode == "None":
                bary_left_freqs = left_freqs
                bary_right_freqs = right_freqs
            else:
                bary_left_freqs = bary_from_topo_freq(left_freqs, beta)
                bary_right_freqs = bary_from_topo_freq(right_freqs, beta)

            common_birdies = []
            for blf, brf in zip(bary_left_freqs, bary_right_freqs):
                common_birdies.extend(
                    np.where((freq_labels >= blf) & (freq_labels <= brf))[0].tolist()
                )
            log.info(
                "The final common birdies list masks"
                f" {len(common_birdies) / len(freq_labels):.6f} of the data"
            )

            strong_birdies_left_freqs, strong_birdies_right_freqs = (
                strong_birdies["left"],
                strong_birdies["right"],
            )
            if not self.barycentric_cleaning:
                strong_bary_left_freqs = bary_from_topo_freq(
                    strong_birdies_left_freqs, beta
                )
                strong_bary_right_freqs = bary_from_topo_freq(
                    strong_birdies_right_freqs, beta
                )
            else:
                strong_bary_left_freqs = strong_birdies_left_freqs
                strong_bary_right_freqs = strong_birdies_right_freqs
            strong_periodic_rfi = []
            for blf, brf in zip(strong_bary_left_freqs, strong_bary_right_freqs):
                strong_periodic_rfi.extend(
                    np.where((freq_labels >= blf) & (freq_labels <= brf))[0].tolist()
                )
            bad_freq_indices = sorted(set(bad_freq_indices).union(strong_periodic_rfi))
            bad_freq_indices = sorted(set(bad_freq_indices).union(common_birdies))
        else:
            birdies = None

        return bad_freq_indices, compared_obs, birdies, medians_path

    def compare_birdies(
        self,
        birdies,
        birdies_left,
        birdies_right,
        other_birdies_left,
        other_birdies_right,
        no_width=False,
    ):
        """
        Compare birdies to adjacent observations.

        Compare the list of birdies from a single observations with the birdies lists
        from its surrounding pointings and outputs a boolean array with the indices of
        the subset of the birdies list that are common to a given fraction of the the
        birdies lists given.

        Parameters
        =======
        birdies: List[int]
            List of frequency indices of the birdies of the observation to be compared.

        birdies_left: List[int]
            List of frequency indices of the left flank of the birdies of the
            observation to be compared.

        birdies_right: List[int]
            List of frequency indices of the right flank of the birdies of the
            observation to be compared.

        other_birdies_left: List[List[int]]
            2D List of frequency indices of the left flank of the birdies from the
            surrounding pointings of the observation of interest.

        other_birdies_right: List[List[int]]
            2D List of frequency indices of the right flank of the birdies from the
            surrounding pointings of the observation of interest.

        no_width: bool
            Only use the peak positions of the observation to be compared

        Returns
        =======
        bool_arr: np.ndarray
            Boolean Array that show which of the birdies of the current observation
            are common in the surrounding observations.
        """
        target_array = np.zeros(self.padded_length // 2)
        index_array = np.arange(self.padded_length // 2)
        for left_obs, right_obs in zip(other_birdies_left, other_birdies_right):
            current_target_array = np.zeros(self.padded_length // 2)
            for left, right in zip(left_obs, right_obs):
                current_target_array[left : right + 1] = 1
            target_array += current_target_array

        if no_width:
            # This only uses the peak position without considering the width of the
            # birdies in the current observation.
            count_without_own_width = target_array[birdies]
            return (
                count_without_own_width
                >= len(other_birdies_left) * self.common_birdie_fraction
            )
        else:
            counter_arr = np.zeros(len(birdies))
            for i in range(len(birdies)):
                birdie_range = target_array[birdies_left[i] : birdies_right[i]]
                try:
                    max_val = np.max(birdie_range)
                    counter_arr[i] = max_val
                except ValueError:
                    log.info(
                        f"ValueError with birdies between indices {birdies_left[i]} :"
                        f" {birdies_right[i]};  max_length : {len(target_array)}"
                    )
            return counter_arr >= len(other_birdies_left) * self.common_birdie_fraction

    def common_birdies(self, observation, freq_spacing, birdies):
        """
        Retrieve common birdies with adjacent observations.

        Find the common birdies to the birdies list computed by DynamicPeriodicFilter in
        this observation with other observations within an RA, Dec range.

        Parameters
        =======
        observation_id: str
            The observation ID of the data being processed.

        freq_spacing: float
            The frequency spacing between adjacent frequency indices in the power spectrum.

        Returns
        =======
        birdies_left_freq_list: np.ndarray
            The topocentric frequency of the leftmost frequency of each birdies.

        birdies_right_freq_list: np.ndarray
            The topocentric frequency of the rightmost frequency of each birdies.

        compared_obs: List[str]
            The observation id's that are used for the comparison
        """
        birdies_array_list = birdies.position
        if not len(birdies_array_list):
            log.info(
                f"The observation {observation.id} has no birdies."
                "Returning an empty birdies list"
            )
            return np.array([]), np.array([]), []
        birdies_left_freq_list = np.asarray(birdies.left)
        birdies_right_freq_list = np.asarray(birdies.right)

        # surround_observations = db_api.get_observations_around_observation(
        #    observation,
        #    self.common_birdies_config["ra_range"],
        #    self.common_birdies_config["dec_range"],
        # )
        surround_observations = db_api.get_observations_before_observation(
            observation, 0, 7, ra_range=1
        )

        compared_obs = []
        surrounding_birdies_left = []
        surrounding_birdies_right = []

        for observ in surround_observations:
            if observation._id != observ._id:
                if observ.birdies_position is not None:
                    if len(observ.birdies_position) > 0:
                        compared_obs.append(observ._id)
                        current_birdies_left_freqs = np.asarray(
                            observ.birdies_left_freq
                        )
                        current_birdies_right_freqs = np.asarray(
                            observ.birdies_right_freq
                        )
                        current_birdies_left_indices = np.floor(
                            current_birdies_left_freqs / freq_spacing
                        ).astype(int)
                        current_birdies_right_indices = np.ceil(
                            current_birdies_right_freqs / freq_spacing
                        ).astype(int)
                        surrounding_birdies_left.append(current_birdies_left_indices)
                        surrounding_birdies_right.append(current_birdies_right_indices)

        if not surrounding_birdies_left:
            log.info(
                "There are no birdies to compare with the observation"
                f" {observation._id}. Returning the birdies left and right frequencies"
                " of the current observation"
            )
            return birdies_left_freq_list, birdies_right_freq_list, compared_obs

        log.info(f"Using {len(compared_obs)} observations for birdie comparison.")

        birdies_left_indices = np.floor(birdies_left_freq_list / freq_spacing).astype(
            int
        )
        birdies_right_indices = np.ceil(birdies_right_freq_list / freq_spacing).astype(
            int
        )
        bool_array = self.compare_birdies(
            birdies_array_list,
            birdies_left_indices,
            birdies_right_indices,
            surrounding_birdies_left,
            surrounding_birdies_right,
        )

        return (
            birdies_left_freq_list[bool_array],
            birdies_right_freq_list[bool_array],
            compared_obs,
        )
