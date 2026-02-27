#!/usr/bin/env python3
import logging

import numpy as np
from pyfftw.interfaces.numpy_fft import rfft
from rfi_mitigation.utilities.cleaner_utils import (
    generalised_spectral_kurtosis,
    get_gaussian_gsk_bounds,
    get_pdf_transforms,
    median_absolute_deviation,
)
from rfi_mitigation.utilities.noise_utils import baseline_estimation_medfilt
from scipy.stats.mstats import kurtosis
from sps_common.constants import L0_NCHAN, L1_NCHAN, TSAMP

log = logging.getLogger(__name__)


class Cleaner:
    """This class is a superclass for other cleaners, collecting the common attributes
    among all methods.
    """

    def __init__(self, spectra_shape):
        # self.spectra = spectra  # assumes shape is (nchan, ntime)
        self.nchan = spectra_shape[0]
        self.ntime = spectra_shape[1]
        self.nsamp = spectra_shape[0] * spectra_shape[1]
        self.cleaner_mask = np.zeros(spectra_shape, dtype=bool)
        self.cleaned = False

    def get_mask(self):
        if self.cleaned:
            return self.cleaner_mask
        else:
            log.error("Cleaner was not executed - mask is empty!")
            return self.cleaner_mask

    def get_masked_fraction(self):
        if self.cleaned:
            return self.cleaner_mask.sum() / self.nsamp
        else:
            log.error("Cleaner was not executed - mask is empty!")
            return self.cleaner_mask.sum() / self.nsamp


class DummyCleaner(Cleaner):
    """This class is a dummy cleaner that does nothing except return a binary mask that
    masks no data.
    """

    def __init__(self, spectra: np.ndarray):
        super().__init__(spectra)

    def clean(self):
        log.info("Running Dummy cleaner")
        self.cleaned = True

    def summary(self):
        return dict(
            nchan=self.nchan,
            ntime=self.ntime,
            nsamp=self.nsamp,
            nmasked=self.cleaner_mask.sum(),
            cleaned=self.cleaned,
        )


class MedianAbsoluteDeviationCleaner(Cleaner):
    """
    This class accepts a spectra and detects RFI along the frequency axis by computing
    the local median and median absolute deviation (MAD) and rejecting samples outside
    the nominated thresholds.

    This helps to recognize data corruption and/or missing data caused by packet loss or
    L0/L1 GPU node failures. It will also pick up on RFI tha other methods may miss.
    """

    def __init__(
        self, spectra: np.ndarray, threshold: float = 4.0, stat_window_size: int = 64
    ):
        super().__init__(spectra)
        self.threshold = threshold
        self.stat_window_size = stat_window_size

    def summary(self):
        return dict(
            nchan=self.nchan,
            ntime=self.ntime,
            nsamp=self.nsamp,
            threshold=self.threshold,
            stat_window_size=self.stat_window_size,
            nmasked=self.cleaner_mask.sum(),
            cleaned=self.cleaned,
        )

    def clean(self):
        log.info("Running Median Absolute Deviation cleaner")
        self.clean_freq()
        self.clean_time()
        self.clean_mean_spectrum()
        self.cleaned = True

    def clean_freq(self):
        step = 1024
        current_stat_window = self.stat_window_size

        # run the MAD cleaner on multiple channel scales
        log.debug(f"cleaner window = {current_stat_window} channels")
        for i in range(self.ntime // step):
            # pre-determine the time-index slice
            tsl = slice(i * step, (i + 1) * step)
            intensity = np.ma.mean(self.spectra[:, tsl], axis=1)

            for j in range(
                current_stat_window,
                self.nchan + current_stat_window,
                current_stat_window,
            ):
                # pre-determine the frequency-index slice
                upper_index = j if j < self.nchan else self.nchan
                lower_index = (
                    j - current_stat_window if (j - current_stat_window) > 0 else 0
                )
                fsl = slice(lower_index, upper_index)
                spec_chunk = intensity[fsl]

                # calculate the median absolute deviation as a proxy for standard
                # deviation
                spec_chunk_med, spec_chunk_std = median_absolute_deviation(spec_chunk)

                # filter out samples outside the nominal threshold defined by thresh
                pval = spec_chunk_med + self.threshold * spec_chunk_std
                nval = spec_chunk_med - self.threshold * spec_chunk_std
                chunk_mask = np.logical_or(spec_chunk > pval, spec_chunk < nval)

                # update the entries of the final data mask
                self.cleaner_mask[fsl, tsl] = chunk_mask[:, np.newaxis]

    def clean_time(self):
        step = 1024

        for i in range(self.ntime // step):
            log.debug(f"cleaning time slice {i}")
            # pre-determine the time-index slice
            tsl = slice(i * step, (i + 1) * step)
            intensity = np.ma.mean(self.spectra[:, tsl], axis=0)

            # calculate the median absolute deviation as a proxy for standard deviation
            time_chunk_med, time_chunk_std = median_absolute_deviation(intensity)

            # filter out samples outside the nominal threshold defined by thresh
            pval = time_chunk_med + self.threshold * time_chunk_std
            nval = time_chunk_med - self.threshold * time_chunk_std
            chunk_mask = np.logical_or(intensity > pval, intensity < nval)

            # update the entries of the final data mask
            self.cleaner_mask[:, tsl] = chunk_mask[np.newaxis, :]

    def clean_mean_spectrum(self, pthresh: float = 3.0, nthresh: float = 1.5):
        # calculate the mean spectrum and the MAD from that collapsed array
        mean_spec = np.ma.mean(self.spectra, axis=-1)
        mean_spec_med, mean_spec_std = median_absolute_deviation(mean_spec)

        # filter samples outside the defined thresholds
        pval = mean_spec_med + pthresh * mean_spec_std
        nval = mean_spec_med - nthresh * mean_spec_std
        mean_spec_mask = np.logical_or(mean_spec > pval, mean_spec < nval)

        # combine the mean spectrum mask with the individual chunk mask
        self.cleaner_mask = np.logical_or(
            self.cleaner_mask, mean_spec_mask[:, np.newaxis]
        )

        # also mask channels where std. dev. over full time slice is 0 (not physical)
        # TODO: does this hold when we use the 3-bit data though?
        self.cleaner_mask = np.logical_or(
            self.cleaner_mask, (np.ma.std(self.spectra, axis=1) == 0)[:, np.newaxis]
        )


class KurtosisCleaner(Cleaner):
    """
    This class accepts a spectra and detects RFI along the frequency axis by computing
    the (time) local kurtosis in each channel and rejecting samples outside the
    nominated percentile.

    This cleaning method flags entire channels (i.e. does not make use of the time
    resolution).
    """

    def __init__(
        self, spectra: np.ndarray, percentile: float = 99.0, stat_window_size: int = 128
    ):
        super().__init__(spectra)
        self.percentile = percentile
        self.stat_window_size = stat_window_size

    def summary(self):
        return dict(
            nchan=self.nchan,
            ntime=self.ntime,
            nsamp=self.nsamp,
            percentile=self.percentile,
            stat_window_size=self.stat_window_size,
            nmasked=self.cleaner_mask.sum(),
            cleaned=self.cleaned,
        )

    def clean(self):
        log.info("Running temporal Kurtosis cleaner")
        self.clean_chunkwise()
        self.cleaned = True

    def clean_chunkwise(self):
        for j in range(
            self.stat_window_size,
            self.ntime + self.stat_window_size,
            self.stat_window_size,
        ):
            # pre-determine the frequency-index slice
            upper_index = j if j < self.ntime else self.ntime
            lower_index = (
                j - self.stat_window_size if (j - self.stat_window_size) > 0 else 0
            )
            tsl = slice(lower_index, upper_index)
            spec_chunk = self.spectra[:, tsl] - np.ma.mean(
                self.spectra[:, tsl], axis=0
            )  # subtract baseline mean
            # the baseline subtraction also means that the kurtosis filter is sensitive
            # to dropouts from
            # L0 packet-loss/GPU node crashes

            # calculate the kurtosis along local time range
            chunk_kur = kurtosis(spec_chunk, bias=True, fisher=True, axis=1)
            # k-stats are used to counteract biased estimators, and Fisher's definition
            # is used (i.e. subtract 3 from the usual definition, so normal-distributed
            # data has a kurtosis of 0)

            chunk_kur_med = np.ma.median(chunk_kur)
            chunk_kur_percentile = np.percentile(chunk_kur, self.percentile)

            # filter out samples outside the nominal threshold defined by thresh
            # NOTE: nominally, a kurtosis outside [-1, 1] indicates significant
            # "tailedness" of a distribution, but in this case, excluding values that
            # exceed the median +/- the 99.9-th percentile should be an appropriate cut
            # to remove extreme values
            pval = chunk_kur_med + chunk_kur_percentile
            nval = chunk_kur_med - chunk_kur_percentile
            chunk_mask = np.logical_or(chunk_kur > pval, chunk_kur < nval)

            # update the entries of the final data mask
            self.cleaner_mask[:, tsl] = chunk_mask[:, np.newaxis]

        # TODO: we could also add a threshold here that says if a channel is more than
        # X% masked, just mask all of it
        freq_mask_sum = self.cleaner_mask.sum(axis=1)
        self.cleaner_mask[np.where(freq_mask_sum >= 0.75 * self.ntime), :] = True


class SpectralKurtosisCleaner(Cleaner):
    """
    This class accepts a spectra and detects RFI corrupted channels by calculating the
    generalized spectral kurtosis (GSK). It is best suited to identification of narrow-
    band impulsive RFI, especially as it uses a median filter approach to remove the
    non-uniform baseline of the GSK statistics in order to detect outliers.

    Here, we discuss some details of the algorithm and choices made in the case of
    CHIME/SPS data. The GSK requires 3 parameters in addition to the intensity data:

    - The scale, `M`, is the number of time samples that are summed together to form the
    GSK estimator. This is notionally 1024, corresponding to ~1 second.

    - The number of onboard integration, `N`, is more precisely described as the number
    of squared Gaussian variates being summed to reach the intensity data.

    - The data distribution shape, denoted as `delta`, is a combination of N and the
    corresponding Gamma distribution shape factor, d, that describes the data.
    Ultimately, it can be considered a "fudge factor" that accounts for arbitrary
    divergences from ideal Gamma distributions (which is the case for quantized data
    and data that is not independent from sample-to-sample).

    Specifically for CHIME/SPS intensity data there are some important details regarding
    the evaluation of N, and ultimately, delta. In an email communication between
    B. Meyers and G. Hellbourg on 2021-03-10:
        - Going from complex voltage data to total intensity involves 4 squared
        samples for each time sample (2 Re/Im and 2 pols).
        - Are you working with coherently or incoherently beamformed data? If it's the
        latter, your N-term would also have to be multiplied by the number of antennas
        summed together. (We are coherently beamforming, thus this factor is 1.)
        - You probably average frequency channels together when you downsample in
        frequency. In that case, you would also have to multiply N by the numbers of
        channels averaged together.

        CHIME/FRB is a slightly special case in that it up-samples the native 1024 L0
        channels to 16384 channels via a FFT method. Specifically, this is described on
        pg. 5 of the system overview paper, which can be found here:
        (https://iopscience.iop.org/article/10.3847/1538-4357/aad188)

        Beamforming occurs after a coarse PFB of 1024 channels. The 1024 x 1024
        beamformed channels are then fed into a fine PFB to increase the frequency
        resolution. The end products are then 1024 x 16k beamformed spectra with 1 ms,
        24.4 kHz resolutions. For each frequency bin of these 16k channel spectra we
        have:

            N = 2 Re/Im * 2 pols * 8 averaged channels * 3 averaged spectra = 96

        If frequency downsampling occurs, you would need to multiply this N by the
        downsample factor. i.e., if we downsampled to 1024 channels, N = 96 * 16 = 1536.

    In terms of estimating the data distribution shape parameters, we are certainly not
    in a regime where analytic values can be used (i.e. delta = 1 if working with
    voltages). To that end, one can estimate the true data shape from the data
    (technically, the first two moments of the data distribution) by selecting a
    typically-good region of the spectrum. For that, we can use the function

        delta = generalised_sk_shape_from_data(
            spec_chunk, good_chan_range[0], good_chan_range[1]
        )

    A more robust approach would be to first flag very clear SK outlier values and
    then compute the median SK estimator value across the full band. We can then
    use this data-driven measurement to rescale the original SK array (rather than
    completely recomputing, which is expensive).
    """

    def __init__(
        self,
        spectra_shape,
        scales=None,
        rfi_threshold_sigma=None,
        plot_diagnostics: bool = False,
    ):
        super().__init__(spectra_shape)

        # set the scale on which the SK cleaner operates
        if scales is None:
            self.scales = [1024]
        elif isinstance(scales, list):
            self.scales = scales
        elif isinstance(scales, int):
            self.scales = [scales]
        else:
            log.warning(
                "Did not understand requested `scales` parameter for SK cleaner - "
                "assuming 1024 time samples!"
            )
            self.scales = [1024]
        self.nscales = len(self.scales)

        # Set the RFI threshold in sigma
        if rfi_threshold_sigma is None:
            self.eta = 25
        else:
            self.eta = rfi_threshold_sigma
        log.debug(f"SK RFI threshold = {self.eta}-sigma equiv.")

        # set per-chunk masked fraction threshold, above which all channels from the
        # M-sample chunk are masked
        self.chunk_mask_frac_threshold = 0.75
        log.debug(
            "Masking fraction threshold to flag entire chunk ="
            f" {self.chunk_mask_frac_threshold}"
        )

        # define a channel range where, usually, there is little-to-no RFI (this is only
        # used to flag really obvious outlier prior to the SK is renormalised)
        self.good_chan_upper = 730
        self.good_chan_lower = 530
        self.freq_scale_factor = self.nchan / 2048
        self.good_chan_range = slice(
            int(self.good_chan_lower * self.freq_scale_factor),
            int(self.good_chan_upper * self.freq_scale_factor),
        )
        log.debug(f"Good channel range (slice) = {self.good_chan_range}")

        # whether to produce plots at different stages of the clean to check validity
        # of SK and median filtering
        self.plot_diagnostics = plot_diagnostics

    def summary(self):
        return dict(
            nchan=self.nchan,
            ntime=self.ntime,
            nsamp=self.nsamp,
            scales=self.scales,
            nscales=self.nscales,
            nmasked=self.cleaner_mask.sum(),
            cleaned=self.cleaned,
        )

    def clean(self, spectra, rfi_mask):
        log.info("Running Spectral Kurtosis cleaner")
        self.clean_all_scales(spectra, rfi_mask)
        self.cleaned = True

    def clean_all_scales(self, spectra, rfi_mask):
        for s in self.scales:
            log.info(f"scale, M = {s} ({self.ntime // s} chunks)")
            self.clean_single_scale(s, spectra, rfi_mask)

    def clean_single_scale(self, scale: int, spectra, rfi_mask):
        n_chan_scrunch = L1_NCHAN / self.nchan
        log.debug(f"Channel downsample factor: {n_chan_scrunch}")
        for ichunk in range(self.ntime // scale):
            start_idx = ichunk * scale
            end_idx = (ichunk + 1) * scale
            spec_chunk = spectra[:, start_idx:end_idx]
            current_mask_frac = rfi_mask[:, start_idx:end_idx].mean()

            # generalized spectral kurtosis estimator
            # (Nita, Gary & Hellbourg 2017, eq. 5)
            # Compute with delta=1 (untrue) but allows for us to re-estimate the
            # correct shape
            d = 1
            n = 96 * n_chan_scrunch
            delta = d * n
            spec_sk = generalised_spectral_kurtosis(spec_chunk, m=scale, delta=delta)

            scl, loc, a, b = get_pdf_transforms(scale, n, d, ret_ab=True)
            # scale the SK spectrum based on the moments
            spec_sk = spec_sk * scl + loc

            # At this point we have two options:
            # 1) Just use the median value as the baseline, since we cannot be sure that
            #    any resduals structure is not due to RFI (and it should therefore be
            #    considered to excision)
            bline = np.nanmedian(spec_sk[self.good_chan_range])
            log.debug(f"SK baseline = {bline}")

            # 2) Run a median filter across the SK band to flatten things out, which is
            #    very expensive and assumes that any baseline structure is instrumental and
            #    not cause by RFI. This also requires careful tuning of the median filter
            #    window size.
            # bline = baseline_estimation_medfilt(
            #     flagged_spec_sk.filled(flagged_spec_sk_med),
            #     k=int(0.5 * self.median_filter_size),
            # )

            # We can also now estimate the RFI bounds. The choice of eta here essentially
            # increases the range of acceptable values. eta=25 seems to work for our data
            # which is highly quantized and therefore doesn't play as nicely with the standard
            # assumed Gaussian statisitcs underlying the approach. eta is basically equivalent
            # to Gaussian "sigma".
            log.debug(f"scale={scale} delta={delta}")
            lower, upper = get_gaussian_gsk_bounds(scale, n, d, eta=self.eta)
            # offset the lower/upper bounds as they are assumed to be centred at unity,
            # and our SK values are not.
            lower += bline - 1
            upper += bline - 1
            log.debug(f"SK RFI bounds ({self.eta}-sigma): lower={lower}  upper={upper}")

            if self.plot_diagnostics:
                import matplotlib.pyplot as plt

                fig = plt.figure()
                plt.plot(spec_sk * scl + loc)
                plt.axhline(bline, ls="--", color="k")
                plt.axhline(lower, ls=":", color="k")
                plt.axhline(upper, ls=":", color="k")
                plt.axvline(
                    self.good_chan_lower * self.freq_scale_factor, ls=":", color="k"
                )
                plt.axvline(
                    self.good_chan_upper * self.freq_scale_factor, ls=":", color="k"
                )
                plt.ylim(0, 3 * np.nanmedian(spec_sk.data))
                plt.savefig(f"raw_sk_chunk{ichunk}.png", bbox_inches="tight")
                plt.close(fig)

            # create RFI mask from GSK estimator
            sk_mask = np.logical_or(spec_sk > upper, spec_sk < lower)

            n_above = np.sum(spec_sk > upper)
            n_below = np.sum(spec_sk < lower)
            log.debug(f"Number of channels excised: above={n_above}  below={n_below}")

            total_chunk_mask_frac = sk_mask.mean() + current_mask_frac
            # diff_mask_frac = total_chunk_mask_frac - current_mask_frac
            if total_chunk_mask_frac > self.chunk_mask_frac_threshold:
                log.warning(
                    f"More than {self.chunk_mask_frac_threshold * 100}% of channels are"
                    f" bad! Flagging entire band for these {scale} samples."
                )
                sk_mask = np.ones_like(sk_mask)

            # if requested, make some diagnostic plots for this data chunk
            if self.plot_diagnostics:
                from rfi_mitigation.utilities.cleaner_utils import (
                    plot_diagnostic_delta,
                    plot_diagnostic_masked_sk,
                    plot_diagnostic_masked_spectrum,
                )

                # plot_diagnostic_baseline(flagged_spec_sk, bline, scale, ichunk)
                plot_diagnostic_delta(delta, ichunk)
                plot_diagnostic_masked_sk(
                    spec_sk, sk_mask, lower, upper, scale, np.mean(delta), ichunk
                )
                plot_diagnostic_masked_spectrum(spec_chunk, sk_mask, ichunk)

            # update the mask chunk-wise
            self.cleaner_mask[:, start_idx:end_idx] = sk_mask[:, np.newaxis]


class StdDevChannelCleaner(Cleaner):
    """
    This class flags channels based on anomalous standard deviation values computed
    across the full beamformed pointing. This helps identify RFI that varies on
    longer timescales than the typical 1-second blocks.

    The cleaner first normalizes each channel by its time-averaged mean to remove
    bandpass structure, then computes the standard deviation across time. Channels
    with anomalously high or low normalized standard deviations are flagged.
    """

    def __init__(
        self,
        spectra_shape,
        threshold: float = 5.0,
    ):
        """
        Initialize the StdDevChannelCleaner.

        Parameters
        ----------
        spectra_shape : tuple
            Shape of the spectra (nchan, ntime)
        threshold : float
            Number of MAD units beyond which channels are flagged.
            Default: 5.0
        """
        super().__init__(spectra_shape)
        self.threshold = threshold

    def summary(self):
        return dict(
            nchan=self.nchan,
            ntime=self.ntime,
            nsamp=self.nsamp,
            threshold=self.threshold,
            nmasked=self.cleaner_mask.sum(),
            cleaned=self.cleaned,
        )

    def clean(self, spectra_shared_name, mask_shared_name, spectra_shape, spec_dtype, time_bin=1024):
        """
        Flag channels based on anomalous standard deviation across full pointing.

        Parameters
        ----------
        spectra_shared_name : str
            Name of shared memory for spectra
        mask_shared_name : str
            Name of shared memory for mask
        spectra_shape : tuple
            Shape of the spectra (nchan, ntime)
        spec_dtype : dtype
            Data type of spectra
        time_bin : int
            Number of time samples to average together. Default: 1024
        """
        from multiprocessing import shared_memory

        log.info("Running Standard Deviation Channel cleaner")

        # Access shared memory
        shared_spectra = shared_memory.SharedMemory(name=spectra_shared_name)
        spectra = np.ndarray(spectra_shape, dtype=spec_dtype, buffer=shared_spectra.buf)
        shared_mask = shared_memory.SharedMemory(name=mask_shared_name)
        rfi_mask = np.ndarray(spectra_shape, dtype=bool, buffer=shared_mask.buf)

        # Replace zeros with NaN (zeros indicate masked regions after zero-filling)
        spectra_work = spectra.copy()
        spectra_work[spectra_work == 0] = np.nan

        # Trim to multiple of time_bin
        nchan, ntime = spectra_shape
        ntime_trimmed = (ntime // time_bin) * time_bin
        spectra_trimmed = spectra_work[:, :ntime_trimmed]

        # Reshape and average: (nchan, ntime_trimmed) -> (nchan, nbins, time_bin) -> (nchan, nbins)
        spectra_binned = spectra_trimmed.reshape(nchan, -1, time_bin).mean(axis=-1)

        # Check the fraction of channels that are already heavily flagged
        channel_mask_fractions = rfi_mask.mean(axis=1)
        heavily_flagged_channels = channel_mask_fractions > 0.5
        num_heavily_flagged = heavily_flagged_channels.sum()
        frac_heavily_flagged = num_heavily_flagged / self.nchan

        log.debug(
            f"{num_heavily_flagged} channels ({frac_heavily_flagged*100:.1f}%) "
            f"already have >90% of samples flagged"
        )

        # Compute global mean for each channel across all bins (ignoring NaNs)
        channel_means = np.nanmean(spectra_binned, axis=1, keepdims=True)

        # Normalize binned data by dividing by global mean to remove bandpass structure
        channel_means[channel_means == 0] = 1.0
        channel_means[np.isnan(channel_means)] = 1.0
        normalized_binned = spectra_binned / channel_means

        # Compute standard deviation across normalized bins for each channel
        channel_stds = np.nanstd(normalized_binned, axis=1)

        # Replace NaN stds with 0 for heavily-flagged channels
        channel_stds[np.isnan(channel_stds)] = 0

        # Always compute statistics from only the cleaner channels (those with <50% flagged)
        # to avoid biased estimates when many channels are already heavily flagged
        clean_channel_stds = channel_stds[~heavily_flagged_channels]
        # Remove zeros from clean channels for statistics
        clean_channel_stds = clean_channel_stds[clean_channel_stds > 0]

        if len(clean_channel_stds) == 0:
            log.error("No clean channels available for statistics - skipping cleaner")
            self.cleaned = True
            shared_spectra.close()
            shared_mask.close()
            return

        log.debug(
            f"Computing statistics from {len(clean_channel_stds)} channels "
            f"with <50% samples flagged"
        )

        # Normalize std by its median (like in the working snippet)
        std_median = np.nanmedian(clean_channel_stds)

        if std_median == 0:
            log.warning("Median std is zero - skipping cleaner")
            self.cleaned = True
            return

        normalized_channel_stds = channel_stds / std_median
        normalized_clean_stds = clean_channel_stds / std_median

        # Use MAD for robust statistics on normalized std values
        # Compute MAD of (normalized_std - 1) to detect deviations from typical behavior
        std_deviations = normalized_clean_stds - 1.0
        std_mad = median_absolute_deviation(std_deviations)[1]  # Get just the MAD value
        log.debug(f"Normalized channel std median: 1.0 (by design), MAD of deviations: {std_mad:.3f}")

        # Flag channels where deviation from 1.0 exceeds threshold
        # This matches: (bpass-1) > thresh*mad from the snippet
        upper_bound = 1.0 + self.threshold * std_mad
        lower_bound = 1.0 - self.threshold * std_mad

        # Flag entire channels that fall outside bounds
        bad_channels = np.logical_or(
            normalized_channel_stds > upper_bound,
            normalized_channel_stds < lower_bound
        )

        # Also flag channels with zero std (non-physical)
        bad_channels = np.logical_or(bad_channels, channel_stds == 0)

        num_flagged = np.sum(bad_channels)
        log.info(
            f"Flagged {num_flagged} channels ({num_flagged/self.nchan*100:.1f}%) "
            f"based on normalized std deviation (bounds: [{lower_bound:.3f}, {upper_bound:.3f}])"
        )

        # Update the cleaner mask - flag entire channels
        self.cleaner_mask[bad_channels, :] = True

        # Also update the shared memory mask directly
        rfi_mask[bad_channels, :] = True

        self.cleaned = True

        # Close shared memory handles
        shared_spectra.close()
        shared_mask.close()


class PowerSpectrumCleaner(Cleaner):
    """
    This class accepts a spectra and detects periodic RFI with an amplitude outside the
    defined threshold, by creating a power spectrum for each frequency channel. This
    cleaner effectively mimics the method employed by PRESTO's rfifind algorithm.

    For a correctly normalised power spectrum (where the mean value is unity), the
    probability of getting a power value >= threhsold is ~exp(-threshold). Given the
    "trials factor" involved, and since we are deciding whether or not to flag an entire
    channel for a large chunk of time, a threshold of ~20 is appropriate.
    """

    def __init__(
        self, spectra: np.ndarray, threshold: float = 20.0, plot_diagnostics=True
    ):
        super().__init__(spectra)
        self.dt = TSAMP  # time sampling
        self.threshold = threshold
        self.plot_diagnostics = plot_diagnostics

    def summary(self):
        return dict(
            nchan=self.nchan,
            ntime=self.ntime,
            nsamp=self.nsamp,
            threshold=self.threshold,
        )

    def clean(self):
        log.info("Running Power Spectrum outlier detection")
        self.flag_per_channel()
        self.cleaned = True

    def flag_per_channel(self):
        log.debug("Computing per-channel power spectra")
        # use the pyFFTW rfft implementation as it's ~25% faster
        per_chan_power_spec = np.abs(rfft(self.spectra, axis=1)) ** 2
        # normalise such that the mean value is approximately 1
        log.debug("Re-normalise power spectrum to have a mean ~ 1")
        per_chan_power_spec /= np.median(per_chan_power_spec, axis=1)[
            :, np.newaxis
        ] / np.log(2)

        fft_mask = np.zeros_like(self.spectra, dtype=bool)

        log.debug(
            "Gather channels with Fourier power exceeding threshold or invalid data"
        )
        chans_above_threshold = [
            c
            for c in range(self.nchan)
            if any(per_chan_power_spec[c, 1:] > self.threshold)
        ]
        fft_mask[chans_above_threshold, :] = True
        log.debug(
            "Number of channels flagged due to excess power:"
            f" {len(chans_above_threshold)}"
        )

        invalid_chans = [
            c
            for c in range(self.nchan)
            if not all(np.isfinite(per_chan_power_spec[c, 1:]))
        ]
        fft_mask[invalid_chans, :] = True
        log.debug(
            f"Number of channels flagged due to invalid data: {len(invalid_chans)}"
        )

        total_num_to_flag = len(chans_above_threshold) + len(invalid_chans)
        log.debug(
            f"Total number of channels to mask = {total_num_to_flag} (out of"
            f" {self.nchan})"
        )
        self.cleaner_mask = np.logical_or(self.spectra.mask, fft_mask)

        if self.plot_diagnostics:
            from rfi_mitigation.utilities.cleaner_utils import (
                plot_diagnostic_power_spectrum,
            )

            power_spec_freqs = np.fft.rfftfreq(self.ntime, d=self.dt)

            for chan in range(0, self.nchan, self.nchan // 8):
                plot_diagnostic_power_spectrum(
                    power_spec_freqs[1:],
                    per_chan_power_spec[chan, 1:],
                    chan,
                    thresh=self.threshold,
                    nchan=self.nchan,
                )
