# RFI mitigation pipeline

try:
    import logging
    import time

    import numpy as np
    from prometheus_client import Summary
    from rfi_mitigation.cleaners import cleaners
    from rfi_mitigation.utilities.cleaner_utils import (
        combine_cleaner_masks,
        known_bad_channels,
    )
except ImportError as err:
    print(err)
    exit(1)

from multiprocessing import shared_memory

log = logging.getLogger(__name__)

rfi_processing_time = Summary(
    "rfi_chunk_processing_seconds",
    "Duration of running RFI mitigation on a chunk",
    ("cleaner", "beam"),
)


class RFIPipelineException(Exception):
    pass


class RFIPipeline:
    """
    This class is responsible for cleaning a chunk of data.

    It accepts a list of SlowPulsarIntensityChunks and applies a dynamic list of RFI
    mitigation techniques.
    """

    def __init__(self, masks_to_apply: dict, make_plots: bool = False):
        """
        Initialise the RFI cleaning pipeline by setting up the cleaners to use.

        Parameters
        ----------
        masks_to_apply: dict
            A dictionary of RFI mitigation techniques to apply to the data.

        make_plots: bool
            Whether to produce diagnotic plots from cleaners that support
            those operations (only useful for debugging and it comes with
            significant performance impacts)
        """
        # Masking application options
        keys = masks_to_apply.keys()
        self.apply_badchan_mask = (
            masks_to_apply["badchan"] if "badchan" in keys else False
        )
        self.apply_kurtosis_filter = (
            masks_to_apply["kurtosis"] if "kurtosis" in keys else False
        )
        self.apply_mad_filter = masks_to_apply["mad"] if "mad" in keys else False
        self.apply_sk_filter = masks_to_apply["sk"] if "sk" in keys else False
        self.apply_powspec_filter = (
            masks_to_apply["powspec"] if "powspec" in keys else False
        )
        self.apply_dummy_filter = masks_to_apply["dummy"] if "dummy" in keys else False
        self.apply_stddev_filter = (
            masks_to_apply["stddev"] if "stddev" in keys else False
        )

        self.plot_diagnostics = make_plots

    def clean(
        self,
        spectra_shared_name,
        mask_shared_name,
        spectra_shape,
        spec_dtype,
        current_slice,
    ):
        """
        Run the requested cleaners on the provided intensity data, updating the masks
        in-place.

        Parameters
        ----------
        spectra_shared_name: str
            Name of shared spectra

        mask_shared_name: str
            Name of shared mask

        spectra_shape: tuple(int)
            Shape of spectra

        spec_dtype:
            dtype of spectra

        current_slice: slice
            Slice of time steps that is being cleaned in this step.
        """
        shared_spectra = shared_memory.SharedMemory(name=spectra_shared_name)
        spectra_full = np.ndarray(
            spectra_shape, dtype=spec_dtype, buffer=shared_spectra.buf
        )
        spectra = spectra_full[:, current_slice]
        shared_mask = shared_memory.SharedMemory(name=mask_shared_name)
        rfi_mask_full = np.ndarray(spectra_shape, dtype=bool, buffer=shared_mask.buf)
        rfi_mask = rfi_mask_full[:, current_slice]

        # avg_nchan = np.mean([s.nchan for s in sps_data_chunk])
        avg_nchan = spectra_shape[0]

        cleaning_start = time.time()

        log.debug(f"chunk shape = {spectra_shape}")
        initial_masked_frac = rfi_mask.sum() / rfi_mask.size
        log.info(f"initial flagged fraction = {initial_masked_frac:g}")

        if self.apply_dummy_filter:
            with rfi_processing_time.labels("dummy", "0").time():
                dummy_start = time.time()
                log.debug("Dummy clean START")
                cleaner = cleaners.DummyCleaner(spectra)
                cleaner.clean()
                before_masked_frac = rfi_mask.mean()

                rfi_mask[:], masked_frac = combine_cleaner_masks(
                    np.array([rfi_mask, cleaner.get_mask()])
                )

                masked_frac = rfi_mask.mean()
                unique_masked_frac = masked_frac - before_masked_frac
                log.debug(f"unique masked frac = {unique_masked_frac:g}")
                log.debug(f"total masked frac = {masked_frac:g}")
                log.debug("Dummy clean END")
                dummy_end = time.time()
                dummy_runtime = dummy_end - dummy_start
                dummy_time_per_chan = dummy_runtime / spectra_shape[0]
                log.debug(f"Took {dummy_runtime} seconds to run DummyCleaner")
                log.debug(f"Corresponds to {1000 * dummy_time_per_chan} ms per channel")

        if self.apply_badchan_mask:
            with rfi_processing_time.labels("badchan_mask", "0").time():
                badchan_start = time.time()
                log.debug("Applying known bad channel mask")
                before_masked_frac = rfi_mask.sum() / rfi_mask.size

                rfi_mask[known_bad_channels(nchan=spectra_shape[0])] = True

                masked_frac = rfi_mask.mean()
                unique_masked_frac = masked_frac - before_masked_frac
                log.debug(f"unique masked frac = {unique_masked_frac:g}")
                log.debug(f"total masked frac = {masked_frac:g}")
                badchan_end = time.time()
                badchan_runtime = badchan_end - badchan_start
                badchan_time_per_chan = badchan_runtime / spectra_shape[0]
                log.debug(f"Took {badchan_runtime} seconds to apply bad channel mask")
                log.debug(
                    f"Corresponds to {1000 * badchan_time_per_chan} ms per channel"
                )

        if self.apply_kurtosis_filter:
            with rfi_processing_time.labels("kurtosis", "0").time():
                kur_start = time.time()
                log.debug("Kurtosis clean START")
                cleaner = cleaners.KurtosisCleaner(spectra)
                cleaner.clean()
                before_masked_frac = rfi_mask.mean()

                rfi_mask[:], masked_frac = combine_cleaner_masks(
                    np.array([rfi_mask, cleaner.get_mask()])
                )

                masked_frac = rfi_mask.mean()
                unique_masked_frac = masked_frac - before_masked_frac
                log.debug(f"unique masked frac = {unique_masked_frac:g}")
                log.debug(f"total masked frac = {masked_frac:g}")
                log.debug("Kurtosis clean END")
                kur_end = time.time()
                kur_runtime = kur_end - kur_start
                kur_time_per_chan = kur_runtime / spectra_shape[0]
                log.debug(f"Took {kur_runtime} seconds to run KurtosisCleaner")
                log.debug(f"Corresponds to {1000 * kur_time_per_chan} ms per channel")

        if self.apply_mad_filter:
            with rfi_processing_time.labels("median_absolute_deviation", "0").time():
                mad_start = time.time()
                log.debug("MAD clean START")
                cleaner = cleaners.MedianAbsoluteDeviationCleaner(spectra)
                cleaner.clean()
                before_masked_frac = rfi_mask.mean()

                rfi_mask[:], masked_frac = combine_cleaner_masks(
                    np.array([rfi_mask, cleaner.get_mask()])
                )

                masked_frac = rfi_mask.mean()
                unique_masked_frac = masked_frac - before_masked_frac
                log.debug(f"unique masked frac = {unique_masked_frac:g}")
                log.debug(f"total masked frac = {masked_frac:g}")
                log.debug("MAD clean END")
                mad_end = time.time()
                mad_runtime = mad_end - mad_start
                mad_time_per_chan = mad_runtime / spectra_shape[0]
                log.debug(
                    f"Took {mad_runtime} seconds to run MedianAbsoluteDeviationCleaner"
                )
                log.debug(f"Corresponds to {1000 * mad_time_per_chan} ms per channel")

        if self.apply_sk_filter:
            with rfi_processing_time.labels("spectral_kurtosis", "0").time():
                speckur_start = time.time()
                log.debug("SK clean START")
                scales = [1024]
                cleaner = cleaners.SpectralKurtosisCleaner(
                    spectra.shape,
                    scales,
                    rfi_threshold_sigma=25,
                    plot_diagnostics=self.plot_diagnostics,
                )
                cleaner.clean(spectra, rfi_mask)
                before_masked_frac = rfi_mask.mean()

                rfi_mask[:], masked_frac = combine_cleaner_masks(
                    np.array([rfi_mask, cleaner.get_mask()])
                )

                masked_frac = rfi_mask.mean()
                unique_masked_frac = masked_frac - before_masked_frac
                log.debug(f"unique masked frac = {unique_masked_frac:g}")
                log.debug(f"total masked frac = {masked_frac:g}")
                log.debug("SK clean END")
                speckur_end = time.time()
                speckur_runtime = speckur_end - speckur_start
                speckur_time_per_chan = speckur_runtime / spectra_shape[0]
                log.debug(
                    f"Took {speckur_runtime} seconds to run SpectralKurtosisCleaner"
                )
                log.debug(
                    f"Corresponds to {1000 * speckur_time_per_chan} ms per channel"
                )

        if self.apply_powspec_filter:
            with rfi_processing_time.labels("power_spectrum", "0").time():
                powspec_start = time.time()
                log.debug("Power spectrum clean START")
                cleaner = cleaners.PowerSpectrumCleaner(
                    spectra, plot_diagnostics=self.plot_diagnostics
                )
                cleaner.clean()
                before_masked_frac = rfi_mask.mean()

                rfi_mask[:], masked_frac = combine_cleaner_masks(
                    np.array([rfi_mask, cleaner.get_mask()])
                )

                masked_frac = rfi_mask.mean()
                unique_masked_frac = masked_frac - before_masked_frac
                log.debug(f"unique masked frac = {unique_masked_frac:g}")
                log.debug(f"total masked frac = {masked_frac:g}")
                log.debug("Power spectrum clean END")
                powspec_end = time.time()
                powspec_runtime = powspec_end - powspec_start
                powspec_time_per_chan = powspec_runtime / spectra_shape[0]
                log.debug(f"Took {powspec_runtime} seconds to run PowerSpectrumCleaner")
                log.debug(
                    f"Corresponds to {1000 * powspec_time_per_chan} ms per channel"
                )

        if self.apply_stddev_filter:
            with rfi_processing_time.labels("stddev_channel", "0").time():
                stddev_start = time.time()
                log.debug("StdDev Channel clean START")
                cleaner = cleaners.StdDevChannelCleaner(spectra.shape)
                cleaner.clean(spectra, rfi_mask)
                before_masked_frac = rfi_mask.mean()

                rfi_mask[:], masked_frac = combine_cleaner_masks(
                    np.array([rfi_mask, cleaner.get_mask()])
                )

                masked_frac = rfi_mask.mean()
                unique_masked_frac = masked_frac - before_masked_frac
                log.debug(f"unique masked frac = {unique_masked_frac:g}")
                log.debug(f"total masked frac = {masked_frac:g}")
                log.debug("StdDev Channel clean END")
                stddev_end = time.time()
                stddev_runtime = stddev_end - stddev_start
                stddev_time_per_chan = stddev_runtime / spectra_shape[0]
                log.debug(f"Took {stddev_runtime} seconds to run StdDevChannelCleaner")
                log.debug(
                    f"Corresponds to {1000 * stddev_time_per_chan} ms per channel"
                )

        # update cleaned flag and append to the list of output instances

        log.info(f"final flagged fraction = {rfi_mask.mean():g}")

        cleaning_end = time.time()
        log.debug(f"Took {cleaning_end - cleaning_start} seconds to clean chunk")
        clean_time_per_chan = (cleaning_end - cleaning_start) / avg_nchan
        log.debug(f"Corresponds to {1000 * clean_time_per_chan} ms per channel")

        if self.plot_diagnostics:
            import matplotlib.pyplot as plt

            plt.figure(constrained_layout=True, figsize=plt.figaspect(0.5))
            plt.imshow(spectra, origin="lower", interpolation="none", aspect="auto")
            plt.title(f"nchan={spectra_shape[0]}  mask frac. = {rfi_mask.mean()}")
            plt.savefig("cleaned_waterfall.png")
            plt.clf()

        shared_spectra.close()
        shared_mask.close()


class RFIGlobalPipeline:
    """
    This class is responsible for cleaning the full beamformed dataset.

    Unlike RFIPipeline which operates on chunks, this pipeline operates on the
    entire dataset at once, allowing cleaners to analyze global statistics across
    the full time series.
    """

    def __init__(self, masks_to_apply: dict, make_plots: bool = False):
        """
        Initialise the global RFI cleaning pipeline.

        Parameters
        ----------
        masks_to_apply: dict
            A dictionary of RFI mitigation techniques to apply to the data.
            Currently supports:
            - "stddev": StdDevChannelCleaner

        make_plots: bool
            Whether to produce diagnostic plots from cleaners that support
            those operations (only useful for debugging and it comes with
            significant performance impacts)
        """
        keys = masks_to_apply.keys()
        self.apply_stddev_filter = (
            masks_to_apply["stddev"] if "stddev" in keys else False
        )

        self.plot_diagnostics = make_plots

    def clean(
        self,
        spectra_shared_name,
        mask_shared_name,
        spectra_shape,
        spec_dtype,
    ):
        """
        Run the requested global cleaners on the full beamformed dataset.

        Parameters
        ----------
        spectra_shared_name: str
            Name of shared spectra

        mask_shared_name: str
            Name of shared mask

        spectra_shape: tuple(int)
            Shape of spectra

        spec_dtype:
            dtype of spectra
        """
        shared_spectra = shared_memory.SharedMemory(name=spectra_shared_name)
        spectra = np.ndarray(
            spectra_shape, dtype=spec_dtype, buffer=shared_spectra.buf
        )
        shared_mask = shared_memory.SharedMemory(name=mask_shared_name)
        rfi_mask = np.ndarray(spectra_shape, dtype=bool, buffer=shared_mask.buf)

        log.debug(f"full data shape = {spectra_shape}")
        initial_masked_frac = rfi_mask.sum() / rfi_mask.size
        log.info(f"initial flagged fraction = {initial_masked_frac:g}")

        if self.apply_stddev_filter:
            with rfi_processing_time.labels("stddev_channel_global", "0").time():
                stddev_start = time.time()
                log.debug("Global StdDev Channel clean START")
                cleaner = cleaners.StdDevChannelCleaner(spectra_shape)
                cleaner.clean(spectra_shared_name, mask_shared_name, spectra_shape, spec_dtype)

                # Re-access mask after cleaner modifies it
                shared_mask = shared_memory.SharedMemory(name=mask_shared_name)
                rfi_mask = np.ndarray(spectra_shape, dtype=bool, buffer=shared_mask.buf)
                before_masked_frac = rfi_mask.mean()

                rfi_mask[:], masked_frac = combine_cleaner_masks(
                    np.array([rfi_mask, cleaner.get_mask()])
                )

                masked_frac = rfi_mask.mean()
                unique_masked_frac = masked_frac - before_masked_frac
                log.debug(f"unique masked frac = {unique_masked_frac:g}")
                log.debug(f"total masked frac = {masked_frac:g}")
                log.debug("Global StdDev Channel clean END")
                stddev_end = time.time()
                stddev_runtime = stddev_end - stddev_start
                stddev_time_per_chan = stddev_runtime / spectra_shape[0]
                log.debug(
                    f"Took {stddev_runtime} seconds to run global StdDevChannelCleaner"
                )
                log.debug(
                    f"Corresponds to {1000 * stddev_time_per_chan} ms per channel"
                )

        # Re-access mask for final reporting
        shared_mask = shared_memory.SharedMemory(name=mask_shared_name)
        rfi_mask = np.ndarray(spectra_shape, dtype=bool, buffer=shared_mask.buf)
        log.info(f"final flagged fraction = {rfi_mask.mean():g}")
        shared_mask.close()

        shared_spectra.close()
        shared_mask.close()
