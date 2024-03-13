#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    from typing import Union, List
    import logging
    import numpy as np
    import time
    from prometheus_client import Summary
    from rfi_mitigation.utilities.cleaner_utils import (
        combine_cleaner_masks,
        known_bad_channels,
    )
    from rfi_mitigation.cleaners import cleaners
    from sps_common.interfaces import SlowPulsarIntensityChunk
except ImportError as err:
    print(err)
    exit(1)

log = logging.getLogger(__name__)

rfi_processing_time = Summary(
    "rfi_chunk_processing_seconds",
    "Duration of running RFI mitigation on a chunk",
    ("cleaner", "beam"),
)


class RFIPipelineException(Exception):
    pass


class RFIPipeline(object):
    """
    This class is responsible for cleaning a chunk of data. It accepts a list of
    SlowPulsarIntensityChunks and applies a dynamic list of RFI mitigation techniques.
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

        self.plot_diagnostics = make_plots

    def clean(self, sps_data_chunk: List[SlowPulsarIntensityChunk]):
        """
        Run the requested cleaners on the provided intensity data, updating the masks
        in-place.

        Parameters
        ----------
        sps_data_chunk: list(SlowPulsarIntensityChunk)
            A list of intensity chunks for cleaning. Each chunk is processed
            independently.

        Returns
        -------
            A list of cleaned intensity chunks, where the mask has been updated and the
            "cleaned" flag toggled. list(SlowPulsarIntensityChunk)

        """
        cleaned_chunks = []
        avg_nchan = np.mean([s.nchan for s in sps_data_chunk])

        cleaning_start = time.time()
        for ichunk, chunk in enumerate(sps_data_chunk):
            log.debug(f"cleaning chunk {ichunk}")
            log.debug(f"chunk shape = {chunk.spectra.shape}")
            initial_masked_frac = chunk.spectra.mask.sum() / chunk.nsamp
            log.info("initial flagged fraction = {0:g}".format(initial_masked_frac))

            if self.apply_dummy_filter:
                with rfi_processing_time.labels("dummy", chunk.beam_number).time():
                    dummy_start = time.time()
                    log.debug("Dummy clean START")
                    cleaner = cleaners.DummyCleaner(chunk.spectra)
                    cleaner.clean()
                    before_masked_frac = chunk.spectra.mask.mean()

                    chunk.spectra.mask, masked_frac = combine_cleaner_masks(
                        np.array([chunk.spectra.mask, cleaner.get_mask()])
                    )

                    masked_frac = chunk.spectra.mask.mean()
                    unique_masked_frac = masked_frac - before_masked_frac
                    log.debug("unique masked frac = {0:g}".format(unique_masked_frac))
                    log.debug("total masked frac = {0:g}".format(masked_frac))
                    log.debug("Dummy clean END")
                    dummy_end = time.time()
                    dummy_runtime = dummy_end - dummy_start
                    dummy_time_per_chan = dummy_runtime / chunk.nchan
                    log.debug(f"Took {dummy_runtime} seconds to run DummyCleaner")
                    log.debug(
                        f"Corresponds to {1000 * dummy_time_per_chan} ms per channel"
                    )

            if self.apply_badchan_mask:
                with rfi_processing_time.labels(
                    "badchan_mask", chunk.beam_number
                ).time():
                    badchan_start = time.time()
                    log.debug("Applying known bad channel mask")
                    before_masked_frac = chunk.spectra.mask.sum() / chunk.nsamp

                    chunk.spectra.mask[known_bad_channels(nchan=chunk.nchan)] = True

                    masked_frac = chunk.spectra.mask.mean()
                    unique_masked_frac = masked_frac - before_masked_frac
                    log.debug("unique masked frac = {0:g}".format(unique_masked_frac))
                    log.debug("total masked frac = {0:g}".format(masked_frac))
                    badchan_end = time.time()
                    badchan_runtime = badchan_end - badchan_start
                    badchan_time_per_chan = badchan_runtime / chunk.nchan
                    log.debug(
                        f"Took {badchan_runtime} seconds to apply bad channel mask"
                    )
                    log.debug(
                        f"Corresponds to {1000 * badchan_time_per_chan} ms per channel"
                    )

            if self.apply_kurtosis_filter:
                with rfi_processing_time.labels("kurtosis", chunk.beam_number).time():
                    kur_start = time.time()
                    log.debug("Kurtosis clean START")
                    cleaner = cleaners.KurtosisCleaner(chunk.spectra)
                    cleaner.clean()
                    before_masked_frac = chunk.spectra.mask.mean()

                    chunk.spectra.mask, masked_frac = combine_cleaner_masks(
                        np.array([chunk.spectra.mask, cleaner.get_mask()])
                    )

                    masked_frac = chunk.spectra.mask.mean()
                    unique_masked_frac = masked_frac - before_masked_frac
                    log.debug("unique masked frac = {0:g}".format(unique_masked_frac))
                    log.debug("total masked frac = {0:g}".format(masked_frac))
                    log.debug("Kurtosis clean END")
                    kur_end = time.time()
                    kur_runtime = kur_end - kur_start
                    kur_time_per_chan = kur_runtime / chunk.nchan
                    log.debug(f"Took {kur_runtime} seconds to run KurtosisCleaner")
                    log.debug(
                        f"Corresponds to {1000 * kur_time_per_chan} ms per channel"
                    )

            if self.apply_mad_filter:
                with rfi_processing_time.labels(
                    "median_absolute_deviation", chunk.beam_number
                ).time():
                    mad_start = time.time()
                    log.debug("MAD clean START")
                    cleaner = cleaners.MedianAbsoluteDeviationCleaner(chunk.spectra)
                    cleaner.clean()
                    before_masked_frac = chunk.spectra.mask.mean()

                    chunk.spectra.mask, masked_frac = combine_cleaner_masks(
                        np.array([chunk.spectra.mask, cleaner.get_mask()])
                    )

                    masked_frac = chunk.spectra.mask.mean()
                    unique_masked_frac = masked_frac - before_masked_frac
                    log.debug("unique masked frac = {0:g}".format(unique_masked_frac))
                    log.debug("total masked frac = {0:g}".format(masked_frac))
                    log.debug("MAD clean END")
                    mad_end = time.time()
                    mad_runtime = mad_end - mad_start
                    mad_time_per_chan = mad_runtime / chunk.nchan
                    log.debug(
                        f"Took {mad_runtime} seconds to run "
                        f"MedianAbsoluteDeviationCleaner"
                    )
                    log.debug(
                        f"Corresponds to {1000 * mad_time_per_chan} ms per channel"
                    )

            if self.apply_sk_filter:
                with rfi_processing_time.labels(
                    "spectral_kurtosis", chunk.beam_number
                ).time():
                    speckur_start = time.time()
                    log.debug("SK clean START")
                    scales = [1024]

                    cleaner = cleaners.SpectralKurtosisCleaner(
                        chunk.spectra,
                        scales,
                        rfi_threshold_sigma=25,
                        plot_diagnostics=self.plot_diagnostics,
                    )
                    cleaner.clean()
                    before_masked_frac = chunk.spectra.mask.mean()

                    chunk.spectra.mask, masked_frac = combine_cleaner_masks(
                        np.array([chunk.spectra.mask, cleaner.get_mask()])
                    )

                    masked_frac = chunk.spectra.mask.mean()
                    unique_masked_frac = masked_frac - before_masked_frac
                    log.debug("unique masked frac = {0:g}".format(unique_masked_frac))
                    log.debug("total masked frac = {0:g}".format(masked_frac))
                    log.debug("SK clean END")
                    speckur_end = time.time()
                    speckur_runtime = speckur_end - speckur_start
                    speckur_time_per_chan = speckur_runtime / chunk.nchan
                    log.debug(
                        f"Took {speckur_runtime} seconds to run SpectralKurtosisCleaner"
                    )
                    log.debug(
                        f"Corresponds to {1000 * speckur_time_per_chan} ms per channel"
                    )

            if self.apply_powspec_filter:
                with rfi_processing_time.labels(
                    "power_spectrum", chunk.beam_number
                ).time():
                    powspec_start = time.time()
                    log.debug("Power spectrum clean START")
                    cleaner = cleaners.PowerSpectrumCleaner(
                        chunk.spectra, plot_diagnostics=self.plot_diagnostics
                    )
                    cleaner.clean()
                    before_masked_frac = chunk.spectra.mask.mean()

                    chunk.spectra.mask, masked_frac = combine_cleaner_masks(
                        np.array([chunk.spectra.mask, cleaner.get_mask()])
                    )

                    masked_frac = chunk.spectra.mask.mean()
                    unique_masked_frac = masked_frac - before_masked_frac
                    log.debug("unique masked frac = {0:g}".format(unique_masked_frac))
                    log.debug("total masked frac = {0:g}".format(masked_frac))
                    log.debug("Power spectrum clean END")
                    powspec_end = time.time()
                    powspec_runtime = powspec_end - powspec_start
                    powspec_time_per_chan = powspec_runtime / chunk.nchan
                    log.debug(
                        f"Took {powspec_runtime} seconds to run PowerSpectrumCleaner"
                    )
                    log.debug(
                        f"Corresponds to {1000 * powspec_time_per_chan} ms per channel"
                    )

            # update cleaned flag and append to the list of output instances
            chunk.cleaned = True
            cleaned_chunks.append(chunk)
            log.info("final flagged fraction = {0:g}".format(chunk.spectra.mask.mean()))

        cleaning_end = time.time()
        log.debug(f"Took {cleaning_end - cleaning_start} seconds to clean chunk")
        clean_time_per_chan = (cleaning_end - cleaning_start) / avg_nchan
        log.debug(f"Corresponds to {1000 * clean_time_per_chan} ms per channel")

        if self.plot_diagnostics:
            import matplotlib.pyplot as plt

            plt.figure(constrained_layout=True, figsize=plt.figaspect(0.5))
            plt.imshow(
                chunk.spectra, origin="lower", interpolation="none", aspect="auto"
            )
            plt.title(f"nchan={chunk.nchan}  mask frac. = {chunk.spectra.mask.mean()}")
            plt.savefig("cleaned_waterfall.png")
            plt.clf()

        return cleaned_chunks
