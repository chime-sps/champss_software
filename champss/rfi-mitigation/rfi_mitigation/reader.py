#!/usr/bin/env python
import logging
import os
from typing import List

import numpy as np
from astropy.time import Time, TimeDelta
from rfi_mitigation import conversion as rfi_conversion
from sps_common.constants import L0_NCHAN
from sps_common.interfaces import SlowPulsarIntensityChunk

log = logging.getLogger(__name__)


class DataReader:
    """
    Generic data reader for the SPS RFI pipeline.

    This class encapsulates functions to read data from disk and populate the correct
    interface to then pass to the pipeline for processing.
    """

    def __init__(self, apply_l1_mask: bool = True):
        self.apply_l1_mask = apply_l1_mask

    def read_files(
        self,
        files_to_read: list,
        is_callback: bool = False,
        is_raw: bool = False,
        sps_freq_downsamp_factor: int = 1,
    ) -> List[SlowPulsarIntensityChunk]:
        """Read the provided file list and populate a variety of class attributes based
        on the information from the data headers, metadata and final shape.
        """
        suffix = ".dat"
        if is_callback:
            suffix = ".msgpack"
        if is_raw:
            suffix = ".msg"

        if all([f.endswith(suffix) for f in files_to_read]):
            if is_raw:
                # assume items are "standard" CHIME/FRB msgpack files
                log.info("Assuming files are standard msgpack format")
                spic_list = self._read_msgpack_files(files_to_read)
            else:
                # assume items are Huffmann-encoded, quantised data (i.e. 3-bit)
                log.info("Assuming files are Huffmann-encoded, quantised msgpack data")
                if sps_freq_downsamp_factor != 1:
                    log.info(
                        "Will downsample in frequency by a factor of"
                        f" {sps_freq_downsamp_factor}"
                    )
                spic_list = self._read_sps_files(
                    files_to_read, channel_downsampling_factor=sps_freq_downsamp_factor
                )

            return spic_list

        if all([f.endswith(".hdf5") for f in files_to_read]):
            # assuming that a single chunk of data is in a single HDF5 file
            log.info(
                "Assuming HDF5 file was written by SlowPulsarIntensityChunk.write()"
            )
            return [SlowPulsarIntensityChunk.read(f) for f in files_to_read]

    def _read_sps_files(
        self, fnames: List[str], channel_downsampling_factor: int = 1
    ) -> List[SlowPulsarIntensityChunk]:
        """
        Read a given FRB -> SPS huffman encoded format data file into a flexible numpy
        masked array.

        Parameters
        ----------
        fnames: list of str
            The paths to a FRB Huffmann-encoded msgpack files to be ingested
        channel_downsampling_factor: int
            The desired downsampling factor in frequency. Default = 1

        Returns
        -------
        spic: list of SlowPulsarIntensityChunk
            A list of interface objects containing the data from the list of files
            recieved as input.
        """

        spic = []

        for fname in fnames:
            log.debug(f"Loading {fname}")
            file_chunks = rfi_conversion.read_huff_msgpack(
                fname, channel_downsampling_factor=channel_downsampling_factor
            )
            for fc in file_chunks:
                spic.append(SlowPulsarIntensityChunk(**fc))

        return spic

    def _read_msgpack_files(self, fnames: List[str]) -> List[SlowPulsarIntensityChunk]:
        """
        Read a given FRB L1 msgpack format data file into a flexible numpy masked array.

        Parameters
        ----------
        fnames: list of str
            The paths to a FRB L1 format msgpack files to be ingested

        Returns
        -------
        spic: list of SlowPulsarIntensityChunk
            A list of interface objects containing the data from the list of files
            recieved as input.
        """
        try:
            from ch_frb_l1 import rpc_client
        except ImportError:
            log.error(
                "Failed to import ch_frb_l1, cannot read FRB L1 msgpack data directly",
                exc_info=True,
            )
            log.error(
                "In order to load FRB L1 msgpack data, please install the ch_frb_l1 "
                "package."
            )

        # get the FRB beam number from the absolute path (assumes data are stored in
        # standard directory structure: YYYY/mm/dd/BEAM/START_END.[suffix])
        abspath = os.path.abspath(fnames[0])
        beam_number = abspath.split("/")[-2]
        if "beam" in beam_number:
            beam_number = int(beam_number.split("_")[-1])

        spic = []

        for i, filename in enumerate(fnames):
            log.debug(f"Loading {filename}")
            chunk = rpc_client.read_msgpack_file(filename)
            intensity, weights = chunk.decode()

            start_time = Time(
                chunk.frame0_nano * 1e-9, format="unix", scale="utc"
            ) + TimeDelta(0.0, chunk.fpga0 * 2.56e-6, format="sec")

            # construct the full msgpack data mask
            msgpack_mask = np.zeros_like(weights, dtype=bool)
            msgpack_mask[~np.isfinite(intensity)] = True
            msgpack_mask[np.where(intensity <= 0)] = True

            if self.apply_l1_mask:
                # if any L1 RFI mask/weight value is zero, we need to mask the data,
                # otherwise leave it
                rfi_mask = np.repeat(
                    chunk.rfi_mask, intensity.shape[0] // L0_NCHAN, axis=0
                )
                msgpack_mask = np.logical_or(msgpack_mask, ~weights.astype(bool))
                msgpack_mask = np.logical_or(msgpack_mask, ~rfi_mask)

            # pack the intensity data into a numpy masked array for easy
            # access and manipulation
            masked_msgpack_data = np.ma.masked_array(
                intensity.astype(np.float32), mask=msgpack_mask
            )

            spic.append(
                SlowPulsarIntensityChunk(
                    spectra=masked_msgpack_data,
                    nchan=masked_msgpack_data.shape[0],
                    ntime=masked_msgpack_data.shape[1],
                    nsamp=masked_msgpack_data.size,
                    start_unix_time=start_time.unix,
                    start_mjd=start_time.mjd,
                    beam_number=beam_number,
                    cleaned=False,
                )
            )

        return spic
