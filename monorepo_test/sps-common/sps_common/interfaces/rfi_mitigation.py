#!/usr/bin/env python
import numpy as np
from astropy.time import Time
from attr import attrib, attrs

from sps_common.constants import TSAMP
from sps_common.conversion import convert_intensity_to_hdf5, read_intensity_hdf5


@attrs
class SlowPulsarIntensityChunk:
    """
    A chunk of intensity data that has been quantized and potentially downsampled. It is
    both the input and output structure of the RFI cleaning pipeline.

    This class is not used in the current loading scheme.

    Attributes:
        spectra (np.ma.MaskedArray): The raw intensity data with whatever previous
        masking has been applied stored in the masked array's `.mask` attribute.

        nchan (int): The number of frequency channels in the spectra.

        ntime (int): The number of time samples in the spectra.

        nsamp (int): The total number of sample values (should be nchan * ntime).

        start_unix_time (float): The Unix time corresponding to the first time sample
        in the spectra.

        start_mjd (float): The Modified Julien Day corresponding to the first time
        sample in the spectra.

        beam_number (int): The CHIME/FRB beam number from which the intensity data was
        recorded.

        cleaned (bool): Whether the data have been passed through the RFIPipeline
        process.
    """

    spectra = attrib(type=np.ma.MaskedArray)
    nchan = attrib(converter=int)
    ntime = attrib(converter=int)
    nsamp = attrib(converter=int)
    start_unix_time = attrib(type=float)
    start_mjd = attrib(type=float)
    beam_number = attrib(converter=int)
    cleaned = attrib(type=bool, default=False)

    @nchan.validator
    def _check_nchan(self, attribute, value):
        choices = [1024, 2048, 4096, 8192, 16384]
        if value not in choices:
            raise ValueError(
                f"Attribute {attribute.name}={value} is invalid. "
                f"Must be one of {choices}"
            )

    @start_unix_time.validator
    def _check_start_unix_time(self, attribute, value):
        min_unix = 1514764800.0  # 2018-01-01 00:00:00 UTC
        if value < min_unix:
            raise ValueError(
                f"Attribute {attribute.name}={value} is invalid. "
                f"Must be greater than {min_unix}"
            )

    @start_mjd.validator
    def _check_start_mjd(self, attribute, value):
        min_mjd = 58119.0  # 2018-01-01 00:00:00 UTC
        if value < min_mjd:
            raise ValueError(
                f"Attribute {attribute.name}={value} is invalid. "
                f"Must be greater than {min_mjd}"
            )

    @beam_number.validator
    def _check_beam_number(self, attribute, value):
        min_row = 0
        max_row = 255
        min_col = 0
        max_col = 4
        choices = np.arange(min_row, max_row + 1, dtype=int)[
            :, np.newaxis
        ] + 1000 * np.arange(min_col, max_col)
        choices = choices.flatten().astype(int)
        if value not in choices:
            raise ValueError(
                f"The CHIME/FRB beam number specified ({attribute.name}={value}) is "
                "invalid. Should be something like N[000-255] where N = [0, 3] and "
                "leading zeros are truncated."
            )

    @classmethod
    def read(cls, fname: str):
        """
        Read a HDF5 that is assumed to be formatted as if it had been written by another
        SlowPulsarIntensityChunk instance. See `write_hdf5_file`.

        Parameters
        ----------
        fname: str
            HDF5 file name to load and unpack into the class attributes

        Returns
        -------
            An instance of this class with data and relevant attributes populated by
            what was in the read HDF5 file.
        """
        i, m, metadata = read_intensity_hdf5(fname)
        spectra = np.ma.array(i, mask=m.astype(bool))
        start_time = Time(metadata["start"], format="unix", scale="utc")

        return cls(
            spectra=spectra,
            nchan=metadata["nchan"],
            ntime=metadata["ntime"],
            nsamp=spectra.size,
            start_unix_time=start_time.unix,
            start_mjd=start_time.mjd,
            beam_number=metadata["beam_number"],
            cleaned=metadata["cleaned"],
        )

    def write(self, odir: str = None, tsamp: float = TSAMP) -> str:
        """
        Take the current RFIPipeline state and write the intensity and masking data to a
        HDF5 file, including useful metadata for downstream processes.

        Parameters
        ----------
        odir: str
            Output path where files are to be written

        Returns
        -------
            The name of the saved file.
        """
        metadata = dict(
            beam_number=self.beam_number,
            nchan=self.nchan,
            ntime=self.ntime,
            start=self.start_unix_time,
            end=self.start_unix_time + self.ntime * tsamp,
            cleaned=self.cleaned,
        )
        fname = "{:.0f}_{:.0f}.hdf5".format(metadata["start"], metadata["end"])
        if odir is not None:
            fname = f"{odir}/{fname}"
        convert_intensity_to_hdf5(self.spectra.data, self.spectra.mask, fname, metadata)

        return fname

    def get_data(self) -> np.ndarray:
        """
        Return the current working raw intensity (without masking information)

        Returns
        -------
            The raw intensity data (i.e. np.ma.MaskedArray.data), which will
            be a 2D np.ndarray (dtype = np.float32)
        """
        return self.spectra.data

    def get_mask(self) -> np.ndarray:
        """
        Return the current working intensity mask.

        Returns
        -------
            The intensity mask (i.e. np.ma.MaskedArray.mask), which will be a
            2D np.ndarray (dtype = bool)
        """
        return self.spectra.mask

    def get_masked_data(self) -> np.ma.MaskedArray:
        """
        Return the current working masked intensity data as a np.ma.MaskedArray.

        Returns
        -------
            The masked intensity data, which will be a 2D np.ma.MaskedArray
            (dtype = np.float32)
        """
        return self.spectra
