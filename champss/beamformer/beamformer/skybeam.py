"""Class for forming the Skybeam."""

import datetime
import logging
from functools import partial
from itertools import product
from multiprocessing import Pool, set_start_method, shared_memory
from typing import Literal

import attr
import numpy as np
import numpy.ma as ma
from astropy.time import Time, TimeDelta
from attr.validators import deep_iterable, instance_of
from beamformer.utilities.common import get_data_list
from numpy._typing import ArrayLike
from rfi_mitigation.pipeline import RFIPipeline
from scipy import linalg
from scipy.signal import detrend
from sps_common.constants import TSAMP
from sps_common.interfaces.beamformer import SkyBeam
from sps_databases import db_api
from spshuff import l1_io
from threadpoolctl import threadpool_limits

log = logging.getLogger(__name__)


def detrend_masked_row(
    data: np.ndarray,
    mask: np.ndarray = None,
    axis: int = -1,
    type: Literal["linear", "constant"] = "linear",
    bp: ArrayLike | int = 0,
    overwrite_data: bool = False,
) -> np.ndarray:
    r"""
    Remove linear or constant trend along axis from masked data. Based on
    scipy.signal.detrend.

    Parameters
    ----------
    data : array_like
        The input data.
    data : mask
        Boolean mask with True at valid indices.
        Should only have entries along the detrended axis.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.

    Notes
    -----
    Detrending can be interpreted as substracting a least squares fit polyonimial:
    Setting the parameter `type` to 'constant' corresponds to fitting a zeroth degree
    polynomial, 'linear' to a first degree polynomial. Consult the example below.

    See Also
    --------
    numpy.polynomial.polynomial.Polynomial.fit: Create least squares fit polynomial.


    Examples
    --------
    The following example detrends the function :math:`x(t) = \sin(\pi t) + 1/4`:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy.signal import detrend
    ...
    >>> t = np.linspace(-0.5, 0.5, 21)
    >>> x = np.sin(np.pi*t) + 1/4
    ...
    >>> x_d_const = detrend(x, type='constant')
    >>> x_d_linear = detrend(x, type='linear')
    ...
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.set_title(r"Detrending $x(t)=\sin(\pi t) + 1/4$")
    >>> ax1.set(xlabel="t", ylabel="$x(t)$", xlim=(t[0], t[-1]))
    >>> ax1.axhline(y=0, color='black', linewidth=.5)
    >>> ax1.axvline(x=0, color='black', linewidth=.5)
    >>> ax1.plot(t, x, 'C0.-',  label="No detrending")
    >>> ax1.plot(t, x_d_const, 'C1x-', label="type='constant'")
    >>> ax1.plot(t, x_d_linear, 'C2+-', label="type='linear'")
    >>> ax1.legend()
    >>> plt.show()

    Alternatively, NumPy's `~numpy.polynomial.polynomial.Polynomial` can be used for
    detrending as well:

    >>> pp0 = np.polynomial.Polynomial.fit(t, x, deg=0)  # fit degree 0 polynomial
    >>> np.allclose(x_d_const, x - pp0(t))  # compare with constant detrend
    True
    >>> pp1 = np.polynomial.Polynomial.fit(t, x, deg=1)  # fit degree 1 polynomial
    >>> np.allclose(x_d_linear, x - pp1(t))  # compare with linear detrend
    True

    Note that `~numpy.polynomial.polynomial.Polynomial` also allows fitting higher
    degree polynomials. Consult its documentation on how to extract the polynomial
    coefficients.
    """
    if type not in ["linear", "l", "constant", "c"]:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = np.asarray(data)
    dtype = data.dtype.char
    if dtype not in "dfDF":
        dtype = "d"
    if type in ["constant", "c"]:
        ret = data - np.mean(data, axis, keepdims=True)
        return ret
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = np.sort(np.unique(np.concatenate(np.atleast_1d(0, bp, N))))
        if np.any(bp > N):
            raise ValueError(
                "Breakpoints must be less than length of data along given axis."
            )

        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdata = np.moveaxis(data, axis, 0)
        newdata_shape = newdata.shape
        newdata = newdata.reshape(N, -1)

        if mask is not None:
            newmask = np.moveaxis(mask, axis, 0)
            newmask = newmask.squeeze()
            if newmask.shape[0] != newmask.size:
                raise ValueError(
                    "Mask should not have multiple dimensions with size larger than 1."
                )

        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in "dfDF":
            newdata = newdata.astype(dtype)

        #        Nreg = len(bp) - 1
        # Find leastsq fit and remove it for each piece
        for m in range(len(bp) - 1):
            Npts = bp[m + 1] - bp[m]
            A = np.ones((Npts, 2), dtype)
            A[:, 0] = np.arange(1, Npts + 1, dtype=dtype) / Npts
            sl = slice(bp[m], bp[m + 1])
            if mask is not None:
                valid_points = newmask[sl]
                A = A[valid_points]
                if A.size:
                    coef, resids, rank, s = linalg.lstsq(A, newdata[sl][valid_points])
                    newdata[sl][valid_points] = newdata[sl][valid_points] - A @ coef
            else:
                coef, resids, rank, s = linalg.lstsq(A, newdata[sl])
                newdata[sl] = newdata[sl] - A @ coef

        # Put data back in original shape.
        newdata[~newmask] = 0
        newdata = newdata.reshape(newdata_shape)
        ret = np.moveaxis(newdata, 0, axis)
        return ret


@attr.s(slots=True)
class SkyBeamFormer:
    """
    SkyBeam class to create skybeam from active pointings.

    Parameters
    =======
    basepath: str
        Base path to the location of the rfi-cleaned intensity data. Default: './'

    extn: str
        Extension of the files used to beamform the pointing. Currently support 'hdf5' produced
        by the rfi pipeline and 'dat' produced by the sps writer. Default: 'hdf5'

    tsamp: float
        Sampling time of the intensity data. Default: native sampling time of the intensity data

    nbits: int
        Number of bits in the beamformed spectra. Default: 32 bits

    nsub: int
        Number of subbands to use when computing the median to replace masked values.
        Default = 16384

    block_size:
        Largest common factor of the number of samples in the intensities. Default = 1024

    masking_timescale: int
        The timescale to compute the median to mask the intensity in number of samples.
        Must be a multiple of block_size. Default = 16384

    mask_channel_frac: float
        The fraction of channel being flagged for the whole channel to be masked. Default = 0.75

    detrend_data: bool
        Whether to detrend the channels in the spectra upon masking. Default = False

    detrend_nsamp: int or None
        The number of samples to detrend the data with.
        Default = None = same as masking_timescale.

    add_local_median: bool
        Whether to add the local median within a subband upon masking. Default = False

    beam_to_normalise: None or int
        The beam number in which to normalise the data with. If left empty,
        no normalisation is done. Default = None.

    flatten_bandpass: bool
        Whether to flatten the bandpass by normalising each channel. Default: False.

    update_db: bool
        Whether to update the database of RFI fraction in the skybeam. Default: True

    min_data_frac: float
        Minimum fraction of data available such that a skybeam is consider completely formed.
        Default: 0.0

    run_rfi_mitigation: bool
        Whether to run RFI mitigation on the 'dat' files from sps writer, if the
        option is chosen. Default = True

    masking_dict: dict
        The dictionary of the configuration for the RFI mitigation process. Default = {}

    active_beams: List[int]
        The list of beam columns to be used for beamforming from 0 to 3. Default [0, 1, 2, 3]
    """

    basepath = attr.ib(default="./", validator=instance_of(str))
    extn = attr.ib(default="hdf5", validator=instance_of(str))
    tsamp = attr.ib(default=TSAMP, validator=instance_of(float))
    nbits = attr.ib(default=32, validator=instance_of(int))
    nsub = attr.ib(default=16384, validator=instance_of(int))
    block_size = attr.ib(default=1024, validator=instance_of(int))
    masking_timescale = attr.ib(default=16384, validator=instance_of(int))
    mask_channel_frac = attr.ib(default=0.75, validator=instance_of(float))
    detrend_data = attr.ib(default=False, validator=instance_of(bool))
    detrend_nsamp = attr.ib(default=None)
    add_local_median = attr.ib(default=False, validator=instance_of(bool))
    beam_to_normalise = attr.ib(default=None)
    flatten_bandpass = attr.ib(default=False, validator=instance_of(bool))
    update_db = attr.ib(default=True, validator=instance_of(bool))
    min_data_frac = attr.ib(default=0.0, validator=instance_of(float))
    run_rfi_mitigation = attr.ib(default=True, validator=instance_of(bool))
    masking_dict = attr.ib(default={}, validator=instance_of(dict), type=dict)
    active_beams = attr.ib(
        default=[0, 1, 2, 3],
        validator=deep_iterable(
            member_validator=instance_of(int), iterable_validator=instance_of(list)
        ),
    )
    rfi_pipeline = attr.ib(init=False)
    max_mask_frac = attr.ib(default=1.0, validator=instance_of(float))

    @extn.validator
    def _validate_extension(self, attribute, value):
        assert value in ["hdf5", "dat"], f"File format {value} not support"

    @nbits.validator
    def _validate_nbits(self, attribute, value):
        assert value in [1, 2, 4, 8, 16, 32, 64, 128], "invalid nbits value"

    @nsub.validator
    def _validate_nsub(self, attribute, value):
        assert value <= 16384, "nsub must be less than 16384"

    @masking_timescale.validator
    def _validate_timescale(self, attribute, value):
        assert value % self.block_size == 0, (
            f"masking timescale must be a multiple of block_size {self.block_size}"
        )

    @mask_channel_frac.validator
    def _validate_mask_channel_frac(self, attribute, value):
        assert 0 <= value <= 1, "mask channel fraction must be between 0 and 1"

    @detrend_nsamp.validator
    def _validate_detrend_nsamp(self, attribute, value):
        if value is not None:
            assert isinstance(value, int), (
                "detrend number of samples must be an integer or None"
            )

    @beam_to_normalise.validator
    def _validate_beam_to_normalise(self, attribute, value):
        if value is not None:
            assert value in [
                0,
                1,
                2,
                3,
            ], "the beam to normalise must be either 0, 1, 2 or 3"

    @active_beams.validator
    def _validate_active_beams(self, attribute, value):
        for v in value:
            assert v in [0, 1, 2, 3], "the active beams must be either 0, 1, 2 or 3"

    def __attrs_post_init__(self):
        """Create RFIPipeline instance."""
        if self.run_rfi_mitigation:
            self.rfi_pipeline = RFIPipeline(self.masking_dict, make_plots=False)

    def form_skybeam(self, active_pointing, num_threads=1):
        """
        Beamform a given active pointing.

        Parameter
        =========
        active_pointing: ActivePointing
            Active pointing class containing the required properties to beamform the pointing.

        num_threads: int
            Number of threads to run the parallel processing by.

        Return
        =======
        skybeam: SkyBeam
            SkyBeam object containing the spectra and related properties of the beamformed data.
        """
        # Fixes some unexpected behaviour with shared memory
        set_start_method("forkserver", force=True)
        pool = Pool(num_threads)
        spec_dtype = f"float{self.nbits}"
        if self.active_beams != [0, 1, 2, 3]:
            new_max_beams = []
            for max_beam in active_pointing.max_beams:
                if int(str(max_beam["beam"]).zfill(4)[0]) in self.active_beams:
                    new_max_beams.append(max_beam)
            active_pointing.max_beams = new_max_beams
            active_pointing.ntime = int(
                1024
                * (
                    active_pointing.max_beams[-1]["utc_end"]
                    - active_pointing.max_beams[0]["utc_start"]
                )
                // 40
                * 40
            )
        utc_start = active_pointing.max_beams[0]["utc_start"]
        beams_start = [beam["utc_start"] for beam in active_pointing.max_beams]
        beams_end = [beam["utc_end"] for beam in active_pointing.max_beams]

        beams_start_idx = [round((time - utc_start) / TSAMP) for time in beams_start]
        beams_end_idx = [round((time - utc_start) / TSAMP) for time in beams_end]
        beam_slices = [
            slice(start, end) for (start, end) in zip(beams_start_idx, beams_end_idx)
        ]
        data_list = get_data_list(
            active_pointing.max_beams,
            basepath=self.basepath,
            extn=self.extn,
            per_beam_list=True,
        )
        spectra_shape = (active_pointing.nchan, active_pointing.ntime)
        buffer_size_spectra = int(spectra_shape[0] * spectra_shape[1] * self.nbits / 8)
        spectra_shared = shared_memory.SharedMemory(
            create=True, size=buffer_size_spectra
        )
        spectra = np.ndarray(spectra_shape, dtype=spec_dtype, buffer=spectra_shared.buf)
        buffer_size_mask = int(spectra_shape[0] * spectra_shape[1] * 1)
        mask_shared = shared_memory.SharedMemory(create=True, size=buffer_size_mask)
        rfi_mask = np.ndarray(spectra_shape, dtype=bool, buffer=mask_shared.buf)
        # Arrays are filled with 0 or False
        rfi_mask[:] = True
        # Set to True to properly account for missing data

        log.info("Created arrays.")

        flattened_data_list = [
            [file, beam_start_idx, beam_end_idx]
            for (beam, beam_start_idx, beam_end_idx) in zip(
                data_list, beams_start_idx, beams_end_idx
            )
            for file in beam
        ]
        raw_spec_slices = pool.starmap(
            partial(
                self.read_to_shared_spectra,
                spectra_shared.name,
                mask_shared.name,
                spectra_shape,
                spec_dtype,
                16384 // active_pointing.nchan,
                utc_start,
            ),
            flattened_data_list,
        )

        # Here I use data fraction as the fraction of time steps which are present at the start.
        # This does not take into account values whether these values
        # are masked or not in the RFI mask.
        data_fraction = np.count_nonzero(spectra.max(0)) / spectra_shape[1]
        log.info(f"Data fraction: {data_fraction}")
        if self.min_data_frac > data_fraction:
            mask_shared.close()
            mask_shared.unlink()
            log.error(f"Data Fraction below {self.min_data_frac}")
            return None, spectra_shared

        raw_spec_slices = [
            slice
            for raw_spec_slice_list in raw_spec_slices
            for slice in raw_spec_slice_list
        ]
        log.info("Finished loading.")
        # For now separate the spectrum that each thread gets one part
        # Splitting it too small might increase memory usage
        # Can set minimal length here
        rfi_scale = max(
            int(np.ceil(spectra_shape[1] / num_threads / 1024) * 1024), 1024 * 1
        )
        log.info(
            "Start RFI cleaning. Current Masking fraction ="
            f" {(rfi_mask.sum() / rfi_mask.size):.4f}"
        )

        # For now always use actual file edges

        # if rfi_scale >= spectra_shape[1]:
        #     self.rfi_pipeline.clean(
        #         spectra_shared.name,
        #         mask_shared.name,
        #         spectra_shape,
        #         spec_dtype,
        #         slice(0, spectra_shape[1]),
        #     )
        # else:
        #     start_idxs = np.arange(0, spectra_shape[1], rfi_scale)
        #     end_idxs = np.roll(start_idxs, -1)
        #     end_idxs[-1] = spectra_shape[1]
        #     slices = [
        #         slice(start_idx, end_idx)
        #         for start_idx, end_idx in zip(start_idxs, end_idxs)
        #     ]
        #     log.info(f"Spectra are split into {len(slices)} chunks for RFI cleaning.")
        #     pool.map(
        #         partial(
        #             self.rfi_pipeline.clean,
        #             spectra_shared.name,
        #             mask_shared.name,
        #             spectra_shape,
        #             spec_dtype,
        #         ),
        #         slices,
        #     )

        log.info(
            f"Spectra are split into {len(raw_spec_slices)} chunks for RFI cleaning."
        )
        pool.map(
            partial(
                self.rfi_pipeline.clean,
                spectra_shared.name,
                mask_shared.name,
                spectra_shape,
                spec_dtype,
            ),
            raw_spec_slices,
        )

        log.info(
            "RFI cleaning finished. New masking Fraction:"
            f" {(rfi_mask.sum() / rfi_mask.size):.4f}"
        )
        log.info("Start filling and normalization.")
        used_nsub = np.min([self.nsub, spectra_shape[0]])
        chan_per_nsub = spectra_shape[0] // used_nsub
        nsub_slices = [
            slice(start_idx, end_idx)
            for start_idx, end_idx in zip(
                range(0, spectra_shape[0] + 1 - chan_per_nsub, chan_per_nsub),
                range(chan_per_nsub, spectra_shape[0] + 2, chan_per_nsub),
            )
        ]
        fill_slices = product(nsub_slices, beam_slices)
        pool.starmap(
            partial(
                self.fill_and_norm_shared_spectra,
                spectra_shared.name,
                mask_shared.name,
                spectra_shape,
                spec_dtype,
                self.masking_timescale,
                detrend_data=self.detrend_data,
                detrend_nsamp=self.detrend_nsamp,
                add_local_median=self.add_local_median,
            ),
            fill_slices,
        )
        # Not getting medians from fill_and_norm_shared_spectra because of a weird crash
        if self.beam_to_normalise is not None:
            pool.map(
                partial(
                    self.norm_channels_to_beam,
                    spectra_shared.name,
                    spectra_shape,
                    spec_dtype,
                    self.beam_to_normalise,
                    beam_slices,
                ),
                nsub_slices,
            )
        log.info("Finished filling and normalization.")
        pool.map(
            partial(
                self.zero_replace_shared_spectra,
                spectra_shared.name,
                mask_shared.name,
                spectra_shape,
                spec_dtype,
                flatten_bandpass=self.flatten_bandpass,
            ),
            nsub_slices,
        )
        completely_masked_channels = rfi_mask.min(axis=1).sum()
        log.info(
            "Fraction of completely masked channels:"
            f" {(completely_masked_channels / rfi_mask.shape[0]):.3f}"
        )
        mask_fraction = rfi_mask.sum() / rfi_mask.size

        if self.update_db:
            log.info("Updating Database")
            self.update_database(active_pointing.obs_id, mask_fraction, utc_start)
        pool.close()
        pool.join()

        if self.max_mask_frac < mask_fraction:
            mask_shared.close()
            mask_shared.unlink()
            log.error(f"Mask Fraction above {self.max_mask_frac}")
            return None, spectra_shared
        skybeam = SkyBeam(
            spectra=spectra,
            ra=active_pointing.ra,
            dec=active_pointing.dec,
            nchan=spectra.shape[0],
            ntime=spectra.shape[1],
            maxdm=active_pointing.maxdm,
            beam_row=active_pointing.beam_row,
            utc_start=float(utc_start),
            obs_id=active_pointing.obs_id,
            nbits=self.nbits,
        )
        mask_shared.close()
        mask_shared.unlink()
        log.info("Beamforming finished.")
        return skybeam, spectra_shared

    def zero_replace_shared_spectra(
        self,
        spectra_shared_name,
        mask_shared_name,
        spectra_shape,
        spec_dtype,
        current_freq_slice,
        flatten_bandpass=False,
    ):
        """
        Mask zeroes, fully mask channels and flatten bandpass.

        Replacing the zeroes in each channel of the spectra with the median of the non
        zero parts. The function also zeroes out the channels where more than 75% of the
        data are masked. Replacing zeroes in this function may be redundant if the
        sps_common.conversion.fil_and_norm function is called earlier.

        Parameters
        ==========
        spectra_shared_name: str
            Name of shared spectra

        mask_shared_name: str
            Name of shared mask

        spectra_shape: tuple(int)
            Shape of spectra

        spec_dtype:
            dtype of spectra

        current_freq_slice: slice
            Slice of frequencies that are processed in this function

        flatten_bandpass: bool
            Whether to flatten the bandpass by normalising each channel by subtracting the mean
            and dividing by the standard deviation. Default: False.
        """
        shared_spectra = shared_memory.SharedMemory(name=spectra_shared_name)
        spectra_full = np.ndarray(
            spectra_shape, dtype=spec_dtype, buffer=shared_spectra.buf
        )
        spectra = spectra_full[current_freq_slice, :]

        shared_mask = shared_memory.SharedMemory(name=mask_shared_name)
        rfi_mask_full = np.ndarray(spectra_shape, dtype=bool, buffer=shared_mask.buf)
        rfi_mask = rfi_mask_full[current_freq_slice, :]
        mask_fraction = rfi_mask.sum() / rfi_mask.size
        if mask_fraction == 1.0:
            shared_spectra.close()
            shared_mask.close()
            return
        _contains_nan = np.isnan(np.min(spectra))
        _masked_frac = rfi_mask.mean()
        _contains_nan = np.isnan(np.min(spectra))
        _local_max = np.max(spectra)
        if _masked_frac > 0.75 or _contains_nan or _local_max == 0:
            spectra[:] = 0
            rfi_mask[:] = True
        else:
            _zero_vals = spectra == 0
            _zero_count = _zero_vals.sum()
            if _zero_count > 0:
                # This happens if one channel is completely for one beam
                # but not for the others
                if flatten_bandpass:
                    # Perform this here to not unnecessarily slice in the normal case
                    spectra[~_zero_vals] = (
                        spectra[~_zero_vals] - spectra[~_zero_vals].mean()
                    ) / spectra[~_zero_vals].std()
                    spectra[_zero_vals] = 0
                else:
                    _local_median = np.median(spectra[~_zero_vals])
                    spectra[_zero_vals] = _local_median
            else:
                if flatten_bandpass:
                    # In an earlier version, the mean and std were computed using only the
                    # nonzero values, but nothing should be 0 at this point
                    spectra[:] = (spectra - spectra.mean()) / spectra.std()
        shared_spectra.close()
        shared_mask.close()

    def update_database(self, obs_id, mask_fraction, utc_start):
        """
        Update the sps database with the mask fraction.

        Only works if the SkyBeam object has an associated obs_id.

        Parameters
        ==========
        obs_id: ObjectID or str
            The observation id of the sky beam.

        mask_fraction: float
            The fraction of masked values.

        utc_start: float
            The utc start time of the sky beam formed.
        """
        start_dt = datetime.datetime.utcfromtimestamp(utc_start)
        if mask_fraction > 0.0:
            log.info(f"Fraction of data masked : {mask_fraction:.3f}")
            db_api.update_observation(
                obs_id,
                {"mask_fraction": mask_fraction, "datetime": start_dt},
            )
        else:
            raise Exception(
                "RFI mask is empty, this implies skybeam has not been formed"
            )

    def read_to_shared_spectra(
        self,
        spectra_shared_name,
        mask_shared_name,
        spectra_shape,
        spec_dtype,
        channel_downsampling_factor,
        utc_start,
        file_name,
        beam_start_idx,
        beam_end_idx,
    ):
        """
        Class for reading raw files directly to shared spectra.

        Based on the sps_common.conversion.read_huff_msgpack (now in rfi_mitigation.conversion)

        Parameters
        ==========
        spectra_shared_name: str
            Name of shared spectra

        mask_shared_name: str
            Name of shared mask

        spectra_shape: tuple(int)
            Shape of spectra

        spec_dtype:
            dtype of spectra

        channel_downsampling_factor: int
            Channel downsampling factor relative to 16k channels.
            (i.e., the number of channels requested is 16384 // channel_downsampling_factor)

        utc_start: int
            Unix utc stamp at the start of the observation

        file_name: str
            File that is being read.

        beam_start_idx: int
            Index of the time step where this beam should start

        beam_end_idx: int
            Index of the time step where this beam should start
        """
        subfiles = []
        try:
            with open(file_name, "rb") as f:
                int_file = l1_io.IntensityFile.from_file(
                    f, shape=(16384 // channel_downsampling_factor, None)
                )
                sps_chunks = int_file.get_chunks()
        except IndexError:
            log.error(f"File {file_name} defect.")
            return []
        last_fpga0 = 0
        last_nfreq = 0
        shared_spectra = shared_memory.SharedMemory(name=spectra_shared_name)
        spectra = np.ndarray(spectra_shape, dtype=spec_dtype, buffer=shared_spectra.buf)
        shared_mask = shared_memory.SharedMemory(name=mask_shared_name)
        rfi_mask = np.ndarray(spectra_shape, dtype=bool, buffer=shared_mask.buf)

        if spectra.shape[0] != sps_chunks[0].chunk_header.nfreq:
            log.error("Shape mismatch between spectra and file.")
            return []

        utc_start_obj = Time(utc_start, format="unix", scale="utc")
        # sort the chunks by start time to loop through
        for i, chunk in enumerate(
            sorted(sps_chunks, key=lambda x: x.chunk_header.fpga0)
        ):
            chunk_header = chunk.chunk_header
            if i == 0:
                subfiles.append([chunk])
                last_fpga0 = chunk_header.fpga0
                last_nfreq = chunk_header.nfreq
                continue
            # split into different subfiles if nchan is different, or they are not contiguous
            # (can be tuned to be not contiguous by n chunks and still be processed together)
            not_contiguous = (
                round((chunk_header.fpga0 - last_fpga0) * 2.56e-6 / TSAMP) != 1024
            )
            different_nchan = chunk_header.nfreq != last_nfreq
            if not_contiguous or different_nchan:
                subfiles.append([chunk])
            else:
                subfiles[-1].extend([chunk])
            last_fpga0 = chunk_header.fpga0
            last_nfreq = chunk_header.nfreq

        spec_slices = []

        for subfile in subfiles:
            for i, chunk in enumerate(subfile):
                current_time = Time(
                    subfile[i].chunk_header.frame0_nano * 1e-9,
                    format="unix",
                    scale="utc",
                ) + TimeDelta(
                    0.0, subfile[i].chunk_header.fpga0 * 2.56e-6, format="sec"
                )
                # Expect indices without cuts
                start_idx = round(
                    (current_time - utc_start_obj).to_value("sec") / TSAMP
                )
                end_idx = start_idx + chunk.chunk_header.ntime

                actual_start_idx = max(start_idx, beam_start_idx, 0)
                actual_end_idx = min(end_idx, beam_end_idx, spectra_shape[1])

                if (actual_end_idx - actual_start_idx) < 1:
                    continue
                else:
                    start_offset = actual_start_idx - start_idx
                    end_offset = actual_end_idx - end_idx
                    spec_slice = slice(actual_start_idx, actual_end_idx)
                    chunk_slice = slice(
                        start_offset, chunk.chunk_header.ntime + end_offset
                    )
                    chunk_intensity = chunk.data[:, chunk_slice]
                    chunk_mask = chunk.bad_mask[:, chunk_slice].astype(bool)
                    chunk_intensity *= np.sqrt(chunk.variance)[:, np.newaxis]
                    chunk_intensity += chunk.means[:, np.newaxis]
                    spectra[:, spec_slice] = chunk_intensity

                    zero_mask = np.zeros_like(chunk_intensity, dtype=bool)
                    zero_mask[chunk_intensity <= 0] = True
                    zero_mask[~np.isfinite(chunk_intensity)] = True
                    full_mask = np.logical_or(zero_mask, ~chunk_mask)
                    rfi_mask[:, spec_slice] = full_mask
                    spec_slices.append(spec_slice)
        shared_spectra.close()
        shared_mask.close()
        return spec_slices

    def fill_and_norm_shared_spectra(
        self,
        spectra_shared_name,
        mask_shared_name,
        spectra_shape,
        spec_dtype,
        nsamp,
        current_freq_slice,
        current_time_slice,
        detrend_data=False,
        detrend_nsamp=None,
        add_local_median=False,
    ):
        """
        Fill and detrend spectra.

        When converting the RFIPipeline spectra to a form to be written to a file, we
        need to fill in the masked data with a reasonable alternative. The code now
        gives several options in forming the masked data. The current default is to
        replace the masked values with a local median over a nsamp segment and then set
        the local median to zero.

        There are options to apply a linear detrending of the data and to retain the
        local median values instead.

        Parameters
        ==========
        spectra_shared_name: str
            Name of shared spectra

        mask_shared_name: str
            Name of shared mask

        spectra_shape: tuple(int)
            Shape of spectra

        spec_dtype:
            dtype of spectra

        nsamp: int
            Minimum number of time samples for each individual segment to process.
            Must be equal or multiple of the
            block_size.

        current_freq_slice: slice
            Slice of frequencies that are processed in this function

        current_freq_slice: slice
            Slice of time steps that are processed in this function

        detrend_data: bool
            Whether to detrend the data after masking. Default = False

        detrend_nsamp: int or None
            The number of samples to detrend together. Default = None (equal to the size of
            data chunk being processed)

        add_local_median: bool
            Whether to add local median to the detrended data. Default = False
        """
        shared_spectra = shared_memory.SharedMemory(name=spectra_shared_name)
        spectra_full = np.ndarray(
            spectra_shape, dtype=spec_dtype, buffer=shared_spectra.buf
        )
        spectra = spectra_full[current_freq_slice, current_time_slice]
        shared_mask = shared_memory.SharedMemory(name=mask_shared_name)
        rfi_mask_full = np.ndarray(spectra_shape, dtype=bool, buffer=shared_mask.buf)
        rfi_mask = rfi_mask_full[current_freq_slice, current_time_slice]
        mask_fraction = rfi_mask.sum() / rfi_mask.size
        if mask_fraction == 1.0:
            spectra[:] = 0
            shared_spectra.close()
            shared_mask.close()
            return

        spectra_masked = np.ma.masked_array(spectra, mask=rfi_mask)
        ntime = spectra.shape[1]
        with threadpool_limits(limits=1, user_api="blas"):
            # determine sets of blocks to process
            num_blocks = ntime // nsamp

            if num_blocks <= 1:
                used_block = ntime
                local_median = np.ma.median(spectra_masked)
                filled_local_medians = local_median
                spectra[rfi_mask] = filled_local_medians
            else:
                if (spectra_masked.shape[0] % num_blocks) == 0:
                    # Solution when shapes are divisible
                    spectra_masked_block = spectra_masked.reshape(
                        spectra_masked.shape[0], num_blocks, -1
                    )
                    used_block = spectra_masked_block.shape[2]
                    local_medians = np.ma.median(spectra_masked_block, axis=2)
                    filled_local_medians = np.repeat(local_medians, used_block, axis=1)
                else:
                    # Slower solution if shapes are not compatible
                    # Current settings num_block==0, but code could be optimised
                    # when actually being used
                    for block_index in range(num_blocks):
                        start_index = block_index * nsamp
                        if block_index == (num_blocks - 1):
                            end_index = spectra_masked.shape[1]
                        else:
                            end_index = (block_index + 1) * nsamp
                        used_block = end_index - start_index
                        local_median = np.ma.median(
                            spectra_masked[:, start_index:end_index], axis=1
                        )
                        current_filled_medians = np.full(
                            (spectra_masked.shape[0], used_block), local_median
                        )
                        if block_index == 0:
                            filled_local_medians = current_filled_medians
                        else:
                            filled_local_medians = np.concatenate(
                                [filled_local_medians, current_filled_medians], axis=1
                            )
                spectra[rfi_mask] = filled_local_medians[rfi_mask]

            # Only tested the first case for now
            if detrend_data and add_local_median:
                # Only wokrs on 1D masks
                if spectra.shape[0] == 1:
                    spectra[:] = (
                        detrend_masked_row(
                            spectra,
                            mask=~rfi_mask,
                            axis=1,
                            type="linear",
                            bp=np.arange(0, spectra.shape[1], detrend_nsamp),
                        )
                        + filled_local_medians
                    )
                else:
                    spectra[:] = (
                        detrend(
                            spectra,
                            axis=1,
                            type="linear",
                            bp=np.arange(0, spectra.shape[1], detrend_nsamp),
                        )
                        + filled_local_medians
                    )
                    # Detrending will introduce trend to masked values
                    # Could at one point use detrending method that uses masked
                    # arrays which this function does not
                if num_blocks <= 1:
                    spectra[rfi_mask] = filled_local_medians
                else:
                    spectra[rfi_mask] = filled_local_medians[rfi_mask]
            elif detrend_data and not add_local_median:
                spectra[:] = detrend(
                    spectra,
                    axis=1,
                    type="linear",
                    bp=np.arange(0, spectra.shape[1], detrend_nsamp),
                )
                spectra[rfi_mask] = 0
            elif not detrend_data and not add_local_median:
                spectra[:] -= filled_local_medians
            else:
                pass
            shared_spectra.close()
            shared_mask.close()
            return

    def norm_channels_to_beam(
        self,
        spectra_shared_name,
        spectra_shape,
        spec_dtype,
        beam_to_norm,
        beam_slices,
        current_freq_slice,
    ):
        """
        Normalize the channels to the median of one chosen beam.

        Parameters
        ==========
        spectra_shared_name: str
            Name of shared spectra

        mask_shared_name: str
            Name of shared mask

        spectra_shape: tuple(int)
            Shape of spectra

        spec_dtype:
            dtype of spectra

        beam_to_norm: int
            Beam number whihc should be used as basis for the normalisation.

        beam_slices: list(slice)
            List of slices of the different beams

        current_freq_slice: slice
            Slice of frequencies that are processed in this function
        """
        shared_spectra = shared_memory.SharedMemory(name=spectra_shared_name)
        spectra_full = np.ndarray(
            spectra_shape, dtype=spec_dtype, buffer=shared_spectra.buf
        )
        spectra = spectra_full[current_freq_slice, :]
        norm_medians = np.median(spectra[:, beam_slices[beam_to_norm]])
        if norm_medians != 0:
            for current_beam, beam_slice in enumerate(beam_slices):
                if beam_to_norm != current_beam:
                    current_medians = np.median(spectra[:, beam_slice])
                    if (norm_medians != 0) and (current_medians != 0):
                        spectra[:, beam_slice] *= norm_medians / current_medians
