import datetime
import logging
from functools import partial
from itertools import repeat
from multiprocessing import Pool

import attr
import numpy as np
from attr.validators import deep_iterable, instance_of
from rfi_mitigation.pipeline import RFIPipeline
from sps_common.constants import TSAMP
from sps_common.conversion import fill_and_norm, read_huff_msgpack, subband
from sps_common.filterbank import get_dtype
from sps_common.interfaces.beamformer import SkyBeam
from sps_common.interfaces.rfi_mitigation import SlowPulsarIntensityChunk
from sps_databases import db_api

from beamformer import NotEnoughDataError
from beamformer.utilities.common import get_data_list, get_intensity_set

log = logging.getLogger(__name__)


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
        assert (
            value % self.block_size == 0
        ), f"masking timescale must be a multiple of block_size {self.block_size}"

    @mask_channel_frac.validator
    def _validate_mask_channel_frac(self, attribute, value):
        assert 0 <= value <= 1, "mask channel fraction must be between 0 and 1"

    @detrend_nsamp.validator
    def _validate_detrend_nsamp(self, attribute, value):
        if not (value is None):
            assert isinstance(
                value, int
            ), "detrend number of samples must be an integer or None"

    @beam_to_normalise.validator
    def _validate_beam_to_normalise(self, attribute, value):
        if not (value is None):
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
        if self.run_rfi_mitigation:
            self.rfi_pipeline = RFIPipeline(self.masking_dict, make_plots=False)

    def form_skybeam(self, active_pointing, num_threads=1):
        """
        Beamform a given active pointing.

        Parameter
        =======
        active_pointing: ActivePointing
            Active pointing class containing the required properties to beamform the pointing.

        num_threads: int
            Number of threads to run the parallel processing by.

        Return
        =======
        skybeam: SkyBeam
            SkyBeam object containing the spectra and related properties of the beamformed data.
        """
        pool = Pool(num_threads)
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
        data_list = get_data_list(
            active_pointing.max_beams,
            basepath=self.basepath,
            extn=self.extn,
            per_beam_list=True,
        )
        spectra = np.zeros(
            shape=(active_pointing.nchan, active_pointing.ntime),
            dtype=get_dtype(self.nbits),
        )
        rfi_mask = np.ones(shape=spectra.shape, dtype=bool)
        processed_count = 0
        return_median = False
        if self.beam_to_normalise is not None:
            return_median = True
            sorted_beam_list = [0, 1, 2, 3]
            sorted_beam_list[self.beam_to_normalise], sorted_beam_list[0] = (
                sorted_beam_list[0],
                sorted_beam_list[self.beam_to_normalise],
            )
            data_list = [data_list[i] for i in sorted_beam_list]
            channel_medians = None
        for i, beam_path in enumerate(data_list):
            if not beam_path:
                # Sometimes there are no beamformed data for a given beam
                log.info(
                    "There are no beamformed data for beam"
                    f' {active_pointing.max_beams[i]["beam"]}'
                )
                continue
            intensity_set = get_intensity_set(
                beam_path, min_set_length=self.masking_timescale, tsamp=self.tsamp
            )
            for intensities in intensity_set:
                if len(intensities) == 1:
                    intensity = self.read_and_rfi(
                        intensities[0], 16384 // active_pointing.nchan
                    )
                else:
                    intensity = self.get_combined_intensity(
                        intensities, active_pointing, pool
                    )
                active_beam = None
                for beam in active_pointing.max_beams:
                    if beam["beam"] == intensity.beam_number:
                        active_beam = beam
                if not active_beam:
                    log.info("Current beam is not used by this sky beam")
                    continue
                log.info(
                    "Processing {:.3f}s of data from beam {}".format(
                        intensity.ntime * self.tsamp, intensity.beam_number
                    )
                )
                if num_threads == 1:
                    masked_intensity, medians = fill_and_norm(
                        intensity.get_masked_data(),
                        intensity.nchan,
                        intensity.ntime,
                        self.masking_timescale,
                        nsub=np.min([self.nsub, intensity.nchan]),
                        block_size=self.block_size,
                        detrend_data=self.detrend_data,
                        detrend_nsamp=self.detrend_nsamp,
                        add_local_median=self.add_local_median,
                        return_median=return_median,
                    )
                else:
                    if self.nsub >= intensity.nchan:
                        masked_subbands, subbands_medians = list(
                            zip(
                                *pool.map(
                                    partial(
                                        fill_and_norm,
                                        nchan=1,
                                        ntime=intensity.ntime,
                                        nsamp=self.masking_timescale,
                                        nsub=1,
                                        block_size=self.block_size,
                                        detrend_data=self.detrend_data,
                                        detrend_nsamp=self.detrend_nsamp,
                                        add_local_median=self.add_local_median,
                                        return_median=return_median,
                                    ),
                                    intensity.get_masked_data(),
                                )
                            )
                        )
                        masked_intensity = np.vstack(masked_subbands)
                        if self.beam_to_normalise is not None:
                            medians = np.hstack(subbands_medians)
                    else:
                        subbands = intensity.get_masked_data().reshape(
                            self.nsub, intensity.nchan // self.nsub, intensity.ntime
                        )
                        masked_subbands, subbands_medians = list(
                            zip(
                                *pool.map(
                                    partial(
                                        fill_and_norm,
                                        nchan=subbands.shape[1],
                                        ntime=intensity.ntime,
                                        nsamp=self.masking_timescale,
                                        nsub=1,
                                        block_size=self.block_size,
                                        detrend_data=self.detrend_data,
                                        detrend_nsamp=self.detrend_nsamp,
                                        add_local_median=self.add_local_median,
                                        return_median=return_median,
                                    ),
                                    subbands,
                                )
                            )
                        )
                        masked_intensity = np.vstack(masked_subbands).reshape(
                            -1, intensity.ntime
                        )
                        if self.beam_to_normalise is not None:
                            medians = np.hstack(subbands_medians)
                if self.beam_to_normalise is not None:
                    log.info(
                        "Normalise data to the beam response of beam"
                        f" {self.beam_to_normalise}"
                    )
                    if channel_medians is None:
                        channel_medians = medians
                    else:
                        masked_intensity *= channel_medians[:, None]
                        masked_intensity /= medians[:, None]
                spectra, rfi_mask, processed_count = self.append_intensity_chunk(
                    spectra,
                    rfi_mask,
                    utc_start,
                    active_beam,
                    masked_intensity,
                    intensity.get_mask(),
                    intensity.start_unix_time,
                    intensity.start_unix_time + intensity.ntime * self.tsamp,
                    processed_count,
                )
                if processed_count >= active_pointing.ntime:
                    break
            if processed_count >= active_pointing.ntime:
                break
        if processed_count / active_pointing.ntime < self.min_data_frac:
            raise NotEnoughDataError(
                f"Only {processed_count / active_pointing.ntime * 100:.1f}% of data is"
                f" available.A minimum of {self.min_data_frac * 100:.1f}% of data is"
                " required to beamformthe pointing."
            )
        log.info("Beamforming completed")
        spectra, rfi_mask = self.zero_replace_spectra(
            spectra,
            rfi_mask,
            nsub=np.min(
                [self.nsub, spectra.shape[0]],
            ),
            flatten_bandpass=self.flatten_bandpass,
        )
        if self.update_db:
            log.info("Updating Database")
            self.update_database(active_pointing.obs_id, rfi_mask, utc_start)
        pool.close()
        pool.join()
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
        return skybeam

    def append_intensity_chunk(
        self,
        spectra,
        rfi_mask,
        spectra_utc_start,
        beam_start_end,
        intensity,
        intensity_mask,
        intensity_utc_start,
        intensity_utc_end,
        processed_count,
    ):
        """
        Filter and apply the intensity data for the chunk used to skybeam.

        Parameters
        =======
        spectra: np.ndarray
            The 2-D spectra where the beamformed skybeam is stored

        rfi_mask: np.ndarray
            The rfi mask of the skybeam

        spectra_utc_start: float
            The utc start time of the skybeam spectra

        beam_start_end: dict
            The dict containing the start and end time of the intensity data
            of the beam to process

        intensity: np.ndarray
            The masked intensity data.

        intensity_mask: np.ndarray
            The masking information of the intensity data

        intensity_utc_start: float
            The start time of the intensity data in unix utc

        intensity_utc_end: float
            The end time of the intensity data in unix utc

        processed_count: int
            The counter to track if the beamforming process is completed

        Return
        =======
        spectra: ndarray
            The updated 2-D spectra where the beamformed skybeam is stored

        rfi_mask: ndarray or None
            The updated rfi mask of the skybeam

        processed_count: int
            The updated counter to track if the beamforming process is completed
        """
        if (
            intensity_utc_end < beam_start_end["utc_start"]
            or intensity_utc_start > beam_start_end["utc_end"]
        ):
            return spectra, rfi_mask, processed_count
        if intensity_utc_start < beam_start_end["utc_start"]:
            start_chunk = int(
                round((beam_start_end["utc_start"] - intensity_utc_start) / self.tsamp)
            )
            intensity_utc_start += start_chunk * self.tsamp
            intensity = intensity[:, start_chunk:]
            intensity_mask = intensity_mask[:, start_chunk:]
        if intensity_utc_end > beam_start_end["utc_end"]:
            from_end_chunk = int(
                round((intensity_utc_end - beam_start_end["utc_end"]) / self.tsamp)
            )
            intensity_utc_end -= from_end_chunk * self.tsamp
            intensity = intensity[:, :-from_end_chunk]
            intensity_mask = intensity_mask[:, :-from_end_chunk]
        spectra, rfi_mask, processed_count = self.append_intensity(
            spectra,
            rfi_mask,
            spectra_utc_start,
            intensity,
            intensity_mask,
            intensity_utc_start,
            intensity_utc_end,
            processed_count,
        )
        return spectra, rfi_mask, processed_count

    def append_intensity(
        self,
        spectra,
        rfi_mask,
        spectra_utc_start,
        intensity,
        intensity_mask,
        intensity_utc_start,
        intensity_utc_end,
        processed_count,
    ):
        """
        Append intensity data chunk to spectra.

        Append the beamformer spectra with the intensity data from the FRB beam with a
        start time for the intensity data chunk.

        Parameters
        =======
        spectra: np.ndarray
            The 2-D spectra where the beamformed skybeam is stored

        rfi_mask: np.ndarray
            The rfi mask of the skybeam

        spectra_utc_start: float
            The utc start time of the skybeam spectra

        intensity: np.ndarray
            The masked intensity data.

        intensity_mask: np.ndarray
            The masking information of the intensity data

        intensity_utc_start: float
            The start time of the intensity data in unix utc

        intensity_utc_end: float
            The end time of the intensity data in unix utc

        processed_count: int
            The counter to track if the beamforming process is completed

        Return
        =======
        spectra: ndarray
            The updated 2-D spectra where the beamformed skybeam is stored

        rfi_mask: ndarray or None
            The updated rfi mask of the skybeam

        processed_count: int
            The updated counter to track if the beamforming process is completed
        """
        if intensity.shape[0] > spectra.shape[0]:
            intensity = subband(intensity, spectra.shape[0])
            intensity_mask = subband(intensity_mask, spectra.shape[0]).astype(bool)
        time_diff = intensity_utc_start - spectra_utc_start
        intensity_length = int(
            round((intensity_utc_end - intensity_utc_start) / self.tsamp)
        )
        ichunk = int(round(time_diff / self.tsamp))
        if ichunk > spectra.shape[1]:
            return spectra, rfi_mask, processed_count
        if ichunk < 0:
            spectra[:, 0 : ichunk + intensity_length] = intensity[
                :, -ichunk:intensity_length
            ]
            processed_count += ichunk + intensity_length
            rfi_mask[:, 0 : ichunk + intensity_length] = intensity_mask[
                :, -ichunk:intensity_length
            ]
        elif ichunk + intensity_length > spectra.shape[1]:
            spectra[:, ichunk:] = intensity[:, : (spectra.shape[1] - ichunk)]
            processed_count += spectra.shape[1] - ichunk
            rfi_mask[:, ichunk:] = intensity_mask[:, : (spectra.shape[1] - ichunk)]
        else:
            spectra[:, ichunk : ichunk + intensity_length] = intensity[
                :, :intensity_length
            ]
            processed_count += intensity.shape[1]
            rfi_mask[:, ichunk : ichunk + intensity_length] = intensity_mask[
                :, :intensity_length
            ]
        return spectra, rfi_mask, processed_count

    def get_combined_intensity(self, data_set, active_pointing, pool=None):
        """
        Get combined intensities from a list of files.

        Load several intensity data and combine them together to form a longer intensity
        chunk.

        Parameters
        =======
        data_set: List(str)
            A list of intensity data to combine.

        active_pointing: ActivePointing
            The ActivePointing object of the pointing to be beamformed. It will be used
            to determine the number of channels for the output intensity chunk.

        pool: multiprocessing.pool
            Pool used for multiprocessing

        Returns
        =======
        spichunk: SlowPulsarIntensityChunk
             The SlowPulsarIntensityChunk object of the combined intensity data.
        """
        start_time = int(data_set[0].split("/")[-1].split("_")[0])
        end_time = int(data_set[-1].split("/")[-1].split("_")[1].split(".")[0]) + 1
        spectra = np.zeros(
            shape=(active_pointing.nchan, int(1024 * (end_time - start_time)))
        )
        rfi_mask = np.ones(
            shape=(active_pointing.nchan, int(1024 * (end_time - start_time)))
        ).astype(bool)
        channel_downsampling_factor = 16384 // active_pointing.nchan
        if pool is not None:
            intensities = pool.starmap(
                self.read_and_rfi, zip(data_set, repeat(channel_downsampling_factor))
            )
            found_start = False
            for intensity in intensities:
                if intensity is not None:
                    if not found_start:
                        start_unix_time = intensity.start_unix_time
                        start_mjd = intensity.start_mjd
                        beam_number = intensity.beam_number
                        cleaned = intensity.cleaned
                        found_start = True
                    spectra, rfi_mask, processed_count = self.append_intensity(
                        spectra,
                        rfi_mask,
                        start_unix_time,
                        intensity.spectra.data,
                        intensity.spectra.mask,
                        intensity.start_unix_time,
                        intensity.start_unix_time + intensity.ntime * self.tsamp,
                        0,
                    )
        else:
            found_start = False
            for i, datapath in enumerate(data_set):
                intensity = self.read_and_rfi(datapath, channel_downsampling_factor)
                if not found_start:
                    start_unix_time = intensity.start_unix_time
                    start_mjd = intensity.start_mjd
                    beam_number = intensity.beam_number
                    cleaned = intensity.cleaned
                    found_start = True
                spectra, rfi_mask, processed_count = self.append_intensity(
                    spectra,
                    rfi_mask,
                    start_unix_time,
                    intensity.spectra.data,
                    intensity.spectra.mask,
                    intensity.start_unix_time,
                    intensity.start_unix_time + intensity.ntime * self.tsamp,
                    0,
                )

        return SlowPulsarIntensityChunk(
            spectra=np.ma.array(spectra, mask=rfi_mask),
            nchan=active_pointing.nchan,
            ntime=spectra.shape[1],
            nsamp=spectra.size,
            start_unix_time=start_unix_time,
            start_mjd=start_mjd,
            beam_number=beam_number,
            cleaned=cleaned,
        )

    def zero_replace_spectra(self, spectra, rfi_mask, nsub=256, flatten_bandpass=False):
        """
        Maske zeroes, fully mask channels and flatten bandpass.

        Replacing the zeroes in each channel of the spectra with the median of the non
        zero parts. The function also zeroes out the channels where more than 75% of the
        data are masked. Replacing zeroes in this function may be redundant if the
        sps_common.conversion.fil_and_norm function is called earlier.

        Parameters
        =======
        spectra: np.ndarray
            A 2D np array of the beamformed spectra

        rfi_mask: np.ndarrau
            A 2D boolean array with the masking information of the spectra

        nsub: int
            Number of subbands used. Default: 256

        flatten_bandpass: bool
            Whether to flatten the bandpass by normalising each channel by subtracting the mean
            and dividing by the standard deviation. Default: False.

        Return
        =======
        spectra: np.ndarray
            A 2D np array of the beamformed spectra, with zeroes being replaced by
            channel median and extra masked data

        rfi_mask: np.ndarrau
            A 2D boolean array with the updated masking information of the spectra
        """
        log.info("Replacing zeroes in spectra and masking partially removed channels.")
        masked_subbands = []
        for i in range(nsub):
            _local_chan = slice(
                i * (spectra.shape[0] // nsub), (i + 1) * (spectra.shape[0] // nsub)
            )
            # We will mask all channels that contain any nans
            # If we fully understand how nans come to be to this may be changed
            _contains_nan = np.isnan(np.min(spectra[_local_chan]))
            _masked_frac = rfi_mask[_local_chan].mean()
            _contains_nan = np.isnan(np.min(spectra[_local_chan]))
            _local_max = np.max(spectra[_local_chan])
            if _masked_frac > 0.75 or _contains_nan or _local_max == 0:
                log.debug(
                    "removing channels {}-{}".format(
                        i * (spectra.shape[0] // nsub),
                        (i + 1) * (spectra.shape[0] // nsub) - 1,
                    )
                )
                spectra[_local_chan] = 0
                rfi_mask[_local_chan] = 1
                masked_subbands.append(i)
                continue
            else:
                _zero_count = (spectra[_local_chan] == 0).sum()
                # The fill_and_norm could have masked zeroes already
                if _zero_count > 0:
                    _local_median = np.median(
                        spectra[_local_chan][spectra[_local_chan] != 0]
                    )
                    spectra[_local_chan][spectra[_local_chan] == 0] = _local_median
                if flatten_bandpass:
                    # In an earlier version the mean and std was computed using only the
                    # nonzero values but nothing should be 0 at this point
                    spectra[_local_chan] = (
                        spectra[_local_chan] - spectra[_local_chan].mean()
                    ) / spectra[_local_chan].std()
        log.info(f"Masked subbands with indices: {masked_subbands}")
        log.info(
            f"Fraction of completely masked subbands: {len(masked_subbands)/nsub} "
        )
        return spectra, rfi_mask

    def update_database(self, obs_id, rfi_mask, utc_start):
        """
        Update the sps database with the mask fraction.

        Only works if the SkyBeam object has an associated obs_id.

        Parameters
        =======
        obs_id: ObjectID or str
            The observation id of the sky beam.

        rfi_mask: np.ndarray
            A 2-D boolean array of the rfi mask.

        utc_start: float
            The utc start time of the sky beam formed.
        """
        mask_fraction = rfi_mask.mean()
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

    def read_and_rfi(self, file_name, channel_downsampling_factor):
        """
        Wrapper function for reading and cleaning files used for parallel processing.

        Parameters
        ==========
        file_name: str
            The file name to be loaded.

        channel_downsampling_factor: int
            Downsampling used in the raw data file.

        Return
        =======
        intensity: SlowPulsarIntensityChunk
            The cleaned intensity.
        """
        try:
            raw_intensity = [
                SlowPulsarIntensityChunk(**o)
                for o in read_huff_msgpack(
                    file_name,
                    channel_downsampling_factor=channel_downsampling_factor,
                )
            ]
        except Exception as e:
            log.warning(f"Error Reading file {file_name}, will skip the file...")
            log.warning(e)
            return
        if self.run_rfi_mitigation:
            log.debug("Running RFI mitigation on raw .dat files")
            intensity = self.rfi_pipeline.clean(raw_intensity)[0]
        else:
            intensity = raw_intensity[0]
        return intensity
