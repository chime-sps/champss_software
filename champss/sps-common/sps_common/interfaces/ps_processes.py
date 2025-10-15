"""Classes used for power spectra processing."""

import logging
from datetime import datetime
from glob import glob
from multiprocessing import shared_memory
from typing import List

import astropy.units as u
import h5py
import numpy as np
import pytz
from astropy.coordinates import SkyCoord
from attr import ib as attrib
from attr import s as attrs
from attr.setters import validate
from attr.validators import instance_of
from easydict import EasyDict
from numpy.lib import recfunctions as rfn
from sps_common.conversion import load_presto_dat, natural_keys
from sps_common.interfaces.single_pointing import (
    SearchAlgorithm,
    check_detection_statistic,
)

log = logging.getLogger(__name__)


@attrs
class DedispersedTimeSeries:
    """
    The interface for dedispersed time series of a single observation.

    Attributes:
        dedisp_ts (np.ndarray): The dedispersion time series in a 2D np.array with shape
        (num_dms, ts_length).

        dms (np.ndarray): The DM values of each dedispersed time series in dedisp_ts.

        ra (float) : The pointing Right Ascension of the observation.

        dec (float) : The pointing Declination of the observation.

        start_mjd (float) : The start time of the observation.

        obs_id (str) : The ID of the observation for sps-database purpose.
    """

    dedisp_ts = attrib(validator=instance_of(np.ndarray))
    dms = attrib(type=np.ndarray)
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    start_mjd = attrib(converter=float)
    obs_id = attrib(converter=str)

    def __attrs_post_init__(self):
        """Convert dm list to array and check for dm inconsistency."""
        if type(self.dms) != np.ndarray:
            self.dms = np.asarray(self.dms)
        if self.dedisp_ts.shape[0] != self.dms.size:
            raise ValueError(
                "The attribute dedisp_ts does not have shape that is consistent with"
                " the size of dms"
            )

    @dedisp_ts.validator
    @dms.validator
    def _validate_array_size(self, attribute, value):
        if value.size == 0:
            raise ValueError(
                f"The attribute {attribute.name} must have size larger than 0"
            )

    @ra.validator
    def _check_ra(self, attribute, value):
        if not 0 <= value < 360.0:
            raise ValueError(
                f"Right Ascension attribute ({attribute.name}={value}) outside range"
                " [0, 360]"
            )

    @dec.validator
    def _check_dec(self, attribute, value):
        if not -90.0 <= value < 90.0:
            raise ValueError(
                f"Declination attribute ({attribute.name}={value}) outside range"
                " [-90, 90]"
            )

    @classmethod
    def from_presto_datfiles(cls, datpath, obs_id, nbits=32, prefix=""):
        """
        Class method to load dedispersed time series in the form of presto .dat file.

        Parameters
        =======
        dat: str
            path to the .dat file to load

        obs_id
            The observation ID corresponding to this data file (for database
            interaction)

        nbits: int
            Number of bits in the .dat file. Default = 32

        prefix: str
            Prefix of the files to load. Default = ""

        Returns
        =======
        PowerSpectrum: class
            The power spectrum class with power spectrum and frequency labels
        """
        ts_set = []
        dms = []
        datfiles = glob(f"{datpath}/{prefix}*.dat")
        datfiles.sort(key=lambda x: float(x.split("DM")[-1].rstrip(".dat")))
        for dat in datfiles:
            ts = load_presto_dat(dat, nbits)
            dms.append(float(dat.split("DM")[-1].rstrip(".dat")))
            ts_set.append(ts)
        dedisp_ts = np.asarray(ts_set)
        dms = np.asarray(dms)
        mjd = None
        ra = None
        dec = None

        try:
            with open(datfiles[0].replace(".dat", ".inf")) as ifile:
                lines = ifile.readlines()
                for line in lines:
                    if "MJD" in line:
                        mjd = float(line.strip().split(" ")[-1])
                    elif "Right Ascension" in line:
                        ra = str(line.strip().split(" ")[-1])
                    elif "Declination" in line:
                        dec = str(line.strip().split(" ")[-1])
        except FileNotFoundError as e:
            log.error(e)
            return None

        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        radeg = coord.ra.deg
        decdeg = coord.dec.deg

        return cls(
            dedisp_ts=dedisp_ts,
            dms=dms,
            ra=radeg,
            dec=decdeg,
            start_mjd=mjd,
            obs_id=obs_id,
        )


@attrs
class PowerSpectra:
    """
    Power spectra of a given observation.

    Attributes:
        power_spectra (np.ndarray or List(np.ndarray)): The power spectra
        in either a 2D np.ndarray or a list of 1D np.ndarray.

        dms (np.ndarray): The DM values of each spectrum in power_spectra.

        freq_labels (np.ndarray): The trial frequency values of each bin in the
        power spectra.

        ra (float) : The pointing Right Ascension of the observation.

        dec (float) : The pointing Declination of the observation.

        datetimes (datetime) : The list of datetime objects
        showing the dates in which the stack consists of

        num_days (int) : The number of days stacked within the power spectra

        beta (float) : The barycentric correction factor of the power spectra
        (0 if num_days > 1)

        bad_freq_indices (List(List(int)) or List(np.ndarray)) : Nested
        lists of bad frequency indices to mask of power spectra from
        different days in the stack.

        obs_id (List) : A list of observation id of the observations in
        the power spectra stack

        power_spectra_shared (multiprocessing.shared_memory.SharedMemory or None):
        Shared memory object storing the power spectrum if it is kept in stored memory.
    """

    power_spectra = attrib()
    dms = attrib(type=np.ndarray)
    freq_labels = attrib(type=np.ndarray)
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    datetimes = attrib(type=List[datetime], on_setattr=validate)  # type: ignore
    num_days = attrib(converter=int)
    beta = attrib(converter=float)
    bad_freq_indices = attrib(type=List[List[int]], on_setattr=validate)  # type: ignore
    obs_id = attrib(type=List[str], on_setattr=validate)  # type: ignore
    rn_medians = attrib(type=np.ndarray, default=None)
    rn_scales = attrib(type=np.ndarray, default=None)
    rn_dm_indices = attrib(type=np.ndarray, default=None)
    power_spectra_shared = attrib(default=None)

    def __attrs_post_init__(self):
        """Convert list to arrays."""
        if type(self.dms) != np.ndarray:
            self.dms = np.asarray(self.dms)
        if type(self.freq_labels) != np.ndarray:
            self.freq_labels = np.asarray(self.freq_labels)

    @power_spectra.validator
    def _validate_power_spectra_shape(self, attribute, value):
        if type(value) == np.ndarray:
            if value.shape != (self.dms.size, self.freq_labels.size):
                raise ValueError(
                    f"{attribute.name} does not have the shape (dms.size,"
                    f" freq_labels.size) of ({self.dms.size}, {self.freq_labels.size}"
                )
        elif type(value) == list:
            if type(value[0]) != np.ndarray:
                raise TypeError(
                    f"{attribute.name} must be either an np.ndarray or a list of"
                    " np.ndarray"
                )
            ps_shape = (len(self.power_spectra), len(self.power_spectra[0]))
            if ps_shape != (self.dms.size, self.freq_labels.size):
                raise ValueError(
                    f"{attribute.name} does not have the shape (dms.size,"
                    f" freq_labels.size) of ({self.dms.size}, {self.freq_labels.size})"
                )
        else:
            raise TypeError(
                f"{attribute.name} must be either an np.ndarray or a list of np.ndarray"
            )

    @datetimes.validator
    def _validate_datetimes(self, attribute, value):
        if type(value) != list:
            raise AttributeError(f"The data type of {attribute.name} is not list.")
        for val in value:
            if type(val) != datetime:
                raise AttributeError(
                    f"The elements of {attribute.name} are not datetime."
                )
            if val.tzinfo != pytz.utc:
                raise ValueError(
                    f"The tzinfo of the elements of {attribute.name} are not pytz.utc"
                )

    @bad_freq_indices.validator
    def _validate_bad_freq_indices(self, attribute, value):
        if type(value) != list:
            raise AttributeError(
                f"The data type of {attribute.name} is not a nested list. It is a"
                f" {type(value)}"
            )
        for val in value:
            if len(val) > 0:
                if type(val) == list:
                    if type(val[0]) != int:
                        raise AttributeError(
                            f"The elements of {attribute.name} are not int."
                        )
                elif type(val) == np.ndarray:
                    if not np.issubdtype(val.dtype, np.integer):
                        raise AttributeError(
                            f"The elements of {attribute.name} are not int."
                        )
                else:
                    raise AttributeError(
                        f"The data type of {attribute.name} is must be either a nested"
                        f" list or a list of np.ndarray. It is a {type(val)}"
                    )

    @obs_id.validator
    def _validate_obs_id(self, attribute, value):
        if type(value) != list:
            raise AttributeError(f"The data type of {attribute.name} is not list.")
        for val in value:
            if type(val) != str:
                raise AttributeError(f"The elements of {attribute.name} are not str.")

    @ra.validator
    def _check_ra(self, attribute, value):
        if not 0 <= value < 360.0:
            raise ValueError(
                f"Right Ascension attribute ({attribute.name}={value}) outside range"
                " [0, 360]"
            )

    @dec.validator
    def _check_dec(self, attribute, value):
        if not -90.0 <= value < 90.0:
            raise ValueError(
                f"Declination attribute ({attribute.name}={value}) outside range"
                " [-90, 90]"
            )

    @num_days.validator
    def _check_num_days(self, attribute, value):
        if not value > 0:
            raise ValueError(
                f"Num days attribute ({attribute.name}={value}) outside must be larger"
                " than 0"
            )

    @classmethod
    def read(cls, filename, nbit=32, use_shared_memory=False):
        """
        Read the power spectrum saved.

        Parameters
        =======
        filename: str
            The filename of the hdf5 power spectra file

        nbit: int
            The number of bits to load the power spectra into. Default = 32

        shared_memory: bool
            Keep spectra in shared memory
            Reading multiple PowerSpectra with shared_memory enabled may lead to the shared
            memory not being cleaned up.

        Returns
        =======
        PowerSpectra: class
            The PowerSpectra class
        """
        h5f = h5py.File(filename, "r")
        # Currently I do not know how to write the hdf5 directly into shared memory
        # Changing the dtyoe of a shared array seems problematic
        # Doing power_spectra = power_spectra.astype(f"f{int(nbit / 8)}") seems to change
        # the values sometimes
        # For now read and convert into normal memory and then create shared version
        power_spectra_ini = np.asarray(h5f.get("power spectra")).astype(
            f"f{int(nbit / 8)}"
        )
        if use_shared_memory:
            buffer_size = int(power_spectra_ini.size * nbit / 8)
            power_spectra_shared = shared_memory.SharedMemory(
                create=True, size=buffer_size
            )
            power_spectra = np.ndarray(
                h5f["power spectra"].shape,
                dtype=f"f{int(nbit / 8)}",
                buffer=power_spectra_shared.buf,
            )
            power_spectra[:] = power_spectra_ini
        else:
            power_spectra_shared = None
            power_spectra = power_spectra_ini
        freq_labels = np.ones(h5f["frequency labels"].shape)
        h5f["frequency labels"].read_direct(freq_labels)
        dms = np.ones(h5f["dms"].shape)
        h5f["dms"].read_direct(dms)
        datetimes_unix = np.ones(h5f["datetimes"].shape)
        h5f["datetimes"].read_direct(datetimes_unix)
        datetimes = [
            datetime.utcfromtimestamp(date).replace(tzinfo=pytz.UTC)
            for date in datetimes_unix
        ]

        ra = h5f.attrs["ra"]
        dec = h5f.attrs["dec"]
        obs_id = h5f.attrs["observation ids"]

        # the bad frequency index list could be empty, so handle that gracefully
        bad_freq_arrays = [key for key in h5f.keys() if "bad frequency indices" in key]
        bad_freq_arrays.sort(key=natural_keys)
        bad_freq_indices = []
        for key in bad_freq_arrays:
            if h5f[key].shape is None:
                bad_freq_indices.append([])
            elif h5f[key].len() == 0:
                bad_freq_indices.append([])
            else:
                bad_freqs = np.ones(h5f[key].shape)
                h5f[key].read_direct(bad_freqs)
                if len(bad_freq_arrays) == 1 and bad_freqs.ndim > 1:
                    bad_freq_indices = bad_freqs.astype(int).tolist()
                else:
                    # read in as a list of np.ndarray instead
                    bad_freq_indices.append(bad_freqs.astype(int))

        num_days = h5f.attrs["number of days"]
        if isinstance(h5f.attrs["beta"], h5py.Empty):
            beta = None
        elif np.isnan(h5f.attrs["beta"]):
            beta = None
        else:
            beta = h5f.attrs["beta"]

        if "rn medians" in h5f.keys():
            rn_medians = np.ones(h5f["rn medians"].shape)
            h5f["rn medians"].read_direct(rn_medians)
            rn_scales = np.ones(h5f["rn scales"].shape)
            h5f["rn scales"].read_direct(rn_scales)
            rn_dm_indices = np.ones(h5f["rn dm indices"].shape)
            h5f["rn dm indices"].read_direct(rn_dm_indices)

            return cls(
                power_spectra=power_spectra,
                dms=dms,
                freq_labels=freq_labels,
                ra=ra,
                dec=dec,
                datetimes=datetimes,
                num_days=num_days,
                beta=beta,
                bad_freq_indices=bad_freq_indices,
                obs_id=obs_id.astype(str).tolist(),
                power_spectra_shared=power_spectra_shared,
                rn_medians=rn_medians,
                rn_scales=rn_scales,
                rn_dm_indices=rn_dm_indices,
            )

        else:
            return cls(
                power_spectra=power_spectra,
                dms=dms,
                freq_labels=freq_labels,
                ra=ra,
                dec=dec,
                datetimes=datetimes,
                num_days=num_days,
                beta=beta,
                bad_freq_indices=bad_freq_indices,
                obs_id=obs_id.astype(str).tolist(),
                power_spectra_shared=power_spectra_shared,
            )

    def write(self, filename, nbit=32):
        """
        Write the PowerSpectra class to a hdf5 file.

        Parameters
        =======
        filename: str
            The filename of the hdf5 power spectra file

        nbit: int
            The number of bits to save the power spectra in. Default = 32
        """

        # CAUTION: When changing this method, make sure to adapt .check_file_keys aswell
        # Otherwise the pipeline may destroy perfectly fine stacks

        self.convert_to_nparray()
        with h5py.File(filename, "w") as h5f:
            h5f.create_dataset(
                "power spectra",
                data=self.power_spectra,
                dtype=f"f{int(nbit / 8)}",
                chunks=(1, self.freq_labels.size),
            )
            h5f.create_dataset("frequency labels", data=self.freq_labels)
            h5f.create_dataset("dms", data=self.dms)
            if self.bad_freq_indices == [[]] or not self.bad_freq_indices:
                # create an empty dataset
                log.warning("WARNING: writing empty dataset for birdies")
                h5f.create_dataset("bad frequency indices 0", dtype="i")
            else:
                for i, index_list in enumerate(self.bad_freq_indices):
                    h5f.create_dataset(
                        f"bad frequency indices {i}", data=index_list, dtype="i"
                    )
            datetimes_unix = np.asarray([date.timestamp() for date in self.datetimes])
            h5f.create_dataset("datetimes", data=datetimes_unix)

            if self.rn_medians is not None:
                h5f.create_dataset(
                    "rn medians", data=self.rn_medians, 
                )
                h5f.create_dataset("rn scales", data=self.rn_scales)
                h5f.create_dataset("rn dm indices", data=self.rn_dm_indices)

            h5f.attrs["ra"] = self.ra
            h5f.attrs["dec"] = self.dec
            h5f.attrs["observation ids"] = self.obs_id
            h5f.attrs["number of days"] = self.num_days

            if self.beta is None:
                # no barycentric correction set/required
                log.warning(
                    "WARNING: berycentric correction not set, writing empty attribute"
                )
                h5f.attrs.create("beta", data=None, dtype="f")
            else:
                h5f.attrs["beta"] = self.beta

    @classmethod
    def check_file_keys(
        cls,
        filename,
        checked_fields=["datetimes", "dms", "frequency labels", "power spectra"],
    ):
        """
        Check if the file has been properly saved by testingt if expected fields are included.

        Parameters
        =======
        filename: str
            The filename of the hdf5 power spectra file

        checked_fields: bool
            The fields that are expected in the power spectra stack.

        Returns
        =======
        file_ok: bool
            Whether the file contains the expected fields
        """
        try:
            h5f = h5py.File(filename, "r")
            file_ok = all(field in h5f.keys() for field in checked_fields)
        except:
            file_ok = False
        return file_ok

    def convert_to_nparray(self):
        """Convert power_spectra attribute to single array if it is a list of arrays."""
        if type(self.power_spectra) == list:
            self.power_spectra = np.array(
                self.power_spectra, dtype=self.power_spectra[0].dtype
            )

    def get_bin_weights(self):
        """
        Return the weight of each frequency bin.

        The weight tells you in how many days the bin was not masked.

        Returns
        =======
        bin_weights: np.ndarray
            The computed bin_weights
        """
        bin_weights = np.ones(len(self.freq_labels)) * self.num_days
        for bad_bins_per_day in self.bad_freq_indices:
            bin_weights[bad_bins_per_day] -= 1
        return bin_weights

    def get_bin_weights_fraction(self):
        """
        Return the fraction of days in which each bin was not masked.

        The fraction tells you in how many days the bin was not masked divided
        by the number of stacked days.

        Returns
        =======
        bin_weights_fraction: np.ndarray
            The computed fraction of unmasked days for each frequency bin
        """
        bin_weights_fraction = self.get_bin_weights() / self.num_days
        return bin_weights_fraction

    def unlink_shared_memory(self):
        """Unlink shared mempry when PowerSpectra is no longer used."""
        if self.power_spectra_shared is not None:
            self.power_spectra_shared.close()
            self.power_spectra_shared.unlink()
            self.power_spectra_shared = None

    def move_to_shared_memory(self):
        """If not in shared memory, move to shared memory."""
        if self.power_spectra_shared is None:
            buffer_size = int(self.power_spectra.nbytes)
            power_spectra_shared = shared_memory.SharedMemory(
                create=True, size=buffer_size
            )
            power_spectra = np.ndarray(
                self.power_spectra.shape,
                dtype=self.power_spectra.dtype,
                buffer=power_spectra_shared.buf,
            )
            power_spectra[:] = self.power_spectra
            self.power_spectra = power_spectra
            self.power_spectra_shared = power_spectra_shared
            log.info("Moved power spectra to shared memory.")


@attrs
class PowerSpectraDetections:
    """
    Power spectra detections of a given observation.

    Attributes:
        detection_list (np.ndarray): The list of detections,
        saved in a structured np.arrary with shape
        (N_detections,) and fields:
            freq (float): frequency of the detection
            dm (float): dispersion measure of the detection
            nharm (int): number of harmonics summed
            sum_pow (float): (incoherent) summed power of the harmonics
            sigma (float): significance of the detection
            injection (bool): whether the detection shares bins with an injection

        freq_spacing (float): The trial frequency spacing of
        between adjacent bins in the power spectra

        ra (float) : The pointing Right Ascension of the observation.

        dec (float) : The pointing Declination of the observation.

        sigma_min (float) : The minimum sigma of the power spectra search process

        obs_id (List) : The observation id of the observation where the
        detections are made.
    """

    detection_list = attrib(type=np.ndarray)
    freq_spacing = attrib(converter=float)
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    sigma_min = attrib(converter=float)
    obs_id = attrib(type=List[str])

    @detection_list.validator
    def _check_detection_list(self, attribute, value):
        dit = [
            ("freq", float),
            ("dm", float),
            ("nharm", int),
            ("harm_pows", [("power", float, (32,)), ("freq", float, (32,))]),
            ("sigma", float),
            ("injection", bool),
        ]
        if value.dtype != np.dtype(dit):
            raise AttributeError(
                f"Array attribute ({attribute.name}, dtype={value.dtype}) is not of the"
                f" correct form.Required dtype is {dit}"
            )

    @obs_id.validator
    def _validate_obs_id(self, attribute, value):
        if type(value) != list:
            raise AttributeError(f"The data type of {attribute.name} is not list.")
        for val in value:
            if type(val) != str:
                raise AttributeError(f"The elements of {attribute.name} are not str.")

    @classmethod
    def read(cls, filename):
        """
        Read the power spectra detections file saved.

        Parameters
        =======
        filename: str
            The filename of the hdf5 power spectra detections file

        Returns
        =======
        PowerSpectra: class
            The PowerSpectraDetections class
        """
        h5f = h5py.File(filename, "r")
        detection_list = np.array(h5f["detection list"])
        freq_spacing = h5f.attrs["frequency spacing"]
        ra = h5f.attrs["ra"]
        dec = h5f.attrs["dec"]
        sigma_min = h5f.attrs["minimum sigma"]
        obs_id = h5f.attrs["observation id"]

        return cls(
            detection_list=detection_list,
            freq_spacing=freq_spacing,
            ra=ra,
            dec=dec,
            sigma_min=sigma_min,
            obs_id=obs_id.astype(str).tolist(),
        )

    def write(self, filename):
        """
        Write the power spectra detections to an hdf5 file.

        Parameters
        =======
        filename: str
            The filename of the hdf5 power spectra detections file
        """
        with h5py.File(filename, "w") as h5f:
            h5f.create_dataset("detection list", data=self.detection_list)
            h5f.attrs["frequency spacing"] = self.freq_spacing
            h5f.attrs["ra"] = self.ra
            h5f.attrs["dec"] = self.dec
            h5f.attrs["minimum sigma"] = self.sigma_min
            h5f.attrs["observation id"] = [str(idx) for idx in self.obs_id]


@attrs
class Cluster:
    """
    A cluster of detections.

    Attributes:
    ===========
    detections (np.ndaray): detections output from PowerSpectra search. A numpy
        structured array with fields "dm", "freq", "sigma", "nharm", "harm_idx",
        "injection"

    Post-init:
    ==========
    max_sig_det (np.ndarray): highest-sigma detection
    freq (float): frequency of the highest-sigma detection
    dm (float): DM of the highest-sigma detection
    sigma (float): sigma of the highest-sigma detection
    nharm (int): nharm of the highest-sigma detection
    harm_idx (np.ndarray): harm_idx of the highest-sigma detection
    injection_index (bool): Index of the injection in the injection_dicts list.
                        -1 if not associated with an injection

    Properties
    ==========
    ndetections (int): Number of detections
    unique_freqs (np.ndarray):  Unique frequencies in detections
    unique_dms (np.ndarray):  Unique DMs in detections
    num_unique_freqs (int): Number of unique freqs in detections
    num_unique_dms (int): Number of unique DMs in detections
    log_p_dm (float): log_10 (Period / DM)
    """

    detections: np.ndarray = attrib()
    max_sig_det: np.ndarray = attrib()
    freq: float = attrib()
    dm: float = attrib()
    sigma: float = attrib()
    nharm: int = attrib()
    harm_idx: np.ndarray = attrib()
    injection_index: float = attrib()

    @classmethod
    def from_raw_detections(cls, detections):
        max_sig_det = detections[np.argmax(detections["sigma"])]
        init_dict = dict(
            max_sig_det=max_sig_det,
            freq=max_sig_det["freq"],
            dm=max_sig_det["dm"],
            sigma=max_sig_det["sigma"],
            nharm=max_sig_det["nharm"],
            harm_idx=max_sig_det["harm_idx"],
            injection_index=max_sig_det["injection"],
            detections=detections,
        )
        return cls(**init_dict)

    @classmethod
    def from_saved_psdc(cls, detections, max_sig_det):
        init_dict = dict(
            max_sig_det=max_sig_det,
            freq=max_sig_det["freq"],
            dm=max_sig_det["dm"],
            sigma=max_sig_det["sigma"],
            nharm=max_sig_det["nharm"],
            harm_idx=max_sig_det["harm_idx"],
            injection_index=max_sig_det["injection"],
            detections=detections,
        )
        return cls(**init_dict)

    def filter_nharm(self):
        """Filter the cluster, only keeping detections which have the same nharm as the
        highest-sigma detection.
        """
        self.detections = self.detections[self.detections["nharm"] == self.nharm]

    def remove_harm_idx(self):
        """Remove the harm_idx field from detections (Note self.harm_idx will still
        exist)
        """
        if "harm_idx" in self.detections.dtype.names:
            self.detections = rfn.drop_fields(self.detections, "harm_idx")

    def remove_harm_pow(self):
        """Remove the harm_pow field from detections."""
        if "harm_pow" in self.detections.dtype.names:
            self.detections = rfn.drop_fields(self.detections, "harm_pow")

    @property
    def ndetections(self):
        """Number of detections."""
        return self.detections.shape[0]

    @property
    def unique_freqs(self):
        """Unique frequencies in detections."""
        return np.unique(self.detections["freq"])

    @property
    def unique_dms(self):
        """Unique DMs in detections."""
        return np.unique(self.detections["dm"])

    @property
    def num_unique_freqs(self):
        """Number of unique frequencies in detections."""
        return len(self.unique_freqs)

    @property
    def num_unique_dms(self):
        """Number of unique DMs in detections."""
        return len(self.unique_dms)

    @property
    def log_p_dm(self):
        """Log (P/DM)"""
        if self.dm == 0 or self.freq == 0:
            return np.NaN
        else:
            return np.log10(1 / (self.freq * self.dm))


@attrs
class PowerSpectraDetectionClusters:
    """
    Clusters in DM-freq-harmonic space of detections/triggers from DBSCAN (run on data
    from a single pointing).

    Attributes
    ==========
    clusters: dict(np.array): A dictionary of Cluster objects

    summary: dict(dict): A dictionary of dictionary showing the summary
    of each cluster

    ra: float: The right ascension of the detections

    dec: float: The declination of the detections

    detection_statistic (SearchAlgorithm) : Whether power spectra (power_spec)
    or hhat was used to find detections in this cluster

    freq_spacing: float: The frequency spacing between adjacent frequency
    bins of the power spectra

    obs_id: List(str): A list of strings of the observation ids in the power
    spectra detections

    datetimes: List(datetime.datetime): A list of datetimes of the power
    spectra detections. For now not included when writing the file.
    """

    clusters = attrib()
    summary = attrib()
    ra: float = attrib()
    dec: float = attrib()
    detection_statistic = attrib(
        type=SearchAlgorithm, validator=check_detection_statistic
    )
    threshold: float = attrib()
    freq_spacing: float = attrib()
    obs_id: List = attrib()
    datetimes: List = attrib(default=[])
    injection_dicts: List = attrib(default=[])

    @clusters.validator
    def validate_cluster(self, attribute, value):
        """Validate if clusters have expected format."""
        for v in value:
            if not isinstance(value[v], Cluster):
                raise TypeError(f"Entry {v} in {attribute.name} must be a Cluster")
            val_dtype = set(value[v].detections.dtype.names)
            acceptable_dtype_names = [
                {"freq", "dm", "nharm", "sigma"},
                {"freq", "dm", "nharm", "harm_idx", "sigma"},
                {"freq", "dm", "nharm", "harm_idx", "harm_pow", "sigma"},
                {"freq", "dm", "nharm", "harm_idx", "harm_pow", "injection", "sigma"},
            ]
            if val_dtype not in acceptable_dtype_names:
                raise TypeError(
                    f"The dtype of detections of entry {v} in {attribute.name} must"
                    " have 'freq', 'dm', 'nharm' and 'sigma' (and optionally"
                    " 'harm_idx', 'harm_pow', 'injection')"
                )

    @summary.validator
    def validate_summary(self, attribute, value):
        """Validate if summary has expected format."""
        for v in value:
            if not all(
                k in value[v].keys()
                for k in [
                    "dm",
                    "freq",
                    "nharm",
                    "sigma",
                    "harm_idx",
                    "injection",
                ]
            ):
                raise TypeError(
                    f"The dictionary for entry {v} for {attribute.name} does not"
                    " contain the correct keys"
                )

    @ra.validator
    def _check_ra(self, attribute, value):
        if not 0 <= value < 360.0:
            raise ValueError(
                f"Right ascension attribute ({attribute.name}={value}) outside range "
                "[0, 360) degrees"
            )

    @dec.validator
    def _check_dec(self, attribute, value):
        if not -30.0 <= value < 90.0:
            raise ValueError(
                f"Declination attribute ({attribute.name}={value}) outside range "
                "[-30, 90) degrees"
            )

    @threshold.validator
    def _check_sigma_threshold(self, attribute, value):
        if value <= 0:
            raise ValueError(f"({attribute.name}={value}) must be greater than 0")

    def __iter__(self):
        """Allow iteration through the class."""
        self.iteration_counter = 0
        return self

    def __next__(self):
        """Iterate through clusters and return EasyDict containing information."""
        if self.iteration_counter < len(self.clusters):
            out_dict = EasyDict(
                {
                    "cluster": self.clusters[self.iteration_counter],
                    "ra": self.ra,
                    "dec": self.dec,
                    "detection_statistic": self.detection_statistic,
                    "obs_id": self.obs_id,
                    "datetimes": self.datetimes,
                }
            )
            self.iteration_counter += 1
            return out_dict
        else:
            raise StopIteration

    @classmethod
    def read(cls, filename):
        """
        Read SinglePointingDetectionClusters from an hdf5 file.

        Parameters
        ----------
        filename: str
            Name of the file to read from.
        """
        h5f = h5py.File(filename, "r")
        clusters = {}
        summary = {}
        cluster_ids = h5f["cluster_ids"]
        max_sig_det_dt = [
            ("freq", float),
            ("dm", float),
            ("nharm", int),
            ("harm_idx", int, (32,)),
            ("injection", bool),
            ("sigma", float),
        ]
        if h5f.attrs["harm_idx_included"]:
            det_dtype = max_sig_det_dt
        else:
            det_dtype = [
                ("freq", float),
                ("dm", float),
                ("nharm", int),
                ("sigma", float),
                ("injection", bool),
            ]

        ra = h5f.attrs["ra"]
        dec = h5f.attrs["dec"]
        detection_statistic = SearchAlgorithm(h5f.attrs["detection_statistic"])
        freq_spacing = h5f.attrs["freq_spacing"]
        obs_id = [i for i in h5f.attrs["obs_id"]]
        threshold = h5f.attrs["threshold"]

        for n, cluster_id in enumerate(cluster_ids):
            detections = np.array(
                h5f[f"clusters_{cluster_id}"][()],
                dtype=det_dtype,
            )
            max_sig_det = np.array(
                h5f[f"max_sig_det_{cluster_id}"][()],
                dtype=max_sig_det_dt,
            )
            summary[cluster_id] = dict(
                dm=max_sig_det["dm"],
                freq=max_sig_det["freq"],
                nharm=max_sig_det["nharm"],
                sigma=max_sig_det["sigma"],
                harm_idx=max_sig_det["harm_idx"],
                injection=max_sig_det["injection"],
            )
            clusters[cluster_id] = Cluster.from_saved_psdc(
                detections=detections,
                max_sig_det=max_sig_det,
            )

        return cls(
            clusters=clusters,
            summary=summary,
            ra=ra,
            dec=dec,
            detection_statistic=detection_statistic,
            threshold=threshold,
            freq_spacing=freq_spacing,
            obs_id=obs_id,
        )

    def write(self, filename):
        """
        Write SinglePointingDetectionClusters to an hdf5 file.

        Parameters
        ----------
        filename: str
            Name of the file to write to
        """
        log.info(f"saved candidates to {filename}")
        cluster_ids = []
        with h5py.File(filename, "w") as h5f:
            for c in self.clusters:
                cluster_ids.append(c)
                h5f.create_dataset(
                    f"clusters_{c}",
                    data=np.array(self.clusters[c].detections),
                )
                h5f.create_dataset(
                    f"max_sig_det_{c}",
                    data=np.array(self.clusters[c].max_sig_det),
                )

            h5f.create_dataset("cluster_ids", data=cluster_ids)
            h5f.attrs["ra"] = self.ra
            h5f.attrs["dec"] = self.dec
            h5f.attrs["detection_statistic"] = self.detection_statistic.value
            h5f.attrs["threshold"] = self.threshold
            h5f.attrs["freq_spacing"] = self.freq_spacing
            h5f.attrs["obs_id"] = self.obs_id
            if "harm_idx" in self.clusters[c].detections.dtype.names:
                h5f.attrs["harm_idx_included"] = True
            else:
                h5f.attrs["harm_idx_included"] = False
