"""Classes for single pointing FFA candidates."""

import datetime
import enum
import logging
import os
import warnings
from itertools import repeat
from multiprocessing import Pool

import click
import numpy as np
import yaml
from attr import attrib, attrs, converters
from attr.validators import deep_iterable, instance_of
MAX_SEARCH_DM_FFA = 2000
MIN_SEARCH_DM_FFA = 0
MAX_SEARCH_FREQ_FFA = 0.5
MIN_SEARCH_FREQ_FFA = 0.05
# Uncomment this when constants have been updated
# from sps_common.constants import (
#     MAX_SEARCH_DM_FFA,
#     MAX_SEARCH_FREQ_FFA,
#     MIN_SEARCH_DM_FFA,
#     MIN_SEARCH_FREQ_FFA,
# )
from sps_common.conversion import convert_ra_dec
from sps_common.interfaces.utilities import (
    filter_class_dict,
    sigma_sum_powers,
    within_range,
)
from sps_common.plotting import plot_candidate

log = logging.getLogger(__name__)


default_config_path = (
    os.path.abspath(os.path.dirname(__file__)) + "/default_sp_plot.yml"
)

detections_dtype = [
    ("dm", float),
    ("freq", float),
    ("sigma", float),
    ("width", int)
]

class SearchAlgorithm_FFA(enum.Enum):
    """
    The search algorithm type.

    Currently only periodogram relevant.
    """

    periodogram_form = 1


def check_detection_statistic_FFA(instance, attribute, value):
    """Check if the detection statistic has the correct type."""
    if not isinstance(value, SearchAlgorithm_FFA):
        raise TypeError(
            f"Detection statistic attribute ({attribute.name}={value}) must be "
            + "of SearchAlgorithm_FFA enum"
        )

@attrs
class SinglePointingCandidate_FFA:
    """
    This is the class instance to store the properties of single pointing FFA candidates.

    Specifically, after the FFA search has gone through the periodogram
    processor. The class instance will include basic properties of the candidate,
    including rotation frequency, dm, dc, ra, dec and highest detection sigma. It
    will also include a feature set to describe the cluster of detections that is
    associated with this candidate. This feature set will then be used by a Machine
    Learning Classifier to determine the likelihood of the candidate being a pulsar.

    Attributes
    ----------
    freq (float): the rotation frequency associated with the
        detection with the highest significance, in Hz

    dm (float): the dispersion measure associated with the
    detection with the highest significance, in pc/cc

    ra (float): the right ascension of the observation, in degrees

    dec (float): the declination of the observation, in degrees

    sigma (float): the highest detection significance out of all the periodogram
        peaks associated with this candidate

    summary (dict): summary information for the candidate. (see line 249 later in this file)

    features (numpy.ndarray): 1D array with a column per feature. (?!?!?!?!)

    detections (np.ndarray): Array decribing the raw detections. (!!!!!!!!)
        Saving those can be turned off. (?)

    injection (bool): bool describing whether or not a given candidate is an injection.

    injection_dict (dict): Dict describing the injection parameters.

    ndetections (int): Number of detections clustered in this candidate.

    unique_freq (np.ndarray): Array containing the unique frequencies.

    unique_dms (np.ndarray): Array containing the unique dms.

    max_sig_det (np.ndarray): A numpy structured array with fields "dm", "freq", "sigma", "width", "width_id", "injection"

    obs_id (list) : An array of observation ids of the observations this candidate came from

    rfi (bool): whether the candidate has been flagged as interference #Is it actually a boolean?

    widths_info (numpy.ndarray): Information about the highest-sigma
        detection of each harmonically related cluster of detections (from which the
        SinglePointingCandidate_FFA was derived) This INCLUDES the main detection.
        Earlier versions excluded the main detection.
        dtype.names = ['freq', 'dm', 'width', 'sigma']

    raw_widths_array (dict[np.array]): Raw harmonic power information
        that includes the powers for a set number of DMs and widths.
        Keys: "powers", "dms", "freqs", "freq_bins"
        "powers", "freqs" and freq_bins share the dimensions
        [dm_trials, harmonics, factor_trials].
        "dms" is one dimensional.
        "factor_trials" is defined by the property "period_factors" in HarmonicFilter.
        The elements of that dimension contain the raw widths where the
        base period is multiplied by [1,2,1/2,3,1/3,...].

    dm_freq_sigma (dict[np.array]): Sigma as a function of dm and frequency
    around the detected dm and frequency.
    Keys: "sigmas", "dms", "freqs"
    "sigmas" is two dimensional, "dms" and "freqs" are one dimensional.
    The sigma values are the maximum of that periodogram.

    dm_sigma_1d (dict[np.array]): Sigma as a function of dm around the detected DM.
                Keys: "sigmas", "dms"
                The sigma values are the maximum of that periodogram.

    pgram_freq_resolution (float): frequency resolution of the Periodogram from
        which this candidate was derived.
    """

    freq = attrib(converter=float)
    dm = attrib(converter=float)
    sigma = attrib(converter=float)
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    features = attrib(type=np.ndarray)
    detection_statistic = attrib(
        type=SearchAlgorithm_FFA,
        validator=check_detection_statistic_FFA,
        converter=SearchAlgorithm_FFA,
    )
    obs_id = attrib(
        validator=deep_iterable(
            member_validator=instance_of(str), iterable_validator=instance_of(list)
        )
    )
    widths_info = attrib(type=np.ndarray)
    rfi = attrib(converter=bool, default=False)
    detections = attrib(type=np.ndarray, default=None)
    pgram_freq_resolution = attrib(converter=float, default=0.0009701276818911235)
    ndetections = attrib(converter=converters.optional(int), default=None)
    unique_freqs = attrib(type=np.ndarray, default=None)
    max_sig_det = attrib(type=np.ndarray, default=None)
    unique_dms = attrib(type=np.ndarray, default=None)
    dm_freq_sigma = attrib(type=dict, default=None)
    raw_widths_array = attrib(type=dict, default=None)
    dm_sigma_1d = attrib(type=dict, default=None)
    sigmas_per_width = attrib(type=dict, default=None)
    injection = attrib(type=bool, default=False)
    injection_dict = attrib(type=dict, default={})
    datetimes = attrib(
        validator=deep_iterable(
            member_validator=instance_of(datetime.datetime),
            iterable_validator=instance_of(list),
        ),
        default=[],
    )
    # Note `ra`, `dec`, `obs_id` and `detection_statistic` are populated from
    # `SinglePointingCandidateCollection`.
    # they are attributes here too because multi-pointing then uses the
    # `SinglePointingCandidate`s outside of the collection

    @freq.validator
    def _check_freq(self, attribute, value):
        if not within_range(value, MIN_SEARCH_FREQ_FFA, MAX_SEARCH_FREQ_FFA):
            raise ValueError(
                f"Frequency attribute ({attribute.name}={value}) outside range "
                f"({MIN_SEARCH_FREQ}, {MAX_SEARCH_FREQ}] Hz"
            )

    @dm.validator
    def _check_dm(self, attribute, value):
        if not within_range(value, MIN_SEARCH_DM_FFA, MAX_SEARCH_DM_FFA):
            raise ValueError(
                f"DM attribute ({attribute.name}={value}) outside range "
                f"[{MIN_SEARCH_DM_FFA}, {MAX_SEARCH_DM_FFA}] pc/cc"
            )

    @sigma.validator
    def _check_sigma(self, attribute, value):
        if value <= 0:
            raise ValueError(f"Sigma ({attribute.name}={value}) must be greater than 0")

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

    @pgram_freq_resolution.validator
    def _check_pgram_freq_resolution(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"Pgram_freq_resolution ({attribute.name}={value}) must be greater"
                " than 0"
            )

    @property
    def summary(self):
        """Return a summary dict of the candidate."""
        out_dict = {
            "freq": self.freq,
            "dm": self.dm,
            "sigma": self.sigma,
            "rfi": self.rfi,
            "ra": self.ra,
            "dec": self.dec,
            "obs_id": self.obs_id,
            "datetimes": self.datetimes,
        }
        return out_dict

    @property
    def width(self):
        """Return width of the main cluster."""
        if self.widths_info is not None:
            # Older version
            return self.widths_info["width"][0]
        else:
            return self.features["width"][0]

    @property
    def dm_max_curve(self):
        """Return the dm curve where the maximum is taken along the frequency axis."""
        return np.nanmax(self.dm_freq_sigma["sigmas"], 1)

    @property
    def freq_max_curve(self):
        """Return the freq curve where the maximum is taken along the dm axis."""
        # with np.errstate(all='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmax(self.dm_freq_sigma["sigmas"], 0)

    @property
    def delta_dm(self):
        """Return diff between min and max dm where the candidate was found."""
        return np.ptp(self.unique_dms)

    @property
    def delta_freq(self):
        """Return diff between min and max freq where the candidate was found."""
        return np.ptp(self.unique_freqs)

    @property
    def dm_best_curve(self):
        """Return the dm curve for the freq value with the freq value of the cluster."""
        # Another possibility would be using the row containing the maximum sigma,
        # but this may actually be something that was not found by the search
        best_freq_index = np.nanargmin(np.abs(self.dm_freq_sigma["freqs"] - self.freq))
        return self.dm_freq_sigma["sigmas"][:, best_freq_index]

    @property
    def freq_best_curve(self):
        """Return the dm curve for the freq value with the freq value of the cluster."""
        # Another possibility would be using the row containing the maximum sigma,
        # but this may actually be something that was not found by the search
        best_dm_index = np.nanargmin(np.abs(self.dm_freq_sigma["dms"] - self.dm))
        return self.dm_freq_sigma["sigmas"][best_dm_index, :]

    @property
    def num_days(self):
        """Calculate the days from the obs ids."""
        return len(self.obs_id)

    @property
    def best_raw_widths(self):
        """Return the raw widths at the candidate DM."""
        best_dm_index = np.nanargmin(
            np.abs(self.raw_widths_array["dms"] - self.dm)
        )
        raw_power_curve = self.raw_widths_array["powers"][best_dm_index, :, 0]
        return raw_power_curve

    @property
    def harm_sigma_curve(self):
        """Return sigma as a function of summed harmonics."""
        raw_power_curve = self.best_raw_widths
        width_sigma_curve = np.zeros(len(raw_power_curve))
        for i in range(len(width_sigma_curve)):
            # sigma_sum_powers also accepts array inputs which might be faster
            width_sigma_curve[i] = sigma_sum_powers(
                raw_power_curve[: i + 1].sum(), self.num_days * (i + 1)
            )
        return width_sigma_curve

    @property
    def masked_harmonics(self):
        """Returns the number of masked harmonics."""
        zero_count = np.count_nonzero(self.best_raw_widths == 0)
        zero_freqs = np.count_nonzero(
            self.raw_widths_array["freqs"][0, :, 0] == 0
        )
        return zero_count - zero_freqs

    @property
    def best_width_sum(self):
        """Returns the number of summed widths which results in the highest sigma."""
        best_width_sum_index = np.nanargmax(self.width_sigma_curve) + 1
        return best_width_sum_index

    @property
    def strongest_width(self):
        """Returns the width which has the strongest raw power."""
        strongest_width_index = np.nanargmax(self.best_raw_widths)
        dm_index = np.nanargmin(np.abs(self.raw_widths_array["dms"] - self.dm))
        freq_labels = self.raw_widths_array["freqs"][dm_index, :, 0]
        strongest_freq = freq_labels[strongest_width_index]
        return strongest_freq

    @property
    def period(self):
        """Returns the period."""
        return 1 / self.freq

    @property
    def ra_hms(self):
        """Returns ra in hh:mm:ss.ss format."""
        ra_hms = convert_ra_dec(self.ra, self.dec)[0]
        return ra_hms[:2] + ":" + ra_hms[2:4] + ":" + ra_hms[4:]

    @property
    def dec_hms(self):
        """Returns dec in hms format."""
        dec_hms = convert_ra_dec(self.ra, self.dec)[1]
        return dec_hms[:2] + ":" + dec_hms[2:4] + ":" + dec_hms[4:]

    def as_dict(self):
        """Return this candidate's properties as a Python dictionary."""
        ret_dict = dict(
            freq=self.freq,
            dm=self.dm,
            sigma=self.sigma,
            features=self.features,
            rfi=self.rfi,
            max_sig_det=self.max_sig_det,
            ndetections=self.ndetections,
            detections=self.detections,
            dm_freq_sigma=self.dm_freq_sigma,
            unique_freqs=self.unique_freqs,
            unique_dms=self.unique_dms,
            ra=self.ra,
            dec=self.dec,
            detection_statistic=self.detection_statistic.value,
            obs_id=self.obs_id,
            raw_widths_array=self.raw_widths_array,
            widths_info=self.widths_info,
            dm_sigma_1d=self.dm_sigma_1d,
            pspec_freq_resolution=self.pgram_freq_resolution,
            injection=self.injection,
            injection_dict=self.injection_dict,
            datetimes=self.datetimes,
        )
        return ret_dict

    def plot_candidate(self, folder="./plots/", config=default_config_path):
        """
        Plots the candidate.

        Parameters
        ----------
        folder: str
                Folder in which the plot should be saved
        config: str or dict
                Path to the yml config containing the plot config or dict
                containing the plot config

        Returns
        -------
        plot_name: str
                Path to the created plot
        """
        plot_name = plot_candidate(self, config, folder, "single_point_cand")
        return plot_name


@attrs
class SinglePointingCandidateCollection_FFA:
    """
    A collection of `SinglePointingCandidate_FFA`s.

    Attributes
    ----------
    candidates (List[SinglePointingCandidate_FFA]): the candidates
    injections (List(dict)): List describing the injections that were performed
                            beffore the search
    injection_indices (List(int)): Indices of the candidates that were injected
    real_indices (List(int)): Indices of the candidates that were  not injected
    """

    candidates = attrib(
        validator=deep_iterable(
            member_validator=instance_of(SinglePointingCandidate_FFA),
            iterable_validator=instance_of(list),
        )
    )
    real_indices = attrib(
        init=False,
        validator=deep_iterable(
            member_validator=instance_of(int),
            iterable_validator=instance_of(list),
        ),
    )
    injection_indices = attrib(
        init=False,
        validator=deep_iterable(
            member_validator=instance_of(int),
            iterable_validator=instance_of(list),
        ),
    )
    injections = attrib(
        validator=deep_iterable(
            member_validator=instance_of(dict),
            iterable_validator=instance_of(list),
        ),
        default=[],
    )

    def __attrs_post_init__(self):
        """Find indices of real detections and injections."""
        all_indices = np.arange(len(self.candidates))
        injection_flags = np.asarray(
            [cand.injection for cand in self.candidates], dtype=bool
        )
        self.real_indices = all_indices[~injection_flags]
        self.injection_indices = all_indices[injection_flags]

    @classmethod
    def read(cls, filename, verbose=True):
        """Read in candidates from (npz) file."""
        data = np.load(filename, allow_pickle=True)
        cand_list = []
        for cand_dict in data["candidate_dicts"]:
            cand_dict = filter_class_dict(SinglePointingCandidate, cand_dict)
            sp_cand = SinglePointingCandidate_FFA(**cand_dict)
            cand_list.append(sp_cand)
        if verbose:
            log.info(f"Read {len(cand_list)} candidates from {filename}")
        return cls(
            candidates=cand_list,
            injections=data.get("injection_dicts", np.empty(0)).tolist(),
        )

    def write(self, filename):
        """Write candidates to npz file."""
        cand_dicts = []
        for spc in self.candidates:
            cand_dicts.append(spc.as_dict())
        save_dict = dict(
            candidate_dicts=cand_dicts,
            injection_dicts=self.injections,
        )
        np.savez(filename, **save_dict)
        log.info(f"saved candidates to {filename}")

    def plot_candidates(
        self,
        sigma_threshold=10,
        dm_threshold=2,
        folder="./plots/",
        config=default_config_path,
        num_threads=8,
    ):
        """Plot all candidates of the collection."""
        # Load layout here so it doesn't have to be read for each plot individually
        if type(config) == str:
            config = yaml.safe_load(open(config))
        if num_threads == 1:
            plots = []
            for candidate in self.candidates:
                if candidate.sigma > sigma_threshold:
                    if candidate.dm > dm_threshold:
                        plot = candidate.plot_candidate(folder=folder, config=config)
                        plots.append(plot)
        else:
            # Based on https://stackoverflow.com/a/56076681
            with Pool(num_threads) as pool:
                functions = [
                    cand.plot_candidate
                    for cand in self.candidates
                    if ((cand.sigma > sigma_threshold) and (cand.dm > dm_threshold))
                ]
                arguments = (folder, config)
                tasks = zip(functions, repeat(arguments))
                futures = [pool.apply_async(*t) for t in tasks]
                plots = [fut.get() for fut in futures]
        return plots

    def return_attribute_array(self, attribute):
        """Given scalar candidate attribute, return array with those attributes."""
        attribute_list = [getattr(cand, attribute, None) for cand in self.candidates]
        return np.asarray(attribute_list)

    def get_real_candidates(self):
        """Get candidates which have not been injected."""
        # Converting to array for easier slicing, could also change the attribute itself to array
        return np.asarray(self.candidates)[self.real_indices]

    def get_injection_candidates(self):
        """Get candidates which have been injected."""
        return np.asarray(self.candidates)[self.injection_indices]

    def test_injection_performance(self, verbose=True):
        """Return dict containing details of the injection performance."""
        injected_candidates = self.get_injection_candidates()
        injected_indices = np.asarray(
            [cand.injection_dict["injection_index"] for cand in injected_candidates]
        )
        unique_indices, unique_counts = np.unique(injected_indices, return_counts=True)
        all_injections_count = len(self.injections)
        injection_candidates_count = len(injected_candidates)
        recovered_injections = len(unique_indices)
        if verbose:
            log.info(f"Injected Pulsars:     {all_injections_count}")
            log.info(f"Injection Candidates: {injection_candidates_count}")
            log.info(f"Recovered Injections: {recovered_injections}")

        not_recovered = np.setdiff1d(np.arange(all_injections_count), unique_indices)
        return_dict = {
            "all_injections": all_injections_count,
            "all_injection_candidates": injection_candidates_count,
            "recovered_injection": recovered_injections,
            "recovered_injection_indices": unique_indices,
            "recovered_injection_counts": unique_counts,
            "not_recovered_indices": not_recovered,
        }

        return return_dict


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--path",
    default="./plot",
    type=click.Path(
        exists=False,
        file_okay=False,
    ),
    help="Path to the folder where the plots are created.",
)
@click.option(
    "--plot-threshold",
    default=10.0,
    type=float,
    help="Sigma threshold above which the candidate plots are created",
)
@click.argument(
    "candidate_files",
    nargs=-1,
    type=click.Path(
        exists=False,
        file_okay=True,
    ),
)
def plot_candidates_FFA(path, plot_threshold, candidate_files):
    """
    Plot candidates from one or more SinglePointintCandidateCollections.

    This script allows plotting of one or more candidate files to a specific folder for
    candidates above a given sigma from the command line and is accessible under the
    name 'plot_candidates'.
    """
    for cand_file in candidate_files:
        print(f"Processing {cand_file}")
        cands = SinglePointingCandidateCollection_FFA.read(cand_file)
        plotted_cands = cands.plot_candidates(
            sigma_threshold=plot_threshold, folder=path
        )
        print(f"Plotted {len(plotted_cands)} candidates.")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--threshold",
    default=0.0,
    type=float,
    help="Sigma threshold above which the candidates are printed.",
)
@click.argument(
    "candidate_files",
    nargs=-1,
    type=click.Path(
        exists=False,
        file_okay=True,
    ),
)
def print_candidates_FFA(threshold, candidate_files):
    """
    Print candidates from one or more SinglePointintCandidateCollections.

    This script allows printing of the basic candidate parameters
    of one or more candidate files for
    candidates above a given sigma from the command line and is accessible under the
    name 'print_candidates'.

    Returns
    -------
    all_cands: dict[np.array]
            Contains DM, F0 and sigma for the printed candidates
    """
    all_cands = {}
    for cand_file in candidate_files:
        print(f"Printing {cand_file}")
        cands = SinglePointingCandidateCollection_FFA.read(cand_file)
        current_cands = []
        for cand in cands.candidates:
            if cand.sigma >= threshold:
                print(
                    f"DM: {cand.dm:7.3f}, F0: {cand.freq:7.4f}, Sigma:"
                    f" {cand.sigma:4.2f}"
                )
                cand_entry = [cand.dm, cand.freq, cand.sigma]
                current_cands.append(cand_entry)
        all_cands[cand_file] = np.asarray(current_cands)
    return all_cands