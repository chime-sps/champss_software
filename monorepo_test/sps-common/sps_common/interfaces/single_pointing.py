"""Classes for single pointing candidates."""

import datetime
import enum
import logging
import os
from itertools import repeat
from multiprocessing import Pool

import click
import numpy as np
import yaml
from attr import attrib, attrs, converters
from attr.validators import deep_iterable, instance_of

from sps_common.constants import (
    MAX_SEARCH_DM,
    MAX_SEARCH_FREQ,
    MIN_SEARCH_DM,
    MIN_SEARCH_FREQ,
)
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


class SearchAlgorithm(enum.Enum):
    """
    The search algorithm type.

    Currently only power_spec relevant.
    """

    power_spec = 1
    hhat = 2
    power_spec_and_hhat = 3


def check_detection_statistic(instance, attribute, value):
    """Check if the detection statistic has the correct type."""
    if not isinstance(value, SearchAlgorithm):
        raise TypeError(
            f"Detection statistic attribute ({attribute.name}={value}) must be "
            + "of SearchAlgorithm enum"
        )
    if value == SearchAlgorithm.power_spec_and_hhat:
        raise ValueError(
            f"Detection statistic attribute ({attribute.name}={value}) must be "
            + "either power_spec or hhat only"
        )


@attrs
class SinglePointingCandidate:
    """
    This is the class instance to store the properties of a single pointing candidates.

    Specifically, after the power spectra detections has gone through the candidates
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

    sigma (float): the highest detection significance out of all the power
        spectrum detections associated with this candidate

    summary (dict): summary information for the candidate.

    features (numpy.ndarray): 1D array with a column per feature.

    detections (np.ndarray): Array decribing the raw detections.
        Saving those can be turned off.

    ndetections (int): Number of detections clusterered in this candidate.

    unique_freq (np.ndarray): Array containing the unique frequencies.

    unique_dms (np.ndarray): Array containing the unique dms.

    max_sig_det (np.ndarray): Arrays describing the highest sigma detections as outlined
        in ps_processes.Cluster

    detection_statistic (SearchAlgorithm) : Whether power spectra (power_spec) or
    hhat was used to find this candidate

    obs_id (list) : A list of observation id of the observations in the power
        spectra stack this candidate came from

    rfi (bool): whether the candidate has been flagged as interference

    harmonics_info (numpy.ndarray): Information about the highest-sigma
        detection of each harmonically related cluster of detections (from which the
        SinglePointingCandidate was derived) This INCLUDES the main detection.
        Earlier versions exluded the main detection.
        dtype.names = ['freq', 'dm', 'nharm', 'sigma']

    raw_harmonic_powers_array (dict[np.array]): Raw harmonic power information
        that includes the powers for a set number of DMs and harmonics.
        Keys: "powers", "dms", "freqs", "freq_bins"
        "powers", "freqs" and freq_bins share the dimensions
        [dm_trials, harmonics, factor_trials].
        "dms" is one dimensional.
        "factor_trials" is defined by the property "period_factors" in HarmonicFilter.
        The elements of that dimension contain the raw harmonic powers where the
        base period is multiplied by [1,2,1/2,3,1/3,...].

    dm_freq_sigma (dict[np.array]): Sigma as a function of dm and frequency
    arround the detected dm and frequency.
    Keys: "sigmas", "dms", "freqs"
    "sigmas" is two dimensional, "dms" and "freqs" are one dimensional.
    The sigma values are the maximum of different harmonic sums.

    dm_sigma_1d (dict[np.array]): Sigma as a function of dm arround the detected DM.
                Keys: "sigmas", "dms"
                The sigma values are the maximum of different harmonic sums.
    sigmas_per_harmonic_sum (dict[np.array]): Sigma for each harmonic sum.
                Keys: "sigmas_unweighted", "sigmas_weighted", "bin_weights"
                "sigma_unweighted": Sigma where nsum is defined by ndays * harm.
                               This assumes not frequency bins are masked in the
                               harmonic sum.
                "sigma_weighted": Sigma where bins that are masked are not included
                                  in nsum.
                "nsum": The number of summands used in the sigma calculation.
                "weight_fraction": nsum divided by the possible maximum of nsum.

    pspec_freq_resolution (float): frequency resolution of the PowerSpectra from
        which this candidate was derived.
    """

    freq = attrib(converter=float)
    dm = attrib(converter=float)
    sigma = attrib(converter=float)
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    features = attrib(type=np.ndarray)
    detection_statistic = attrib(
        type=SearchAlgorithm,
        validator=check_detection_statistic,
        converter=SearchAlgorithm,
    )
    obs_id = attrib(
        validator=deep_iterable(
            member_validator=instance_of(str), iterable_validator=instance_of(list)
        )
    )
    harmonics_info = attrib(type=np.ndarray)
    rfi = attrib(converter=bool, default=False)
    detections = attrib(type=np.ndarray, default=None)

    pspec_freq_resolution = attrib(converter=float, default=0.0009701276818911235)
    ndetections = attrib(converter=converters.optional(int), default=None)
    unique_freqs = attrib(type=np.ndarray, default=None)
    max_sig_det = attrib(type=np.ndarray, default=None)
    unique_dms = attrib(type=np.ndarray, default=None)
    dm_freq_sigma = attrib(type=dict, default=None)
    raw_harmonic_powers_array = attrib(type=dict, default=None)
    dm_sigma_1d = attrib(type=dict, default=None)
    sigmas_per_harmonic_sum = attrib(type=dict, default=None)
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
        if not within_range(value, MIN_SEARCH_FREQ, MAX_SEARCH_FREQ):
            raise ValueError(
                f"Frequency attribute ({attribute.name}={value}) outside range "
                f"({MIN_SEARCH_FREQ}, {MAX_SEARCH_FREQ}] Hz"
            )

    @dm.validator
    def _check_dm(self, attribute, value):
        if not within_range(value, MIN_SEARCH_DM, MAX_SEARCH_DM):
            raise ValueError(
                f"DM attribute ({attribute.name}={value}) outside range "
                f"[{MIN_SEARCH_DM}, {MAX_SEARCH_DM}] pc/cc"
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

    @pspec_freq_resolution.validator
    def _check_pspec_freq_resolution(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"Pspec_freq_resolution ({attribute.name}={value}) must be greater"
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
    def nharm(self):
        """Return nharm of the main cluster."""
        return self.harmonics_info["nharm"][0]

    @property
    def dm_max_curve(self):
        """Return the dm curve where the maximum is taken along the frequency axis."""
        return self.dm_freq_sigma["sigmas"].max(1)

    @property
    def freq_max_curve(self):
        """Return the freq curve where the maximum is taken along the dm axis."""
        return self.dm_freq_sigma["sigmas"].max(0)

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
        best_freq_index = np.argmin(np.abs(self.dm_freq_sigma["freqs"] - self.freq))
        return self.dm_freq_sigma["sigmas"][:, best_freq_index]

    @property
    def freq_best_curve(self):
        """Return the dm curve for the freq value with the freq value of the cluster."""
        # Another possibility would be using the row containing the maximum sigma,
        # but this may actually be something that was not found by the search
        best_dm_index = np.argmin(np.abs(self.dm_freq_sigma["dms"] - self.dm))
        return self.dm_freq_sigma["sigmas"][best_dm_index, :]

    @property
    def num_days(self):
        """Calculate the days from the obs ids."""
        return len(self.obs_id)

    @property
    def best_raw_harmonic_powers(self):
        """Return the raw harmonic powers at the candidate DM."""
        best_dm_index = np.argmin(
            np.abs(self.raw_harmonic_powers_array["dms"] - self.dm)
        )
        raw_power_curve = self.raw_harmonic_powers_array["powers"][best_dm_index, :, 0]
        return raw_power_curve

    @property
    def harm_sigma_curve(self):
        """Return sigma as a function of summed harmonics."""
        raw_power_curve = self.best_raw_harmonic_powers
        dm_sigma_curve = np.zeros(len(raw_power_curve))
        for i in range(len(dm_sigma_curve)):
            # sigma_sum_powers aso accepts array inputs which might be faster
            dm_sigma_curve[i] = sigma_sum_powers(
                raw_power_curve[: i + 1].sum(), self.num_days * (i + 1)
            )
        return dm_sigma_curve

    @property
    def masked_harmonics(self):
        """Returns the number of masked harmonics."""
        zero_count = np.count_nonzero(self.best_raw_harmonic_powers == 0)
        zero_freqs = np.count_nonzero(
            self.raw_harmonic_powers_array["freqs"][0, :, 0] == 0
        )
        return zero_count - zero_freqs

    @property
    def best_harmonic_sum(self):
        """Returns the number of summed harmonics which results in the highest sigma."""
        best_harmonic_sum_index = np.argmax(self.harm_sigma_curve) + 1
        return best_harmonic_sum_index

    @property
    def strongest_harmonic_frequency(self):
        """Returns the frequency of the harmonic which has the strongest raw power."""
        strongest_harm_index = np.argmax(self.best_raw_harmonic_powers)
        dm_index = np.argmin(np.abs(self.raw_harmonic_powers_array["dms"] - self.dm))
        freq_labels = self.raw_harmonic_powers_array["freqs"][dm_index, :, 0]
        strongest_freq = freq_labels[strongest_harm_index]
        return strongest_freq

    @property
    def period(self):
        """Returns the period."""
        return 1 / self.freq

    @property
    def strongest_harmonic_period(self):
        """Returns the period of the harmonic which has the strongest raw power."""
        return 1 / self.strongest_harmonic_frequency

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

    @property
    def sigma_unweighted(self):
        """Returns the best sigma without correction nsum."""
        return np.max(self.sigmas_per_harmonic_sum["sigmas_unweighted"])

    @property
    def masked_fraction_at_best_sigma(self):
        """Returns the fraction of masked bins at the the best weighted sigma."""
        return (
            1
            - self.sigmas_per_harmonic_sum["weight_fraction"][
                np.argmax(self.sigmas_per_harmonic_sum["sigmas_weighted"])
            ]
        )

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
            raw_harmonic_powers_array=self.raw_harmonic_powers_array,
            harmonics_info=self.harmonics_info,
            dm_sigma_1d=self.dm_sigma_1d,
            sigmas_per_harmonic_sum=self.sigmas_per_harmonic_sum,
            pspec_freq_resolution=self.pspec_freq_resolution,
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
class SinglePointingCandidateCollection:
    """
    A collection of `SinglePointingCandidate`s.

    Attributes
    ----------
    candidates (List[SinglePointingCandidate]): the candidates
    """

    candidates = attrib(
        validator=deep_iterable(
            member_validator=instance_of(SinglePointingCandidate),
            iterable_validator=instance_of(list),
        )
    )

    @classmethod
    def read(cls, filename, verbose=True):
        """Read in candidates from (npz) file."""
        data = np.load(filename, allow_pickle=True)
        cand_list = []
        # allow backward-compatibility with old files
        if "harmonics_info" not in data["candidate_dicts"][0]:
            log.warning(
                "Old file without harmonics_info, populating them all with an empty"
                " array"
            )
            harm_dtype = [
                ("freq", float),
                ("dm", float),
                ("nharm", int),
                ("sigma", float),
            ]
            dummy_harm_info = np.array([], dtype=harm_dtype)

        for cand_dict in data["candidate_dicts"]:
            if "harmonics_info" not in cand_dict:
                cand_dict["harmonics_info"] = dummy_harm_info
            # Allow backwards compability
            cand_dict = filter_class_dict(SinglePointingCandidate, cand_dict)
            cand_list.append(SinglePointingCandidate(**cand_dict))

        if verbose:
            log.info(f"Read {len(cand_list)} candidates from {filename}")
        return cls(candidates=cand_list)

    def write(self, filename):
        """Write candidates to npz file."""
        cand_dicts = []
        for spc in self.candidates:
            cand_dicts.append(spc.as_dict())
        save_dict = dict(
            candidate_dicts=cand_dicts,
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
def plot_candidates(path, plot_threshold, candidate_files):
    """
    Plot candidates from one or more SinglePointintCandidateCollections.

    This script allows plotting of one or more candidate files to a specific folder for
    candidates above a given sigma from the command line and is accessible under the
    name 'plot_candidates'.
    """
    for cand_file in candidate_files:
        print(f"Processing {cand_file}")
        cands = SinglePointingCandidateCollection.read(cand_file)
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
def print_candidates(threshold, candidate_files):
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
        cands = SinglePointingCandidateCollection.read(cand_file)
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
