"""Classes for single pointing candidates."""

import datetime
import os
from enum import Enum

import attr
import numpy as np
import numpy.lib.recfunctions as rfn
from attr import attrib, attrs
from attr.validators import deep_iterable, instance_of
from sps_common.constants import (
    MAX_SEARCH_DM,
    MAX_SEARCH_FREQ,
    MIN_SEARCH_DM,
    MIN_SEARCH_FREQ,
)
from sps_common.interfaces.single_pointing import (
    SinglePointingCandidate,
    SinglePointingCandidateCollection,
)
from sps_common.interfaces.utilities import within_range
from sps_common.plotting import plot_candidate

default_config_path = (
    os.path.abspath(os.path.dirname(__file__)) + "/default_mp_plot.yml"
)


class CandidateClassificationLabel(Enum):
    """Enumeration of possible candidate classifications."""

    RFI = 0
    Ambiguous = 1
    Astro = 2


@attrs
class CandidateClassification:
    """
    Classification label plus the strength of the association, in the opinion of the
    classifier.

    Attributes
    ----------
    label: RFI/Astro/Ambiguous

    grade (float in the range 0-1):

        indicator of the strength of the classification, 1 being the highest
    """

    label: CandidateClassificationLabel = attrib()
    grade: float = attrib()


class KnownSourceLabel(Enum):
    """Enumeration of possible candidate classifications."""

    Unknown = 0
    Known = 1


@attrs
class KnownSourceClassification:
    """
    Classification label plus the strength of the association towards known sources.

    Attributes
    ----------
    label: Unknown/Known

    matches (np.ndarray):

        A structured array of known sources associated with the
        candidates, where `dtype` of the array is:
        ```
        [
            ('source_name', 'u1'),
            ('pos_ra_deg', '<f4'),
            ('pos_dec_deg', '<f4'),
            ('spin_period_s', '<f4'),
            ('dm', '<f4'),
            ('likelihood', '<f4')
        ]
        ```
    """

    label: KnownSourceLabel = attrib()
    matches: np.ndarray = attrib()


@attrs
class MultiPointingCandidate:
    """
    The result of multi-pointing grouping, collecting many single-pointing candidate
    groups. Single-pointing candidate summaries will be combined and reduced, and
    single-pointing feature sets will be reduced to a single set for the group.

    Attributes
    ----------
    best_freq (float):

        the rotation frequency associated with the single-pointing with the
        highest detection significance, in Hz

    mean_freq (float):

        the mean candidate rotation frequency from all combined single-pointing
        candidates, in Hz

    delta_freq (float):

        the uncertainty in the mean candidate rotation frequency, in Hz

    best_dm (float):

        the dispersion measure associated with the single-pointing with the
        highest detection significance, in pc/cc

    mean_dm (float):

        the mean candidate dispersion measure, in pc/cc

    delta_dm (float):

        the uncertainty in the mean candidate dispersion measure, in pc/cc

    best_dc (float):

        the nominal duty cycle of the candidate with the highest detection
        significance (0<dc<1)

    ra (float):

        the centroid right ascension of condensed single-pointing candidates,
        in degrees

    dec (float):

        the centroid declination of condensed single-pointing candidates, in
        degrees

    best_sigma (float):

        the detection significance value associated with the single-pointing
        with the highest detection significance

    summary (dict):

        summary information for the candidate based on the combination of
        grouped single-pointing candidate summaries

    features (numpy.ndarray):

        1D array with a column per feature, where the feature value is some
        reduction from the complete set of single-pointing candidate features

    position_features (numpy.ndarray):

        1D array with a column per feature, where the feature values are
        computed from the distribution of pointings that constitute the group

    classification (sps_multi_pointing.interfaces.CandidateClassification):

        the result of running the `classifier` on the candidate

    known_source (sps_multi_pointing.interfaces.KnownSourceClassification):

        the result of running the `known_source_sifter` on the candidate

    position_sigmas (np.ndarray):

        2d array containing (ra, dec, sigma, distance_t_centroid) for the
        strongest candidate in each pointing

    summed_raw_harmonic_powers (np.ndarray):

        The summed raw_harmonic_powers for all candidates for shared dms.
        Will be changed to use a dict and only use the strongest candidate
        in each beam in a future version.

    best_candidate (SinglePointingCandidate):

        The candidate with the highest sigma.

    all_dms (np.ndarray):

        The DMs of all candidates.

    all_freqs (np.ndarray:

        The frequencies of all candidates.

    all_sigmas (np.ndarray):

        The sigmas of all candidates.
    """

    all_dms = attrib(type=list)
    all_freqs = attrib(type=list)
    all_sigmas = attrib(type=list)
    best_freq = attrib(converter=float)
    mean_freq = attrib(converter=float)
    delta_freq = attrib(converter=float)
    best_dm = attrib(converter=float)
    mean_dm = attrib(converter=float)
    delta_dm = attrib(converter=float)
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    best_sigma = attrib(converter=float)
    summary = attrib(type=dict)
    features = attrib(type=np.ndarray)
    position_features = attrib(type=np.ndarray)
    obs_id = attrib(
        validator=deep_iterable(
            member_validator=instance_of(str), iterable_validator=instance_of(list)
        )
    )
    position_sigmas = attrib(type=np.ndarray)
    summed_raw_harmonic_powers = attrib(type=np.ndarray)
    all_summaries = attrib(type=list)
    best_candidate_object = attrib(type=SinglePointingCandidate, default=None)
    best_nharm = attrib(converter=int, default=None)
    classification: CandidateClassification = attrib(default=None)
    known_source: KnownSourceClassification = attrib(default=None)
    datetimes = attrib(
        validator=deep_iterable(
            member_validator=instance_of(datetime.datetime),
            iterable_validator=instance_of(list),
        ),
        default=[],
    )
    # I want best_candidate to be a SinglePointingCandidate, but when loading a file I get a dict
    # I fix this in post_init. A custom converter would also work probabaly.

    def __attrs_post_init__(self):
        """Convert the best_candidate from a dict to a SinglePointingCandidate."""
        # self.best_candidate_object = self.best_candidate

    # @best_freq.validator
    @mean_freq.validator
    def _check_freq(self, attribute, value):
        if not within_range(value, MIN_SEARCH_FREQ, MAX_SEARCH_FREQ):
            raise ValueError(
                f"Frequency attribute ({attribute.name}={value}) outside range "
                f"({MIN_SEARCH_FREQ}, {MAX_SEARCH_FREQ}] Hz"
            )

    @delta_freq.validator
    def _check_delta_freq(self, attribute, value):
        if not within_range(value, 0, MAX_SEARCH_FREQ):
            raise ValueError(
                f"Frequency attribute ({attribute.name}={value}) outside range "
                f"[0, {MAX_SEARCH_FREQ}] Hz"
            )

    #  @best_dm.validator
    @mean_dm.validator
    def _check_dm(self, attribute, value):
        if not within_range(value, MIN_SEARCH_DM, MAX_SEARCH_DM):
            raise ValueError(
                f"DM attribute ({attribute.name}={value}) outside range "
                f"[{MIN_SEARCH_DM}, {MAX_SEARCH_DM}] pc/cc"
            )

    @delta_dm.validator
    def _check_delta_dm(self, attribute, value):
        if not within_range(value, 0, MAX_SEARCH_DM):
            raise ValueError(
                f"DM attribute ({attribute.name}={value}) outside range "
                f"[0, {MAX_SEARCH_DM}] pc/cc"
            )

    # @best_freq_arr.validator
    # @best_dm_arr.validator
    def _check_array(self, attribute, value):
        if type(value) != np.ndarray:
            raise TypeError(
                f"The array ({attribute.name}={value}) is must be a numpy.ndarray"
            )
        if value[0].size != 2:
            raise ValueError(
                f"The elements of the array ({attribute.name}={value}) must have size"
                " of 2"
            )

    # @best_dc.validator
    # def _check_best_dc(self, attribute, value):
    #     if not 0 <= value < 1:
    #         raise ValueError(
    #             f"Duty cycle attribute ({attribute.name}={value}) outside range [0, 1)"
    #         )

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

    @best_sigma.validator
    def _check_best_sigma(self, attribute, value):
        if value <= 0:
            raise ValueError(
                f"Sigma attribute ({attribute.name}={value}) must be greater than 0"
            )

    # @features.validator
    @position_features.validator
    def _check_features(self, attribute, value):
        if type(value) != np.ndarray:
            raise TypeError(
                f"Feature attribute ({attribute.name}={value}) must be a numpy.ndarray"
            )
        try:
            if len(value.dtype.names) > 0:
                if rfn.structured_to_unstructured(value).size != len(value.dtype.names):
                    raise ValueError(
                        f"Feature attribute ({attribute.name}={value})"
                        + " does not have the same length as its dtype"
                    )
        except TypeError:
            raise TypeError(
                f"Feature attribute ({attribute.name}={value}) is not a structured"
                " array"
            )

    @property
    def num_candidates(self):
        """Return the number of grouped candidates."""
        return len(self.all_dms)

    @property
    def known_source_string(self):
        """Returns a string describing the knonw sources."""
        known_source_string = ""
        for source in self.known_source.matches:
            known_source_string += (
                f"\n{source[0]} ({source[1]:.2f}, {source[2]:.2f}): {source[5]:.5f}"
            )
            known_source_string += f"\n F0: {1/source[3]:.4f}, DM: {source[4]:.2f}"
        return known_source_string

    def single_candidate(self, index):
        """Load a SinglePointingCandidate based on an index."""
        cand_collection = SinglePointingCandidateCollection.read(
            self.all_summaries[index]["file_name"], verbose=False
        )
        candidate = cand_collection.candidates[self.all_summaries[index]["cand_index"]]
        return candidate

    @property
    def best_candidate(self):
        """
        Load the best candidate.

        all_summaries need to be sorted.
        """
        return self.single_candidate(0)

    def as_dict(self):
        """Return this candidate's properties as a Python dictionary."""
        if self.classification is not None:
            classification_dict = dict(
                label=self.classification.label.name, grade=self.classification.grade
            )
        else:
            classification_dict = None
        if self.known_source is not None:
            known_source_dict = dict(
                label=self.known_source.label.name,
                matches=self.known_source.matches.tolist(),
                matches_names=self.known_source.matches.dtype.names,
            )
        else:
            known_source_dict = None
        cand_dict = dict(
            best_freq=self.best_freq,
            #        best_freq_arr=self.best_freq_arr.tolist(),
            mean_freq=self.mean_freq,
            delta_freq=self.delta_freq,
            best_dm=self.best_dm,
            #        best_dm_arr=self.best_dm_arr.tolist(),
            mean_dm=self.mean_dm,
            delta_dm=self.delta_dm,
            # best_dc=self.best_dc,
            ra=self.ra,
            dec=self.dec,
            best_sigma=self.best_sigma,
            summary=self.summary,
            features=self.features.tolist(),
            features_names=self.features.dtype.names,
            position_features=self.position_features.tolist(),
            position_features_names=self.position_features.dtype.names,
            obs_id=self.obs_id,
            classification=classification_dict,
            known_source=known_source_dict,
            # best_candidate=self.best_candidate.as_dict(),
            all_dms=self.all_dms,
            all_freqs=self.all_freqs,
            all_sigmas=self.all_sigmas,
            all_summaries=self.all_summaries,
            datetimes=self.datetimes,
            best_nharm=self.best_nharm,
        )
        return cand_dict

    @classmethod
    def read(cls, filename):
        """Read MultiPointingCandidate from disk."""
        mpc_dict = np.load(filename, allow_pickle=True)["properties"][()]
        if mpc_dict["classification"]:
            mpc_dict["classification"] = CandidateClassification(
                **mpc_dict["classification"]
            )
        if mpc_dict["known_source"]:
            mpc_dict["known_source"] = KnownSourceClassification(
                **mpc_dict["known_source"]
            )
        return cls(**mpc_dict)

    def write(self, file_prefix):
        """Write MultiPointingCandidate to disk."""
        if self.classification:
            class_value = self.classification.label.name
        else:
            class_value = "none"
        filename = file_prefix + "_f_{:.3f}_DM_{:.3f}_class_{}.npz".format(
            self.best_freq, self.best_dm, class_value
        )
        properties = attr.asdict(self)
        np.savez(filename, properties=properties)
        return filename

    def plot_candidate(self, path="./plots_mp/", config=default_config_path):
        """Create candidate plot for MultiPointingCandidate."""
        if self.known_source:
            known_source_label = self.known_source.label.value
        else:
            known_source_label = 0
        if known_source_label:
            known_source_name = str(self.known_source.matches[0][0])
            path = path + "/known/"
            file_name = f"multi_pointing_cand_{known_source_name}"
        else:
            path = path + "/unknown/"
            file_name = "multi_point_cand"
        file_path = plot_candidate(
            self.best_candidate,
            default_config_path,
            path,
            file_name,
            mp_cand=self,
        )
        return file_path


@attrs
class PulsarCandidate:
    """The final result of the multi-pointing processing: an unknown pulsar candidate
    ready for verification
    """
