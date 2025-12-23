"""
The CHIME/FRB L2/L3 Known Source Sifter contains an actor styled class that determines
if an event is likely originating from a known source.

Adapted to SPS 20210204
"""

import logging
import operator
import os

import attr
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.time import Time
from attr.validators import instance_of
from sps_common.interfaces import KnownSourceClassification, KnownSourceLabel
from sps_databases.db_api import get_nearby_known_sources
from sps_multi_pointing.known_source_sifter import known_source_filters
from sps_common.interfaces.utilities import angular_separation

logger = logging.getLogger(__name__)

kss_path = os.path.dirname(__file__)


@attr.s(slots=True)
class KnownSourceSifter:
    """
    Known source sifter SPS version, adapted from CHIME/FRB L2/L3. The KnownSourceSifter
    is initialised with a set of ks_filters used to determine the likelihood of a
    candidate is associated with a known source.

    Parameters
    -------
    ks_filters: list
        A list of known source filters and their relative weights.
        Default = [["compare_position", 1.0], ["compare_dm", 1.0], ["compare_frequency", 1.0]]

    threshold: float
        The likelihood threshold to determine if a candidate matches a known source.
        Default = 1.0

    rfi_check: dict
        Dictionary containing details of a quick RFI check before the knwon source sifter is run.
        Dicotinary keys should be the name of a candidate attribute and each field should be a dict
        containing a 'threshold' key which is the upper limit for that parameter.

    use_unknown_freq: bool
        Also use known sources without known frequency.
        Default = False
    """

    ks_filters = attr.ib(
        default=[
            ["compare_position", 1.0],
            ["compare_dm", 2.0],
            ["compare_frequency", 4.0],
        ],
        validator=instance_of(list),
    )
    threshold = attr.ib(default=1.0, validator=instance_of(float))
    rfi_check = attr.ib(default={}, validator=instance_of(dict))
    use_unknown_freq = attr.ib(default=False, validator=instance_of(bool))
    filter_scraper_and_f1_nan = attr.ib(default=True, validator=instance_of(bool))
    config_init = attr.ib(init=False)
    ks_database = attr.ib(init=False)
    ks_filter_names = attr.ib(init=False)
    ks_filter_weights = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.config_init = False
        self.ks_filter_names = []
        self.ks_filter_weights = []
        self.load_ks_database()
        self.load_filters()

    def load_ks_database(self):
        # get nearby known sources and make it a numpy array
        ks_collection = get_nearby_known_sources(0, 0, np.infty)  # get all sources
        # Filter scraper sources and sources with no errors
        if self.filter_scraper_and_f1_nan:
            ks_collection = [
                ks
                for ks in ks_collection
                if np.isfinite(ks.spin_period_s_error)
                and "psr_scraper" not in ks.survey
            ]
        ks_database = np.empty(
            shape=len(ks_collection),
            dtype=[
                ("source_name", "<U15"),
                ("pos_ra_deg", "<f4"),
                ("pos_dec_deg", "<f4"),
                ("pos_error_semimajor_deg", "<f4"),
                ("pos_error_semiminor_deg", "<f4"),
                ("pos_error_theta_deg", "<f4"),
                ("dm", "<f4"),
                ("dm_error", "<f4"),
                ("spin_period_s", "<f4"),
                ("current_spin_period_s", "<f4"),
                ("spin_period_s_error", "<f4"),
            ],  # the dtype has some unused fields trimmed out
        )
        for i, ks in enumerate(ks_collection):
            ks_database[i] = (
                ks.source_name,
                ks.pos_ra_deg,
                ks.pos_dec_deg,
                ks.pos_error_semimajor_deg,
                ks.pos_error_semiminor_deg,
                ks.pos_error_theta_deg,
                ks.dm,
                ks.dm_error,
                ks.spin_period_s,
                known_source_filters.change_spin_period(ks, Time.now()),
                ks.spin_period_s_error,
            )

        self.ks_database = ks_database
        logger.info(f"{ks_database.size} known sources loaded")

    def classify(self, candidate, pos_filter=False, known_source_radius=5.0):
        """
        Process a MultiPointingCandidate class instance with and adding matches to the
        known_source_matches field.

        Parameters
        -------
        candidate: MultiPointingCandidate
            A MultiPointingCandidate class instance.
        pos_filter: bool
            Whether to filter the known sources by region around the candidate. Default = False
        known_source_radius: float
            Radius in degrees around the candidate to look for known sources, if pos_filter is applied.
            Default = 5.0

        Returns
        -------
        candidate: MultiPointingCandidate
            MultiPointingCandidate class instance with additional field of known_source_matches.
        """
        rfi_label = self.apply_rfi_check(candidate)
        if rfi_label:
            candidate.known_source = KnownSourceClassification(
                KnownSourceLabel(0), np.asarray([])
            )
            return candidate

        ks_db_freq, ks_db_no_freq, n_ks = known_sources_subset(
            ks_database=self.ks_database,
            pos_filter=pos_filter,
            ra=candidate.ra,
            dec=candidate.dec,
            radius=known_source_radius,
        )
        if n_ks == 0:
            candidate.known_source = KnownSourceClassification(
                KnownSourceLabel(0), np.asarray([])
            )
            return candidate
        prob_freq = self.calculate_response(candidate, ks_db_freq, n_ks)
        ks_db_freq = ks_db_freq[
            ["source_name", "pos_ra_deg", "pos_dec_deg", "spin_period_s", "dm"]
        ]
        if self.use_unknown_freq:
            prob_no_freq = self.calculate_response(
                candidate, ks_db_no_freq, n_ks, freq=False
            )
            ks_db_no_freq = ks_db_no_freq[
                ["source_name", "pos_ra_deg", "pos_dec_deg", "spin_period_s", "dm"]
            ]
            known_sources = np.concatenate(
                (
                    ks_db_freq[np.where(prob_freq > self.threshold)],
                    ks_db_no_freq[np.where(prob_no_freq > self.threshold)],
                )
            )
            known_sources_prob = np.array(
                np.concatenate(
                    (
                        prob_freq[np.where(prob_freq > self.threshold)],
                        prob_no_freq[np.where(prob_no_freq > self.threshold)],
                    )
                ),
                dtype=[("likelihood", "<f4")],
            )
        else:
            known_sources = ks_db_freq[np.where(prob_freq > self.threshold)]
            known_sources_prob = np.array(
                prob_freq[np.where(prob_freq > self.threshold)],
                dtype=[("likelihood", "<f4")],
            )

        known_source_list = rfn.merge_arrays(
            (known_sources, known_sources_prob), flatten=True
        )
        if known_source_list.size == 0:
            known_source_list = np.array(
                [],
                dtype=[
                    ("source_name", "u1"),
                    ("pos_ra_deg", "<f4"),
                    ("pos_dec_deg", "<f4"),
                    ("spin_period_s", "<f4"),
                    ("dm", "<f4"),
                    ("likelihood", "<f4"),
                ],
            )
        if len(known_source_list) > 0:
            candidate.known_source = KnownSourceClassification(
                KnownSourceLabel(1), known_source_list
            )
        else:
            candidate.known_source = KnownSourceClassification(
                KnownSourceLabel(0), known_source_list
            )
        return candidate

    def load_filters(self):
        """
        Initializes the filters defined in the configuration file.

        Function appends to `ks_filter_names` and `ks_filter_weights`.
        """
        # initialize filters
        for filt in self.ks_filters:
            # check if this filter definition exists
            if hasattr(known_source_filters, filt[0]):
                self.ks_filter_names.append(filt[0])
                self.ks_filter_weights.append(float(filt[1]))
                logger.info(f"Filter '{filt[0]}' loaded")
            else:
                logger.error(
                    f"Filter '{filt[0]}' "
                    + "is not defined in "
                    + "known_source_filters.py!"
                )

        nof = len(self.ks_filter_names)
        if nof:
            logger.info(f"Read {nof} filters")
            self.config_init = True
        else:
            logger.error(
                "There are no filters defined! "
                + "No known sources will be recognized!"
            )

    def calculate_response(self, candidate, ks_sub_database, n_ks, freq=True):
        """
        Calculates statistical match of an L2 event to a list of known sources. Function
        loops over the filters listed in the KS sifter configuration file and defined in
        the known_source_filters.py script.

        Parameters
        ----------
        candidate : class
            Class should be ``MultiPointingCandidate``.
        ks_sub_database : array_like
            List containing known sources in the neighborhood of candidates.
        n_ks: int
            Number of known sources within the region of the candidate

        Returns
        -------
        probability : float, array
            An array with the probabilities representing the
            likelihoods that the event is associated with neighbouring
            known sources.
        """
        assert self.config_init is True

        # call the filters one-by-one
        bayes_factor = np.zeros(len(ks_sub_database))
        for i, filt in enumerate(self.ks_filter_names):
            if not freq:
                if filt == "compare_frequency":
                    continue
            # get the function call
            function = getattr(known_source_filters, filt)
            # function adds the grade to the object header
            if i == 0:
                bayes_factor = self.ks_filter_weights[i] * function(
                    candidate,
                    ks_sub_database,
                    self.ks_filter_weights[i],
                )
            else:
                # Btot = B1 * B2 * ... * Bn
                bayes_factor *= self.ks_filter_weights[i] * function(
                    candidate,
                    ks_sub_database,
                    self.ks_filter_weights[i],
                )

        # calculate posterior probability
        prior = 1.0 / n_ks
        probability = bayes_factor * prior / (1 + bayes_factor * prior)

        return probability

    def apply_rfi_check(self, candidate):
        """
        Simple RFI check for candidates with clear RFI-like parameters.

        Parameters
        ----------
        candidate : class
            Class should be ``MultiPointingCandidate``.

        Returns
        -------
        rfi_label: bool
            True if was identified as RFI
        """
        for check in self.rfi_check:
            threshold = self.rfi_check[check]["threshold"]
            check_val = getattr(candidate, check, None)
            comparator = getattr(operator, self.rfi_check[check].get("operator", "gt"))
            if check_val is not None:
                if comparator(check_val, threshold):
                    return True
        return False


def known_sources_subset(ks_database, pos_filter=False, ra=0, dec=0, radius=5.0):
    """
    Retrieve known sources with angular separation less than `radius` of the provided
    coordinates if prompted and split the database to sources with spin period and
    without spin period.

    Parameters
    ----------
    ks_database : array_like
        CHIME/FRB known sources database.
    pos_filter : bool
        Whether to narrow down the ks database to within a region of sky
    pos_ra_deg : float
        Right ascension of the center of the circle, in degrees. Default=0
    pos_dec_deg : float
        Declination of the center of the circle, in degrees. Default=0
    radius : float
        Maximum angular separation of sources to consider. Default=5.0
    Returns
    -------
    ks_database_freq : array_like
        Part of the known sources database with spin periods
    ks-database_no_freq : array_like
        Part of the known sources database without spin periods
    """
    if pos_filter:
        # select on angular separation
        _, separation = angular_separation(
            ks_database["pos_ra_deg"],
            ks_database["pos_dec_deg"],
            ra,
            dec,
        )

        sky_mask = np.where(separation < radius)[0]

        # sort on angular separation
        sorted_idx = separation[sky_mask].argsort()

        ks_region = ks_database[sky_mask[sorted_idx]]
    else:
        ks_region = ks_database

    # count number of known sources in the region
    ks_names = []
    for ks_name in ks_region["source_name"]:
        ks_names.append(ks_name.split("_")[0])
    n_ks = np.unique(ks_names).size

    # split ks_database to sources with spin period and sources without spin period
    freq_mask = np.where(~np.isnan(ks_region["spin_period_s"]))[0]
    no_freq_mask = np.where(np.isnan(ks_region["spin_period_s"]))[0]
    return (
        ks_region[freq_mask],
        ks_region[no_freq_mask],
        n_ks,
    )
