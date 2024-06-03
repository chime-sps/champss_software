#!/usr/bin/env python3
import logging
import math
from typing import List, Tuple

import astropy.coordinates as coordinates
import numpy as np
import scipy
from attr import attrib, attrs
from easydict import EasyDict
from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)
from sklearn.cluster import DBSCAN
from sps_common.interfaces import MultiPointingCandidate, SinglePointingCandidate

log = logging.getLogger(__name__)


def get_cand_freq_and_dm(cand: SinglePointingCandidate) -> Tuple[float, float]:
    """
    Return the necessary summary information from a given SinglePointingCandidate for
    clustering.

    Parameters
    ----------
    cand: SinglePointingCandidate
        An individual single pointing candidate

    Returns
    -------
    (freq, dm): tuple[float, float]
        The summary frequency and DM of a given candidate
    """
    return cand.freq, cand.dm


@attrs
class SinglePointingCandidateGroup:
    """
    A collection of SinglePointingCandidates that are considered to be in a cluster.

    This class provides the mechanisms to convert the collection into a single
    MultiPointingCandidate object.
    """

    group_members = attrib()
    group_features = attrib(default=None)
    position_based_features = attrib(default=None)
    group_summary = attrib(factory=dict)
    group_attributes = attrib(factory=dict)
    best_candidate = attrib(type=SinglePointingCandidate, default=None)

    def __attrs_post_init__(self):
        # Broken for some reason, will check later. length of features is different
        # self.reduce_member_features()
        self.extract_position_features()
        self.reduce_member_summaries()

    def reduce_member_features(self, operation: str = "max_sigma"):
        """
        Reduce many individual feature sets into a single feature set that represents
        the grouped collection of single pointing candidates.

        Parameters
        ----------
        operation: str
            A string that describes how the features from each
            SinglePointingCandidate will be reduced. Default = 'max_sigma'.
            Options are:
                'median' - Use the median value of each feature
                'max' - Use the maximum value of each feature
                'max_sigma' - Use the features of the highest-sigma
                              SinglePointingCandidate in the group
        """
        features = [
            structured_to_unstructured(spc.features)
            for spc in self.group_members
            if spc.features
        ]
        if not features:
            self.group_features = np.array((), dtype=[])
            return
        # do the reduction along the candidate axis
        if operation == "max":
            reduced_features = np.max(features, axis=0)
        elif operation == "median":
            reduced_features = np.median(features, axis=0)
        elif operation == "max_sigma":
            sigmas = [spc.sigma for spc in self.group_members]
            spc_idx = sigmas.index(max(sigmas))
            reduced_features = structured_to_unstructured(
                self.group_members[spc_idx].features
            )

        else:
            log.warn("Did not understand the given operation, taking maximum value")
            reduced_features = np.max(features, axis=0)
        features_dtype = self.group_members[0].features.dtype
        self.group_features = unstructured_to_structured(
            reduced_features, dtype=features_dtype
        )

    def extract_position_features(self):
        """
        Create position-based features from the distribution of pointing positions
        present in the group members.

        Updates the group RA/dec attributes.
        """
        all_ra_dec = np.asarray([(spc.ra, spc.dec) for spc in self.group_members])
        all_sigmas = np.asarray([spc.sigma for spc in self.group_members])

        # These two arrays can contain multiple values per pointing
        # The could be filtered beforehand but I filter them out now

        # This could probabaly be done more effectively
        sigma_sorted_indices = np.argsort(all_sigmas)[::-1]
        ra_dec = all_ra_dec[sigma_sorted_indices]
        sigmas = all_sigmas[sigma_sorted_indices]

        # After sorting np unique will give the index of the first appearance of the first row
        ra_dec, beam_indices = np.unique(ra_dec, axis=0, return_index=True)
        sigmas = sigmas[beam_indices]
        weights = all_sigmas**2
        cartesian_points = coordinates.spherical_to_cartesian(
            sigmas**2, ra_dec[:, 1] / 180 * math.pi, ra_dec[:, 0] / 180 * math.pi
        )

        centroid_point = np.mean(cartesian_points, axis=1)
        centroid_spherical = coordinates.cartesian_to_spherical(*centroid_point)

        centroid_ra = centroid_spherical[2].deg
        centroid_dec = centroid_spherical[1].deg

        distances_to_centroid = (
            coordinates.angular_separation(
                centroid_ra / 180 * math.pi,
                centroid_dec / 180 * math.pi,
                ra_dec[:, 0] / 180 * math.pi,
                ra_dec[:, 1] / 180 * math.pi,
            )
            * 180
            / math.pi
        )

        mean_ra = scipy.stats.circmean(all_ra_dec[:, 0] / 180 * math.pi) * 180 / math.pi
        mean_dec = np.mean(all_ra_dec[:, 1])

        std_ra = scipy.stats.circstd(all_ra_dec[:, 0] / 180 * math.pi) * 180 / math.pi
        std_dec = np.std(all_ra_dec[:, 1])

        delta_ra = min(np.ptp(ra_dec[:, 1]), 360 - np.ptp(ra_dec[:, 1]))
        delta_dec = np.ptp(ra_dec[:, 1])

        self.position_sigmas = np.array(
            [
                (position[0], position[1], sigma, distance)
                for position, sigma, distance in zip(
                    ra_dec, sigmas, distances_to_centroid
                )
            ]
        )
        self.group_attributes.update({"ra": centroid_ra, "dec": centroid_dec})
        self.position_based_features = np.array(
            (mean_ra, std_ra, delta_ra, mean_dec, delta_dec, std_dec),
            dtype=[
                ("mean_ra", float),
                ("std_ra", float),
                ("delta_ra", float),
                ("mean_dec", float),
                ("std_dec", float),
                ("delta_dec", float),
            ],
        )

    def reduce_member_summaries(self):
        """Reduce many individual summaries into a single summary dictionary that
        represents the grouped collection of single pointing candidates.
        """
        # summaries = [spc.summary for spc in self.group_members]

        # Sort by sigma
        sorted_members = sorted(self.group_members, key=lambda d: d.sigma, reverse=True)

        # this is dumb, but for now, the summary is reduced to the summary of
        # the SinglePointingCandidate with the largest sigma value -
        # specifically, the first instance of it the maximum value
        max_sigma = sorted_members[0].sigma
        max_sigma_idx = 0
        max_sigma_cand = sorted_members[0]
        obs_id = max_sigma_cand.obs_id
        datetimes = max_sigma_cand.datetimes
        # TODO: change this to however we really want to reduce the summaries
        reduced_summary = max_sigma_cand

        # For now add only the strongest raw harmonics to the candidates
        # I am not sure at this moment if summing the raw harmonic powers helps or needs to be done differently

        # For adding the raw harmonics the dm_range can be different.
        # To circumvent this I ass only those dm values which are present in all candidates

        # Method commented out for now because candidates arrays are not kept in memory at first
        """
        if len(self.group_members) > 1:
            min_dm = max(
                [
                    min(spc.raw_harmonic_powers_array["dms"])
                    for spc in self.group_members
                ]
            )
            max_dm = min(
                [
                    max(spc.raw_harmonic_powers_array["dms"])
                    for spc in self.group_members
                ]
            )
            all_dms = (
                self.group_members[max_sigma_idx]
                .raw_harmonic_powers_array["dms"]
                .copy()
            )
            all_dms = all_dms[(all_dms >= min_dm) & (all_dms <= max_dm)]

            summed_raw_harmonic_powers = np.zeros(
                (
                    len(all_dms),
                    *self.group_members[max_sigma_idx]
                    .raw_harmonic_powers_array["powers"]
                    .shape[1:],
                )
            )
            for spc in self.group_members:
                try:
                    current_dms = spc.raw_harmonic_powers_array["dms"].copy()
                    current_dms_truncated = current_dms[
                        (current_dms >= min_dm) & (current_dms <= max_dm)
                    ]
                    same_dm = np.array_equal(current_dms_truncated, all_dms)
                    if same_dm:
                        summed_raw_harmonic_powers += spc.raw_harmonic_powers_array[
                            "powers"
                        ][(current_dms >= min_dm) & (current_dms <= max_dm), :, :]
                    else:
                        log.warn(
                            f"Raw harmonic powers could not be added because {current_dms_truncated} and {all_dms} are not the same."
                        )
                except:
                    log.warn(
                        f"Raw harmonic of candidate with shape {spc.raw_harmonic_powers.shape} could not be added to expected shape {summed_raw_harmonic_powers.shape}."
                    )
                    log.warn(
                        f"Best DM: {max_sigma_summary['dm']}, Current DM: {spc.dm}"
                    )
        else:
            # For now setting this to None which should save memory, If None should grab the powers from the best candidate
            summed_raw_harmonic_powers = None
        """
        summed_raw_harmonic_powers = None

        spc_freqs = [spc.freq for spc in self.group_members]
        spc_dms = [spc.dm for spc in self.group_members]
        spc_sigmas = [spc.sigma for spc in self.group_members]

        attributes = {
            "best_sigma": max_sigma,
            "best_dm": max_sigma_cand["dm"],
            "mean_dm": np.mean(spc_dms),
            "delta_dm": np.ptp(spc_dms),
            "best_freq": max_sigma_cand["freq"],
            "all_freqs": spc_freqs,
            "all_dms": spc_dms,
            "all_sigmas": spc_sigmas,
            "mean_freq": np.mean(spc_freqs),
            "delta_freq": np.ptp(spc_freqs),
            "obs_id": obs_id,
            "datetimes": datetimes,
            "summed_raw_harmonic_powers": summed_raw_harmonic_powers,
            "all_summaries": sorted_members,
            "best_nharm": max_sigma_cand["nharm"],
        }

        self.group_summary.update(reduced_summary)
        self.group_attributes.update(attributes)

    def as_candidate(self) -> MultiPointingCandidate:
        """
        From the group members, and pre-calculated group information, populate and
        return a MultiPointingCandidate object.

        Returns
        -------
            A MultiPointingCandidate representation of the group
        """
        return MultiPointingCandidate(
            summary=self.group_summary,
            features=None,  # self.group_features,
            position_features=self.position_based_features,
            position_sigmas=self.position_sigmas,
            **self.group_attributes,
        )


@attrs
class SinglePointingCandidateGrouper:
    """Take a list of SinglePointingCandidate objects and group them into individual
    MultiPointingCandidate objects.
    """

    dbscan_eps = attrib(default=0.75)
    dbscan_min_samples = attrib(default=1)  # default=0 throws an error
    freq_spacing = attrib(default=9.70127682e-04)
    dm_spacing = attrib(default=0.10119793310713615)
    dm_scale = attrib(default=1)
    freq_scale = attrib(default=1)

    def group(self, cands: List[EasyDict]) -> List[MultiPointingCandidate]:
        """
        The top-level function that executes grouping and return the formed
        MultiPointingCandidate objects.

        Parameters
        ----------
        cands: list[EasyDict]
            A list of candidates to group based on their properties

        Returns
        -------
            A list of MultiPointingCandidates
        """
        # create a list of summary info for each candidate, where every item
        # is a tuple in the form (frequency, dm), and the index corresponds to
        # the candidate index in the input list

        f_dm_pairs = map(get_cand_freq_and_dm, cands)
        data = np.asarray(
            [
                [
                    i,
                    p[0] / self.freq_spacing * self.freq_scale,
                    p[1] / self.dm_spacing * self.dm_scale,
                ]
                for i, p in enumerate(f_dm_pairs)
            ]
        )
        # run DBSCAN clustering algorithm on the frequency and DM data
        dbres = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(
            data[:, 1:]
        )
        labels = dbres.labels_
        uniq_labels = set(labels)
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[dbres.core_sample_indices_] = True

        # iterate over the unique labels to access which SinglePointingCandidates have
        # been clustered together
        mp_cands = []
        for k in uniq_labels:
            cluster_member_mask = labels == k
            clustered_cands = data[core_samples_mask & cluster_member_mask]
            clustered_cands_idx = list(set(clustered_cands[:, 0].astype(int)))
            # for each cluster, create a SinglePointingCandidate object and add its
            # MultiPointingCandidate representation to the list
            group = SinglePointingCandidateGroup(
                [cands[idx] for idx in clustered_cands_idx]
            )
            mp_cands.append(group.as_candidate())

        return mp_cands
