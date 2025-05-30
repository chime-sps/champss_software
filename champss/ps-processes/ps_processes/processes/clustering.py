#!/usr/bin/env python

import itertools
import logging
import warnings
from collections import OrderedDict

import colorcet as cc
import numpy as np
from attr import ib as attribute
from attr import s as attrs
from attr.validators import instance_of
from matplotlib import pyplot as plt
from scipy.sparse import SparseEfficiencyWarning
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import paired_distances
from sklearn.neighbors import radius_neighbors_graph, sort_graph_by_row_values
from sps_common.interfaces import Cluster

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


log = logging.getLogger(__name__)


def locatebin(bins, n):
    """
    Helper function for set_merge.

    Find the bin where list n has ended up: Follow bin references until
    we find a bin that has not moved.
    """
    while bins[n] != n:
        n = bins[n]
    return n


# Based on Q and replies here https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections/9400562#9400562
# used code in https://github.com/rikpg/IntersectionMerge
# to check different methods for speed on data derived from 2 <HarmonicallyRelatedClustersCollection>s
# one with 2941 entries and another with 12983
# alexis was 1st and 2nd respectively (for speed)
def set_merge(data, out="bins"):
    """
    Data = list of sets Merge sets which intersect together until all that remain are
    mutually exclusive sets.

    Returns:
        An index map for data
        e.g.  [0,0,2,0,2,5,6,6,0]
        means elements at index 0,1,3,8 are all in a group
        elements at index 2,4 are in a group
        element at 5 is in its own group
        elements at 6,7 are in a group

    Credit: alexis on stackoverflow https://stackoverflow.com/a/9453249/18740127
    """
    if not isinstance(data[0], set):
        raise TypeError("data must be a list of sets")
    bins = list(range(len(data)))  # Initialize each bin[n] == n
    nums = dict()

    data = [
        set(m) for m in data
    ]  # Convert to sets, also don't want to change original!
    for r, row in enumerate(data):
        for num in row:
            if num not in nums:
                # New number: tag it with a pointer to this row's bin
                nums[num] = r
                continue
            else:
                dest = locatebin(bins, nums[num])
                if dest == r:
                    continue  # already in the same bin

                if dest > r:
                    dest, r = r, dest  # always merge into the smallest bin

                data[dest].update(data[r])
                data[r] = None
                # Update our indices to reflect the move
                bins[r] = dest
                r = dest

    # Filter out the empty bins
    if out != "bins":
        have = [m for m in data if m]  # removed this line
        return have
    else:
        # added this as (normally) all I care about is the mapping
        final_bins = [locatebin(bins, i) for i in bins]
        return final_bins


def filter_duplicates_freq_dm(dets):
    """
    Filter detections with the same DM, freq but different nharm, only keeping the
    detection with the highest sigma.

    Args:
        dets (np.ndarray): detections, structured nnumpy array which must have the
            fields "freq", "dm", "sigma"
    """
    tmp = [
        [k, dets[k]["freq"], dets[k]["dm"], dets[k]["sigma"]] for k in range(len(dets))
    ]
    st = sorted(tmp, key=lambda st: (st[1], st[2], st[3]))
    grouped_by_freq_dm = [
        list(v)[:-1] for k, v in itertools.groupby(st, key=lambda st: (st[1], st[2]))
    ]  # the [:-1] grabs the lower-sigma duplicates ([] if there's only one)
    # flatten and grab index
    idx_to_remove = [item[0] for group in grouped_by_freq_dm for item in group]
    log.info(
        "Filtering out duplicate freq,dm detections at multiple nharms removes"
        f" {len(idx_to_remove)} detections"
    )
    return np.delete(dets, idx_to_remove)


def group_duplicates_freq(dets, ignorenharm1=False):
    """
    Groups duplicates based on frequency.

    Input:
    ------
    dets (structured numpy array with 'freq' and 'sigma' fields) = detections

    Output:
    ------
    harmonics, idx_to_skip

    Where harmonics is a dict in the form of
    m: [m, n, o, p]
    where m,n,o,p are all indices of dets
    m was the highest-sigma and m,n,o,p all have the same frequency

    idx_to_skip is a list containing all indices which have been found as a harmonic of something else
    (in the example above n,o,p would be in idx_to_skip, but m would not)
    """
    tmp = [[k, dets[k]["freq"], dets[k]["sigma"]] for k in range(len(dets))]
    st = sorted(tmp, key=lambda st: (st[1], st[2]))
    duplicates_main_idx = [
        list(v)[-1][0] for k, v in itertools.groupby(st, key=lambda st: st[1])
    ]
    duplicates_harmonics = [
        [x[0] for x in list(v)[:-1]]
        for k, v in itertools.groupby(st, key=lambda st: st[1])
    ]

    harmonics = {}
    idx_to_skip = []
    for i, m in enumerate(duplicates_main_idx):
        all_idxs = [m, *duplicates_harmonics[i]]
        normal = True
        if ignorenharm1:
            # need to find a non-nharm=1 detection for the main index
            normal = False
            if dets["nharm"][m] == 1:
                nharms = [dets["nharm"][x] for x in all_idxs]
                if nharms.count(1) == len(nharms):
                    # if all nharm=1 don't care
                    normal = True
                else:
                    # select highest-sig detection with nharm!=1
                    for ii, nharm in enumerate(nharms):
                        if nharm > 1:
                            harmonics[all_idxs[ii]] = all_idxs
                            del all_idxs[ii]
                            idx_to_skip.extend(all_idxs)
                            break
            else:
                normal = True
        if normal:
            harmonics[m] = all_idxs
            idx_to_skip.extend(duplicates_harmonics[i])

    return harmonics, idx_to_skip


def rogue_harmpow_filter_presto(detections):
    """
    Filter out detections where one harmonic is much stronger than the others Based on
    presto's sifting.reject_rogueharmpow which has two criteria: # Max-power harmonic is
    at least 2x more powerful # than the next highest-power harmonic, and is the # 4+th
    harmonic our of 8+ harmonics and # Max-power harmonic is at least 3x more powerful #
    than the next highest-power harmonic, and is the # 2+th harmonic our of 4+ harmonics
    # NB # if the harmonics are [0th, 1st, 2nd, 3rd, 4th, ...] # how the function is
    written in presto 2+th harmonic actually means # 3rd or higher # since i) the
    condition is a > rather than >=, and ii) it's that the index is >2.

    Args:
        detections (np.array): detections output from PowerSpectra search - numpy structured array with fields "dm", "freq", "sigma", "nharm", "harm_idx", "harm_pow"
    """
    if "harm_pow" not in detections.dtype.names:
        log.warning(
            "No harm_pow field in detections, cannot filter out rogue harm powers"
        )
        return detections

    filter_out_idx = set()
    for i in np.where(detections["nharm"] >= 8)[0]:
        harm_pows = detections[i]["harm_pow"][: detections[i]["nharm"]]
        maxharm = np.argmax(harm_pows)
        maxpow = harm_pows[maxharm]
        sortedpows = np.sort(harm_pows)
        if maxharm > 4 and maxpow > 2 * sortedpows[-2]:
            filter_out_idx.add(i)
        elif maxharm > 2 and maxpow > 3 * sortedpows[-2]:
            filter_out_idx.add(i)
    for i in np.where(detections["nharm"] == 4)[0]:
        harm_pows = detections[i]["harm_pow"][: detections[i]["nharm"]]
        maxharm = np.argmax(harm_pows)
        maxpow = harm_pows[maxharm]
        sortedpows = np.sort(harm_pows)
        if maxharm > 2 and maxpow > 3 * sortedpows[-2]:
            filter_out_idx.add(i)
    filter_out_idx = sorted(list(filter_out_idx))
    return np.delete(detections, filter_out_idx)


def rogue_harmpow_filter_presto_tweak(detections):
    """
    Filter out detections where one harmonic is much stronger than the others Based on
    presto's sifting.reject_rogueharmpow which has two criteria: # Max-power harmonic is
    at least 2x more powerful # than the next highest-power harmonic, and is the # 4+th
    harmonic our of 8+ harmonics and # Max-power harmonic is at least 3x more powerful #
    than the next highest-power harmonic, and is the # 2+th harmonic our of 4+
    harmonics.

    This version of the function is tweaked so that the above conditions apply how
    I'd interpret them rather than how the code is writted in presto

    Args:
        detections (np.array): detections output from PowerSpectra search - numpy structured array with fields "dm", "freq", "sigma", "nharm", "harm_idx", "harm_pow"
    """
    if "harm_pow" not in detections.dtype.names:
        log.warning(
            "No harm_pow field in detections, cannot filter out rogue harm powers"
        )
        return detections

    filter_out_idx = set()
    for i in np.where(detections["nharm"] >= 8)[0]:
        harm_pows = detections[i]["harm_pow"][: detections[i]["nharm"]]
        maxharm = np.argmax(harm_pows)
        maxpow = harm_pows[maxharm]
        sortedpows = np.sort(harm_pows)
        if maxharm >= 3 and maxpow > 2 * sortedpows[-2]:
            filter_out_idx.add(i)
        elif maxharm >= 1 and maxpow > 3 * sortedpows[-2]:
            filter_out_idx.add(i)
    for i in np.where(detections["nharm"] == 4)[0]:
        harm_pows = detections[i]["harm_pow"][: detections[i]["nharm"]]
        maxharm = np.argmax(harm_pows)
        maxpow = harm_pows[maxharm]
        sortedpows = np.sort(harm_pows)
        if maxharm >= 1 and maxpow > 3 * sortedpows[-2]:
            filter_out_idx.add(i)
    filter_out_idx = sorted(list(filter_out_idx))
    return np.delete(detections, filter_out_idx)


def rogue_harmpow_filter_alt(detections):
    """
    Filter out detections where one harmonic is much stronger than the others.

    Uses the criteria:
        Max-power harmonic is >= 2 * the sum of all other harmonics
        plus nharm >= 4
        plus it's not the fundamental harmonic

    Args:
        detections (np.array): detections output from PowerSpectra search - numpy structured array with fields "dm", "freq", "sigma", "nharm", "harm_idx", "harm_pow"
    """
    if "harm_pow" not in detections.dtype.names:
        log.warning(
            "No harm_pow field in detections, cannot filter out rogue harm powers"
        )
        return detections

    filter_out_idx = []
    for i in np.where(detections["nharm"] >= 4)[0]:
        harm_pows = detections[i]["harm_pow"][: detections[i]["nharm"]]
        maxharm = np.argmax(harm_pows)
        maxpow = harm_pows[maxharm]
        if maxharm == 0:
            continue
        sortedpows = np.sort(harm_pows)
        sumothers = sortedpows[:-1].sum()
        if maxpow > 2 * sumothers:
            filter_out_idx.append(i)
    return np.delete(detections, filter_out_idx)


def intersect2d_ind_filter0(ar1, ar2):
    """
    Find row wise overlap between two arrays.

    The arrays are assumed to have unique entries except for 0 which will be ignored.
    This is my attempt to perform np.intersect1d on 2D arrays.
    Parameters
    ----------
    ar1: np.ndarray
        The first array. Should be 2D or 1D.

    ar2: np.ndarray
        The second array. Should be 2D or 1D.

    Returns
    -------
    int2d_split: List(np.ndarray)
        The overlap per row

    ar1_indices_split: List(np.ndarray)
        The indices of the overlap in the first array

    ar2_indices_split: List(np.ndarray)
        The indices of the overlap in the second array
    """
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)
    if len(ar1.shape) == 1:
        ar1 = ar1[None, :]
        ar2 = ar2[None, :]
    aux = np.concatenate((ar1, ar2), -1)
    aux_sort_indices = np.argsort(aux, axis=1, kind="mergesort")
    aux = np.take_along_axis(aux, aux_sort_indices, axis=1)
    mask = (aux[:, :-1] == aux[:, 1:]) & (aux[:, :-1] != 0)
    # based on https://stackoverflow.com/posts/53918643/revisions
    take = mask.sum(axis=1)
    int2d_flat = aux[:, :-1][mask]
    int2d_split = np.split(int2d_flat, np.cumsum(take)[:-1])
    ar1_indices = aux_sort_indices[:, :-1][mask]
    ar2_indices = aux_sort_indices[:, 1:][mask] - ar1.shape[1]
    ar1_indices_split = np.split(ar1_indices, np.cumsum(take)[:-1])
    ar2_indices_split = np.split(ar2_indices, np.cumsum(take)[:-1])
    return int2d_split, ar1_indices_split, ar2_indices_split


@attrs(slots=True)
class Clusterer:
    """
    Class to perform clustering on detections.

    Attributes:
    ===========
    cluster_scale_factor: float
        The scale factor to apply between the frequency spacing and the dm spacing during the clustering process.
        A number larger than 1 indicates the numerical spacing between adjacent frequency bin is larger than that
        of adjacent DMs.

    dbscan_eps: float
        From the sklearn documentation :
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN
        parameter to choose appropriately for your data set and distance function.

    dbscan_min_samples: int
        The minimum number of detections point being grouped together for a cluster to be considered as a real
        candidate. Default = 5

    sigma_detection_threshold: int
        The sigma detection threshold used in the power spectra search

    max_ndetect: int
        Maximum number of detections to cluster in DM,freq,harmonic

    group_duplicate_freqs: bool
        Whether to group identical frequencies together before doing the harmonic computations
        (As the harmonic metric is based on the indices summed in the raw power spectrum, this does have an effect)

    metric_method: str
        Method used to calcualte the harmonic metric
        Options:
            "rhp_norm_by_min_nharm": 1 - (the intersection of the two raw harmonic power bins, divided by the max nharm)
                                     (aka how much the bins intersect / max possible they could intersect)
            "rhp_overlap": 1 - (the intersection of the two raw harmonic power bins / the max nharm used in the search)
                           (this would downweight nharm=1 matches)

    metric_combination: str
        Method used to combine the DM-freq distance and harmonic-relation metrics
        Options:
            "multiply": multiply the two together; the harmonic metric modifies the distance metric
            "replace": where the harmonic metric found some relation (aka it is not 1) replace the distance metric
                       value with the harmonic metric value, scaled to be within dbscan_eps

    clustering_method: str
        DBSCAN / HDBSCAN


    min_freq: float
        Don't calculate the harmonic metric for any detection below this freq

    ignore_nharm1: bool
        Don't calculate the harmonic metric for any detection with nharm=1

    rogue_harmpow_scheme: str
        Which scheme to use for rogue harm power rejection:
            "presto" = the same criteria as in presto's sifting
            "tweak" = my tweak of presto's code to actually do what the comment says it does
            "alt" = alternate scheme, must have 4+ harmonics, the max is not the fundamental, and the max is >= 2* sum of the others
    """

    # cluster_scale_factor: float = attribute(default=10)
    freq_scale_factor: float = attribute(default=1)
    dm_scale_factor: float = attribute(default=0.1)
    dbscan_eps: float = attribute(default=1)
    dbscan_min_samples: int = attribute(default=5)
    max_ndetect: int = attribute(
        default=50000
    )  # 32-bit max_ndetect x max_ndetect matrix is ~4GB
    sigma_detection_threshold: int = attribute(default=5)
    group_duplicate_freqs: bool = attribute(default=True)
    metric_method: str = attribute(default="rhp_norm_by_min_nharm")
    metric_combination: str = attribute(default="multiply")
    clustering_method: str = attribute(default="DBSCAN")
    min_freq: float = attribute(default=0)
    ignore_nharm1: bool = attribute(default=False)
    rogue_harmpow_scheme: str = attribute(default="tweak")
    filter_nharm: bool = attribute(default=False)
    remove_harm_idx: bool = attribute(default=False)
    cluster_dm_cut: float = attribute(default=-1)
    overlap_scale: float = attribute(default=1)
    add_dm_when_replace: bool = attribute(default=False)
    num_threads = attribute(validator=instance_of(int), default=8)
    use_sparse: bool = attribute(default=True)
    grouped_freq_dm_scale: float = attribute(default=1)
    use_dbscan_filter: bool = attribute(default=True)
    dbscan_filter_whole_freqs: bool = attribute(default=True)
    dbscan_filter_broad_dm: bool = attribute(default=False)
    dbscan_filter_broad_threshold: bool = attribute(default=100)

    @metric_method.validator
    def _validate_metric_method(self, attribute, value):
        assert value in [
            "rhp_norm_by_min_nharm",
            "rhp_overlap",
            "power_overlap",
            "power_overlap_array",
        ], "metric_method must be either 'rhp_norm_by_min_nharm' or 'rhp_overlap'"

    @metric_combination.validator
    def _validate_metric_combination(self, attribute, value):
        assert value in [
            "multiply",
            "replace",
        ], "metric_combination must be either 'multiply' or 'replace'"

    @clustering_method.validator
    def _validate_clustering_method(self, attribute, value):
        assert value in [
            "DBSCAN",
            "HDBSCAN",
        ], "clustering_method must be either 'DBSCAN' or 'HDBSCAN'"

    @rogue_harmpow_scheme.validator
    def _validate_rogue_harmpow_scheme(self, attribute, value):
        assert value in [
            "presto",
            "tweak",
            "alt",
        ], "harmpow_scheme must be 'presto', 'tweak' or 'alt'"

    def calculate_metric_rhp_overlap(self, rhplist, idx0, idx1, *args):
        """Calculate harmonic distance between two detections
        1 - scale*len(set.intersection(rhp0, rhp1))

        Args:
            rhplist (list): list of sets of PowerSpectra bins used in the sum (aka non-zero values of 'harm_idx' in detections)
            idx0 (int): index of 1st detection (must correspond to the same index in rhplist)
            idx1 (int): index of 2nd detection
            scale (int, optional): scale multiplies the length of the set intersection. Defaults to 1.

        Returns:
            float: value for the harmonic distance
        """
        rhp0 = rhplist[idx0]
        rhp1 = rhplist[idx1]
        out_metric = 1 - len(set.intersection(rhp0, rhp1))
        return out_metric

    def calculate_metric_rhp_overlap_normbyminnharm(
        self, rhplist, idx0, idx1, detections, **kwargs
    ):
        """Calculate harmonic distance between two detections
        1 - len(set.intersection(rhp0, rhp1)) / min(detections['nharm'][idx0], detections['nharm'][idx1])

        Args:
            rhplist (list): list of sets of PowerSpectra bins used in the sum (aka non-zero values of 'harm_idx' in detections)
            idx0 (int): index of 1st detection (must correspond to the same index in rhplist)
            idx1 (int): index of 2nd detection
            detections (np.ndarray): detections output from PowerSpectra search - numpy structured array with fields "dm", "freq", "sigma", "nharm", "harm_idx", "harm_pow"

        Returns:
            _type_: _description_
        """
        rhp0 = rhplist[idx0]
        rhp1 = rhplist[idx1]
        out_metric = 1 - len(set.intersection(rhp0, rhp1)) / min(
            detections["nharm"][idx0], detections["nharm"][idx1]
        )
        return out_metric

    def calculate_metric_power_overlap(self, rhplist, idx0, idx1, detections, **kwargs):
        """
        Calculate harmonic distance based on the power in overlapping bins.

        Args:
            rhplist (list): list of sets of PowerSpectra bins used in the sum (aka non-zero values of 'harm_idx' in detections)
            idx0 (int): index of 1st detection (must correspond to the same index in rhplist)
            idx1 (int): index of 2nd detection
            detections (np.ndarray): detections output from PowerSpectra search - numpy structured array with fields "dm", "freq", "sigma", "nharm", "harm_idx", "harm_pow"

        Returns:
            _type_: _description_
        """
        intersect_bins, intersect_idx0, intersect_idx1 = np.intersect1d(
            detections[idx0]["harm_idx"],
            detections[idx1]["harm_idx"],
            return_indices=True,
        )
        total_power_1 = np.sum(detections[idx0]["harm_pow"])
        total_power_2 = np.sum(detections[idx1]["harm_pow"])
        intersec_power_1 = np.sum(detections[idx0]["harm_pow"][intersect_idx0])
        intersec_power_2 = np.sum(detections[idx1]["harm_pow"][intersect_idx1])
        power_overlap = max(
            intersec_power_1 / total_power_1, intersec_power_2 / total_power_2
        )
        out_metric = 1 - power_overlap
        return out_metric

    def calculate_metric_power_overlap_array(
        self, rhplist, idxs0, idxs1, detections, **kwargs
    ):
        """
        Calculate overlap of power between two detections. Can be used on arrays.

        Args:
            rhplist (list): list of sets of PowerSpectra bins used in the sum (aka non-zero values of 'harm_idx' in detections). Not used here
            idx0 (list(int)): indices of 1st detection (must correspond to the same index in rhplist)
            idx1 (list(int)): indices of 2nd detection
            detections (np.ndarray): detections output from PowerSpectra search - numpy structured array with fields "dm", "freq", "sigma", "nharm", "harm_idx", "harm_pow"

        Returns:
            out_metric (np.ndarray): (1 - power_overlap). Will be 0 for complete overlap and 1
        """
        intersect_bins, intersect_idx0, intersect_idx1 = intersect2d_ind_filter0(
            detections["harm_idx"][idxs0],
            detections["harm_idx"][idxs1],
        )
        if np.isscalar(idxs0):
            powers_0 = np.array(
                [
                    (row.sum(), row[indices].sum())
                    for row, indices in zip(
                        detections["harm_pow"][idxs0][None, :], intersect_idx0
                    )
                ]
            )
            powers_1 = np.array(
                [
                    (row.sum(), row[indices].sum())
                    for row, indices in zip(
                        detections["harm_pow"][idxs1][None, :], intersect_idx1
                    )
                ]
            )
        else:
            powers_0 = np.array(
                [
                    (row.sum(), row[indices].sum())
                    for row, indices in zip(
                        detections["harm_pow"][idxs0], intersect_idx0
                    )
                ]
            )
            powers_1 = np.array(
                [
                    (row.sum(), row[indices].sum())
                    for row, indices in zip(
                        detections["harm_pow"][idxs1], intersect_idx1
                    )
                ]
            )
        powers_overlap = np.stack(
            (
                powers_0[:, 1] / powers_0[:, 0],
                powers_1[:, 1] / powers_1[:, 0],
            ),
            axis=1,
        ).max(1)
        out_metric = 1 - powers_overlap
        return out_metric

    def cluster(
        self,
        detections_in,
        cluster_dm_spacing,
        cluster_df_spacing,
        scheme="combined",
        plot_fname="",
    ):
        """
        Cluster detections in freq-dm-harmonic space.
        Does 2-3 things to thin down the number of detections before custering:
            a) filters out detections with one strong harmonic power using rogue_harmpow_filter_presto
            b) filters out detections with the same freq,dm but different nharm
            If the number of detections is > self.max_ndetect:
            c) raises the sigma threshold until the number of detections is less than self.max_ndetect. This subset is what will be clustered.

        Args:
            detections_in (np.ndaray): detections output from PowerSpectra search - numpy structured array with fields "dm", "freq", "sigma", "nharm", "harm_idx", "harm_pow"
            cluster_dm_spacing (float): spacing between DM trials
            scheme (str, optional): determines qhat kind of clustering to perform
                "combined": cluster based on combined metric from DM-freq and harmonic distances
                "dmfreq": cluster only in DM-freq space
                "harm": cluster only in harmonic space
                Defaults to "combined"
            plot_fname (str, optional): if not '', a plot of the detections and clusters will be made and saved to this file
        Returns:
            np.ndarray: subset of detections_in actually used for the clustering
            np.ndarray: labels resulting from the clustering
            float: sigma lower limit used during clustering
        """
        log.info("Starting clustering")
        if scheme not in ["combined", "dmfreq", "harm"]:
            raise AttributeError(
                f'Invalid value for scheme ({scheme}). Valid options are "combined",'
                ' "dmfreq", "harm"'
            )

        # Set rogue harm powers filtering method
        if self.rogue_harmpow_scheme == "presto":
            filter_rogue_harmpows = rogue_harmpow_filter_presto
        elif self.rogue_harmpow_scheme == "tweak":
            filter_rogue_harmpows = rogue_harmpow_filter_presto_tweak
        elif self.rogue_harmpow_scheme == "alt":
            filter_rogue_harmpows = rogue_harmpow_filter_alt

        # Filter out rogue harmonic powers
        detections_filtered = filter_rogue_harmpows(detections_in)
        log.info(
            "Rogue harmonic power filter reduced detections from"
            f" {len(detections_in)} to {len(detections_filtered)}"
        )
        del detections_in

        # Filter out duplicate freq,dm detections (with different nharm)
        detections_filtered = filter_duplicates_freq_dm(detections_filtered)
        log.info(
            f"Duplicate freq,dm filter reduced detections to {len(detections_filtered)}"
        )

        if self.use_dbscan_filter:
            log.info("Running dbscan filter")
            data_filter = np.vstack(
                (
                    detections_filtered["dm"]
                    / cluster_dm_spacing
                    * self.dm_scale_factor,
                    detections_filtered["freq"]
                    / cluster_df_spacing
                    * self.freq_scale_factor,
                ),
                dtype=np.float32,
            ).T
            metric_array = radius_neighbors_graph(
                data_filter, 1.1 * self.dbscan_eps, p=2, mode="distance"
            )
            metric_array = sort_graph_by_row_values(
                metric_array, warn_when_not_sorted=False
            )
            db_filter = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                metric="precomputed",
            ).fit(metric_array)
            bad_freqs = []
            filtered_labels_low_dm = []
            filtered_labels_broad_dm = []
            filtered_indices_low_dm = []
            filtered_indices_broad_dm = []
            for i in range(max(db_filter.labels_) + 1):
                filtered = False
                current_indices = np.arange(detections_filtered.shape[0])[
                    db_filter.labels_ == i
                ]
                det_sample = detections_filtered[current_indices]
                det_max_sigma_pos = np.argmax(det_sample["sigma"])
                det_max_sigma_dm = det_sample["dm"][det_max_sigma_pos]
                det_dm_ptp = np.ptp(det_sample["dm"])
                if det_max_sigma_dm <= self.cluster_dm_cut:
                    filtered_indices_low_dm.extend(current_indices)
                    filtered_labels_low_dm.append(i)
                    bad_freqs.append(
                        (
                            det_sample["freq"].mean(),
                            det_sample["freq"].min(),
                            det_sample["freq"].max(),
                        )
                    )
                    filtered = True
                if self.dbscan_filter_broad_dm:
                    if self.dbscan_filter_broad_threshold < det_dm_ptp:
                        filtered_indices_broad_dm.extend(current_indices)
                        filtered_labels_broad_dm.append(i)
                        if not filtered:
                            bad_freqs.append(
                                (
                                    det_sample["freq"].mean(),
                                    det_sample["freq"].min(),
                                    det_sample["freq"].max(),
                                )
                            )
                        filtered = True

            filtered_labels = filtered_labels_low_dm + filtered_labels_broad_dm
            filtered_indices = filtered_indices_low_dm + filtered_indices_broad_dm
            bad_freqs = np.asarray(bad_freqs)
            bad_low_dm_freqs = len(filtered_indices)
            log.info(
                f"Dbscan filter removed {len(filtered_indices_low_dm)} detections in low DM clusters."
            )
            if self.dbscan_filter_broad_dm:
                log.info(
                    f"Dbscan filter removed {len(filtered_indices_broad_dm)} detections in broad DM clusters."
                )
                log.info(
                    f"Filtered {len(set(filtered_indices))} from low or broad clusters."
                )

            if self.dbscan_filter_whole_freqs:
                for i in range(max(db_filter.labels_) + 1):
                    if i not in filtered_labels:
                        current_indices = np.arange(detections_filtered.shape[0])[
                            db_filter.labels_ == i
                        ]
                        det_sample = detections_filtered[current_indices]
                        mean_freq = det_sample["freq"].mean()
                        for row in bad_freqs:
                            if mean_freq > row[1] and mean_freq < row[2]:
                                filtered_indices.extend(current_indices)
                                break
                bad_all_freqs = len(filtered_indices)
                log.info(
                    "Dbscan filter removed additonal"
                    f" {bad_all_freqs - bad_low_dm_freqs} detection with same"
                    " frequencies as already filtered DM clusters."
                )
            mask = np.full(detections_filtered.shape[0], True)
            mask[filtered_indices] = False
            detections_filtered = detections_filtered[mask]

            del data_filter
            del metric_array
            del db_filter

        detections = detections_filtered[
            detections_filtered["sigma"] > self.sigma_detection_threshold
        ]

        # thin down detections if there are too many
        log.info(
            "Number of detections over threshold sigma"
            f" ({self.sigma_detection_threshold}): {len(detections)}"
        )
        sig_limit = self.sigma_detection_threshold
        if len(detections) > self.max_ndetect:
            log.info(
                f"The number of detections {len(detections)} is greater than the max"
                f" for the usual clustering scheme {self.max_ndetect}"
            )
            sig_limit = self.sigma_detection_threshold + 1
            selection = np.where(detections_filtered["sigma"] > sig_limit)[0]
            while len(selection) > self.max_ndetect:
                sig_limit += 0.2
                selection = np.where(detections_filtered["sigma"] > sig_limit)[0]
            detections = detections_filtered[selection]
            log.warning(
                f"The minimum sigma has been raised to {sig_limit} to reduce the number"
                f" of detections to {len(detections)}"
            )
        del detections_filtered

        # make data products necessary for clustering and making the harmonic metric
        data = np.vstack(
            (
                detections["dm"] / cluster_dm_spacing * self.dm_scale_factor,
                detections["freq"] / cluster_df_spacing * self.freq_scale_factor,
            ),
            dtype=np.float32,
        ).T

        rhps = [set(det["harm_idx"][: det["nharm"]]) for det in detections]

        # Find duplicate frequencies and only keep the highest-sigma detection for the harmonic metric caluclation
        if self.group_duplicate_freqs:
            log.info("Grouping duplicate frequencies together first")
            harm, idx_to_skip = group_duplicates_freq(
                detections, ignorenharm1=self.ignore_nharm1
            )
            # set lookup is faster
            idx_to_skip = set(idx_to_skip)
            log.info(
                f"Grouping duplicate frequencies removed {len(idx_to_skip)} detections"
                " from the harmonic metric computation"
            )
        else:
            harm = None
            idx_to_skip = []

        if scheme in ["combined", "dmfreq"]:
            log.info("Starting freq-DM distance metric computation")
            if self.use_sparse:
                metric_array = radius_neighbors_graph(
                    data, 1.1 * self.dbscan_eps, p=2, mode="distance"
                )
                metric_array.sort_indices()
            else:
                metric_array = pairwise_distances(data, n_jobs=self.num_threads)

            log.info("Finished freq-DM distance metric computation")

        if scheme in ["combined", "harm"]:
            # Set harmonic metric method
            if self.metric_method == "rhp_norm_by_min_nharm":
                calculate_harm_metric = self.calculate_metric_rhp_overlap_normbyminnharm
                log.debug(
                    "calculate_harm_metric set to"
                    " calculate_metric_rhp_overlap_normbyminnharm"
                )
            elif self.metric_method == "rhp_overlap":
                calculate_harm_metric = self.calculate_metric_rhp_overlap
                log.debug("calculate_harm_metric set to calculate_metric_rhp_overlap")
            elif self.metric_method == "power_overlap":
                calculate_harm_metric = self.calculate_metric_power_overlap
                log.debug("calculate_harm_metric set to calculate_metric_power_overlap")
            elif self.metric_method == "power_overlap_array":
                calculate_harm_metric = self.calculate_metric_power_overlap_array
                log.debug(
                    "calculate_harm_metric set to calculate_metric_power_overlap_array"
                )

            # Organise raw harmonic power bins into supersets
            # This restricts the parameter space for which you need to calcualte the harmonic metric
            #
            # Exclude things from the harmonic metric if desired
            bin_map = range(len(rhps))
            if self.min_freq and self.ignore_nharm1:
                bin_map = list(
                    np.where(
                        (detections["freq"] > self.min_freq) & (detections["nharm"] > 1)
                    )[0]
                )
            elif self.min_freq:
                bin_map = list(np.where(detections["freq"] > self.min_freq)[0])
            elif self.ignore_nharm1:
                bin_map = list(np.where(detections["nharm"] > 1)[0])

            rhps_subsec = [rhp for i, rhp in enumerate(rhps) if i in bin_map]

            bins = set_merge(rhps_subsec)
            # from https://stackoverflow.com/a/53999192
            groups = OrderedDict()
            for i, v in enumerate(bins):
                if bin_map[i] in idx_to_skip:
                    continue
                try:
                    groups[v].append(bin_map[i])
                except KeyError:
                    groups[v] = [bin_map[i]]
            # if cared about actual value of v that should probably be bin_map[v]

            largest_group_size = max([len(group) for group in groups.values()])
            log.info(
                f"Largest group size for metric computation is {largest_group_size}"
            )
            grouped_ids = [g for g in list(groups.values()) if len(g) > 1]
            del groups

            log.info("Starting harmonic distance metric computation")
            if scheme not in ["combined", "dmfreq"]:
                metric_array = np.ones((data.shape[0], data.shape[0]), dtype=np.float32)

            if self.metric_combination == "multiply":
                threshold_for_new_vals = 1
            elif self.metric_combination == "replace":
                threshold_for_new_vals = self.dbscan_eps
            else:
                threshold_for_new_vals = np.inf

            # to save on memory should probably alter the DMfreq_dist_metric in-place instead
            all_indices_0 = []
            all_indices_1 = []
            all_metric_vals = []
            if self.metric_method == "power_overlap_array":
                chunk_size = 2000
                index_pairs = [
                    index_pair
                    for id_group in grouped_ids
                    for index_pair in list(itertools.combinations(id_group, 2))
                ]
                index_pairs = np.asarray(index_pairs)
                index_pairs = np.split(
                    index_pairs, np.arange(chunk_size, index_pairs.shape[0], chunk_size)
                )
            else:
                index_pairs = [
                    index_pair
                    for id_group in grouped_ids
                    for index_pair in list(itertools.combinations(id_group, 2))
                ]
            for split in index_pairs:
                if self.metric_method == "power_overlap_array":
                    rows_0 = split[:, 0]
                    rows_1 = split[:, 1]
                else:
                    rows_0 = split[0]
                    rows_1 = split[1]
                    split = np.array(split).reshape(1, 2)
                metric_vals = (
                    calculate_harm_metric(rhps, rows_0, rows_1, detections)
                    * self.overlap_scale
                )
                if np.isscalar(metric_vals):
                    metric_vals = np.array(metric_vals).reshape(1)

                used_indices = metric_vals < threshold_for_new_vals
                metric_vals = metric_vals[used_indices]
                split = split[used_indices]
                if not len(metric_vals):
                    continue
                if self.group_duplicate_freqs:
                    indices_0 = []
                    indices_1 = []
                    group_metric_vals = []

                    for index, row in enumerate(split):
                        indices_0_part = np.tile(harm[row[0]], len(harm[row[1]]))
                        indices_1_part = np.repeat(harm[row[1]], len(harm[row[0]]))
                        indices_0.append(indices_0_part)
                        indices_1.append(indices_1_part)

                        group_metric_vals.append(
                            [
                                metric_vals[index],
                            ]
                            * len(indices_0_part)
                        )
                    indices_0 = np.concatenate(indices_0)
                    indices_1 = np.concatenate(indices_1)
                    metric_vals = np.concatenate(group_metric_vals)
                else:
                    indices_0 = split[:, 0]
                    indices_1 = split[:, 1]
                if self.add_dm_when_replace and self.metric_combination == "replace":
                    dm_dists = paired_distances(
                        data[indices_0, :1], data[indices_1, :1]
                    )
                    metric_vals += dm_dists
                    used_indices_dm = metric_vals < self.dbscan_eps
                    metric_vals = metric_vals[used_indices_dm]
                    indices_0 = indices_0[used_indices_dm]
                    indices_1 = indices_1[used_indices_dm]
                    if not len(metric_vals):
                        continue

                all_indices_0.append(indices_0)
                all_indices_1.append(indices_1)
                all_metric_vals.append(metric_vals)

            all_indices_0 = np.concatenate(all_indices_0)
            all_indices_1 = np.concatenate(all_indices_1)
            all_metric_vals = np.concatenate(all_metric_vals)

            if scheme == "combined":
                if self.metric_combination == "multiply":
                    metric_array[all_indices_0, all_indices_1] *= metric_vals
                    metric_array[indices_1, indices_0] *= metric_array[
                        all_indices_0, all_indices_1
                    ]
                elif self.metric_combination == "replace":
                    metric_array[all_indices_0, all_indices_1] = all_metric_vals
                    metric_array[all_indices_1, all_indices_0] = all_metric_vals
            else:
                metric_array[all_indices_0, all_indices_1] = all_metric_vals
                metric_array[all_indices_1, all_indices_0] = all_metric_vals
            group_indices_0 = []
            group_indices_1 = []
            if self.grouped_freq_dm_scale != 0:
                all_dm_dists = []
            for i in range(metric_array.shape[0]):
                # metric_array[i, i] = 0
                # No min dist_for sparse array needed because diagonal is set in
                # sklearn dbscan method
                if i in idx_to_skip:
                    continue
                if self.group_duplicate_freqs:
                    index_0, index_1 = np.meshgrid(harm[i], harm[i])
                    index_0 = index_0.flatten()
                    index_1 = index_1.flatten()

                    group_indices_0.append(index_0)
                    group_indices_1.append(index_1)
                    if self.grouped_freq_dm_scale != 0:
                        dm_dists = (
                            paired_distances(data[index_0, :1], data[index_1, :1])
                            * self.grouped_freq_dm_scale
                        )
                        all_dm_dists.append(dm_dists)
            log.info("Updating grouped frequencies.")
            group_indices_0 = np.concatenate(group_indices_0)
            group_indices_1 = np.concatenate(group_indices_1)
            if self.grouped_freq_dm_scale != 0:
                metric_array[group_indices_0, group_indices_1] = np.concatenate(
                    all_dm_dists
                )
            else:
                metric_array[group_indices_0, group_indices_1] = 0

            log.info("Finished harmonic distance metric computation")

        if self.use_sparse:
            metric_array = sort_graph_by_row_values(
                metric_array, warn_when_not_sorted=False
            )
        if self.clustering_method == "DBSCAN":
            log.info("Starting DBSCAN")
            db = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                metric="precomputed",
            ).fit(metric_array)
            log.info("Finished DBSCAN")
        elif self.clustering_method == "HDBSCAN":
            log.info("Starting HDBSCAN")
            db = HDBSCAN(
                min_samples=self.dbscan_min_samples,
                metric="precomputed",
            ).fit(metric_array)
            log.info("Finished HDBSCAN")

        nclusters = len(np.unique(db.labels_))
        if -1 in db.labels_:
            nclusters -= 1
        log.info(f"Clusters found: {nclusters}")

        if plot_fname:
            plot_clusters(detections, db.labels_, fname=plot_fname)

        return detections, db.labels_, sig_limit

    def make_clusters(
        self,
        detections_in,
        cluster_dm_spacing,
        cluster_df_spacing,
        plot_fname="",
        only_injections=False,
    ):
        """
        Make clusters from detections. This calls the cluster function, and packages up
        the results nicely.

        Args:
            detections_in (np.ndaray): detections output from PowerSpectra search - numpy structured array
                with fields "dm", "freq", "sigma", "nharm", "harm_idx", "harm_pow", "injection"
            cluster_dm_spacing (float): spacing between DM trials
            plot_fname (str, optional): A plot of the detections and clusters will be made and saved to
                this file. Defaults to "".
            filter_nharm (bool, optional): If True, for each cluster, only keep detections where the nharm
                matches that of the highest-sigma detection. Defaults to False.
            remove_harm_idx (bool, optional): If True, remove the "harm_idx" field from the cluster
                detections. Defaults to False.
            cluster_dm_cut (float, optional): Filter all clusters equal or below this DM.
            only_injections: bool Whether non-injections are filtered out. Default: False

        Returns:
            clusters (dict): A dict of Cluster objects
            summary (np.ndarray): A summary dict for the highest-sigma detection in all clusters found.
                                  Fields are "cluster_id", "freq", "sigma", "nharm", "injection"
                                  cluster_id corresponds to the keys in clusters
            sig_limit (float): The minimum sigma used when clustering.
                               If there were many detections this may be higher than the limi used in the search
        """
        detections, cluster_labels, sig_limit = self.cluster(
            detections_in,
            cluster_dm_spacing,
            cluster_df_spacing,
            scheme="combined",
            plot_fname=plot_fname,
        )
        unique_labels = np.unique(cluster_labels)
        clusters = {}
        summary = {}
        zero_dm_count = 0

        if not np.all(unique_labels == -1):
            # Could use old labels, but new labels prevent gaps if cluster is filtered out
            current_label = 0
            for lbl in unique_labels:
                if lbl == -1:
                    continue
                cluster = Cluster.from_raw_detections(detections[cluster_labels == lbl])
                if self.cluster_dm_cut >= cluster.dm:
                    zero_dm_count += 1
                    continue
                if self.filter_nharm:
                    cluster.filter_nharm()
                if self.remove_harm_idx:
                    cluster.remove_harm_idx()
                    cluster.remove_harm_pow()
                if only_injections and cluster.injection_index == -1:
                    continue
                clusters[current_label] = cluster
                summary[current_label] = dict(
                    freq=cluster.freq,
                    dm=cluster.dm,
                    sigma=cluster.sigma,
                    nharm=cluster.nharm,
                    harm_idx=cluster.harm_idx,
                    injection=cluster.injection_index,
                )
                current_label += 1
        if zero_dm_count:
            log.info(
                f"Filtered {zero_dm_count} clusters below or equal"
                f" {self.cluster_dm_cut} DM."
            )
        used_detections_len = len(detections)
        return clusters, summary, sig_limit, used_detections_len


def plot_clusters(
    detections,
    labels,
    ax=None,
    fname="",
    parameters=None,
    labels_want=None,
    return_plt=False,
    plot_all_dets=False,
):
    """
    Plot clusters.

    Args:
        detections (np.ndarray): Detections output from PowerSpectra search - numpy structured array with fields "dm", "freq", "sigma", "nharm", "harm_idx", "harm_pow"
        labels (np.ndarray): Labels resulting from clustering (must be the same length as detections). If None will plot all detections as noise
        ax (plt.axis, optional): pyplot axis. Defaults to None.
        fname (str, optional): Save plot to this filename. Defaults to "".
        parameters (dict, optional): Any parameters and values wish to add to the plot title. Defaults to None.
        labels_want (list, optional): Only plot clusters corresponding to the subset of labels in this list. Defaults to None.
        return_plt (bool, optional): Whether to return fig, ax (otherwise plt.show() is called). Defaults to False.
        plot_all_dets (bool, optional): Plot x's for all detections, even if clustered. Defaults to False.

    Returns:
        fig, ax (only if return_plt=True)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    labels = labels if labels is not None else np.array([-1] * detections.shape[0])
    # Black removed and is used for noise instead.
    if labels_want is None:
        unique_labels = set(labels)
    else:
        unique_labels = set(labels_want)
    colors = [cc.cm.glasbey(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_index = np.where(labels == k)[0]
        ax.plot(
            detections["dm"][class_index],
            detections["freq"][class_index],
            "x" if k == -1 else "o",
            markerfacecolor=tuple(col),
            markeredgecolor=tuple(col),
            markersize=4 if k == -1 else 8,
            alpha=0.2,  # if k != -1 else 1,
        )
        if plot_all_dets and k != -1:
            ax.plot(
                detections["dm"][class_index],
                detections["freq"][class_index],
                "x",
                markerfacecolor=tuple(col),
                markeredgecolor=tuple(col),
                markersize=4,
                alpha=1,
            )
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    preamble = "Estimated"
    title = f"{preamble} number of clusters: {n_clusters_}"
    if labels_want is not None:
        title += " | clusters: " + ",".join([f"{lw}" for lw in labels_want])
    if parameters is not None:
        parameters_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        title += f"\n{parameters_str}"
    ax.set_title(title)
    if return_plt:
        return fig, ax
    else:
        plt.tight_layout()
        if fname:
            plt.savefig(fname)
        else:
            plt.show()
            plt.close()
