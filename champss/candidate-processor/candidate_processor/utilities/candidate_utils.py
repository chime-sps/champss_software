import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sps_databases import db_api, db_utils

log = logging.getLogger(__name__)


def clean_power_spec(data, threshold):
    sigmas = np.asarray([d[4] for d in data])
    to_pop = np.where(sigmas < threshold)[0]
    log.info(f"Number of low sigma candidates to remove = {len(to_pop)}")
    data = np.delete(data, to_pop, axis=0)

    return data


def clean_hhat(data):
    """
    Identify bad frequencies in hhat array. Performs a median across the DM space and
    identifies common frequencies. Astrophysical frequencies are expected to be
    clustered near a small DM range.

    Paramaters
    ==========
    data: np.ndarray
        hhat array to clean

    Returns
    =======
    data: np.ndarray
        cleaned hhat array
    """

    spec = np.nanmedian(data[range(0, data.shape[0], 10)], axis=0)
    mean = np.nanmean(spec)
    std = np.nanstd(spec)
    out = mean + 5 * std
    m = np.nanmedian(spec)
    outliers = [idx for idx in range(len(spec)) if spec[idx] > out]
    data[:, outliers] = m
    return data


def extract_bright_hhat(
    data, dm_labels=None, freq_labels=None, dc_label=None, threshold=5.0, filename=None
):
    """
    Identify bright hhat triggers in the hhat array above an input threshold.

    Paramaters
    ==========
    data: np.ndarray (ndm, nfreq)
        hhat array for a given duty cycle to clean
    dm_labels: np.array
        array of DMs hhat was computed over. Default: None
    freq_labels: np.array
        array of spin frequencies hhat was computed over. Default: None
    dc_label: float
        Duty cycle of the array. Default: None
    threshold: float
        hhat noise threshold. Default: 5.0
    filename: str
        filename of the plot file.

    Returns
    =======
    wh: np.ndarray
        Array of indices containing hhat values above the threshold.
    """

    wh = np.where(data > threshold)
    if (
        dm_labels is not None
        and freq_labels is not None
        and dc_label is not None
        and filename is not None
    ):
        values_to_plot = data[wh[0], wh[1]]
        dms_to_plot = dm_labels[wh[0]]
        freqs_to_plot = freq_labels[wh[1]]
        figure = plt.figure(figsize=(5, 5))
        plt.scatter(freqs_to_plot, dms_to_plot, s=values_to_plot, c="black")
        plt.xscale("Log")
        plt.xlabel("spin freq Hz")
        plt.ylabel("DM (pc/cc)")
        plt.title(f"Duty cycle: {dc_label * 100} %")
        plt.savefig(filename)
        plt.clf()
    return wh


def perform_clustering(data, dc_value, min_group_size=5, filename=None):
    """
    Perform a clustering analysis on the hhat-array for a given duty cycle. This uses
    Scikit Learn's DBSCAN algorithm.

    Paramaters
    ==========
    data: np.ndarray (ndm,nfreq)
        hhat array to perform a clustering analysis on
    dc_value: float
        value of the duty cycle for each element in this h-hat array.
    min_group_size: int
        minimum group size to consider as a cluster. Default: 5
    filename: str
        filename of the plot file.


    Returns
    =======
    groups: dict of np.ndarray
        groups of clustered hhat points.
    """
    log.info(f"Number of points to cluster = {len(data)}")
    # Compute DBSCAN
    db = DBSCAN(eps=0.125, min_samples=3).fit(data[:, 0:2])
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    groups = {}
    for k in unique_labels:
        class_member_mask = labels == k
        xy = data[class_member_mask & core_samples_mask]
        if k != -1 and len(xy) >= min_group_size:
            groups[k] = xy
            if filename is not None:
                plt.scatter(xy[:, 0], xy[:, 1], c="black", s=5)
    log.info("Estimated number of clusters: %d" % len(groups))
    log.info("Estimated number of noise points: %d" % n_noise_)
    if filename is not None:
        plt.xscale("Log")
        plt.xlabel("Spin Freq Hz")
        plt.ylabel("DM (pc/cc)")
        plt.title("Clustering -- Duty cycle %.2f" % (dc_value))
        plt.savefig(filename)
        plt.clf()
    return groups


def rescale_clusters(
    groups,
    curr_freq_diff,
    curr_dm_diff,
    original_freq_diff,
    original_dm_diff,
    freq_labels,
):
    """
    Rescale values to original level and make the final groups that will become
    candidates.

    Paramaters
    ==========
    groups: dict of np.ndarray
        hhat array to clean
    curr_freq_diff: float
        current frequency spacing in the data.
    curr_dm_diff: float
        current dm spacing in the data.
    original_freq_diff: float
        original frequency spacing in the data.
    original_dm_diff: float
        original dm spacing in the data.
    freq_labels: np.array
        array of spin frequencies hhat was computed over. Default: None

    Returns
    =======
    final_groups: dict of np.ndarray
        final rescaled groups
    final_group_idx: dict of np.ndarray
        indices of hhats in the group.
    """
    to_remove = []
    freq_labels.sort()
    for g in groups:
        if groups[g].shape[0] <= 5:
            to_remove.append(g)
        else:
            scaled_freqs = groups[g][:, 0]
            freqs = scale_data(
                scaled_freqs, current_diff=curr_freq_diff, diff=original_freq_diff
            )
            groups[g][:, 0] = freqs
            scaled_dms = groups[g][:, 1]
            dms = scale_data(
                scaled_dms, current_diff=curr_dm_diff, diff=original_dm_diff
            )
            groups[g][:, 1] = dms
    for g in sorted(to_remove, reverse=True):
        del groups[g]
    final_groups = {}
    for g in groups:
        final_groups[g] = [tuple(grp) for grp in groups[g]]
    return final_groups


def rescale_power_spec_clusters(
    groups, curr_freq_diff, curr_dm_diff, original_freq_diff, original_dm_diff
):
    """
    Rescale values to original level and make the final groups that will become
    candidates.

    Paramaters
    ==========
    groups: dict of np.ndarray
        hhat array to clean
    curr_freq_diff: float
        current frequency spacing in the data.
    curr_dm_diff: float
        current dm spacing in the data.
    original_freq_diff: float
        original frequency spacing in the data.
    original_dm_diff: float
        original dm spacing in the data.

    Returns
    =======
    final_groups: dict of np.ndarray
        final rescaled groups
    final_group_idx: dict of np.ndarray
        indices of hhats in the group.
    """
    to_remove = []
    for g in groups:
        if groups[g].shape[0] <= 5:
            to_remove.append(g)
        else:
            scaled_freqs = groups[g][:, 0]
            freqs = scale_data(
                scaled_freqs, current_diff=curr_freq_diff, diff=original_freq_diff
            )
            groups[g][:, 0] = freqs
            scaled_dms = groups[g][:, 1]
            dms = scale_data(
                scaled_dms, current_diff=curr_dm_diff, diff=original_dm_diff
            )
            groups[g][:, 1] = dms
    for g in sorted(to_remove, reverse=True):
        del groups[g]
    final_groups = {}
    for g in groups:
        final_groups[g] = [tuple(gp) for gp in groups[g]]
    return final_groups


def scale_data(data, current_diff=0.0, diff=0.125):
    """
    Scale the data for clustering analysis. increase the spacing in the data to an
    optimal level.

    Parameters
    ==========
    data: np.array
        array of values to update the spacing of.
    current_diff: float
        current spacing in the data
    diff: float
        new spacing of the data

    returns
    =======
    data: np.array
        array of values spaced to the new level
    """
    data /= current_diff
    data *= diff
    return data


def make_rec_array(curr_group):
    """
    Make a dictionary of array into a dictionary of record array of type dtype=[("f",
    "float"), ("dm", "float"), ("dc", "float"), ("sigma", "float")].

    Parameters
    ==========
    curr_group: dict of tuple

    Returns
    =======
    curr_group: dict of record array
    """
    for g in curr_group:
        # tuple_g = [tuple(gp) for gp in curr_group[g]]
        group = np.array(
            curr_group[g],
            dtype=[
                ("f", "float"),
                ("dm", "float"),
                ("dc", "float"),
                ("sigma", "float"),
            ],
        )
        # remove zero dm candidates
        curr_group[g] = group
    return curr_group


def group_summary(group):
    """
    Make a summary dictionary of each group and store the following information:
    frequency, duty cycle, DM of highest hhat value, min DM of the group and max DM of the group. Also removes zero DM candidate groups from the data

    Paramaters
    ==========
    group: dict of record array

    harm_filter: boolean trigger for harmonic filter

    Returns
    =======
    summary: dict of information stated above.
    """
    summary = {}
    zero_dm_idx = []
    for g in group:
        f = group[g]["f"][np.argmax(group[g]["sigma"])]
        dm = group[g]["dm"][np.argmax(group[g]["sigma"])]
        summary[g] = {}
        summary[g]["f"] = f
        summary[g]["dc"] = group[g]["dc"][0]
        summary[g]["dm"] = group[g]["dm"][np.argmax(group[g]["sigma"])]
        summary[g]["min_dm"] = np.min(group[g]["dm"])
        summary[g]["max_dm"] = np.max(group[g]["dm"])
        summary[g]["sigma"] = np.max(group[g]["sigma"])
        summary[g]["harmonics"] = {}
        # mark candidate group with DM < 1 as RFI
        if dm < 0.99:
            summary[g]["rfi"] = True
        else:
            summary[g]["rfi"] = False

    return group, summary


def harmonic_filter(summary, freq_spacing, dm_thresh=0.99, f_thresh=0.1):
    """
    Associate any candidates of the same frequency with each other. Apply a harmonic
    filter to group candidates that are harmonically related to the brightest detection.
    It searches up to 32 harmonics above and 8 harmonics below the brightest detection.
    Does not group harmonics for candidates with f<f_thresh or dm<dm_thresh.

    Parameters
    ----------
    summary: dict of summary from group_summary

    freq_spacing: frequency spacing of the input data

    Returns
    -------
    summary: dict of summary with entries that are harmonically related to a brighter detection removed
    """
    idx_to_skip = []
    harmonics = {}  # main_harmonic: [list, of, harmonics]
    log.info("Finding duplicate frequencies")
    duplicate_indices = {}  # main_index: [sub, indices]
    # associate any candidates which have the same frequency with each other
    # pick a "main" frequency, and don't include the others in the harmonic search
    group_summary = pd.DataFrame.from_dict(summary, orient="index")
    unique_freqs, counts = np.unique(group_summary["f"], return_counts=True)
    duplicate_freqs = unique_freqs[np.where(counts > 1)]
    for freq in duplicate_freqs:
        group = group_summary.loc[group_summary["f"] == freq]
        highdm = group.loc[group["dm"] > dm_thresh]
        # select a "main" index. highest hhat with dm > threshold
        # or if all under dm threshold, just highest hhat
        if highdm.empty:
            main_index = group.loc[group["sigma"] == group["sigma"].max()][0]
        else:
            main_index = highdm.loc[highdm["sigma"] == highdm["sigma"].max()].index[0]
        sub_index = list(group[group.index != main_index].index)
        duplicate_indices[main_index] = sub_index
        idx_to_skip.extend(sub_index)
        log.debug(f"frequency {freq} assoc with candidates {main_index}, {sub_index}")
    idx_to_skip.sort()
    log.debug("idx to skip")
    log.debug(idx_to_skip)

    log.info("Finding harmonically related candidates")
    # find harmonically assicated candidates
    sorted_summary_idx = sorted(
        summary, key=lambda i: summary[i]["sigma"], reverse=True
    )
    for n, idx in enumerate(sorted_summary_idx):
        harms_n = []
        log.debug(f"processing candidate {sorted_summary_idx[n]}")
        if idx in idx_to_skip:
            log.debug(f"skipping {sorted_summary_idx[n]} - in skip list")
            continue
        # if it's a duplicate, add those as harmonics
        # this needs to go above the low dm/freq filter
        try:
            harms_n.extend(duplicate_indices[sorted_summary_idx[n]])
            log.debug(
                f"{sorted_summary_idx[n]} has duplicate freq candidates"
                f" {duplicate_indices[sorted_summary_idx[n]]}"
            )
        except (
            KeyError
        ):  # error raised on the duplicates dict when there are no duplicates
            pass
        # skip low dm candidates from forming harmonic groups
        if summary[sorted_summary_idx[n]]["dm"] < dm_thresh:
            log.debug(f"skipping {sorted_summary_idx[n]} - low dm")
            if harms_n:
                harms_n.sort()
                harmonics[sorted_summary_idx[n]] = harms_n
                log.debug(f"harmonics of {sorted_summary_idx[n]}:\n{harms_n}")
            continue
        # don't group candidates with f < 0.1
        if summary[sorted_summary_idx[n]]["f"] < f_thresh:
            log.debug(f"skipping {sorted_summary_idx[n]} - low freq")
            if harms_n:
                harms_n.sort()
                harmonics[sorted_summary_idx[n]] = harms_n
                log.debug(f"harmonics of {sorted_summary_idx[n]}:\n{harms_n}")
            continue
        # identify harmonics
        for m in np.arange(n + 1, len(sorted_summary_idx)):
            is_harm = False
            if sorted_summary_idx[m] in idx_to_skip:
                continue
            max_f = np.max(
                np.asarray(
                    [
                        summary[sorted_summary_idx[m]]["f"],
                        summary[sorted_summary_idx[n]]["f"],
                    ]
                )
            )
            min_f = np.min(
                np.asarray(
                    [
                        summary[sorted_summary_idx[m]]["f"],
                        summary[sorted_summary_idx[n]]["f"],
                    ]
                )
            )
            harm = round(max_f / min_f, 0)
            remainder = np.min([max_f % min_f, min_f - (max_f % min_f)])

            if remainder < freq_spacing * (harm / 2):
                if (
                    summary[sorted_summary_idx[n]]["f"]
                    <= summary[sorted_summary_idx[m]]["f"]
                ) and (harm <= 32):
                    is_harm = True
                elif (
                    summary[sorted_summary_idx[n]]["f"]
                    > summary[sorted_summary_idx[m]]["f"]
                ) and (harm <= 8):
                    is_harm = True

                if is_harm:
                    log.debug(f"found harmonic {sorted_summary_idx[m]}")
                    harms_n.append(sorted_summary_idx[m])
                    idx_to_skip.append(sorted_summary_idx[m])
                    # add duplicates if there are any
                    try:
                        harms_n.extend(duplicate_indices[sorted_summary_idx[m]])
                        log.debug(
                            f"{sorted_summary_idx[m]} has duplicates"
                            f" {duplicate_indices[sorted_summary_idx[m]]}"
                        )
                    except KeyError:
                        pass
        if harms_n:
            harms_n.sort()
            harmonics[sorted_summary_idx[n]] = harms_n
            log.debug(f"harmonics of {sorted_summary_idx[n]}:\n{harms_n}")
    log.debug("all harmonics:")
    log.debug(str(harmonics))
    # add identified harmonics to the summary entries
    for key_id, id_list in harmonics.items():
        for harm_id in np.unique(id_list):
            summary[key_id]["harmonics"][harm_id] = summary[harm_id]
    # delete the harmonics' entries from summary
    for idxd in np.unique(idx_to_skip):
        del summary[idxd]
    return summary


def update_database(obsid, summary):
    conn = db_utils.connect()
    no_cands = 0
    no_harm = 0
    no_rfi = 0
    for s in summary:
        no_cands += len(summary[s])
        for idx in summary[s]:
            if summary[s][idx]["rfi"]:
                no_rfi += 1
            no_harm += len(summary[s][idx]["harmonics"])
            no_cands += len(summary[s][idx]["harmonics"])
    payload = {
        "num_total_candidates": no_cands,
        "num_rfi_candidates": no_rfi,
        "num_harmonics": no_harm,
    }
    db_api.update_observation(obsid, payload)


def plot_group(
    group, figsize=(10, 10), plot_title="Candidates", filename="candidates.png"
):
    """
    Plots hhat vs DM for each groups.

    Parameters
    ==========
    group: dict of record array
    figsize: tuple
        matplotlib figure size. Default: (10,10)
    plot_title: str
        figure title
    filename: str
        file name of the saved png.

    Returns
    =======
    None
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(plot_title, fontsize=16)
    for i, g in enumerate(group):
        plt.subplot(len(group) // 4 + 1, 4, i + 1)
        plt.scatter(group[g]["dm"], group[g]["sigma"], s=4)
        plt.grid()
        plt.xlabel("DM")
        plt.ylabel("sigma -- %.3f Hz" % (group[g]["f"][np.argmax(group[g]["sigma"])]))
    plt.savefig(filename)
    plt.clf()


def dc_to_f_spacing_factor(dc):
    nphi_set = np.asarray([192, 128, 96, 64, 48, 32, 24, 16, 8])
    dc_set = np.asarray(
        [
            2.0 / 192,
            2.0 / 128,
            2.0 / 96,
            2.0 / 64,
            2.0 / 48,
            2.0 / 32.0,
            2.0 / 24,
            2.0 / 16,
            2.0 / 8,
        ]
    )
    if dc < np.min(dc_set):
        f_spacing_factor = 192 / 8
    else:
        nphi_idx = np.argwhere(dc_set <= dc)[-1]
        f_spacing_factor = nphi_set[nphi_idx] / 8
    return f_spacing_factor
