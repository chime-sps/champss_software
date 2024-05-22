import logging

import numpy as np
from candidate_processor.utilities import candidate_utils
from sps_common import conversion

log = logging.getLogger(__name__)


def process(
    data,
    dm_labels,
    freq_labels,
    f_spacing_factor,
    dc_value,
    dc_idx,
    threshold=5.5,
    min_group_size=5,
    clean=False,
    root_plot_dir="./",
    make_plots=True,
    harm_filter=False,
):
    """
    Perform a clustering analysis on h-hat array (ndm, nfreq) for a given duty cycle.
    Outputs a file containing candidates.

    Paramaters
    ==========
    data: np.ndarray (ndm, nfreq)
        hhat array for a given duty cycle to clean
    dm_labels: np.array
        array of DMs hhat was computed over.
    freq_labels: np.array
        array of spin frequencies hhat was computed over.
    dc_value: float
        Duty cycle of the array.
    dc_idx: int
        Duty cycle index of the array.
    threshold: float
        hhat noise threshold. Default: 5.5
    min_group_size: int
        minimum group size to consider as a cluster. Default: 5
    clean: bool
        perform cleaning on the hhat array? Default: True.
    root_plot_dir: str
        root path to put all the plots
    make_plots: bool
        make diagnostic plots during analysis
    Returns
    =======
    None
    """
    if clean:
        log.info("cleaning hhat")
        data = candidate_utils.clean_hhat(data)
    log.info("Extracting bright hhat")
    if not make_plots:
        filename = None
    else:
        filename = root_plot_dir + f"hhat_{dc_idx}.png"
    good_idx = candidate_utils.extract_bright_hhat(
        data,
        dm_labels,
        freq_labels,
        dc_value,
        threshold=threshold,
        filename=filename,
    )
    good_hhat = data[good_idx[0], good_idx[1]]
    log.info("performing clustering analysis:")
    orig_freq_diff = np.diff(freq_labels)[0]
    orig_dm_diff = np.diff(dm_labels)[0]
    new_freq_diff = orig_dm_diff / f_spacing_factor
    new_dm_diff = orig_dm_diff
    scaled_freqs = candidate_utils.scale_data(
        freq_labels[good_idx[1]], current_diff=orig_freq_diff, diff=new_freq_diff
    )
    scaled_dms = candidate_utils.scale_data(
        dm_labels[good_idx[0]], current_diff=orig_dm_diff, diff=new_dm_diff
    )
    data_cluster = np.dstack(
        (scaled_freqs, scaled_dms, np.repeat(dc_value, len(scaled_freqs)), good_hhat)
    )[0]
    if not make_plots:
        filename = None
    else:
        filename = root_plot_dir + f"clusters_{dc_idx}.png"
    groups = candidate_utils.perform_clustering(
        data_cluster,
        dc_value,
        min_group_size=min_group_size,
        filename=filename,
    )
    log.info("rescaling clusters")
    final_group = candidate_utils.rescale_clusters(
        groups,
        new_freq_diff,
        new_dm_diff,
        orig_freq_diff,
        orig_dm_diff,
        freq_labels,
    )
    final_group = candidate_utils.make_rec_array(final_group)
    final_group, summary = candidate_utils.group_summary(final_group)
    if harm_filter:
        summary = candidate_utils.harmonic_filter(
            summary, orig_freq_diff * f_spacing_factor
        )
    if not make_plots:
        filename = None
    else:
        filename = root_plot_dir + f"candidates_{dc_idx}.png"
        log.info("plotting candidates")
        candidate_utils.plot_group(
            final_group,
            figsize=(20, 20),
            plot_title="Candidates --> duty cycle {}".format(
                summary[list(summary.keys())[0]]["dc"] * 100
            ),
            filename=filename,
        )
    log.info("saving candidates")
    np.savez(
        root_plot_dir + f"candidates_dc{dc_idx}.npz",
        group=final_group,
        group_summary=summary,
    )
    return summary, final_group


def process_full_hhat(
    min_group_size,
    threshold,
    filename,
    make_plots=True,
    root_plot_dir="./",
    harm_filter=False,
    obsid=None,
    update_db=False,
):
    """
    Given an hhat array (ndm, nfreq,ndc), perform a clustering analysis for each duty
    cycle.

    Parameters
    ==========
    min_group_size: int
        minimum group size to consider as a cluster.
    threshold: float
        hhat noise threshold.
    filename: str
        hhat hdf5 filename (must be sliced by duty cycle -- "dc")
    make_plots: bool
        Do you want diagnostic plots?
    root_plot_dir: str
        path where to output the plot files.
    harm_filter: bool
        apply harmonic filter to the final candidate list?
    obsid: int or None
        obsid of the pointing
    update_db: bool
        Do you want to update the database with candidates properties?

    Returns
    =======
    summary: dict
        contains a summary of the groups for each duty cycle
    groups: dict
        contains all the groups of clusters for each duty cycle.
    """
    h5f, dm_labels, freq_labels, dc_labels, sliced_by = conversion.read_hhat_hdf5(
        filename
    )
    assert (
        sliced_by == "dc"
    ), "H-hat array must be sliced by duty cycle for this analysis to happen"
    summary = {}
    groups = {}
    for dc_idx in range(11):
        dc_value = dc_labels[dc_idx]
        data = conversion.read_hdf5_dataset(h5f, dataset_key=f"{dc_idx}")
        if freq_labels.ndim == 2:
            fls = freq_labels[:, dc_idx]
        else:
            fls = freq_labels
        f_spacing_factor = candidate_utils.dc_to_f_spacing_factor(dc_value)
        log.info(
            "frequency spacing factor for dc of {} is {}".format(
                dc_value, f_spacing_factor
            )
        )
        s, g = process(
            data,
            dm_labels,
            fls,
            f_spacing_factor,
            dc_value,
            dc_idx,
            threshold=threshold,
            min_group_size=min_group_size,
            clean=False,
            root_plot_dir=root_plot_dir,
            make_plots=make_plots,
            harm_filter=harm_filter,
        )
        summary[dc_labels[dc_idx]] = s
        groups[dc_labels[dc_idx]] = g
    if update_db:
        candidate_utils.update_database(obsid, summary)
    return summary, groups


def process_power_spec(
    min_group_size,
    threshold,
    candidate_files,
    make_plots=True,
    root_plot_dir="./",
    harm_filter=False,
    obsid=None,
    update_db=False,
    save_pregroup=True,
):
    """
    Given an hhat array (ndm, nfreq,ndc), perform a clustering analysis for each duty
    cycle.

    Parameters
    ==========
    min_group_size: int
        minimum group size to consider as a cluster.
    threshold: float
        sigma threshold of candidates.
    candidate_files: list(str)
        names of candidate files to process
    make_plots: bool
        Do you want diagnostic plots?
    root_plot_dir: str
        path where to output the plot files.
    harm_filter: bool
        apply a harmonic filter to the resultant candidate groups
    obsid: int or None
        obsid of the pointing
    update_db: bool
        Do you want to update the database with candidates properties?
    save_pregroup: bool
        Do you want to save all PS candidates before grouping?

    Returns
    =======
    summary: dict
        contains a summary of the groups for each duty cycle
    groups: dict
        contains all the groups of clusters for each duty cycle.
    """
    cand_list = []
    for c in candidate_files:
        try:
            cl, cand_labels, freq_spacing = conversion.read_hdf5_cands(c)
        except:
            log.error(f"Error opening {c}")
            continue
        cl = cl[cl[:, 1].argsort()]
        pop_dup = []
        pointer = 0
        for i in range(len(cl)):
            if i == 0:
                pointer = i
                continue
            if cl[i][1] == cl[i - 1][1] and cl[i][0] == cl[pointer][0]:
                if cl[i][4] > cl[pointer][4]:
                    pop_dup.append(pointer)
                    pointer = i
                elif cl[i][4] < cl[pointer][4]:
                    pop_dup.append(i)
            else:
                pointer = i

        # log.info('Removing {} duplicate detections from {}'.format(len(pop_dup), c))
        cand_list.append(np.delete(cl, pop_dup, axis=0))
    cand_list = np.vstack(cand_list)
    log.info(f"total number of candidates = {len(cand_list)}")
    # save candidates before grouping
    if save_pregroup:
        log.info("saving pre-grouping PS candidates")
        np.save(root_plot_dir + "candidates_pregroup.npy", cand_list)
    summary = {}
    groups = {}
    s, g = process_spec(
        cand_list,
        freq_spacing=freq_spacing,
        threshold=threshold,
        min_group_size=min_group_size,
        clean=True,
        root_plot_dir=root_plot_dir,
        make_plots=make_plots,
        harm_filter=harm_filter,
    )
    summary["0.0"] = s
    groups["0.0"] = g
    if update_db:
        candidate_utils.update_database(obsid, summary)
    return summary, groups


def process_spec(
    data,
    freq_spacing=9.70127682e-04,
    threshold=5.5,
    min_group_size=5,
    clean=False,
    root_plot_dir="./",
    make_plots=True,
    harm_filter=False,
):
    """
    Perform a clustering analysis on h-hat array (ndm, nfreq) for a given duty cycle.
    Outputs a file containing candidates.

    Paramaters
    ==========
    data: np.ndarray (ndm, nfreq)
        hhat array for a given duty cycle to clean
    freq_spacing: float
        frequency spacing between adjacent frequency values in the original search data
    threshold: float
        sigma threshold. Default: 5.5
    min_group_size: int
        minimum group size to consider as a cluster. Default: 5
    clean: bool
        perform cleaning on the hhat array? Default: True.
    root_plot_dir: str
        root path to put all the plots
    make_plots: bool
        make diagnostic plots during analysis?
    harm_filter: bool
        filter harmonically related candidates?

    Returns
    =======
    None
    """
    log.info(f"Threshold: {threshold}")
    if clean:
        log.info("cleaning power spec")
        data = candidate_utils.clean_power_spec(data, threshold)
    log.info("Extracting bright power_spec")
    if not make_plots:
        filename = None
    else:
        filename = root_plot_dir + "power_spec.png"
    dm_labels = np.asarray([d[0] for d in data])
    freq_labels = np.asarray([d[1] for d in data])

    log.info("performing clustering analysis:")
    orig_freq_diff = freq_spacing
    orig_dm_diff = 0.125
    new_freq_diff = orig_dm_diff
    new_dm_diff = orig_dm_diff
    scaled_freqs = candidate_utils.scale_data(
        freq_labels, current_diff=orig_freq_diff, diff=new_freq_diff
    )
    scaled_dms = candidate_utils.scale_data(
        dm_labels, current_diff=orig_dm_diff, diff=new_dm_diff
    )

    data_cluster = np.dstack(
        (scaled_freqs, scaled_dms, np.zeros(shape=len(scaled_freqs)), data[:, -1])
    )[0]
    if not make_plots:
        filename = None
    else:
        filename = root_plot_dir + "clusters.png"
    groups = candidate_utils.perform_clustering(
        data_cluster,
        0.0,
        min_group_size=min_group_size,
        filename=filename,
    )
    log.info("rescaling clusters")
    final_group = candidate_utils.rescale_power_spec_clusters(
        groups,
        new_freq_diff,
        new_dm_diff,
        orig_freq_diff,
        orig_dm_diff,
    )
    final_group = candidate_utils.make_rec_array(final_group)
    final_group, summary = candidate_utils.group_summary(final_group)
    log.info("filtering the harmonics")
    if harm_filter:
        summary = candidate_utils.harmonic_filter(summary, freq_spacing)
    if not make_plots:
        filename = None
    else:
        filename = root_plot_dir + "candidates.png"
        log.info("plotting candidates")
        candidate_utils.plot_group(
            final_group,
            figsize=(20, 20),
            plot_title="Candidates --> duty cycle {}".format(
                summary[list(summary.keys())[0]]["dc"] * 100
            ),
            filename=filename,
        )
    log.info("saving candidates")
    np.savez(
        root_plot_dir + "candidates.npz",
        group=final_group,
        group_summary=summary,
    )
    return summary, final_group
