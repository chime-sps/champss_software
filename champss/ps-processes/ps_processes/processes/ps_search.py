"""Class for searching in power spectra."""

import logging
import time
from functools import partial
from multiprocessing import Pool, shared_memory
import yaml
import datetime

import numpy as np
import pandas as pd
from attr import ib as attribute
from attr import s as attrs
from attr import converters
from attr.validators import instance_of
from ps_processes.processes import ps_inject
from ps_processes.processes.clustering import Clusterer
from rfi_mitigation.cleaners.periodic import StaticPeriodicFilter
from sps_common.constants import MIN_SEARCH_FREQ
from sps_common.interfaces import PowerSpectraDetectionClusters, SearchAlgorithm
from sps_common.interfaces.utilities import (
    harmonic_sum,
    powersum_at_sigma,
    sigma_sum_powers,
    get_arc_for_beam,
    angular_separation,
)
from sps_common.constants import TSAMP
from sps_databases import db_api

log = logging.getLogger(__name__)


@attrs(slots=True)
class PowerSpectraSearch:
    """
    Class to do incoherent summing of power spectra to search for pulsars.

    Parameters:
    =======
    num_harm: int
        The number of harmonics to search to. Must be a power of 2 up to 32. Default = 32

    sigma_min: float
        The minimum sigma to determine a candidate. Default = 5.0

    padded_length: int
        The padded length of the time series (default is 2**20 = 1048576), but this
        will depend on declination. The power spectra will have half the size.

    precompute_harms: bool
        Whether to precompute the harmonic bins prior to the search process. Default = True

    num_threads: int
        The number of threads to run the parallel processing of the FFT process. Default = 8

    use_nsum_per_bin: bool
        Use the RFI mask when calculation nsum per frequency bin

    update_db: bool
        Whether to update the observation database with search metrics. Default = True

    cluster_scale_factor: float
        Realtive scaling of DM and freq dimensions during clustering
        Values > 1 mean cluster more in DM space and less in freq space

    dbscan_eps: float
        from sklearn DBSCAN docs:
        The maximum distance between two samples for one to be considered as in the neighborhood of
        the other. This is not a maximum bound on the distances of points within a cluster. This
        is the most important DBSCAN parameter to choose appropriately for your data set and distance
        function.
        This is ignored if clustering_method is set to HDBSCAN

    dbscan_min_samples: int
        from sklearn DBSCAN docs:
        The number of samples (or total weight) in a neighborhood for a point to be considered
        as a core point. This includes the point itself. If min_samples is set to a higher
        value, DBSCAN will find denser clusters, whereas if it is set to a lower value,
        the found clusters will be more sparse.
        This has the same function in HDBSCAN

    clustering_max_ndetect: int
        Maximum number of detections to cluster. If the number of detections exceeds this value the threshold sigma will be raised until it doesn't.

    group_duplicate_freqs: bool
        Whether to first group duplicate frequencies during clustering, and cluster all of those points based on the highest-sigma detection
        This speeds up the calculation of the harmonic metric

    clustering_metric_method: str
        Method used to calcualte the harmonic metric
        Options:
            "rhp_norm_by_min_nharm": 1 - (the intersection of the two raw harmonic power bins, divided by the max nharm)
                                     (aka how much the bins intersect / max possible they could intersect)
            "rhp_overlap": 1 - (the intersection of the two raw harmonic power bins / the max nharm used in the search)
                           (this would downweight nharm=1 matches)

    clustering_metric_combination: str
        Method used to combine the DM-freq distance and harmonic-relation metrics
        Options:
            "multiply": multiply the two together; the harmonic metric modifies the distance metric
            "replace": where the harmonic metric found some relation (aka it is not 1) replace the distance metric
                       value with the harmonic metric value, scaled to be within dbscan_eps

    clustering_method: str
        Clustering method to use: DBSCAN or HDBSCAN

    clustering_min_freq: float
        During clustering don't calculate the harmonic metric for any detection below this freq

    clustering_ignore_nharm1: bool
        During clustering don't calculate the harmonic metric for any detection with nharm=1

    filter_by_nharm: bool
        If True, for each cluster, only keep detections where the nharm matches that of the highest-sigma detection.

    cluster_dm_cut: float
        Filter all clusters that are equal or below this DM threshold

    known_source_threshold: float
        Filter all known sources that were ever above this threshold in an observation of this pointing

    use_stack_threshold: str
        Whether to use the stack sigma for the known source threshold. Otherwise uses sigma
        from single observations.
    """

    cluster_config = attribute(validator=instance_of(dict))
    arc_filter_config = attribute(validator=instance_of(dict))
    num_harm = attribute(validator=instance_of(int), default=32)
    sigma_min = attribute(validator=instance_of(float), default=5.0)
    padded_length = attribute(validator=instance_of(int), default=1048576)
    precompute_harms = attribute(validator=instance_of(bool), default=True)
    num_threads = attribute(validator=instance_of(int), default=8)
    cluster_scale_factor: float = attribute(default=10)
    use_nsum_per_bin: bool = attribute(default=False)
    mp_chunk_size: bool = attribute(default=10)
    skip_first_n_bins: int = attribute(default=2)
    injection_overlap_threshold: bool = attribute(default=0.5)
    injection_dm_threshold: int = attribute(default=10.0)
    known_source_threshold: float = attribute(
        default=np.inf,
        validator=instance_of(float),
        converter=converters.optional(float),
    )
    filter_birdies: bool = attribute(default=False)
    mean_bin_sigma_threshold: float = attribute(default=0)
    only_use_stack_threshold = attribute(validator=instance_of(bool), default=False)
    full_harm_bins = attribute(init=False)
    update_db = attribute(default=True, validator=instance_of(bool))
    max_search_frequency: float = attribute(default=np.inf)

    def __attrs_post_init__(self):
        """Precompute bins used in harmonic summing."""
        if self.precompute_harms:
            self.full_harm_bins = np.vstack(
                (
                    np.arange(0, self.padded_length // 2),
                    harmonic_sum(self.num_harm, np.zeros(self.padded_length // 2))[1],
                )
            ).astype(np.int32)

    @num_harm.validator
    def _validate_num_harm(self, attribute, value):
        assert value in [
            1,
            2,
            4,
            8,
            16,
            32,
        ], "The number of harmonics must be a power of 2 up to 32"

    def search(
        self,
        pspec,
        injection_path,
        injection_indices,
        only_injections,
        scale_injections=False,
    ):
        """
        Run the search.

        Runs a harmonic summing of the power spectrum over a number of harmonics and
        search for candidates.

        Parameters
        ==========
        pspec: PowerSpectra
            PowerSpectra object with the properties of the power spectra to search

        inject: bool
            whether or not to inject an artificial pulse into the power spectrum

        injection_path: str
            Path to injection file or string describing default injection type.
            If 'random', then an integer may be sent in following the string to
            clarify the number of random profiles to inject. Ex: random4

        injection_indices: list
            Indices of injection file entries that are injected.

        only_injections: bool
            Whether non-injections are filtered out. Default: False

        scale_injection: bool
            Whether to scale the injection so that the detected sigma should be
            the same as the input sigma. Default: False

        Returns
        =======
        ps_detections: PowerSpectraDetectionClusters
            PowerSpectraDetectionClusters object with the properties of all the
            detections clusters from the pointing.
        """
        ps_length = ((len(pspec.freq_labels)) // self.num_harm) * self.num_harm
        # compute harmonic bins based on power spectra properties
        if not self.precompute_harms:
            self.full_harm_bins = np.vstack(
                (
                    np.arange(0, ps_length),
                    harmonic_sum(self.num_harm, np.zeros(ps_length))[1],
                )
            ).astype(np.int32)

        if injection_path is not None:
            injection_dicts = []
            if "random" in injection_path:
                if injection_path == "random":
                    n_injections = 1
                else:
                    n_injections = int(injection_path[-1])

                for i in range(n_injections):
                    injection_dict = ps_inject.main(
                        pspec,
                        self.full_harm_bins,
                        scale_injections=scale_injections,
                    )
                    injection_dicts.append(injection_dict)
            else:
                try:
                    with open(injection_path) as file:
                        injection_list = yaml.safe_load(file)
                    injection_df = pd.DataFrame(injection_list)
                except:
                    injection_df = pd.read_pickle(injection_path)
                # injection_df are the initial injection parameters
                # injection_dicts are final injection parameters
                # Some entries from injection_list may create multiple injection or none
                if len(injection_indices) == 0:
                    injection_indices = np.arange(len(injection_df))
                for injection_index in injection_indices:
                    log.info("Injecting at:")
                    log.info(f"DM: {injection_df.iloc[injection_index]['DM']}")
                    log.info(
                        f"frequency: {injection_df.iloc[injection_index]['frequency']}"
                    )
                    injection_dict = injection_df.iloc[injection_index].to_dict()

                    injection_dict = ps_inject.main(
                        pspec,
                        self.full_harm_bins,
                        injection_dict,
                        scale_injections=scale_injections,
                    )
                    injection_dicts.append(injection_dict)
            for injection_index, injection_dict in enumerate(injection_dicts):
                injection_dict["injection_index"] = injection_index
        else:
            injection_dicts = []
            log.info("No artificial pulse injected.")
        search_start = time.time()
        filtered_sources = []
        if self.known_source_threshold is not np.inf:
            log.info(
                f"Will remove known pulsars based on a threshold of {self.known_source_threshold} sigma."
            )
            pointing_id = db_api.get_observation(pspec.obs_id[0]).pointing_id
            current_pointing = db_api.get_pointing(pointing_id)
            if self.only_use_stack_threshold:
                previous_detections = current_pointing.strongest_pulsar_detections_stack
            else:
                # This will merge the two fields with the stack taking precedece on overlap
                previous_detections = (
                    current_pointing.strongest_pulsar_detections
                    | current_pointing.strongest_pulsar_detections_stack
                )

            filtered_psr_names = [
                pulsar
                for pulsar in previous_detections
                if previous_detections[pulsar]["sigma"] > self.known_source_threshold
            ]

            # Filter based on arc
            if self.arc_filter_config["filter_if_kst_active"]:
                # Get psr list from config
                all_arc_psrs = self.arc_filter_config["filtered_pulsars"]
                arc = get_arc_for_beam(
                    current_pointing.beam_row,
                    pspec.datetimes[-1]
                    + datetime.timedelta(seconds=current_pointing.length * TSAMP / 2),
                    delta_x=90,
                    samples=401,
                )
                nearby_arc_psrs = []
                for psr in all_arc_psrs:
                    config_entry = all_arc_psrs[psr]
                    db_entry = db_api.get_known_source_by_names(psr)[0]
                    used_dist = config_entry.get(
                        "arc_length", self.arc_filter_config["default_arc_length"]
                    )
                    if np.abs(pspec.ra - db_entry.pos_ra_deg) < used_dist:
                        nearby_arc_psrs.append(psr)
                    for arc_ra, arc_dec in arc:
                        arc_dist = angular_separation(
                            arc_ra,
                            arc_dec,
                            db_entry.pos_ra_deg,
                            db_entry.pos_dec_deg,
                        )[1]
                        if arc_dist < self.arc_filter_config["arc_search_dist"]:
                            filtered_psr_names.append(psr)
                            break

            filtered_sources = db_api.get_known_source_by_names(filtered_psr_names)

            if len(filtered_sources):
                static_filter = StaticPeriodicFilter.from_ks_list(filtered_sources)
                bad_freq_indices = static_filter.apply_static_mask(pspec.freq_labels, 0)

                dummy_spec = np.zeros(len(pspec.freq_labels))
                dummy_spec[bad_freq_indices] = 1
                dummy_harmonics = dummy_spec[self.full_harm_bins]
                log.info(
                    f"Filter {len(bad_freq_indices)} indices because of pulsars"
                    f" {filtered_psr_names}"
                )
                self.full_harm_bins[dummy_harmonics != 0] = 0

        if self.filter_birdies:
            # This allows filtering birdies using an updated birdie list
            # if some birdies were not filtered during ps creation
            static_filter = StaticPeriodicFilter()
            bad_freq_indices = static_filter.apply_static_mask(pspec.freq_labels, 0)

            dummy_spec = np.zeros(self.padded_length // 2)
            dummy_spec[bad_freq_indices] = 1
            dummy_harmonics = dummy_spec[self.full_harm_bins]
            log.info(f"Filter {len(bad_freq_indices)} indices based on birdie list.")
            self.full_harm_bins[dummy_harmonics != 0] = 0

        if self.mean_bin_sigma_threshold:
            # Filter out bins where mean sigma across all DM trials would be above some threshold
            mean_spec = pspec.power_spectra.mean(0)
            mean_sigma = sigma_sum_powers(mean_spec, pspec.get_bin_weights())
            bad_freq_indices = np.arange(pspec.power_spectra.shape[1])[
                mean_sigma >= self.mean_bin_sigma_threshold
            ]

            dummy_spec = np.zeros(self.padded_length // 2)
            dummy_spec[bad_freq_indices] = 1
            dummy_harmonics = dummy_spec[self.full_harm_bins]
            log.info(
                f"Filter {len(bad_freq_indices)} indices based on mean bin sigma above"
                f" {self.mean_bin_sigma_threshold}."
            )
            self.full_harm_bins[dummy_harmonics != 0] = 0

        # Calculate the used number of each days in each pixel for each harmonic sum
        # From that calculate the power_cutoff.
        # Calculating for each DM trial would be slower.
        # The harmonics are also currently hard-coded in the search routine currently
        all_harmonic_vals = np.array([1, 2, 4, 8, 16, 32])
        if self.use_nsum_per_bin:
            power_cutoff_per_harmonic = np.zeros(
                (6, self.padded_length // 2), dtype=float
            )
            nsum_per_harmonic = np.zeros((6, self.padded_length // 2), dtype=int)
            num_days_per_bin = pspec.get_bin_weights()
            # For masked bins make sure that 0th bins is 0
            num_days_per_bin[0] = 0
            if len(filtered_sources):
                num_days_per_bin[bad_freq_indices] = 0
            for idx_harm, harm in enumerate(all_harmonic_vals):
                nsum_harm_bins = self.full_harm_bins[:harm]
                nsum_current_harmonic = num_days_per_bin[nsum_harm_bins].sum(0)
                nsum_per_harmonic[idx_harm, :] = nsum_current_harmonic
                power_cutoff_per_harmonic[idx_harm, :] = powersum_at_sigma(
                    self.sigma_min, nsum_current_harmonic
                )
                power_cutoff_per_harmonic[idx_harm, nsum_current_harmonic == 0] = np.inf
        else:
            nsum_per_harmonic = all_harmonic_vals * pspec.num_days
            power_cutoff_per_harmonic = powersum_at_sigma(
                self.sigma_min, nsum_per_harmonic
            )

        pool = Pool(self.num_threads)
        detection_dtype = [
            ("freq", float),
            ("dm", float),
            ("nharm", int),
            (
                "harm_idx",
                (
                    int,
                    32,
                ),
            ),
            (
                "harm_pow",
                (
                    float,
                    32,
                ),
            ),
            ("sigma", float),
            ("injection", int),
        ]
        # multiprocessing pool to run the search as a parallel process.
        log.info(
            f"Start searching {len(pspec.dms)} dm trials using"
            f" {self.num_harm} harmonics."
        )
        # Move spectra to shared memory if they are not already.
        # Alternatively the search_candidates function could be rewritten
        # to work with shared and non-shared memory
        pspec.move_to_shared_memory()
        if not self.mp_chunk_size:
            self.mp_chunk_size = len(pspec.dms) // self.num_threads + 1
        log.info(
            f"Number of dm trials in single multi processing job: {self.mp_chunk_size}"
        )

        dm_split = [
            pspec.dms[i : i + self.mp_chunk_size]
            for i in range(0, len(pspec.dms), self.mp_chunk_size)
        ]
        full_indices = np.arange(len(pspec.dms))
        dm_indices = [
            full_indices[i : i + self.mp_chunk_size]
            for i in range(0, len(pspec.dms), self.mp_chunk_size)
        ]
        detection_list = pool.starmap(
            partial(
                self.search_candidates,
                pspec.power_spectra_shared.name,
                pspec.power_spectra.shape,
                pspec.power_spectra.dtype,
                self.num_harm,
                self.full_harm_bins,
                pspec.freq_labels,
                nsum_per_harmonic,
                power_cutoff_per_harmonic,
                injection_dicts,
                self.max_search_frequency,
                self.skip_first_n_bins,
                self.injection_overlap_threshold,
                self.injection_dm_threshold,
            ),
            zip(dm_indices, dm_split),
        )
        pool.close()
        pool.join()

        search_end = time.time()
        log.debug(f"Took {search_end - search_start} seconds to run search")

        # convert the full detection list into a numpy structured array
        detections = np.array(
            [j for sub in detection_list for j in sub], dtype=detection_dtype
        )
        log.info(f"Total number of detections={len(detections)}")
        # if len(detections) == 0:
        #     log.warning("No detections made. Further processing will not be completed.")
        #     return None
        log.info("Clustering the detections")
        clusterer = Clusterer(
            **self.cluster_config,
            sigma_detection_threshold=self.sigma_min,
            num_threads=self.num_threads,
        )
        cluster_dm_spacing = pspec.dms[1] - pspec.dms[0]
        cluster_df_spacing = pspec.freq_labels[1]
        (
            clusters,
            summary,
            clustering_sigma_min,
            used_detections_len,
        ) = clusterer.make_clusters(
            detections,
            cluster_dm_spacing,
            cluster_df_spacing,
            plot_fname="",
            only_injections=only_injections,
        )
        # if len(clusters) == 0:
        #     log.warning("No clusters found. Further processing will not be completed.")
        #     return None

        log.info(f"Total number of candidate clusters={len(clusters)}")
        # Only update db for single day obs, proper stack logging tba
        if len(pspec.obs_id) == 1:
            if self.update_db:
                payload = {
                    "num_detections": len(detections),
                    "num_detections_used": used_detections_len,
                    "num_clusters": len(clusters),
                    "detection_threshold": clustering_sigma_min,
                }

                db_api.update_observation(pspec.obs_id[0], payload)
        for injection_index, injection_dict in enumerate(injection_dicts):
            # remove unneeded dict entries
            injection_dict.pop("bins")
            injection_dict.pop("dms")
        return (
            PowerSpectraDetectionClusters(
                clusters=clusters,
                summary=summary,
                ra=pspec.ra,
                dec=pspec.dec,
                detection_statistic=SearchAlgorithm(1),
                threshold=clustering_sigma_min,
                freq_spacing=cluster_df_spacing,
                obs_id=pspec.obs_id,
                datetimes=pspec.datetimes,
                injection_dicts=injection_dicts,
            ),
            detections,
        )

    @staticmethod
    def search_candidates(
        shared_spectra_name,
        shared_spectra_shape,
        shared_spectra_type,
        num_harm,
        full_harm_bins,
        freq_labels,
        nsum_per_harmonic,
        power_cutoff_per_harmonic,
        injection_dicts,
        cutoff_frequency,
        skip_n_bins,
        injection_overlap_threshold,
        injection_dm_threshold,
        dm_indices,
        dms,
    ):
        """
        Search in a single dm trial.

        The static method to run the FFT to search the power spectra for detections in a
        parallel loop.A static function that does not depend on any other function in
        the object is required to run a multiprocessing task. This is due to
        multiprocessing will pickle the function to spread out the jobs to different
        cores, and a function that relies on other functions in the class object will
        not be pickle-able.

        Parameters
        ----------
        shared_spectra_name: str
            The name of the shared mempry object where the shared power spectra are saved.

        shared_spectra_shape: np.ndarray.shape
            Shape of the shared spectra.

        shared_spectra_type: np.dtype
            Type of the array values in the shared spectra.

        num_harm: int
            The number of harmonics to search.

        full_harm_bins: np.ndarray
            The full array to determine the frequency bins to add together in harmonic summing.

        freq_labels: np.ndarray
            The frequency labels of the power spectrum.

        nsum_per_harmonic: np.ndarray
            The number of summed elements for each harmonic sum.
            Is 2D if self.use_nsum_per_bin otherwise 1D.
            In the 2D case nsum is given for all frequency bins.

        power_cutoff_per_harmonic: np.ndarray
            The power cutoff for each harmonic sum. If the summed power is above this value
            it will be considered a candidate.
            Is 2D if self.use_nsum_per_bin otherwise 1D.
            In the 2D case the power cutoff is given for all frequency bins.

        injection_dicts: list
            Dictionaries containing the injection specifics, Fields "bins" and "dms" are
            important here.

        dm_tuple: (int, float)
            The DM of the power spectrum to search and the index of the dm trial.

        Returns
        -------
        detection_list: List[tuple]
            A list of tuples containing the properties of the individual detections
            made in the search process.
        """
        # log.debug(f"Working on DM={dm} with {num_harm} harmonics")
        shared_spectra = shared_memory.SharedMemory(name=shared_spectra_name)
        power_spectra = np.ndarray(
            shared_spectra_shape, dtype=shared_spectra_type, buffer=shared_spectra.buf
        )
        detection_list = []
        power_spectra = power_spectra[:, : len(full_harm_bins[0])]
        freq_labels = freq_labels[: len(full_harm_bins[0])]
        allowed_harmonics = [1, 2, 4, 8, 16, 32]
        for dm_index, dm in zip(dm_indices, dms):
            power_spectrum = power_spectra[dm_index, :]
            for idx_harm, harm in enumerate(allowed_harmonics):
                harm_start = time.time()
                if harm > num_harm:
                    continue
                log.debug(f"Working on the harmonic={harm} sum")
                harm_bins = full_harm_bins[:harm]

                if idx_harm == 0:
                    harm_sum_powers = power_spectrum[harm_bins].sum(0)
                else:
                    last_harm = allowed_harmonics[idx_harm - 1]
                    np.add(
                        harm_sum_powers,
                        power_spectrum[harm_bins[last_harm:harm, :]].sum(0),
                        out=harm_sum_powers,
                    )

                power_cutoff = power_cutoff_per_harmonic[idx_harm]
                used_nsum = nsum_per_harmonic[idx_harm]
                detection_idx = np.where(harm_sum_powers > power_cutoff)[0]
                if not len(detection_idx):
                    continue
                last_detection_freq = None
                last_detection_sigma = None
                if type(used_nsum) == np.ndarray:
                    used_nsum_detec = used_nsum[detection_idx]
                else:
                    used_nsum_detec = used_nsum
                sigmas = sigma_sum_powers(
                    harm_sum_powers[detection_idx], used_nsum_detec
                )
                for idx_count, idx in enumerate(detection_idx):
                    replace_last = False
                    detection_freq = freq_labels[idx] / harm
                    # skipping candidates with very short and very high frequencies

                    if detection_freq <= skip_n_bins * freq_labels[1]:
                        continue
                    if detection_freq > cutoff_frequency:
                        break

                    if sigmas is None:
                        if type(used_nsum) == np.ndarray:
                            used_nsum_detec_loop = used_nsum[idx]
                        else:
                            used_nsum_detec_loop = used_nsum
                        sigma = sigma_sum_powers(
                            harm_sum_powers[idx], used_nsum_detec_loop
                        )
                    else:
                        sigma = sigmas[idx_count]
                        if np.isnan(sigma):
                            if type(used_nsum) == np.ndarray:
                                used_nsum_detec_loop = used_nsum[idx]
                            else:
                                used_nsum_detec_loop = used_nsum
                            # Array input should work properly
                            log.error("Sigma calculation for array produced nan")
                            sigma = sigma_sum_powers(
                                harm_sum_powers[idx], used_nsum_detec_loop
                            )
                    if (
                        last_detection_freq
                        and np.abs(detection_freq - last_detection_freq)
                        < MIN_SEARCH_FREQ * 1.1
                    ):
                        if sigma < last_detection_sigma:
                            continue
                        else:
                            replace_last = True

                    sorted_harm_bins = sorted(harm_bins[:harm, idx].astype(int))
                    injected_index = -1

                    for list_index, injection_dict in enumerate(injection_dicts):
                        injected_bins = injection_dict["bins"]
                        injected_dms = injection_dict["dms"]
                        if dm_index in injected_dms:
                            injection_overlap = np.intersect1d(
                                sorted_harm_bins, injected_bins
                            )
                            if (
                                injection_overlap.size / len(sorted_harm_bins)
                                > injection_overlap_threshold
                                and np.abs(injection_dict["DM"] - dm)
                                < injection_dm_threshold
                            ):
                                injected_index = list_index

                    if replace_last:
                        detection_list[-1] = (
                            detection_freq,
                            dm,
                            harm,
                            tuple(
                                np.pad(
                                    sorted_harm_bins, (0, 32 - len(sorted_harm_bins))
                                )
                            ),
                            tuple(
                                np.pad(
                                    power_spectrum[sorted_harm_bins],
                                    (0, 32 - len(sorted_harm_bins)),
                                )
                            ),
                            sigma,
                            injected_index,
                        )
                    else:
                        detection_list.append(
                            (
                                detection_freq,
                                dm,
                                harm,
                                tuple(
                                    np.pad(
                                        sorted_harm_bins,
                                        (0, 32 - len(sorted_harm_bins)),
                                    )
                                ),
                                tuple(
                                    np.pad(
                                        power_spectrum[sorted_harm_bins],
                                        (0, 32 - len(sorted_harm_bins)),
                                    )
                                ),
                                sigma,
                                injected_index,
                            )
                        )
                    last_detection_freq = detection_freq
                    last_detection_sigma = sigma
                harm_end = time.time()
                log.debug(
                    f"Took {harm_end - harm_start} seconds to do harmonic={harm} sum"
                )
        return detection_list

    def summarise(self, clusters, cluster_harm_idx):
        """
        Produce a summary of the clusters.

        Parameters
        ----------
        clusters: dict(np.array):
            A dictionary of numpy structure array of each cluster. Each cluster array consists
            of dtype=[("freq", float), ("dm", float), ("nharm", int),
            ("sigma", float),] for each point in cluster.

        cluster_harm_idx: dict(np.array)
            A dictionary of numpy array showing the indices of the harmonics for the most
            significant detection in each cluster.

        Returns
        -------
        summary: dict(dict):
            A dictionary of dictionary showing the summary of each cluster. Each summary dict
            consists of the following properties : ['freq', 'dm', 'nharm', 'sigma',
            'harm_idx', 'min_dm', 'max_dm', 'harmonics', 'rfi'] for the
            most significant detection, the dm range of all the detections, and whether
            a cluster is considered RFI (dm < 1.0).
        """
        summary = {}
        for c in clusters:
            cluster_max = clusters[c][np.argmax(clusters[c]["sigma"])]
            summary[c] = {}
            for i, prop in enumerate(clusters[c].dtype.names):
                summary[c][prop] = cluster_max[i]
            summary[c]["harm_idx"] = cluster_harm_idx[c]
            summary[c]["min_dm"] = np.min(clusters[c]["dm"])
            summary[c]["max_dm"] = np.max(clusters[c]["dm"])
            summary[c]["harmonics"] = {}
            # mark candidate group with DM < 1 as RFI
            if summary[c]["dm"] < 0.99:
                summary[c]["rfi"] = True
            else:
                summary[c]["rfi"] = False
        return summary
