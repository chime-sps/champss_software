@attrs
class Features_FFA:
    """
    Class contains a list of FeatureGenerator objects and a corresponding list of
    DataGetter objects With make these will be used on a HarmonicallyRelatedClusters
    instance to generate features and output a SinglePointingCandidate. Also computes
    candidate arrays.

    Example useage:
    # initialise from config, a feature-configuration dictionary
    features = Features.from_config(config)
    # run on psdc, a PowerSpectraDetectionClusters instance
    # to generate spc, a SinglePointingCandidate
    spc = features.make(hrc)

    Attributes:
    -----------
    generators (List(FeatureGenerator)): Initialized FeatureGenerator_FFA objects
    datagetters (List(DataGetter)): Initialized DataGetter_FFA objects
    """

    generators = attrib(
        validator=deep_iterable(
            member_validator=instance_of(FeatureGenerator_FFA),
            iterable_validator=instance_of(list),
        )
    )
    datagetters = attrib(
        validator=deep_iterable(
            member_validator=instance_of(DataGetter_FFA),
            iterable_validator=instance_of(list),
        )
    )
    _dtype_structure: Dict = attrib()
    # Some attribute describing the array creation
    pool_bins: int = attrib(default=0)
    period_factors: int = attrib(default=1)
    max_nharm: int = attrib(default=32)
    array_ranges: dict = attrib(
        default={
            "dm_in_raw": 20,
            "dm_in_dm_freq": 150,
            "freq_in_dm_freq": 40,
            "dm_in_dm_1d": 500,
        }
    )
    write_detections_to_candidates: bool = attrib(
        validator=instance_of(bool), default=True
    )
    num_threads = attrib(validator=instance_of(int), default=4)
    allowed_harmonics: np.ndarray = attrib(default=np.asarray([1, 2, 4, 8, 16, 32]))

    def __attrs_post_init__(self):
        """Truncate allowed harmonics."""
        self.allowed_harmonics = self.allowed_harmonics[
            self.allowed_harmonics <= self.max_nharm
        ]

    # links the index in datagetters/generators to the dtype name
    @_dtype_structure.validator
    def check_dtype_structure(self, attribute, value):
        """
        This serves as a check that:
         a) datagetters[i] and generators[i] are compatible
            e.g. a PropertyDataGetter and a PropertyGenerator
         b) the combination of the two corresponds to the name which is in
            _dtype_structure
        """
        for key in list(value.keys()):
            tst_name = make_combined_name(self.datagetters[key], self.generators[key])
            if tst_name != value[key]:
                raise ValueError(
                    f"Made name {tst_name} from DataGetter and FeatureGenerator"
                    f"Name in _dtype_structure is {value[key]}"
                )

    def print_dtype_structure(self):
        for datacode_entry in self._dtype_structure:
            print(datacode_entry[0], ":")
            for individual_feature_entry in datacode_entry[1]:
                print("\t", individual_feature_entry)

    @classmethod
    def from_config(cls, config: Dict, config_arrays: Dict, num_threads: int):
        gens = []
        datgets = []
        dt_struct = {}
        fit_flag_names = [
            "rel_err",
            "diff_from_detection",
            "ndof",
            "rms_err",
        ]
        stat_flag_names = [
            "weighted",
            "about_peak",
        ]
        datacode_i = 0
        i = 0  # tracking index in generators/datagetters
        for datacode, indiv_features in config.items():
            # properties are a special case
            if datacode == "Property":
                for prop in indiv_features:
                    gens.append(PropertyGenerator(property=prop))
                    datgets.append(PropertyDataGetter(datacode=datacode))
                    i += 1

            else:
                for feature_config in indiv_features:
                    try:
                        featureclass = getattr(feat, feature_config["feature"])
                        flag_list = feature_config.get("flags", [])
                        options = feature_config.get("options", {})

                        # Stats:
                        if issubclass(featureclass, Stat):
                            stat_flags = flags_from_list(stat_flag_names, flag_list)
                            datgets.append(
                                StatDataGetter(datacode=datacode, **stat_flags)
                            )
                            gens.append(StatGenerator(statclass=featureclass))
                        # Fits
                        elif issubclass(featureclass, Fit):
                            fit_flags = flags_from_list(fit_flag_names, flag_list)
                            datgets.append(FitDataGetter(datacode=datacode))
                            gens.append(
                                FitGenerator(fitclass=featureclass, **fit_flags)
                            )
                    except Exception as ex:
                        log.exception(
                            f'{feature_config["feature"]} could not be initialized due'
                            " to the following"
                        )
                        log.exception(ex)

                    i += 1

            datacode_i += 1

        if len(gens) != len(datgets):
            raise AttributeError(
                "Coding bug! - length of gens and datgets lists must be the same"
            )

        # construct _dtype_structure
        log.info("Features initialized as the following:")
        for i in range(0, len(gens)):
            dt_struct[i] = make_combined_name(datgets[i], gens[i])
            log.info(dt_struct[i])

        return cls(
            generators=gens,
            datagetters=datgets,
            dtype_structure=dt_struct,
            **config_arrays,
            num_threads=num_threads,
        )

    def make_single_pointing_candidate(
        self,
        pspec_meta_data: EasyDict,
        full_harm_bins,
        injection_dicts,
        cluster_dict: EasyDict,
    ) -> SinglePointingCandidate:
        """
        Create candidate from PowerSpectraDetectionCluster.

        Compute/Fit/Get the features and candidate arrays from a
        PowerSpectraDetectionCluster dictionary and return the corresponding
        SinglePointingCandidate.
        """
        shared_spectra = shared_memory.SharedMemory(name=pspec_meta_data.shared_name)
        power_spectra = np.ndarray(
            pspec_meta_data.shape,
            dtype=pspec_meta_data.dtype,
            buffer=shared_spectra.buf,
        )
        cluster = cluster_dict.cluster
        # Calculate candidate arrays
        (
            raw_harmonic_powers_array_dict,
            dm_freq_sigma_dict,
            dm_sigma_1d_dict,
            sigmas_per_harmonic_sum_dict,
        ) = self.create_candidate_arrays(
            cluster, full_harm_bins, power_spectra, pspec_meta_data
        )

        cluster.dm_freq_sigma = dm_freq_sigma_dict

        # get a value and dtype for each feature
        vals = []
        dts = []
        for i in range(len(self.generators)):
            val, dt = self.generators[i].make(self.datagetters[i].get(cluster))
            if (
                type(val) == list
            ):  # if multiple values output from one FeatureGenerator, e.g. fits
                if len(val) == len(self._dtype_structure[i]):
                    vals.extend(val)
                    for j in range(len(dt)):
                        dts.append(tuple([self._dtype_structure[i][j], dt[j]]))
                else:
                    log.warning(
                        "Skipping feature - number of returned values did not match"
                        f" the length of _dtype_structure: {self._dtype_structure[i]}"
                    )
            else:  # normal case of a single-valued feature
                vals.append(val)
                dts.append(tuple([self._dtype_structure[i], dt]))

        if self.write_detections_to_candidates:
            written_detections = rfn.structured_to_unstructured(
                cluster.detections[["dm", "freq", "sigma", "nharm", "injection"]]
            )
        else:
            written_detections = None

        """
        harm_dtype = [("freq", float), ("dm", float), ("nharm", int), ("sigma", float)]
        harmonics_tmp = []
        # Earlier versions excluded the main peak in harmonics_info
        main_cluster_max = hrc.main_cluster[np.argmax(hrc.main_cluster["sigma"])]
        harmonics_tmp.append(main_cluster_max)
        for ii, harm_cluster in hrc.harmonics_clusters.items():
            cluster_max = harm_cluster[np.argmax(harm_cluster["sigma"])]
            harmonics_tmp.append(cluster_max)
        harmonics_info = np.array(harmonics_tmp, dtype=harm_dtype)
        """
        if cluster.injection_index != -1:
            injected = True
            injection_dict = injection_dicts[cluster.injection_index]
        else:
            injected = False
            injection_dict = {}
        spc_init_dict = dict(
            freq=cluster.freq,
            dm=cluster.dm,
            detections=written_detections,
            ndetections=cluster.ndetections,
            max_sig_det=cluster.max_sig_det,
            unique_freqs=cluster.unique_freqs,
            unique_dms=cluster.unique_dms,
            sigma=cluster.sigma,
            injection=injected,
            injection_dict=injection_dict,
            ra=cluster_dict.ra,
            dec=cluster_dict.dec,
            features=np.array([tuple(vals)], dtype=dts),
            detection_statistic=cluster_dict.detection_statistic,
            obs_id=cluster_dict.obs_id,
            datetimes=cluster_dict.get("datetimes", []),
            rfi=None,
            dm_freq_sigma=dm_freq_sigma_dict,
            raw_harmonic_powers_array=raw_harmonic_powers_array_dict,
            harmonics_info=None,  # harmonics_info,
            dm_sigma_1d=dm_sigma_1d_dict,
            sigmas_per_harmonic_sum=sigmas_per_harmonic_sum_dict,
            pspec_freq_resolution=pspec_meta_data.freq_labels[1],
        )
        return SinglePointingCandidate(**spc_init_dict)

    def create_candidate_arrays(
        self,
        cluster: Cluster,
        full_harm_bins: np.ndarray,
        power_spectra: np.ndarray = None,
        pspec_meta_data: EasyDict = EasyDict({}),
    ) -> Tuple[dict, dict, dict, dict]:
        """
        Create arrays for the candidate files.

        Create raw_harmonic_powers_array, dm_freq_sigma and dm_sigma_1d for one group of
        harmonically related clusters.

        Parameters
        ----------
        clusterdict (Cluster): Cluster for which the candidate arrays are created
        full_harm_bins (np.ndarray): Array containing the information about which bins
                                    are used during harmonic summing
        power_spectra (PowerSpectra): The power spectra object from which to derive the arrays
        ps (np.ndarray): The array containing values of the power spectra

        Returns
        -------
        (raw_harmonic_powers_array, dm_freq_sigma and dm_sigma_1d ) (dict, dict, dict):
            A tuple of dicts containing the arrays and labels
        """
        if power_spectra is None:
            return None, None, None, None
        else:
            ps = power_spectra
            cluster_freq = cluster.freq
            best_dm_trial = np.abs(pspec_meta_data.dms - cluster.dm).argmin()
            dm_idx_min, dm_idx_max = get_min_max_index(
                best_dm_trial, self.array_ranges["dm_in_raw"], len(pspec_meta_data.dms)
            )

            # When only a harmonic of the pulsar is found, writing out additional
            # periods may help finding a more pulsar-like raw_harmonic_powers.
            # Cast factor to float to make mypy happy
            used_factors = [1.0]
            for i in range(2, self.period_factors + 1):
                used_factors.extend([float(i), 1 / i])
            # raw_harmonic_powers, freq_labels and freq_bins share the dimensions
            # [dm_trials, harmonic, factor_trials]
            # dm_labels are only a 1-d array and give the dm values for the dm trials
            raw_harmonic_powers_array = np.zeros(
                (dm_idx_max - dm_idx_min, self.max_nharm, len(used_factors))
            )

            dm_labels = pspec_meta_data.dms[np.arange(dm_idx_min, dm_idx_max)]
            freq_labels = np.zeros(
                (dm_idx_max - dm_idx_min, self.max_nharm, len(used_factors))
            )
            freq_bins = np.zeros(
                (dm_idx_max - dm_idx_min, self.max_nharm, len(used_factors))
            )
            freq_bin_weights = pspec_meta_data.freq_bin_weights
            for factor_index, factor in enumerate(used_factors):
                current_freq = cluster_freq * factor

                # If max(allowed_harmonics) * current_freq is larger than the maximum
                # frequency in the power spectrum
                # fewer harmonics are saved
                max_harm_cluster = min(
                    self.max_nharm,
                    np.floor(pspec_meta_data.freq_labels[-1] / current_freq).astype(
                        int
                    ),
                )
                used_harmonics = self.allowed_harmonics[
                    self.allowed_harmonics <= max_harm_cluster
                ]
                if max_harm_cluster not in self.allowed_harmonics:
                    max_harm_cluster = max(used_harmonics)
                nearest_bins = np.abs(
                    current_freq * max_harm_cluster - pspec_meta_data.freq_labels
                ).argmin()
                harm_bins_truncated = full_harm_bins[:max_harm_cluster, :]
                harm_bins_sorted = harm_bins_truncated[
                    np.argsort(harm_bins_truncated[:, -1], axis=0), :
                ]
                # Create bins for raw harmonics
                harm_bins = harm_bins_sorted[:, nearest_bins]
                bin_steps = np.arange(-self.pool_bins, self.pool_bins + 1)
                harm_bins_broadened = (
                    harm_bins[:, np.newaxis] + bin_steps[np.newaxis, :]
                )

                # Here we are stepping through the dm trials used for the raw_harmonic_powers
                for array_dm_index, dm_index in enumerate(
                    range(dm_idx_min, dm_idx_max)
                ):
                    # This sclicing operation finds for each harmonic what shift allows
                    # finding the strongest peak
                    best_bin_positions = harm_bins_broadened[
                        np.arange(harm_bins_broadened.shape[0]),
                        ps[dm_index][harm_bins_broadened].argmax(-1),
                    ]
                    raw_harmonic_powers_array[
                        array_dm_index, : len(best_bin_positions), factor_index
                    ] = ps[dm_index][best_bin_positions]
                    freq_labels[
                        array_dm_index, : len(best_bin_positions), factor_index
                    ] = pspec_meta_data.freq_labels[best_bin_positions]
                    freq_bins[
                        array_dm_index, : len(best_bin_positions), factor_index
                    ] = best_bin_positions

                # Now we are writing out the values out only when factor==1 for the
                # dm-sigma and dm-freq-sigma array
                if factor == 1:
                    dm_freq_sigma_full = np.zeros(
                        (
                            2 * self.array_ranges["dm_in_dm_freq"] + 1,
                            2 * self.array_ranges["freq_in_dm_freq"] + 1,
                            len(used_harmonics),
                        )
                    )
                    f0_idx_min_sigma, f0_idx_max_sigma = get_min_max_index(
                        nearest_bins,
                        self.array_ranges["freq_in_dm_freq"],
                        full_harm_bins.shape[1],
                    )
                    dm_idx_min_sigma, dm_idx_max_sigma = get_min_max_index(
                        best_dm_trial,
                        self.array_ranges["dm_in_dm_freq"],
                        len(pspec_meta_data.dms),
                    )
                    freq_labels_sigma = (
                        pspec_meta_data.freq_labels[
                            np.arange(f0_idx_min_sigma, f0_idx_max_sigma)
                        ]
                        / max_harm_cluster
                    )
                    dm_labels_sigma = pspec_meta_data.dms[
                        np.arange(dm_idx_min_sigma, dm_idx_max_sigma)
                    ]

                    harm_bins_current_pos = harm_bins_sorted[
                        :, f0_idx_min_sigma:f0_idx_max_sigma
                    ]
                    for harm_index, harm in enumerate(used_harmonics):
                        harm_bins_current = harm_bins_current_pos[:harm, :]
                        harm_sum_powers = ps[
                            dm_idx_min_sigma:dm_idx_max_sigma, harm_bins_current
                        ].sum(1)
                        ndays_per_bin = freq_bin_weights[harm_bins_current].sum(0)
                        sigma = sigma_sum_powers(harm_sum_powers, ndays_per_bin)
                        dm_freq_sigma_full[:, :, harm_index] = sigma

                    dm_freq_sigma = np.nanmax(dm_freq_sigma_full, 2)
                    try:
                        dm_freq_nharm = used_harmonics[
                            np.nanargmax(dm_freq_sigma_full, 2)
                        ]
                    except ValueError:
                        # When slice contains only nan
                        dm_freq_nharm = np.full(dm_freq_sigma.shape, np.nan)
                        dm_freq_nharm[~np.isnan(dm_freq_sigma)] = used_harmonics[
                            np.nanargmax(
                                dm_freq_sigma_full[~np.isnan(dm_freq_sigma)], 1
                            )
                        ]

                    dm_freq_sigma_dict = {
                        "sigmas": dm_freq_sigma.astype(np.float16),
                        "dms": dm_labels_sigma,
                        "freqs": freq_labels_sigma,
                        "nharm": np.nan_to_num(dm_freq_nharm).astype(np.int8),
                    }

                    # Now write out the 1d dm series
                    dm_idx_min_1d, dm_idx_max_1d = get_min_max_index(
                        best_dm_trial,
                        self.array_ranges["dm_in_dm_1d"],
                        len(pspec_meta_data.dms),
                    )
                    dm_sigma_full = np.zeros(
                        (
                            min(
                                self.array_ranges["dm_in_dm_1d"] * 2 + 1,
                                len(pspec_meta_data.dms),
                            ),
                            len(used_harmonics),
                        )
                    )
                    dm_labels_1d = pspec_meta_data.dms[
                        np.arange(dm_idx_min_1d, dm_idx_max_1d)
                    ]
                    # I am not entirely sure if summing over the possible harmonics
                    # is the best way
                    # Maybe just creating it for one sum would be more helpful
                    for harm_index, harm in enumerate(used_harmonics):
                        harm_bins_current = harm_bins[:harm]
                        harm_sum_powers = ps[
                            dm_idx_min_1d:dm_idx_max_1d, harm_bins_current
                        ].sum(1)
                        ndays_per_bin = freq_bin_weights[harm_bins_current].sum(0)
                        sigma = sigma_sum_powers(harm_sum_powers, ndays_per_bin)
                        dm_sigma_full[:, harm_index] = sigma

                    dm_sigma_1d = np.nanmax(dm_sigma_full, 1)
                    dm_sigma_1d_dict = {
                        "sigmas": dm_sigma_1d.astype(np.float16),
                        "dms": dm_labels_1d,
                    }

                    # Calculate single sigma values
                    # This could also be done in the dm_sigma_1d loop
                    harmonic_sums = np.full(len(used_harmonics), np.nan)
                    unweighted_nsum = np.full(len(used_harmonics), np.nan)
                    weighted_nsum = np.full(len(used_harmonics), np.nan)
                    for harm_index, harm in enumerate(used_harmonics):
                        harm_bins_current = harm_bins[:harm]
                        harm_sum_powers = ps[best_dm_trial, harm_bins_current].sum(0)
                        harmonic_sums[harm_index] = harm_sum_powers
                        unweighted_nsum[harm_index] = harm * pspec_meta_data.num_days
                        ndays_per_bin = freq_bin_weights[harm_bins_current].sum(0)
                        weighted_nsum[harm_index] = ndays_per_bin

                    sigmas_unweighted = sigma_sum_powers(harmonic_sums, unweighted_nsum)
                    sigmas_weighted = sigma_sum_powers(harmonic_sums, weighted_nsum)
                    weight_fraction = weighted_nsum / unweighted_nsum
                    pad_length = len(self.allowed_harmonics) - len(used_harmonics)

                    sigmas_per_harmonic_sum_dict = {
                        "sigmas_unweighted": np.pad(
                            sigmas_unweighted, (0, pad_length), constant_values=np.nan
                        ).astype(np.float16),
                        "sigmas_weighted": np.pad(
                            sigmas_weighted, (0, pad_length), constant_values=np.nan
                        ).astype(np.float16),
                        "nsum": np.pad(
                            np.nan_to_num(weighted_nsum),
                            (0, pad_length),
                            constant_values=0,
                        ).astype(np.uint16),
                        "weight_fraction": np.pad(
                            weight_fraction, (0, pad_length), constant_values=np.nan
                        ),
                    }

            raw_harmonic_powers_array_dict = {
                "powers": raw_harmonic_powers_array.astype(np.float16),
                "dms": dm_labels,
                "freqs": freq_labels,
                "freq_bins": freq_bins.astype(np.uint32),
            }

            return (
                raw_harmonic_powers_array_dict,
                dm_freq_sigma_dict,
                dm_sigma_1d_dict,
                sigmas_per_harmonic_sum_dict,
            )

    def make_single_pointing_candidate_collection(
        self, psdc: PowerSpectraDetectionClusters, power_spectra
    ) -> SinglePointingCandidateCollection:
        """
        Make a list of `HarmonicallyRelatedClusters`obejcts into a
        SinglePointingCandidateCollection. (Generate features, adds metadata)

        hrc_list: a list of HarmonicallyRelatedClusters instances
        """
        log.info(
            "Making SinglePointingCandidates from a list of"
            f" {len(psdc.clusters)} PowerSpectraDetectionClusters"
        )
        # Prepare array creation

        if power_spectra is not None:
            power_spectra.convert_to_nparray()
            ps = power_spectra.power_spectra
            full_harm_bins = np.vstack(
                (
                    np.arange(0, len(power_spectra.freq_labels)),
                    harmonic_sum(
                        self.allowed_harmonics.max(),
                        np.zeros(len(power_spectra.freq_labels)),
                    )[1],
                )
            ).astype(int)
        else:
            log.warning(
                "PowerSpectra object provided to the harmonic filter is None. Not all"
                " candidate properties will be created."
            )
        """
        spcs = []
        for index, cluster_dict in enumerate(psdc):
            log.info(f"Processing PowerSpectraDetectionCluster {index}")
            spc = self.make_single_pointing_candidate(
                cluster_dict, power_spectra, full_harm_bins
            )
            spcs.append(spc)
        """
        pspec_meta_data = EasyDict(
            {
                "shared_name": power_spectra.power_spectra_shared.name,
                "shape": power_spectra.power_spectra.shape,
                "dtype": power_spectra.power_spectra.dtype,
                "dms": power_spectra.dms,
                "freq_labels": power_spectra.freq_labels,
                "num_days": power_spectra.num_days,
                "freq_bin_weights": power_spectra.get_bin_weights(),
            }
        )
        with Pool(self.num_threads) as pool:
            spcs = pool.map(
                partial(
                    self.make_single_pointing_candidate,
                    pspec_meta_data,
                    full_harm_bins,
                    psdc.injection_dicts,
                ),
                psdc,
            )
            pool.close()
            pool.join()
        spcc_init_dict = dict(candidates=spcs, injections=psdc.injection_dicts)

        return SinglePointingCandidateCollection(**spcc_init_dict)
