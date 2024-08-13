import logging
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, shared_memory
from typing import Dict, List, Tuple

import candidate_processor.utilities.features as feat
import numpy as np
from attr import attrib, attrs
from attr.validators import deep_iterable, in_, instance_of
from candidate_processor.utilities.features import Fit, Mean, Stat
from easydict import EasyDict
from numpy.lib import recfunctions as rfn
from sps_common.interfaces.ps_processes import Cluster, PowerSpectraDetectionClusters
from sps_common.interfaces.single_pointing import (
    SinglePointingCandidate,
    SinglePointingCandidateCollection,
)
from sps_common.interfaces.utilities import harmonic_sum, sigma_sum_powers

log = logging.getLogger(__name__)
NOPOINT = 999999


# Need to be able to test if classes are subclasses of others a fair bit
def make_issubclass_validator(parent):
    def subclass_validator(instance, attribute, value):
        if not issubclass(value, parent):
            raise TypeError(
                f"{attribute.name} must be a subclass of {parent}Received {value}"
            )

    return subclass_validator


###################################################################
######################## DataGetter Classes ########################


@attrs
class DataGetter:
    """
    Parent for classes which get the correct kind of data from the
    HarmonicallyRelatedClusters so it can be supplied to a FeatureGenerator. All have a
    get function. Mainly exists so you can check if something is a DataGetter.

    All DataGetter subclasses have a corresponding FeatureGenerator subclass
    and are written so their get methods provide the correct input parameters
    for the make method of their FeatureGenerator

    Methods:
    --------
    get(hrc:HarmonicallyRelatedClusters): Get data from a HarmonicallyRelatedClusters object
    make_name() : Make a name to describe itself (used to form/check _dtype_structure in Features)
    """

    def get(self, hrc):
        """Get correct kind of data from a HarmonicallyRelatedClusters."""
        raise NotImplementedError("Subclasses must implement this method")

    def make_name(self):
        """Make feature from data output from a DataGetter."""
        raise NotImplementedError("Subclasses must implement this method")


@attrs
class PropertyDataGetter(DataGetter):
    """Get data when want to extract a property of a HarmonicallyRelatedClusters
    instance The get method simply returns the HarmonicallyRelatedClusters.
    """

    datacode: str = attrib()

    @datacode.validator
    def check_property_datacode(self, attribute, value):
        if value != "Property":
            raise ValueError(
                f"Datacode attribute ({attribute.name}={value}) is incorrect"
                f"Should only use PropertyDataGetter if {attribute.name}='Property'"
            )

    def get(self, hrc):
        """Just returns the hrc."""
        return hrc

    def make_name(self):
        """Returns ''."""
        return {}


@attrs
class StatDataGetter(DataGetter):
    """
    Class to store what kind of data to get for a statistics calculations.

    Example of use on hrc, a HarmonicallyRelatedClusters instance:
    sdg = StatDataGetter(datacode="DM", weighted=True)
    sdg.get(hrc)

    This will return the data, weights, and the point about which to calculate things.
    In the example, as it didn't specify about_peak=True, that point will be the weighted mean
    of the DM data.

    Attributes:
    -----------
    datacode (str): Code for what kind of data to grab
              At the moment must be one of ["Property", "DCoverF", "DM"]
    weighted (bool, default=False): Whether to weight the statistic by sigma
    about_peak (bool, default=False): Whether to calculate, e.g. skewness, about the point of max sigma rather than the mean as is standard
    """

    datacode: str = attrib(validator=in_(["DCoverF", "DM"]))
    weighted: bool = attrib(default=False)
    about_peak: bool = attrib(default=False)

    def get(self, hrc):
        """
        Get the data corresponding to the datacode, weights, and point about which to
        calculate things.

        Returns (data, weights, point)
        """
        no_valid_data = True
        # set the weights
        if self.weighted:
            weights = hrc.main_cluster["sigma"]
        else:
            weights = 1.0 + np.zeros_like(hrc.main_cluster["sigma"])

        # get the data corresponding to the correct datacode
        # and the point if about_peak
        if self.datacode == "DM":
            data = hrc.main_cluster["dm"]
            no_valid_data = False
            if self.about_peak:
                point = hrc.dm
            else:
                point = NOPOINT
        elif self.datacode == "DCoverF":
            try:
                data = hrc.main_cluster["dc"] / hrc.main_cluster["freq"]
                if self.about_peak:
                    point = hrc.dc / hrc.f
            except ValueError:
                log.warning(
                    "Trying to compute dc/f but there is no dc field in main_cluster"
                    " for this HarmonicallyRelatedClusters"
                )
                return (None, None, None)

        # if not about_peak, compute correct point
        if self.about_peak:
            if point == NOPOINT and not no_valid_data:
                raise AttributeError(
                    "about_peak == True, have valid data, but point has not been set."
                    " Check code"
                )
        else:
            point = Mean.compute(data, weights).value

        return data, weights, point

    def make_name(self):
        """
        Two pieces of information are relevent and should be included in the
        final name of the feature: the data the statistic is derived from and
        any flags affecting how it was derived

        Therefore this function returns a dictionary like
        {'name': <datacode>, "add_on": <flag1>_<flag2>}
        e.g.
        {'name': 'DM', 'add_on': 'about_peak'}
        {'name': 'DCoverF', 'add_on': 'weighted_about_peak'}
        """
        flag_name = ""
        if self.weighted:
            flag_name += "weighted"
        if self.about_peak:
            if len(flag_name) > 0:
                flag_name += "_"
            flag_name += "about_peak"
        return {"name": self.datacode, "add_on": flag_name}


@attrs
class FitDataGetter(DataGetter):
    datacode: str = attrib(
        validator=in_(
            [
                "dm_sigma",
                "freq_sigma",
            ]
        )
    )

    def get(self, hrc):
        if self.datacode == "dm_sigma":
            best_2d_freq = (np.abs(hrc.dm_freq_sigma["freqs"] - hrc.freq)).argmin()
            xdata = hrc.dm_freq_sigma["dms"]
            ydata = hrc.dm_freq_sigma["sigmas"][:, best_2d_freq]
            compare_point = [hrc.dm, hrc.sigma]  # x, y
        if self.datacode == "freq_sigma":
            best_2d_dm = (np.abs(hrc.dm_freq_sigma["dms"] - hrc.dm)).argmin()
            xdata = hrc.dm_freq_sigma["freqs"]
            ydata = hrc.dm_freq_sigma["sigmas"][best_2d_dm, :]
            compare_point = [hrc.freq, hrc.sigma]  # x, y
        return xdata, ydata, compare_point

    def make_name(self):
        return {"name": self.datacode}


###################################################################
######################## Generator Classes ########################


class FeatureGenerator:
    """
    Empty parent class - mainly in order to check if things are a FeatureGenerator.

    FeatureGenerator's store what class they are using to fit/calculate/etc,
    and all have a make method which takes in data from the class's
    corresponding DataGetter and returns the feature value/s and dtype

    Methods:
    --------
    make(data output from DataGetter): make the feature
    make_name(): Make a name to describe itself (used to form/check _dtype_structure in Features)
    """

    def make(self, data):
        """Make feature from data output from a DataGetter."""
        raise NotImplementedError("Subclasses must implement this method")

    def make_name(self):
        """Make feature from data output from a DataGetter."""
        raise NotImplementedError("Subclasses must implement this method")


@attrs
class PropertyGenerator(FeatureGenerator):
    """
    Get a property of a HarmonicallyRelatedCluster. Properties which are iterable are
    not allowed/will be blocked.

    Example usage with pdg a PropertyDataGetter and hrc a HarmonicallyRelatedClusters:
    pg = PropertyGenerator(propery="num_harmonics")
    pg.make(pdg.get(hrc))

    Attributes:
    -----------
    property (str): Name of the property
    """

    property: str = attrib()

    def make(self, hrc):
        """Get the value and type of the specified property."""
        value = getattr(hrc, self.property)
        dtype = type(value)
        try:
            iter(value)
            log.warning(
                f"{self.property} - properties which are iterable cannot be used as"
                " features"
            )
            return (np.nan, float)
        except TypeError:
            return value, dtype

    def make_name(self):
        """Returns the name of the property."""
        return {"name": self.property}


@attrs
class StatGenerator(FeatureGenerator):
    """Compute a statistic from a HarmonicallyRelatedClusters object."""

    statclass = attrib(validator=make_issubclass_validator(Stat))

    def make(self, input):
        """Compute the statistic from data, weights and a point."""
        data, weights, point = input
        if data is None:
            return (np.nan, float)
        stat = self.statclass.compute(data, weights, point)
        return stat.value, stat.dtype

    def make_name(self):
        """Returns the statclass name."""
        return {"name": self.statclass.__name__}


@attrs
class FitGenerator(FeatureGenerator):
    """
    Fit data from a HarmonicallyRelatedClusters object.

    Attributes:
    -----------
    fitclass: class in features.py which will be used for the fit.

    rel_err (bool, default=False): also output the relative uncertainties for
        the fitted parameters.

    diff_from_detection (bool, default=False): e.g. when fitting a gaussian to
        the DM-sigma peak via FitGauss, you fit for the peak amplitude and
        location. Comparing these to the sigma and SM of the actual detection
        is probably useful. This option outputs <fitted_value> - <detection value>
        for any fitted parameters for which that makes sense.
        (compare_point in FitDataGetter.get is what grabs the detection values)

    ndof (bool, default=False): output the number of degrees of freedom for the
        fit.

    rms_err (bool, default=False): output the root mean square error from the
        fit.
    """

    fitclass = attrib(validator=make_issubclass_validator(Fit))
    rel_err: bool = attrib(default=False)
    diff_from_detection: bool = attrib(default=False)
    ndof: bool = attrib(default=False)
    rms_err: bool = attrib(default=False)
    # amalgamate: str = attrib(default="max_sigma", validator=in_(["max_sigma", "mean"]))
    # max_iters: int = attrib(default=10, validator=instance_of(int))

    def make(self, input):
        """Fit the data."""
        xdata, ydata, compare_point = input
        if xdata is None or ydata is None:
            expected_length = len(self.make_name()["indiv_names"])
            return ([np.nan] * expected_length, [float] * expected_length)

        fit = self.fitclass.fit(xdata, ydata, compare_point)

        vals, dts = fit.output(
            do_rel_err=self.rel_err,
            do_diff=self.diff_from_detection,
            do_ndof=self.ndof,
            do_rms_err=self.rms_err,
        )

        return vals, dts

    def make_name(self):
        # this is kind of clunky and annoying, but the names get made when
        # initializing, so it has to be independent of the feature itself
        fitclass_to_indiv_names = {
            "FitGauss": {
                "base": ["amplitude", "mu", "gauss_sigma"],
                "diff": ["amplitude-detection", "mu-detection"],
            },
            "FitGaussWidth": {
                "base": ["gauss_sigma"],
                "diff": [],
            },
        }

        name = self.fitclass.__name__

        indiv_names = fitclass_to_indiv_names[name]["base"]
        if self.rel_err:
            tmp = deepcopy(indiv_names)
            for nm in tmp:
                indiv_names.append(nm + "_rel_err")
        if self.diff_from_detection:
            indiv_names.extend(fitclass_to_indiv_names[name].get("diff", []))
        if self.ndof:
            indiv_names.append("ndof")
        if self.rms_err:
            indiv_names.append("rms_err")

        return {"name": name, "indiv_names": indiv_names}


###################################################################


def checktype_datagetter_and_featuregenerator(
    datagetter: DataGetter, featuregenerator: FeatureGenerator
):
    get2gen = {
        StatDataGetter: StatGenerator,
        PropertyDataGetter: PropertyGenerator,
        FitDataGetter: FitGenerator,
    }

    if not isinstance(featuregenerator, get2gen[datagetter.__class__]):
        raise TypeError(
            f"DataGetter ({datagetter}) and FeatureGenerator ({featuregenerator}) are"
            " incompatible"
        )


def make_combined_name(datagetter: DataGetter, featuregenerator: FeatureGenerator):
    """
    Set up make_name() for each DataGetter and FeatureGenerator so that it.

    returns a dictionary with:
      - a 'name' field as the main name - e.g. DM, Mean, etc
      - an 'add_on' field with a string containing stuff to add to the name
        e.g. a combination of flags or options, like 'weighted_about_peak'
      - an 'indiv_names' field if the FeatureGenerator returns multiple values,
        e.g. for a Fit
        this should be a list of the individual names of the parameters, e.g.
        for a gaussian fit you might have ["A", "mu", "sigma"]

    If any of the fields aren't applicaple to the DataGetter/FeatureGenerator, just
    don't include them.
    If the DataGetter/FeatureGenerator contributes nothing to the name of the
    feature then just return an empty dictionary

    If you have a datagetter.make_name() like:
        {'name': 'dat', 'add_on': '1_2'}
    and a featuregenerator.make_name() like:
        {'name': 'feat', 'add_on': 'A_B'}

    the name returned will be

    'dat_feat_1_2_A_B'

    if instead the featuregenerator.make_name() was
        {'name': 'feat', 'add_on': 'A_B', 'indiv_names': ["l", "m", "n"]}
    this function will return a list:

    ['dat_feat_1_2_A_B_l', 'dat_feat_1_2_A_B_m', 'dat_feat_1_2_A_B_n']
    """
    dg_name = datagetter.make_name()
    fg_name = featuregenerator.make_name()

    checktype_datagetter_and_featuregenerator(datagetter, featuregenerator)

    main_name_parts = [
        dg_name.get("name", ""),
        fg_name.get("name", ""),
        dg_name.get("add_on", ""),
        fg_name.get("add_on", ""),
    ]
    main_name = "_".join([part for part in main_name_parts if part != ""])

    # deal with if the feature returns multiple parameters, like for a fit
    if fg_name.get("indiv_names") is not None:
        many_names = []
        for indiv_name in fg_name.get("indiv_names"):
            many_names.append(main_name + "_" + indiv_name)
        return many_names
    else:
        return main_name


def flags_from_list(flag_names: List[str], flag_list: List[str]) -> Dict:
    """Helper function for Features.from_config Outputs a dictionary with keys
    flag_names, and value True/False depending whether it appears in flag_list.
    """
    out = {}
    for flag_name in flag_names:
        out[flag_name] = flag_name in flag_list
    return out


@attrs
class Features:
    """
    Class contains a list of FeatureGenerator objects and a corresponding list of
    DataGetter objects With make these will be used on a HarmonicallyRelatedClusters
    instance to generate features and output a SinglePointingCandidate. Also computes
    candidate arrays.

    Example useage:
    # initialise from config, a feature-configuration dicitonary
    fetures = Features.from_config(config)
    # run on psdc, a PowerSpectraDetectionClusters instance
    # to generate spc, a SinglePointingCandidate
    spc = fetures.make(hrc)

    Attributes:
    -----------
    generators (List(FeatureGenerator)): Initialized FeatureGenerator objects
    datagetters (List(DataGetter)): Initialized DataGetter objects
    """

    generators = attrib(
        validator=deep_iterable(
            member_validator=instance_of(FeatureGenerator),
            iterable_validator=instance_of(list),
        )
    )
    datagetters = attrib(
        validator=deep_iterable(
            member_validator=instance_of(DataGetter),
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
            del injection_dict["dms"]
            del injection_dict["bins"]
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


def get_min_max_index(center_index, index_range, array_len):
    """
    Get minimum and maximum index for given range in array.

    Returns the indices that are used for the candidate properties when trying out
    different dms or freqs. When at the edge of the range the the bboundaries are
    shifted that always the full number of dm trials is written out.

    Parameters ========== center_index (int): The index of the best trial
    index_range(int): The number of trials that are written around the best dm trial in
    both directions. array_len (int): The number of total trials in the investigated
    array.

    Returns ======= min_index (int): The lower index. max_index (int): The upper index.
    """
    idx_min = center_index - index_range
    idx_max = center_index + index_range + 1
    idx_min = idx_min if idx_min >= 0 else 0
    idx_max = idx_max if idx_max <= array_len else array_len

    if idx_min == 0 and idx_max != array_len:
        idx_max = min(index_range * 2 + 1, array_len)
    if idx_max == array_len and idx_min != 0:
        idx_min = max(array_len - (index_range * 2 + 1), 0)

    return idx_min, idx_max
