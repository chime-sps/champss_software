import datetime as dt
import enum
import logging

import numpy as np
from attr import Factory, asdict, attrib, attrs, converters, validators
from attr.setters import convert, validate
from bson.objectid import ObjectId
from sps_common.interfaces import (
    CandidateClassificationLabel,
    KnownSourceLabel,
    SearchAlgorithm,
)
from sps_common.interfaces.utilities import filter_class_dict

log = logging.getLogger(__name__)


class ObservationStatus(enum.Enum):
    scheduled = 1
    complete = 2
    failed = 3
    incomplete = 4


class ProcessStatus(enum.Enum):
    scheduled = 1
    complete = 2
    failed = 3
    incomplete = 4
    blocked = 5


class DailyStatus(enum.Enum):
    created = 0
    scheduled = 1
    finishedPipeline = 2
    finishedMultiPointing = 3
    finishedClassification = 4
    finishedFolding = 5


class DatabaseError(Exception):
    pass


processing_tier_limits = range(10, 110, 10)
processing_tier_names = [f"max-{max_limit}GB" for max_limit in processing_tier_limits]


@attrs
class Pointing:
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    beam_row = attrib(converter=int)
    length = attrib(converter=int)
    ne2001dm = attrib(converter=float)
    ymw16dm = attrib(converter=float)
    maxdm = attrib(converter=float)
    nchans = attrib(converter=int)
    search_algorithm = attrib(validator=validators.in_(SearchAlgorithm))
    # Strongest pulsars in a single day observations
    strongest_pulsar_detections = attrib(
        default={},
        converter=dict,
    )
    strongest_pulsar_detections_stack = attrib(
        default={},
        converter=dict,
    )
    last_changed = attrib(
        validator=validators.instance_of(dt.datetime), default=Factory(dt.datetime.now)
    )
    _id = attrib(
        default=None,
        alias="_id",
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )

    @property
    def id(self):
        return self._id

    @classmethod
    def from_db(cls, doc):
        """Create a `Pointing` instance from a MongoDB document."""
        # We have
        doc["search_algorithm"] = SearchAlgorithm(doc["search_algorithm"])
        filtered_doc = filter_class_dict(cls, doc)
        obj = cls(**filtered_doc)
        return obj

    def to_db(self):
        """Return a MongoDB document version of this instance."""
        doc = asdict(self)
        doc["_id"] = ObjectId(self.id)
        doc["search_algorithm"] = self.search_algorithm.value
        return doc

    # observations = relationship("Observation", back_populates="pointing")
    def observations(self, db):
        return [
            Observation.from_db(obs)
            for obs in db.observations.find({"pointing_id": ObjectId(self.id)})
        ]

    # ps_stack = relationship("PsStack", back_populates="pointing")
    def ps_stack(self, db):
        return PsStack.from_db(
            db.ps_stacks.find_one({"pointing_id": ObjectId(self.id)})
        )

    # hhat_stack = relationship("HhatStack", back_populates="pointing")
    def hhat_stack(self, db):
        return HhatStack.from_db(
            db.hhat_stacks.find_one({"pointing_id": ObjectId(self.id)})
        )


@attrs
class Observation:
    pointing_id = attrib(converter=str)
    datetime = attrib(validator=validators.instance_of(dt.datetime))
    datapath = attrib(converter=str)
    status = attrib(validator=validators.in_(ObservationStatus), type=ObservationStatus)
    _id = attrib(
        default=None,
        alias="_id",
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    mask_fraction = attrib(
        default=None,
        converter=converters.optional(float),
        on_setattr=convert,  # type: ignore
    )
    beta = attrib(
        default=None,
        converter=converters.optional(float),
        on_setattr=convert,  # type: ignore
    )
    barycentring_mode = attrib(default=None, converter=str)
    barycentric_cleaning = attrib(
        default=None,
        converter=converters.optional(bool),
        on_setattr=convert,  # type: ignore
    )
    birdies = attrib(
        default=None,
        converter=converters.optional(np.asarray),
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(np.int64),
                iterable_validator=validators.instance_of(np.ndarray),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    birdies_position = attrib(
        default=None,
        converter=converters.optional(np.asarray),
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(np.int64),
                iterable_validator=validators.instance_of(np.ndarray),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    birdies_height = attrib(
        default=None,
        converter=converters.optional(np.asarray),
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(float),
                iterable_validator=validators.instance_of(np.ndarray),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    birdies_left_freq = attrib(
        default=None,
        converter=converters.optional(np.asarray),
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(float),
                iterable_validator=validators.instance_of(np.ndarray),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    birdies_right_freq = attrib(
        default=None,
        converter=converters.optional(np.asarray),
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(float),
                iterable_validator=validators.instance_of(np.ndarray),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    birdie_file = attrib(
        default=None,
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    compared_obs = attrib(
        default=None,
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(str),
                iterable_validator=validators.instance_of(list),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    dm0_rn_file = attrib(
        default=None,
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    dm0_rn_medians = attrib(
        default=None,
        converter=converters.optional(np.asarray),
    )
    dm0_rn_scales = attrib(
        default=None,
        converter=converters.optional(np.asarray),
    )
    rn_file = attrib(
        default=None,
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    rn_medians = attrib(
        default=None,
        converter=converters.optional(np.asarray),
    )
    rn_scales = attrib(
        default=None,
        converter=converters.optional(np.asarray),
    )
    mean_power = attrib(
        default=None,
        converter=converters.optional(float),
        on_setattr=convert,  # type: ignore
    )
    std_power = attrib(
        default=None,
        converter=converters.optional(float),
        on_setattr=convert,  # type: ignore
    )
    add_to_stack = attrib(
        default=None,
        converter=converters.optional(bool),
        on_setattr=convert,  # type: ignore
    )
    qc_test = attrib(
        default=None,
        converter=converters.optional(dict),
        on_setattr=convert,  # type: ignore
    )
    frac_recovered_samples = attrib(
        default=None,
        converter=converters.optional(float),
        on_setattr=convert,  # type: ignore
    )
    num_total_candidates = attrib(
        default=None,
        converter=converters.optional(int),
        on_setattr=convert,  # type: ignore
    )
    num_rfi_candidates = attrib(
        default=None,
        converter=converters.optional(int),
        on_setattr=convert,  # type: ignore
    )
    num_harmonics = attrib(
        default=None,
        converter=converters.optional(int),
        on_setattr=convert,  # type: ignore
    )
    path_candidate_file = attrib(
        default=None,
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    num_detections = attrib(
        default=None,
        converter=converters.optional(int),
        on_setattr=convert,  # type: ignore
    )
    num_detections_used = attrib(
        default=None,
        converter=converters.optional(int),
        on_setattr=convert,  # type: ignore
    )
    detection_threshold = attrib(
        default=None,
        converter=converters.optional(int),
        on_setattr=convert,  # type: ignore
    )
    num_clusters = attrib(
        default=None,
        converter=converters.optional(int),
        on_setattr=convert,  # type: ignore
    )
    log_file = attrib(
        default=None,
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    last_changed = attrib(
        validator=validators.instance_of(dt.datetime), default=Factory(dt.datetime.now)
    )

    @datetime.validator
    def _validate_datetime(self, attribute, value):
        if not (value.tzinfo and value.utcoffset().total_seconds() == 0):
            raise ValueError(f"The tzinfo of {attribute.name} = {value} is not utc")

    @property
    def id(self):
        return self._id

    @classmethod
    def from_db(cls, doc, load_birdies=True, load_rn=True):
        """Create an `Observation` instance from a MongoDB document."""
        try:
            doc["status"] = ObservationStatus(doc["status"])
            birdie_file = doc.get("birdie_file", None)
            if birdie_file is not None and load_birdies:
                try:
                    birdie_info = np.load(birdie_file)
                    doc.update(birdie_info.items())
                except (FileNotFoundError, OSError) as e:
                    log.warning(f"Could not load birdie_file at {birdie_file}.")
                    log.warning(e)
            rn_file = doc.get("rn_file", None)
            if rn_file is not None and load_rn:
                try:
                    rn_arrays = np.load(rn_file)
                    doc["rn_medians"] = rn_arrays["medians"]
                    doc["rn_scales"] = rn_arrays["scales"]
                except (FileNotFoundError, OSError) as e:
                    log.warning(f"Could not load dm0 red noise file at {rn_file}.")
                    log.warning(e)
            dm0_rn_file = doc.get("dm0_rn_file", None)
            if dm0_rn_file is not None and load_rn:
                try:
                    dm0_rn_arrays = np.load(dm0_rn_file)
                    doc["dm0_rn_medians"] = dm0_rn_arrays["medians"]
                    doc["dm0_rn_scales"] = dm0_rn_arrays["scales"]
                except (FileNotFoundError, OSError) as e:
                    log.warning(f"Could not load dm0 red noise file at {dm0_rn_file}.")
                    log.warning(e)
            filtered_doc = filter_class_dict(cls, doc)

            obj = cls(**filtered_doc)
        except TypeError as e:
            log.warning(f"Could not load observation with dic {doc}.")
            log.warning(e)
            return None
        return obj

    def to_db(self):
        """Return a MongoDB document version of this instance."""
        doc = asdict(self)
        doc["_id"] = ObjectId(self.id)
        doc["pointing_id"] = ObjectId(self.pointing_id)
        doc["status"] = self.status.value
        return doc

    # pointing = relationship("Pointing", back_populates="observations")
    def pointing(self, db):
        return Pointing.from_db(db.pointings.find_one(ObjectId(self.pointing_id)))


@attrs
class Candidate:
    pointing_id = attrib(converter=str)
    observation_id = attrib(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(str),
            iterable_validator=validators.instance_of(list),
        )
    )
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    sigma = attrib(converter=float)
    freq = attrib(converter=float)
    dm = attrib(converter=float)
    dc = attrib(converter=float)
    num_days = attrib(converter=int)
    classification_label = attrib(
        validator=validators.in_(CandidateClassificationLabel)
    )
    classification_grade = attrib(converter=float)
    known_source_label = attrib(validator=validators.in_(KnownSourceLabel))
    known_source_matches = attrib(
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(list),
                iterable_validator=validators.instance_of(list),
            )
        )
    )
    last_changed = attrib(
        validator=validators.instance_of(dt.datetime), default=Factory(dt.datetime.now)
    )
    _id = attrib(
        default=None,
        alias="_id",
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )

    @property
    def id(self):
        return self._id

    @classmethod
    def from_db(cls, doc):
        """Create a `Candidate` instance from a MongoDB document."""
        doc["classification_label"] = CandidateClassificationLabel(
            doc["classification_label"]
        )
        doc["known_source_label"] = KnownSourceLabel(doc["known_source_label"])
        filtered_doc = filter_class_dict(cls, doc)
        obj = cls(**filtered_doc)
        return obj

    def to_db(self):
        """Return a MongoDB document version of this instance."""
        doc = asdict(self)
        doc["_id"] = ObjectId(self.id)
        doc["pointing_id"] = ObjectId(self.pointing_id)
        doc["observation_id"] = [ObjectId(obs) for obs in self.observation_id]
        doc["classification_label"] = self.classification_label.value
        doc["known_source_label"] = self.known_source_label.value
        return doc

    # pointing = relationship("Pointing", back_populates="observations")
    def pointing(self, db):
        return Pointing.from_db(db.pointings.find_one(ObjectId(self.pointing_id)))

    # observations = relationship("Observation", back_populates="pointing")
    def observations(self, db):
        return [
            Observation.from_db(db.observations.find_one(ObjectId(obs)))
            for obs in self.observation_id
        ]


@attrs
class PsStack:
    """
    PsStack class to store properties of a power series stack into the database.

    Parameters
    ==========
    pointing_id: str or None
        The id of the pointing.
    datapath_month: str
        Path to the saved monthly stack.
    datapath_cumul: str
        Path to the saved cumulative stack.
    datetimes_month: List(datetime.datetime)
        List of the observation dates in the current monthly stack.
    num_days_month: int
        Number of days in the current monthly stack.
    datetimes_cumul: List(datetime.datetime)
        List of the observation dates in the monthly stack.
    num_days_cumul: int
        Number of days in the cumulative stack.
    datetimes_per_month: List(List(datetime.datetime))
        Nested list of the observation dates in all processed monthly stacks.
        This includes stacks that were not added to the cumulative stack.
    qc_test_per_month: List(dict())
        List of the quality test in all processed monthly stacks.
        This includes stacks that were not added to the cumulative stack.
    qc_label_per_month: List(bool)
        List of quality labels in all processed monthly stacks.
        This determines if the monthly stacks were added to the cumulative stack.
    qc_test_cumul: List(dict())
        List of quality test of cumulative stack after adding new monthly stack.
    qc_label_per_month: List(bool)
        List of quality labels of cumulative stack after adding new monthly stack.
    """

    pointing_id = attrib(converter=str)
    datapath_month = attrib(converter=str)
    datapath_cumul = attrib(converter=str)
    datetimes_month = attrib(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(dt.datetime),
            iterable_validator=validators.instance_of(list),
        )
    )
    num_days_month = attrib(converter=int)
    datetimes_cumul = attrib(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(dt.datetime),
            iterable_validator=validators.instance_of(list),
        )
    )
    num_days_cumul = attrib(converter=int)
    _id = attrib(
        default=None,
        alias="_id",
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    datetimes_per_month = attrib(
        default=[],
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.deep_iterable(
                    member_validator=validators.instance_of(dt.datetime),
                    iterable_validator=validators.instance_of(list),
                ),
                iterable_validator=validators.instance_of(list),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    qc_test_per_month = attrib(
        default=[],
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(dict),
                iterable_validator=validators.instance_of(list),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    qc_label_per_month = attrib(
        default=[],
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(bool),
                iterable_validator=validators.instance_of(list),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    qc_test_cumul = attrib(
        default=[],
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(dict),
                iterable_validator=validators.instance_of(list),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    cumul_stack_quality_label = attrib(
        default=[],
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(bool),
                iterable_validator=validators.instance_of(list),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    path_candidate_files = attrib(
        default=[],
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(str),
                iterable_validator=validators.instance_of(list),
            )
        ),
        on_setattr=validate,  # type: ignore
    )
    last_changed = attrib(
        validator=validators.instance_of(dt.datetime), default=Factory(dt.datetime.now)
    )

    @datetimes_month.validator
    @datetimes_cumul.validator
    def _validate_datetimes(self, attribute, value):
        for val in value:
            if not (val.tzinfo and val.utcoffset().total_seconds() == 0):
                raise ValueError(
                    f"The tzinfo of the elements of {attribute.name} = {val.tzinfo} are"
                    " not utc"
                )

    @datetimes_per_month.validator
    def _validate_datetimes_per_month(self, attribute, value):
        for month in value:
            for val in month:
                if not (val.tzinfo and val.utcoffset().total_seconds() == 0):
                    raise ValueError(
                        f"The tzinfo of the elements of {attribute.name} ="
                        f" {val.tzinfo} are not utc"
                    )

    @property
    def id(self):
        return self._id

    @classmethod
    def from_db(cls, doc):
        """Create an `PsStack` instance from a MongoDB document."""
        filtered_doc = filter_class_dict(cls, doc)
        obj = cls(**filtered_doc)
        return obj

    def to_db(self):
        """Return a MongoDB document version of this instance."""
        doc = asdict(self)
        doc["_id"] = ObjectId(self.id)
        doc["pointing_id"] = ObjectId(self.pointing_id)
        return doc

    # pointing = relationship("Pointing", back_populates="ps_stack")
    def pointing(self, db):
        return Pointing.from_db(db.pointings.find_one(ObjectId(self.pointing_id)))


@attrs
class HhatStack:
    pointing_id = attrib(converter=str)
    datapath_month = attrib(converter=str)
    datapath_cumul = attrib(converter=str)
    datetimes_month = attrib(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(dt.datetime),
            iterable_validator=validators.instance_of(list),
        )
    )
    num_days_month = attrib(converter=int)
    datetimes_cumul = attrib(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(dt.datetime),
            iterable_validator=validators.instance_of(list),
        )
    )
    num_days_cumul = attrib(converter=int)
    sliced_by = attrib(converter=str)
    r_value = attrib(converter=float)
    _id = attrib(
        default=None,
        alias="_id",
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )

    @datetimes_month.validator
    @datetimes_cumul.validator
    def _validate_datetimes(self, attribute, value):
        for val in value:
            if not (val.tzinfo and val.utcoffset().total_seconds() == 0):
                raise ValueError(
                    f"The tzinfo of the elements of {attribute.name} = {val.tzinfo} are"
                    " not utc"
                )

    @property
    def id(self):
        return self._id

    @classmethod
    def from_db(cls, doc):
        """Create an `HhatStack` instance from a MongoDB document."""
        filtered_doc = filter_class_dict(cls, doc)
        obj = cls(**filtered_doc)
        return obj

    def to_db(self):
        """Return a MongoDB document version of this instance."""
        doc = asdict(self)
        doc["_id"] = ObjectId(self.id)
        doc["pointing_id"] = ObjectId(self.pointing_id)
        return doc

    # pointing = relationship("Pointing", back_populates="hhat_stack")
    def pointing(self, db):
        return Pointing.from_db(db.pointings.find_one(ObjectId(self.pointing_id)))


@attrs
class KnownSource:
    source_type = attrib(converter=int)
    source_name = attrib(converter=str)
    pos_ra_deg = attrib(converter=float)
    pos_dec_deg = attrib(converter=float)
    pos_error_semimajor_deg = attrib(converter=float)
    pos_error_semiminor_deg = attrib(converter=float)
    pos_error_theta_deg = attrib(converter=float)
    dm = attrib(converter=float)
    dm_error = attrib(converter=float)
    spin_period_s = attrib(converter=float)
    spin_period_s_error = attrib(converter=float)
    dm_galactic_ne_2001_max = attrib(converter=float)
    dm_galactic_ymw_2016_max = attrib(converter=float)
    spin_period_derivative = attrib(default=0, converter=float, type=float)
    spin_period_derivative_error = attrib(default=0, converter=float, type=float)
    spin_period_epoch = attrib(default=0, converter=float, type=float)
    detection_history = attrib(default=[], type=list)
    survey = attrib(
        default=[],
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(str),
                iterable_validator=validators.instance_of(list),
            )
        ),
        on_setattr=validate,
    )
    last_changed = attrib(
        validator=validators.instance_of(dt.datetime), default=Factory(dt.datetime.now)
    )
    _id = attrib(
        default=None,
        alias="_id",
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )

    @property
    def id(self):
        return self._id

    @classmethod
    def from_db(cls, doc):
        """Create a `KnownSource` instance from a MongoDB document."""
        filtered_doc = filter_class_dict(cls, doc)
        obj = cls(**filtered_doc)
        return obj

    def to_db(self):
        """Return a MongoDB document version of this instance."""
        doc = asdict(self)
        doc["_id"] = ObjectId(self.id)
        return doc


@attrs
class FollowUpSource:
    # "source_type of 'known_source', 'sd_candidate', 'md_candidate', governing followup plan"
    source_type = attrib(converter=str)
    source_name = attrib(converter=str)
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    dm = attrib(converter=float)
    f0 = attrib(converter=float)
    dm_galactic_ne_2001_max = attrib(converter=float)
    dm_galactic_ymw_2016_max = attrib(converter=float)
    pepoch = attrib(default=0, converter=float, type=float)
    candidate_sigma = attrib(default=0, converter=float, type=float)
    folding_history = attrib(default=[], type=list)
    coherentsearch_history = attrib(default=[], type=list)
    followup_duration = attrib(default=1, converter=int, type=int)
    path_to_ephemeris = attrib(
        default=None, converter=converters.optional(str), type=str
    )
    path_to_timfile = attrib(default=None, converter=converters.optional(str), type=str)
    path_to_candidates = attrib(
        default=[],
        validator=validators.optional(
            validators.deep_iterable(
                member_validator=validators.instance_of(str),
                iterable_validator=validators.instance_of(list),
            )
        ),
        on_setattr=validate,
    )
    active = attrib(default=True, converter=bool, type=bool)
    last_changed = attrib(
        validator=validators.instance_of(dt.datetime), default=Factory(dt.datetime.now)
    )
    _id = attrib(
        default=None,
        alias="_id",
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )

    @property
    def id(self):
        return self._id

    @classmethod
    def from_db(cls, doc):
        """Create a `KnownSource` instance from a MongoDB document."""
        filtered_doc = filter_class_dict(cls, doc)
        obj = cls(**filtered_doc)
        return obj

    def to_db(self):
        """Return a MongoDB document version of this instance."""
        doc = asdict(self)
        doc["_id"] = ObjectId(self.id)
        return doc


@attrs
class Process:
    pointing_id = attrib(converter=str)
    ra = attrib(converter=float)
    dec = attrib(converter=float)
    datetime = attrib(validator=validators.instance_of(dt.datetime))
    date = attrib(converter=str)
    nchan = attrib(converter=int)
    ntime = attrib(converter=int)
    maxdm = attrib(converter=float)
    status = attrib(validator=validators.in_(ProcessStatus), type=ProcessStatus)
    obs_status = attrib(
        validator=validators.in_(ObservationStatus), type=ObservationStatus
    )
    quality_label = attrib(
        converter=converters.optional(bool),
        on_setattr=convert,  # type: ignore
    )
    is_in_stack = attrib(
        converter=converters.optional(bool),
        on_setattr=convert,  # type: ignore
    )
    folded_status = attrib(
        converter=converters.optional(bool),
        on_setattr=convert,  # type: ignore
        default=False,
    )
    last_changed = attrib(
        validator=validators.instance_of(dt.datetime), default=Factory(dt.datetime.now)
    )
    process_time = attrib(converter=converters.optional(float), default=None)
    _id = attrib(
        default=None,
        alias="_id",
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    obs_id = attrib(
        default=None,
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    error_message = attrib(
        default=None,
        converter=converters.optional(str),
        on_setattr=convert,  # type: ignore
    )
    max_memory_usage = attrib(
        default=None,
        converter=converters.optional(float),
        on_setattr=convert,  # type: ignore
    )
    max_cpu_usage = attrib(
        default=None,
        converter=converters.optional(float),
        on_setattr=convert,  # type: ignore
    )

    @property
    def id(self):
        return self._id

    @property
    def stack_finished(self):
        """Tells you whether you still need to run the stacking process."""
        if self.is_in_stack:
            return True
        else:
            if self.quality_label:
                return False
            else:
                return True

    @property
    def ram_requirement(self):
        return min(100.0, (4 + self.maxdm * 0.05 + self.ntime * 1.35e-5))

    @property
    def tier(self):
        ram_requirement = self.ram_requirement
        for index, mem_limit in enumerate(processing_tier_limits):
            if ram_requirement < mem_limit:
                break
        tier_name = processing_tier_names[index]
        tier_limit = processing_tier_limits[index]
        return tier_name, tier_limit

    @classmethod
    def from_db(cls, doc):
        """Create a `Process` instance from a MongoDB document."""
        doc["status"] = ProcessStatus(doc["status"])
        doc["obs_status"] = ObservationStatus(doc["obs_status"])
        filtered_doc = filter_class_dict(cls, doc)
        obj = cls(**filtered_doc)
        return obj

    def to_db(self):
        """Return a MongoDB document version of this instance."""
        doc = asdict(self)
        doc["_id"] = ObjectId(self.id)
        doc["pointing_id"] = ObjectId(self.pointing_id)
        doc["status"] = self.status.value
        doc["obs_status"] = self.obs_status.value
        return doc


@attrs
class DailyRun:
    date = attrib(validator=validators.instance_of(dt.date))
    # status = attrib(validator=validators.in_(DailyStatus), type=DailyStatus)
    schedule_result = attrib(
        default={},
        converter=converters.optional(dict),
    )
    pipeline_result = attrib(
        default={},
        converter=converters.optional(dict),
    )
    multipointing_result = attrib(
        default={},
        converter=converters.optional(dict),
    )
    classification_result = attrib(
        default={},
        converter=converters.optional(dict),
    )
    folding_result = attrib(
        default={},
        converter=converters.optional(dict),
    )
    last_changed = attrib(
        validator=validators.instance_of(dt.datetime), default=Factory(dt.datetime.now)
    )
    plots = attrib(
        default={},
        converter=dict,
    )

    @property
    def id(self):
        return self._id

    @classmethod
    def from_db(cls, doc):
        """Create a `DailyRun` instance from a MongoDB document."""
        filtered_doc = filter_class_dict(cls, doc)
        obj = cls(**filtered_doc)
        return obj

    def to_db(self):
        """Return a MongoDB document version of this instance."""
        doc = asdict(self)
        doc["_id"] = ObjectId(self.id)
        return doc

    @property
    def status(self):
        steps = ["schedule", "pipeline", "multipointing", "classification", "folding"]
        status = 0
        for step in steps:
            if getattr(self, f"{step}_result", {}) != {}:
                status += 1
            else:
                break
        return status
