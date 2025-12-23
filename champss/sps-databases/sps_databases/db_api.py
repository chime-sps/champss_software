"""Simpe API to gather collection data using pymongo."""

import datetime as dt

import pymongo
import pytz
from bson.objectid import ObjectId
from sps_common.constants import SIDEREAL_S
from sps_databases import db_utils
from sps_databases.models import (
    Candidate,
    DatabaseError,
    FollowUpSource,
    HhatStack,
    KnownSource,
    Observation,
    ObservationStatus,
    Pointing,
    Process,
    ProcessStatus,
    PsStack,
    DailyRun,
)


def create_pointing(payload):
    """
    Create a `Pointing` instance and inserts it into the database.

    Parameters
    ----------
    payload: dict
        dictionary of attributes for the pointing, as expected by `models.Pointing`

    Returns
    -------
    pointing: sps_database.models.Pointing
        the new instance, including the created MongoDB document id
    """
    db = db_utils.connect()
    pointing = Pointing(**payload)
    doc = pointing.to_db()
    del doc["_id"]
    rc = db.pointings.insert_one(doc)
    pointing._id = rc.inserted_id
    return pointing


def get_pointing(pointing_id):
    """
    Find an `Pointing` instance with the given id.

    Parameters
    ----------
    pointing_id: bson.objectid.ObjectId or string
        key for the id lookup in the "pointings" collection,
        as expected by pymongo.collection.Collection.find()

    Returns
    -------
    pointing: sps_database.models.Pointing
        the pointing instance with the given id

    Exceptions
    ----------
    Raises a runtime exception if the pointing id does not exist
    """
    db = db_utils.connect()
    if isinstance(pointing_id, str):
        pointing_id = ObjectId(pointing_id)
    return Pointing.from_db(db.pointings.find_one(pointing_id))


def update_pointing(pointing_id, payload):
    """
    Updates pointing to the given attribute and values as a dict.

    Parameters
    ----------
    pointing_id: str or ObjectId
        The id of the observation to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    pointing: dict
        The dict of the updated pointing entry
    """
    db = db_utils.connect()
    payload["last_changed"] = dt.datetime.now()
    if isinstance(pointing_id, str):
        pointing_id = ObjectId(pointing_id)
    return db.pointings.find_one_and_update(
        {"_id": pointing_id},
        {"$set": payload},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def create_observation(payload):
    """
    Create an `Observation` instance and inserts it into the database.

    Parameters
    ----------
    payload: dict
        dictionary of attributes for the observation,
        as expected by `models.Observation`

    Returns
    -------
    observation: sps_database.models.Observation
        the new instance, including the created MongoDB document id
    """
    db = db_utils.connect()
    observation = Observation(**payload)
    doc = observation.to_db()
    del doc["_id"]
    query = {k: doc[k] for k in doc.keys() & {"pointing_id", "datetime"}}
    rc = db.observations.find_one_and_replace(
        query, doc, upsert=True, return_document=pymongo.ReturnDocument.AFTER
    )
    observation._id = rc["_id"]
    return observation


def get_observation(observation_id):
    """
    Find an `Observation` instance with the given id.

    Parameters
    ----------
    observation_id: bson.objectid.ObjectId or string
        key for the id lookup in the "observations" collection,
        as expected by pymongo.collection.Collection.find()

    Returns
    -------
    observation: sps_database.models.Observation
        the observation instance with the given id

    Exceptions
    ----------
    Raises a runtime exception if the pointing id does not exist
    """
    db = db_utils.connect()
    if isinstance(observation_id, str):
        observation_id = ObjectId(observation_id)
    obs_dic = db.observations.find_one(observation_id)
    if obs_dic is None:
        raise DatabaseError(
            f"Could not find observation with id {observation_id} in database."
        )
    else:
        return Observation.from_db(obs_dic)


def update_observation(observation_id, payload):
    """
    Updates observation to the given attribute and values as a dict.

    Parameters
    ----------
    observation_id: str or ObjectId
        The id of the observation to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    observation: dict
        The dict of the updated observation entry
    """
    db = db_utils.connect()
    payload["last_changed"] = dt.datetime.now()
    if isinstance(observation_id, str):
        observation_id = ObjectId(observation_id)
    new_obs = db.observations.find_one_and_update(
        {"_id": observation_id},
        {"$set": payload},
        return_document=pymongo.ReturnDocument.AFTER,
        upsert=False,
    )
    if new_obs is None:
        raise DatabaseError("Trying to update observation that does not exist.")
    else:
        return new_obs


def get_observations(pointing_id):
    """
    Returns a list of `Observation` instance with the given pointing id.

    Parameters
    ----------
    pointing_id: bson.objectid.ObjectId or string
        pointing_id to lookup in the "observations" collection,
        as expected by pymongo.collection.Collection.find()

    Returns
    -------
    observations: list(sps_database.models.Observation)
        List of Observation objects from the database
    """
    db = db_utils.connect()
    if isinstance(pointing_id, str):
        pointing_id = ObjectId(pointing_id)
    return [
        obs
        for o in db.observations.find({"pointing_id": pointing_id})
        if (obs := Observation.from_db(o)) is not None
    ]


def get_observation_date(pointing_id, datetime):
    """
    Finds the `Observation` instance for a given pointing id at a given date.

    Parameters
    ----------
    pointing_id: bson.objectid.ObjectId or string
        pointing_id to lookup in the "observations" collection,
        as expected by pymongo.collection.Collection.find()

    datetime: datetime.datetime
        The datetime.datetime object of the date
        to search for the `Observation` instance.

    Returns
    -------
    observation: sps_database.models.Observation or None
        The Observation object from the database
        for the given date, if it exists. Otherwise return None.
    """
    db = db_utils.connect()
    if type(pointing_id) == str:
        pointing_id = ObjectId(pointing_id)
    obs = db.observations.find_one(
        {
            "pointing_id": pointing_id,
            "datetime": {"$gte": datetime, "$lte": datetime + dt.timedelta(days=1)},
        }
    )
    if obs:
        return Observation.from_db(obs)
    else:
        return None


def get_observations_around_observation(observation, ra_range, dec_range):
    """
    Returns `Observation` list within RA/Dec range around an observation.

    Parameters
    ----------
    observation: sps_database.models.Observation
        The Observation object as defined in sps_database.models

    ra_range: The RA range around the observation of interest in degrees.

    dec_range: The Dec range around the observation of interest in degrees.

    Returns
    -------
    observations: list(sps_database.models.Observation)
        List of `Observation` objects from the database around
        the observation of interest
    """
    if ra_range > 180:
        ra_range = 180
    db = db_utils.connect()
    pointing_id = ObjectId(observation.pointing_id)
    pointing = db.pointings.find_one(pointing_id)
    ra_min = pointing["ra"] - ra_range
    if ra_min < 0:
        ra_min += 360
        lower_edge = True
    else:
        lower_edge = False
    ra_max = pointing["ra"] + ra_range
    if ra_max > 360:
        ra_max -= 360
        upper_edge = True
    else:
        upper_edge = False

    if lower_edge or upper_edge:
        ra_query = {"$or": [{"ra": {"$gte": ra_min}}, {"ra": {"$lte": ra_max}}]}
    else:
        ra_query = {
            "ra": {
                "$gte": ra_min,
                "$lte": ra_max,
            }
        }

    surrounding_pointings = [
        ObjectId(p["_id"])
        for p in db.pointings.find(
            {
                **ra_query,
                "dec": {
                    "$gte": pointing["dec"] - dec_range,
                    "$lte": pointing["dec"] + dec_range,
                },
            }
        )
    ]
    return [
        obs
        for o in db.observations.find(
            {
                "pointing_id": {"$in": surrounding_pointings},
                "datetime": {
                    "$gte": observation.datetime - dt.timedelta(hours=ra_range / 15),
                    "$lte": observation.datetime + dt.timedelta(hours=ra_range / 15),
                },
            }
        )
        if (obs := Observation.from_db(o)) is not None
    ]


def get_observations_before_observation(
    observation, time_before, num_rows, row_offset=0, ra_range=2.5
):
    """
    Returns `Observation` list at given seconds before the current observation.

    List of 'Observation' is a row of length (num_rows * 2 + 1).

    Parameters
    -------
    observation: sps_database.models.Observation
        The Observation object as defined in sps_database.models

    time_before: The time before the input observations in seconds

    num_rows: The number of rows above and below the input observation to retrieve.

    row_offset: The offset in number of rows from the input observation to
    retrieve previous observations. Positive number is towards the North.

    Returns
    -------
    observations: list(sps_database.models.Observation)
        List of `Observation` objects from the database before the
        observation of interest
    """
    db = db_utils.connect()
    pointing_id = ObjectId(observation.pointing_id)
    pointing = db.pointings.find_one(pointing_id)
    ra_before = pointing["ra"] - 360 * (time_before / SIDEREAL_S)
    if ra_before < 0:
        ra_before += 360
    ra_min = ra_before - ra_range
    ra_query = {}
    if ra_min < 0:
        ra_min += 360
        lower_edge = True
    else:
        lower_edge = False
    ra_max = ra_before + ra_range
    if ra_max > 360:
        ra_max -= 360
        upper_edge = True
    else:
        upper_edge = False
    if lower_edge or upper_edge:
        ra_query = {"$or": [{"ra": {"$gte": ra_min}}, {"ra": {"$lte": ra_max}}]}
    else:
        ra_query = {
            "ra": {
                "$gte": ra_min,
                "$lte": ra_max,
            }
        }

    earlier_pointings = [
        ObjectId(p["_id"])
        for p in db.pointings.find(
            {
                **ra_query,
                "beam_row": {
                    "$gte": pointing["beam_row"] - num_rows + row_offset,
                    "$lte": pointing["beam_row"] + num_rows + row_offset,
                },
            }
        )
    ]
    return [
        obs
        for o in db.observations.find(
            {
                "pointing_id": {"$in": earlier_pointings},
                "datetime": {
                    "$gte": observation.datetime - dt.timedelta(hours=ra_range / 15),
                    "$lte": observation.datetime + dt.timedelta(hours=ra_range / 15),
                },
            }
        )
        if (obs := Observation.from_db(o)) is not None
    ]


def delete_observation(observation_id):
    """
    Delete an observation entry with the given observation id.

    Parameters
    ----------
    observation_id: str or ObjectId
        The observation_id of the observation entry to be deleted

    Returns
    -------
    observation: dict
        The dict of the observation entry deleted
    """
    db = db_utils.connect()
    if isinstance(observation_id, str):
        observation_id = ObjectId(observation_id)
    return db.observations.find_one_and_delete({"_id": observation_id})


def create_candidate(payload):
    """
    Create an `Candidate` instance and inserts it into the database.

    Parameters
    ----------
    payload: dict
        dictionary of attributes for the observation, as expected
        by `models.Candidate`

    Returns
    -------
    candidate: sps_database.models.Candidate
        the new instance, including the created MongoDB document id
    """
    db = db_utils.connect()
    candidate = Candidate(**payload)
    doc = candidate.to_db()
    del doc["_id"]
    query = {k: doc[k] for k in doc.keys() & {"observation_id", "freq", "dm"}}
    rc = db.candidates.find_one_and_replace(
        query, doc, upsert=True, return_document=pymongo.ReturnDocument.AFTER
    )
    candidate._id = rc["_id"]
    return candidate


def get_candidates(observation_id):
    """
    Find all `Candidate` instance with the given observation id.

    Parameters
    ----------
    pointing_id: bson.objectid.ObjectId or string
        key for the id lookup in the "candidates" collection,
        as expected by pymongo.collection.Collection.find()

    Returns
    -------
    candidates: List(sps_database.models.Candidate)
        the observation instance with the given id

    Exceptions
    ----------
    Raises a runtime exception if the pointing id does not exist
    """
    db = db_utils.connect()
    if isinstance(observation_id, str):
        observation_id = ObjectId(observation_id)
    return [
        Candidate.from_db(cand)
        for cand in db.candidates.find({"observation_id": observation_id})
    ]


def update_candidate(candidate_id, payload):
    """
    Updates a candidate entry with the given attribute/values as a dict.

    Parameters
    ----------
    candidate_id: str or ObjectId
        The id of the candidate to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    observation: dict
        The dict of the updated candidate entry
    """
    db = db_utils.connect()
    payload["last_changed"] = dt.datetime.now()
    if isinstance(candidate_id, str):
        candidate_id = ObjectId(candidate_id)
    return db.candidates.find_one_and_update(
        {"_id": candidate_id},
        {"$set": payload},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def delete_candidate(candidate_id):
    """
    Delete an candidate entry with the given candidate id.

    Parameters
    ----------
    candidate_id: str or ObjectId
        The candidate_id of the candidate entry to be deleted

    Returns
    -------
    candidate: dict
        The dict of the candidate entry deleted
    """
    db = db_utils.connect()
    if isinstance(candidate_id, str):
        candidate_id = ObjectId(candidate_id)
    return db.observations.find_one_and_delete({"_id": candidate_id})


def create_ps_stack(payload):
    """
    Create a `PsStack` instance and inserts it into the database.

    Parameters
    ----------
    payload: dict
        dictionary of attributes for the stack, as expected by `models.PsStack`

    Returns
    -------
    stack: sps_database.models.PsStack
        the new instance, including the created MongoDB document id
    """
    db = db_utils.connect()
    ps_stack = PsStack(**payload)
    doc = ps_stack.to_db()
    del doc["_id"]
    query = {k: doc[k] for k in doc.keys() & {"pointing_id"}}
    rc = db.ps_stacks.find_one_and_replace(
        query, doc, upsert=True, return_document=pymongo.ReturnDocument.AFTER
    )
    ps_stack._id = rc["_id"]
    return ps_stack


def get_ps_stack(pointing_id):
    """
    Find an `PsStack` instance for the given pointing.

    Parameters
    ----------
    pointing_id: bson.objectid.ObjectId or string
        key for the pointing lookup, as expected by
        pymongo.collection.Collection.find()

    Returns
    -------
    stack: sps_database.models.PsStack or None
        the stack instance for the given pointing, or None if
        no stacks of a given pointing_id exists
    """
    db = db_utils.connect()
    if isinstance(pointing_id, str):
        pointing_id = ObjectId(pointing_id)
    ps_stack = db.ps_stacks.find_one({"pointing_id": pointing_id})
    if ps_stack:
        return PsStack.from_db(ps_stack)
    else:
        return None


def update_ps_stack(pointing_id, payload):
    """
    Updates a power spectra stack entry of a given pointing.

    The PS stack entry is updated with the given attribute
    and values as a dict.

    Parameters
    -------
    pointing_id: str or ObjectId
        The pointing id of the power spectra stack to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    observation: dict
        The dict of the updated power spectra stack database entry
    """
    db = db_utils.connect()
    payload["last_changed"] = dt.datetime.now()
    if isinstance(pointing_id, str):
        pointing_id = ObjectId(pointing_id)
    return db.ps_stacks.find_one_and_update(
        {"pointing_id": pointing_id},
        {"$set": payload},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def append_ps_stack(pointing_id, payload):
    """
    Appends to a power spectra stack entry of a given pointing.

    The PS stack entry is apppended to with the given attribute
    and values as a dict.

    Parameters
    ---------
    pointing_id: str or ObjectId
        The pointing id of the power spectra stack to be updated
    payload: dict
        The dict of the attributes and values which are appended

    Returns
    -------
    observation: dict
        The dict of the updated power spectra stack database entry
    """
    db = db_utils.connect()
    payload_time = {"last_changed": dt.datetime.now()}
    if isinstance(pointing_id, str):
        pointing_id = ObjectId(pointing_id)
    return db.ps_stacks.find_one_and_update(
        {"pointing_id": pointing_id},
        {"$push": payload, "$set": payload_time},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def create_hhat_stack(payload):
    """
    Create an `HhatStack` instance and inserts it into the database.

    Parameters
    ----------
    payload: dict
        dictionary of attributes for the stack,
        as expected by `models.HhatStack`

    Returns
    -------
    stack: sps_database.models.HhatStack
        the new instance, including the created MongoDB document id
    """
    db = db_utils.connect()
    hhat_stack = HhatStack(**payload)
    doc = hhat_stack.to_db()
    del doc["_id"]
    query = {k: doc[k] for k in doc.keys() & {"pointing_id"}}
    rc = db.hhat_stacks.find_one_and_replace(
        query, doc, upsert=True, return_document=pymongo.ReturnDocument.AFTER
    )
    hhat_stack._id = rc["_id"]
    return hhat_stack


def get_hhat_stack(pointing_id):
    """
    Find an `HhatStack` instance for the given pointing.

    Parameters
    ----------
    pointing_id: bson.objectid.ObjectId or string
        key for the pointing lookup, as expected by
        pymongo.collection.Collection.find()

    Returns
    -------
    stack: sps_database.models.HhatStack
        the stack instance for the given pointing

    Exceptions
    ----------
    Raises a runtime exception if the pointing id does not exist
    """
    db = db_utils.connect()
    if isinstance(pointing_id, str):
        pointing_id = ObjectId(pointing_id)
    return HhatStack.from_db(db.hhat_stacks.find_one({"pointing_id": pointing_id}))


def update_hhat_stack(pointing_id, payload):
    """
    Updates a hhat stack entry of a given pointing.

    The hhat stack entry is updated with the given attribute and
    values as a dict.

    Parameters
    ----------
    pointing_id: str or ObjectId
        The pointing_id of the hhat stack to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    observation: dict
        The dict of the updated hhat stack entry
    """
    db = db_utils.connect()
    if isinstance(pointing_id, str):
        pointing_id = ObjectId(pointing_id)
    return db.hhat_stacks.find_one_and_update(
        {"pointing_id": pointing_id},
        {"$set": payload},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def create_known_source(payload):
    """
    Create a `KnownSource` instance and inserts it into the database.

    Parameters
    ----------
    payload: dict
        dictionary of attributes for the stack, as expected by `models.KnownSource`

    Returns
    -------
    known_source: sps_database.models.KnownSource
        the new instance, including the created MongoDB document id
    """
    db = db_utils.connect()
    known_source = KnownSource(**payload)
    doc = known_source.to_db()
    del doc["_id"]
    query = {k: doc[k] for k in doc.keys() & {"source_name"}}
    rc = db.known_sources.find_one_and_replace(
        query, doc, upsert=True, return_document=pymongo.ReturnDocument.AFTER
    )
    known_source._id = rc["_id"]
    return known_source


def update_known_source(id, payload):
    """
    Updates a known_source with the given attribute and values as a dict.

    Parameters
    ----------
    id: str or ObjectId
        The id of the known source to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    observation: dict
        The dict of the updated known source
    """
    db = db_utils.connect()
    payload["last_changed"] = dt.datetime.now()
    if isinstance(id, str):
        id = ObjectId(id)
    return db.known_sources.find_one_and_update(
        {"_id": id},
        {"$set": payload},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def get_nearby_known_sources(ra, dec, radius=5.0):
    """
    Gets all known sources from the database near the given (ra, dec) position.

    To find all sources, set radius to infinity.

    Parameters
    ----------
    ra, dec: float
        The Right Ascension and Declination, in degrees, of the sky position
        around which to search for known sources.

    radius: float
        The radius, in degrees, within which all known sources will be
        returned.

    Returns
    -------
    known_sources: list of KnownSource
        The known sources in the database within the desired distance.
    """
    db = db_utils.connect()

    from astropy import units as u
    from astropy.coordinates import angular_separation

    return [
        KnownSource.from_db(source)
        for source in db.known_sources.find()
        if angular_separation(
            ra * u.degree,
            dec * u.degree,
            source["pos_ra_deg"] * u.degree,
            source["pos_dec_deg"] * u.degree,
        )
        .to(u.degree)
        .value
        < radius
    ]


def get_known_source_by_name(name):
    """
    Gets all known sources that have a name.

    Parameters
    ----------
    name: str, or tuple or list of strs
        The name(s) of known sources to seek.
        If name is a str, then returns either one `KnownSource` or None.
        If name is a tuple or list, then returns a list of `KnownSource`s.
        This list may be empty if none of the names are found.

    Returns
    -------
    known_sources: KnownSource, or list of KnownSource
        The known source(s) with the desired name(s).
    """
    # name is a list; caller is seeking multiple names
    if isinstance(name, list) or isinstance(name, tuple):
        accum = []
        for one_name in name:
            ks = get_known_source_by_name(one_name)
            if ks is not None:
                accum.append(ks)
        return accum

    # name is a string; caller is seeking single object
    db = db_utils.connect()
    ks = db.known_sources.find_one({"source_name": name})
    return [KnownSource.from_db(ks)]


def get_known_source_by_names(names):
    """
    Alternative method to get_known_source_by_name which uses a single db query.

    Parameters
    ----------
    name: str or list of strs

    Returns
    -------
    known_sources: list of KnownSource
        The known source(s) with the desired name(s).
    """
    db = db_utils.connect()
    if isinstance(names, str):
        names = [names]
    return [
        KnownSource.from_db(ks)
        for ks in db.known_sources.find({"source_name": {"$in": names}})
    ]


def delete_known_source(ks_id):
    """
    Deletes the known source with the given id.

    Parameters
    ----------
    ks_id: str or ObjectId
        The objectid of the KnownSource to be deleted.
    """
    db = db_utils.connect()
    if isinstance(ks_id, str):
        ks_id = ObjectId(ks_id)
    return db.known_sources.find_one_and_delete({"_id": ks_id})


def get_detections_by_date(datetime):
    """
    Finds all detections of known pulsars on a given date.

    Parameters
    ----------
    datetime: datetime.datetime
        The datetime.datetime object of the date

    Returns
    -------
    list of dictionaries: List of all detection history entries on the given date.
    """
    db = db_utils.connect()
    detecs = []
    for source in db.known_sources.find({"detection_history": {"$exists": True}}):
        for c in source["detection_history"]:
            if c["datetime"].date() == datetime.date():
                c["source_name"] = source["source_name"]
                detecs.append(c)
    return detecs


def create_followup_source(payload):
    """
    Create a `FollowUpSource` instance and inserts it into the database.

    Parameters
    ----------
    payload: dict
        dictionary of attributes for the stack, as expected by `models.FollowUpSource`

    Returns
    -------
    followup_source: sps_database.models.FollowUpSource
        the new instance, including the created MongoDB document id
    """
    db = db_utils.connect()
    followup_source = FollowUpSource(**payload)
    doc = followup_source.to_db()
    del doc["_id"]
    query = {k: doc[k] for k in doc.keys() & {"source_name"}}
    rc = db.followup_sources.find_one_and_replace(
        query, doc, upsert=True, return_document=pymongo.ReturnDocument.AFTER
    )
    followup_source._id = rc["_id"]
    return followup_source._id


def get_followup_source(followup_source_id):
    """
    Find a `FollowUpSource` instance with the given id.

    Parameters
    ----------
    followup_source_id: bson.objectid.ObjectId or string
        key for the id lookup in the "followup_sources" collection,
        as expected by pymongo.collection.Collection.find()

    Returns
    -------
    followup_source: sps_database.models.FollowUpSource
        the followup source instance with the given id

    Exceptions
    ----------
    Raises a runtime exception if the followup source id does not exist
    """
    db = db_utils.connect()
    if isinstance(followup_source_id, str):
        followup_source_id = ObjectId(followup_source_id)
    return FollowUpSource.from_db(db.followup_sources.find_one(followup_source_id))


def update_followup_source(id, payload):
    """
    Updates a followup_source with the given attribute and values as a dict.

    Parameters
    ----------
    id: str or ObjectId
        The id of the followup source to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    followup_source: dict
        The dict of the updated followup source
    """
    db = db_utils.connect()
    payload["last_changed"] = dt.datetime.now()
    if isinstance(id, str):
        id = ObjectId(id)
    return db.followup_sources.find_one_and_update(
        {"_id": id},
        {"$set": payload},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def delete_followup_source(fs_id):
    """
    Deletes the followup source with the given id.

    Parameters
    ----------
    fs_id: str or ObjectId
        The objectid of the FollowUpSource to be deleted.
    """
    db = db_utils.connect()
    if isinstance(fs_id, str):
        fs_id = ObjectId(fs_id)
    return db.followup_sources.find_one_and_delete({"_id": fs_id})


def create_process(payload, assume_new=False):
    """
    Create an `Process` instance and inserts it into the database.

    Parameters
    ----------
    payload: dict
        dictionary of attributes for the process,
        as expected by `models.Process`

    Returns
    -------
    observation: sps_database.models.Process
        the new instance, including the created MongoDB document id
    """
    db = db_utils.connect()
    process = Process(**payload)
    doc = process.to_db()
    del doc["_id"]
    if not assume_new:
        query = {k: doc[k] for k in doc.keys() & {"pointing_id", "datetime"}}

        old_proc = db.processes.find_one(query)
        if old_proc:
            doc["status"] = old_proc["status"]
        rc = db.processes.find_one_and_replace(
            query, doc, upsert=True, return_document=pymongo.ReturnDocument.AFTER
        )
        process._id = rc["_id"]
    else:
        # Not sure if that ever would cause trouble
        inserted = db.processes.insert_one(doc)
        process._id = inserted.inserted_id
    return process


def create_process_from_active_pointing(active_pointing, assume_new=False):
    """
    Create a `Process` instance from an active pointing.

    Parameters
    ----------
    active_pointing: ActivePointing
        Active pointing class object used to form a payload
        to call db_api.create_process

    Returns
    -------
    observation: sps_database.models.Process
        the new instance, including the created MongoDB document id
    """
    datetime = dt.datetime.utcfromtimestamp(
        active_pointing.max_beams[0]["utc_start"]
    ).replace(tzinfo=pytz.UTC)
    date_string = datetime.strftime("%Y/%m/%d")
    if not assume_new:
        obs = get_observation_date(active_pointing.pointing_id, datetime)
        if obs is not None:
            obs_status = obs.status
            quality_label = obs.add_to_stack
            if obs_status.value == 2:
                status = ProcessStatus.complete
            else:
                status = ProcessStatus.scheduled
        else:
            obs_status = ObservationStatus.scheduled
            quality_label = None
            status = ProcessStatus.scheduled
        is_in_stack = obs_in_stack(obs)
    else:
        obs_status = ObservationStatus.scheduled
        quality_label = None
        status = ProcessStatus.scheduled
        is_in_stack = False
    doc = {
        "pointing_id": active_pointing.pointing_id,
        "ra": active_pointing.ra,
        "dec": active_pointing.dec,
        "datetime": datetime,
        "date": date_string,
        "nchan": active_pointing.nchan,
        "ntime": active_pointing.ntime,
        "maxdm": active_pointing.maxdm,
        "status": status,
        "obs_status": obs_status,
        "quality_label": quality_label,
        "is_in_stack": is_in_stack,
        "obs_id": active_pointing.obs_id,
    }
    process = create_process(doc, assume_new=assume_new)
    return process


def get_process(process_id):
    """
    Find an `Process` instance with the given id.

    Parameters
    ----------
    process_id: bson.objectid.ObjectId or string
        key for the id lookup in the "processes" collection,
        as expected by pymongo.collection.Collection.find()

    Returns
    -------
    process: sps_database.models.process
        the process instance with the given id

    Exceptions
    ----------
    Raises a runtime exception if the process id does not exist maybe
    """
    db = db_utils.connect()
    if isinstance(process_id, str):
        process_id = ObjectId(process_id)
    return Process.from_db(db.processes.find_one(process_id))


def update_process(process_id, payload):
    """
    Updates process to the given attribute and values as a dict.

    Parameters
    ----------
    process_id: str or ObjectId
        The id of the process to be updated

    payload: dict
        The dict of the attributes and values to be updated

    Returns
    -------
    observation: dict
        The dict of the updated process entry
    """
    db = db_utils.connect()
    payload["last_changed"] = dt.datetime.now()
    if isinstance(process_id, str):
        process_id = ObjectId(process_id)
    return db.processes.find_one_and_update(
        {"_id": process_id},
        {"$set": payload},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def dequeue_process(process_id):
    """
    Finds scheduled process, changes its status to blocked, and returns it.

    Parameters
    ----------
    process_id: str or ObjectId
        The id of the process to be updated

    Returns
    -------
    process: dict
        The dict of the updated process entry
    """
    db = db_utils.connect()
    if isinstance(process_id, str):
        process_id = ObjectId(process_id)
    return db.processes.find_one_and_update(
        {"_id": process_id, "status": ProcessStatus.scheduled},
        {"$set": ProcessStatus.blocked},
        return_document=pymongo.ReturnDocument.AFTER,
    )


def get_process_from_active_pointing(active_pointing, assume_new=False):
    """
    Find an `Process` instance based on an active pointing.

    Parameters
    ----------
    active_pointing: ActivePointing
        The active_pointing which the process should correspond to

    Returns
    -------
    process: sps_database.models.process
        the process instance based on the given active_pointing
    """
    db = db_utils.connect()
    datetime = dt.datetime.utcfromtimestamp(
        active_pointing.max_beams[0]["utc_start"]
    ).replace(tzinfo=pytz.UTC)
    if not assume_new:
        query = {
            "datetime": {"$gte": datetime, "$lte": datetime + dt.timedelta(days=1)},
            "pointing_id": active_pointing.pointing_id,
        }
        process = db.processes.find_one(query)
    else:
        process = None
    if process:
        if active_pointing.obs_id and process.id:
            process = update_process(process.id, {"obs_id": active_pointing.obs_id})
    else:
        process = create_process_from_active_pointing(
            active_pointing, assume_new=assume_new
        )
    return process


def obs_in_stack(observation):
    """
    Check if Observation is contained in the monthly or cumulative stack.

    Parameters
    ----------
    observation: Observation
        The Observation that gets tested

    Returns
    -------
    is_in_stack: boolean
        Truth value if the Observation is in the stack
    """
    if observation is None:
        return False
    else:
        ps_stack = get_ps_stack(observation.pointing_id)
        is_in_stack = False
        if ps_stack is not None:
            if observation.datetime in ps_stack.datetimes_month:
                is_in_stack = True
            if observation.datetime in ps_stack.datetimes_cumul:
                is_in_stack = True
        return is_in_stack


def get_dates(obs_id_list):
    """
    Get dates for multiple observations.

    Get sorted datetimes for a list of obs_ids. May fail if the not connected to the
    correct database.

    Parameters
    ----------
    obs_id_list: list(obs_id)
        The list of obsevrations ids.

    Returns
    -------
    sorted_dates: list(datetime.datetime)
        The datetimes of the observations.
    """
    dates = []
    for obs_id in obs_id_list:
        cand = get_observation(obs_id)
        dates.append(cand.datetime)
    sorted_dates = sorted(dates)
    return sorted_dates


def update_pulsars_in_pointing(pointing_id, pulsar_name, pulsar_dict, stack=False):
    """
    Updates the strongest_pulsar_detections in a pointing.

    Parameters
    ----------
    pointing_id: str or ObjectId
        The id of the pointing to be updated

    pulsar_name: str
        TName of the pulsar as given in the source_name in the known sources

    pulsar_dict: dict
        Dict of the new pulsar

    Returns
    -------
    pointing: dict
        The dict of the updated pointing entry
    """
    if stack:
        attribute_to_update = "strongest_pulsar_detections_stack"
    else:
        attribute_to_update = "strongest_pulsar_detections"
    initial_pointing = get_pointing(pointing_id)
    new_pulsar_dict = {pulsar_name: pulsar_dict}
    if getattr(initial_pointing, attribute_to_update, {}) == {}:
        payload = {attribute_to_update: new_pulsar_dict}
        updated_pointing = update_pointing(pointing_id, payload)
    else:
        payload = {
            attribute_to_update: getattr(initial_pointing, attribute_to_update, {})
        }
        old_pulsar_info = payload[attribute_to_update].get(pulsar_name, {})
        if old_pulsar_info == {}:
            payload[attribute_to_update][pulsar_name] = pulsar_dict
            updated_pointing = update_pointing(pointing_id, payload)
        else:
            old_max = payload[attribute_to_update][pulsar_name].get("sigma", 0)
            new_max = pulsar_dict.get("sigma", 0)
            if new_max > old_max:
                payload[attribute_to_update][pulsar_name] = pulsar_dict
                updated_pointing = update_pointing(pointing_id, payload)
            else:
                updated_pointing = initial_pointing
    return updated_pointing


def update_daily_run(date, payload, upsert=True):
    """
    Updates a power spectra stack entry of a given pointing.

    The PS stack entry is updated with the given attribute
    and values as a dict.

    Parameters
    -------
    date: datetime.datetime
        The date that should be updated

    payload: dict
        The dict of the attributes and values to be updated

    upsert: bool
        If daily run object should eb created if not pre-existing.
        For now always create, to simplify.

    Returns
    -------
    DailyRun: dict
        The dict of the updated DailyRun entry
    """
    db = db_utils.connect()
    payload["last_changed"] = dt.datetime.now()
    return db.daily_runs.find_one_and_update(
        {"date": date.replace(tzinfo=None)},
        {"$set": payload},
        return_document=pymongo.ReturnDocument.AFTER,
        upsert=upsert,
    )


def get_daily_run(date):
    """
    Updates a power spectra stack entry of a given pointing.

    The PS stack entry is updated with the given attribute
    and values as a dict.

    Parameters
    -------
    date: datetime.datetime
        The date that should be updated

    Returns
    -------
    DailyRun: DailyRun
        The DailyRun object for the database entry
    """
    db = db_utils.connect()
    return DailyRun.from_db(db.daily_runs.find_one({"date": date.replace(tzinfo=None)}))
