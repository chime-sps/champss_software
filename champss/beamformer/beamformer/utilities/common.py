import datetime
import json
import logging
from glob import glob
import importlib
import sys

import ephem
import numpy as np
import pytz
from astropy import units as u
from astropy.coordinates import SkyCoord
from beamformer import CURRENT_POINTING_MAP, NoSuchPointingError
from sps_common import constants
from sps_databases import db_utils, models

if importlib.util.find_spec("cfbm"):
    import cfbm as beam_model

log = logging.getLogger(__name__)

if "cfbm" in sys.modules:
    config = beam_model.current_config
    try:
        beammod = beam_model.current_model_class(config)
    except FileNotFoundError:
        log.error("Missing files for beam model. Will try to load them.")
        from cfbm.bm_data import get_data

        get_data.main()
        beammod = beam_model.current_model_class(config)

    beam_transit = beam_model.config.chime

_storage_path = constants.STORAGE_PATH


def find_next_transit(ra: float, dec: float, ref_time: datetime.datetime) -> float:
    """
    Find the next transit time for a given ra, dec and reference time.

    Args:
        ra (float): Right Ascension of the sky position in degrees
        dec (float): Declination of the sky position in degrees
        ref_time (datetime.datetime): Datetime object of the a given time

    Returns:
        float: Transit time in utc in unix format
    """
    if "cfbm" not in sys.modules:
        sys.exit(
            "ImportError: Using find_next_transit() function requires FRB"
            " beam_model. Please install cfbm if you plan to use this function."
        )
    coord = ephem.Equatorial(ephem.degrees(str(ra)), ephem.degrees(str(dec)))
    body = ephem.FixedBody()
    body._ra = coord.ra
    body._dec = coord.dec
    beam_transit.date = ephem.Date(ref_time)
    transit_time_dt = beam_transit.next_transit(body).datetime()
    transit_time = transit_time_dt.replace(tzinfo=pytz.utc).timestamp()

    return transit_time


def find_closest_pointing(ra, dec, mode="database"):
    coord = SkyCoord(ra * u.deg, dec * u.deg)
    ra_max = ra + (1 / np.cos(np.pi * dec / 180))
    if ra_max > 360:
        ra_max -= 360
    ra_min = ra - (1 / np.cos(np.pi * dec / 180))
    if ra_min <= 0:
        ra_min += 360
    if mode == "database":
        db = db_utils.connect()
        if ra_min > ra_max:
            pointings = list(
                db.pointings.find(
                    {
                        "$or": [
                            {
                                "ra": {
                                    "$gt": ra_min,
                                }
                            },
                            {
                                "ra": {
                                    "$lt": ra_max,
                                }
                            },
                        ],
                        "dec": {"$gt": dec - 1, "$lt": dec + 1},
                    }
                )
            )
        else:
            pointings = list(
                db.pointings.find(
                    {
                        "ra": {
                            "$gt": ra_min,
                            "$lt": ra_max,
                        },
                        "dec": {"$gt": dec - 1, "$lt": dec + 1},
                    }
                )
            )
    elif mode == "local":
        pointings = []
        with open(CURRENT_POINTING_MAP) as infile:
            full_pointings = json.load(infile)
        for p in full_pointings:
            if ra_min > ra_max:
                if p["ra"] > ra_min or p["ra"] < ra_max:
                    if dec - 1.0 < p["dec"] < dec + 1.0:
                        pointings.append(p)
            else:
                if ra_min < p["ra"] < ra_max:
                    if dec - 1.0 < p["dec"] < dec + 1.0:
                        pointings.append(p)
    else:
        raise ValueError(
            "unrecognized mode. Available modes are 'database' and 'local' "
        )
    if not pointings:
        raise NoSuchPointingError(
            f"No pointings within one degree of target ra, dec: ({ra:.2f}, {dec:.2f}"
        )

    p_coords = np.zeros((len(pointings), 2))
    for i, p in enumerate(pointings):
        ra, dec = p["ra"], p["dec"]
        p_coords[i, :] = [ra, dec]
    catalogue = SkyCoord(ra=p_coords[:, 0] * u.degree, dec=p_coords[:, 1] * u.degree)
    pointing = pointings[np.argmin(catalogue.separation(coord))]

    return models.Pointing.from_db(pointing)


def get_data_list(max_beams, basepath, extn, per_beam_list=False, max_file_length=120):
    """
    Search for the SPS intensity data corresponding to the required FRB beams Data path
    for CHIME/SPS is always in the format:
    YYYY/MM/DD/BEAM_NUMBER/STARTCTIME_STOPCTIME.extn Requires get_max_beam_list to be
    computed first.

    Parameters
    =======
    max_beams: List(dict)
        The list of dictionary of set of beams and their transit time for a giving active pointing. The individual
        dict entry of the list must have the keys {'beam', 'utc_start', 'utc_end'}

    basepath: str
        The base path to the location of the intensity data, i.e. the path before the sub directory
        YYYY/MM/DD/BEAM/

    extn: str
        The file extension of the intensity data to look for in either 'msg', 'msgpack', 'dat' or 'hdf5'

    max_file_length: int
        Some raw files are malformed and will have time stamps that are farther apart than expected.
        This allows filtering out all files that are too supposedly too long. Default: 1200 s
    """
    if extn not in ["msg", "msgpack", "dat", "hdf5"]:
        raise ValueError(
            "The input extension is not valid, the valid options are 'msg', 'msgpack',"
            " 'dat', or 'hdf5'"
        )
    data_list = []
    for beam in max_beams:
        paths = []
        beam_data_list = []
        start_date = datetime.datetime.utcfromtimestamp(beam["utc_start"]).date()
        end_date = datetime.datetime.utcfromtimestamp(beam["utc_end"]).date()
        paths.append(
            "{}/{}/{}/{}/{}/".format(
                basepath,
                str(start_date.year).zfill(4),
                str(start_date.month).zfill(2),
                str(start_date.day).zfill(2),
                str(beam["beam"]).zfill(4),
            )
        )
        if end_date > start_date:
            paths.append(
                "{}/{}/{}/{}/{}/".format(
                    basepath,
                    str(end_date.year).zfill(4),
                    str(end_date.month).zfill(2),
                    str(end_date.day).zfill(2),
                    str(beam["beam"]).zfill(4),
                )
            )
        for path in paths:
            datapacks = sorted(glob(path + f"/*.{extn}"))
            for data in datapacks:
                try:
                    data_start = int(data.split("/")[-1].split("_")[0])
                    data_end = int(data.split("/")[-1].split("_")[1].split(".")[0]) + 1
                except ValueError as error:
                    print(
                        "The prefix of the intensity file is not in"
                        " 'STARTCTIME_ENDCTIME' format"
                    )
                    print(error)
                    continue
                if (data_end - data_start) < max_file_length:
                    if data_start <= beam["utc_end"] and data_end >= beam["utc_start"]:
                        if per_beam_list:
                            beam_data_list.append(data)
                        else:
                            data_list.append(data)
        if per_beam_list:
            data_list.append(beam_data_list)

    return data_list


def get_all_data_list(active_pointings, datapath=_storage_path, extn=".dat"):
    """
    Get the list of data required the all active pointings in a given format.

    Parameters
    =======
    active_pointings: list(ActivePointing)
        A list of ActivePointing class instances

    storage_path: str
        The base path of where the intensity files are stored. (Default : path determined in sps-common)

    extn: str
        The extension of the data files in either 'msg', 'msgpack', 'dat' or 'hdf5'

    Returns
    =======

    raw_data_list: list
        The list of raw data used by the set of active pointings in the format given
    """
    raw_data_list = []
    for p in active_pointings:
        raw_data_list.extend(
            get_data_list(max_beams=p.max_beams, basepath=datapath, extn=extn)
        )
    return sorted(list(set(raw_data_list)))


def get_intensity_set(data_list, min_set_length=1024, tsamp=0.00098304):
    """
    Split a list of intensity data into subsets based on the minimum length of each
    subset.

    Parameters
    =======
    data_list: List(str)
        A list of location of the intensity data with <unix_start_time>_<unix_end_time>.<format>.
        The script assumes the list is sorted by time.

    min_set_length: int
        The minimum length of each subset of intensity data in number of time samples. Default = 1024.

    tsamp: float
        The sampling time of the intensity data in seconds. Default = 0.00098304 s

    Returns
    =======
    split_list: List(List(str))
         A nested list where each subset of data is grouped into.
    """
    list_end_time = int(data_list[-1].split("/")[-1].split("_")[1].split(".")[0])
    min_batch_time = min_set_length * tsamp
    full_list = []
    part_list = []
    for data in data_list:
        start_time = int(data.split("/")[-1].split("_")[0])
        if not part_list:
            part_list.append(data)
            part_start_time = start_time
            continue
        else:
            if start_time - part_start_time < min_batch_time:
                part_list.append(data)
                if data == data_list[-1]:
                    full_list.append(part_list)
            elif list_end_time - start_time < min_batch_time:
                part_list.append(data)
                if data == data_list[-1]:
                    full_list.append(part_list)
            else:
                full_list.append(part_list)
                part_list = [data]
                part_start_time = start_time
    return full_list
