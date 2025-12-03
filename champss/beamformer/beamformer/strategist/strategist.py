import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import attr
import numpy as np
import pytz
import sps_databases.models as spsdb_models
from beamformer import CURRENT_POINTING_MAP
from beamformer.utilities.common import (
    beammod,
    find_closest_pointing,
    find_next_transit,
)
from sps_common import constants
from sps_common.interfaces.beamformer import ActivePointing
from sps_databases import db_api, db_utils, models

_sidereal_s = constants.SIDEREAL_S
_storage_path = constants.STORAGE_PATH
_tsamp = constants.TSAMP


@attr.s(slots=True)
class PointingStrategist:
    """
    PointingStrategist class to determine the properties of a pointings to be processed
    between the start and end utc time for given sets of FRB beam rows.

    Parameters
    =======
    from_db: bool
        Whether to read the current pointing list from sps-database. Default: True

    create_db: bool
        Whether to create a database entry for each active pointing determined. Default: True

    basepath: str
        The base path to the location to process the pointings. Default: './'

    split_long_pointing: bool
        Whether to split long pointings into smaller sub pointings to be processed independently.
        Default: False

    max_length: int
        Max length of a pointing in number of samples to be processed.
        Default: 2*22 (65.5 minutes, highest factor of 2 which easily fits in memory)
    """

    from_db = attr.ib(default=True, validator=attr.validators.instance_of(bool))
    create_db = attr.ib(default=True, validator=attr.validators.instance_of(bool))
    basepath = attr.ib(default="./", validator=attr.validators.instance_of(str))
    split_long_pointing = attr.ib(
        default=False, validator=attr.validators.instance_of(bool)
    )
    max_length = attr.ib(default=2**22, validator=attr.validators.instance_of(int))
    _mapper = attr.ib(init=False, repr=False)

    def __attrs_post_init__(self):
        if not self.from_db:
            self.create_db = False
        # Initialize DM mapper for arbitrary pointing calculations
        from beamformer.strategist.mapper import PointingMapper
        self._mapper = PointingMapper()

    def get_pointings(self, utc_start, utc_end, beam_row):
        """
        Get the pointings to process between utc start and utc end along with their
        properties. The outputs will be served as input to SkyBeamFormer.form_skybeam.

        Parameters
        =======

        utc_start: float
            The UTC start time of the set of pointings to be processed

        utc_end: float
            The UTC end time of the set of pointings to be processed

        beam_row: np.ndarray
            The FRB beam rows of the pointings to be processed in a 1-D numpy array

        Returns
        =======

        active_pointings: list
            The list of active pointings as class instances with ra, dec, nchan, ntime, maxdm, max_beams, beam_row and obs_id
        """
        if self.from_db:
            active_pointings = self.pointings_from_db(utc_start, utc_end, beam_row)
        else:
            active_pointings = self.pointings_from_map(utc_start, utc_end, beam_row)
        return active_pointings

    def get_single_pointing(self, ra, dec, date, use_grid=True):
        """
        Get the single pointing to process for the given date along with their
        properties. The output will be served as input to SkyBeamFormer.form_skybeam.

        Parameters
        =======

        ra: float
            The right ascension of the target location

        dec: float
            The declination of the target location

        date: datetime
            The datetime object of the date to be processed

        use_grid: bool
            Whether to snap coordinates to the pointing map grid (default: True).
            If False, creates a pointing at exact RA/Dec coordinates.

        Returns
        =======

        active_pointing: ActivePointing
            The active pointing as class instances with ra, dec, nchan, ntime, maxdm, max_beams, beam_row and obs_id

        Raises
        ------
        NoSuchPointingError
            Raised if requested coordinates are not found in the pointing map.
        """
        if use_grid:
            if self.from_db:
                mode = "database"
            else:
                mode = "local"
            pointing = find_closest_pointing(ra, dec, mode=mode)
        else:
            # Create pointing at exact coordinates without grid snapping
            pointing = self._create_arbitrary_pointing(ra, dec)
        return self.active_pointing_from_pointing(pointing, date)

    def active_pointing_from_pointing(self, pointing, date):
        """
        Get an ActivePointing instance from a Pointing dictionary obtained from the
        pointing map.

        Parameters
        ==========

        pointing: Pointing
            A Pointing of the properties of a pointing as obtained from a pointing map

        date: datetime
            A datetime object of the given date.

        Returns
        =======

        active_pointing: ActivePointing
            The active pointing as class instances with ra, dec, nchan, ntime, maxdm, max_beams, beam_row and obs_id
        """
        date_utc = date.replace(tzinfo=pytz.utc).timestamp()
        max_beams = self.get_max_beam_list(
            pointing.ra,
            pointing.dec,
            pointing.length,
            pointing.beam_row,
            date_utc,
        )
        if self.split_long_pointing and pointing.length > self.max_length:
            active_pointings = []
            split_length, split_max_beams = self.split_pointing(
                pointing.length, max_beams
            )
            for i in range(len(split_max_beams)):
                active_pointings.append(
                    ActivePointing(
                        ra=pointing.ra,
                        dec=pointing.dec,
                        nchan=pointing.nchans,
                        ntime=split_length,
                        maxdm=pointing.maxdm,
                        max_beams=split_max_beams[i],
                        beam_row=pointing.beam_row,
                        sub_pointing=i,
                        pointing_id=getattr(pointing, '_id', None),
                    )
                )
        else:
            active_pointings = [
                ActivePointing(
                    ra=pointing.ra,
                    dec=pointing.dec,
                    nchan=pointing.nchans,
                    ntime=pointing.length,
                    maxdm=pointing.maxdm,
                    max_beams=max_beams,
                    beam_row=pointing.beam_row,
                    pointing_id=getattr(pointing, '_id', None),
                )
            ]
        if self.create_db:
            for i, p in enumerate(active_pointings):
                active_pointings[i] = self.create_database_entry(p)
        return active_pointings

    def _find_beam_row_for_dec(self, dec):
        """
        Find the FRB beam row whose declination is closest to the target declination.
        Uses upper transit only (beam_row < 224) to avoid degeneracy for circumpolar sources.

        Parameters
        ==========
        dec: float
            Target declination in degrees

        Returns
        =======
        beam_row: int
            The beam row (0-223) with the closest declination to the target
        """
        from datetime import datetime
        # Use a reference time to get equatorial coordinates for each beam_row
        ref_time = datetime(2021, 3, 20, 20, 5, 17)
        decs = []
        # Only check upper transit beam_rows (0-223) to avoid circumpolar degeneracy
        for i in range(224):
            ra_beam, dec_beam = beammod.get_equatorial_from_position(
                            0, beammod.reference_angles[i], ref_time
                            )
            decs.append(dec_beam)
        decs = np.array(decs)

        return int(np.argmin(np.abs(decs - dec)))

    def _create_arbitrary_pointing(self, ra, dec):
        """
        Create a Pointing object at arbitrary RA/Dec coordinates.

        Parameters
        ==========
        ra: float
            Right ascension in degrees
        dec: float
            Declination in degrees

        Returns
        =======
        pointing: Pointing (from sps_common.interfaces.beamformer)
            A Pointing object with calculated properties for the exact coordinates
        """
        from sps_common.interfaces.beamformer import Pointing

        # Calculate standard pointing attributes
        beam_row = self._find_beam_row_for_dec(dec)
        ne2001_dm, ymw16_dm = self._mapper.get_ne2001_ymw16(ra, dec)
        maxdm = self._mapper.get_max_dm(
            ra, dec, ne2001_dm, ymw16_dm,
            exp=self._mapper.exp,
            excess=self._mapper.excess,
            excess_fac=self._mapper.excess_fac,
            extragalactic=self._mapper.extragalactic
        )
        nchans = self._mapper.get_nchans(maxdm)

        # Get the pointing length for this beam row
        beam_index = np.where(self._mapper.beams == beam_row)[0]
        length = self._mapper.get_length(beam_index[0])

        pointing = Pointing(
            ra=ra,
            dec=dec,
            beam_row=beam_row,
            length=length,
            ne2001dm=ne2001_dm,
            ymw16dm=ymw16_dm,
            maxdm=maxdm,
            nchans=nchans
        )

        return pointing

    def pointings_from_map(self, utc_start, utc_end, beam_row):
        """
        Obtain the list of active pointings to process from the local database.

        Parameters
        =======

        utc_start: float
            The UTC start time of the set of pointings to be processed

        utc_end: float
            The UTC end time of the set of pointings to be processed

        beam_row: np.ndarray
            The FRB beam rows of the pointings to be processed in a 1-D numpy array

        Returns
        =======

        pointings_to_process: list(ActivePointing)
            The list of active pointings to process as class instances with ra, dec, nchan, ntime, maxdm, max_beams, beam_row and obs_id
        """
        with open(CURRENT_POINTING_MAP) as infile:
            pointings = json.load(infile)
        pointings_to_process = []
        for b in beam_row:
            ra_start, dec_start = beammod.get_equatorial_from_position(
                0,
                beammod.reference_angles[b],
                datetime.utcfromtimestamp(utc_start),
            )
            ra_end, dec_end = beammod.get_equatorial_from_position(
                0,
                beammod.reference_angles[b],
                datetime.utcfromtimestamp(utc_end),
            )
            for p in pointings:
                if p["beam_row"] == b:
                    if (
                        (utc_end - utc_start > _sidereal_s - 160 and ra_end > ra_start)
                        or (ra_end > ra_start and ra_start < p["ra"] < ra_end)
                        or (
                            ra_start > ra_end
                            and (p["ra"] > ra_start or p["ra"] < ra_end)
                        )
                    ):
                        p["max_beams"] = self.get_max_beam_list(
                            p["ra"], p["dec"], p["length"], p["beam_row"], utc_start
                        )
                        if p["max_beams"][0]["utc_start"] < utc_end:
                            if (
                                self.split_long_pointing
                                and p["length"] > self.max_length
                            ):
                                split_length, split_max_beams = self.split_pointing(
                                    p["length"], p["max_beams"]
                                )
                                for i in range(len(split_max_beams)):
                                    pointings_to_process.append(
                                        ActivePointing(
                                            ra=p["ra"],
                                            dec=p["dec"],
                                            nchan=p["nchans"],
                                            ntime=split_length,
                                            maxdm=p["maxdm"],
                                            max_beams=split_max_beams[i],
                                            beam_row=p["beam_row"],
                                            sub_pointing=i,
                                        )
                                    )
                            else:
                                pointings_to_process.append(
                                    ActivePointing(
                                        ra=p["ra"],
                                        dec=p["dec"],
                                        nchan=p["nchans"],
                                        ntime=p["length"],
                                        maxdm=p["maxdm"],
                                        max_beams=p["max_beams"],
                                        beam_row=p["beam_row"],
                                    )
                                )
        return pointings_to_process

    def pointings_from_db(self, utc_start, utc_end, beam_row):
        """
        Obtain the list of active pointings to process from sps-databases.

        Parameters
        =======
        utc_start: float
            The UTC start time of the set of pointings to be processed

        utc_end: float
            The UTC end time of the set of pointings to be processed

        beam_row: np.ndarray
            The FRB beam rows of the pointings to be processed in a 1-D numpy array

        Returns
        =======
        pointings_to_process: list(ActivePointing)
            The list of active pointings to process as class instances with ra, dec, nchan, ntime, maxdm, max_beams, beam_row and obs_id
        """
        db = db_utils.connect()
        pointings_to_process = []
        for b in beam_row:
            ra_start, dec_start = beammod.get_equatorial_from_position(
                0,
                beammod.reference_angles[b],
                datetime.utcfromtimestamp(utc_start),
            )
            ra_end, dec_end = beammod.get_equatorial_from_position(
                0,
                beammod.reference_angles[b],
                datetime.utcfromtimestamp(utc_end),
            )
            if utc_end - utc_start > _sidereal_s - 160 and ra_start < ra_end:
                qs = db.pointings.find({"beam_row": int(b)})
            elif ra_start < ra_end:
                # ra_start < ra < ra_end
                # fmt: off
                qs = db.pointings.find({
                    "beam_row": int(b),
                    "ra": {"$gt": ra_start, "$lt": ra_end}
                })
                # fmt: on
            else:
                # ra < ra_end || ra > ra_start
                # fmt: off
                qs = db.pointings.find({"$and": [
                    {"beam_row": int(b)},
                    {"$or": [{"ra": {"$gt": ra_start}}, {"ra": {"$lt": ra_end}}]}
                ]})
                # fmt: on
            for p in qs:
                # p["pointing_id"] = p.pop("_id", None)
                # p.pop("search_algorithm", None)
                pointing = models.Pointing.from_db(p)
                max_beams = self.get_max_beam_list(
                    pointing.ra,
                    pointing.dec,
                    pointing.length,
                    pointing.beam_row,
                    utc_start,
                )

                if max_beams[0]["utc_start"] < utc_end:
                    if self.split_long_pointing and pointing.length > self.max_length:
                        split_length, split_max_beams = self.split_pointing(
                            pointing.length, max_beams
                        )
                        for i in range(len(split_max_beams)):
                            active_pointing = ActivePointing(
                                ra=pointing.ra,
                                dec=pointing.dec,
                                nchan=pointing.nchans,
                                ntime=split_length,
                                maxdm=pointing.maxdm,
                                max_beams=split_max_beams[i],
                                beam_row=pointing.beam_row,
                                sub_pointing=i,
                                pointing_id=pointing._id,
                            )
                            if self.create_db:
                                active_pointing = self.create_database_entry(
                                    active_pointing
                                )
                            pointings_to_process.append(active_pointing)
                    else:
                        active_pointing = ActivePointing(
                            ra=pointing.ra,
                            dec=pointing.dec,
                            nchan=pointing.nchans,
                            ntime=pointing.length,
                            maxdm=pointing.maxdm,
                            max_beams=max_beams,
                            beam_row=pointing.beam_row,
                            pointing_id=pointing._id,
                        )
                        if self.create_db:
                            active_pointing = self.create_database_entry(
                                active_pointing
                            )
                        pointings_to_process.append(active_pointing)

        return pointings_to_process

    def split_pointing(self, ntime, max_beams):
        """
        Split a pointing over the maximum allowed length into subsets that are smaller.

        Parameters
        =======
        ntime: int
            Length of the pointing to split in number of samples

        max_beams: List(dict)
            A list of dictionary defining the FRB beams in transit for the pointing to split

        Returns
        =======
        split_ntime: int
            Length of a sub pointing that is split.

        full_beam_set: List(List(dict))
            List of the max beams of all the sub pointings post splitting.
        """
        split_factors = np.array([2, 4, 5, 8, 10, 20, 40])
        if ntime / 40 > self.max_length:
            for i in range(ntime // self.max_length + 1, ntime // 40960):
                if ntime % i == 0:
                    split_fac = i
                    break
        else:
            split_fac = split_factors[
                np.min(np.where(ntime / split_factors < self.max_length))
            ]
        split_ntime = ntime // split_fac
        full_beam_set = []
        pointing_utc_start = max_beams[0]["utc_start"]
        for i in range(split_fac):
            beam_set = []
            start = pointing_utc_start + (i * split_ntime * _tsamp)
            end = pointing_utc_start + ((i + 1) * split_ntime * _tsamp)
            for beam in max_beams:
                if beam["utc_start"] <= start and beam["utc_end"] >= end:
                    beam_set.append(
                        {"beam": beam["beam"], "utc_start": start, "utc_end": end + 5}
                    )
                    break
                if beam["utc_start"] <= start <= beam["utc_end"] <= end:
                    beam_set.append(
                        {
                            "beam": beam["beam"],
                            "utc_start": start,
                            "utc_end": beam["utc_end"],
                        }
                    )
                elif start <= beam["utc_start"] <= end <= beam["utc_end"]:
                    beam_set.append(
                        {
                            "beam": beam["beam"],
                            "utc_start": beam["utc_start"],
                            "utc_end": end + 5,
                        }
                    )
                elif beam["utc_start"] >= start and beam["utc_end"] <= end:
                    beam_set.append(
                        {
                            "beam": beam["beam"],
                            "utc_start": beam["utc_start"],
                            "utc_end": beam["utc_end"],
                        }
                    )
            full_beam_set.append(beam_set)
        return split_ntime, full_beam_set

    def get_max_beam_list(self, ra, dec, ntime, beam_row, ref_time):
        """
        Function to compute the FRB beam used to beamform the pointing across the
        transit. The output is saved in max_beam. The FRB beam to used is computed on 10
        second intervals.

        Parameters
        =======
        ra: float
            The right acsencion of the pointing to process

        dec: float
            The declination of the pointing to process

        ntime: int
            The length of the pointing to process in number of time samples

        beam_row: int
            The beam row of the pointing to process

        ref_time: float
            The reference time in unix utc time to determine the next transit time of the pointing to process

        Returns
        =======
        max_beams: List(dict)
            A list of dictionary showing the FRB beam in transit, along with their start and end time.
        """
        max_beams = []
        transit_time = find_next_transit(ra, dec, datetime.utcfromtimestamp(ref_time))
        utc_start = transit_time - ntime * _tsamp / 2
        utc_end = transit_time + (ntime * _tsamp * 0.75)
        beam_arr = np.arange(beam_row, beam_row + 4000, 1000)
        if 214 <= beam_row <= 222:
            beam_arr = np.concatenate(
                (beam_arr, np.arange(beam_row + 1, beam_row + 4001, 1000))
            )
        if beam_row == 221:
            beam_arr = np.concatenate(
                (beam_arr, np.arange(beam_row + 2, beam_row + 4002, 1000))
            )
        max_beam_now = -1
        for time in np.arange(int(utc_start), int(utc_end), 10):
            ref_pos = beammod.get_position_from_equatorial(
                ra, dec, datetime.utcfromtimestamp(time + 5.0)
            )
            if -0.6 < ref_pos[0] < 1.0 and ref_pos[1] < 40.68:
                ref_pos = np.asarray([ref_pos])
                ref_freq = np.asarray([480.5])
                max_beam_arg = beammod.get_sensitivity(
                    beam_arr, ref_pos, ref_freq
                ).argmax()
                max_beam = beam_arr[max_beam_arg]
                if max_beam != max_beam_now:
                    if len(max_beams) > 0:
                        max_beams[-1]["utc_end"] = time
                    max_beams.append({"beam": max_beam, "utc_start": time})
                    max_beam_now = max_beam
            else:
                if len(max_beams) > 0 and "utc_end" not in max_beams[-1]:
                    max_beams[-1]["utc_end"] = time

        return max_beams

    def create_database_entry(self, active_pointing: ActivePointing) -> ActivePointing:
        """
        Creates an observation entry in the sps-database for a given active pointing.

        Parameter
        =======
        active_pointing: ActivePointing
            The ActivePointing instance to enter into the database

        Returns
        =======
        active_pointing: ActivePointing
            The ActivePointing object with updated obs_id.
        """
        p_start_time = active_pointing.max_beams[0]["utc_start"]
        p_start_time_dt = datetime.utcfromtimestamp(p_start_time).replace(
            tzinfo=pytz.utc
        )
        observation = db_api.create_observation(
            {
                "pointing_id": active_pointing.pointing_id,
                "datapath": "{}/{}/{}/{}/{:.02f}_{:.02f}/".format(
                    self.basepath,
                    str(p_start_time_dt.year).zfill(4),
                    str(p_start_time_dt.month).zfill(2),
                    str(p_start_time_dt.day).zfill(2),
                    active_pointing.ra,
                    active_pointing.dec,
                ),
                "status": spsdb_models.ObservationStatus.scheduled,
                "datetime": p_start_time_dt,
            }
        )
        active_pointing.obs_id = observation._id
        return active_pointing


def get_single_active_pointing(
    ra: float,
    dec: float,
    utc_date: float,
    from_db: bool = True,
    create_db: bool = True,
    basepath: str = "./",
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Get the single active pointing's properties give its location and date.

    Parameters
    ==========
    from_db: bool
        Whether to read the current pointing list from sps-database. Default: True

    create_db: bool
        Whether to create a database entry for each active pointing determined. Default: True

    basepath: str
        The base path to the location to process the pointings. Default: './'

    ra: float
        The right ascension of the target location

    dec: float
        The declination of the target location

    utc_date: float
        The POSIX UTC time in seconds of the date desired

    Returns
    =======
    results: Tuple[Dict[str, Any], List[str], List[str]]
        The results of calling the PointStrategist's get_single_pointing method, containing
        information about the retrieved active pointing
    """
    pointing_strategist = PointingStrategist(
        from_db=from_db, create_db=create_db, basepath=basepath
    )
    date = datetime.utcfromtimestamp(utc_date)
    active_pointing = pointing_strategist.get_single_pointing(ra=ra, dec=dec, date=date)

    return (
        {
            "ra": active_pointing.ra,
            "dec": active_pointing.dec,
            "nchan": active_pointing.nchan,
            "ntime": active_pointing.ntime,
            "maxdm": active_pointing.maxdm,
            "max_beams": active_pointing.max_beams,
            "beam_row": active_pointing.beam_row,
            "sub_pointing": active_pointing.sub_pointing,
            "pointing_id": active_pointing.pointing_id,
        },
        [],
        [],
    )
