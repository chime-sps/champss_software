#!/usr/bin/env python

import attr
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from beamformer import PATH

"""
Note:
    map_ymw16.npy & map_ne2001.npy were copied from CHIME/FRB frb-l2l3 Repo:
    https://github.com/CHIMEFRB/frb-l2l3/tree/master/config/data/dm_checker
    on date: May 28, 2020
"""


@attr.s
class DMMap:
    map_ymw16 = np.load(PATH + "/data/YMW16_map.npy").T
    map_ne2001 = np.load(PATH + "/data/NE2001_map.npy").T
    interp_map_ymw16 = LinearNDInterpolator(map_ymw16[:2].T, map_ymw16[2].T)
    interp_map_ne2001 = LinearNDInterpolator(map_ne2001[:2].T, map_ne2001[2].T)

    def get_dm_ne2001(self, latitude: float, longitude: float) -> np.ndarray:
        """
        Uses a map of DM values generate by the NE2001 software to obtain maximum Galactic
        component of DM, given a position of the source.

        Parameters
        ----------

        latitude : float
            Declination, in units of degrees.

        longitude : float
            Right ascension, in units of degrees.

        Returns
        -------

        dm_ne2001
            Galactic DM predicted by NE2001, in units of pc cm^-3.


        Notes
        -----
        The NE2001 map is stored in a numpy file (NE2001_map.npy) that contains a non-uniform
        grid of right ascension, declination, and the corresponding maximum DM. The grid was
        first generated over a grid of Galactic coordinates that sample more points close to the
        Galactic plane (where DM varies more strongly for small changes in position), and fewer
        points farther way from the plane.
        """
        return self.interp_map_ne2001(longitude, latitude)

    def get_dm_ymw16(self, latitude: float, longitude: float) -> np.ndarray:
        """
        Uses a map of DM values generate by the YMW16 software to obtain maximum Galactic
        component of DM, given a position of the source.

        Parameters
        ----------

        latitude : float
            Declination, in units of degrees.

        longitude : float
            Right ascension, in units of degrees.

        Returns
        -------

        dm_ymw16
            Galactic DM predicted by YMW16, in units of pc cm^-3.


        Notes
        -----
        The YMW16 map is stored in a numpy file (YMW16_map.npy) that contains a non-uniform
        grid of right ascension, declination, and the corresponding maximum DM. The YMW16 grid
        was first generated using the same set of coordinates used to compute the NE2001 map.
        """

        # interpolate map, and used interpolated result to find approximate DM at header coordinates.
        return self.interp_map_ymw16(longitude, latitude)
