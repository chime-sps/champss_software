#!/usr/bin/env python3

import logging

import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.coordinates import EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from sps_common.constants import CHIME_ELEV, CHIME_LAT, CHIME_LON, SEC_PER_DAY

# define CHIME's Earth location
CHIME_LOCATION = EarthLocation(
    lat=CHIME_LAT * u.deg, lon=CHIME_LON * u.deg, height=CHIME_ELEV * u.m
)

# use the JPL planetary ephemeris
solar_system_ephemeris.set("jpl")

LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(levelname)s::%(module)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
logger = logging.getLogger()


def get_barycentric_correction(
    ra: str, dec: str, mjd: float, convention: str = "radio"
):
    """
    Compute the instantaneous barycentric velocity correction towards a given direction
    at a given time.

    Parameters
    ----------
    ra: str
        Right ascension formatted as hh:mm:ss.ss
    dec: str
        Declination formatted as dd:mm:ss.ss
    mjd: float
        Time in MJD
    convention: str, optional
        One of "optical", "radio", or "relativistic" which specifies the
        velocity convention used. Default is "radio"

    Returns
    -------
    float
        Barycentric velocity correction as a fraction of the speed of light
    """
    coord = SkyCoord(ra, dec, frame="icrs", unit=(u.hourangle, u.deg))
    t = Time(mjd, format="mjd", scale="utc")

    logger.info(f"RA (J2000) : {coord.ra.to_string(u.hour)}")
    logger.info(f"Dec (J2000) : {coord.dec.to_string(u.degree)}")
    logger.info(f"Topocentric MJD : {t.mjd}")

    vb_optical = coord.radial_velocity_correction(
        kind="barycentric", obstime=t, location=CHIME_LOCATION
    )

    if convention == "optical":
        vb = vb_optical
    elif convention == "radio":
        vb = vb_optical.to(u.Hz, u.doppler_optical(1 * u.Hz)).to(
            vb_optical.unit, u.doppler_radio(1 * u.Hz)
        )
    elif convention == "relativistic":
        vb = vb_optical.to(u.Hz, u.doppler_optical(1 * u.Hz)).to(
            vb_optical.unit, u.doppler_relativistic(1 * u.Hz)
        )
    else:
        logger.warning(
            "convention not recognised, must be one of: optical, radio, relativistic"
        )
        logger.warning("Defaulting to RADIO convention")
        vb = vb_optical.to(u.Hz, u.doppler_optical(1 * u.Hz)).to(
            vb_optical.unit, u.doppler_radio(1 * u.Hz)
        )

    logger.info(
        "barycentric velocity (fraction of speed of light) = {}".format(
            vb.value / c.value
        )
    )

    return vb.value / c.value


def get_mean_barycentric_correction(
    ra: str, dec: str, start_mjd: float, duration: float, convention: str = "radio"
):
    """
    Compute the mean barycentric velocity correction towards a given direction over a
    certain time span.

    Parameters
    ----------
    ra: str
        Right ascension formatted as hh:mm:ss.ss
    dec: str
        Declination formatted as dd:mm:ss.ss
    start_mjd: float
        Start time in MJD
    duration: float
        Duration over which to calculate the mean correction, in seconds
    convention: str, optional
        One of "optical", "radio", or "relativistic" which specifies the
        velocity convention used. Default is "radio"

    Returns
    -------
    float
        The mean Barycentric velocity correction as a fraction of the speed of
        light
    """

    nsteps = 100
    mjds = np.linspace(start_mjd, start_mjd + duration / SEC_PER_DAY, nsteps)
    coord = SkyCoord(ra, dec, frame="icrs", unit=(u.hourangle, u.deg))
    times = Time(mjds, format="mjd", scale="utc")

    logger.info(f"RA (J2000) : {coord.ra.to_string(u.hour)}")
    logger.info(f"Dec (J2000) : {coord.dec.to_string(u.degree)}")
    logger.info(f"Topocentric MJD : {np.min(mjds)} - {np.max(mjds)}")

    vb_corr = []
    for t in times:
        vb_optical = coord.radial_velocity_correction(
            kind="barycentric", obstime=t, location=CHIME_LOCATION
        )

        if convention == "optical":
            vb = vb_optical
        elif convention == "radio":
            vb = vb_optical.to(u.Hz, u.doppler_optical(1 * u.Hz)).to(
                vb_optical.unit, u.doppler_radio(1 * u.Hz)
            )
        elif convention == "relativistic":
            vb = vb_optical.to(u.Hz, u.doppler_optical(1 * u.Hz)).to(
                vb_optical.unit, u.doppler_relativistic(1 * u.Hz)
            )
        else:
            logger.warning(
                "convention not recognised, must be one of: optical, radio, "
                "relativistic"
            )
            logger.warning("Defaulting to RADIO convention")
            vb = vb_optical.to(u.Hz, u.doppler_optical(1 * u.Hz)).to(
                vb_optical.unit, u.doppler_radio(1 * u.Hz)
            )

        vb_corr.append(vb.value)

    mean_vb = np.mean(vb_corr)

    logger.info(
        "mean barycentric velocity (fraction of speed of light) = {}".format(
            mean_vb / c.value
        )
    )

    return mean_vb / c.value


def barycenter_timeseries(topo_ts: np.ndarray, beta: float, metadata: bool = False):
    """
    Apply the barycentric velocity correction to a time series by duplicating or
    removing specific samples.

    Parameters
    ----------
    topo_ts: np.ndarray
        The topocentric time series
    beta: float
        The barycentric Doppler parameter, v/c
    metadata: bool
        Whether to return additional information about what occurred during
        the correction step (a list of topcentric time series indices where a
        correction was made, and the number of samples added or removed)

    Returns
    -------
    bary_ts: np.ndarray
        The barycentered time series, which has had sampled removed or
        duplicated
    """
    # what indices in the topocentric time series will be "corrected"
    correction_index = []
    correction_required = []  # +1 = add a value, -1 = remove a value
    total_nsamp = topo_ts.size
    bary_ts = np.copy(topo_ts)
    one_over_abs_beta = 1 / np.abs(beta)
    logger.info(f"nsamps in topocentric time series = {topo_ts.size}")
    logger.info(f"nsamps before first correction = {int(round(one_over_abs_beta))}")

    logger.debug("determining correction indices")
    n = 1
    while n * one_over_abs_beta <= total_nsamp:
        correction_index.append(int(round(n * one_over_abs_beta)))
        # NOTE: there is a sign convention issue somewhere, but normally:
        # if beta < 0 then we need to add samples, if beta > 0 we need to
        # remove samples
        # ----------------------
        # however in our case:
        # if beta < 0 then we need to remove samples, if beta > 0 we need to
        # add samples
        if beta < 0:
            correction_required.append(-1)
        else:
            correction_required.append(1)
        n += 1

    correction_index = np.array(correction_index)
    correction_required = np.array(correction_required)
    logger.info(f"# samples to be added: {np.count_nonzero(correction_required == 1)}")
    logger.info(
        "# samples to be removed: {}".format(
            np.count_nonzero(correction_required == -1)
        )
    )

    logger.info("barycentering time series")
    nadded = 0
    nremoved = 0
    for i in range(len(correction_index)):
        idx_offset = nadded - nremoved
        if correction_required[i] > 0:
            # define the replacement value as just a duplication
            replace_val = bary_ts[correction_index[i] + idx_offset]
            # need to +1 the index since np.insert puts the new value BEFORE
            # the given index
            bary_ts = np.insert(
                bary_ts, correction_index[i] + idx_offset + 1, replace_val
            )
            nadded += 1
        else:
            bary_ts = np.delete(bary_ts, correction_index[i] + idx_offset)
            nremoved += 1

    if metadata:
        return bary_ts, correction_index, nadded, nremoved
    else:
        return bary_ts


def bary_from_topo_freq(topo_freq, beta, convention="radio"):
    """
    Convert from topocentric to barycentric frequency.

    :param topo_freq: Topocentric frequency in Hz
    :type topo_freq: float
    :param beta: The barycentric velocity correction (as a fraction of the speed of
        light)
    :type beta: float
    :param convention: The Doppler convention to use when converting frequency
    :type convention: str
    :return: Barycentric frequency in Hz
    :rtype: float
    """
    if convention == "radio":
        return topo_freq * (1 - beta)
    elif convention == "optical":
        return topo_freq * 1 / (1 + beta)
    elif convention == "relativistic":
        return topo_freq * np.sqrt(1 - beta**2) / (1 + beta)
    else:
        logger.warning(
            "convention not recognised, must be one of: optical, radio, relativistic"
        )
        logger.warning("Defaulting to RADIO convention")
        return topo_freq * (1 - beta)


def topo_from_bary_freq(bary_freq, beta, convention="radio"):
    """
    Convert from barycentric to topocentric frequency. Assuming the radio convention.

    :param bary_freq: Barycentric frequency in Hz
    :type bary_freq: float
    :param beta: The barycentric velocity correction (as a fraction of the speed of
        light)
    :type beta: float
    :param convention: The Doppler convention to use when converting frequency
    :type convention: str
    :return: Topocentric frequency in Hz
    :rtype: float
    """
    if convention == "radio":
        return bary_freq / (1 - beta)
    elif convention == "optical":
        return bary_freq / (1 / (1 + beta))
    elif convention == "relativistic":
        return bary_freq / (np.sqrt(1 - beta**2) / (1 + beta))
    else:
        logger.warning(
            "convention not recognised, must be one of: optical, radio, relativistic"
        )
        logger.warning("Defaulting to RADIO convention")
        return bary_freq / (1 - beta)


def bary_from_topo_mjd(topo_mjd, ra, dec):
    """
    Given a topocentric time, correct it to the barycenter based on the source
    direction.

    :param topo_mjd: Topocentric MJD (UTC scale)
    :type topo_mjd: float
    :param ra: Source Right ascension, in hh:mm:ss.ss
    :type ra: str
    :param dec: Source Declination, in dd:mm:ss.ss
    :type dec: str
    :return: barycentric MJD (UTC scale)
    :rtype: float
    """
    topo_time = Time(topo_mjd, format="mjd", scale="utc")
    coord = SkyCoord(ra, dec, frame="icrs", unit=(u.hourangle, u.deg))
    ltt_bary = topo_time.light_travel_time(
        coord, location=CHIME_LOCATION, kind="barycentric", ephemeris="jpl"
    )
    bary_mjd = (topo_time.tdb + ltt_bary).utc.mjd

    return bary_mjd
