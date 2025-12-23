"""
This module defines the known source filters for the known source sifter. An unused
template function is defined below.

Adpated from frb-l2l3 on 20200202.
"""

import numpy as np
from astropy.time import Time
from sps_common.interfaces.utilities import angular_separation

# from sps_common.constants import TELESCOPE_ROTATION_ANGLE
# from sps_common.config import search_freq_range_from_dc
TELESCOPE_ROTATION_ANGLE = -0.071  # in degrees


def compare_position(candidate, known_sources, weight, **kwargs):
    """
    Compare the sky position of `event` with the sky positions of the sources in
    `known_sources` using the Bayes factor. The function first calculates the angular
    sky separation between `event` and the sources in `known_sources`, and then computes
    the Bayes factor based on the likelihoods of the event being associated with a known
    source and being associated with a new random sky position. The response for each
    known source is stored in the `known_source_metrics` variable of `self`.

    Parameters
    ----------
    candidate : class
        Class should be ``MultiPointingCandidate``.
    known_sources : array_like
        A list of known sources.
    weight : float
        The relative weight for this filter compared to other filters.

    Returns
    -------
    response : float, array
        A list of floats, with the Bayes factor for all known sources in
        `known_sources`.

    Notes
    -----
    For a derivation of the Bayes factor for angular separation of two
    astronomical sources, see e.g. [1]_.

    References
    ----------
    .. [1] Budavari, T. & Szalay, A. S. 2008, ApJ, 679, 301-309
    """
    # calculate angular separations
    angle, sep = angular_separation(
        candidate.ra,
        candidate.dec,
        known_sources["pos_ra_deg"],
        known_sources["pos_dec_deg"],
    )

    # in SPS, the position uncertainty is calculated assuming a gaussian ellipsoid beam of EW FWHM of
    # 0.4 degree and NS FWHM of 0.4 / cos(zenith angle) and an angle of from true North from sps_common.
    sigma_event = position_uncertainty(
        0.4 / (2 * np.sqrt(2 * np.log(2))) / np.cos(np.abs(candidate.dec - 49.32)),
        0.4 / (2 * np.sqrt(2 * np.log(2))),
        TELESCOPE_ROTATION_ANGLE,
        angle,
    )

    # if the angle between source A and source B is x degrees,
    # the angle between source B and source A is (x + 180) % 360 degrees
    sigma_ks = position_uncertainty(0.1, 0.1, 0.0, (angle + 180) % 360)

    bayes_factor = position_bayes(sep, sigma_event, sigma_ks)

    # add to the header dictionary 'known_source_metrics'
    # if the key already exists its values are updated
    # if isinstance(candidate.known_source_metrics, dict):
    #    candidate.known_source_metrics.update({"position_weight": weight})
    #    for i, ks_name in enumerate(known_sources["source_name"]):
    #        candidate.known_source_metrics.update(
    #            {"position_bayes_factor_{}".format(ks_name): bayes_factor[i]}
    #        )
    # else:
    #    candidate.known_source_metrics = {"position_weight": weight}
    #    for i, ks_name in enumerate(known_sources["source_name"]):
    #        candidate.known_source_metrics.update(
    #            {"position_bayes_factor_{}".format(ks_name): bayes_factor[i]}
    #        )

    return bayes_factor


def compare_dm(candidate, known_sources, weight, **kwargs):
    """
    Compare the dispersion measures (DMs) of `event` with the DMs of the sources in
    `known_sources` using the Bayes factor. The function first calculates the DM offsets
    between `event` and the sources in `known_sources`, and then computes the Bayes
    factor based on the likelihoods of the event being associated with a known source
    and being associated with a new random DM. The Bayes factor for each known source is
    stored in the `known_source_metrics` variable of `self`.

    Parameters
    ----------
    candidate : class
        Class should be ``MultiPointingCandidate``.
    known_sources : array_like
        A list of known sources.
    weight : float
        The relative weight for this filter compared to other filters.

    Returns
    -------
    bayes_factor : float, array
        A list of floats, with the Bayes factor for all known sources in
        `known_sources`.

    Notes
    -----
    For a derivation of the Bayes factor for a Gaussian variable without
    free parameters, see e.g. section 5.4 in [1]_.

    References
    ----------
    .. [1] Ivezic, Z., Connelly, A. J., VanderPlas, J. T. & Gray, A.
       2014, Statistics, Data Mining, and machine Learning in Astronomy,
       Princeton University Press, 2014
    """
    mu_min = 0.0
    mu_max = 4357  # maximum of all pointins, may also only want the maximum of used pointings

    if candidate.delta_dm == 0:
        used_delta_dm = 1.0
    else:
        used_delta_dm = candidate.delta_dm
    # calculate Bayes factor for all fine-grained steps and take the maximum
    bayes_factor = gaussian_bayes(
        candidate.best_dm,
        used_delta_dm,
        known_sources["dm"],
        mu_min,
        mu_max,
        sigma_mu=known_sources["dm_error"],
    )

    # add to the header dictionary 'known_source_metrics'
    # if the key already exists its values are updated
    # if isinstance(candidate.known_source_metrics, dict):
    #    candidate.known_source_metrics.update({"dm_weight": weight})
    #    for i, ks_name in enumerate(known_sources["source_name"]):
    #        candidate.known_source_metrics.update(
    #            {"dm_bayes_factor_{}".format(ks_name): bayes_factor[i]}
    #        )
    # else:
    #    candidate.known_source_metrics = {"dm_weight": weight}
    #    for i, ks_name in enumerate(known_sources["source_name"]):
    #        candidate.known_source_metrics.update(
    #            {"dm_bayes_factor_{}".format(ks_name): bayes_factor[i]}
    #        )

    return bayes_factor


def compare_frequency(
    candidate, known_sources, weight, frac_harm=4, max_harm=16, **kwargs
):
    """
    Compare the dispersion measures (DMs) of `event` with the DMs of the sources in
    `known_sources` using the Bayes factor. The function first calculates the DM offsets
    between `event` and the sources in `known_sources`, and then computes the Bayes
    factor based on the likelihoods of the event being associated with a known source
    and being associated with a new random DM. The Bayes factor for each known source is
    stored in the `known_source_metrics` variable of `self`.

    Parameters
    ----------
    candidate : class
        Class should be ``MultiPointingCandidate``.
    known_sources : array_like
        A list of known sources.
    weight : float
        The relative weight for this filter compared to other filters.
    frac_harm: int
        Fractional harmonics of the known source period to match up to. E.g. frac_harm=4 means search to
        periods of P/2, P/3 and P/4
    max_harm: int
        Harmonics of the known source period to match up to. E.g max_harm=8 means search to periods of
        2P, 3P, 4P, 5P, 6P, 7P, 8P

    Returns
    -------
    bayes_factor : float, array
        A list of floats, with the Bayes factor for all known sources in
        `known_sources`.

    Notes
    -----
    For a derivation of the Bayes factor for a Gaussian variable without
    free parameters, see e.g. section 5.4 in [1]_.
    References
    ----------
    .. [1] Ivezic, Z., Connelly, A. J., VanderPlas, J. T. & Gray, A.
       2014, Statistics, Data Mining, and machine Learning in Astronomy,
       Princeton University Press, 2014
    """

    # dc is deprecated in simple ps search
    # mu_min, mu_max = search_freq_range_from_dc(candidate.best_dc)
    mu_min, mu_max = search_freq_range_from_dc(0)

    # calculate Bayes factor for all fine-grained steps and take the maximum
    if frac_harm < 1:
        frac_harm = 1
    if max_harm < 1:
        max_harm = 1
    frac_harms = 1 / np.arange(2, frac_harm + 1)
    num_harms = np.arange(1, max_harm + 1)
    harms = np.concatenate((frac_harms, num_harms))
    bayes_factor = np.zeros(len(known_sources))
    cand_nharm = candidate.best_nharm
    if candidate.delta_freq == 0:
        used_delta_freq = 9.70127682e-04 / cand_nharm
    else:
        used_delta_freq = candidate.delta_freq
    current_period = known_sources["current_spin_period_s"]
    for harm in harms:
        bayes_factor_harm = gaussian_bayes(
            candidate.best_freq * harm,
            used_delta_freq * harm,
            1 / current_period,
            mu_min,
            mu_max,
            sigma_mu=known_sources["spin_period_s_error"] / current_period**2,
        )
        bayes_factor = np.max((bayes_factor, bayes_factor_harm), axis=0)

    # add to the header dictionary 'known_source_metrics'
    # if the key already exists its values are updated
    # if isinstance(candidate.known_source_metrics, dict):
    #    candidate.known_source_metrics.update({"freq_weight": weight})
    #    for i, ks_name in enumerate(known_sources["source_name"]):
    #        candidate.known_source_metrics.update(
    #            {"freq_bayes_factor_{}".format(ks_name): bayes_factor[i]}
    #        )
    # else:
    #    candidate.known_source_metrics = {"freq_weight": weight}
    #    for i, ks_name in enumerate(known_sources["source_name"]):
    #        candidate.known_source_metrics.update(
    #            {"freq_bayes_factor_{}".format(ks_name): bayes_factor[i]}
    #        )

    return bayes_factor


def position_uncertainty(semi_major, semi_minor, ellipse_angle, source_angle):
    """
    Calculate the uncertainty in position in the direction of a known source, given the
    positional uncertainty ellipse of a known source.

    Parameters
    ----------
    semi_major : float
        The ellipse's semi-major axis in degrees.
    semi_minor : float
        The ellipse's semi-minor axis in degrees.
    ellipse_angle : float
        The rotation angle of the ellipse in degrees (East of North).
    source_angle : float
        The angle towards the other sky location.

    Returns
    -------
    uncertainty : float
        The positional uncertainty in degrees in the direction of
        `source_angle`.

    Notes
    -----
    The parametric equation of an ellipse, with the semi-major axis
    pointing North for :math:`\\theta` is 0:
    .. math: x(t) = a \\cos(t) \\sin(\\theta) - b \\sin(t) \\cos(\\theta)
    .. math: y(t) = a \\cos(t) \\cos(\\theta) + b \\sin(t) \\sin(\\theta)
    Where, :math:`a` is the semi-major axis, :math:`b` is the semi-minor
    axis, :math:`t` is the clockwise angle around the ellipse for which
    x and y are calculated, and :math:`\\theta` is the rotation angle of
    the ellipse.
    The positional uncertainty is caculated using the Pythagorean theorem
    and :math:`x` and :math:`y`.
    """
    # convert decimal degrees to radians
    deg2rad = np.pi / 180
    semi_major = semi_major * deg2rad
    semi_minor = semi_minor * deg2rad
    ellipse_angle = ellipse_angle * deg2rad
    source_angle = source_angle * deg2rad

    # point on the error ellipse's side in the direction of a known source
    x = semi_major * np.cos(source_angle) * np.sin(ellipse_angle) - semi_minor * np.sin(
        source_angle
    ) * np.cos(ellipse_angle)
    y = semi_major * np.cos(source_angle) * np.cos(ellipse_angle) + semi_minor * np.sin(
        source_angle
    ) * np.sin(ellipse_angle)

    # Pythagoras
    position_uncertainty = np.sqrt(x**2 + y**2)

    # convert radians back to decimal degrees
    return position_uncertainty / deg2rad


def gaussian_bayes(x, sigma_x, mu, mu_min, mu_max, sigma_mu=None, wrap_around=False):
    """
    Calculates the Bayes factor for a Gaussian variable.

    Parameters
    ----------
    x : float, array
        Observed value (e.g. DM of the event).
    sigma_x : float, array
        Uncertainty on the observed value (e.g. uncertainty on the DM
        of the event).
    mu : float, array
        Mean of Gaussian distribution to compare to (e.g. the DM of a
        known source).
    mu_min : float, array
        The lower limit on the variable being compared (e.g. range of
        DMs in CHIME/FRB, so 0 pc cm-3).
    mu_max : float, array
        The upper limit on the variable being compared (e.g. range of
        DMs in CHIME/FRB, so 13000 pc cm-3).
    wrap_around : bool
        Calculate the distance between the variable and the mean using
        modulo arithmetic if variable wraps around (as e.g. RA does).
        (Default: False)

    Returns
    -------
    bayes_factor : float, array
        Results of the comparisons.

    Notes
    -----
    For a derivation of the Bayes factor for a Gaussian variable without
    free parameters, see e.g. section 5.4 in [1]_.

    References
    ----------
    .. [1] Ivezic, Z., Connelly, A. J., VanderPlas, J. T. & Gray, A.
       2014, Statistics, Data Mining, and machine Learning in Astronomy,
       Princeton University Press, 2014
    """
    # match dimensions of `x` and `mu`
    mu = np.atleast_1d(mu)
    x = np.array([x] * len(mu)).T
    if sigma_mu is None:
        sigma = sigma_x
    else:
        nan_idx = np.where(np.isnan(sigma_mu))
        sigma_mu[nan_idx] = mu[nan_idx] * 0.01
        sigma = np.sqrt(sigma_x**2 + sigma_mu**2)

    prefactor = 1.0 / (np.sqrt(2 * np.pi) * sigma) * (mu_max - mu_min)
    if wrap_around:
        # distance in modulo arithmetic
        d1 = (x - mu).T % mu_max
        d2 = (mu - x).T % mu_max
        distance = np.minimum(d1, d2)
    else:
        distance = np.abs(x - mu).T
    exponent = -1.0 / (2 * sigma**2) * distance**2
    bayes_factor = prefactor * np.exp(exponent)
    return bayes_factor


def position_bayes(angular_separation, sigma1, sigma2):
    """
    Calculates the Bayes factor for an angular separation.

    Parameters
    ----------
    angular_separation : float, array
        Angular separation between two sources in degrees.
    sigma1 : float, array
        Position uncertainty for the event in the direction
        of the known sources it is being compared with.
    sigma2 : float, array
        Position uncertainty for the known sources in the direction
        of the event.

    Returns
    -------
    bayes_factor : float, array
        Results of the comparisons.

    Notes
    -----
    See e.g. [1]_.

    References
    ----------
    .. [1] Budavari, T. & Szalay, A. S. 2008, ApJ, 679, 301-309
    """
    squared_sigma = sigma1**2 + sigma2**2
    prefactor = 2.0 / squared_sigma
    exponent = -1.0 * angular_separation**2 / (2 * squared_sigma)
    bayes_factor = prefactor * np.exp(exponent)
    return bayes_factor


def search_freq_range_from_dc(dc, base_freq=50.0, nphi=0):
    """
    Script to return the frequency search range for a given duty cycle.

    Parameters
    ----------
    dc: float
        The duty cycle of the search (0 for power spectrum search).

    base_freq: float
        The base frequency of the hhat search (not applicable for dc == 0). Default = 50.0 Hz

    nphi: int
        The nphi value used for hhat search (not applicable for dc == 0). Default = 0

    Returns
    -------
    freq_min: float
        The minimum frequency of the search (preset values based on the type of search done)

    freq_max: float
        The maximum frequency of the search
    """
    freq_min = 0.01
    if dc == 0:
        freq_min = 9.70127682e-04
        freq_max = 9.70127682e-04 * (2**20) / 10
    elif nphi != 0:
        freq_max = base_freq
    else:
        nphi_set = np.asarray([256, 192, 128, 96, 64, 48, 32, 24, 16, 8, 0])
        nphi_dc = nphi_set[np.where(nphi_set > 2.0 / dc)][0]
        if nphi_dc == 0:
            freq_max = base_freq * 4
        else:
            freq_max = base_freq * 32 / nphi_dc

    return freq_min, freq_max


def change_spin_period(source, new_epoch):
    """
    Calculates period of known source based on new epoch, rather than the observation
    epoch.

    Parameters
    ----------
    source: KnownSource
        The known source class object to be updated

    new_epoch: astropy Time
        Date of new epoch

    Returns
    -------
    P: int
        The updated period of the known source
    """
    P0 = source.spin_period_s
    P1 = source.spin_period_derivative
    if P1 > 0 and ~np.isnan(source.spin_period_epoch):
        detect_epoch = Time(source.spin_period_epoch, format="mjd")
        new_epoch = Time(new_epoch)
        diff_epoch = (new_epoch - detect_epoch).sec
        P = P0 + diff_epoch * P1
    else:
        P = P0
    return P


if __name__ == "__main__":
    print(angular_separation(200.478971, 55.185900, 200.806433, 55.247994))
    print(angular_separation(0.0, 20.0, 180.0, 20.0))
    print(position_uncertainty(1.0, 2.0, 0.0, 0.0))
    print(position_uncertainty(1.0, 2.0, 0.0, 45.0))
    print(position_uncertainty(1.0, 2.0, 0.0, 90.0))
