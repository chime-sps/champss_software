"""Utility functions used in the interfaces and in other repositories."""

import logging

import numpy as np
from scipy.special import chdtrc, chdtri, loggamma, ndtr, ndtri
import importlib
import sys

# Import beam model for arc calculation
if importlib.util.find_spec("cfbm"):
    import cfbm as beam_model

    bm_config = beam_model.current_config
    beammod = beam_model.current_model_class(bm_config)


log = logging.getLogger(__name__)


def within_range(value, min_value, max_value):
    """
    Function to determine if float is within a min and max value range.

    The relative and absolute difference between the numbers are 1e-5 and
    1e-8 respectively.

    Parameters
    ----------
    value: float
        The value to check.

    min_value: float
        The minimum value of the range to check.

    max_value
        The maximum value of the range to check.

    Returns
    -------
    is_within_range: bool
        Whether the given value is within the given range.
    """
    if value <= min_value:
        if np.isclose(value, min_value):
            return True
        else:
            return False
    elif value >= max_value:
        if np.isclose(value, max_value):
            return True
        else:
            return False
    else:
        return True


def powersum_at_sigma(sigma, nsum):
    """
    Returns the approximate sum of 'nsum' normalized powers.

    Equivalent to a detection of significance 'sigma'. Adapted from presto.

    Parameters
    ----------
        sigma: float or np.ndarray
            The sigma that should be reached by the sums.
            This value should not exceed 8.2.
            Otherwise the result will be inf.
        nsum: int or np.ndarrar
            The number of summands.

    Returns
    -------
        float or np.ndarray
            The approximate sum that is equivalent to a detection
            of significance 'sigma'.
    """
    return 0.5 * chdtri(2.0 * nsum, 1.0 - ndtr(sigma))


def sigma_sum_powers(power, nsum, max_iter=20):
    """
    Returns an approx equivalent Gaussian sigma.

    For noise to exceed a sum of 'nsum' normalized powers given
    by 'power' in a power spectrum. Adapted from presto.

    Parameters
    ----------
        power: float or np.ndarray
            The summed power.
            If both power and nsum are arrays they need to be broadcastable together
            (the trailing dimensions should have the same shape).
        nsum: int or np.ndarray
            The number of summands.
            nsum should not have more dimensions than power
            (i.e. it is not possible currently to calculate sigma for the same power
            with multiple nsum values).
        max_iter: int
            Maximum number of iteration when computing G(p,x).
            When power~nsum more iterations are needed but these cases will calculated
            byt the alternate method, so al relatively small iteration count is enough.
            Default: 20

    Returns
    -------
        float or np.ndarray
            The approximate equivalent Gaussian sigma.
            If the output would be smaller than ~-8, it will result in np.inf.
    """
    if not np.isfinite(power).any() or not np.isfinite(nsum).any():
        if np.isscalar(power):
            return np.nan
        else:
            return np.full(power.shape, np.nan, dtype=power.dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = equivalent_gaussian_sigma(prob_sum_powers(power, nsum))
    # sigma > ~38.4: np,nan
    # sigma > ~38: slightly deviates from log_prob_sum_powers_asymptotic result
    # sigma < ~-8: -np.inf
    # power=nsum=0: np.nan
    if np.isscalar(sigma):
        if (np.isnan(sigma) and power != 0) or sigma > 38:
            power = float(power)
            return extended_equiv_gaussian_sigma(
                log_prob_sum_powers_asymptotic(power, nsum)
            )
        else:
            return sigma
    else:
        case_array = np.logical_or(
            np.logical_and(np.isnan(sigma), power != 0), sigma > 38
        )
        # equivalent_gaussian_sigma only works up to ~38.45 sigma
        # at the end of this range the results slightly deviate from
        # the method using log_prob_sum_powers_asymptotic
        if case_array.any():
            # nsum may have a smaller shape than power when power come from the different
            # dms which all share the same nsum
            if np.isscalar(nsum):
                nsum = np.broadcast_to(nsum, power.shape)
            elif len(power.shape) != len(nsum.shape):
                nsum = np.broadcast_to(nsum, power.shape)
            sigma[case_array] = extended_equiv_gaussian_sigma(
                log_prob_sum_powers_asymptotic(
                    power[case_array], nsum[case_array], max_iter
                )
            )
        return sigma


def prob_sum_powers(power, nsum):
    """
    Return the probability for noise to exceed 'power' in the sum of 'nsum'.

    'nsum' being the normalized powers from a power spectrum. Adapted from presto.

    Parameters
    ----------
        power: float or np.ndarray
            The summed power.
        nsum: int or np.ndarray
            The number of summands.

    Returns
    -------
        float or np.ndarray
            The resulting probability.
    """
    return chdtrc(2 * nsum, 2.0 * power)


def log_prob_sum_powers_asymptotic(power, nsum, max_iter=200):
    """
    Return the log of the probability for noise to exceed 'power' in the sum of 'nsum'.

    'nsum' being the normalized powers from a power spectrum. This version allows
    the use of very large powers by using asymptotic expansions from
    Abramowitz and Stegun Chap 6. This function always uses this expansion
    unlike the original presto function Adapted from presto.

    Parameters
    ----------
        power: float or np.ndarray
            The summed power.
        nsum: int or np.ndarray
            The number of summands.
        max_iter: int
            Maximum number of iteration when computing G(p,x).

    Returns
    -------
        float or np.ndarray
            The log of resulting probability.
    """
    return log_upper_incomplete_gamma(nsum, power, max_iter) - loggamma(nsum)


def equivalent_gaussian_sigma(p):
    """
    Return the equivalent gaussian sigma.

    Corresponding to the cumulative gaussian probability p.
    In other words, return x, such that Q(x) = p, where Q(x) is the
    cumulative normal distribution. For very small p the asymptotic
    approximation is used.

    Adapted from presto.

    Parameters
    ----------
        p: float or np.ndarray
            The cumulative gaussian probability.

    Returns
    -------
        float or np.ndarray
            The gaussian sigma.
    """
    sigma = ndtri(1.0 - p)

    # This will return np.inf above ~8 sigma and np.-inf below -8
    # Above ~ 7.4 the result will diverge from the two methods using
    # extended_equiv_gaussian_sigma
    # The older threshold of np.log(p)==30, resulted in sigma=7.357
    if np.isscalar(sigma):
        if np.isposinf(sigma) or sigma > 7.3:
            return extended_equiv_gaussian_sigma(np.log(p))
        else:
            return sigma
    else:  # Array input
        case_array = np.logical_or(np.isposinf(sigma), sigma > 7.3)
        if case_array.any():
            # This will return nan above ~38
            sigma[case_array] = extended_equiv_gaussian_sigma(np.log(p[case_array]))
        return sigma


def extended_equiv_gaussian_sigma(logp):
    """
    Return the extended equivalent gaussian sigma.

    Corresponding to the log of the cumulative gaussian probability logp.
    In other words, return x, such that Q(x) = p, where Q(x)
    is the cumulative normal distribution.  This version uses the rational
    approximation from Abramowitz and Stegun, eqn 26.2.23.  Using the log(P)
    as input gives a much extended range. Adapted from presto.

    Parameters
    ----------
        logp: float or np.ndarray
            The lof of the cumulative gaussian probability.

    Returns
    -------
        float or np.ndarray
            The equivalent gaussian sigma.
    """
    t = np.sqrt(-2.0 * logp)
    num = 2.515517 + t * (0.802853 + t * 0.010328)
    denom = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308))
    return t - num / denom


def g_abergel_moisan(p, x, stop_value=np.finfo(float).eps, max_iter=200):
    """
    Algorithm1 from Abergel and Moisan https://hal.science/hal-01329669/document.

    Calculates G(p,x). This provides the normalisation needed to be able to
    calculate the upper incomplete gamma function over a large range of input values.

    Parameters
    ----------
        p: int or np.ndarray
            The parameter p of the upper incomplete gamma function.
            This value should be positive.
        x: float or np.ndarray
            The lower bound x of the integral in the upper incomplete gamma function.
            This value should be not negative.
        stop_value: float
            The value used for stopping the iterative process.
            Once 'delta' is below that value the iteration is stopped.
            As default the machine precision of floats is used.
        max_iter: int
            Maximum number of iteration when computing G(p,x). For small
            iterations there can be deviations when p~x and thus sigma~0.
            FOr most values iterations of about 10 would be sufficient.
            Default: 200

    Returns
    -------
        g: float or np.ndarray
            The function G(p,x) used by Abergel and Moisan.


    """
    dm = 10e-300  # number near minimal floating-point value

    # If float32 is used the default stop_value threshold will
    # not be reached for array input
    if type(x) == np.ndarray:
        x = x.astype(np.float64)
    else:
        x = float(x)

    def get_an_bn_x_s_p(p, x, n):
        if n % 2:
            n_odd = int((n - 1) / 2)
            an = n_odd * x
        else:
            n_even = int(n / 2)
            an = -(p - 1 + n_even) * x
        bn = p - 1 + n
        return an, bn

    def get_an_bn_x_l_p(p, x, n):
        alphan = -(n - 1) * (n - p - 1)
        betan = x + 2 * n - 1 - p
        return alphan, betan

    case_array = x <= p
    a1 = 1.0
    b1 = np.where(case_array, p, x + 1 - p)
    f = a1 / b1
    C = a1 / dm
    D = 1 / b1
    n = 2
    delta = 100
    if type(x) == np.ndarray:
        g = np.full(x.shape, np.inf)
        if np.any(x[np.resize(p == 0, x.shape)]):
            log.warning("Array where p=0!=x found. This should not happen.")
    else:
        g = np.array(np.inf)
        if p == 0 != x:
            log.warning(
                f"p=0!=x in g_abergel_moisan found. This should not happen. p:{p},"
                f" x={x}"
            )
    # If p=x=0 the result will be nan
    # For the use with sigma_sum_powers if p=0, x should also be 0
    g[x == 0] = np.nan
    while np.isinf(g).any():
        an_x_s_p, bn_x_s_p = get_an_bn_x_s_p(p, x, n)
        an_x_l_p, bn_x_l_p = get_an_bn_x_l_p(p, x, n)
        an = np.where(case_array, an_x_s_p, an_x_l_p)
        bn = np.where(case_array, bn_x_s_p, bn_x_l_p)
        D = D * an + bn
        D = np.where(D == 0, dm, D)
        C = bn + an / C
        C = np.where(C == 0, dm, C)
        D = 1 / D
        delta = C * D
        f = f * delta
        n = n + 1
        new_vals_indices = np.fabs(delta - 1) <= stop_value
        g[new_vals_indices] = f[new_vals_indices]
        if n > max_iter:
            log.debug(
                "g_abergel_moisan has not converged after 200 steps."
                + f" Current (delta - 1): {np.nanmax(np.fabs(delta - 1))}"
                + f"{C} {D} {delta} "
                + f"{p} {x}"
            )
            g[np.isinf(g)] = f[np.isinf(g)]
            break
    return g


def log_upper_incomplete_gamma(p, x, max_iter):
    """
    Compute the log of the upper incomplete gamma function.

    Based on Abergel and Moisan
    https://hal.science/hal-01329669/document

    Parameters
    ----------
        p: int or np.ndarray
            The parameter p of the upper incomplete gamma function.
            This value should be positive.
        x: float or np.ndarray
            The lower bound x of the integral in the upper incomplete gamma functtion.
            This value should be not negative.
        max_iter: int
            Maximum number of iteration when computing G(p,x).

    Returns
    -------
        float or np.ndarray
            The log of the upper incomplete gamma function
    """
    G = g_abergel_moisan(p, x, max_iter=max_iter)
    case_array = x <= p

    # Rho and Sigma are defined in Abergel and Moisan (26)
    log_gamma_p_x_s_p = loggamma(p)
    rho = np.where(
        case_array, 1 - np.e ** (-x + p * np.log(x) - log_gamma_p_x_s_p) * G, G
    )
    sig = np.where(case_array, log_gamma_p_x_s_p, -x + p * np.log(np.abs(x)))
    return np.log(rho) + sig


def filter_class_dict(used_class, used_dict):
    """
    Filter unwanted keys from a dict that is used as kwargs for given class.

    Parameters
    ----------
    used_class :
        The class which will use the filtered dictionary for initialisation.
        Properties need to be created with attr. Attributes that are present in the
        dictionary but not in the class will be filtered out.
    used_dict : dict
        The dictionary that will be filtered.

    Returns
    -------
    filtered: dict
        The filtered dictionary
    """
    filtered = {
        attribute.name: used_dict[attribute.name]
        for attribute in used_class.__attrs_attrs__
        if attribute.name in used_dict
    }
    return filtered


# harmonic sum functions
def get_frac_harmonic(num, denom, spectrum):
    """
    Return spectrum values corresponding to harmonic num/denom ​

        Parameters
        ----------
        num : integer or float
            Numerator of the fractional harmonic
        denom : integer or float
            Denominator of the fractional harmonic
        spectrum : float or double array
            Spectrum that you want to do harmonic summing of
    ​
        Returns
        -------
        spectrum : array of the same type of spectrum
            Selected elements of the spectrum for harmonic num/denom
        bins: array of integer
            The array showing the exact frequency bins of the original
            spectrum used by the harmonic spectrum.
    """
    inds = np.arange(len(spectrum), dtype=float)
    # get the closest indices for the fractional harmonics
    # they need to be integers to use them as indices
    bins = np.round(inds * num / denom).astype(int)
    return spectrum[bins], bins


def harmonic_sum(numharm, spectrum, partial=None, partial_n=1):
    """
    Perform a top-down harmonic sum of a spectrum ​

        Parameters
        ----------
        numharm : integer
            Number of harmonics to sum (2, 4, 8, 16, or 32)
        spectrum : float or double array
            Spectrum to perform harmonic summing on
        partial : float or double array
            partially harmonic sum spectrum, default is None
        partial_n : int, optional
            Number of harmonics in partial spectrum
    ​
        Returns
        -------
        spectrum : array of same type of the spectrum
            The full harmonic summed spectrum
        harm_bins : array of shape (numharm-partial_n, len(spectrum))
            The frequency bins used by all the harmonics that are computed.
    """
    if numharm == 1:
        harm_bins = np.zeros((1, (len(spectrum))))
    else:
        harm_bins = np.zeros((numharm - partial_n, (len(spectrum))))
    harm_count = 0
    if partial_n == 1:
        partial = spectrum.copy()  # This is the high harmonic
        if numharm == 1:
            harm_bins = np.arange(len(spectrum))
            harm_count += 1
    if numharm > partial_n and partial_n < 2:
        harm_spec, harm_bin = get_frac_harmonic(1, 2, spectrum)
        np.add(partial, harm_spec, partial)
        if harm_bins.shape[0] == 1:
            harm_bins = harm_bin
        else:
            harm_bins[harm_count] = harm_bin
        harm_count += 1
        partial_n = 2
    if numharm > partial_n and partial_n < 4:
        # Note that the 1/2 == 2/4 has already been added
        harm_spec, harm_bin = get_frac_harmonic(1, 4, spectrum)
        np.add(partial, harm_spec, partial)
        harm_bins[harm_count] = harm_bin
        harm_count += 1
        harm_spec, harm_bin = get_frac_harmonic(3, 4, spectrum)
        np.add(partial, harm_spec, partial)
        harm_bins[harm_count] = harm_bin
        harm_count += 1
        partial_n = 4
    if numharm > partial_n and partial_n < 8:
        # 2/8, 4/8, and 6/8 have already been added
        for ii in [1, 3, 5, 7]:
            harm_spec, harm_bin = get_frac_harmonic(ii, 8, spectrum)
            np.add(partial, harm_spec, partial)
            harm_bins[harm_count] = harm_bin
            harm_count += 1
        partial_n = 8
    if numharm > partial_n and partial_n < 16:
        # [even]/16 have all been added
        for ii in np.arange(1, 16, 2):
            harm_spec, harm_bin = get_frac_harmonic(ii, 16, spectrum)
            np.add(partial, harm_spec, partial)
            harm_bins[harm_count] = harm_bin
            harm_count += 1
        partial_n = 16
    if numharm > partial_n and partial_n < 32:
        # [even]/32 have all been added
        for ii in np.arange(1, 32, 2):
            harm_spec, harm_bin = get_frac_harmonic(ii, 32, spectrum)
            np.add(partial, harm_spec, partial)
            harm_bins[harm_count] = harm_bin
            harm_count += 1
        partial_n = 32
    return partial, harm_bins


def angular_separation(ra1, dec1, ra2, dec2):
    """
    Calculates the angular separation between two celestial objects.

    Parameters
    ----------
    ra1 : float, array
        Right ascension of the first source in degrees.
    dec1 : float, array
        Declination of the first source in degrees.
    ra2 : float, array
        Right ascension of the second source in degrees.
    dec2 : float, array
        Declination of the second source in degrees.

    Returns
    -------
    angle : float, array
        Angle between the two sources in degrees, where 0 corresponds
        to the positive Dec axis.
    angular_separation : float, array
        Angular separation between the two sources in degrees.

    Notes
    -----
    The angle between sources is calculated with the Pythagorean theorem
    and is used later to calculate the uncertainty in the event position
    in the direction of the known source.
    Calculating the angular separation using spherical geometry gives
    poor accuracy for small (< 1 degree) separations, and using the
    Pythagorean theorem fails for large separations (> 1 degrees).
    Transforming the spherical geometry cosine formula into one that
    uses haversines gives the best results, see e.g. [1]_. This gives:
    .. math:: \\mathrm{hav} d = \\mathrm{hav} \\Delta\\delta +
              \\cos\\delta_1 \\cos\\delta_2 \\mathrm{hav} \\Delta\\alpha
    Where we use the identity
    :math:`\\mathrm{hav} \\theta = \\sin^2(\\theta/2)` in our
    calculations.
    The calculation might give inaccurate results for antipodal
    coordinates, but we do not expect such calculations here..
    The source angle (or bearing angle) :math:`\\theta` from a point
    A(ra1, dec1) to a point B(ra2, dec2), defined as the angle in
    clockwise direction from the positive declination axis can be
    calculated using:
    .. math:: \\tan(\\theta) = (\\alpha_2 - \\alpha_1) /
              (\\delta_2 - \\delta_1)
    In NumPy :math:`\\theta` can be calculated using the arctan2
    function. Note that for negative :math:`\\theta` a factor :math:`2\\pi`
    needs to be added. See also the documentation for arctan2.

    References
    ----------
    .. [1] Sinnott, R. W. 1984, Sky and Telescope, 68, 158
    Examples
    --------
    >>> print(angular_separation(200.478971, 55.185900, 200.806433, 55.247994))
    (79.262937451490941, 0.19685681276638525)
    >>> print(angular_separation(0., 20., 180., 20.))
    (90.0, 140.0)
    """
    # convert ra and dec to numpy arrays
    ra1 = np.array(ra1)
    ra2 = np.array(ra2)
    dec1 = np.array(dec1)
    dec2 = np.array(dec2)

    # convert decimal degrees to radians
    deg2rad = np.pi / 180
    ra1 = ra1 * deg2rad
    dec1 = dec1 * deg2rad
    ra2 = ra2 * deg2rad
    dec2 = dec2 * deg2rad

    # delta works
    dra = ra1 - ra2
    ddec = dec1 - dec2

    # haversine formula
    hav = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2.0) ** 2
    angular_separation = 2 * np.arcsin(np.sqrt(hav))

    # angle in the clockwise direction from the positive dec axis
    # note the minus signs in front of `dra` and `ddec`
    source_angle = np.arctan2(-dra, -ddec)
    if isinstance(source_angle, np.ndarray):
        source_angle[source_angle < 0] += 2 * np.pi
    elif source_angle < 0:
        source_angle += 2 * np.pi

    # convert radians back to decimal degrees
    return source_angle / deg2rad, angular_separation / deg2rad


def get_arc_for_beam(beam_no, time, delta_x=90, samples=101):
    """
    Get the EW arc in RA and Dec for a given beam at a given time.

    Parameters
    ----------
    beam_no : int
        Beam number for beam of interest.
    time : datetime object
        Time of interest (central transit time).
    delta_x : float
        Number of degrees away from meridian that arc subtends. Default
        is 90 (horizon to horizon).
    samples : int
        Number of points in the arc.

    Returns
    -------
    eq_positions : ndarray
        Array of (RA, Dec) positions along the arc.
    """
    if "cfbm" not in sys.modules:
        return None

    eq_positions = []
    beam_pos = np.squeeze(beammod.get_beam_positions(beam_no, freqs=beammod.clamp_freq))
    y = beam_pos[1]

    xes = np.linspace(
        -delta_x * np.cos(np.deg2rad(y)), delta_x * np.cos(np.deg2rad(y)), samples
    )

    for x in xes:
        eq_positions.append(beammod.get_equatorial_from_position(x, y, time))

    return np.array(eq_positions)
