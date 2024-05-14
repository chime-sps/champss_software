#!/usr/bin/env python3

import logging
from typing import Tuple
import numpy as np
from scipy.stats import kstest, chi2
from .utilities import analytical_chi2_cdf, get_ks_distance


log = logging.getLogger(__name__)
rng = np.random.default_rng()


def validate_ps_chisqr_outlier_bins(
    ps: np.ndarray,
    ndays: int = 1,
    red_noise_nbins: int = 1000,
    scale_factor: float = 1,
    remove_zeros: bool = True,
    min_count: int = 25,
    max_count: int = 1000,
    ntests: int = 4,
) -> bool:
    """
    For a pure-noise power spectrum, we expect a perfect chi2 distribution.
    Knowing this, we can compute the expected number of samples above certain
    threshold values and then compare those (with some scaling factor) to the
    input power spectrum.

    The threshold values here are set according to what seems reasonable in
    terms of how many outlier values we want to see. Nominally, we want to do
    a handful of tests with different thresholds that are still near the tail
    to make sure.

    Parameters
    ---------
    :param ps: An array of power spectrum amplitudes
    :type ps: numpy.ndarray

    :param ndays: The number of days included in the provided power spectrum. Default = 1
    :type ndays: int

    :param red_noise_nbins: The typical number of power spectrum bins corrupted by red noise. Default = 1000
    :type red_noise_nbins: int

    :param scale_factor: The fudge factor by which the theoretically expected number of
    "outliers" is multiplied to allow for some larger range. Default = 1
    :type scale_factor: float

    :param remove_zeros: Whether to remove zeros in the spectra. This is currently performed
        after red noise removal. This will change length of the time spectrum.
    :type remove_zeros: bool

    :min_count: The minimum number of outliers at which level the expected outliers are compared
    :type min_count: int

    :max_count: The maximum number of outliers at which level the expected outliers are compared
    :type max_count: int

    :ntests: The number of tests which are performed at different outlier levels
    :type ntests: int


    :return: is_ok
    :rtype: bool
    """
    # We want to ignore the bins corrupted by red noise, so let's hide them
    # from the comparisons done here. Also account for the Leahy-normalisation.
    _ps = 2 * ps[red_noise_nbins:]

    # Spectra contain zeros due to RFI masking. In order to compare to a
    # chi-square distribution these need to be removed first.
    if remove_zeros:
        _ps = _ps[_ps != 0]
    log.debug(f"PS sample size = {_ps.size}")

    # Since we may not have ndays=1, we have to think in the other direction
    # and ask: what values of X make sense such that len(ps) * P(x>X) and within
    # a certain range.
    n_outlier_tests = np.linspace(min_count, max_count, ntests, dtype=int)
    n_outlier_tests_probs = n_outlier_tests / _ps.size
    # TODO BM: Should we actually be using the full PS size, or just the number of unmasked values?
    # LK: Option added to remove zeroes. size after removal is always used
    threshold_values = chi2(2 * ndays).isf(n_outlier_tests_probs)
    log.debug(f"chi2 outlier probs: {n_outlier_tests_probs}")
    log.debug(f"chi2 outlier thresholds: {threshold_values}")

    # log.debug("checking if outlier numbers match expectation")
    found_outliers = []
    expected_outliers = []
    for i, (th, ne) in enumerate(zip(threshold_values, n_outlier_tests)):
        n_above = (_ps > th).sum()
        max_allowed = int(np.ceil(scale_factor * ne))
        found_outliers.append(n_above)
        expected_outliers.append(max_allowed)

    found_outliers = np.asarray(found_outliers)
    expected_outliers = np.asarray(expected_outliers)
    diff_outliers = found_outliers - expected_outliers
    frac_outliers = found_outliers / expected_outliers
    return expected_outliers, found_outliers, diff_outliers, frac_outliers


def compare_ps_to_chisqr_kstest(
    ps: np.ndarray,
    ndays: int = 1,
    lower_percentile_cut: float = None,
    upper_percentile_cut: float = None,
    sig_p: float = 0.05,
    red_noise_nbins: int = 1000,
    plots: bool = False,
    extras: bool = False,
    remove_zeros: bool = True,
    dof_factor: float = 1,
) -> Tuple[float, float]:
    """
    In theory, the noise-only power spectrum containing ndays of data has 2 * ndays
    degrees-of-freedom. If this is not the case, then it implies there is some
    significant corruption of the power spectrum, which likely means we should not
    use it in the long-term stack.

    We perform a KS test to determine whether the power spectrum amplitudes match a chi-squared
    distribution with the appropraite d.o.f. The null hypothesis is that the distributions are
    identical.

    NOTE: it turns out that given the number of samples we test, we're actually in the regime
    where the test power is extreme and so this approach is TOO senstive. Nonetheless, it's a
    useful tool to have available to diagnose power spectra.

    Parameters
    ---------
    :param ps: An array of power spectrum amplitudes
    :type ps: numpy.ndarray

    :param ndays: The number of days included in the provided power spectrum.
        Default = 1
    :type ndays: int

    :param lower_percentile_cut: Samples below this percentile value are ignored.
        Default = None (i.e., no limit)
    :type lower_percentile_cut: float

    :param upper_percentile_cut: Samples above this percentile value are ignored.
        Default = None (i.e., no limit)
    :type upper_percentile_cut: float

    :param sig_p: The p-value significance threshold. Is only used in the plotting routine.
        Default = 0.05
    :type sig_p: float

    :param red_noise_nbins: The typical number of power spectrum bins corrupted by red noise.
        Default is 1000
    :type red_noise_nbins: int

    :param plots: Whether to produce diagnostic plots (only useful for debugging)
    :type plots: bool

    :param extras: Whether to compute extra information, like KS distances and
        equivalent threshold values, and then return these.
    :type extras: bool

    :param remove_zeros: Whether to remove zeros in the spectra. This is currently performed
        after red noise removal and before the percentile cuts are made
    :type remove_zeros: bool

    :param dof_factor: Factor with which the number of degrees of freedom is multiplicated
    :type dof_factor

    :return: (is_valid, statistic, pvalue, [ks_dist, ks_thresh])
        is_valid: Whether the input power spectrum matches the theoretical chi-square distribution
        statistic: two-sided KS statistic
        pvalue: p-value corresponding to the KS statistic

        if extras is True, then also
        ks_dist: The KS distance for each sample value from the test CDF
        ks_thresh: The equivalent KS distance corresponding to a `sig_p`
    :rtype: tuple
    """
    # For a power spectrum where we stack N days of data, the degrees of freedom is 2 * N
    # dof_factor allows changing this value.
    # Depending on the normalisation a slightly smaller value may create a distribution
    # that is closer to the distribution in the observation, eg 0.986
    dof = 2 * ndays * dof_factor

    # We know there is red noise present, which is not chi-square distributed. Even
    # if red-noise removal was attempted, these low-frequency bins are still corrupted
    # and their amplitudes are biased low, thus we exclude them first before computing
    # additional statistical tests.
    if isinstance(ps, np.ma.MaskedArray):
        # strip the masked values as we don't want them causing issues (NaNs, etc.)
        # Question: Are masked arrays currently used? Masked values are set to zero, I think.
        _ps_truncated = np.ma.compressed(ps[red_noise_nbins:])
    else:
        _ps_truncated = ps[red_noise_nbins:]

    # Spectra contain zeros due to RFI masking. In order to compare to a
    # chi-square distribution these need to be removed first
    if remove_zeros:
        _ps_truncated = _ps_truncated[_ps_truncated != 0]

    # We expect that there may be *some* high-power contamination in random bins,
    # especially if we are comparing multi-day stacks where real pulsar signals
    # will be present. We also want to get rid of the really low values since they
    # should not affect our search/stacks, the lowest ~1% of values should suffice.

    l_percentile = None
    u_percentile = None
    if lower_percentile_cut == "None":
        lower_percentile_cut = None
    if upper_percentile_cut == "None":
        upper_percentile_cut = None
    if lower_percentile_cut:
        l_percentile_threshold = 100 * lower_percentile_cut
        l_percentile = np.percentile(_ps_truncated, l_percentile_threshold)
    if upper_percentile_cut:
        u_percentile_threshold = 100 * upper_percentile_cut
        u_percentile = np.percentile(_ps_truncated, u_percentile_threshold)

    if lower_percentile_cut and upper_percentile_cut:
        good_idxs = np.where(
            np.logical_and(
                _ps_truncated >= l_percentile,
                _ps_truncated <= u_percentile,
            )
        )
        ngood = len(good_idxs[0])
    elif lower_percentile_cut and not upper_percentile_cut:
        good_idxs = np.where(_ps_truncated >= l_percentile)
        ngood = len(good_idxs[0])
    elif not lower_percentile_cut and upper_percentile_cut:
        good_idxs = np.where(_ps_truncated <= u_percentile)
        ngood = len(good_idxs[0])
    else:
        good_idxs = ...  # basically, all indexes are fine
        ngood = len(_ps_truncated[good_idxs])

    log.debug(
        f"{ngood} out of {len(_ps_truncated)} within upper/lower percentile cut"
        f" (val={l_percentile}, {u_percentile})"
    )
    ps_truncated = _ps_truncated[good_idxs]
    nsamps = len(ps_truncated)
    if l_percentile:
        truncated_chi2_cutoff_lower = 2 * l_percentile
    else:
        truncated_chi2_cutoff_lower = 0
    if u_percentile:
        truncated_chi2_cutoff_upper = 2 * u_percentile
    else:
        truncated_chi2_cutoff_upper = np.inf

    # When comparing the power spectrum to the chi2 distribution, it is important to
    # note that the normalisation used matters! Leahy normalisation (used in high-energy
    # regimes) ensures the power spectrum has a mean value of 2, which is what one
    # expects for a chi-squared distribution with 2 degrees of freedom. Thus, since
    # our data are normalised in the "radio" convention to have a mean of 1, we need
    # to multiply the powers by 2 before comparing to the chi-squared distribution.
    # e.g. slide 11 of https://heasarc.gsfc.nasa.gov/docs/xrayschool-2003/XraySchool_timing.pdf
    ps_truncated = 2 * ps_truncated

    # Calculate the 1-sample KS statistic and p-value. In this case, since we have a lower
    # sample bound, we need to compute the CDF ourselves with the lower integration bound
    # set to the percentile value used above to exclude very-low value samples.
    # If the p-value returned is below the significance threshold, then we reject the
    # null hypothesis that the data is distributed like a chi-squared distribution,
    # and we reject the power spectrum, otherwise it can move along to any other tests
    # and/or stacking.

    result = kstest(
        ps_truncated,
        analytical_chi2_cdf,
        args=(
            dof,
            truncated_chi2_cutoff_lower,  # integration lower limit
            truncated_chi2_cutoff_upper,  # integration upper limit
        ),
    )

    r_pval = result.pvalue
    r_stat = result.statistic

    if plots:
        import matplotlib.pyplot as plt

        # plot the raw power spectrum
        fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
        m = np.zeros_like(ps, dtype=bool)
        m[:red_noise_nbins] = True
        plt.plot(
            np.ma.masked_array(ps, mask=m),
            label="raw (static + RN cut)",
            alpha=0.5,
        )
        if upper_percentile_cut is not None:
            m[np.where(ps > u_percentile)] = True
            plt.plot(
                np.ma.masked_array(ps, mask=m),
                label="truncated (static + RN + percentile cut)",
                alpha=0.5,
            )
        plt.ylim(-1, 1.1 * np.ma.masked_array(ps, mask=m).max())
        plt.xlabel("Bin")
        plt.ylabel("Power, $|FFT[x(t)]|^2$")
        plt.title("Power spectra input")
        plt.legend()
        plt.savefig("raw_and_masked_PS.png", bbox_inches="tight")
        plt.close(fig)

        # plot the PDF and CDF for comparison
        x = np.logspace(
            *np.log10(chi2.interval(1 - 9e-6, dof)),
            500,
        )
        bins = np.logspace(
            np.log10(np.min(ps_truncated)),
            np.log10(np.max(ps_truncated)),
            250,
        )
        xmin, xmax = 0.75 * np.min(ps_truncated), 1.25 * np.max(ps_truncated)

        fig, (axPDF, axCDF) = plt.subplots(
            ncols=2,
            figsize=plt.figaspect(0.5),
            constrained_layout=True,
        )
        pdf_hist_kwargs = {"density": True, "alpha": 0.2}
        axPDF.hist(ps_truncated, bins=bins, label="truncated data", **pdf_hist_kwargs)
        axPDF.plot(x, chi2.pdf(x, dof), color="r", label=f"$\chi^2$ PDF (dof={dof})")
        if upper_percentile_cut is not None:
            axPDF.axvline(
                2 * u_percentile, color="k", ls="--", label="upper percentile cut"
            )
        if lower_percentile_cut:
            axPDF.axvline(
                2 * l_percentile, color="k", ls="--", label="lower percentile cut"
            )
        axPDF.set_title("PDF")
        axPDF.set_ylabel("PDF, $f_k(x)$")
        axPDF.set_xlabel("Sample value")
        axPDF.legend(loc="lower left")
        axPDF.set_yscale("log")
        axPDF.set_xscale("log")
        axPDF.set_xlim(xmin, xmax)

        cdf_hist_kwargs = {"density": True, "cumulative": True, "alpha": 0.2}
        axCDF.hist(ps_truncated, bins=bins, label="truncated data", **cdf_hist_kwargs)
        axCDF.plot(x, chi2.cdf(x, dof), color="r", label=f"$\chi^2$ CDF (dof={dof})")
        axCDF.plot(
            x,
            analytical_chi2_cdf(
                x,
                dof,
                llim=truncated_chi2_cutoff_lower,
                ulim=truncated_chi2_cutoff_upper,
            ),
            color="g",
            label=f"Truncated $\chi^2$ CDF (dof={dof})",
        )
        if upper_percentile_cut is not None:
            axCDF.axvline(
                2 * u_percentile, color="k", ls="--", label="upper percentile cut"
            )
        if lower_percentile_cut:
            axCDF.axvline(
                2 * l_percentile, color="k", ls="--", label="lower percentile cut"
            )
        axCDF.set_title("CDF")
        axCDF.set_ylabel("CDF, $F_k(x)$")
        axCDF.set_xlabel("Sample value")
        axCDF.legend(loc="lower right")
        axCDF.set_yscale("log")
        axCDF.set_xscale("log")
        axCDF.set_xlim(xmin, xmax)

        plt.suptitle(
            f"(nsamp = {ps_truncated.size:g}, KS stat = {r_stat:g}, p = {r_pval:g})"
        )
        plt.savefig("raw_and_masked_PS_vs_chi2.png")
        plt.close(fig)

    if extras:
        # get the KS distance stats for the sample and plot them
        ks_dist, ks_thresh = get_ks_distance(
            ps_truncated,
            dof=dof,
            llim=truncated_chi2_cutoff_lower,
            ulim=truncated_chi2_cutoff_upper,
            pval=sig_p,
        )
        if plots:
            fig = plt.figure(figsize=plt.figaspect(0.5), constrained_layout=True)
            plt.plot(ps_truncated, ks_dist, ls="none", marker="o", ms=5)
            plt.axhline(
                ks_thresh,
                ls="--",
                color="k",
                label=f"KS stat. (for p = {sig_p}) = {ks_thresh:g}",
            )
            plt.xlabel("Sample value")
            plt.ylabel("KS distance")
            plt.title(f"Per-sample KS statistic with N={nsamps}")
            plt.legend()
            plt.savefig("ks_distance.png")
            plt.close(fig)

    if extras:
        return r_stat, r_pval, [ks_dist, ks_thresh]
    else:
        return r_stat, r_pval
