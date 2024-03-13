from attr import attrs, attrib
import numpy as np
import logging
from typing import ClassVar, Tuple, List
import yaml
import sys
from sps_common.interfaces.single_pointing import SinglePointingCandidate
from scipy.special import erf
from scipy.optimize import curve_fit

log = logging.getLogger(__name__)


################################ Stats Classes ################################


@attrs
class Stat:
    """
    Parent class for features which are statistics

    Attributes
    ----------
    value - the value of the statistic calculated
    dtype - the dtype of the value

    Every Stat must have a compute classmethod of the form
    compute(cls, data, weights, point, **kwargs)
    This should have functionality to compute whatever statistic you want,
    from the data, about point, and using the weights.
    """

    value: float = attrib()
    dtype: np.dtype = attrib(init=False, default=float)

    @classmethod
    def compute(cls, data, weights, point, **kwargs):
        """Compute statistic from data, using weights, about point"""
        raise NotImplementedError("Subclasses must implement this method")


@attrs
class Mean(Stat):
    how_combine: ClassVar[str] = "mean"

    @classmethod
    def compute(cls, data, weights, *args, **kwargs):
        mn = (weights * data).sum() / weights.sum()
        return cls(value=mn)


@attrs
class Variance(Stat):
    @classmethod
    def compute(cls, data, weights, point, **kwargs):
        var = (weights * (data - point) ** 2).sum() / weights.sum()
        return cls(value=var)


@attrs
class MAD(Stat):
    """Mean Absolute Deviation"""

    @classmethod
    def compute(cls, data, weights, point, **kwargs):
        mad = np.sum(weights * np.abs(data - point)) / weights.sum()
        return cls(value=mad)


@attrs
class StandardDeviation(Stat):
    @classmethod
    def compute(cls, data, weights, point, **kwargs):
        var = Variance.compute(data, weights, point, **kwargs)
        sd = np.sqrt(var.value)
        return cls(value=sd)


@attrs
class Skewness(Stat):
    convention: str = attrib()

    @classmethod
    def compute(cls, data, weights, point, convention="Pearson", **kwargs):
        var = Variance.compute(data, weights, point, **kwargs)
        # third moment
        m3 = np.sum(weights * (data - point) ** 3) / weights.sum()
        # Pearson
        skew = m3 / var.value ** (3 / 2)
        return cls(value=skew, convention=convention)


@attrs
class Kurtosis(Stat):
    convention: str = attrib()

    @classmethod
    def compute(cls, data, weights, point, convention="Fisher", **kwargs):
        var = Variance.compute(data, weights, point, **kwargs)
        # fourth moment
        m4 = np.sum(weights * (data - point) ** 4) / weights.sum()
        # Fisher
        kurt = m4 / var.value**2 - 3.0
        return cls(value=kurt, convention=convention)


@attrs
class Min(Stat):
    @classmethod
    def compute(cls, data, *args, **kwargs):
        return cls(value=min(data))


@attrs
class Max(Stat):
    @classmethod
    def compute(cls, data, *args, **kwargs):
        return cls(value=max(data))


@attrs
class Range(Stat):
    """maximum value - minimum value"""

    @classmethod
    def compute(cls, data, *args, **kwargs):
        range = max(data) - min(data)
        return cls(value=range)


################################ Fit Classes ################################


@attrs
class Fit:
    """Parent class for features which are from a fit

    Attributes:
    -----------
    pars (ndarray): The parameters resulting from the fit
    par_names (List[str]): The names of said parameters
    covs (ndarray): The covariance matrix resulting from the fit
    ndof (int): Number of degrees of freedom
    rms_err (float): The rms error of the resulting fit. Calculated via
        sqrt( (ydata - model)**2 / ndof )

    compare_pars_to (np.ndarray): The detected values, you wish to compare the
        pars to if the FitGenerator has diff_from_detection=True

        e.g.
        for a FitGauss fit to DM-sigma the pars are [amplitude, mu, gauss_sigma]
        and you would want to compare amplitude to the sigma of the detection,
        mu to the dm of the detection, and gauss_sigma to nothing, so in that case
        compare_pars_to would be [detection_sigma, dection_dm, NaN]


    """

    pars: np.ndarray = attrib()
    par_names: List[str] = attrib()
    covs: np.ndarray = attrib()
    ndof: int = attrib()
    rms_err: float = attrib()
    compare_pars_to: np.array = attrib()

    @classmethod
    def fit(cls, xdata, ydata, **kwargs):
        """Fit a function to x- and ydata"""
        raise NotImplementedError("Subclasses must implement this method")

    def output(self, do_rel_err, do_diff, do_ndof, do_rms_err):
        """Output desired values
        do_rel_err => also output relative error on fit parameters
        do_diff => also output the difference between the fitted value and that
            of the detection (where applicable)
        do_ndof => also output the reduced chisquared for the fit (and the
            number of degrees of freedom)
            NB the reduced chisquared may need to be taken with a giant grain of
            salt, not checking that its assumptions are true.
        do_rms_err => also output the root mean square error from the fit
        """
        out_vals = list(self.pars)
        par_len = len(self.pars)
        out_dts = [float] * par_len

        if do_rel_err:
            for i in range(par_len):
                out_vals.append(np.sqrt(self.covs[i, i]) / self.pars[i])
            out_dts.extend([float] * par_len)

        if do_diff:
            pars = self.pars[~np.isnan(self.compare_pars_to)]
            compare = self.compare_pars_to[~np.isnan(self.compare_pars_to)]
            out_vals.extend(list(pars - compare))
            out_dts.extend([float] * len(pars))

        if do_ndof:
            out_vals.append(self.ndof)
            out_dts.append(int)

        if do_rms_err:
            out_vals.append(self.rms_err)
            out_dts.append(float)

        return out_vals, out_dts


############ Fit to a Gaussian #############


def gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


@attrs
class FitGauss(Fit):
    """
    Fit to a Gaussian via scipy's curve_fit (non-linear least squares)

    Fits for:
    amplitude
    peak_dm
    gauss_sigma (since sigma is already used elsewhere)
    """

    @classmethod
    def fit(cls, xdata, ydata, compare_point):
        par_names = ["amplitude", "mu", "gauss_sigma"]
        # compare_point is in the format x_value_to_compare, y_value_to_compare
        # need to reshuffle so compare amplitude to y, mu to x
        compare_pars_to = np.array(
            [
                compare_point[1],
                compare_point[0],
                np.NaN,
            ]
        )
        init_pars = np.array([ydata.max(), xdata.mean(), 1])
        numpars = init_pars.shape[0]
        ndof = len(xdata) - numpars

        try:
            if ndof <= 0:
                raise AttributeError(
                    "number of degrees of freedom <= 0; not enough data for fit"
                )
            pars, covs = curve_fit(gauss, xdata, ydata, p0=init_pars)
            sum_sq_err = np.sum((ydata - gauss(xdata, pars[0], pars[1], pars[2])) ** 2)
            rms_err = np.sqrt(sum_sq_err / ndof)
        except Exception as ex:
            log.info("FitGauss failed due to the following Exception")
            log.info(ex)
            pars = np.empty((numpars,))
            pars[:] = np.NaN
            covs = np.empty((numpars, numpars))
            covs[:] = np.NaN
            rms_err = np.NaN

        init_dict = dict(
            pars=pars,
            par_names=par_names,
            covs=covs,
            ndof=ndof,
            rms_err=rms_err,
            compare_pars_to=compare_pars_to,
        )
        return cls(**init_dict)
