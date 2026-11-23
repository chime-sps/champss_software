"""
Filters for interference-like stack candidates.

Three criteria, tested on the candidates of stack_2026_06_03_binary_1 (1786 candidates
without and 11 with a known-pulsar match; none of the known-pulsar candidates was
flagged by them):

- fit_failed: the Gaussian fit to the sigma-vs-DM curve failed, i.e. the relative error
  of the fitted amplitude (feature dm_sigma_FitGauss_amplitude_rel_err) is large.
- wobbly: the windowed sigma-vs-DM curve is jagged with several separate peaks, as for
  narrowband interference that appears at many DMs.
- one_day_burst: a daily candidate at the same frequency exists on exactly one day of
  the stack and is strong there. The stack signal then most likely comes from that
  single day.

The measures and flags are stored in SinglePointingCandidate.rfi_filters. The features
array (used by the classifier) is not changed.
"""

import glob
import logging
import os

import numpy as np
from candidate_processor.utilities.features import FitGauss
from scipy.signal import find_peaks

log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "enabled": False,
    # Set SinglePointingCandidate.rfi to True if any filter flags the candidate
    "flag_as_rfi": False,
    # Which sigma-vs-DM map the shape measures and the Gaussian fit use: "convolved"
    # (the map with the candidate's convolution width, dm_freq_sigma["convolve_arr"];
    # falls back to the plain map for candidates without one) or "plain"
    # (dm_freq_sigma["sigmas"]).
    "curve": "convolved",
    # source "fit": Gaussian fit to the DM curve of the map above; "feature": the
    # existing feature dm_sigma_FitGauss_amplitude_rel_err (plain map)
    "fit_failed": {"source": "fit", "amplitude_rel_err_max": 1.0},
    "wobbly": {"roughness_min": 0.03, "min_peaks": 4},
    "one_day_burst": {
        "enabled": True,
        # Folder with the daily output (<basepath>/YYYY/MM/DD/<ra>_<dec>/). If None, the
        # processed root folder of the stack is used.
        "daily_basepath": None,
        "freq_tolerance": 0.03,
        # Daily candidates at least this strong count as a detection on that day
        "min_daily_sigma": 6.0,
        # One-day burst: detected on exactly one day, with at least this sigma
        "min_burst_sigma": 9.0,
    },
}


def merged_config(config):
    """DEFAULT_CONFIG updated with the given (possibly partial) config dict."""
    out = {
        k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()
    }
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value
    return out


def candidate_map(dm_freq_sigma, use_convolved=True):
    """
    The sigma map of a candidate as (sigmas [n_dm, n_freq], dms, freqs, is_convolved).

    With use_convolved, the map computed with the candidate's convolution width and
    harmonic count (dm_freq_sigma["convolve_arr"], "convolve_freqs") is used when it
    exists; it is only stored for candidates with convolve > 1. Otherwise the plain
    map (dm_freq_sigma["sigmas"], "freqs") is used. Returns None if not available.
    """
    if not dm_freq_sigma:
        return None
    dms = np.asarray(dm_freq_sigma.get("dms", []), dtype=float)
    if use_convolved and dm_freq_sigma.get("convolve_arr") is not None:
        sigmas = np.asarray(dm_freq_sigma["convolve_arr"], dtype=float)
        freqs = np.asarray(dm_freq_sigma.get("convolve_freqs", []), dtype=float)
        # The array is allocated for the full window; at the edges of the DM or
        # frequency range only the first rows/columns are filled.
        sigmas = sigmas[: len(dms), : len(freqs)]
        is_convolved = True
    else:
        sigmas = np.asarray(dm_freq_sigma.get("sigmas", []), dtype=float)
        freqs = np.asarray(dm_freq_sigma.get("freqs", []), dtype=float)
        is_convolved = False
    if sigmas.ndim != 2 or len(dms) == 0 or len(freqs) == 0:
        return None
    if sigmas.shape[0] != len(dms) and sigmas.shape[1] == len(dms):
        sigmas = sigmas.T
    if sigmas.shape != (len(dms), len(freqs)):
        return None
    return sigmas, dms, freqs, is_convolved


def dm_curve_shape(dm_freq_sigma, use_convolved=True):
    """
    Shape measures of the sigma-vs-DM curve of a candidate.

    The curve is the maximum over frequency of the candidate's sigma map at each DM
    (see candidate_map for which map is used).

    Returns a dict with
        npeaks: number of separate peaks above 0.6 x the maximum (at least 1 DM apart)
        roughness: standard deviation of the point-to-point differences / maximum
        lowdm_ratio: maximum within 1 DM of the lowest DM / overall maximum
        extent: fraction of the DM range above half the maximum
    or None if the curve is not available.
    """
    found = candidate_map(dm_freq_sigma, use_convolved)
    if found is None or len(found[1]) < 10:
        return None
    sigmas, dms, _, _ = found
    with np.errstate(invalid="ignore"):
        curve = np.nanmax(sigmas, axis=1)
    ok = np.isfinite(curve)
    curve, dms = curve[ok], dms[ok]
    if len(curve) < 10:
        return None
    peak = curve.max()
    if peak <= 0:
        return None
    step = np.median(np.diff(dms))
    peaks, _ = find_peaks(
        curve, height=0.6 * peak, distance=max(1, int(round(1.0 / max(step, 1e-3))))
    )
    low = curve[dms <= dms.min() + 1.0]
    return {
        "npeaks": int(len(peaks)),
        "roughness": float(np.std(np.diff(curve)) / peak),
        "lowdm_ratio": float(low.max() / peak) if len(low) else float("nan"),
        "extent": float(np.mean(curve > 0.5 * peak)),
    }


def fit_dm_curve(dm_freq_sigma, freq, dm, sigma, use_convolved=True):
    """
    Gaussian fit (features.FitGauss, as for the dm_sigma features) to the sigma-vs-DM
    curve at the frequency of the candidate in its sigma map.

    With the convolved map the fitted amplitude can be compared with the candidate's
    sigma, which comes from the convolved search; the plain map of a convolved
    candidate peaks much lower, so a fit bounded to the candidate's sigma fails there.

    Returns a dict with amplitude, mu, gauss_sigma, their relative errors, rms_err and
    whether the convolved map was used, or None if no map is available.
    """
    found = candidate_map(dm_freq_sigma, use_convolved)
    if found is None:
        return None
    sigmas, dms, freqs, is_convolved = found
    column = int(np.argmin(np.abs(freqs - freq)))
    ydata = sigmas[:, column]
    ok = np.isfinite(ydata)
    fit = FitGauss.fit(dms[ok], ydata[ok], [dm, sigma])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.sqrt(np.diag(fit.covs)) / fit.pars
    out = {"convolved": bool(is_convolved), "rms_err": float(fit.rms_err)}
    for name, value, err in zip(fit.par_names, fit.pars, rel_err):
        out[name] = float(value)
        out[f"{name}_rel_err"] = float(err)
    return out


def feature_value(candidate, name):
    """Value of a named feature of a candidate, or None if it does not exist."""
    features = candidate.features
    if (
        features is None
        or features.dtype.names is None
        or name not in features.dtype.names
    ):
        return None
    return float(np.asarray(features[name]).ravel()[0])


def _daily_candidates(daily_basepath, ra, dec, day, cache):
    """(freq, sigma) of the daily candidates of a pointing on a day (cached)."""
    key = (round(ra, 2), round(dec, 2), day)
    if key not in cache:
        folder = os.path.join(
            daily_basepath, day.strftime("%Y/%m/%d"), f"{ra:.2f}_{dec:.2f}"
        )
        found = None
        for filename in glob.glob(f"{folder}/*_power_spectra_candidates.npz"):
            try:
                data = np.load(filename, allow_pickle=True)
                found = [
                    (float(data[k].item()["freq"]), float(data[k].item()["sigma"]))
                    for k in data.files
                    if k.startswith("candidate_")
                ]
                break
            except Exception as error:
                log.debug(f"Could not read {filename}: {error}")
        cache[key] = found
    return cache[key]


def daily_crossmatch(candidate, daily_basepath, freq_tolerance, min_daily_sigma, cache):
    """
    Daily candidates at the frequency of a stack candidate.

    Returns (days with a daily candidate file, days with a daily candidate within
    freq_tolerance and at least min_daily_sigma, strongest such daily sigma).
    """
    days_checked = 0
    hits = []
    for day in candidate.datetimes or []:
        daily = _daily_candidates(
            daily_basepath, candidate.ra, candidate.dec, day, cache
        )
        if daily is None:
            continue
        days_checked += 1
        sigmas = [
            sigma
            for freq, sigma in daily
            if abs(freq - candidate.freq) < freq_tolerance and sigma >= min_daily_sigma
        ]
        if sigmas:
            hits.append(max(sigmas))
    return days_checked, len(hits), max(hits) if hits else 0.0


def apply_rfi_filters(candidates, config, daily_basepath=None):
    """
    Compute the filter measures and flags for a list of SinglePointingCandidates and
    store them in candidate.rfi_filters (in place). Manual candidates are skipped.

    Returns the number of flagged candidates.
    """
    config = merged_config(config)
    burst_config = config["one_day_burst"]
    if burst_config.get("daily_basepath"):
        daily_basepath = burst_config["daily_basepath"]
    cache = {}
    n_flagged = 0
    for candidate in candidates:
        if candidate.manual_candidate:
            continue
        result = {}
        use_convolved = config["curve"] == "convolved"
        shape = dm_curve_shape(candidate.dm_freq_sigma, use_convolved)
        if shape:
            result.update(shape)
        fit = fit_dm_curve(
            candidate.dm_freq_sigma,
            candidate.freq,
            candidate.dm,
            candidate.sigma,
            use_convolved,
        )
        if fit:
            result.update({f"dm_fit_{k}": v for k, v in fit.items()})
        feature_rel_err = feature_value(
            candidate, "dm_sigma_FitGauss_amplitude_rel_err"
        )
        result["feature_fit_amplitude_rel_err"] = feature_rel_err
        if config["fit_failed"].get("source", "fit") == "feature":
            rel_err = feature_rel_err
        else:
            rel_err = fit["amplitude_rel_err"] if fit else None
        result["fit_amplitude_rel_err"] = rel_err
        result["fit_failed"] = bool(
            rel_err is not None
            and (
                not np.isfinite(rel_err)
                or rel_err > config["fit_failed"]["amplitude_rel_err_max"]
            )
        )
        result["wobbly"] = bool(
            shape
            and shape["roughness"] > config["wobbly"]["roughness_min"]
            and shape["npeaks"] >= config["wobbly"]["min_peaks"]
        )
        result["one_day_burst"] = False
        if burst_config.get("enabled") and daily_basepath:
            checked, hit_days, max_sigma = daily_crossmatch(
                candidate,
                daily_basepath,
                burst_config["freq_tolerance"],
                burst_config["min_daily_sigma"],
                cache,
            )
            result.update(
                daily_days_checked=checked,
                daily_hit_days=hit_days,
                daily_max_sigma=max_sigma,
            )
            result["one_day_burst"] = bool(
                hit_days == 1 and max_sigma >= burst_config["min_burst_sigma"]
            )
        result["flagged"] = bool(
            result["fit_failed"] or result["wobbly"] or result["one_day_burst"]
        )
        candidate.rfi_filters = result
        if result["flagged"]:
            n_flagged += 1
            if config["flag_as_rfi"]:
                candidate.rfi = True
    log.info(
        f"RFI filters flagged {n_flagged} of"
        f" {sum(not c.manual_candidate for c in candidates)} candidates."
    )
    return n_flagged
