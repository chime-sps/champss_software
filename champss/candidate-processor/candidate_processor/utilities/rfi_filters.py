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
from scipy.signal import find_peaks

log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "enabled": False,
    # Set SinglePointingCandidate.rfi to True if any filter flags the candidate
    "flag_as_rfi": False,
    "fit_failed": {"amplitude_rel_err_max": 1.0},
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


def dm_curve_shape(dm_freq_sigma):
    """
    Shape measures of the windowed sigma-vs-DM curve of a candidate.

    The curve is the maximum over frequency of dm_freq_sigma["sigmas"] (shape
    n_dm x n_freq) at each DM.

    Returns a dict with
        npeaks: number of separate peaks above 0.6 x the maximum (at least 1 DM apart)
        roughness: standard deviation of the point-to-point differences / maximum
        lowdm_ratio: maximum within 1 DM of the lowest DM / overall maximum
        extent: fraction of the DM range above half the maximum
    or None if the curve is not available.
    """
    if not dm_freq_sigma:
        return None
    sigmas = np.asarray(dm_freq_sigma.get("sigmas", []), dtype=float)
    dms = np.asarray(dm_freq_sigma.get("dms", []), dtype=float)
    if sigmas.ndim != 2 or len(dms) < 10:
        return None
    if sigmas.shape[0] != len(dms) and sigmas.shape[1] == len(dms):
        sigmas = sigmas.T
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
        shape = dm_curve_shape(candidate.dm_freq_sigma)
        if shape:
            result.update(shape)
        rel_err = feature_value(candidate, "dm_sigma_FitGauss_amplitude_rel_err")
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
