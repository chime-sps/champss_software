"""
Highest fluctuation frequency worth storing in a stack, per DM trial.

Within each channel, incoherent dedispersion leaves the pulse smeared over
t_smear = 2 K DM df / f^3 (df the channel width, f the observing frequency), so with the
sampling time the effective time resolution is t_eff = sqrt(tsamp^2 + t_smear^2). Power at
fluctuation frequencies well above 1 / t_eff is mostly lost, so stacks may keep only the
bins below factor / t_eff (factor 0.5: the Nyquist frequency of the effective resolution),
never above the true Nyquist frequency. The smearing is taken at ref_freq_mhz: the band
centre (600 MHz) by default, where it is 2.4 times that at the top of the band.

This is a storage approximation, not exact: a pulsar whose emission comes mostly from the
top of the band keeps some power above the cut. Steep pulsar spectra and the heavier RFI
masking above 600 MHz both move the band's effective centre lower, which makes the
band-centre cutoff more conservative than a flat, unmasked band suggests.
"""

import numpy as np

from sps_common.constants import DM_CONSTANT, FREQ_BOTTOM, FREQ_TOP, TSAMP

DEFAULT_REF_FREQ_MHZ = 600.0
DEFAULT_FACTOR = 0.5


def smearing_cutoff(
    dms, nchan, ref_freq_mhz=DEFAULT_REF_FREQ_MHZ, factor=DEFAULT_FACTOR, tsamp=TSAMP
):
    """
    Cutoff frequency (Hz) for each DM: factor / t_eff, at most the Nyquist frequency.

    Parameters
    ==========
    dms: array_like
        DM trials (pc cm^-3).
    nchan: int
        Number of frequency channels the spectra were beamformed with.
    ref_freq_mhz: float
        Observing frequency at which the intra-channel smearing is evaluated.
    factor: float
        Cutoff in units of 1 / t_eff.
    tsamp: float
        Sampling time (s).
    """
    if nchan <= 0 or ref_freq_mhz <= 0 or factor <= 0:
        raise ValueError("nchan, ref_freq_mhz and factor must be positive.")
    chan_width = (FREQ_TOP - FREQ_BOTTOM) / nchan
    smear = 2 * DM_CONSTANT * np.abs(np.asarray(dms, dtype=np.float64)) * chan_width
    smear /= ref_freq_mhz**3
    t_eff = np.sqrt(tsamp**2 + smear**2)
    return np.minimum(factor / t_eff, 0.5 / tsamp)


def kept_bins(freq_labels, dms, nchan, **kwargs):
    """
    Number of leading bins of each DM row at or below that row's cutoff frequency.

    freq_labels must increase. Returns an int64 array with one entry per DM.
    """
    cutoff = smearing_cutoff(dms, nchan, **kwargs)
    return np.searchsorted(np.asarray(freq_labels), cutoff, side="right").astype(np.int64)


def freq_cut_settings(freq_cut):
    """
    The full settings of a freq_cut configuration: nchan (required), ref_freq_mhz and
    factor (defaults filled in), as stored in a stack's freq_cut_* attributes.
    """
    unknown = set(freq_cut) - {"nchan", "ref_freq_mhz", "factor"}
    if unknown:
        raise ValueError(f"Unknown freq_cut settings: {sorted(unknown)}")
    if freq_cut.get("nchan") is None:
        raise ValueError("freq_cut needs the number of channels (nchan).")
    return {
        "nchan": int(freq_cut["nchan"]),
        "ref_freq_mhz": float(freq_cut.get("ref_freq_mhz", DEFAULT_REF_FREQ_MHZ)),
        "factor": float(freq_cut.get("factor", DEFAULT_FACTOR)),
    }


def stored_freq_cut(h5f):
    """The freq_cut settings a stack was written with, or None."""
    if "freq_cut_nchan" not in h5f.attrs:
        return None
    return {
        "nchan": int(h5f.attrs["freq_cut_nchan"]),
        "ref_freq_mhz": float(h5f.attrs["freq_cut_ref_freq_mhz"]),
        "factor": float(h5f.attrs["freq_cut_factor"]),
    }
