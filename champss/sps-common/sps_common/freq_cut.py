"""
Highest fluctuation frequency worth storing in a stack, per DM trial.

The effective time resolution of a DM trial adds three terms in quadrature:

    t_eff = sqrt(tsamp^2 + t_smear^2 + t_DM^2)

- tsamp, the sampling time;
- t_smear = 2 K DM df / f^3, the smearing within each channel (df the channel width), taken
  at ref_freq_mhz: the band centre (600 MHz) by default, 2.4 times that at the band top;
- t_DM, the smearing from the DM step: a pulsar whose DM falls between two trials keeps a
  residual delay across the band of up to half the local trial spacing. t_DM is its RMS
  over a uniform DM offset, (spacing / 2) / sqrt(3), the spacing taken from the trials
  themselves (in samples of delay across the band), so it follows whatever grid is used.

Power at fluctuation frequencies well above 1 / t_eff is mostly lost, so stacks may keep
only the bins below factor / t_eff (factor 0.5: the Nyquist frequency of the effective
resolution), never above the true Nyquist frequency.

This is a storage approximation, not exact: a pulsar whose emission comes mostly from the
top of the band keeps some power above the cut. Steep pulsar spectra and the heavier RFI
masking above 600 MHz both move the band's effective centre lower, which makes the
band-centre cutoff more conservative than a flat, unmasked band suggests.
"""

import numpy as np

from sps_common.constants import DM_CONSTANT, FREQ_BOTTOM, FREQ_TOP, TSAMP

DEFAULT_REF_FREQ_MHZ = 600.0
DEFAULT_FACTOR = 0.5
# Samples of delay across the band per unit DM
_DT_PER_DM = DM_CONSTANT * (FREQ_BOTTOM**-2 - FREQ_TOP**-2)


def dm_step_smearing(dms, nchan, trials=None, tsamp=TSAMP, max_snr_loss=0.1, window=4):
    """
    The RMS DM-step smearing (s) at each DM: (local trial spacing / 2) / sqrt(3), the
    spacing in samples of delay across the band.

    trials: the DM trials in use (sorted), whose local spacing, averaged over window
    trials on each side (the S/N-loss grid's trials are whole samples, so neighbouring
    steps alternate), is interpolated at dms. None: the spacing of the S/N-loss grid
    (sps_common.dm_grid.snr_loss_dt_grid) at each DM, from its formula.
    """
    dms = np.atleast_1d(np.asarray(dms, dtype=np.float64))
    dt = dms * _DT_PER_DM / tsamp
    if trials is not None and np.size(trials) >= 2:
        tdt = np.asarray(trials, dtype=np.float64) * _DT_PER_DM / tsamp
        n = tdt.size
        idx = np.arange(n)
        lo, hi = np.maximum(idx - window, 0), np.minimum(idx + window, n - 1)
        spacing = (tdt[hi] - tdt[lo]) / (hi - lo)
        step = np.interp(dt, tdt, spacing)
    else:
        eta = np.sqrt(1 / (1 - max_snr_loss) ** 2 - 1)
        smear_per_dt = (
            2 * (FREQ_TOP - FREQ_BOTTOM) / nchan / FREQ_BOTTOM**3
            / (FREQ_BOTTOM**-2 - FREQ_TOP**-2)
        )  # fmt: skip
        step = np.maximum(1.0, 2 * eta * np.sqrt(2 + (smear_per_dt * dt) ** 2))
    return step / 2 / np.sqrt(3) * tsamp


def smearing_cutoff(
    dms, nchan, ref_freq_mhz=DEFAULT_REF_FREQ_MHZ, factor=DEFAULT_FACTOR, tsamp=TSAMP,
    trials=None,
):
    """
    Cutoff frequency (Hz) for each DM: factor / t_eff, at most the Nyquist frequency, t_eff
    including the DM-step smearing of these trials (see dm_step_smearing).

    Parameters
    ==========
    dms: array_like
        DMs (pc cm^-3) at which to give the cutoff.
    nchan: int
        Number of frequency channels the spectra were beamformed with.
    ref_freq_mhz: float
        Observing frequency at which the intra-channel smearing is evaluated.
    factor: float
        Cutoff in units of 1 / t_eff.
    tsamp: float
        Sampling time (s).
    trials: array_like or None
        The DM trials in use, whose spacing gives the DM-step smearing; None: the S/N-loss
        grid's (see dm_step_smearing).
    """
    if nchan <= 0 or ref_freq_mhz <= 0 or factor <= 0:
        raise ValueError("nchan, ref_freq_mhz and factor must be positive.")
    chan_width = (FREQ_TOP - FREQ_BOTTOM) / nchan
    dms = np.atleast_1d(np.asarray(dms, dtype=np.float64))
    smear = 2 * DM_CONSTANT * np.abs(dms) * chan_width / ref_freq_mhz**3
    t_dm = dm_step_smearing(dms, nchan, trials=trials, tsamp=tsamp)
    t_eff = np.sqrt(tsamp**2 + smear**2 + t_dm**2)
    return np.minimum(factor / t_eff, 0.5 / tsamp)


def kept_bins(freq_labels, dms, nchan, **kwargs):
    """
    Number of leading bins of each DM row at or below that row's cutoff frequency.

    dms are all of a stack's DM trials, so their spacing gives the DM-step smearing.
    freq_labels must increase. Returns an int64 array with one entry per DM.
    """
    cutoff = smearing_cutoff(dms, nchan, trials=dms, **kwargs)
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
