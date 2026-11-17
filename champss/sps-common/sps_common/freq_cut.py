"""
Highest fluctuation frequency worth storing in a stack, per DM trial.

The effective time resolution of a DM trial adds three terms in quadrature:

    t_eff = sqrt(tsamp^2 + t_smear^2 + t_DM^2)

- tsamp, the sampling time;
- t_smear = 2 K DM df / f^3, the smearing within each channel (df the channel width), taken
  at ref_freq_mhz: the band centre (600 MHz) by default, 2.4 times that at the band top;
- t_DM, the smearing from the DM step: a pulsar whose DM falls between two trials keeps a
  residual delay across the band of up to half the trial spacing. t_DM is its RMS over a
  uniform DM offset, (step / 2) / sqrt(3), with the step given analytically by the rule
  the DM grid is built from (dm_grid_step): for the S/N-loss grid
  max(1, 2 eta sqrt(w^2 + 1 + (s d)^2)) samples of delay across the band, a smooth
  function of DM; for a uniform grid the constant dm_step.

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
# The DM grid the trials follow (dedisp.fdmt.max_snr_loss, dm_step); max_snr_loss 0: uniform
DEFAULT_MAX_SNR_LOSS = 0.1
DEFAULT_DM_STEP = 2
# Samples of delay across the band per unit DM
_DT_PER_DM = DM_CONSTANT * (FREQ_BOTTOM**-2 - FREQ_TOP**-2)


def dm_grid_step(dms, nchan, max_snr_loss=DEFAULT_MAX_SNR_LOSS, dm_step=DEFAULT_DM_STEP,
                 pulse_width=1.0):
    """
    The DM grid's trial spacing at each DM, in samples of delay across the band, from the
    rule the grid is built with: for the S/N-loss grid (sps_common.dm_grid.snr_loss_dt_grid,
    max_snr_loss > 0) max(1, 2 eta w_eff), w_eff = sqrt(pulse_width^2 + 1 + (s d)^2) with
    s the lowest channel's smearing per sample of delay d; for the uniform grid
    (max_snr_loss 0 or None) the constant dm_step. A smooth function of DM: the grid's
    trials are this rule's steps rounded to whole samples.
    """
    dms = np.atleast_1d(np.asarray(dms, dtype=np.float64))
    if not max_snr_loss:
        return np.full(dms.shape, float(dm_step))
    eta = np.sqrt(1 / (1 - max_snr_loss) ** 2 - 1)
    smear_per_dt = (
        2 * (FREQ_TOP - FREQ_BOTTOM) / nchan / FREQ_BOTTOM**3
        / (FREQ_BOTTOM**-2 - FREQ_TOP**-2)
    )  # fmt: skip
    d = np.abs(dms) * _DT_PER_DM / TSAMP
    return np.maximum(1.0, 2 * eta * np.sqrt(pulse_width**2 + 1 + (smear_per_dt * d) ** 2))


def dm_step_smearing(dms, nchan, max_snr_loss=DEFAULT_MAX_SNR_LOSS, dm_step=DEFAULT_DM_STEP,
                     tsamp=TSAMP):
    """The RMS DM-step smearing (s) at each DM: (dm_grid_step / 2) / sqrt(3) samples."""
    return dm_grid_step(dms, nchan, max_snr_loss, dm_step) / 2 / np.sqrt(3) * tsamp


def smearing_cutoff(
    dms, nchan, ref_freq_mhz=DEFAULT_REF_FREQ_MHZ, factor=DEFAULT_FACTOR, tsamp=TSAMP,
    max_snr_loss=DEFAULT_MAX_SNR_LOSS, dm_step=DEFAULT_DM_STEP,
):
    """
    Cutoff frequency (Hz) for each DM: factor / t_eff, at most the Nyquist frequency, t_eff
    including the DM-step smearing of the grid (see dm_grid_step).

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
    max_snr_loss, dm_step: float, int
        The DM grid (dedisp.fdmt.max_snr_loss and dm_step): the S/N-loss grid for
        max_snr_loss > 0, else the uniform grid of every dm_step-th delay.
    """
    if nchan <= 0 or ref_freq_mhz <= 0 or factor <= 0:
        raise ValueError("nchan, ref_freq_mhz and factor must be positive.")
    chan_width = (FREQ_TOP - FREQ_BOTTOM) / nchan
    dms = np.atleast_1d(np.asarray(dms, dtype=np.float64))
    smear = 2 * DM_CONSTANT * np.abs(dms) * chan_width / ref_freq_mhz**3
    t_dm = dm_step_smearing(dms, nchan, max_snr_loss, dm_step, tsamp)
    t_eff = np.sqrt(tsamp**2 + smear**2 + t_dm**2)
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
    The full settings of a freq_cut configuration: nchan (required), ref_freq_mhz, factor,
    max_snr_loss and dm_step (the DM grid; defaults filled in, max_snr_loss None stored as
    0, the uniform grid), as stored in a stack's freq_cut_* attributes.
    """
    unknown = set(freq_cut) - {"nchan", "ref_freq_mhz", "factor", "max_snr_loss", "dm_step"}
    if unknown:
        raise ValueError(f"Unknown freq_cut settings: {sorted(unknown)}")
    if freq_cut.get("nchan") is None:
        raise ValueError("freq_cut needs the number of channels (nchan).")
    return {
        "nchan": int(freq_cut["nchan"]),
        "ref_freq_mhz": float(freq_cut.get("ref_freq_mhz", DEFAULT_REF_FREQ_MHZ)),
        "factor": float(freq_cut.get("factor", DEFAULT_FACTOR)),
        "max_snr_loss": float(freq_cut.get("max_snr_loss", DEFAULT_MAX_SNR_LOSS) or 0.0),
        "dm_step": int(freq_cut.get("dm_step", DEFAULT_DM_STEP)),
    }


def stored_freq_cut(h5f):
    """
    The freq_cut settings a stack was written with, or None. Stacks written before the
    DM grid was part of the settings get the defaults (the grid then in use).
    """
    if "freq_cut_nchan" not in h5f.attrs:
        return None
    a = h5f.attrs
    return {
        "nchan": int(a["freq_cut_nchan"]),
        "ref_freq_mhz": float(a["freq_cut_ref_freq_mhz"]),
        "factor": float(a["freq_cut_factor"]),
        "max_snr_loss": float(a.get("freq_cut_max_snr_loss", DEFAULT_MAX_SNR_LOSS)),
        "dm_step": int(a.get("freq_cut_dm_step", DEFAULT_DM_STEP)),
    }
