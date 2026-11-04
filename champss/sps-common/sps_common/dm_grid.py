"""DM trial grids and the DM scale used to cluster detections on them."""

import numpy as np
from sps_common.constants import (
    FREQ_BOTTOM,
    FREQ_TOP,
    REFERENCE_DM_SPACING,
    TSAMP,
)


def snr_loss_dt_grid(
    nchan,
    max_dt,
    max_snr_loss=0.1,
    pulse_width=1.0,
    freq_bottom=FREQ_BOTTOM,
    freq_top=FREQ_TOP,
    tsamp=TSAMP,
):
    """
    Delay trials spaced so a DM offset costs at most max_snr_loss in a smeared pulse.

    Each step is 2 * eta * w_eff samples, at least 1, where w_eff adds pulse_width, one
    sample and the smearing of the lowest channel in quadrature. This matches the
    "snr_loss" method of dmtlib.generate_optimal_dm_grid. It is kept here so that the
    grid, and so the stacks using it, cannot change with a library update.

    Parameters
    =========
    nchan: int
        Number of frequency channels.
    max_dt: int
        The largest delay to search, in time samples across the band. Always included.
    max_snr_loss: float
        Maximum fractional S/N loss between trials, for a pulse in the time domain.
    pulse_width: float
        Intrinsic pulse width in time samples.

    Returns
    =======
    dt_trials: np.ndarray
        Sorted unique integer delays from 0 to max_dt.
    """
    if not 0 < max_snr_loss < 1:
        raise ValueError(f"max_snr_loss must be between 0 and 1, got {max_snr_loss}")

    eta = np.sqrt(1 / (1 - max_snr_loss) ** 2 - 1)
    # Smearing of the lowest channel per sample of delay across the band
    smear_per_dt = (
        2
        * (freq_top - freq_bottom)
        / nchan
        / freq_bottom**3
        / (1 / freq_bottom**2 - 1 / freq_top**2)
    )
    trials = [0.0]
    while trials[-1] < max_dt:
        w_eff = np.sqrt(pulse_width**2 + 1 + (smear_per_dt * trials[-1]) ** 2)
        trials.append(trials[-1] + max(1.0, 2 * eta * w_eff))
    trials[-1] = max_dt
    return np.unique(np.round(trials).astype(int))


def clustering_dm_spacing(dms, reference=REFERENCE_DM_SPACING):
    """
    DM scale for clustering: the spacing of a uniform grid, otherwise a fixed reference.

    The first gap of a non-uniform grid says nothing about the rest of it, and using it
    would silently change the DM distance over which detections are linked.

    Parameters
    =========
    dms: np.ndarray
        DM trials.
    reference: float
        Scale to use when the trials are not uniformly spaced.

    Returns
    =======
    spacing: float
        The DM scale in pc/cm^3.
    """
    gaps = np.diff(dms)
    if np.allclose(gaps, gaps[0]):
        return dms[1] - dms[0]
    return reference
