import numpy as np


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
