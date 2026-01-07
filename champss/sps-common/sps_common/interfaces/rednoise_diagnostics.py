import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from ps_processes.processes.ps_stack import PowerSpectraStack
from sps_common.interfaces import PowerSpectra

def get_from_stack(stack_path):

    ps_stack = PowerSpectra.read(stack_path)
    
    try:
        medians = ps_stack.rn_medians
        scales = ps_stack.rn_scales
        scales = scales.astype('int')
        DMs = ps_stack.dms[ps_stack.rn_dm_indices[0].astype(int)]
        freq_labels = ps_stack.freq_labels
        return freq_labels, DMs, medians, scales
    
    except:
        print("This power spectrum does not have saved rednoise information. Please rerun power spectrum creation with saving enabled.")
        return


def freq_to_period(x):
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, 1 / x, np.inf)

def period_to_freq(x):
    return np.where(x != 0, 1 / x, np.inf)

def plot_medians(freq_labels, DMs, medians, scales, title = 'Rednoise Medians'):
    
    fig, ax = plt.subplots(1, figsize = (10, 8))
    N = medians.shape[1]
    norm = Normalize(vmin=DMs.min(), vmax=DMs.max())
    colors = plt.cm.jet(norm(DMs))
    
    for day in range(medians.shape[0]):
        print(f'day: {day + 1}')
        day_medians = medians[day]
        day_scales = scales[day]
        day_freqs = freq_labels[day]
        if len(day_scales.shape) == 2:
            freq_idx = np.cumsum(day_scales[0]) #only need one... people keep changing the shape of this :/
        else:
            freq_idx = np.cumsum(day_scales)
        all_freqs = np.zeros((len(day_medians), len(day_freqs)))
        all_freqs[:] = day_freqs
        if len(day_freqs) == len(day_scales):
            segments = [np.column_stack([all_freqs[i], day_medians[i]]) for i in range(N)]
        else:
            segments = [np.column_stack([all_freqs[i, freq_idx], day_medians[i]]) for i in range(N)]
        lc = LineCollection(segments, colors=colors, alpha = 0.01, norm = norm)
        ax.add_collection(lc)


    secax = ax.secondary_xaxis('top', functions=(freq_to_period, period_to_freq))
    secax.set_xlabel('Period (s)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Median Power')
    ax.set_title(title)
    cbar = fig.colorbar(ScalarMappable(cmap=plt.cm.jet, norm = norm), ax=ax)
    cbar.set_label('DMs')
    plt.legend()
    plt.show()
