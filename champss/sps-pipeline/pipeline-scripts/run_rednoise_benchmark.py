import numpy as np
from sps-common.sps_common.utilities import rednoise_diagnostics

median_path = './benchmark/2022/06/*/*/medians.npz'

info = np.load(median_path)
freq_labels = info['freq_labels']
DMs = info['dms']
medians = info['medians']
scales = info['scales']

rednoise_diagnostics.plot_medians(freq_labels, DMs, medians, scales, title = 'Benchmark Rednoise Medians')
