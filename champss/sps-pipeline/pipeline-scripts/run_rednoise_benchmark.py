import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from glob import glob
from sps_common.interfaces import rednoise_diagnostics
import os

median_pattern = './benchmark/2022/06/*/*/medians.npz'
median_files = sorted(glob(median_pattern))

if not median_files:
    raise FileNotFoundError(f"No medians.npz files found matching pattern: {median_pattern}")

print(f"Found {len(median_files)} median files:")
for f in median_files:
    print(f"  {f}")

# Create output directory for plots
output_dir = './benchmark/rednoise_plots'
os.makedirs(output_dir, exist_ok=True)

# Loop through all median files and create benchmark plots
for median_path in median_files:
    print(f"\nProcessing median file: {median_path}")

    info = np.load(median_path)
    freq_labels = info['freq_labels']
    DMs = info['dms']
    medians = info['medians']
    scales = info['scales']

    # Extract date from path for plot title (e.g., ./benchmark/2022/06/18/...)
    date_str = median_path.split('/')[3]  # Gets '18' from the path
    title = f'Benchmark Rednoise Medians - 2022/06/{date_str}'

    # Plot medians (this will create the figure but not display it due to Agg backend)
    rednoise_diagnostics.plot_medians(freq_labels, DMs, medians, scales, title=title)

    # Save the plot
    output_file = os.path.join(output_dir, f'rednoise_medians_202206{date_str}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Saved plot to: {output_file}")

print(f"\nAll plots saved to: {output_dir}")
