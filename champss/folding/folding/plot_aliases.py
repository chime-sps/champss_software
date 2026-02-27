import glob
import os
import re

import click
import matplotlib.pyplot as plt
import numpy as np

from folding.utilities.archives import get_SN, readpsrarch


def get_alias_factor_from_label(label):
    """
    Convert alias label to numeric factor.

    Parameters
    ----------
    label : str
        Alias label (e.g., "1_16", "1_2", "2", "16")

    Returns
    -------
    factor : float
        Numeric alias factor
    """
    if "_" in label:
        parts = label.split("_")
        return float(parts[0]) / float(parts[1])
    else:
        return float(label)


def plot_aliases(alias_results, output_path=None):
    """
    Plot profiles and S/N vs alias factor.

    Parameters
    ----------
    alias_results : dict
        Dictionary mapping alias labels to archive file paths,
        as returned by fold_candidate with --fold-aliases
    output_path : str, optional
        Path to save the plot. If None, saves to directory of first archive

    Returns
    -------
    output_path : str
        Path to the saved plot
    """
    if not alias_results:
        print("No alias data to plot")
        return None

    # Load data from archives
    alias_data = {}
    for label, archive_path in alias_results.items():
        # Use .FT file if available
        ft_path = archive_path.replace(".ar", ".FT")
        if os.path.exists(ft_path):
            read_path = ft_path
        else:
            read_path = archive_path

        factor = get_alias_factor_from_label(label)

        try:
            data, params = readpsrarch(read_path)
            # Sum over time and frequency to get profile
            profile = data.squeeze()
            sn = get_SN(profile)

            alias_data[label] = {
                "factor": factor,
                "profile": profile,
                "params": params,
                "sn": sn,
                "file": read_path,
            }
        except Exception as e:
            print(f"Error reading {read_path}: {e}")
            continue

    if not alias_data:
        print("No alias data could be loaded")
        return None

    # Sort by factor
    sorted_labels = sorted(alias_data.keys(), key=lambda x: alias_data[x]["factor"])
    n_aliases = len(sorted_labels)

    # Extract data for plotting
    factors = [alias_data[label]["factor"] for label in sorted_labels]
    sns = [alias_data[label]["sn"] for label in sorted_labels]
    profiles = [alias_data[label]["profile"] for label in sorted_labels]

    # Normalize profiles for display
    profiles_norm = []
    for prof in profiles:
        prof_norm = prof - np.median(prof)
        prof_norm = prof_norm / np.max(np.abs(prof_norm))
        profiles_norm.append(prof_norm)

    # Create figure
    fig = plt.figure(figsize=(10, 6))

    # Left panel: profiles as waterfall
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    plt.subplots_adjust(wspace=0.05)

    print(len(profiles_norm), profiles_norm[0].shape)

    for i in range(len(profiles_norm)):
        profile = profiles_norm[i]
        xaxis = np.linspace(0, 1, len(profile))
        ax1.plot(xaxis, profile + i, color='k')

    # Set y-ticks to show alias factors
    ytick_positions = np.arange(n_aliases)
    ytick_labels = [f"{alias_data[label]['factor']:.3g}" for label in sorted_labels]
    ax1.set_yticks(ytick_positions)
    ax1.set_yticklabels(ytick_labels)

    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Alias Factor")
    ax1.set_title("Profiles at Different Alias Factors")

    # Right panel: S/N vs alias factor
    ax2 = plt.subplot2grid((1, 3), (0, 2))

    ax2.semilogy(sns, factors, "o")
    ax2.set_ylabel("Alias Factor")
    ax2.set_xlabel("S/N")
    ax2.set_title("S/N vs Alias Factor")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    # Save figure
    if output_path is None:
        first_file = list(alias_results.values())[0]
        output_path = os.path.join(os.path.dirname(first_file), "alias_plot.png")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved alias plot to {output_path}")

    # Print summary
    print("\nAlias S/N Summary:")
    print("-" * 40)
    for label in sorted_labels:
        factor = alias_data[label]["factor"]
        sn = alias_data[label]["sn"]
        print(f"  {factor:8.4f}x : S/N = {sn:6.1f}")

    return output_path


def load_alias_dict_from_dir(alias_dir):
    """
    Construct alias_results dictionary from files in a directory.

    Parameters
    ----------
    alias_dir : str
        Path to the aliases directory containing alias_*.ar files

    Returns
    -------
    alias_results : dict
        Dictionary mapping alias labels to archive file paths
    """
    alias_results = {}

    # Find all .ar files in the directory
    ar_files = glob.glob(os.path.join(alias_dir, "alias_*.ar"))

    for ar_file in ar_files:
        # Extract alias label from filename (e.g., "alias_1_2.ar" -> "1_2")
        basename = os.path.basename(ar_file)
        match = re.match(r"alias_(.+)\.ar", basename)
        if match:
            label = match.group(1)
            alias_results[label] = ar_file

    return alias_results


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("alias_dir", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for the plot. Default: alias_dir/alias_plot.png",
)
def main(alias_dir, output):
    """
    Plot profiles and S/N for frequency alias folding results.

    ALIAS_DIR is the path to the directory containing alias_*.ar files.
    """
    alias_results = load_alias_dict_from_dir(alias_dir)

    if not alias_results:
        print(f"No alias_*.ar files found in {alias_dir}")
        return

    plot_aliases(alias_results, output_path=output)


if __name__ == "__main__":
    main()
