import click
import numpy as np
from folding.archive_utils import *
from load_profiles import *
from phase_aligned_search import *


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--dm", type=float, required=True, help="DM")
@click.option("--f0", type=float, required=True, help="F0")
@click.option("--ra", type=float, required=True, help="RA")
@click.option("--dec", type=float, required=True, help="DEC")
@click.option(
    "--phase_accuracy",
    type=float,
    default=1.0 / 64,
    help="required accuracy in pulse phase, which determines step size",
)
@click.option(
    "--load_only",
    is_flag=True,
    help="Load archives without changing their epoch",
)
@click.option(
    "--full_plot",
    is_flag=True,
    help="Apply P0 and P1 from phase search to archives and display diagnostic plot",
)
@click.option(
    "--no_search",
    is_flag=True,
    help="Don't perform P0/P1 search",
)
def main(
    ra, dec, dm, f0, phase_accuracy, load_only=False, full_plot=False, no_search=False
):
    # Find values directly from par file
    directory = (
        "/data/chime/sps/archives/candidates/"
        + f"{round(ra,2)}"
        + "_"
        + f"{round(dec,2)}"
    )
    print(f"Searching in directory {directory}...")
    # par_file = find_oldest_obs(directory, f0, dm, message=False)
    par_file = find_central_obs(directory, f0, dm, message=False)
    par_vals = read_par(par_file)

    DM_incoherent = par_vals["DM"]
    F0_incoherent = par_vals["F0"]
    P0_incoherent = 1 / F0_incoherent
    RA = par_vals["RAJD"]
    DEC = par_vals["DECJD"]

    if not no_search:
        data = load_profiles(
            directory, F0_incoherent, DM_incoherent, RA, DEC, load_only
        )

        # Eventually delta f0_max will be determined using power vs. freq plot
        f0_min = F0_incoherent - 15e-6
        f0_max = F0_incoherent + 15e-6
        f0_lims = (f0_min, f0_max)
        delta_f0max = f0_max - f0_min

        f1_max = (
            1e-13  # Upper limit based on known pulsars, or expected barycentric shift
        )
        f1_lims = (
            -f1_max,
            f1_max,
        )  # Negative or positive to account for position error
        delta_f1max = 2 * f1_max

        T = data["T"]  # Time from first observation to last observation
        npbin = data["npbin"]  # Number of phase bins
        M_f0 = int(npbin * phase_accuracy)
        # factor of 2, since we reference to central observation
        f0_points = 2 * int(delta_f0max * T * npbin / M_f0)
        f1_points = 2 * int(0.5 * delta_f1max * T**2 * npbin / M_f0)

        print(f"Running search with {f0_points} f0 bins, {f1_points} f1 bins")

        explore_grid = ExploreGrid(data, f0_lims, f1_lims, f0_points, f1_points)
        f0s, f1s, chi2_grid, optimal_parameters = explore_grid.output()

        np.savez(directory + "/explore_grid.npz", f0s=f0s, f1s=f1s, chi2_grid=chi2_grid)

        explore_grid.plot(squeeze=False)

    if full_plot:
        # Rewrite new ephemeris using new P0 and P1

        f0_optimal = optimal_parameters[0] + F0_incoherent
        print(f0_optimal)
        f1_optimal = optimal_parameters[1]

        first_obs_par = find_oldest_obs(directory, F0_incoherent, DM_incoherent)
        optimal_par_file = directory + "/" + "optimal_par"

        with open(first_obs_par) as input:
            with open(optimal_par_file, "w") as output:
                for line in input:
                    if line.strip("\n")[0:2] != "F0":
                        output.write(line)
                output.write("\t".join(["F0", str(f0_optimal)]) + "\n")
                output.write("\t".join(["F1", str(-f1_optimal)]) + "\n")

        subprocess.run(
            ["pam", "-E", optimal_par_file, "-m", "cand*.newar"], cwd=directory
        )

        # Create archive scrunched in time
        subprocess.run(["pam", "-T", "-e", ".T", "cand*.newar"], cwd=directory)

        # Create archive scrunched in frequency
        subprocess.run(["pam", "-F", "-e", ".F", "cand*.newar"], cwd=directory)

        # Create archive scrunched in both
        subprocess.run(["pam", "-FT", "-e", ".FT", "cand*.newar"], cwd=directory)

        # Concatenate the modified archives
        subprocess.run(["psradd", "*.T", "-o", "added.T"], cwd=directory)

        explore_grid.plot(squeeze=True)


if __name__ == "__main__":
    main()
