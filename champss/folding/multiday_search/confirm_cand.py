import datetime

import click
import numpy as np
from folding.archive_utils import read_par
from multiday_search.load_profiles import load_profiles
from multiday_search.phase_aligned_search import ExploreGrid
from sps_databases import db_api, db_utils


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--fs_id",
    type=str,
    default="",
    help="FollowUpSource ID, to fold from database values",
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database.",
)
@click.option(
    "--db-host",
    default="sps-archiver",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
)
@click.option(
    "--phase_accuracy",
    type=float,
    default=1.0 / 64,
    help="required accuracy in pulse phase, which determines step size",
)
@click.option(
    "--full_plot",
    is_flag=True,
    help="Apply P0 and P1 from phase search to archives and display diagnostic plot",
)
@click.option(
    "--write-to-db",
    is_flag=True,
    help="Set folded_status to True in the processes database.",
)
def main(
    fs_id,
    db_port,
    db_host,
    db_name,
    phase_accuracy,
    full_plot=False,
    write_to_db=False,
):
    db_utils.connect(host=db_host, port=db_port, name=db_name)
    print(fs_id)
    source = db_api.get_followup_source(fs_id)
    print(source)
    source_type = source.source_type
    if source_type != "md_candidate":
        log.error(f"Source {fs_id} is not a multi-day candidate, exiting...")
        return

    if source.folding_history:
        fold_dates = [entry["date"].date() for entry in source.folding_history]
        fold_SN = [entry["SN"] for entry in source.folding_history]
        archives = [entry["archive_fname"] for entry in source.folding_history]
    else:
        log.error(f"Source {fs_id} has no folding history in db, exiting...")
        return

    par_file = source.path_to_ephemeris
    par_vals = read_par(par_file)
    DM_incoherent = par_vals["DM"]
    F0_incoherent = par_vals["F0"]
    RA = par_vals["RAJD"]
    DEC = par_vals["DECJD"]

    data = load_profiles(archives)

    # Compute on 1 day alias
    f0_min = F0_incoherent - 15e-6
    f0_max = F0_incoherent + 15e-6
    f0_lims = (f0_min, f0_max)
    delta_f0max = f0_max - f0_min

    f1_max = 1e-12  # Upper limit based on known pulsars, or expected barycentric shift
    f1_lims = (
        -f1_max,
        f1_max,
    )  # Negative or positive to account for position error
    delta_f1max = 2 * f1_max

    T = data["T"]  # Time from first observation to last observation
    print(T)
    npbin = data["npbin"]  # Number of phase bins
    M_f0 = int(npbin * phase_accuracy)
    # factor of 2, since we reference to central observation
    f0_points = 2 * int(delta_f0max * T * npbin / M_f0)
    f1_points = 2 * int(0.5 * delta_f1max * T**2 * npbin / M_f0)

    print(f"Running search with {f0_points} f0 bins, {f1_points} f1 bins")

    explore_grid = ExploreGrid(data, f0_lims, f1_lims, f0_points, f1_points)
    f0s, f1s, chi2_grid, optimal_parameters = explore_grid.output()

    np.savez(
        data["directory"] + "/explore_grid.npz", f0s=f0s, f1s=f1s, chi2_grid=chi2_grid
    )
    explore_grid.plot(squeeze=False)

    coherentsearch_summary = {
        "date": datetime.datetime.now(),
        "SN": np.max(explore_grid.SNmax),
        "f0": optimal_parameters[0],
        "f1": optimal_parameters[1],
        "profile": explore_grid.profiles_aligned.sum(0).tolist(),
        "gridsearch_file": data["directory"] + "/explore_grid.npz",
    }
    coherentsearch_history = source.coherentsearch_history
    search_dates = [entry["date"].date() for entry in coherentsearch_history]
    coherentsearch_history.append(coherentsearch_summary)
    if write_to_db:
        log.info("Updating FollowUpSource with coherent search results")
        db_api.update_followup_source(
            fs_id, {"coherentsearch_history": coherentsearch_history}
        )

    if not full_plot:
        # Silence Workflow errors, requires results, products, plots
        return coherentsearch_summary, [], []
    else:
        # Rewrite new ephemeris using new P0 and P1

        f0_optimal = optimal_parameters[0] + F0_incoherent
        print(f0_optimal)
        f1_optimal = optimal_parameters[1]

        optimal_par_file = directory + "/" + "optimal_par"

        with open(par_file) as input:
            with open(optimal_par_file, "w") as output:
                for line in input:
                    if line.strip("\n")[0:2] != "F0":
                        output.write(line)
                ### rewrite without \t
                output.write("\t".join(["F0", str(f0_optimal)]) + "\n")
                output.write("\t".join(["F1", str(-f1_optimal)]) + "\n")

        subprocess.run(
            ["pam", "-E", optimal_par_file, "-e", ".newar", "cand*.ar"], cwd=directory
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
