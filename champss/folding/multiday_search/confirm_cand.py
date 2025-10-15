import datetime
import logging

import click
import numpy as np

# set these up before importing any SPS packages
log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from folding.archive_utils import read_par
from multiday_search.load_profiles import load_profiles, load_unwrapped_archives
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
    default="sps-archiver1",
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
    help="Required accuracy in pulse phase, which determines step size.",
)
@click.option(
    "--nday",
    default=0,
    type=int,
    help="Number of days to search. Default is to search all available archives."
)
@click.option(
    "--foldpath",
    default=None,
    type=str,
    help="Path for created files during fold step.",
)
@click.option(
    "--write-to-db",
    is_flag=True,
    help="Set folded_status to True in the processes database.",
)
@click.option(
    "--check-cands",
    is_flag=True,
    help="Output possible candidates.",
)
def main(
    fs_id,
    db_port,
    db_host,
    db_name,
    phase_accuracy,
    nday,
    foldpath,
    write_to_db=False,
    check_cands=False,
):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    if check_cands:
        print("Possible candidates:")
        candidates = db.followup_sources.find({"source_type": "md_candidate"})
        for cand in candidates:
            print(
                cand["_id"],
                cand["path_to_ephemeris"],
                len(cand["folding_history"]),
                "archives",
            )
        return

    source = db_api.get_followup_source(fs_id)
    print(fs_id, source)
    source_type = source.source_type
    if source_type != "md_candidate":
        log.error(f"Source {fs_id} is not a multi-day candidate, exiting...")
        return

    if source.folding_history:
        fold_dates = [entry["date"].date() for entry in source.folding_history]
        fold_SN = [entry["SN"] for entry in source.folding_history]
        archives = [entry["archive_fname"] for entry in source.folding_history]

        if len(archives) < 2:
            log.error(f"Source {fs_id} does not have more than 1 archive required to run the search, exiting...")
            return
    else:
        log.error(f"Source {fs_id} has no folding history in db, exiting...")
        return

    # sort by date
    combined = sorted(zip(fold_dates, fold_SN, archives), key=lambda x: x[0])
    fold_dates, fold_SN, archives = map(list, zip(*combined))

    # find consecutive chunks
    chunks = []
    current_chunk = [(fold_dates[0], fold_SN[0], archives[0])]

    for i in range(1, len(fold_dates)):
        if fold_dates[i] - fold_dates[i-1] <= datetime.timedelta(days=30):
            current_chunk.append((fold_dates[i], fold_SN[i], archives[i]))
        else:
            chunks.append(current_chunk)
            current_chunk = [(fold_dates[i], fold_SN[i], archives[i])]

    chunks.append(current_chunk)

    # keep the longest chunk, warn about trimmed ones
    longest = max(chunks, key=len)
    for ch in chunks:
        if ch is not longest:
            start, end = ch[0][0], ch[-1][0]
            log.warning(f"Source {fs_id}: trimming chunk from {start} to {end} (gap >30d)")

    fold_dates, fold_SN, archives = map(list, zip(*longest))

    # take only first nday if requested
    if nday:
        fold_dates = fold_dates[:nday]
        fold_SN = fold_SN[:nday]
        archives = archives[:nday]

    log.info(f"Folding {len(archives)} days of data: {fold_dates}")

    par_file = source.path_to_ephemeris
    par_vals = read_par(par_file)
    DM_incoherent = par_vals["DM"]
    F0_incoherent = par_vals["F0"]
    RA = par_vals["RAJD"]
    DEC = par_vals["DECJD"]

    data = load_profiles(archives)
    print(f"Loaded {len(data['profiles'])} profiles")

    if foldpath is not None:
        data["directory"] = f"{foldpath}/candidates/{RA:.02f}_{DEC:.02f}/"

    dF0 = 1 / 86164.1  # 1 day alias (can reduce by 2x if necessary)
    f0_min = F0_incoherent - dF0
    f0_max = F0_incoherent + dF0
    f0_lims = (f0_min, f0_max)
    delta_f0max = f0_max - f0_min

    f1_max = 1e-12  # Upper limit based on known pulsars, or expected barycentric shift
    f1_lims = (
        -f1_max,
        f1_max,
    )  # Negative or positive to account for position error
    delta_f1max = 2 * f1_max

    T = data["Tmax_from_reference"]  # Time from first observation to last observation
    npbin = data["npbin"]  # Number of phase bins
    M_f0 = npbin * phase_accuracy
    M_f0 = int(np.max((M_f0, 1)))  # To make sure M_f0 does not return 0
    # factor of 2, since we reference to central observation
    f0_points = 2 * int(delta_f0max * T * npbin / M_f0)
    f1_points = 2 * int(np.max((0.5 * delta_f1max * T**2 * npbin / M_f0, 1)))

    print(f"Running search with {f0_points} f0 bins, {f1_points} f1 bins")
    explore_grid = ExploreGrid(data, f0_lims, f1_lims, f0_points, f1_points)
    f0s, f1s, chi2_grid, optimal_parameters = explore_grid.output()

    np.savez(
        data["directory"] + f"cand_{F0_incoherent}_{DM_incoherent}_explore_grid.npz",
        f0s=f0s,
        f1s=f1s,
        chi2_grid=chi2_grid,
        SN=np.max(explore_grid.SNmax),
        nobs=len(data["profiles"]),
    )

    plot_name = explore_grid.plot(fullplot=False)
    coherentsearch_summary = {
        "date": datetime.datetime.now(),
        "SN": float(np.max(explore_grid.SNmax)),
        "f0": float(optimal_parameters[0]),
        "f1": float(optimal_parameters[1]),
        # "profile": explore_grid.profiles_aligned.sum(0).tolist(),
        "gridsearch_file": data["directory"] + "/explore_grid.npz",
        "path_to_plot": plot_name,
    }
    coherentsearch_summary["date"] = coherentsearch_summary["date"].strftime("%Y%m%d")
    if write_to_db:
        log.info("Updating FollowUpSource with coherent search results")
        db_api.update_followup_source(
            fs_id, {"coherentsearch_history": [coherentsearch_summary]}
        )

    # Rewrite new ephemeris using new F0 and F1

    f0_optimal = optimal_parameters[0]  # + F0_incoherent
    f1_optimal = optimal_parameters[1]

    optimal_par_file = par_file.replace(".par", "_optimal.par")
    if foldpath is not None:
        optimal_par_file = data['directory'] + optimal_par_file.split('/')[-1]

    print(f"Writing new par file to {optimal_par_file}")
    with open(par_file) as input:
        with open(optimal_par_file, "w") as output:
            output.write("# Created: " + str(datetime.datetime.now()) + "\n")
            output.write("# F0 and F1 from CHAMPSS coherent search\n")
            for line in input:
                key = line.split()[0]
                if key == "F0":
                    F0_output = f"F0 {str(f0_optimal)} 1 \n"
                    output.write(F0_output)
                    F1_output = f"F1 {str(-f1_optimal)} 1 \n"
                    output.write(F1_output)
                elif key == "RAJ":
                    RA_output = f"{line.strip()} 1 \n"
                    output.write(RA_output)
                elif key == "DECJ":
                    DEC_output = f"{line.strip()} 1 \n"
                    output.write(DEC_output)
                elif key == 'PEPOCH':
                    PEPOCH_output = f"PEPOCH {data['PEPOCH']} 0 \n"
                    output.write(PEPOCH_output)
                else:
                    output.write(line)

    explore_grid.plot(fullplot=True)

    return coherentsearch_summary, [plot_name], [plot_name]


if __name__ == "__main__":
    main()
