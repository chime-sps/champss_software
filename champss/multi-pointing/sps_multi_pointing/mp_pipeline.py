"""Combine multiple SinglePointingCandidates."""

import datetime as dt
import logging
import os
import shutil
from functools import partial
from glob import glob
from multiprocessing import Pool

import click
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scheduler.utils import convert_date_to_datetime
from sps_common.interfaces.multi_pointing import KnownSourceLabel
from sps_databases import db_api, db_utils
from sps_multi_pointing import classifier, data_reader, grouper, utilities
from sps_multi_pointing.known_source_sifter.known_source_sifter import KnownSourceSifter
import tqdm

log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)


def load_config():
    """
    Combines default and user-specified configuration settings.

    User-specified settings can be given in two forms: as a YAML file in the
    current directory named "sps_config.yml", or as command-line arguments.

    The format of the file is (all sections optional):
    ```
    logging:
      format: string for the `logging.formatter`
      level: logging level for the root logger
      modules:
        module_name: logging level for the submodule `module_name`
        module_name2: etc.
    ```

    Returns
    -------
    The `omegaconf` configuration object merging all the default configuration
    with the (optional) user-specified overrides.
    """
    base_config = OmegaConf.load(os.path.dirname(__file__) + "/sps_config.yml")
    if os.path.exists("./sps_config.yml"):
        user_config = OmegaConf.load("./sps_config.yml")
    else:
        user_config = OmegaConf.create()

    return OmegaConf.merge(base_config, user_config)


def apply_logging_config(config, log_file="./logs/default.log"):
    """
    Applies logging settings from the given configuration.

    Logging settings are under the 'logging' key, and include:
    - format: string for the `logging.formatter`
    - level: logging level for the root logger
    - modules: a dictionary of submodule names and logging level to be
               applied to that submodule's logger
    """
    log_stream.setFormatter(
        logging.Formatter(fmt=config.logging.format, datefmt="%b %d %H:%M:%S")
    )

    if config.logging.get("file_logging", False):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(fmt=config.logging.format, datefmt="%b %d %H:%M:%S")
        )
        logging.root.addHandler(file_handler)
    logging.root.setLevel(config.logging.level.upper())
    log.debug("Set default level to: %s", config.logging.level)

    if "modules" in config.logging:
        for module_name, level in config.logging["modules"].items():
            logging.getLogger(module_name).setLevel(level.upper())
            log.debug("Set %s level to: %s", module_name, level)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-o",
    "--output",
    default="./",
    type=str,
    help="Base path for the output.",
)
@click.option(
    "--file-path",
    default=None,
    type=str,
    help="Path to candidates files.",
)
@click.option(
    "--get-from-db/--do-not-get-from-db",
    default=False,
    is_flag=True,
    help="Read file locations from database instead of using --file-path.",
)
@click.option(
    "--date",
    type=click.DateTime(
        ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d:%H", "%Y-%m-%d-%H", "%Y/%m/%d/%H"]
    ),
    required=False,
    help="First date of candidates when grabbing from db. Default = All days.",
)
@click.option(
    "--ndays",
    default=1,
    type=float,
    help="Number of days to process when --date is used for the first day.",
)
@click.option(
    "--plot",
    default=False,
    is_flag=True,
    help="Plots the multi pointing grouper output",
)
@click.option(
    "--plot-cands/--no-plot-cands",
    default=False,
    help="Whether to create candidate plots",
)
@click.option("--db/--no-db", default=False, help="Whether to write to database")
@click.option("--csv/--no-csv", default=True, help="Whether to write summary csv.")
@click.option(
    "--plot-threshold",
    default=0.0,
    type=float,
    help="Sigma threshold above which the candidate plots are created",
)
@click.option(
    "--plot-dm-threshold",
    default=2.0,
    type=float,
    help="DM threshold above which the candidate plots are created",
)
@click.option(
    "--plot-all-pulsars/--no-plot-all-pulsars",
    default=False,
    help="Plot all known pulsars if plot_cands is turned on.",
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
    "--num-threads",
    default=32,
    type=int,
    help="Number of threads for parallel jobs.",
)
@click.option(
    "--run-name",
    default=None,
    type=str,
    help="Name of the output folder inside the output/mp_runs folder.",
)
def cli(
    output,
    file_path,
    get_from_db,
    date,
    ndays,
    plot,
    plot_cands,
    plot_all_pulsars,
    db,
    csv,
    plot_threshold,
    plot_dm_threshold,
    db_port,
    db_host,
    db_name,
    num_threads,
    run_name,
):
    """Slow Pulsar Search multiple-pointing candidate processing."""
    date = convert_date_to_datetime(date)

    config = load_config()
    if run_name:
        run_name = run_name.replace("/", "")
        run_label = run_name
    else:
        run_label = f"spsmp_{dt.datetime.now().isoformat()}"
    out_folder = f"{os.path.abspath(output)}/mp_runs/{run_label}"
    if os.path.isdir(out_folder):
        log.error(f"Found existing folder {out_folder}. Will delete that folder.")
        shutil.rmtree(out_folder)
    os.makedirs(out_folder, exist_ok=False)
    os.makedirs(out_folder + "/candidates/", exist_ok=False)
    log_file = out_folder + "/run.log"
    apply_logging_config(config, log_file=log_file)
    pool = Pool(num_threads)
    if plot_cands:
        os.makedirs(out_folder + "/plots/", exist_ok=False)
    with open(f"{out_folder}/run_parameters.txt", "a") as run_parameters:
        run_parameters.write(str(locals()))

    if db or get_from_db:
        db_client = db_utils.connect(host=db_host, port=db_port, name=db_name)
    # Load the files
    if file_path:
        log.info("Getting files from folder.")
        files = glob(file_path + "/*_candidates.npz")
    elif get_from_db:
        log.info("Getting files from database")
        query = {
            "datetime": {"$gte": date, "$lte": date + dt.timedelta(days=ndays)},
            "path_candidate_file": {"$ne": None},
        }
        all_obs = list(db_client.observations.find(query))
        files = [obs["path_candidate_file"] for obs in all_obs]
    else:
        log.error("Need to use either --file-path or --get-from-db")

    log.info(f"Number of candidate collections: {len(files)}")
    if len(files) == 0:
        log.error("No files found. Will exit.")
        return
    sp_cands = list(
        tqdm.tqdm(pool.imap(data_reader.read_cands_summaries, files), total=len(files))
    )
    # Filter out None
    sp_cands = [sp_cand_list for sp_cand_list in sp_cands if sp_cand_list is not None]
    # Get dates if not already done. Only needed for old candidates prior 2024/03
    for sp_cand_list in sp_cands:
        old_dates = sp_cand_list[0].get("datetimes", None)
        if len(old_dates) == 0:
            datetimes = db_api.get_dates(sp_cand_list[0].obs_id)
            for cand in sp_cand_list:
                cand.datetimes = datetimes
    # Transform list of lists to list
    sp_cands = [sp_cand for sp_cand_list in sp_cands for sp_cand in sp_cand_list]
    log.info(f"Number of single-pointing candidates: {len(sp_cands)}")
    # np.save("all_cands.npy", sp_cands)
    # np.savez("all_cands.npz", sp_cands)

    # Run Grouper
    sp_grouper = grouper.SinglePointingCandidateGrouper(
        **OmegaConf.to_container(config.grouper)
    )
    mp_cands = sp_grouper.group(sp_cands, num_threads=num_threads)
    log.info(f"Number of multi-pointing candidates: {len(mp_cands)}")
    try:
        db_client = db_utils.connect(host=db_host, port=db_port, name=db_name)
        kss = KnownSourceSifter(**OmegaConf.to_container(config.sifter))
    except Exception as e:
        log.error(f"Could not grab known source list. {e}")
        kss = None
    cand_classifier = classifier.CandidateClassifier(
        **OmegaConf.to_container(config.classifier)
    )

    # For now throw some diagnostic metrics into that dict, could be extended to use
    # a property of the candidate
    log.info(
        "Will now write out candidates, run known source sifter and create candidate plots."
    )
    proc_output = list(
        tqdm.tqdm(
            pool.imap(
                partial(
                    utilities.process_mp_candidate,
                    cand_classifier,
                    kss,
                    csv,
                    plot_cands,
                    plot_threshold,
                    plot_dm_threshold,
                    plot_all_pulsars,
                    out_folder,
                ),
                mp_cands,
            ),
            total=len(mp_cands),
        )
    )

    mp_cands = [single_output[0] for single_output in proc_output]
    summary_dicts = [single_output[1] for single_output in proc_output]

    log.info("Now running database updates.")
    # Run db operations which are tricky in multiprocessing calls
    detected_known_pulsars = 0
    for cand in mp_cands:
        if cand.known_source.label == KnownSourceLabel.Known:
            detected_known_pulsars += 1
            """
            log.info( "Candidate: %s -- (%.2f, %.2f) f=%.3f, DM=%.1f, sigma=%.3f", ",
            ".join(cand.known_source.matches["source_name"]), cand.ra, cand.dec,
            cand.best_freq, cand.best_dm,

            cand.best_sigma, )
            """
            if db is True:
                obs = db_api.get_observation(cand.obs_id[0])
                detection_dict = {
                    "ra": cand.ra,
                    "dec": cand.dec,
                    "freq": cand.best_freq,
                    "dm": cand.best_dm,
                    "sigma": cand.best_sigma,
                    "obs_id": cand.obs_id.tolist(),
                    "datetime": obs.datetime,
                }
                for c in cand.known_source.matches:
                    pulsar = c["source_name"]
                    ks = db_api.get_known_source_by_names(pulsar)[0]
                    # For now only update detection history with single obs
                    if len(cand.obs_id) <= 1:
                        pre_exist = False
                        new_index = 0
                        for i, b in reversed(list(enumerate(ks.detection_history))):
                            if b["obs_id"] == cand.obs_id:
                                pre_exist = True
                                if b["sigma"] < cand.best_sigma:
                                    ks.detection_history[i] = detection_dict
                                    payload = {
                                        "detection_history": vars(ks)[
                                            "detection_history"
                                        ]
                                    }
                                    db_api.update_known_source(ks._id, payload)
                                break
                            if b["datetime"].date() < obs.datetime.date():
                                new_index = i + 1
                                break
                        if pre_exist is False:
                            ks.detection_history.insert(new_index, detection_dict)
                            payload = {
                                "detection_history": vars(ks)["detection_history"]
                            }
                            db_api.update_known_source(ks._id, payload)
                    # Updated strongest detection in pointings
                    for summary in cand.all_summaries:
                        obs_id = summary["obs_id"]
                        # May want to include the pointing id in the summaries?
                        pointing_id = db_api.get_observation(obs_id[0]).pointing_id
                        pointing_dict = {
                            "sigma": summary["sigma"],
                            "freq": summary["freq"],
                            "dm": summary["dm"],
                        }
                        if len(cand.obs_id) <= 1:
                            pointing_dict["obs_id"] = obs_id
                            update_stack = False
                        else:
                            # Saving all ob_ids might bloat the pointing db
                            pointing_dict["num_days"] = len(obs_id)
                            update_stack = True
                        updated_pointing = db_api.update_pulsars_in_pointing(
                            pointing_id, pulsar, pointing_dict, stack=update_stack
                        )
        """
        I don't think  in the current state without filtering
        we want to write candidates to the database.
        if db:
            pointing_id = db_api.get_observation(cand.obs_id[0]).pointing_id
            payload = {
                "ra": cand.ra,
                "dec": cand.dec,
                "sigma": cand.best_sigma,
                "freq": cand.best_freq,
                "dm": cand.best_dm,
                "num_days": len(cand.obs_id),
                "classification_label": cand.classification.label,
                "classification_grade": cand.classification.grade,
                "known_source_label": cand.known_source.label,
                "known_source_matches": structured_to_unstructured(
                    cand.known_source.matches
                )
                .astype(str)
                .tolist(),
                "observation_id": [str(idx) for idx in cand.obs_id],
                "pointing_id": pointing_id,
            }
            db_api.create_candidate(payload)
        """
    if csv:
        df = pd.DataFrame(summary_dicts)
        csv_name = f"{out_folder}/all_mp_cands.csv"
        df.to_csv(csv_name)
    else:
        csv_name = ""

    if plot:
        import matplotlib.pyplot as plt

        plt.switch_backend("AGG")

        fig = plt.figure(figsize=(10, 5))
        ra_dec = np.asarray(
            [[cand.ra, cand.dec, cand.known_source.label.value] for cand in mp_cands]
        )
        ra_dec_known = ra_dec[np.where(ra_dec[:, 2] == 1)]
        ra_dec = np.unique(ra_dec, axis=0)
        plt.scatter(
            ra_dec[:, 0], ra_dec[:, 1], c="k", s=10, alpha=0.5
        )  # marker=[shape[idx] for idx in ra_dec[:, 2]])
        plt.scatter(ra_dec_known[:, 0], ra_dec_known[:, 1], c="r", s=50, marker="*")
        plt.xlabel("RA")
        plt.ylabel("Dec")
        summary_plot_path = f"{out_folder}/Multi_Pointing_Groups.png"
        plt.savefig(summary_plot_path, bbox_inches="tight", dpi=240)
        plt.close(fig)
        output_plots = [summary_plot_path]
    else:
        output_plots = []

    return (
        {
            "container_name": os.environ.get("CONTAINER_NAME"),
            "node_name": os.environ.get("NODE_NAME"),
            "log_file": log_file,
            "num_files": len(files),
            "num_sp_cands": len(sp_cands),
            "num_mp_cands": len(mp_cands),
            "known_pulsar_detections": detected_known_pulsars,
            "csv_file": csv_name,
        },
        [],
        output_plots,
    )
