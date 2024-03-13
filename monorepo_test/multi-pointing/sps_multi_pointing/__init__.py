"""Combine multiple SinglePointingCandidates."""

import datetime as dt
import logging
import os
from glob import glob

import click
import numpy as np
import pandas as pd
from easydict import EasyDict
from numpy.lib.recfunctions import structured_to_unstructured
from omegaconf import OmegaConf
from sps_common.constants import MIN_SEARCH_FREQ
from sps_common.interfaces.multi_pointing import KnownSourceLabel
from sps_common.interfaces.single_pointing import SinglePointingCandidateCollection
from sps_databases import db_api, db_utils

from sps_multi_pointing import classifier, grouper
from sps_multi_pointing.known_source_sifter.known_source_sifter import KnownSourceSifter

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


def apply_logging_config(config):
    """
    Applies logging settings from the given configuration.

    Logging settings are under the 'logging' key, and include:
    - format: string for the `logging.formatter`
    - level: logging level for the root logger
    - modules: a dictionary of submodule names and logging level to be applied to
        that submodule's logger
    """
    log_stream.setFormatter(
        logging.Formatter(fmt=config.logging.format, datefmt="%b %d %H:%M:%S")
    )

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
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
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
    default="localhost",
    type=str,
    help="Host used for the mongodb database.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database.",
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
):
    """Slow Pulsar Search multiple-pointing candidate processing."""
    config = load_config()
    apply_logging_config(config)
    run_label = f"spsmp_{dt.datetime.now().isoformat()}"
    out_folder = f"{os.path.abspath(output)}/mp_runs/{run_label}"
    os.makedirs(out_folder, exist_ok=False)
    os.makedirs(out_folder + "/candidates/", exist_ok=False)
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
            # "path_candidate_file":{"$ne":None}
        }
        all_obs = list(db_client.observations.find(query))
        files = [obs["path_candidate_file"] for obs in all_obs]
    else:
        log.error("Need to use either --file-path or --get-from-db")
    sp_cands = []
    log.info(f"Number of candidate collections: {len(files)}")
    if len(files) == 0:
        log.error("No files found. Will exit.")
        return
    for file in files:
        try:
            spcc = SinglePointingCandidateCollection.read(file)
            # datetimes may not be included in the canidates already
            datetimes = spcc.candidates[0].datetimes
            if not datetimes:
                try:
                    datetimes = db_api.get_dates(spcc.candidates[0].obs_id)
                except:
                    log.error("Could not grab datetimes. May not have access to db.")
                    datetimes = []
            for cand_index, candidate in enumerate(spcc.candidates):
                cand_summary = EasyDict(candidate.summary)
                cand_summary["file_name"] = file
                cand_summary["cand_index"] = cand_index
                cand_summary["features"] = candidate.features
                cand_summary["datetimes"] = datetimes

                sp_cands.append(cand_summary)
        except Exception as e:
            log.error(f"Can't process file {file} because of {e}.")
    log.info(f"Number of single-pointing candidates: {len(sp_cands)}")
    # Run Grouper
    sp_grouper = grouper.SinglePointingCandidateGrouper(
        **OmegaConf.to_container(config.grouper)
    )
    mp_cands = sp_grouper.group(sp_cands)
    log.info(f"Number of multi-pointing candidates: {len(mp_cands)}")
    try:
        db_client = db_utils.connect(host=db_host, port=db_port, name=db_name)
        kss = KnownSourceSifter(**OmegaConf.to_container(config.sifter))
    except:
        log.error("Could not grab known source list.")
        kss = None
    cand_classifier = classifier.CandidateClassifier(
        **OmegaConf.to_container(config.classifier)
    )

    # For now throw some diagnostic metrics into that dict, could be extended to use
    # a property of the candidate
    summary_dicts = []

    for cand in mp_cands:
        # Run Classifier
        cand = cand_classifier.classify(cand)

        # Run KSS
        if kss is not None:
            cand = kss.classify(cand, pos_filter=True)
        if cand.known_source.label == KnownSourceLabel.Known:
            log.info(
                "Candidate: %s -- (%.2f, %.2f) f=%.3f, DM=%.1f, sigma=%.3f",
                ", ".join(cand.known_source.matches["source_name"]),
                cand.ra,
                cand.dec,
                cand.best_freq,
                cand.best_dm,
                cand.best_sigma,
            )
            if len(cand.obs_id) <= 1 and db is True:
                obs = db_api.get_observation(cand.obs_id[0])
                detection_dict = {
                    "ra": cand.ra,
                    "dec": cand.dec,
                    "freq": cand.best_freq,
                    "dm": cand.best_dm,
                    "sigma": cand.best_sigma,
                    "obs_id": cand.obs_id,
                    "datetime": obs.datetime,
                }
                for c in cand.known_source.matches:
                    pulsar = c["source_name"]
                    ks = db_api.get_known_source_by_name(pulsar)[0]
                    pre_exist = False
                    new_index = 0
                    for i, b in reversed(list(enumerate(ks.detection_history))):
                        if b["obs_id"] == cand.obs_id:
                            pre_exist = True
                            if b["sigma"] < cand.best_sigma:
                                ks.detection_history[i] = detection_dict
                                payload = {
                                    "detection_history": vars(ks)["detection_history"]
                                }
                                db_api.update_known_source(ks._id, payload)
                            break
                        if b["datetime"].date() < obs.datetime.date():
                            new_index = i + 1
                            break
                    if pre_exist is False:
                        ks.detection_history.insert(new_index, detection_dict)
                        payload = {"detection_history": vars(ks)["detection_history"]}
                        db_api.update_known_source(ks._id, payload)
        if db:
            pointing_id = db_api.get_observation(cand.obs_id[0]).pointing_id
            payload = {
                "ra": cand.ra,
                "dec": cand.dec,
                "sigma": cand.best_sigma,
                "freq": cand.best_freq,
                "dm": cand.best_dm,
                "dc": cand.best_dc,
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
        # Save the candidate to file
        # `Multi_Pointing_Groups_f_<<F>>_DM_<<DM>>_class_<<Astro|RFI|Ambiguous>>.npz`
        mp_cand_file_base = f"{out_folder}/candidates/Multi_Pointing_Groups"
        mp_cand_file_name = cand.write(mp_cand_file_base)
        plot_path = ""
        if plot_cands:
            if plot_all_pulsars and cand.known_source.label.value:
                plot_path = cand.plot_candidate(path=f"{out_folder}/plots/")
            else:
                if cand.best_freq > (MIN_SEARCH_FREQ * 2):
                    if cand.best_sigma > plot_threshold:
                        if cand.best_dm >= plot_dm_threshold:
                            plot_path = cand.plot_candidate(path=f"{out_folder}/plots/")
        if plot_path != "":
            plot_path = os.path.join(os.getcwd(), plot_path)

        if csv:
            cand_dict = {
                "mean_freq": cand.mean_freq,
                "mean_dm": cand.mean_dm,
                "sigma": cand.best_sigma,
                "ra": cand.ra,
                "dec": cand.dec,
                "ncands": len(cand.all_summaries),
                "delta_ra": cand.position_features["delta_ra"],
                "delta_dec": cand.position_features["delta_dec"],
                "file_name": mp_cand_file_name,
                "plot_path": plot_path,
                "known_source_label": cand.known_source.label.value,
            }
            if cand.known_source.label.value:
                # Extracting only the first entry of the known sources
                cand_ks_dict = {
                    "known_source_likelihood": cand.known_source.matches["likelihood"][
                        0
                    ],
                    "known_source_name": cand.known_source.matches["source_name"][0],
                    "known_source_p0": cand.known_source.matches["spin_period_s"][0],
                    "known_source_dm": cand.known_source.matches["dm"][0],
                    "known_source_ra": cand.known_source.matches["pos_ra_deg"][0],
                    "known_source_dec": cand.known_source.matches["pos_dec_deg"][0],
                }
                cand_dict = {**cand_dict, **cand_ks_dict}
                # previously this used an operation which only worked after Python 3.9
                # cand_dict = cand_dict | cand_ks_dict

            summary_dicts.append(cand_dict)

    if csv:
        df = pd.DataFrame(summary_dicts)
        csv_name = f"{out_folder}/all_mp_cands.csv"
        df.to_csv(csv_name)

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
