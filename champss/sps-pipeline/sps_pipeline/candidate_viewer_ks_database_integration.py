import click
import numpy as np

import sps_pipeline.candidate_viewer as cv
from sps_databases import db_utils, db_api
from sps_common.interfaces import MultiPointingCandidate


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--survey",
    type=str,
    required=True,
    help="The survey used on the candidate viewer site, eg dailycands, stackcands.",
)
@click.option(
    "--folder",
    type=str,
    default=None,
    help="The folder withing the survey. If not given all folders will be used",
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
    default="sps-processing",
    type=str,
    help="Name used for the mongodb database.",
)
def move_derived_values_to_ks_db(
    survey,
    folder,
    db_port,
    db_host,
    db_name,
):
    db = db_utils.connect(host=db_host, port=db_port, name=db_name)
    db_config = {
        "user": "automation",
        "password": "",  # no password for automation user
        "host": "sps-archiver1",
        "database": "champss",
        "port": 3306,
    }
    cqv = cv.CandidateViewerQuery(survey, db_config)

    all_psr_cands = cqv.get_ratings(
        folder=folder, classification="<any_known>", with_metadata=True
    )

    success = 0
    for site_cand in all_psr_cands:
        print(site_cand["result"])
        ks = db_api.get_known_source_by_names(site_cand["result"])
        if len(ks):
            ks = ks[0]
        else:
            print(f"{site_cand['result']} not found in db.")
            continue
        try:
            cand = MultiPointingCandidate.read(site_cand["metadata"]["input_file"])
        except FileNotFoundError:
            print(f"{site_cand['metadata']['input_file']} not found.")
            continue
        champss_derived_parameters = {
            "dm": cand.best_dm,
            "dm_error": cand.best_candidate_features[
                "dm_sigma_FitGaussWidth_gauss_sigma"
            ][0],
            "spin_period_s": 1 / cand.best_freq,
            "spin_period_s_error": cand.best_candidate_features[
                "freq_sigma_FitGaussWidth_gauss_sigma"
            ][0]
            / (cand.best_freq**2),
            # "pos_error_semimajor_deg": float(cand.position_features["std_ra"]),
            # "pos_error_semiminor_deg": float(cand.position_features["std_dec"]),
        }
        for value in champss_derived_parameters.values():
            if not np.isfinite(value):
                print(f"Parameters for {site_cand['result']} contain nan.")
                continue
        if champss_derived_parameters["dm_error"] > 20:
            print(
                f"DM Error for {site_cand['result']} too big ({champss_derived_parameters['dm_error']})."
            )
            continue
        freq_error = cand.best_candidate_features[
            "freq_sigma_FitGaussWidth_gauss_sigma"
        ][0]
        if freq_error > 0.002:
            print(f"Frequency error for {site_cand['result']} too big ({freq_error}).")
            continue

        payload = {"champss_derived_parameters": champss_derived_parameters}
        db_api.update_known_source(ks._id, payload)
        print(f"Updated {site_cand['result']} with {champss_derived_parameters}")
        success += 1
    print(f"Updated {success} candidates successfully.")


if __name__ == "__main__":
    move_derived_values_to_ks_db()
