import datetime
import click
from omegaconf import OmegaConf
import logging
import beamformer.skybeam as bs
from beamformer.strategist.strategist import PointingStrategist
from sps_databases import db_utils
from sps_pipeline import dedisp
import multiprocessing

import matplotlib

matplotlib.use("Agg")  # Avoids overhead of interactable plot GUI backends
from sps_pipeline import utils
# from sps_common.interfaces import SinglePointingCandidate_FFA, SearchAlgorithm_FFA

from .FFA_search import FFA_search, get_folding_pars


def apply_logging_config(level):
    """
    Applies logging settings from the given configuration
    Logging settings are under the 'logging' key, and include:
    - format: string for the `logging.formatter`
    - level: logging level for the root logger
    - modules: a dictionary of submodule names and logging level to be applied to that submodule's logger
    """
    log_stream.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s >> %(message)s", datefmt="%b %d %H:%M:%S"
        )
    )
    logging.root.setLevel(level)
    log.debug("Set default level to: %s", level)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--psr",
    type=str,
    default=None,
    required=False,
    help="PSR of known pulsar. Default is None",
)
@click.option(
    "--date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Date of data to process.",
)
@click.option(
    "--plot/--no-plot",
    default=False,
    help="Whether or not to generate periodogram plots and pulse profile at best DM",
)
@click.option(
    "--generate_candidates/--no_candidates",
    "-c",
    default=False,
    help="Whether or not to generate candidates for the given observation",
)
@click.option(
    "--stack/--no-stack", "-s", default=False, help="Whether to stack the periodograms"
)
@click.option("--ra", type=click.FloatRange(-180, 360), default=None)
@click.option("--dec", type=click.FloatRange(-90, 90), default=None)
@click.option(
    "--dm_downsample",
    type=int,
    default=8,
    help="The downsampling factor in DM trials. Must be an integer >= 1. Default is 8",
)
@click.option(
    "--dm_min", default=0, help="Only search for DMs above this value. Default is 0"
)
@click.option(
    "--dm_max",
    type=float,
    default=None,
    help="Only search for DMs smaller than this value. Default is None",
)
@click.option("--num_threads", type=int, default=16, help="Number of threads to use")
def main(
    psr,
    date,
    plot,
    generate_candidates,
    stack,
    ra,
    dec,
    dm_downsample,
    dm_min,
    dm_max,
    num_threads,
):
    """
    Simulates part of the pipeline in order to generate dedispersed time series. For testing purposes mostly
    Example:
    FFA_search.py --psr B0154+61 --date 20240612 --stack --generate_candidates
    Input: See click options
    Output: If specified, creates candidates, plots, and/or writes to the periodogram stacks. Nothing returned directly
    """

    apply_logging_config("INFO")
    db_utils.connect(host="sps-archiver1", port=27017, name="sps-processing")

    datetime.date = utils.convert_date_to_datetime(date)
    date_string = date.strftime("%Y/%m/%d")

    config_file = "/home/ltarabout/FFA/sps_config_ffa.yml"
    config = OmegaConf.load(config_file)
    if psr is not None:
        ra, dec = get_folding_pars(psr)
    elif ra is None or dec is None:
        log.error("Please provide either a PSR or a position in (ra,dec)")
        return
    if plot:
        generate_candidates = True

    pst = PointingStrategist(create_db=False)
    ap = pst.get_single_pointing(ra, dec, date)
    ra = ap[0].ra
    dec = ap[0].dec
    obs_id = ap[0].obs_id
    # print(ap)

    sbf = bs.SkyBeamFormer(
        extn="dat",
        update_db=False,
        min_data_frac=0.5,
        basepath="/mnt/beegfs-client/raw/",
        # basepath="/data/chime/sps/raw/",
        add_local_median=True,
        detrend_data=True,
        detrend_nsamp=32768,
        masking_timescale=512000,
        run_rfi_mitigation=True,
        masking_dict=dict(
            weights=True,
            l1=True,
            badchan=True,
            kurtosis=False,
            mad=False,
            sk=True,
            powspec=False,
            dummy=False,
        ),
        beam_to_normalise=1,
    )
    skybeam, spectra_shared = sbf.form_skybeam(ap[0], num_threads=num_threads)

    dedisp_ts = dedisp.run_fdmt(ap[0], skybeam, config, num_threads)

    FFA_search(
        dedisp_ts,
        obs_id,
        date,
        ra,
        dec,
        True,
        False,
        plot,
        False,
        generate_candidates,
        stack,
        dm_downsample,
        dm_min,
        dm_max,
        2,
        5,
        7,
        0.005,
        num_threads,
        "./",
    )

    del dedisp_ts
    del skybeam, spectra_shared, sbf, pst, ap

    return


log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)  # import in this way for easy reload

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
