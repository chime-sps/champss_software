import argparse

from ps_processes.ps_pipeline import PowerSpectraPipeline


def main():
    """Run power spectrum search with sps database."""
    parser = argparse.ArgumentParser(
        description="Run power spectrum processes with sps database interaction."
    )
    parser.add_argument(
        "--obs",
        help="The observation ID of the data to process.",
    )
    parser.add_argument(
        "--basepath",
        type=str,
        help="Path to the base directory of the SPS data products.",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        help="Path to the sub directory containing the SPS data products.",
    )
    parser.add_argument(
        "--no-ps",
        dest="ps_creation",
        action="store_false",
        help="Do not create power spectra from the dedispersed time series.",
    )
    parser.add_argument(
        "--no-search",
        dest="ps_search",
        action="store_false",
        help="Do not search the power spectra for detections.",
    )
    parser.add_argument(
        "--stack",
        dest="ps_stack",
        action="store_true",
        help="Stack the power spectra created to the monthly stack.",
    )
    parser.add_argument(
        "--write-spectra",
        dest="write_ps",
        action="store_true",
        help=(
            "Write power spectra created to"
            " 'basepath/subdir/<ra>_<dec>_power_spectra.hdf5'."
        ),
    )
    parser.add_argument(
        "--no-write-detections",
        dest="write_ps_detections",
        action="store_false",
        help=(
            "Do not write power spectra detections to"
            " 'basepath/subdir/<ra>_<dec>_power_spectra_detections.hdf5'."
        ),
    )
    parser.add_argument(
        "--no-update-db",
        dest="db",
        action="store_false",
        help="Do not update the sps-database.",
    )
    parser.add_argument(
        "--tsamp",
        type=float,
        default=0.00098304,
        help="The sampling time of the data.",
    )
    parser.add_argument(
        "--no-norm",
        dest="norm",
        action="store_false",
        help="Do not normalise the dedispersed time series.",
    )
    parser.add_argument(
        "--no-bary",
        dest="bary",
        action="store_false",
        help="Do not barycentre the dedispersed time series.",
    )
    parser.add_argument(
        "--padded_length",
        dest="padded",
        type=int,
        default=1048576,
        help="Set the padded length of the dedispersed time series.",
    )
    parser.add_argument(
        "--no-qc",
        dest="qc",
        action="store_false",
        help="Do not run quality control on the dedispersed time series.",
    )
    parser.add_argument(
        "--no-zero-replace",
        dest="replace",
        action="store_false",
        help=(
            "Do not run zero replacement of bad frequency values in the power spectra."
        ),
    )
    parser.add_argument(
        "--no-rednoise",
        dest="rednoise",
        action="store_false",
        help="Do not run rednoise removal on the dedispersed time series.",
    )
    parser.add_argument(
        "--harm",
        type=int,
        default=32,
        help="Number of harmonics to search the data with.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="The minimum sigma for candidate search.",
    )
    parser.add_argument(
        "--precompute-harms",
        dest="precompute",
        action="store_true",
        help=(
            "Precomputes the indices used for harmonic summing prior to the harmonic"
            " search operation."
        ),
    )
    args = parser.parse_args()
    obs_id = args.obs
    basepath = args.basepath
    subdir = args.subdir
    run_ps_creation = args.ps_creation
    run_ps_search = args.ps_search
    run_ps_stack = args.ps_stack
    write_ps = args.write_ps
    write_ps_detections = args.write_ps_detections
    update_db = args.db
    tsamp = args.tsamp
    normalise = args.norm
    barycenter = args.bary
    padded_length = args.padded
    qc = args.qc
    zero_replace = args.replace
    remove_rednoise = args.rednoise
    num_harm = args.harm
    if num_harm not in [1, 2, 4, 8, 16, 32]:
        print("Setting number of harmonics to 32")
        num_harm = 32
    sigma_min = args.sigma
    precompute_harms = args.precompute

    ps_processes = PowerSpectraPipeline(
        run_ps_creation,
        run_ps_search,
        run_ps_stack,
        write_ps,
        write_ps_detections,
        update_db,
        tsamp,
        normalise,
        barycenter,
        padded_length,
        qc,
        zero_replace,
        remove_rednoise,
        num_harm,
        sigma_min,
        precompute_harms,
        qc,
    )
    ps_processes.process(obs_id, basepath, subdir)


if __name__ == "__main__":
    main()
