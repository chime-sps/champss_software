#!/usr/bin/env python3
import argparse
import logging
from glob import glob

import numpy as np
import prometheus_client
from rfi_mitigation.pipeline import RFIPipeline
from rfi_mitigation.reader import DataReader

log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
logging.root.setLevel(logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log_stream.setFormatter(
    logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s >> %(message)s", "%b %d %H:%M:%S"
    )
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("file_path", help="Path to *.msg files")

    parser.add_argument(
        "-n", metavar="NAME", help="Target source name", type=str, default="Unknown"
    )

    parser.add_argument(
        "-p",
        metavar="RA DEC",
        help=(
            'Sky coordinates as a single space-separated string ("hh:mm:ss.s'
            ' dd:mm:ss.s")'
        ),
        type=str,
        default="05:34:31 22:00:52",
    )

    parser.add_argument(
        "-b",
        metavar="BEAM_NUM",
        help="CHIME/FRB beam number (purely house-keeping)",
        type=int,
        default=162,
    )

    parser.add_argument(
        "-i",
        metavar=("S_IDX", "E_IDX"),
        help="First and last second of data to load",
        nargs=2,
        type=int,
        default=[0, 100],
    )

    parser.add_argument(
        "-l",
        metavar="NUM_TO_LOAD",
        type=int,
        help="Number of msgpack files to load simultaneously",
        default=4,
    )

    parser.add_argument(
        "-d",
        help=(
            "Downsample the data in time and frequency by this factor. First number"
            " corresponds to time downsampling factor and must divide 1024. Second"
            " number corresponds to frequency downsampling and must divide 16384."
        ),
        nargs=2,
        metavar=("TFACT", "FFACT"),
        type=int,
        default=[1, 1],
    )

    parser.add_argument(
        "--sk", help="Run the Spectral Kurtosis filter", action="store_true"
    )
    parser.add_argument(
        "--mad", help="Run the Median Absolute Deviation filter", action="store_true"
    )
    parser.add_argument("--kur", help="Apply kurtosis filter", action="store_true")
    parser.add_argument(
        "--ps", help="Run the Power Spectrum filter", action="store_true"
    )
    parser.add_argument(
        "--l1", help="Apply the L1 data mask from msgpack data", action="store_true"
    )
    parser.add_argument(
        "--badchan", help="Apply the empirical bad channel mask", action="store_true"
    )
    parser.add_argument(
        "--chanthresh",
        help="Apply a mask to channel with >75%% of data masked",
        action="store_true",
    )
    parser.add_argument(
        "--quant",
        help="Quantize the data before filtering (mimics predicted input data)",
        action="store_true",
    )
    parser.add_argument(
        "--callback",
        help="Is the data recorded from intensity callbacks from L4 to L1?",
        action="store_true",
    )
    parser.add_argument(
        "--raw", help="is the data raw msgpack data?", action="store_true"
    )
    parser.add_argument(
        "--profile",
        help="Run profiling code to examine performace",
        action="store_true",
    )
    parser.add_argument("--plots", help="Create diagnostic plots", action="store_true")

    args = parser.parse_args()

    file_path = args.file_path
    srcname = args.n
    srcra, srcdec = args.p.split(" ")
    srcra = srcra.replace(":", "")
    srcdec = srcdec.replace(":", "")
    beam = f"{args.b:04d}"
    start, end = args.i
    loadmax = args.l
    tfact, ffact = args.d

    apply_l1 = args.l1
    apply_badchan = args.badchan
    quantize = args.quant
    apply_sk = args.sk
    apply_mad = args.mad
    apply_kur = args.kur
    apply_powspec = args.ps
    apply_chanthresh = args.chanthresh
    is_callback = args.callback
    is_raw = args.raw

    if is_callback:
        msgpack_list = np.sort(np.array(glob(f"{file_path}/*.msgpack")))
    elif is_raw:
        msgpack_list = np.sort(np.array(glob(f"{file_path}/*.msg")))
    else:
        msgpack_list = np.sort(np.array(glob(f"{file_path}/*.dat")))

    # split the total requested time into 100 second chunks
    ntoload = abs(end - start)
    if ntoload > loadmax:
        log.warning(f"request > {loadmax} seconds total, will split processing")
        nbatches = (
            ntoload // loadmax if ntoload % loadmax == 0 else ntoload // loadmax + 1
        )
        log.debug(f"will process in {nbatches} batches...")

        file_ranges = [
            (start + i * loadmax, start + (i + 1) * loadmax)
            for i in range(ntoload // loadmax)
        ]
        if ntoload % loadmax != 0:
            # add the left over time to the final file
            file_ranges.append(
                (file_ranges[-1][-1], file_ranges[-1][-1] + ntoload % loadmax)
            )
    else:
        file_ranges = [(start, start + ntoload)]

    log.debug(f"first msgpack file = {file_ranges[0][0]}")
    log.debug(f"last msgpack file = {file_ranges[-1][-1]}")

    masking_dict = dict(
        badchan=apply_badchan,
        kurtosis=apply_kur,
        mad=apply_mad,
        sk=apply_sk,
        powspec=apply_powspec,
        dummy=False,
    )
    # initialise RFI Pipeline and Data Reader instances
    log.debug("initialise RFIPipeline")
    rfiPipe = RFIPipeline(masking_dict, make_plots=args.plots)
    log.debug("initialise DataReader")
    reader = DataReader(apply_l1_mask=apply_l1)

    if args.profile:
        import cProfile
        import io
        import pstats

        pr = cProfile.Profile()
        pr.enable()

    for ifile, (fs, fe) in enumerate(file_ranges):
        if len(msgpack_list[fs:fe]) > 0:
            # read data from disk into the correct class structure
            log.debug("read files")
            chunks = reader.read_files(
                msgpack_list[fs:fe],
                is_callback=is_callback,
                is_raw=is_raw,
                sps_freq_downsamp_factor=ffact,
            )
            # clean the data and store the cleaned chunks ready for later processing
            log.debug("clean files")
            cleaned_chunks = rfiPipe.clean(chunks)

            # dump cleaned data to HDF5 files, one chunk per file
            log.debug("write output")
            for c in cleaned_chunks:
                c.write()

            prometheus_client.write_to_textfile(
                "pipeline_batch_db.prom", registry=prometheus_client.REGISTRY
            )

    if args.profile:
        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(0.1)
        print(s.getvalue())
