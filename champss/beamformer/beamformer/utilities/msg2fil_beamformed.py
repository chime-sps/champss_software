import argparse
import datetime
import glob
import sys

import numpy as np
import pytz
from astropy.time import Time
import cfbm as beam_model
from iautils import cascade
from iautils.conversion import chime_intensity, filterbank
from sps_common.constants import FREQ_BOTTOM, FREQ_TOP, TSAMP, L1_NCHAN

beammod = beam_model.current_model_class(beam_model.current_config)

SCRUNCH_FACTORS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# CHIME channel bandwidth
CHANNEL_BANDWIDTH = (FREQ_TOP - FREQ_BOTTOM) / L1_NCHAN


def create_filterbank_header(cas, filename, psr, ra_hms, dec_hms, start_time_mjd):
    # create filterbank header with presets and from file
    fil_header = {}
    fil_header["telescope_id"] = 20  # number for CHIME
    fil_header["machine_id"] = 20  # number for CHIME/FRB backend
    fil_header["data_type"] = 1  # filterbank
    # PRESTO and sigproc read at most 80 characters for header messages
    fil_header["rawdatafile"] = filename[:80]
    fil_header["source_name"] = psr
    fil_header["barycentric"] = 0
    fil_header["pulsarcentric"] = 0
    fil_header["src_raj"] = ra_hms  # [hhmmss.s]
    fil_header["src_dej"] = dec_hms  # [ddmmss.s]
    fil_header["tstart"] = start_time_mjd
    fil_header["tsamp"] = 0.00098304
    fil_header["nbits"] = 32
    fil_header["nbeams"] = 1
    fil_header["ibeam"] = 0
    # first channel `fch1` in sigproc is the highest frequency
    # `foff` is negative to signify this
    ftop = 800.1953125
    fbottom = 400.1953125
    channel_bandwidth = float(np.abs(ftop - fbottom)) / cas.max_beam.nchan
    fil_header["fch1"] = ftop - channel_bandwidth / 2.0  # MHz
    fil_header["foff"] = -1.0 * channel_bandwidth  # MHz
    fil_header["nchans"] = cas.max_beam.nchan
    fil_header["nifs"] = 1

    return fil_header


def convert_ra_dec(ra_deg, dec_deg):
    hhr = ("0" + str(int(ra_deg * 24 / 360)))[-2:]
    mmr = ("0" + str(int((ra_deg * 24 / 360 - int(ra_deg * 24 / 360)) * 60)))[-2:]
    ssr = (
        (ra_deg * 24 / 360 - int(ra_deg * 24 / 360)) * 60
        - int((ra_deg * 24 / 360 - int(ra_deg * 24 / 360)) * 60)
    ) * 60
    ssr = ("0" + "%.2f" % ssr)[-5:]
    ra_hms = hhr + mmr + ssr

    if dec_deg < 0:
        dec_deg = -1.0 * dec_deg
        hht = ("0" + str(int(dec_deg)))[-2:]
        hhd = "-" + hht
    else:
        hhd = ("0" + str(int(dec_deg)))[-2:]
    mmd = ("0" + str(int((dec_deg - int(dec_deg)) * 60)))[-2:]
    ssd = (
        "0"
        + "%.2f"
        % (((dec_deg - int(dec_deg)) * 60 - int((dec_deg - int(dec_deg)) * 60)) * 60)
    )[-5:]
    dec_hms = hhd + mmd + ssd

    return ra_hms, dec_hms


def get_best_beam(ra_deg, dec_deg, y_beam, ref_time, ref_freq=480.5):
    """Get the brightest beam from a row of beams."""
    beam_arr = np.asarray([y_beam, y_beam + 1000, y_beam + 2000, y_beam + 3000])
    ref_pos = beammod.get_position_from_equatorial(ra_deg, dec_deg, ref_time)
    if not -0.6 <= ref_pos[0] <= 1.0:
        print("position is outside the grid. Exiting")
        return "out"
    ref_pos = np.asarray([[ref_pos]])
    ref_freq = np.asarray([ref_freq])
    beam_max = beammod.get_sensitivity(beam_arr, ref_pos, ref_freq).argmax()

    return beam_max


def get_beam_sen(ra_deg, dec_deg, y_beam, ref_time, fscrunch):
    """Get relative sensitivity of all 4 beams on a line The relative sensitivity will
    add up to 1 with only values > 0.5 retained.
    """
    beam_arr = np.asarray([y_beam, y_beam + 1000, y_beam + 2000, y_beam + 3000])
    ref_pos = beammod.get_position_from_equatorial(ra_deg, dec_deg, ref_time)
    if not -0.6 <= ref_pos[0] <= 1.0 or not ref_pos[1] <= 41.68:
        print("position is outside the grid. Exiting")
        return "out"
    ref_pos = np.asarray([ref_pos])
    ref_freq = np.asarray(np.arange(800.1953125, 400.1953125, -400 * fscrunch / 16384))
    beam_sen = beammod.get_sensitivity(beam_arr, ref_pos, ref_freq)[0, :, :]
    # compute the relative sensitivity
    for freq in range(len(beam_sen[0])):
        for beam in range(len(beam_sen[:, 0])):
            if beam_sen[beam, freq] < 0.5 * np.max(beam_sen[:, freq]):
                beam_sen[beam, freq] = 0
        beam_sen[:, freq] = beam_sen[:, freq] / np.sum(beam_sen[:, freq] ** 2)
        # Test without sqrt
        # beam_sen[:,freq] = np.sqrt(beam_sen[:,freq])

    return np.float32(beam_sen)


def get_start_time_RA_dec(msg, msg_len, beam_no):
    """Get start time and RA, Dec of a set of beams."""
    (
        intensity,
        weights,
        fpga0s,
        fpgaNs,
        binning,
        rfi_masks,
        frame0_nanos,
    ) = chime_intensity.unpack_datafiles([msg])
    # get start and mid time
    frame0_ctime = frame0_nanos[0] * 1e-9
    start_time = datetime.datetime.fromtimestamp(frame0_ctime)
    start_time += datetime.timedelta(0.0, fpga0s[0] * 2.56e-6)
    start_time_mjd = Time(start_time.isoformat(), format="isot").mjd
    mid_time = start_time + datetime.timedelta(0.0, (msg_len / 2) * fpgaNs[0] * 2.56e-6)
    # get RA DEC in deg
    y_pos = beammod._clamping(int(beam_no), beammod.clamp_freq)
    x_pos = 0
    ra_deg, dec_deg = beammod.get_equatorial_from_position(x_pos, y_pos[0][0], mid_time)
    return start_time, start_time_mjd, ra_deg, dec_deg, fpgaNs


def cas2fil(cas, outfile, fil_header, newfile=True, verbose=False):
    """
    Convert a Cascade object to a filterbank file.

    Parameters
    ----------
    cas : Cascade
        Cascade object to convert to filterbank file.
    outfile : file
        The filterbank file.
    verbose : bool
        Script will be talkative (default: script will be quiet).
    """
    if newfile:
        filterbank.create_filterbank_file(
            outfile,
            fil_header,
            cas.max_beam.intensity * cas.max_beam.weights,
            verbose=verbose,
        )
    else:
        filterbank.append_spectra(
            outfile, cas.max_beam.intensity * cas.max_beam.weights, verbose=verbose
        )


def caslist2fil(caslist, outfile, fil_header, newfile=True, verbose=False):
    """
    Convert a Cascade object to a filterbank file.

    Parameters
    ----------
    cas : Cascade
        Cascade object to convert to filterbank file.
    outfile : file
        The filterbank file.
    verbose : bool
        Script will be talkative (default: script will be quiet).
    """
    spectra = []
    for key, cas in caslist.items():
        if spectra == []:
            spectra = cas.max_beam.intensity * cas.max_beam.weights
        else:
            spectra += cas.max_beam.intensity * cas.max_beam.weights
    spectra_std = np.std(spectra, axis=1)
    spectra_std[spectra_std == 0.0] = 1.0

    if newfile:
        filterbank.create_filterbank_file(
            outfile, fil_header, spectra / spectra_std[:, np.newaxis], verbose=verbose
        )
    else:
        filterbank.append_spectra(
            outfile, spectra / spectra_std[:, np.newaxis], verbose=verbose
        )


def convert_chunk(msg_chunk, beam, dm, fscrunch=1, zerodm=False):
    """Load a list of msg files into a cascade.Cascade instance."""

    (
        intensity,
        weights,
        fpga0s,
        fpgaNs,
        binning,
        rfi_masks,
        frame0_nanos,
    ) = chime_intensity.unpack_datafiles(msg_chunk)

    if (
        not all(item is None for item in frame0_nanos)
        and np.unique(frame0_nanos).size > 1
    ):
        raise Exception(
            "Some of the msg files have different frame0 " + "times, aborting."
        )
    frame0_nanos = frame0_nanos[0]

    if frame0_nanos is None:
        frame0_nanos = 0
        print("WARNING old msgpack without frame0_nanos! No absolute time!")

    fpga_time = np.array(2.56 * fpga0s[0] + frame0_nanos / 1e3, dtype=np.int64)
    event_time = pytz.utc.localize(datetime.datetime.fromtimestamp(fpga_time / 1e6))

    print("Creating cascade object..")
    cas = cascade.Cascade(
        intensities=[intensity],
        weights=[weights],
        beam_nos=[beam],
        fpga0s=[fpga0s],
        fpgaNs=[fpgaNs],
        dm=dm,
        event_time=event_time,
        fpga_time=fpga_time,
        fbottom=FREQ_BOTTOM,
        dt=TSAMP * binning,
        df=CHANNEL_BANDWIDTH,
    )

    # replace missing channels with mean of good channels
    cas.find_and_replace_missing_rfi_packets()

    if fscrunch not in SCRUNCH_FACTORS:
        fscrunch = 4
        print("WARNING fscrunch is not a factor of 2.. setting it to 4!")

    nsub = 16384 // fscrunch
    # give warning when nchan > 4096
    if nsub > 4096:
        print(
            "WARNING sigproc spectrum lengths are capped at 4096 "
            + "channels by default, need to update `reader.h` and "
            + "`header.h` and recompile before reading this file!"
        )

    # subband the data
    cas.process_cascade(dm, nsub, dedisperse=False, subband=True, zerodm=zerodm)

    return cas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam0glob", dest="beam0", help="path to msgpacks for beam0")
    parser.add_argument("--beam1glob", dest="beam1", help="path to msgpacks for beam1")
    parser.add_argument("--beam2glob", dest="beam2", help="path to msgpacks for beam2")
    parser.add_argument("--beam3glob", dest="beam3", help="path to msgpacks for beam3")
    parser.add_argument(
        "--beamno",
        type=int,
        dest="beam_no",
        help="the beam number of the set of observation",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10,
        dest="chunksize",
        help="the chunk size in number of msgpacks to compute the beam to be chosen",
    )
    parser.add_argument("--psr", type=str, dest="psr", help="name of the pulsar")
    parser.add_argument(
        "-o", type=str, dest="filename", help="output filterbank file name"
    )
    parser.add_argument(
        "--fscrunch",
        type=int,
        default=1,
        dest="fscrunch",
        help="scrunch factor in frequency",
    )
    parser.add_argument(
        "--listonly",
        dest="list_only",
        action="store_true",
        help="write out a list of msgpack instead of creating the filterbank file",
    )
    parser.add_argument(
        "--radec",
        nargs=2,
        default=[None, None],
        dest="ra_dec",
        help="specify ra and dec",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        dest="add",
        help="add nearby beams together instead of appending brightest beam",
    )

    args = parser.parse_args()

    beam_msg = [[], [], [], []]
    beam_msg[0] = sorted(glob.glob(args.beam0 + "/*.msg*"))
    beam_msg[1] = sorted(glob.glob(args.beam1 + "/*.msg*"))
    beam_msg[2] = sorted(glob.glob(args.beam2 + "/*.msg*"))
    beam_msg[3] = sorted(glob.glob(args.beam3 + "/*.msg*"))
    msgchunksize = args.chunksize
    psr = args.psr
    filename = args.filename
    beam_no = args.beam_no
    fscrunch = args.fscrunch
    list_only = args.list_only
    ra_dec = args.ra_dec
    add = args.add

    if add:
        list_only = False

    print(len(beam_msg[0]), len(beam_msg[1]), len(beam_msg[2]), len(beam_msg[3]))
    # find the range of msgpack that are consistent with each other
    startmsg_0 = float(beam_msg[0][0].split("_")[-1].split(".")[0])
    startmsg_1 = float(beam_msg[1][0].split("_")[-1].split(".")[0])
    startmsg_2 = float(beam_msg[2][0].split("_")[-1].split(".")[0])
    startmsg_3 = float(beam_msg[3][0].split("_")[-1].split(".")[0])

    diff = [startmsg_1 - startmsg_0]
    diff.append(startmsg_2 - startmsg_0)
    diff.append(startmsg_3 - startmsg_0)

    startpack = np.zeros(4)
    if np.max(diff) < 0:
        startpack[0] = 0
        startpack[1] = -diff[0]
        startpack[2] = -diff[1]
        startpack[3] = -diff[2]
    else:
        startpack[0] = np.max(diff)
        startpack[1] = np.max(diff) - diff[0]
        startpack[2] = np.max(diff) - diff[1]
        startpack[3] = np.max(diff) - diff[2]
    startpack = [int(a) for a in startpack]

    beam_msg[0] = beam_msg[0][startpack[0] :]
    beam_msg[1] = beam_msg[1][startpack[1] :]
    beam_msg[2] = beam_msg[2][startpack[2] :]
    beam_msg[3] = beam_msg[3][startpack[3] :]

    endpack = np.min(
        [len(beam_msg[0]), len(beam_msg[1]), len(beam_msg[2]), len(beam_msg[3])]
    )

    beam_msg[0] = beam_msg[0][:endpack]
    beam_msg[1] = beam_msg[1][:endpack]
    beam_msg[2] = beam_msg[2][:endpack]
    beam_msg[3] = beam_msg[3][:endpack]

    print(beam_msg[0][0], beam_msg[1][0], beam_msg[2][0], beam_msg[3][0])

    if (
        len(beam_msg[0]) != len(beam_msg[1])
        or len(beam_msg[0]) != len(beam_msg[2])
        or len(beam_msg[0]) != len(beam_msg[3])
    ):
        print("Length of msgpack files of each beam is not equal. Exiting...")
        sys.exit(0)

    if beam_no > 255 or beam_no < 0:
        print("Beam number is not right. Exiting")
        sys.exit(0)

    # First, get details of the observation
    start_time, start_time_mjd, ra_deg, dec_deg, fpgaNs = get_start_time_RA_dec(
        beam_msg[0][0], len(beam_msg[0]), args.beam_no
    )

    if not ra_dec == [None, None]:
        ra_deg = float(ra_dec[0])
        dec_deg = float(ra_dec[1])

    # track the first block
    first_block = True

    msg_list = []
    for i in range(int(len(beam_msg[0]) / msgchunksize)):
        beam_ref_time = start_time + datetime.timedelta(
            0.0, (msgchunksize / 2 + i * msgchunksize) * fpgaNs[0] * 2.56e-6
        )
        if add:
            beam_sen = get_beam_sen(ra_deg, dec_deg, beam_no, beam_ref_time, fscrunch)
            if beam_sen == "out":
                continue
        else:
            beam_max = get_best_beam(
                ra_deg, dec_deg, beam_no, beam_ref_time, ref_freq=480.5
            )
            if beam_max == "out":
                continue
            if list_only:
                msg_list.extend(
                    beam_msg[beam_max][i * msgchunksize : (i + 1) * msgchunksize]
                )
                continue
        if add:
            caslist = {}
            for j in range(len(beam_sen[:, 0])):
                if np.max(beam_sen[j, :]) > 0:
                    caslist[f"cas{j}"] = convert_chunk(
                        beam_msg[j][i * msgchunksize : (i + 1) * msgchunksize],
                        beam_no + (j * 1000),
                        0,
                        fscrunch=fscrunch,
                        zerodm=False,
                    )
                    caslist[f"cas{j}"].max_beam.weights = (
                        caslist[f"cas{j}"].max_beam.weights
                        * beam_sen[j, :][:, np.newaxis]
                    )
        else:
            cas = convert_chunk(
                beam_msg[beam_max][i * msgchunksize : (i + 1) * msgchunksize],
                beam_no + (beam_max * 1000),
                0,
                fscrunch=fscrunch,
                zerodm=False,
            )

        if first_block:
            ra_hms, dec_hms = convert_ra_dec(ra_deg, dec_deg)
            outfile = open(filename, "wb")
            if add:
                fil_header = create_filterbank_header(
                    caslist[f"cas{j}"],
                    filename,
                    psr,
                    ra_hms,
                    dec_hms,
                    start_time_mjd + (i * msgchunksize / 86400.0),
                )
                caslist2fil(caslist, outfile, fil_header, newfile=True, verbose=False)
            else:
                fil_header = create_filterbank_header(
                    cas,
                    filename,
                    psr,
                    ra_hms,
                    dec_hms,
                    start_time_mjd + (i * msgchunksize / 86400.0),
                )
                cas2fil(cas, outfile, fil_header, newfile=True, verbose=False)
            first_block = False
        else:
            if add:
                caslist2fil(caslist, outfile, fil_header, newfile=False, verbose=False)
            else:
                cas2fil(cas, outfile, fil_header, newfile=False, verbose=False)

    if list_only:
        with open(filename.rstrip(".fil") + "_msglist.txt", "w+") as listfile:
            listfile.write(f"{ra_deg} {dec_deg}\n")
            for lines in msg_list:
                listfile.write(lines + "\n")


if __name__ == "__main__":
    main()
