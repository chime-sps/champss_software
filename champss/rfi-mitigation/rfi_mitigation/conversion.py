# Deprecated functions moved here to free sps-common from spshuff dependency
import sys

import importlib
from spshuff import l1_io
from sps_common.filterbank import write_to_filterbank
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from sps_common.constants import TSAMP
import numpy as np

# Conditional import for beam_model
if importlib.util.find_spec("cfbm"):
    import cfbm as beam_model


def read_huff_msgpack(filename, channel_downsampling_factor=1):
    """
    Reader for Huffman encoded SPS data file. This function reads all data chunks
    from a single file and checks that:
        1) the time ordering is correct,
        2) that the number of frequency channels remains the same
           (if it does not then splits the file up accordingly into "subfiles"), and
        3) splits non-contiguous chunks into different subfiles to process.

    The output is essentially a list of dictionaries, where each dictionary
    contains all the data and metadata associated with a single contiguous subfile.

    This function is superseeded by the new reading function contained in beamformer.

    Parameters
    ==========
    filename: str
        file to read

    channel_downsampling_factor: int
        Channel downsampling factor relative to 16k channels.
        (i.e., the number of channels requested is 16384 // channel_downsampling_factor)

    Returns
    =======
    output_subfiles: list of dict
        A "subfile" contents, including the masked spectra (nchan, ntime), and various
        metadata values/flags.

    NOTES: The output of this function can be converted into a SlowPulsarIntensityChunk
    object by simply executing:
    >>> from sps_common.interfaces.rfi_mitigation import SlowPulsarIntensityChunk
    >>> spic = SlowPulsarIntensityChunk(**output_subfiles)


    """
    subfiles = []
    with open(filename, "rb") as f:
        int_file = l1_io.IntensityFile.from_file(
            f, shape=(16384 // channel_downsampling_factor, None)
        )
        fh = int_file.fh
        sps_chunks = int_file.get_chunks()
    last_fpga0 = 0
    last_nfreq = 0
    # sort the chunks by start time to loop through
    for i, chunk in enumerate(sorted(sps_chunks, key=lambda x: x.chunk_header.fpga0)):
        chunk_header = chunk.chunk_header
        if i == 0:
            subfiles.append([chunk])
            last_fpga0 = chunk_header.fpga0
            last_nfreq = chunk_header.nfreq
            continue
        # split into different subfiles if nchan is different, or they are not contiguous
        # (can be tuned to be not contiguous by n chunks and still be processed together)
        not_contiguous = (
            round((chunk_header.fpga0 - last_fpga0) * 2.56e-6 / TSAMP) != 1024
        )
        different_nchan = chunk_header.nfreq != last_nfreq
        if not_contiguous or different_nchan:
            subfiles.append([chunk])
        else:
            subfiles[-1].extend([chunk])
        last_fpga0 = chunk_header.fpga0
        last_nfreq = chunk_header.nfreq

    output_subfiles = []
    # produce a spic dict for each subfile
    for subfile in subfiles:
        nchan = subfile[0].chunk_header.nfreq
        ntime = (
            round(
                (subfile[-1].chunk_header.fpga0 - subfile[0].chunk_header.fpga0)
                * 2.56e-6
                / TSAMP
            )
            + 1024
        )
        start_fpga0 = subfile[0].chunk_header.fpga0
        start_time = Time(
            subfile[0].chunk_header.frame0_nano * 1e-9, format="unix", scale="utc"
        ) + TimeDelta(0.0, subfile[0].chunk_header.fpga0 * 2.56e-6, format="sec")
        intensity = np.zeros(shape=(nchan, ntime))
        rfi_mask = np.ones(shape=(nchan, ntime), dtype=bool)
        for i, chunk in enumerate(subfile):
            start_idx = round(
                (chunk.chunk_header.fpga0 - start_fpga0) * 2.56e-6 / TSAMP
            )
            chunk_time_slice = slice(start_idx, start_idx + chunk.chunk_header.ntime)
            chunk_intensity = chunk.data
            chunk_intensity *= np.sqrt(chunk.variance)[:, np.newaxis]
            chunk_intensity += chunk.means[:, np.newaxis]
            intensity[:, chunk_time_slice] = chunk_intensity
            zero_mask = np.zeros_like(chunk_intensity, dtype=bool)
            zero_mask[np.where(chunk_intensity <= 0)] = True
            zero_mask[~np.isfinite(chunk_intensity)] = True
            chunk_mask = chunk.bad_mask.astype(bool)
            full_mask = np.logical_or(zero_mask, ~chunk_mask)
            rfi_mask[:, chunk_time_slice] = full_mask
        masked_spectrum = np.ma.array(intensity, mask=rfi_mask)
        subfile_as_dict = dict(
            spectra=masked_spectrum,
            nchan=masked_spectrum.shape[0],
            ntime=masked_spectrum.shape[1],
            nsamp=masked_spectrum.size,
            start_unix_time=start_time.unix,
            start_mjd=start_time.mjd,
            beam_number=fh.beam_number,
            cleaned=False,
        )
        output_subfiles.append(subfile_as_dict)

    return output_subfiles


def convert_intensity_to_filterbank(
    filelist, channel_downsampling_factor=1, opath=None
):
    """
    Function to convert sps intensity data (.dat) to filterbank format without
    beamforming.

    Parameters
    ==========
    filelist: list
        List of intensity data filenames from which are to be converted to the filterbank.
        Note that the list must be sorted in time.

    channel_downsampling_factor: int
        Channel downsampling factor relative to 16k channels.
        (i.e., the number of channels requested is 16384 // channel_downsampling_factor)

    opath: str
        Output path to store the filterbank file. If None same path as input .dat files is used.

    Returns
    =======
    None.
    """

    # check if beam model is imported
    if "cfbm" not in sys.modules:
        sys.exit(
            "ImportError: Using convert_intensity_to_filterbank() function requires FRB"
            " beam_model. Please install cfbm if you plan to use this function."
        )

    nchan = 16384 // channel_downsampling_factor
    ntime = 0

    # read data from each intensity file
    for i, datfile in enumerate(filelist):
        outsubfiles = read_huff_msgpack(datfile, channel_downsampling_factor)
        ntime += outsubfiles[0]["ntime"]
        if i == 0:
            # read start time from the first file
            start_unix_time = int(outsubfiles[0]["start_unix_time"])
            start_mjd = outsubfiles[0]["start_mjd"]
            beam = outsubfiles[0]["beam_number"]
            spectra = outsubfiles[0]["spectra"].data
        else:
            spectra = np.hstack((spectra, outsubfiles[0]["spectra"].data))

    # get the end time for filename
    end_unix_time = int(start_unix_time + ntime * TSAMP)

    # convert the dtype to 32 bit float for storage
    spectra = spectra.astype(np.float32)

    # outfilename -> start_time + end_time + beam no
    if opath:
        oname = opath + "{:d}_{:d}_beam_{:04d}.fil".format(
            start_unix_time, end_unix_time, beam
        )
    else:
        oname = filelist[0][:-25] + "{:d}_{:d}_beam_{:04d}.fil".format(
            start_unix_time, end_unix_time, beam
        )

    # Get beam sky position. Position is computed taking the mid epoch of the exposure.
    bm = beam_model.current_model_class(beam_model.current_config)
    beam_pos = bm.get_beam_positions([beam], freqs=bm.config['clamp_freq'])
    time_mid = (start_unix_time + end_unix_time) / 2
    beam_skypos = SkyCoord(
        *bm.get_equatorial_from_position(beam_pos[0, 0, 0], beam_pos[0, 0, 1], time_mid)
        * u.deg
    )

    # Update position in file header
    srcname = "Stationary Beam"
    srcra = beam_skypos.ra.to_string(unit=u.hr, sep=":")
    srcdec = beam_skypos.dec.to_string(sep=":")

    write_to_filterbank(
        spectra, nchan, ntime, beam, start_mjd, 32, srcname, srcra, srcdec, oname
    )
