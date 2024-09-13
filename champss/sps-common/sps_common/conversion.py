import importlib
import re
import sys

import h5py
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from sps_common.constants import TSAMP
from sps_common.filterbank import write_to_filterbank
from sps_common.files import open_file
from spshuff import l1_io

# import line_profiler
# profiler = line_profiler.LineProfiler()

# Conditional import for beam_model
if importlib.util.find_spec("beam_model"):
    import beam_model


def convert_ra_dec(ra_deg, dec_deg):
    """
    Converts RA and Dec from degrees to hhmmss.ss format readable by filterbank files.

    Parameters
    =======
    ra_deg: float
        RA in degrees

    dec_deg: float
        Dec in degrees

    Returns
    =======
    ra_hms: str
        RA in hhmmss.s string

    dec_hms: str
        Dec in ddmmss.s string
    """
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


def unix_to_mjd(unix_time):
    """
    Converts unix utc time to MJD.

    Parameters
    =======
    unix_time: float
        Unix UTC time

    Returns
    =======
    time.mjd: float
        UTC time in MJD
    """
    time = Time(unix_time, format="unix")
    return time.mjd


def load_presto_dat(dat, bit):
    """
    Loads a presto dedispersed timeseries in .dat format.

    Parameters
    =======
    dat: str
        The name of the .dat file
    bit: int
        The number of bits in the .dat file

    Returns
    =======
    timeseries: ndarray
        1-D array of the dedispersed time series
    """
    if bit >= 32:
        dtype = f"float{bit}"
    else:
        dtype = f"uint{bit}"

    with open(dat, "rb") as datfile:
        timeseries = np.fromfile(datfile, dtype=dtype)

    return timeseries


def read_hdf5_dataset(h5f, dataset_key="0"):
    """
    Read a dataset from a hdf5 file witha given key.

    Parameters
    ==========
    hf5: hdf5 File object
    dataset_key: str
        key of the dataset to read

    Returns
    =======
    data: np.ndarray
        dataset from the hdf5 file
    """
    data = np.ones(h5f[dataset_key].shape)
    h5f[dataset_key].read_direct(data)
    return data


def read_hhat_hdf5(filename):
    """
    Read hdf5 file containing the hhat array.

    Parameters
    ==========
    filename: str
        hdf5 file to read

    Returns
    =======
    hf5: hdf5 file object
    dml: np.array
        dm labels
    freql: np.array
        frequency labels
    dcl: np.array
        duty cycle labels
    sliced_by: str
        parameter that the data was sliced by : 'dm' or 'dc'
    """
    h5f = h5py.File(filename, "r")
    dcl = np.ones(h5f["dc_labels"].shape)
    h5f["dc_labels"].read_direct(dcl)
    dml = np.ones(h5f["dm_labels"].shape)
    h5f["dm_labels"].read_direct(dml)
    freql = np.ones(h5f["freq_labels"].shape)
    h5f["freq_labels"].read_direct(freql)
    sliced_by = h5f.attrs["sliced_by"]
    if isinstance(sliced_by, bytes):
        sliced_by = sliced_by.decode()
    return h5f, dml, freql, dcl, sliced_by


def write_hhat_hdf5(
    data, dc_labels, dm_labels, freq_labels, filename, slice_param="dc"
):
    """
    Write hhat array to hdf5 file sliced by either DM ('DM', 'dm') or duty cycle ('DC',
    'dc').

    Parameters
    ==========
    data: np.ndarray
        hhat array to write to disk.
    dc_labels: np.array
        duty cycle labels
    dm_labels: np.array
        dm labels
    freq_labels: np.array
        frequency labels
    filename: str
        hdf5 file to write to.
    slice_param: str
        parameter that the data will be sliced across : 'DM', 'dm' or 'DC', 'dc'

    Returns
    =======
    None
    """
    assert slice_param in ["dm", "dc", "DC", "DM"]
    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("dc_labels", data=dc_labels)
        h5f.create_dataset("dm_labels", data=dm_labels)
        h5f.create_dataset("freq_labels", data=freq_labels)
        h5f.attrs["sliced_by"] = slice_param.lower()
        if slice_param in ["dm", "DM"]:
            for i, dml in enumerate(dm_labels):
                h5f.create_dataset(f"{i}", data=data[i, :, :])
        if slice_param in ["dc", "DC"]:
            for i, dcl in enumerate(dc_labels):
                h5f.create_dataset(f"{i}", data=data[:, :, i])


def subband(intensity, nsub):
    """
    Subbands the data to a specified number of subbands.

    Parameters
    ----------

    intensity: 2D array of intensities (nfreq,ntime)
    nsub: int -- number of subbands to reduce data to

    Returns
    -------

    intensity: subbanded 2D array
    """
    if nsub == intensity.shape[0]:
        # no need to subband
        return intensity
    return (
        intensity.reshape(nsub, int(intensity.shape[0] / nsub), intensity.shape[1], 1)
        .mean(axis=1)
        .mean(axis=2)
    )


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
    with open_file(filename, "rb") as f:
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
    if "beam_model" not in sys.modules:
        sys.exit(
            "ImportError: Using convert_intensity_to_filterbank() function requires FRB"
            " beam_model. Please install beam_model if you plan to use this function."
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

    # Get beam sky position. Postion is computed taking the mid epoch of the exposure.
    bm = beam_model.current_model_class(beam_model.current_config)
    beam_pos = bm.get_beam_positions([beam], freqs=bm.clamp_freq)
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


def convert_intensity_to_hdf5(spec, mask, fname, metadata):
    """
    Take the spectra intensity and mask (separately) and write the data to a HDF5 file
    with 'intensity' and 'mask' datasets. To read the data, one would need to do
    something like: >>> import h5py >>> h5f = h5py.File(fname, 'r') >>> intensity =
    h5f['intensity']  # 2D array >>> mask = h5f['mask']  # 2D array, (0 = not masked, 1
    = masked)

    Parameters
    ----------
    spec: np.ndarray
        The raw intensity data in float format, nominally with shape (nchan, ntime)

    mask: np.ndarray
        The corresponding mask, 0 = not masked, 1 = masked

    fname: str
        Desired output HDF5 file name

    metadata: dict
        A dictionary containing metadata required for later pipeline stages

    Returns
    -------
    None
    """

    with h5py.File(fname, "w") as h5f:
        h5f.create_dataset("intensity", data=spec.astype(np.float32))
        h5f.create_dataset("mask", data=mask.astype(np.int8))
        for k, v in metadata.items():
            h5f.attrs[k] = v


def read_intensity_hdf5(fname):
    """
    Take the hdf5 file and extract the datasets 'intensity' and 'mask' into numpy arrays
    of intensity and mask values.

    Parameters
    ----------
    fname: str
        HDF5 file name to read.

    Returns
    -------
    i: np.ndarray
        Array of intensity values of shape (nchan,ntime).

    m: np.ndarray
        Array of boolean mask of shape (nchan,ntime).
    """

    h5f = h5py.File(fname, "r")
    intensity = h5f["intensity"]  # 2D array
    mask = h5f["mask"]  # 2D array, (0 = not masked, 1 = masked)
    i = np.ones(intensity.shape)
    m = np.ones(mask.shape)
    intensity.read_direct(i)
    mask.read_direct(m)
    metadata = h5f.attrs
    return i, m, metadata


def write_spectrum_to_hdf5(
    spectrum, freq_labels, dm, num_days, beta, bad_freq_indices, filename
):
    """
    Write a power spectrum into a hdf5 file.

    Parameters
    =======
    spectrum: np.array
        1-D array of a power spectrum

    freq_labels: np.array
        1-D array of the frequency labels of the power spectrum

    dm: float
        The DM of the power spectrum

    num_days: int
        Number of days the power spectrum stack has

    beta: float
        Barycentric velocity correction that was applied (units of c)

    bad_freq_indices: list of int
        A list of indices corresponding to corrupted power spectrum frequencies

    filename: str
        The filename to save the power spectrum
    """
    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("spectrum", data=spectrum)
        h5f.create_dataset("freq_labels", data=freq_labels)
        if bad_freq_indices is None or not bad_freq_indices:
            # create an empty dataset
            print("WARNING: writing empty dataset for birdies")
            h5f.create_dataset("bad_freq_indices", dtype="i")
        else:
            h5f.create_dataset("bad_freq_indices", data=bad_freq_indices, dtype="i")

        h5f.attrs["dm"] = dm
        h5f.attrs["num_days"] = num_days
        if beta is None:
            # no barycentric correction set/required
            print("WARNING: berycentric correction not set, writing empty attribute")
            h5f.attrs.create("beta", data=None, dtype="f")
        else:
            h5f.attrs["beta"] = beta


def read_hdf5_spectrum(filename):
    """
    Read the power spectrum saved.

    Parameters
    =======
    filename: str
        The filename of the power spectrum

    Returns
    =======
    spectrum: np.array
        1-D array of a power spectrum

    freq_labels: np.array
        1-D array of the frequency labels of the power spectrum

    dm: float
        The DM of the power spectrum

    num_days: int
        Number of days the power spectrum stack has

    beta: float
        Barycentric velocity correction that was applied (units of c)

    bad_freq_indices: list of int
        A list of indices corresponding to corrupted power spectrum frequencies
    """
    h5f = h5py.File(filename, "r")
    spectrum = np.ones(h5f["spectrum"].shape)
    h5f["spectrum"].read_direct(spectrum)
    freq_labels = np.ones(h5f["freq_labels"].shape)
    h5f["freq_labels"].read_direct(freq_labels)

    # the bad frequency index list could be empty, so handle that gracefully

    if h5f["bad_freq_indices"].shape is None:
        bad_freq_indices = []
    elif h5f["bad_freq_indices"].len() == 0:
        bad_freq_indices = []
    else:
        bad_freq_indices = np.ones(h5f["bad_freq_indices"].shape)
        h5f["bad_freq_indices"].read_direct(bad_freq_indices)

    dm = h5f.attrs["dm"]
    num_days = h5f.attrs["num_days"]
    if isinstance(h5f.attrs["beta"], h5py.Empty):
        beta = None
    elif np.isnan(h5f.attrs["beta"]):
        beta = None
    else:
        beta = h5f.attrs["beta"]

    return spectrum, freq_labels, bad_freq_indices, dm, num_days, beta


def write_cands_to_hdf5(cand_list, cand_labels, freq_spacing, filename):
    """
    Write power spectrum search candidate list into a hdf5 file.

    Parameters
    =======
    cand_list: list
        2-D list of the candidate list

    cand_labels: list
        1-D array of the labels of the candidates

    freq_spacing: float
        frequency spacing between adjacent Fourier bins

    filename: str
        The filename to save the candidate list
    """
    cand_list = np.asarray(cand_list)
    with h5py.File(filename, "w") as h5f:
        h5f.create_dataset("candidate list", data=cand_list)
        h5f.attrs["candidate labels"] = cand_labels
        h5f.attrs["frequency spacing"] = freq_spacing


def read_hdf5_cands(filename):
    """
    Read the power spectrum search candidate list in hdf5 format.

    Parameters
    =======
    filename: str
        The filename of the candidates list

    Returns
    =======
    cand_list: np.adarray
        2-D array of the candidate list

    cand_labels: list
        1-D array of the labels of the candidates

    freq_spacing: float
        frequency spacing between adjacent Fourier bins
    """
    h5f = h5py.File(filename, "r")
    cand_list = np.ones(h5f["candidate list"].shape)
    h5f["candidate list"].read_direct(cand_list)
    cand_labels = h5f.attrs["candidate labels"]
    freq_spacing = h5f.attrs["frequency spacing"]

    return cand_list, cand_labels, freq_spacing


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]
