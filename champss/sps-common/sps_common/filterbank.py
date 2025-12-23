"""A module to write filterbank files. Slightly modified version of:
https://github.com/scottransom/presto/blob/master/lib/python/filterbank.py
by Patrick Lazarus.

Ziggy Pleunis, ziggy.pleunis@physics.mcgill.ca

"""


import numpy as np
from sps_common import sigproc
from sps_common.constants import FREQ_BOTTOM, FREQ_TOP, TSAMP


def create_filterbank_file(outfile, header, spectra=None, nbits=32, verbose=False):
    """
    Write filterbank header and spectra to file.

    Parameters
    ----------
    outfile : file
        The filterbank file.
    header : dict
        Dictionary of header parameters and values.
    spectra : Spectra
        Spectra to write to file.
        (Default: don't write any spectra, i.e. write out header only.)
    nbits : int
        The number of bits per sample of the filterbank file. This value
        always overrides the value in the header dictionary.
        (Default: 32, i.e. each sample is a 32-bit float.)
    verbose : bool
        If `True`, be verbose.
        (Default: be quiet.)
    """
    dtype = get_dtype(nbits)
    header["nbits"] = nbits

    if is_float(nbits):
        tinfo = np.finfo(dtype)
    else:
        tinfo = np.iinfo(dtype)
    dtype_min = tinfo.min
    dtype_max = tinfo.max

    outfile.write(sigproc.addto_hdr("HEADER_START", None))

    for parameter in list(header.keys()):
        if parameter not in sigproc.header_params:
            # only add recognized parameters
            continue
        if verbose:
            print(f"Writing header parameter '{parameter}'")
        value = header[parameter]
        outfile.write(sigproc.addto_hdr(parameter, value))

    outfile.write(sigproc.addto_hdr("HEADER_END", None))

    if spectra is not None:
        # CHIME/FRB spectra have shape (channels, time_samples) so tranpose
        fil_data = spectra.T
        # Removed flipping of band as the SPS data are in the correct order already
        np.clip(fil_data, dtype_min, dtype_max, out=fil_data)
        fil_data.tofile(outfile, format=dtype)


def append_spectra(outfile, spectra, nbits=32, verbose=False):
    """
    Append filterbank spectra to file.

    Parameters
    ----------
    outfile : file
        The filterbank file.
    spectra : Spectra
        Spectra to append to file.
    nbits : int
        The number of bits per sample of the filterbank file. This value
        always overrides the value in the header dictionary.
        (Default: 32, i.e. each sample is a 32-bit float.)
    verbose : bool
        If `True`, be verbose.
        (Default: be quiet.)
    """
    dtype = get_dtype(nbits)
    if is_float(nbits):
        tinfo = np.finfo(dtype)
    else:
        tinfo = np.iinfo(dtype)
    dtype_min = tinfo.min
    dtype_max = tinfo.max

    # CHIME/FRB spectra have shape (channels, time_samples) so tranpose
    fil_data = spectra.T
    # Removed flipping of band, matching create_filterbank_file
    np.clip(fil_data, dtype_min, dtype_max, out=fil_data)
    # outfile.seek(0, os.SEEK_END)
    #outfile.write(fil_data.astype(dtype))
    fil_data.astype(dtype).tofile(outfile)

def is_float(nbits):
    """
    For a given number of bits per sample return `True` if it corresponds to floating-
    point samples in filterbank files.

    Parameters
    ----------
    nbits: int
        Number of bits per sample, as recorded in the filterbank file's
        header.

    Returns
    -------
    isfloat : bool
        `True`, if `nbits` indicates the data in the file are encoded
        as floats.
    """
    check_nbits(nbits)
    if nbits == 32:
        return True
    else:
        return False


def check_nbits(nbits):
    """
    Given a number of bits per sample check to make sure `filterbank.py` can cope with
    it.

    Parameters
    ----------
    nbits : int
        Number of bits per sample, as recorded in the filterbank file's
        header.

    Raises
    ------
    ValueError
        If `filterbank.py` cannot cope.
    """
    if nbits not in [32, 16, 8]:
        raise ValueError(
            "`filterbank.py` only supports files with 8- or "
            "16-bit integers, or 32-bit floats "
            "(nbits provided: {})!".format(nbits)
        )


def get_dtype(nbits):
    """
    For a given number of bits per sample return a numpy-recognized dtype.

    Parameters
    ----------
    nbits : int
        Number of bits per sample, as recorded in the filterbank file's
        header.

    Returns
    -------
    dtype : dtype
        A numpy dtype string.
    """
    check_nbits(nbits)

    if is_float(nbits):
        dtype = f"float{nbits}"
    else:
        dtype = "uint%d" % nbits

    return dtype


def write_to_filterbank(
    spectra, nchan, ntime, tsamp, beam, start_mjd, nbits, srcname, srcra, srcdec, oname
):
    """
    Writes the beamformed spectra to a filterbank file.

    Parameters
    =======
    spectra: ndarray
        2-D ndarray containing the beamformed spectra in shape (nchan, ntime)

    nchan: int
        Number of channels in the spectra

    ntime: int
        Number of time samples in the spectra

    tsamp: float
        Sampling time in seconds

    beam: int
        FRB beam row of the beamformed spectra

    start_mjd: float
        Start time of the beamformed spectra in MJD

    nbits: int
        Number of bits in the beamformed spectra

    srcname: str
        Source name of the beamformed spectra

    srcra: str
        Source RA in the format of hh:mm:ss.s or hhmmss.s

    srcdec: str
        Source Dec in the format of dd:mm:ss.s or ddmmss.s

    oname: str
        Output filterbank file name of the beamformed spectra

    Returns
    =======
    None
    """
    chan_bw = np.abs(FREQ_TOP - FREQ_BOTTOM) / nchan
    fb_header = dict(
        nsamples=int(ntime),
        nchans=int(nchan),
        fch1=FREQ_TOP - chan_bw / 2.0,
        foff=-1.0 * chan_bw,
        nbeams=1,
        ibeam=int(beam),
        nifs=1,
        tsamp=tsamp,
        tstart=start_mjd,
        data_type=1,
        telescope_id=20,
        machine_id=20,
        nbits=int(nbits),
        barycentric=0,
        pulsarcentric=0,
        source_name=str(srcname),
        src_raj=srcra.replace(":", ""),
        src_dej=srcdec.replace(":", ""),
    )

    with open(oname, "wb+") as fb_out:
        create_filterbank_file(
            fb_out, fb_header, spectra=spectra, nbits=fb_header["nbits"]
        )
