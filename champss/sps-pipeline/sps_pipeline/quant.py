"""Executes the quantization and down-sampling pipeline component."""

import datetime
import logging
import os
from glob import glob
from os import path

import numpy as np
import pytz
from ch_frb_l1 import rpc_client
from prometheus_client import Summary
from sps_common.conversion import subband
from spshuff import l1_io


log = logging.getLogger(__package__)

archive_root = "/data/chime"

quant_processing_time = Summary(
    "quant_chunk_processing_seconds",
    "Duration of running quantization on a 1s data chunk",
    ("beam",),
)


def _overlap(msg_file, start, end):
    """How much does the range (start1, end1) overlap with (start2, end2)"""
    chunk = rpc_client.read_msgpack_file(msg_file)
    chunk_start = chunk.frame0_nano * 1e-9 + (chunk.fpga0 * 2.56e-6)
    chunk_end = chunk_start + (chunk.fpgaN * 2.56e-6)
    log.debug("/".join(msg_file.split("/")[-2:]), chunk_start, chunk_end)
    return max(
        max((chunk_end - start), 0)
        - max((chunk_end - end), 0)
        - max((chunk_start - start), 0),
        0,
    )


def run(utc_start, utc_end, beam_row, nchan):
    """
    Execute the quantization step on a set of data.

    Parameters
    ----------
    utc_start: float
        The unix UTC start time of the set of data to be quantized.

    utc_end: float
        The unix UTC end time of the set of data to be quantized.

    beam_row: int
        The FRB beam row of the set of data to be quantized.

    nchan: int
        The number of channels to be downsampled to by the quantisation process.
    """
    log.info(
        f"row: {beam_row}, nchan: {nchan}",
        utc_start,
        utc_end,
    )

    date_start = (
        datetime.datetime.utcfromtimestamp(utc_start).replace(tzinfo=pytz.utc).date()
    )
    date_end = (
        datetime.datetime.utcfromtimestamp(utc_end).replace(tzinfo=pytz.utc).date()
    )
    dates = list({date_start, date_end})

    for date in dates:
        file_path = path.join(archive_root, date.strftime("/intensity/raw/%Y/%m/%d"))
        for beam_id in [beam_row + 1000 * i for i in range(4)]:
            msgpack_list = sorted(glob(f"{file_path}/{beam_id:04d}/chunk*.msg"))
            log.debug("Quantize", len(msgpack_list), "data chunks for beam", beam_id)
            for filename in filter(
                lambda mf: _overlap(mf, utc_start, utc_end) > 0, msgpack_list
            ):
                with quant_processing_time.labels(beam_id).time():
                    convert_msgpack_to_huff(filename, nchan)


def convert_msgpack_to_huff(filename, nsub=16384, root=archive_root + "/sps/raw"):
    """
    Converts L1 intensity msgpack data to huffman encoded binary data.

    Parameters
    ==========
    filename: str
        msgpack file to convert to huffman encoded binary file
    root: str
        output root path, from where to start the YYYY/MM/DD/BEAM structure for files.
    Returns
    =======
    new_filename: str
        huffman encoded data file
    """
    chunk = rpc_client.read_msgpack_file(filename)
    intensity, weights = chunk.decode()
    intensity *= weights
    rfi_mask = np.repeat(chunk.rfi_mask, 16, axis=0)
    intensity = subband(intensity, nsub)
    rfi_mask = subband(rfi_mask, nsub).astype(int).astype(bool)
    means = np.mean(intensity, axis=1)
    variance = np.std(intensity, axis=1)
    intensity -= means[:, np.newaxis]
    intensity /= variance[:, np.newaxis]
    intensity[np.isnan(intensity)] = 0.0
    frame0_nano = chunk.frame0_nano
    nrfifreq = chunk.nrfifreq
    beam = chunk.beam
    nbins = chunk.nt
    unix_start = chunk.frame0_nano * 1e-9 + (chunk.fpga0 * 2.56e-6)
    unix_end = unix_start + (chunk.fpgaN * 2.56e-6)
    dt = datetime.datetime.fromtimestamp(unix_start)
    new_filename = path.join(root, dt.strftime("%Y/%m/%d"), f"{beam:04d}")
    os.makedirs(new_filename, exist_ok=True)
    new_filename += "/{:.0f}_{:.0f}.dat".format(
        unix_start,
        unix_end,
    )
    nfreq, ntime = intensity.shape[0], intensity.shape[1]
    with open(new_filename, "w") as f:
        fh = l1_io.FileHeader.from_fields(beam, nbins, unix_start, unix_end)
        ch = l1_io.ChunkHeader.from_fields(
            nfreq, ntime, frame0_nano, chunk.fpgaN, chunk.fpga0
        )
        sps_chunk = l1_io.Chunk(
            ch, means, variance, rfi_mask, intensity, quantize_now=False
        )
        int_file = l1_io.IntensityFile(fh, [sps_chunk])
        int_file.to_file(file=f)
    return new_filename
