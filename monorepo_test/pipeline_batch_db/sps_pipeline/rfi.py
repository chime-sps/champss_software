"""Executes the RFI mitigation pipeline component
"""
from glob import glob
import logging
from omegaconf import OmegaConf
import os
import datetime as dt
import pytz
from os import path
from prometheus_client import Summary

from spshuff import l1_io
from rfi_mitigation.pipeline import RFIPipeline
from rfi_mitigation.reader import DataReader

from . import utils


log = logging.getLogger(__package__)

archive_root = "/data/chime"

rfi_processing_time = Summary(
    "rfi_total_chunk_processing_seconds",
    "Duration of running RFI cleaning on a data chunk",
    ("beam",),
)


def _overlap(msg_file, start, end):
    """how much does the range (start1, end1) overlap with (start2, end2)"""
    with open(msg_file, "rb") as f:
        fh = l1_io.FileHeader.from_file(f)
    chunk_start = fh.start
    chunk_end = fh.end
    log.debug("/".join(msg_file.split("/")[-2:]), chunk_start, chunk_end)
    return max(
        max((chunk_end - start), 0)
        - max((chunk_end - end), 0)
        - max((chunk_start - start), 0),
        0,
    )


def run(beams_start_end, config, basepath="./"):
    """Execute the RFI excision step on a set of data.

    Parameters
    ----------
    utc_start: float
        The unix UTC start time of the set of data for RFI excision.

    utc_end: float
        The unix UTC end time of the set of data for RFI excision.

    beam_row: int
        The FRB beam row of the set of data for RFI excision.

    config:
        OmegaConf instance with the configuration to use

    basepath: str
        Folder which is used to store data. Default: './'

    """
    masking_dict = OmegaConf.to_container(config.rfi)

    rfipipe = RFIPipeline(masking_dict, make_plots=False)
    reader = DataReader(apply_l1_mask=masking_dict["l1"])

    def filter_func(mf):
        """
        Select only files that have more than an empty header and whose data falls
        within the pointing's transit.

        Parameters
        ----------
        mf: str
            The msgpack file path

        Returns
        -------
            If the data file is an empty header, False, otherwise it returns whether
            any of the data in the file corresponds to the transit time of the desired
            pointing.
        """
        if os.stat(mf).st_size > 1024:
            return _overlap(mf, beam_utc_start, beam_utc_end) > 0
        else:
            log.warning(f"file size too small - skipping: {mf}")
            return False

    for beam in beams_start_end:
        beam_id = beam["beam"]
        date_start = (
            dt.datetime.utcfromtimestamp(beam["utc_start"])
            .replace(tzinfo=pytz.utc)
            .date()
        )
        date_end = (
            dt.datetime.utcfromtimestamp(beam["utc_start"])
            .replace(tzinfo=pytz.utc)
            .date()
        )
        beam_path_start = path.join(date_start.strftime("%Y/%m/%d"), f"{ beam_id :04d}")
        beam_path_end = path.join(date_end.strftime("%Y/%m/%d"), f"{beam_id :04d}")
        os.makedirs(path.join(basepath, beam_path_start), exist_ok=True)
        os.makedirs(path.join(basepath, beam_path_end), exist_ok=True)
        beam_paths = list(set([beam_path_start, beam_path_end]))

        for beam_path in beam_paths:
            spshuff_files = sorted(
                glob(os.path.join(archive_root, "sps/raw", beam_path, "*.dat"))
            )

            beam_utc_start = beam["utc_start"]
            beam_utc_end = beam["utc_end"]
            it = filter(filter_func, spshuff_files)
            for spshuff_batch in list(zip(*[it])):
                with rfi_processing_time.labels(beam_id).time():
                    log.info(f"Data being processed : {spshuff_batch}")
                    # default reader mode is SPS quantized data
                    chunks = reader.read_files(spshuff_batch)
                    cleaned = rfipipe.clean(chunks)
                    for c in cleaned:
                        c.write(path.join(basepath, beam_path))


def get_data_to_clean(active_pointings):
    beams_start_end = []
    for p in active_pointings:
        if not beams_start_end:
            beams_start_end = p.max_beams
            continue
        current_beam_list = [b["beam"] for b in beams_start_end]
        for beam in p.max_beams:
            beam_idx = [i for i, b in enumerate(current_beam_list) if b == beam["beam"]]
            if not beam_idx:
                beams_start_end.append(beam)
                continue
            beam_idx = beam_idx[0]
            if beams_start_end[beam_idx]["utc_start"] > beam["utc_start"]:
                beams_start_end[beam_idx]["utc_start"] = beam["utc_start"]
            if beams_start_end[beam_idx]["utc_end"] < beam["utc_end"]:
                beams_start_end[beam_idx]["utc_end"] = beam["utc_end"]
    return beams_start_end
