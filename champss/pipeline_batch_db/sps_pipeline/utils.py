"""Shared utility routines for the SPS pipeline."""
import datetime as dt

import pytz


def transit_time(pointing):
    """
    Calculate `pointing`s transit time in CHIME beam 1xxx.

    Parameters
    ----------
    pointing: sps_common.interfaces.beamformer.ActivePointing
        Sky pointing to process.

    Returns
    -------
    datetime.datetime
        Time of the middle of the pointing's transit
    """

    N_beam = len(pointing.max_beams)
    for i, b in enumerate(pointing.max_beams):
        if i == 0:
            utc_start = b["utc_start"]
        if i == N_beam - 1:
            utc_end = b["utc_end"]
    utc_transit = utc_start + (utc_end - utc_start) / 2
    transit_datetime = dt.datetime.utcfromtimestamp(utc_transit)
    return transit_datetime.replace(tzinfo=pytz.UTC)


def pointing_interval(pointing):
    """
    Returns the start and end times of the `pointing`

    Parameters
    ----------
    pointing: sps_common.interfaces.beamformer.ActivePointing
        Sky pointing to process.

    Returns
    -------
    Tuple(float, float)
        UTC timestamp for the start and end of the pointing

    Raises
    ------
    ValueError
        If `pointing` does not have transit info for beam 1xxx.
    """
    p_utc_start = min([b["utc_start"] for b in pointing.max_beams])
    p_utc_end = max([b["utc_end"] for b in pointing.max_beams])
    return p_utc_start, p_utc_end


def get_pointings_from_list(datlist):
    start_times = []
    end_times = []
    new_start = True
    last_end = None
    for i, dat in enumerate(datlist):
        dat = dat.split("/")[-1]
        start_time = int(dat.split("_")[0])
        end_time = int(dat.split("_")[1].split(".")[0])
        if last_end is not None:
            if (start_time - last_end) > 40:
                end_times.append(last_end)
                new_start = True
        if new_start:
            start_times.append(start_time)
            new_start = False
        if end_time - start_time > 40:
            end_times.append(start_time + 40)
            new_start = True
        elif i == len(datlist) - 1:
            end_times.append(start_time + 40)
        last_end = end_time

    return start_times, end_times


def convert_date_to_datetime(date):
    if isinstance(date, str) or isinstance(date, int):
        for date_format in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
            try:
                date = dt.datetime.strptime(str(date), date_format)
                break
            except ValueError:
                continue
    return date
