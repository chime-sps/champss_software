import datetime as dt
import logging
import math
from typing import Dict, List, NamedTuple, Set

import numpy as np
import pytz
import trio

try:
    from contextlib import AsyncExitStack
except ImportError:
    # backport for Py3.6
    from async_exit_stack import AsyncExitStack

from beamformer.strategist import strategist

log = logging.getLogger("scheduler")


# how should often the next schedule batch be computed (in seconds)
scheduling_interval: int = 1200

# the amount of overlap before the batch (in seconds)
scheduling_overlap_before: int = 300

# the amount of overlap after the batch (in seconds)
scheduling_overlap_after: int = 1500


class BeamUpdateParams(NamedTuple):
    """Utility class for holding beam update parameters in format that sorts by time."""

    time: int  # time of update
    id: int  # associated pointing
    beam: int  # beam to update
    activate: bool  # whether the beam should be turned on or off
    nchans: int  # number for the pointing


async def generate_pointings(
    active_beams: Set[int], send_channels: Dict[int, trio.abc.SendChannel]
):
    """Task that continually calculates current pointings for `active_beams` and sends them to the respective beam's `send_channel`.

    Pointings are calculated in chunks, using `beamformer.strategist` package.
    The time interval for this calculation is one hour long, but the
    calculation is run every 40 minutes, to make sure any execution delays
    don't cause pointing holes.

    Parameters
    ----------

    active_beams: Set(int)
        Beams for which to calculate pointings and send beam updates

    send_channels: Dict[int, trio.abc.SendChannel]
        Map from beam id to the channel for schedule updates
    """
    last_index = 1
    last_time = -math.inf
    now = dt.datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
    log.info("Initialize the next few minutes of pointings")
    last_index, last_time = await update_pointing_schedule(
        active_beams,
        now,
        scheduling_interval + scheduling_overlap_before + scheduling_overlap_after,
        last_index,
        last_time,
        send_channels,
    )

    async with AsyncExitStack() as stack:
        for ch in send_channels.values():
            await stack.enter_async_context(ch)
        while True:
            now = dt.datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
            log.debug(
                "Calculate pointings %.2f - %.2f, last schedule at %.2f",
                now - scheduling_overlap_before,
                now + scheduling_interval + scheduling_overlap_after,
                last_time,
            )
            last_index, last_time = await update_pointing_schedule(
                active_beams,
                now - scheduling_overlap_before,
                scheduling_interval + scheduling_overlap_after,
                last_index,
                last_time,
                send_channels,
            )
            time_to_next_update = max(
                now
                + scheduling_interval
                - dt.datetime.utcnow().replace(tzinfo=pytz.utc).timestamp(),
                0,
            )
            log.info(
                "Wait %d s to calculate the next pointing batch", time_to_next_update
            )
            await trio.sleep(time_to_next_update)


async def update_pointing_schedule(
    active_beams: Set[int],
    start_utc: int,
    duration: int,
    last_index: int,
    last_time: float,
    send_channels: Dict[int, trio.abc.SendChannel],
):
    """Calculate the next batch of pointings for `beam_rows` and send them to the respective beam's `send_channel`.

    Parameters
    ----------

    active_beams: Set(int)
        Beams for which to calculate pointings and send beam updates

    start_utc: int
        UTC timestamp to start the schedule

    duration: int
        How far from `start_utc` to plan the schedule

    last_index: int
        Greatest pointing id seen in the previous batch

    last_time: float
        Greatest pointing transit time seen in the previous batch

    send_channels: Dict[int, trio.abc.SendChannel]
        Map from beam id to the channel for schedule updates

    Returns
    -------

    last_index: int
        new value of last_index after processing the new schedule batch

    last_time: int
        new value of last_time after processing the new schedule batch
    """
    # The strategist works per-row, so we have to give it the rows that include active beams
    beam_rows = set([b % 1000 for b in active_beams])
    aps = await trio.to_thread.run_sync(
        get_active_pointings,
        list(beam_rows),
        start_utc,
        duration,
    )
    log.debug("Got pointings")
    now = dt.datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
    batch_max_time = -math.inf
    for i, ap in enumerate(aps):
        i += last_index
        for b in ap.max_beams:
            if b["beam"] not in active_beams:
                continue

            # Catch if #3 is still possible:
            if "utc_end" not in b or "utc_start" not in b:
                log.error(
                    "Pointing %d (%.2f, %.2f) has transit time properties for beam %04d: %s",
                    i,
                    ap.ra,
                    ap.dec,
                    b["beam"],
                    ap,
                )
                exit(1)

            beam_schedule_channel = send_channels[b["beam"]]
            if b["utc_end"] + 10 > now:
                # add 20 seconds buffer to before start and 10s after end time of the beam transits
                log.info(
                    "Schedule pointing %d / %04d (%.2f, %.2f) @ %d - %d",
                    i,
                    b["beam"],
                    ap.ra,
                    ap.dec,
                    b["utc_start"] - 20,
                    b["utc_end"] + 10,
                )
                await beam_schedule_channel.send(
                    BeamUpdateParams(
                        b["utc_start"] - 20,
                        i,
                        b["beam"],
                        True,
                        ap.nchan,
                    ),
                )
                await beam_schedule_channel.send(
                    BeamUpdateParams(b["utc_end"] + 10, i, b["beam"], False, ap.nchan),
                )
            else:
                log.debug(
                    "Too late for pointing %d / %04d (%.2f, %.2f) @ %.2f: now is %.2f",
                    i,
                    b["beam"],
                    ap.ra,
                    ap.dec,
                    b["utc_end"],
                    now,
                )
            batch_max_time = max(batch_max_time, b["utc_end"] + 10)
    return last_index + len(aps), batch_max_time


def get_active_pointings(beam_rows: List[int], start_utc: float, duration: float):
    """Return active pointings in `beam_rows` in time interval `(start_utc, start_utc+duration)`

    Parameters
    ----------

    beam_rows: List[int]
        Beam row for which to calculate pointings.

    start_utc: int
        UTC timestamp to start the schedule

    duration: int
        How far from `start_utc` to plan the schedule

    Returns
    -------
    List[beamformer.strategist.ActivePointing]"""
    strategy = strategist.PointingStrategist(from_db=False)
    return strategy.get_pointings(
        utc_start=start_utc,
        utc_end=start_utc + duration,
        beam_row=np.array(beam_rows),
    )
