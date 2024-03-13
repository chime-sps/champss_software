import datetime as dt
import heapq
import logging
import math
import subprocess  # nosec

import pytz
import trio

from controller.l1_rpc import get_beam_ip

log = logging.getLogger("issuer")

SPS_DATA_DIR = "/sps-archiver/chime/sps/raw"


async def pointing_beam_control(new_pointing_listen, pointing_done_announce):
    """
    Task that issues beam pointing updates on a generated schedule.

    One instance handles issuing updates for one L1 beam.

    Parametes
    ------
    new_pointing_listen: trio.ReceiveChannel
        Channel from which to receive notifications of schedule updates sent by
        `sps_controller.pointer.generate_pointings()`

    pointing_done_announce: trio.SendChannel
        Channel on which to send notifications of completed beam pointing
        updates, as a tuple of beam row and pointing id.
    """
    scheduled_updates = []  # priority queue of the beam update schedule
    beam_active = {}  # map of this task's beam's active pointings to their nchans
    done = False
    async with new_pointing_listen:
        async with pointing_done_announce:
            while scheduled_updates or not done:
                now = dt.datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
                # decide how long to wait for a new pointing update from the
                # `new_pointing` channel: either until the next scheduled update,
                # or forever if there isn't one
                if scheduled_updates:
                    time_to_next_update = max(0, (scheduled_updates[0].time - now))
                else:
                    time_to_next_update = math.inf

                if not done and time_to_next_update > 0:
                    log.debug(
                        "Wait for next pointing update up to %.1f s",
                        time_to_next_update,
                    )
                    with trio.move_on_after(time_to_next_update):
                        # get next pointing update
                        try:
                            b = await new_pointing_listen.receive()
                            log.debug("Received %s", b)
                            heapq.heappush(scheduled_updates, b)
                        except trio.EndOfChannel:
                            log.debug("No more pointing updates")
                            done = True
                        continue  # back to the top of the scheduling loop

                b = heapq.heappop(scheduled_updates)
                log.debug(
                    "Next pointing %d / %04d @ %.1f - %s",
                    b.id,
                    b.beam,
                    b.time,
                    "ON" if b.activate else "OFF",
                )
                now = dt.datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
                time_to_next_update = max(0, (b.time - now))
                if time_to_next_update > 600:
                    log.info(
                        "Too long until the next update: %.1f s", time_to_next_update
                    )
                    break
                if time_to_next_update > 0:
                    log.debug(
                        "Wait for next scheduled update: %.1f s",
                        time_to_next_update,
                    )
                    await trio.sleep(time_to_next_update)

                if b.activate:
                    max_nchans = max(beam_active.values()) if beam_active else 0
                    beam_active[b.id] = b.nchans
                    new_max_nchans = max(beam_active.values())

                    log.info(
                        "START %d / %04d: %d",
                        b.id,
                        b.beam,
                        b.nchans,
                    )
                    # We want to save using the maximum number
                    # of channels required by all the currently
                    # active pointings
                    if new_max_nchans != max_nchans:
                        log.info(
                            "INCREASE %04d: %d -> %d",
                            b.beam,
                            max_nchans,
                            new_max_nchans,
                        )
                        try:
                            output = subprocess.run(
                                [
                                    "rpc-client --spulsar-writer-params"
                                    f" {b.beam} {new_max_nchans} 1024 5"
                                    f" {SPS_DATA_DIR} tcp://{get_beam_ip(b.beam)}:5555"
                                ],
                                shell=True,  # nosec
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                # capture_output=True,
                                # text=True,
                                timeout=20,
                            )
                        except TimeoutError as e:
                            log.info(e)
                            output = f"Unable to run rpc-client on {b.beam}"
                        except subprocess.TimeoutExpired as e:
                            log.info(e)
                            output = f"Unable to run rpc-client on {b.beam}"
                        log.info(output)
                else:
                    # Remove the pointing from the beam's active list
                    max_nchans = max(beam_active.values())
                    del beam_active[b.id]

                    # If the beam still has any active pointings,
                    # we have to check whether their max `nchans`
                    # has changed
                    if beam_active:
                        log.info(
                            "END %d / %04d: %d",
                            b.id,
                            b.beam,
                            b.nchans,
                        )
                        new_max_nchans = max(beam_active.values())
                        if new_max_nchans != max_nchans:
                            log.info(
                                "REDUCE %04d: %d -> %d",
                                b.beam,
                                max_nchans,
                                new_max_nchans,
                            )
                            try:
                                output = subprocess.run(
                                    [
                                        "rpc-client --spulsar-writer-params"
                                        f" {b.beam} {new_max_nchans} 1024 5"
                                        f" {SPS_DATA_DIR} tcp://{get_beam_ip(b.beam)}:5555"
                                    ],
                                    shell=True,  # nosec
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    # capture_output=True,
                                    # text=True,
                                    timeout=20,
                                )
                            except TimeoutError as e:
                                log.info(e)
                                output = f"Unable to run rpc-client on {b.beam}"
                            except subprocess.TimeoutExpired as e:
                                log.info(e)
                                output = f"Unable to run rpc-client on {b.beam}"
                            log.info(output)

                        # After the last beam has completed,
                        # we can announce to the listeners
                        # that the pointing is done
                        if b.beam >= 3000:
                            log.info("Pointing %d / %04d done", b.id, b.beam)
                            await pointing_done_announce.send((b.beam % 1000, b.id))
                    else:
                        # No more active pointings, turn the beam off
                        log.info("OFF %d / %04d", b.id, b.beam)
