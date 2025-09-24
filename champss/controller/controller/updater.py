import datetime as dt
import heapq
import logging
import math
import os
import signal
import subprocess  # nosec

import pytz
import trio
from controller import rpc_client
from controller.l1_rpc import get_beam_ip

log = logging.getLogger("issuer")


async def pointing_beam_control(new_pointing_listen, pointing_done_announce, basepath, source="champss"):
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
    max_combined_age = 600
    allowed_timeout = 10
    client = rpc_client.RpcClient({})
    last_update = dt.datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()

    # Quick and dirty way of changing mount name
    local_path = basepath.replace("/sps-archiver2/", "/mnt/beegfs-client/").replace(
        "/sps-archiver1/", "/data/"
    )

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
                # Alternatively could also initialize client here, since each instance only contains one beam
                # Shouls not make a difference
                if b.beam not in client.servers:
                    log.debug(f"Will try adding server {b.beam}")
                    client.add_server(b.beam, f"tcp://{get_beam_ip(b.beam)}:5555")
                    log.debug(f"Added server {b.beam}")
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
                now = dt.datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
                data_folder = (
                    local_path
                    + dt.datetime.utcnow().strftime("/%Y/%m/%d/")
                    + f"{str(b.beam).zfill(4)}"
                )
                # update_age makes sure that in the case of a file writing error, the cal to restart
                # writing is not repeated all the time
                update_age = now - last_update
                try:
                    folder_age = now - os.path.getmtime(data_folder)
                except FileNotFoundError:
                    log.warning(f"Data Folder {data_folder} not found")
                    # Check previous day because I am not sure how it performs at the date change
                    previous_data_folder = (
                        local_path
                        + (dt.datetime.utcnow() - dt.timedelta(days=1)).strftime(
                            "/%Y/%m/%d/"
                        )
                        + f"{str(b.beam).zfill(4)}"
                    )
                    try:
                        folder_age = now - os.path.getmtime(data_folder)
                        log.info(
                            "Checked age of data older of previous day:"
                            f" {previous_data_folder}"
                        )
                    except FileNotFoundError:
                        folder_age = max_combined_age + 1
                # Only update if folder and recent update are old, otherwise, will perform too many calls
                combined_age = min(update_age, folder_age)
                if combined_age > max_combined_age:
                    log.warning(
                        f"Folder {data_folder} is too old ({folder_age:.2f}) or was not"
                        " found."
                    )
                    update_and_folder_too_old = True
                else:
                    update_and_folder_too_old = False
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
                    if new_max_nchans != max_nchans or update_and_folder_too_old:
                        try:
                            with timeout(
                                allowed_timeout,
                                error_message=f"Unable to update beam {b.beam}",
                            ):
                                last_update = now
                                output = client.set_spulsar_writer_params(
                                    b.beam,
                                    new_max_nchans,
                                    1024,
                                    5,
                                    basepath,
                                    source,
                                    timeout=-1,
                                    servers=[b.beam],
                                )
                                log.debug(output)
                        except TimeoutError as te:
                            log.warning(
                                f"Could not update beam {b.beam}. rpc-client timed out."
                            )

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
                        if new_max_nchans != max_nchans or update_and_folder_too_old:
                            log.info(
                                "REDUCE %04d: %d -> %d",
                                b.beam,
                                max_nchans,
                                new_max_nchans,
                            )
                            try:
                                with timeout(
                                    allowed_timeout,
                                    error_message=f"Unable to update beam {b.beam}",
                                ):
                                    last_update = now
                                    output = client.set_spulsar_writer_params(
                                        b.beam,
                                        new_max_nchans,
                                        1024,
                                        5,
                                        basepath,
                                        source,
                                        timeout=-1,
                                        servers=[b.beam],
                                    )
                                    log.debug(output)
                            except TimeoutError as te:
                                log.warning(
                                    f"Could not update beam {b.beam}. rpc-client timed"
                                    " out."
                                )

                        # After the last beam has completed,
                        # we can announce to the listeners
                        # that the pointing is done
                        if b.beam >= 3000:
                            log.info("Pointing %d / %04d done", b.id, b.beam)
                            await pointing_done_announce.send((b.beam % 1000, b.id))
                    else:
                        # No more active pointings, turn the beam off
                        log.info("OFF %d / %04d", b.id, b.beam)


class timeout:
    # Taken from https://stackoverflow.com/a/22348885
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
