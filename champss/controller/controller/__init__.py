# Controller for the Slow Pulsar Search (SPS) L1 system.

import logging
import math
import os
import signal
import subprocess  # nosec
import time
from datetime import datetime
from itertools import islice
from typing import Tuple

import click
import trio
from controller.l1_rpc import get_beam_ip, get_node_beams
from controller.pointer import generate_pointings
from controller.updater import pointing_beam_control, timeout
from scheduling.scheduleknownpulsars import is_beam_recording

log = logging.getLogger("spsctl")
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    # Based on function in itertools which is added in 3.12
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


async def announce_pointing_done(pointing_done_listen):
    """
    Task that listens for pointing completed messages from the updater(s).

    Parameters
    ----------
    pointing_done_listen: trio.ReceiveChannel

        Trio channel on which `updater.pointing_beam_control()` sends "pointing
        complete" messages. The channel can have multiple updater producers,
        and the messages are a tuple of beam row and pointing id.
    """
    async with pointing_done_listen:
        async for beam_row, pointing_id in pointing_done_listen:
            print(f"  Row {beam_row} / pointing {pointing_id} DONE")


async def entry_point(active_beams, basepath, source="champss"):
    """
    Parent for the pointer and updater tasks.

    Parameters
    ----------
    active_beams: Set(int)
        Beams for which to calculate pointings and send beam updates
    """
    pointing_done_announce, pointing_done_listen = trio.open_memory_channel(math.inf)
    beam_schedule_channels = {}
    try:
        async with trio.open_nursery() as nursery:
            log.debug("strat: spawning announcer...")
            nursery.start_soon(announce_pointing_done, pointing_done_listen)
            for beam_id in active_beams:
                new_pointing_announce, new_pointing_listen = trio.open_memory_channel(
                    math.inf
                )
                log.debug(f"strat: spawning controller for beam {beam_id}...")
                nursery.start_soon(
                    pointing_beam_control,
                    new_pointing_listen,
                    pointing_done_announce.clone(),
                    basepath,
                    source,
                )
                beam_schedule_channels[beam_id] = new_pointing_announce

            log.debug(f"strat: spawning pointer for {active_beams}...")
            nursery.start_soon(generate_pointings, active_beams, beam_schedule_channels)
    except BaseExceptionGroup:
        pass
        # When cancelling the batch acquisition this Exception will be thrown by trio.


def start_beam(beam: int, basepath: str, source: str, channels: int, ntime: int):
    """
    Start beam acquisition with a fixed channelisation.

    Args:
        beam (int): Beam number
    """
    proc = subprocess.Popen(
        [
            f"rpc-client --spulsar-writer-params {beam} {channels} {ntime} 5"
            f" {basepath} {source} tcp://{get_beam_ip(beam)}:5555"
        ],
        shell=True,  # nosec
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def stop_beam(beam: int, basepath: str, source: str):
    """
    Stop beam acquisition.

    Args:
        beam (int): Beam number
    """
    proc = subprocess.Popen(
        [
            f"rpc-client --spulsar-writer-params {beam} {0} 1024 5"
            f" {basepath} {source} tcp://{get_beam_ip(beam)}:5555"
        ],
        shell=True,  # nosec
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def stop_all_beams(active_beams, basepath, source="champss", batchsize=20):
    """
    Stop all beams.

    Args:
        active_beams list(int): Beam number
    """
    # Creating processes for all beams at once will lead to a crash
    batched_beams = batched(active_beams, batchsize)
    for beam_batch in batched_beams:
        procs = [stop_beam(beam, basepath, source) for beam in beam_batch]
        time.sleep(0.1)
        for proc, beam in zip(procs, beam_batch):
            try:
                output = proc.communicate(timeout=5)
                if "parameters successfully applied" in str(output[0]):
                    output = f"Successfully stopped {beam}"
            except subprocess.TimeoutExpired as e:
                output = f"Unable to stop beam {beam}"
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            finally:
                log.info(output)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--host",
    multiple=True,
    metavar="HOSTNAME",
    help=(
        "Generate the schedule only for beam(s) running on HOSTNAME(s). (Repeat to"
        " include multiple hosts.)"
    ),
)
@click.argument("rows", type=click.IntRange(min=0, max=223), nargs=-1, required=True)
@click.option(
    "--loglevel",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Set logging level.",
)
@click.option("--logtofile", is_flag=True, help="Enable file logging.")
@click.option(
    "--basepath",
    type=str,
    default="/sps-archiver2/raw/",
    help="Path on L1 cf nodes to a CHAMPSS mount.",
)
@click.option(
    "--source",
    type=str,
    default="champss",
    help=(
        "The chime_slow_pulsar_writer object to use on L1, must be either 'champss' or 'slow'. "
        "Do not use 'slow' before consulting with the Slow team"
    ),
)
@click.option(
    "--nocleanup",
    is_flag=True,
    help="Do not clean up. Used to not interfere with batch processing.",
)
@click.option(
    "--logid",
    type=str,
    default="",
    help="Additonal identifier to sepearte the log_files.",
)
@click.option(
    "--channels",
    type=int,
    default=-1,
    help="Number of recorded frequency channels. If -1 use the pointing map.",
)
@click.option(
    "--ntime",
    type=int,
    default=1024,
    help="Size of chunks. Only used in conjunction with --channels.",
)
def cli(
    host: Tuple[str],
    rows: Tuple[int],
    loglevel: str,
    logtofile: bool,
    basepath: str,
    source: str,
    nocleanup: bool,
    logid: str,
    channels: int,
    ntime: int,
):
    """
    L1 controller for Slow Pulsar Search.

    ROW is the beam row(s) to record (in the range of 0-223), ints separated by spaces.
    Default: all.

    Can be cancelled with Ctrl-C, or with the stop_acq command for the appropriate beam
    row(s).
    """
    if logtofile:
        if logid:
            log_append = "_" + logid
        else:
            log_append = ""
        log_filename = (
            f"./logs/spsctl_{datetime.now().strftime('%Y_%m_%d')}{log_append}.log"
        )
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        logging.basicConfig(
            filename=log_filename,
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=loglevel,
            force=True,
        )

    log.info("Starting controller...")
    log.debug("Hosts: %s", host)
    log.debug("Rows: %s", rows)
    active_beams = {
        x for r in [[r, r + 1000, r + 2000, r + 3000] for r in rows] for x in r
    }
    if host:
        host_beams = set()
        for h in host:
            host_beams.update(get_node_beams(h))
        active_beams.intersection_update(host_beams)
    log.info("Controlling only beams: %s", sorted(list(active_beams)))

    try:
        if channels == -1:
            trio.run(entry_point, active_beams, basepath, source)
        else:
            first_loop = True
            while True:
                for beam in active_beams:
                    is_recording = is_beam_recording(beam, basepath)
                    if not is_recording:
                        proc = start_beam(beam, basepath, source, channels, ntime)
                    if first_loop:
                        log.info("Started all beams.")
                        first_loop = False
                    time.sleep(100)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received. Exiting...")
    except Exception as err:
        log.error("Unknown error detected: %s", err)
    finally:
        log.info("Stopping acquisition...")
        if not nocleanup:
            stop_all_beams(active_beams, basepath, source)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--host",
    multiple=True,
    metavar="HOSTNAME",
    help=(
        "Generate the schedule only for beam(s) running on HOSTNAME(s). (Repeat to"
        " include multiple hosts.)"
    ),
)
@click.argument("rows", type=click.IntRange(min=0, max=223), nargs=-1, required=True)
@click.option(
    "--loglevel",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Set logging level.",
)
@click.option("--logtofile", is_flag=True, help="Enable file logging.")
@click.option(
    "--basepath",
    default=[
        "/sps-archiver2/raw/",
        "/sps-archiver3/raw/",
        "/sps-archiver4/raw/",
        "/sps-archiver5/raw/",
    ],
    help="Path on L1 cf nodes to a CHAMPSS mount.",
    multiple=True,
)
@click.option(
    "--source",
    type=str,
    default="champss",
    help=(
        "The chime_slow_pulsar_writer object to use on L1, must be either 'champss' or 'slow'. "
        "Do not use 'slow' before consulting with the Slow team"
    ),
)
@click.option(
    "--batchsize",
    type=int,
    default=10,
    help="Jobs per row.",
)
@click.option("--splitlogs", is_flag=True, help="Split the logs per job.")
def cli_batched(
    host: Tuple[str],
    rows: Tuple[int],
    loglevel: str,
    logtofile: bool,
    basepath: str,
    source: str,
    batchsize: int,
    splitlogs: bool,
):
    """
    Batched wrapper for L1 controller.

    Same arguments as spsctl/cli(), except for --batchsize which defines the rows per
    jobs.

    Needed because otherwise generate_pointings can't catch up for many beams.
    """
    batched_rows = batched(rows, batchsize)
    all_procs = []
    log.info(f"Will record {len(rows)} pointings.")
    try:
        for count, batch in enumerate(batched_rows):
            batch_str = [str(i) for i in batch]
            current_basepath = basepath[count % len(basepath)]
            command_list = [
                "spsctl",
                *batch_str,
                "--basepath",
                current_basepath,
                "--source",
                source,
                "--loglevel",
                loglevel,
                "--nocleanup",  # Disabling cleanup makes it easier for wrapper to cleanup
            ]
            if host:
                for hostname in host:
                    command_list.extend(["--host", hostname])
            if logtofile:
                command_list.append("--logtofile")
            if splitlogs:
                command_list.extend(["--logid", f"b{count}"])
            proc = subprocess.Popen(command_list)
            all_procs.append(proc)
        log.info(f"Started {len(all_procs)} recording jobs.")
        # proc.wait in order for the program not to end, so that i can be easily cancelled
        # Not sure if that can fail in any case
        proc.wait()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received. Will exit batch recording.")
    except Exception as err:
        log.error("Unknown error detected: %s", err)
    finally:
        log.info("Exiting batch recording.")
        log.info(
            f"Logs for stopped beams should appear. Will try to stop beam rows {rows}."
        )
        # Kill all spsctl
        for proc in all_procs:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Make sure that all rpc-clients are stopped
        stop_acq.callback(
            host=host, rows=rows, debug=False, basepath=basepath[0], source=source
        )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--host",
    multiple=True,
    metavar="HOSTNAME",
    help=(
        "Stop acquisition only for beam(s) running on HOSTNAME(s). (Repeat to include"
        " multiple hosts.)"
    ),
)
@click.argument("rows", type=click.IntRange(min=0, max=223), nargs=-1, required=True)
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option(
    "--basepath",
    type=str,
    default="/sps-archiver2/raw/",
    help="Path on L1 cf nodes to a CHAMPSS mount.",
)
@click.option(
    "--source",
    type=str,
    default="champss",
    help=(
        "The chime_slow_pulsar_writer object to use on L1, must be either 'champss' or 'slow'. "
        "Do not use 'slow' before consulting with the Slow team"
    ),
)
def stop_acq(
    host: Tuple[str], rows: Tuple[int], debug: bool, basepath: str, source: str
):
    if debug:
        # Set logging level to debug
        log.setLevel(logging.DEBUG)

    log.info("Stopping acquisition using stop_acq...")
    log.debug("Hosts: %s", host)
    log.debug("Rows: %s", rows)
    active_beams = {
        x for r in [[r, r + 1000, r + 2000, r + 3000] for r in rows] for x in r
    }
    if host:
        host_beams = set()
        for h in host:
            host_beams.update(get_node_beams(h))
        active_beams.intersection_update(host_beams)
    log.info("Stopping beams: %s", sorted(list(active_beams)))
    sorted_active_beams = sorted(list(active_beams))
    stop_all_beams(sorted_active_beams, basepath, source)
