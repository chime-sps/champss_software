#!/usr/bin/env python
"""
Distributed incoherent beam creation across the compute cluster.

Splits a long time range into chunks, submits each chunk as a Docker Swarm
job via Workflow, waits for completion, then concatenates the results into
a single .npz file compatible with skybeam.py subtraction.

Example usage:
    # Create a 2-hour incoherent beam using 1/16 of beams, 120s chunks
    python create_incoherent_beam_distributed.py \\
        --date 20250806 \\
        --unix-start 1754438500 --unix-end 1754445700 \\
        --beam-fraction 16 \\
        -o /data/chime/sps/incoh_beam/incoh_20250806.npz

    # Same but with explicit beam range
    python create_incoherent_beam_distributed.py \\
        --date 20250806 \\
        --unix-start 1754438500 --unix-end 1754445700 \\
        --beam-min 0 --beam-max 255 \\
        --chunk-duration 120 \\
        -o /data/chime/sps/incoh_beam/incoh_20250806.npz
"""
import click
import logging
import os
import shutil
import time
import uuid

import numpy as np

from scheduler.workflow import (
    schedule_workflow_job,
    wait_for_no_tasks_in_states,
    docker_swarm_pending_states,
    docker_swarm_running_states,
)

log = logging.getLogger(__name__)


def split_time_range(unix_start, unix_end, chunk_duration):
    """
    Split a time range into contiguous chunks.

    Parameters
    ----------
    unix_start : float
        Start time in Unix seconds
    unix_end : float
        End time in Unix seconds
    chunk_duration : float
        Duration of each chunk in seconds

    Returns
    -------
    list[tuple[float, float]]
        List of (chunk_start, chunk_end) tuples
    """
    chunks = []
    cs = unix_start
    while cs < unix_end:
        ce = min(cs + chunk_duration, unix_end)
        chunks.append((cs, ce))
        cs = ce
    return chunks


def submit_chunk_jobs(
    chunks,
    date,
    chunk_dir,
    datpath,
    nchan,
    num_processes,
    beam_fraction,
    beam_min,
    beam_max,
    docker_image,
    docker_memory_reservation,
    workflow_buckets_name,
):
    """
    Submit Docker Swarm jobs for each chunk.

    Parameters
    ----------
    chunks : list[tuple[float, float]]
        List of (chunk_start, chunk_end) tuples
    date : str
        Date string
    chunk_dir : str
        Directory for intermediate chunk files
    datpath : str
        Root directory for raw data
    nchan : int
        Number of frequency channels
    num_processes : int
        Number of parallel processes per chunk
    beam_fraction : int or None
        Use 1/N of beams
    beam_min : int or None
        Minimum beam ID
    beam_max : int or None
        Maximum beam ID
    docker_image : str
        Docker image name
    docker_memory_reservation : float
        Memory reservation per job in GB
    workflow_buckets_name : str
        Workflow buckets collection name

    Returns
    -------
    work_ids : list[str]
        List of Workflow work IDs
    chunk_files : list[str]
        List of expected chunk output file paths
    """
    work_ids = []
    chunk_files = []

    for i, (cs, ce) in enumerate(chunks):
        chunk_output = os.path.join(
            chunk_dir, f"chunk_{i:04d}_{cs:.0f}_{ce:.0f}.npz"
        )
        chunk_files.append(chunk_output)

        docker_name = (
            f"incoh-beam-{date}-{i:04d}-{cs:.0f}"
        )
        docker_mounts = [
            f"{datpath}:{datpath}",
            f"{chunk_dir}:{chunk_dir}",
        ]

        workflow_params = {
            "date": date,
            "unix_start": cs,
            "unix_end": ce,
            "output_path": chunk_output,
            "datpath": datpath,
            "nchan": nchan,
            "num_processes": num_processes,
            "beam_fraction": beam_fraction,
            "beam_min": beam_min,
            "beam_max": beam_max,
        }
        workflow_tags = ["incoh-beam", date, f"chunk-{i:04d}"]

        # Throttle: wait until no pending incoh-beam jobs before submitting
        # the next one, to avoid overloading Docker Swarm's queue
        wait_for_no_tasks_in_states(
            docker_swarm_pending_states,
            docker_service_name_prefix="incoh-beam",
        )

        log.info(
            f"Submitting chunk {i + 1}/{len(chunks)}: "
            f"{cs:.1f} to {ce:.1f} ({ce - cs:.1f}s)"
        )

        work_id = schedule_workflow_job(
            docker_image=docker_image,
            docker_mounts=docker_mounts,
            docker_name=docker_name,
            docker_memory_reservation=docker_memory_reservation,
            workflow_buckets_name=workflow_buckets_name,
            workflow_function="beamformer.incoherent_beam.create_chunk",
            workflow_params=workflow_params,
            workflow_tags=workflow_tags,
            timeout=60 * 30,  # 30 minute timeout per chunk
        )
        work_ids.append(work_id)

    return work_ids, chunk_files


def wait_for_completion():
    """Wait for all incoh-beam Docker Swarm jobs to finish."""
    log.info("Waiting for all chunk jobs to complete...")
    wait_for_no_tasks_in_states(
        docker_swarm_running_states,
        docker_service_name_prefix="incoh-beam",
    )
    log.info("All chunk jobs have finished.")


def concatenate_chunks(chunk_files):
    """
    Load and concatenate chunk .npz files along the time axis.

    Parameters
    ----------
    chunk_files : list[str]
        List of chunk .npz file paths (in time order)

    Returns
    -------
    combined_data : np.ndarray
        Combined incoherent beam array [nfreq, ntime]
    combined_unix_start : float
        Start time of combined array
    combined_unix_end : float
        End time of combined array
    beam_ids : list
        Beam IDs from the first chunk
    nchan : int
        Number of frequency channels
    date : str
        Date string
    missing_chunks : list[int]
        Indices of missing/failed chunks
    """
    all_data = []
    combined_unix_start = None
    combined_unix_end = None
    beam_ids = None
    nchan = None
    date = None
    missing_chunks = []

    for i, chunk_file in enumerate(chunk_files):
        if not os.path.exists(chunk_file):
            missing_chunks.append(i)
            log.warning(f"Chunk {i} missing: {chunk_file}")
            continue

        try:
            chunk = np.load(chunk_file, allow_pickle=True)
            data = chunk['data']  # Shape: [nfreq, ntime_chunk]
            cs = float(chunk['unix_start'])
            ce = float(chunk['unix_end'])

            all_data.append(data)

            if combined_unix_start is None or cs < combined_unix_start:
                combined_unix_start = cs
            if combined_unix_end is None or ce > combined_unix_end:
                combined_unix_end = ce

            # Save metadata from first successful chunk
            if beam_ids is None:
                beam_ids = chunk['beam_ids']
                nchan = int(chunk['nchan'])
                date = str(chunk['date'])

            log.info(
                f"  Loaded chunk {i}: shape={data.shape}, "
                f"time={cs:.1f}-{ce:.1f}"
            )
        except Exception as e:
            missing_chunks.append(i)
            log.error(f"Chunk {i} failed to load ({chunk_file}): {e}")
            continue

    if len(all_data) == 0:
        raise RuntimeError("No chunks loaded successfully")

    # Concatenate along time axis
    combined_data = np.concatenate(all_data, axis=1)
    log.info(
        f"Combined shape: {combined_data.shape} "
        f"({combined_unix_start:.1f} to {combined_unix_end:.1f})"
    )

    return (
        combined_data, combined_unix_start, combined_unix_end,
        beam_ids, nchan, date, missing_chunks,
    )


@click.command()
@click.option(
    '--date', type=str, required=True,
    help='Date in format yyyymmdd, yyyy-mm-dd, or yyyy/mm/dd',
)
@click.option(
    '--unix-start', type=float, required=True,
    help='Start time in Unix seconds',
)
@click.option(
    '--unix-end', type=float, required=True,
    help='End time in Unix seconds',
)
@click.option(
    '--chunk-duration', type=float, default=120.0,
    help='Duration of each chunk in seconds (default: 120)',
)
@click.option(
    '--beam-fraction', type=int, default=16,
    help='Use 1/N of beams, regularly spaced (default: 16 for ~64 beams)',
)
@click.option(
    '--beam-min', type=int, default=None,
    help='Minimum beam ID (last 3 digits). Alternative to --beam-fraction.',
)
@click.option(
    '--beam-max', type=int, default=None,
    help='Maximum beam ID (last 3 digits). Alternative to --beam-fraction.',
)
@click.option(
    '--output', '-o', required=True,
    help='Output file path for final combined incoherent beam (.npz)',
)
@click.option(
    '--datpath', default='/mnt/beegfs-client/raw',
    help='Root directory for raw data',
)
@click.option(
    '--nchan', type=int, default=1024,
    help='Number of frequency channels (default: 1024)',
)
@click.option(
    '--num-processes', type=int, default=32,
    help='Number of parallel processes per chunk (default: 32)',
)
@click.option(
    '--tmpdir', default='/data/chime/sps/incoh_beam/chunks',
    help='Directory for intermediate chunk files',
)
@click.option(
    '--docker-image',
    default='sps-archiver1.chime:5000/champss_software:latest',
    help='Docker image name',
)
@click.option(
    '--docker-memory-reservation', type=float, default=40.0,
    help='Memory reservation per chunk job in GB (default: 40)',
)
@click.option(
    '--workflow-buckets-name', default='champss-incoh-beam',
    help='Workflow buckets collection name',
)
@click.option(
    '--cleanup/--no-cleanup', default=True,
    help='Remove intermediate chunk files after combining (default: cleanup)',
)
def create_incoherent_beam_distributed(
    date, unix_start, unix_end, chunk_duration,
    beam_fraction, beam_min, beam_max,
    output, datpath, nchan, num_processes,
    tmpdir, docker_image, docker_memory_reservation,
    workflow_buckets_name, cleanup,
):
    """
    Create an incoherent beam over a long time range by distributing
    the work across the compute cluster.

    The time range is split into chunks (default 120s each). Each chunk
    is submitted as a Docker Swarm job that creates a partial incoherent
    beam. After all chunks complete, the results are concatenated into
    a single .npz file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Validate beam selection
    if beam_min is not None or beam_max is not None:
        if beam_min is None or beam_max is None:
            raise click.BadParameter(
                "Must specify both --beam-min and --beam-max or neither"
            )
        # When using explicit beam range, don't also use fraction
        beam_fraction = None

    total_duration = unix_end - unix_start
    log.info("=" * 60)
    log.info("Distributed Incoherent Beam Creation")
    log.info("=" * 60)
    log.info(f"Date: {date}")
    log.info(f"Time range: {unix_start:.1f} to {unix_end:.1f} "
             f"({total_duration:.1f}s = {total_duration / 60:.1f} min)")
    log.info(f"Chunk duration: {chunk_duration:.1f}s")
    log.info(f"Data path: {datpath}")
    log.info(f"Output: {output}")
    if beam_fraction is not None:
        log.info(f"Beam fraction: 1/{beam_fraction}")
    else:
        log.info(f"Beam range: {beam_min} to {beam_max}")
    log.info(f"Docker image: {docker_image}")
    log.info(f"Memory reservation: {docker_memory_reservation} GB per chunk")

    # Split time range into chunks
    chunks = split_time_range(unix_start, unix_end, chunk_duration)
    log.info(f"Split into {len(chunks)} chunks")

    # Create temporary directory for chunk outputs
    job_id = str(uuid.uuid4())[:8]
    chunk_dir = os.path.join(tmpdir, f"incoh_{date}_{job_id}")
    os.makedirs(chunk_dir, exist_ok=True)
    log.info(f"Chunk output directory: {chunk_dir}")

    # Submit all chunk jobs
    t_submit_start = time.time()
    work_ids, chunk_files = submit_chunk_jobs(
        chunks=chunks,
        date=date,
        chunk_dir=chunk_dir,
        datpath=datpath,
        nchan=nchan,
        num_processes=num_processes,
        beam_fraction=beam_fraction,
        beam_min=beam_min,
        beam_max=beam_max,
        docker_image=docker_image,
        docker_memory_reservation=docker_memory_reservation,
        workflow_buckets_name=workflow_buckets_name,
    )
    t_submit_end = time.time()
    log.info(
        f"All {len(chunks)} jobs submitted in "
        f"{t_submit_end - t_submit_start:.1f}s"
    )

    # Wait for all jobs to finish
    t_wait_start = time.time()
    wait_for_completion()
    t_wait_end = time.time()
    log.info(f"All jobs completed in {t_wait_end - t_wait_start:.1f}s")

    # Concatenate chunk results
    log.info("Concatenating chunk results...")
    (
        combined_data, combined_unix_start, combined_unix_end,
        beam_ids, nchan_actual, date_actual, missing_chunks,
    ) = concatenate_chunks(chunk_files)

    if missing_chunks:
        n_missing = len(missing_chunks)
        n_total = len(chunks)
        log.warning(
            f"{n_missing}/{n_total} chunks failed: {missing_chunks}"
        )
        if n_missing > n_total * 0.25:
            log.error(
                "More than 25% of chunks failed. Output may be unreliable."
            )

    # Create output directory if needed
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save combined result
    log.info(f"Saving combined incoherent beam to {output}...")
    np.savez(
        output,
        data=combined_data.astype(np.float32),
        unix_start=combined_unix_start,
        unix_end=combined_unix_end,
        nsamp=combined_data.shape[1],
        beam_ids=beam_ids,
        num_beams=len(beam_ids),
        nchan=combined_data.shape[0],
        ntime=combined_data.shape[1],
        date=date_actual,
    )

    # Cleanup intermediate files
    if cleanup:
        log.info(f"Cleaning up chunk directory: {chunk_dir}")
        shutil.rmtree(chunk_dir)

    # Summary
    log.info("=" * 60)
    log.info("SUCCESS: Distributed incoherent beam created")
    log.info("=" * 60)
    log.info(f"File: {output}")
    log.info(f"Shape: {combined_data.shape} (nfreq, ntime)")
    log.info(f"Time range: {combined_unix_start:.1f} to "
             f"{combined_unix_end:.1f}")
    log.info(f"Chunks: {len(chunks) - len(missing_chunks)}/{len(chunks)} "
             f"successful")
    log.info(f"Number of beams used: {len(beam_ids)}")
    log.info(f"Statistics:")
    log.info(f"  Mean: {np.nanmean(combined_data):.2f}")
    log.info(f"  Median: {np.nanmedian(combined_data):.2f}")
    log.info(f"  Std: {np.nanstd(combined_data):.2f}")
    log.info(f"  NaN fraction: {np.isnan(combined_data).mean():.4f}")


if __name__ == '__main__':
    create_incoherent_beam_distributed()
