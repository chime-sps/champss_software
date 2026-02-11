"""
Core functions for creating incoherent beams from CHIME SPS data.

The incoherent beam is formed by:
1. Reading data from multiple beams to create [nbeam, nfreq, ntime] array
2. Masking fully-zero time bins and frequency channels as NaN in each beam
3. Taking the MEAN over beam axis
4. Result: 2D array [nfreq, ntime] for channelized subtraction

This module provides:
- Core functions used by both the single-chunk CLI script and the
  distributed workflow.
- ``create_chunk`` – a Workflow-callable entry point for creating an
  incoherent beam for a single time chunk.
"""

import datetime as dt
import glob
import logging
import os

import numpy as np
from functools import partial
from multiprocessing import Pool, shared_memory
from spshuff import l1_io
from rfi_mitigation.utilities.cleaner_utils import known_bad_channels

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def convert_date_to_datetime(date):
    """
    Convert date string to datetime object.
    Accepts formats: YYYY-MM-DD, YYYYMMDD, YYYY/MM/DD
    """
    if isinstance(date, (str, int)):
        for date_format in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
            try:
                date = dt.datetime.strptime(str(date), date_format)
                break
            except ValueError:
                continue
    return date


# ---------------------------------------------------------------------------
# Data I/O
# ---------------------------------------------------------------------------

def extract_data_trange(file_list, tstart, nsamp):
    """
    Extract data across multiple files for a given time range.

    Parameters
    ----------
    file_list : list[str]
        List of files belonging to one beam (in time order).
    tstart : float
        Requested start time (Unix seconds).
    nsamp : int
        Number of samples to extract.

    Returns
    -------
    data : np.ndarray or None
        Array with shape (1024, nsamp). Returns None if the interval
        cannot be satisfied.
    """
    tsamp = 0.00098304
    tend = tstart + nsamp * tsamp

    # Gather file coverage ranges
    file_ranges = []
    for fname in file_list:
        try:
            with open(fname, "rb") as f:
                int_file = l1_io.IntensityFile.from_file(
                    f, shape=(1024, None)
                )
                fh = int_file.fh
                file_ranges.append((fname, fh.start, fh.end))
        except Exception as e:
            log.warning(f"Failed to read {fname}: {e}")
            continue

    if len(file_ranges) == 0:
        log.warning("No valid files found")
        return None

    # Find files overlapping the requested interval
    overlap_files = [
        fname for fname, fstart, fend in file_ranges
        if not (fend <= tstart or fstart >= tend)
    ]
    if not overlap_files:
        log.warning("No files contain requested range")
        return None

    data_blocks = []
    for fname, fstart, fend in file_ranges:
        if fname not in overlap_files:
            continue
        try:
            with open(fname, "rb") as f:
                int_file = l1_io.IntensityFile.from_file(
                    f, shape=(1024, None)
                )

                chunks = int_file.get_chunks()
                file_data = []
                for chunk in chunks:
                    file_data.append(chunk.get_data(apply_mask=True))
                file_data = np.concatenate(file_data, axis=1)

                # build time axis for this file
                taxis = fstart + np.arange(file_data.shape[1]) * tsamp

                # restrict to relevant section
                mask = (taxis >= tstart) & (taxis < tend)
                if np.any(mask):
                    data_blocks.append(file_data[:, mask])
        except Exception as e:
            log.warning(f"Error reading {fname}: {e}")
            continue

    if not data_blocks:
        log.warning("Files overlap but no samples extracted")
        return None

    # Concatenate in time
    data = np.concatenate(data_blocks, axis=1)

    # Set fully-zero time bins to NaN to avoid jumps in median from data
    # dropouts.  Check which time samples are all zeros across all channels.
    zero_time_bins = np.all(data == 0, axis=0)
    if np.any(zero_time_bins):
        num_zero_bins = np.sum(zero_time_bins)
        log.debug(
            f"Found {num_zero_bins} fully-zero time bins "
            f"({num_zero_bins / data.shape[1] * 100:.2f}%), setting to NaN"
        )
        data[:, zero_time_bins] = np.nan

    # Ensure we have exactly nsamp samples
    if data.shape[1] < nsamp:
        log.warning(
            f"Requested {nsamp} samples, but only {data.shape[1]} available"
        )
        return data
    else:
        return data[:, :nsamp]


def group_files_by_beam(file_list):
    """
    Group CHIME files by beam ID (extracted from path).

    File structure: datpath/yyyy/mm/dd/beamno/filename.dat

    Parameters
    ----------
    file_list : list[str]
        List of file paths.

    Returns
    -------
    dict
        Dictionary mapping beam_id -> list of files for that beam.
    """
    from collections import defaultdict
    grouped = defaultdict(list)

    for fname in file_list:
        parts = fname.split('/')
        if len(parts) < 3:
            log.warning(f"Unexpected path structure: {fname}")
            continue

        # The beam number is the directory name before the filename
        # e.g., /path/to/data/2025/01/08/1234/file.dat -> beam 1234
        try:
            beam_id = int(parts[-2])
            grouped[beam_id].append(fname)
        except (ValueError, IndexError) as e:
            log.warning(f"Could not extract beam ID from {fname}: {e}")
            continue

    # Sort files within each beam by filename (which contains timestamp)
    for beam_id in grouped:
        grouped[beam_id] = sorted(grouped[beam_id])

    return dict(grouped)


def process_beam_to_shared(
    data_shared_name,
    shape,
    beam_index,
    beam_id,
    file_list,
    tstart,
    nsamp,
):
    """
    Process a single beam: read data and write full 2D (nchan, ntime) to
    shared memory.

    Parameters
    ----------
    data_shared_name : str
        Name of shared memory for data array
    shape : tuple
        Shape of shared array (nbeams, nchan, nsamp)
    beam_index : int
        Sequential index in array (0, 1, 2, ..., nbeams-1)
    beam_id : int
        Actual beam ID (e.g., 0000, 0001, ..., 1255, 2255, 3255)
    file_list : list[str]
        List of files for this beam
    tstart : float
        Start time (Unix seconds)
    nsamp : int
        Number of samples

    Returns
    -------
    int
        Beam ID (for tracking)
    """
    shared_data = shared_memory.SharedMemory(name=data_shared_name)
    data_array = np.ndarray(shape, dtype=np.float32, buffer=shared_data.buf)

    try:
        beam_data = extract_data_trange(file_list, tstart, nsamp)

        if beam_data is None:
            log.warning(f"Beam {beam_id}: No data extracted")
            data_array[beam_index, :, :] = np.nan
            shared_data.close()
            return beam_id

        nchan, ntime = beam_data.shape

        if ntime < nsamp:
            padded = np.full((nchan, nsamp), np.nan, dtype=np.float32)
            padded[:, :ntime] = beam_data
            data_array[beam_index, :, :] = padded
            log.debug(
                f"Beam {beam_id}: Padded from {ntime} to {nsamp} time samples"
            )
        else:
            data_array[beam_index, :, :] = beam_data[:, :nsamp]

        log.info(
            f"Beam {beam_id} (index {beam_index}): Processed successfully "
            f"({len(file_list)} files, {nchan} channels, "
            f"{min(ntime, nsamp)} time samples)"
        )

    except Exception as e:
        log.error(f"Beam {beam_id}: Error processing - {str(e)}")
        import traceback
        log.debug(traceback.format_exc())
        data_array[beam_index, :, :] = np.nan

    finally:
        shared_data.close()

    return beam_id


def extract_data_allbeams(
    file_list, tstart, nsamp, beam_range=None, beam_fraction=None,
    num_processes=1, nchan=1024,
):
    """
    Extract and process data from all beams in parallel, creating
    incoherent beam via mean.

    Parameters
    ----------
    file_list : list[str]
        List of all data files
    tstart : float
        Start time (Unix seconds)
    nsamp : int
        Number of samples to extract
    beam_range : tuple or None
        (min_beam, max_beam) to filter beams by last 3 digits, or None
    beam_fraction : int or None
        Use 1/N of beams, regularly spaced (e.g., 8 for 1/8), or None
        for all
    num_processes : int
        Number of parallel processes
    nchan : int
        Number of frequency channels

    Returns
    -------
    incoh_beam : np.ndarray
        Incoherent beam array of shape (nchan, nsamp)
    beam_ids : list[int]
        List of beam IDs used in median calculation
    """
    log.info("Grouping files by beam...")
    grouped = group_files_by_beam(file_list)
    beam_ids = sorted(grouped.keys())

    if len(beam_ids) == 0:
        log.error("No beams found in file list")
        return None, []

    log.info(f"Found {len(beam_ids)} beams in total")

    # Filter by beam range if specified
    # Beam structure: XYYY where X=0-3 (column), YYY=000-255 (beam within
    # column). When filtering, use the last 3 digits (YYY) across all columns.
    if beam_range is not None:
        beam_min, beam_max = beam_range
        beam_ids = [
            bid for bid in beam_ids if beam_min <= (bid % 1000) <= beam_max
        ]
        log.info(
            f"Filtered to {len(beam_ids)} beams with beam number "
            f"(last 3 digits) in range [{beam_min}, {beam_max}]"
        )

    # Filter by beam fraction (regularly spaced sampling)
    if beam_fraction is not None:
        original_count = len(beam_ids)
        beam_ids = beam_ids[::beam_fraction]
        log.info(
            f"Sampled 1/{beam_fraction} of beams: "
            f"{len(beam_ids)}/{original_count} beams (regularly spaced)"
        )

    if len(beam_ids) == 0:
        log.error("No beams found after filtering")
        return None, []

    log.info(
        f"Processing {len(beam_ids)} beams: {min(beam_ids)} to "
        f"{max(beam_ids)}"
    )
    if len(beam_ids) > 5:
        log.info(f"  Example beam IDs: {beam_ids[:5]}...")
    else:
        log.info(f"  Beam IDs: {beam_ids}")

    nbeams = len(beam_ids)
    beam_id_to_index = {beam_id: idx for idx, beam_id in enumerate(beam_ids)}

    log.info(f"Created mapping for {nbeams} beams")
    log.info(f"  Beam ID range: {min(beam_ids)} to {max(beam_ids)}")
    log.info(f"  Array index range: 0 to {nbeams - 1}")

    # Create shared memory for 3D data array: (nbeams, nchan, nsamp)
    shape = (nbeams, nchan, nsamp)
    buffer_size = int(np.prod(shape) * np.dtype(np.float32).itemsize)

    log.info(
        f"Creating shared memory array: shape={shape}, "
        f"size={buffer_size / (1024**2):.1f} MB"
    )
    data_shared = shared_memory.SharedMemory(create=True, size=buffer_size)
    data_array = np.ndarray(shape, dtype=np.float32, buffer=data_shared.buf)
    data_array[:] = np.nan  # Initialize with NaN

    log.info(
        f"Processing beams in parallel using {num_processes} processes..."
    )
    pool = Pool(num_processes)
    pool.starmap(
        partial(
            process_beam_to_shared,
            data_shared.name,
            shape,
        ),
        [
            (beam_id_to_index[beam_id], beam_id, grouped[beam_id],
             tstart, nsamp)
            for beam_id in beam_ids
        ],
    )
    pool.close()
    pool.join()

    log.info(f"Finished processing all {nbeams} beams")

    # Copy to regular array before cleaning up shared memory
    data_3d = np.array(data_array)  # Shape: (nbeams, nchan, nsamp)

    data_shared.close()
    data_shared.unlink()

    log.info(f"Loaded 3D array: shape={data_3d.shape} (nbeams, nchan, nsamp)")

    # Mask fully-zero time bins and frequency channels before taking median
    log.info("Masking fully-zero time bins and frequency channels...")

    for beam_idx in range(nbeams):
        beam_data = data_3d[beam_idx, :, :]  # Shape: (nchan, nsamp)

        zero_time_bins = np.all(beam_data == 0, axis=0)
        if np.any(zero_time_bins):
            beam_data[:, zero_time_bins] = np.nan

        zero_freq_channels = np.all(beam_data == 0, axis=1)
        if np.any(zero_freq_channels):
            beam_data[zero_freq_channels, :] = np.nan

        data_3d[beam_idx, :, :] = beam_data

    total_masked = np.isnan(data_3d).sum()
    total_elements = data_3d.size
    log.info(
        f"Masked {total_masked}/{total_elements} values "
        f"({total_masked / total_elements * 100:.2f}%) as NaN"
    )

    # Take mean along beam axis
    log.info("Computing mean over beam axis to create incoherent beam...")
    incoh_beam = np.nanmean(data_3d, axis=0)  # Shape: (nchan, nsamp)

    log.info(f"Incoherent beam shape: {incoh_beam.shape} (nchan, nsamp)")

    return incoh_beam, beam_ids


def get_datfiles_for_date_and_time(
    date_str, unix_start, unix_end, datpath='/mnt/beegfs-client/raw',
):
    """
    Get list of .dat files for a given date and time range.

    File structure: datpath/yyyy/mm/dd/beamno/*.dat

    Parameters
    ----------
    date_str : str
        Date in format yyyymmdd, yyyy-mm-dd, or yyyy/mm/dd
    unix_start : float
        Start time in Unix seconds
    unix_end : float
        End time in Unix seconds
    datpath : str
        Root directory for raw data

    Returns
    -------
    list[str]
        List of .dat files in the time range
    """
    date = convert_date_to_datetime(date_str)
    date_path = date.strftime("%Y/%m/%d")

    pattern = os.path.join(datpath, date_path, '*', '*.dat')
    log.debug(f"Searching for files matching: {pattern}")

    datfiles = sorted(glob.glob(pattern))
    log.info(f"Found {len(datfiles)} total .dat files for date {date_str}")

    if len(datfiles) == 0:
        return []

    datfiles_range = []
    for datfile in datfiles:
        fname = os.path.basename(datfile)
        try:
            t_start = int(fname.split('_')[0])
            t_end = int(fname.split('_')[1].split('.')[0])
            if (t_start > unix_start - 45) and (t_end < unix_end + 37):
                datfiles_range.append(datfile)
        except (ValueError, IndexError) as e:
            log.debug(
                f"Could not parse timestamps from filename {fname}: {e}"
            )
            continue

    log.info(f"Filtered to {len(datfiles_range)} files in time range")
    return datfiles_range


# ---------------------------------------------------------------------------
# Workflow-callable entry point
# ---------------------------------------------------------------------------

def create_chunk(
    date,
    unix_start,
    unix_end,
    output_path,
    datpath="/mnt/beegfs-client/raw",
    nchan=1024,
    num_processes=32,
    beam_fraction=None,
    beam_min=None,
    beam_max=None,
):
    """
    Workflow-callable function to create an incoherent beam for a single
    time chunk.

    This function is designed to be called by the Workflow runner inside a
    Docker container.  It wraps the core functions from this module.

    Workflow module path: ``beamformer.incoherent_beam.create_chunk``

    Parameters
    ----------
    date : str
        Date string (yyyymmdd, yyyy-mm-dd, or yyyy/mm/dd)
    unix_start : float
        Start time in Unix seconds
    unix_end : float
        End time in Unix seconds
    output_path : str
        Full path to save the output .npz file
    datpath : str
        Root directory for raw data
    nchan : int
        Number of frequency channels (always 1024)
    num_processes : int
        Number of parallel processes for beam reading
    beam_fraction : int or None
        Use 1/N of beams (e.g., 16 for 1/16), or None for all
    beam_min : int or None
        Minimum beam ID (last 3 digits), or None
    beam_max : int or None
        Maximum beam ID (last 3 digits), or None

    Returns
    -------
    tuple
        (results_dict, [], []) per Workflow convention
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    unix_start = float(unix_start)
    unix_end = float(unix_end)

    tsamp = 0.00098304
    nsamp = int((unix_end - unix_start) / tsamp)

    log.info("=" * 60)
    log.info(
        f"Creating incoherent beam chunk: {unix_start:.1f} to "
        f"{unix_end:.1f} ({nsamp} samples)"
    )
    log.info("=" * 60)

    beam_range = None
    if beam_min is not None and beam_max is not None:
        beam_range = (int(beam_min), int(beam_max))

    if beam_fraction is not None:
        beam_fraction = int(beam_fraction)

    # Find data files
    datfiles = get_datfiles_for_date_and_time(
        date, unix_start, unix_end, datpath
    )

    if len(datfiles) == 0:
        log.error("No data files found for this chunk")
        return (
            {"output_path": output_path, "status": "failed",
             "error": "no data files"},
            [], [],
        )

    # Create incoherent beam for this chunk
    incoh_beam, beam_ids = extract_data_allbeams(
        datfiles,
        tstart=unix_start,
        nsamp=nsamp,
        beam_range=beam_range,
        beam_fraction=beam_fraction,
        num_processes=num_processes,
        nchan=nchan,
    )

    if incoh_beam is None:
        log.error("Failed to extract data for this chunk")
        return (
            {"output_path": output_path, "status": "failed",
             "error": "extraction failed"},
            [], [],
        )

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save chunk
    np.savez(
        output_path,
        data=incoh_beam.astype(np.float32),
        unix_start=unix_start,
        unix_end=unix_end,
        nsamp=nsamp,
        beam_ids=beam_ids,
        num_beams=len(beam_ids),
        nchan=incoh_beam.shape[0],
        ntime=incoh_beam.shape[1],
        date=date,
    )

    log.info(f"Chunk saved to {output_path}")
    log.info(f"  Shape: {incoh_beam.shape} (nfreq, ntime)")
    log.info(f"  Beams used: {len(beam_ids)}")
    log.info(f"  NaN fraction: {np.isnan(incoh_beam).mean():.4f}")

    return (
        {
            "output_path": output_path,
            "status": "success",
            "unix_start": unix_start,
            "unix_end": unix_end,
            "num_beams": len(beam_ids),
            "shape": list(incoh_beam.shape),
        },
        [], [],
    )
