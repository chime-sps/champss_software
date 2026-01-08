#!/usr/bin/env python
"""
Script to create and save an incoherent beam from CHIME SPS data.

The incoherent beam is formed by:
1. Reading data from multiple beams
2. Applying bad channel mask to each beam
3. Frequency-averaging each beam to get I(time)
4. Taking the MEDIAN over beams (robust to outliers)
5. Resulting in a 1D time series I(time) for achromatic subtraction

This targets achromatic power variations in time.
"""
import click
import numpy as np
import glob
import os
import logging
import datetime as dt
from astropy.time import Time
from spshuff import l1_io
from multiprocessing import Pool, set_start_method, shared_memory
from functools import partial
from rfi_mitigation.utilities.cleaner_utils import known_bad_channels

log = logging.getLogger(__name__)


def convert_date_to_datetime(date):
    """
    Convert date string to datetime object.
    Accepts formats: YYYY-MM-DD, YYYYMMDD, YYYY/MM/DD

    Matches the logic from scheduler.utils.convert_date_to_datetime
    """
    if isinstance(date, str) or isinstance(date, int):
        for date_format in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
            try:
                date = dt.datetime.strptime(str(date), date_format)
                break
            except ValueError:
                continue
    return date


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
    data : np.ndarray
        Array with shape (1024, nsamp). Returns None if the interval cannot be satisfied.
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
    overlap_files = [fname for fname, fstart, fend in file_ranges if not (fend <= tstart or fstart >= tend)]
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

    # Ensure we have exactly nsamp samples
    if data.shape[1] < nsamp:
        log.warning(f"Requested {nsamp} samples, but only {data.shape[1]} available")
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
        # Extract beam ID from path: datpath/yyyy/mm/dd/beamno/
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
    bad_channels
):
    """
    Process a single beam: read data, apply bad channel mask, frequency-average,
    and write to shared memory array at sequential index.

    Parameters
    ----------
    data_shared_name : str
        Name of shared memory for data array
    shape : tuple
        Shape of shared array (nbeams, nsamp)
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
    bad_channels : list
        List of bad channel indices to mask

    Returns
    -------
    int
        Beam ID (for tracking)
    """
    # Access shared memory
    shared_data = shared_memory.SharedMemory(name=data_shared_name)
    data_array = np.ndarray(shape, dtype=np.float32, buffer=shared_data.buf)

    try:
        # Extract 2D data for this beam (nchan, ntime)
        beam_data = extract_data_trange(file_list, tstart, nsamp)

        if beam_data is None:
            log.warning(f"Beam {beam_id}: No data extracted")
            # Fill with NaN to indicate no data
            data_array[beam_index, :] = np.nan
            shared_data.close()
            return beam_id

        nchan = beam_data.shape[0]

        # Apply bad channel mask
        channel_mask = np.ones(nchan, dtype=bool)
        if len(bad_channels) > 0:
            # Ensure bad channel indices are within bounds
            valid_bad_chans = [ch for ch in bad_channels if 0 <= ch < nchan]
            channel_mask[valid_bad_chans] = False
            log.debug(f"Beam {beam_id}: Masking {len(valid_bad_chans)} bad channels")

        # Frequency-average: I(time) = mean over frequency (excluding bad channels)
        # Use nanmean to handle any NaN values in the data
        if np.sum(channel_mask) > 0:
            freq_avg = np.nanmean(beam_data[channel_mask, :], axis=0)
        else:
            log.warning(f"Beam {beam_id}: All channels masked!")
            freq_avg = np.full(beam_data.shape[1], np.nan)

        # Handle case where all channels are bad or data is all NaN
        if np.all(np.isnan(freq_avg)):
            log.warning(f"Beam {beam_id}: All NaN after frequency averaging")
            data_array[beam_index, :] = np.nan
        else:
            # Pad or trim to match nsamp
            if len(freq_avg) < nsamp:
                # Pad with NaN if too short
                padded = np.full(nsamp, np.nan, dtype=np.float32)
                padded[:len(freq_avg)] = freq_avg
                data_array[beam_index, :] = padded
                log.debug(f"Beam {beam_id}: Padded from {len(freq_avg)} to {nsamp} samples")
            else:
                data_array[beam_index, :] = freq_avg[:nsamp]

        log.info(f"Beam {beam_id} (index {beam_index}): Processed successfully ({len(file_list)} files, {nchan} channels)")

    except Exception as e:
        log.error(f"Beam {beam_id}: Error processing - {str(e)}")
        import traceback
        log.debug(traceback.format_exc())
        data_array[beam_index, :] = np.nan

    finally:
        shared_data.close()

    return beam_id


def extract_data_allbeams(file_list, tstart, nsamp, beam_range=None, num_processes=1, nchan=1024):
    """
    Extract and process data from all beams in parallel, creating I(beam, time) array.

    Parameters
    ----------
    file_list : list[str]
        List of all data files
    tstart : float
        Start time (Unix seconds)
    nsamp : int
        Number of samples to extract
    beam_range : tuple or None
        (min_beam, max_beam) to filter beams, or None for all
    num_processes : int
        Number of parallel processes
    nchan : int
        Number of frequency channels (for bad channel mask)

    Returns
    -------
    data_array : np.ndarray
        Array of shape (nbeams, nsamp) containing I(beam, time)
    beam_ids : list[int]
        List of beam IDs corresponding to rows in data_array
    """
    # Group files by beam
    log.info("Grouping files by beam...")
    grouped = group_files_by_beam(file_list)
    beam_ids = sorted(grouped.keys())

    if len(beam_ids) == 0:
        log.error("No beams found in file list")
        return None, []

    log.info(f"Found {len(beam_ids)} beams in total")

    # Filter by beam range if specified
    if beam_range is not None:
        beam_min, beam_max = beam_range
        beam_ids = [bid for bid in beam_ids if beam_min <= bid <= beam_max]
        log.info(f"Filtered to {len(beam_ids)} beams in range [{beam_min}, {beam_max}]")

    if len(beam_ids) == 0:
        log.error("No beams found in specified range")
        return None, []

    log.info(f"Processing {len(beam_ids)} beams: {min(beam_ids)} to {max(beam_ids)}")
    log.info(f"  Example beam IDs: {beam_ids[:5]}..." if len(beam_ids) > 5 else f"  Beam IDs: {beam_ids}")

    # Get bad channels from pipeline
    log.info(f"Loading bad channel mask for {nchan} channels...")
    bad_channels = known_bad_channels(nchan=nchan)
    log.info(f"Masking {len(bad_channels)} bad channels")

    # Create beam_id to sequential index mapping
    # beam_ids are like [0, 1, 2, ..., 255, 1255, 2255, 3255]
    # We map them to sequential indices [0, 1, 2, ..., len(beam_ids)-1]
    nbeams = len(beam_ids)
    beam_id_to_index = {beam_id: idx for idx, beam_id in enumerate(beam_ids)}

    log.info(f"Created mapping for {nbeams} beams")
    log.info(f"  Beam ID range: {min(beam_ids)} to {max(beam_ids)}")
    log.info(f"  Array index range: 0 to {nbeams-1}")

    # Create shared memory for data array: I(beam, time)
    # Use sequential indices, so array size is (nbeams, nsamp)
    shape = (nbeams, nsamp)
    buffer_size = int(np.prod(shape) * np.dtype(np.float32).itemsize)

    log.info(f"Creating shared memory array: shape={shape}, size={buffer_size/(1024**2):.1f} MB")
    data_shared = shared_memory.SharedMemory(create=True, size=buffer_size)
    data_array = np.ndarray(shape, dtype=np.float32, buffer=data_shared.buf)
    data_array[:] = np.nan  # Initialize with NaN

    # Process beams in parallel
    log.info(f"Processing beams in parallel using {num_processes} processes...")
    pool = Pool(num_processes)
    pool.starmap(
        partial(
            process_beam_to_shared,
            data_shared.name,
            shape,
        ),
        [(beam_id_to_index[beam_id], beam_id, grouped[beam_id], tstart, nsamp, bad_channels)
         for beam_id in beam_ids]
    )
    pool.close()
    pool.join()

    log.info(f"Finished processing all {nbeams} beams")

    # Copy to regular array before cleaning up shared memory
    data_beams = np.array(data_array)

    # Clean up shared memory
    data_shared.close()
    data_shared.unlink()

    # data_beams is already in the right order (rows correspond to beam_ids in order)
    return data_beams, beam_ids


def get_datfiles_for_date_and_time(date_str, unix_start, unix_end, datpath='/mnt/beegfs-client/raw'):
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
    # Convert date string to datetime, then to path format (yyyy/mm/dd)
    # Matches logic from pipeline.py
    date = convert_date_to_datetime(date_str)
    date_path = date.strftime("%Y/%m/%d")

    # Construct path pattern: datpath/yyyy/mm/dd/*/*.dat
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
            # Filename format: unixstart_unixend.dat
            t_start = int(fname.split('_')[0])
            t_end = int(fname.split('_')[1].split('.')[0])
            # Include files with some overlap (with buffer of ~45-37 seconds)
            if (t_start > unix_start - 45) and (t_end < unix_end + 37):
                datfiles_range.append(datfile)
        except (ValueError, IndexError) as e:
            log.debug(f"Could not parse timestamps from filename {fname}: {e}")
            continue

    log.info(f"Filtered to {len(datfiles_range)} files in time range")
    return datfiles_range


@click.command()
@click.option('--date', type=str, required=True, help='Date in format yyyymmdd, yyyy-mm-dd, or yyyy/mm/dd')
@click.option('--unix-start', type=float, required=True, help='Start time in Unix seconds')
@click.option('--unix-end', type=float, required=True, help='End time in Unix seconds')
@click.option('--nsamp', type=int, default=None, help='Number of samples to extract (overrides unix-end if specified)')
@click.option('--beam-min', type=int, default=None, help='Minimum beam ID to include (optional)')
@click.option('--beam-max', type=int, default=None, help='Maximum beam ID to include (optional)')
@click.option('--output', '-o', required=True, help='Output file path for incoherent beam (.npz)')
@click.option('--datpath', default='/mnt/beegfs-client/raw', help='Root directory for raw data (default: /mnt/beegfs-client/raw)')
@click.option('--nchan', type=int, default=1024, help='Number of frequency channels (default: 1024)')
@click.option('--num-processes', type=int, default=32, help='Number of parallel processes (default: 32)')
def create_incoherent_beam(date, unix_start, unix_end, nsamp, beam_min, beam_max,
                          output, datpath, nchan, num_processes):
    """
    Create and save an incoherent beam from CHIME SPS data.

    The incoherent beam is formed by:
    1. Reading data from multiple beams
    2. Applying bad channel mask to each beam (same as pipeline)
    3. Frequency-averaging each beam to get I(time)
    4. Taking the MEDIAN over beams (robust to outliers)
    5. Result: 1D time series I(time) for achromatic subtraction

    Example usage:
        python create_incoherent_beam.py --date 20250806 --unix-start 1754438500
               --unix-end 1754438600 --beam-min 0 --beam-max 100 -o incoh_beam.npz
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    log.info("=" * 60)
    log.info("Creating Incoherent Beam (Median-based, Achromatic)")
    log.info("=" * 60)

    # Calculate number of samples if not specified
    tsamp = 0.00098304
    if nsamp is None:
        nsamp = int((unix_end - unix_start) / tsamp)
        log.info(f"Calculated nsamp = {nsamp} from time range")
    else:
        log.info(f"Using specified nsamp = {nsamp}")

    # Get beam range
    beam_range = None
    if beam_min is not None and beam_max is not None:
        beam_range = (beam_min, beam_max)
        log.info(f"Beam range: {beam_min} to {beam_max}")
    elif beam_min is not None or beam_max is not None:
        raise click.BadParameter("Must specify both --beam-min and --beam-max or neither")
    else:
        log.info("Using all available beams")

    log.info(f"Date: {date}")
    log.info(f"Time range: {unix_start} to {unix_end} (Unix seconds)")
    log.info(f"Number of samples: {nsamp}")
    log.info(f"Number of channels: {nchan}")
    log.info(f"Data path: {datpath}")
    log.info(f"Output file: {output}")

    # Get list of data files
    log.info("Finding data files...")
    datfiles_range = get_datfiles_for_date_and_time(date, unix_start, unix_end, datpath)

    if len(datfiles_range) == 0:
        log.error("No data files found for specified date and time range")
        # Convert date to show correct path format
        date_obj = convert_date_to_datetime(date)
        date_path = date_obj.strftime("%Y/%m/%d")
        log.error(f"  Searched in: {datpath}/{date_path}/*/")
        log.error(f"  Time range: {unix_start} to {unix_end}")
        return

    log.info(f"Found {len(datfiles_range)} data files in time range")

    # Extract data from all beams -> I(beam, time)
    log.info("Extracting and frequency-averaging data from beams...")
    data_beams, beam_ids = extract_data_allbeams(
        datfiles_range,
        tstart=unix_start,
        nsamp=nsamp,
        beam_range=beam_range,
        num_processes=num_processes,
        nchan=nchan
    )

    if data_beams is None:
        log.error("Failed to extract data")
        return

    log.info(f"Data shape: {data_beams.shape} (nbeams, ntime)")
    log.info(f"Number of beams: {len(beam_ids)}")
    log.info(f"Beam IDs: min={min(beam_ids)}, max={max(beam_ids)}")

    # Check how many beams have valid data
    valid_beams = ~np.all(np.isnan(data_beams), axis=1)
    num_valid = np.sum(valid_beams)
    log.info(f"Valid beams (with data): {num_valid}/{len(beam_ids)}")

    if num_valid == 0:
        log.error("No beams have valid data")
        return

    if num_valid < 3:
        log.warning(f"Only {num_valid} valid beams - median may not be robust")

    # Take median over beam axis -> I(time)
    log.info("Computing median over beam axis (robust to outliers)...")
    # Use nanmedian to ignore NaN values (beams with no data)
    incoh_beam = np.nanmedian(data_beams, axis=0)

    log.info(f"Incoherent beam shape: {incoh_beam.shape} (1D time series)")

    # Check for NaN values in final result
    nan_frac = np.isnan(incoh_beam).mean()
    if nan_frac > 0:
        log.warning(f"Incoherent beam contains {nan_frac*100:.2f}% NaN values")
        if nan_frac > 0.5:
            log.error("More than 50% of incoherent beam is NaN - insufficient data")
            return

    # Save to file
    log.info(f"Saving incoherent beam to {output}...")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.info(f"Created directory: {output_dir}")

    # Save with metadata
    # NOTE: 'data' is now 1D time series, not 2D (nchan, ntime)
    np.savez(
        output,
        data=incoh_beam.astype(np.float32),  # 1D time series I(time)
        unix_start=unix_start,
        unix_end=unix_end,
        nsamp=nsamp,
        beam_ids=beam_ids,
        num_beams=num_valid,
        beam_range=beam_range,
        date=date,
        ntime=len(incoh_beam),
        nchan=nchan,  # Store nchan used for bad channel mask
        achromatic=True,  # Flag indicating this is for achromatic subtraction
    )

    log.info("=" * 60)
    log.info("SUCCESS: Incoherent beam created and saved")
    log.info("=" * 60)
    log.info(f"File: {output}")
    log.info(f"Shape: {incoh_beam.shape} (1D time series)")
    log.info(f"Number of beams used: {num_valid}")
    log.info(f"Method: Median over beams (robust to outliers)")
    log.info(f"Usage: Achromatic subtraction (same value from all channels)")

    # Print some statistics
    log.info("Statistics:")
    log.info(f"  Mean: {np.nanmean(incoh_beam):.2f}")
    log.info(f"  Median: {np.nanmedian(incoh_beam):.2f}")
    log.info(f"  Std: {np.nanstd(incoh_beam):.2f}")
    log.info(f"  Min: {np.nanmin(incoh_beam):.2f}")
    log.info(f"  Max: {np.nanmax(incoh_beam):.2f}")
    log.info(f"  NaN fraction: {np.isnan(incoh_beam).mean():.4f}")


if __name__ == '__main__':
    create_incoherent_beam()
