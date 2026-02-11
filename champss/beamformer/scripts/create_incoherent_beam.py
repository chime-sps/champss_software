#!/usr/bin/env python
"""
Script to create and save an incoherent beam from CHIME SPS data.

The incoherent beam is formed by:
1. Reading data from multiple beams to create [nbeam, nfreq, ntime] array
2. Masking fully-zero time bins and frequency channels as NaN in each beam
3. Taking the MEDIAN over beam axis (robust to outliers)
4. Result: 2D array [nfreq, ntime] for channelized subtraction
"""
import click
import numpy as np
import os
import logging

from beamformer.incoherent_beam import (
    convert_date_to_datetime,
    extract_data_allbeams,
    get_datfiles_for_date_and_time,
)

log = logging.getLogger(__name__)


@click.command()
@click.option('--date', type=str, required=True, help='Date in format yyyymmdd, yyyy-mm-dd, or yyyy/mm/dd')
@click.option('--unix-start', type=float, required=True, help='Start time in Unix seconds')
@click.option('--unix-end', type=float, required=True, help='End time in Unix seconds')
@click.option('--nsamp', type=int, default=None, help='Number of samples to extract (overrides unix-end if specified)')
@click.option('--beam-min', type=int, default=None, help='Minimum beam ID to include (optional, last 3 digits)')
@click.option('--beam-max', type=int, default=None, help='Maximum beam ID to include (optional, last 3 digits)')
@click.option('--beam-fraction', type=int, default=None, help='Use 1/N of beams, regularly spaced (e.g., 8 for 1/8, 16 for 1/16). Alternative to --beam-min/--beam-max.')
@click.option('--output', '-o', required=True, help='Output file path for incoherent beam (.npz)')
@click.option('--datpath', default='/mnt/beegfs-client/raw', help='Root directory for raw data (default: /mnt/beegfs-client/raw)')
@click.option('--nchan', type=int, default=1024, help='Number of frequency channels (default: 1024)')
@click.option('--num-processes', type=int, default=32, help='Number of parallel processes (default: 32)')
def create_incoherent_beam(date, unix_start, unix_end, nsamp, beam_min, beam_max, beam_fraction,
                          output, datpath, nchan, num_processes):
    """
    Create and save an incoherent beam from CHIME SPS data.

    The incoherent beam is formed by:
    1. Reading data from multiple beams to create [nbeam, nfreq, ntime] array
    2. Masking fully-zero time bins and frequency channels as NaN in each beam
    3. Taking the MEDIAN over beam axis (robust to outliers)
    4. Result: 2D array [nfreq, ntime] for channelized subtraction

    Example usage:
        # Use specific beam range (last 3 digits, across all 4 columns)
        python create_incoherent_beam.py --date 20250806 --unix-start 1754438500
               --unix-end 1754438600 --beam-min 120 --beam-max 150 -o incoh_beam.npz

        # Use 1/8 of all beams (regularly spaced) for faster processing
        python create_incoherent_beam.py --date 20250806 --unix-start 1754438500
               --unix-end 1754438600 --beam-fraction 8 -o incoh_beam.npz
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    log.info("=" * 60)
    log.info("Creating Incoherent Beam (Median-based, 2D [nfreq, ntime])")
    log.info("=" * 60)

    # Calculate number of samples if not specified
    tsamp = 0.00098304
    if nsamp is None:
        nsamp = int((unix_end - unix_start) / tsamp)
        log.info(f"Calculated nsamp = {nsamp} from time range")
    else:
        log.info(f"Using specified nsamp = {nsamp}")

    # Validate beam selection options
    if beam_fraction is not None and (beam_min is not None or beam_max is not None):
        raise click.BadParameter("Cannot specify both --beam-fraction and --beam-min/--beam-max")

    # Get beam range
    beam_range = None
    if beam_min is not None and beam_max is not None:
        beam_range = (beam_min, beam_max)
        log.info(f"Beam range: {beam_min} to {beam_max} (last 3 digits)")
    elif beam_min is not None or beam_max is not None:
        raise click.BadParameter("Must specify both --beam-min and --beam-max or neither")
    elif beam_fraction is not None:
        if beam_fraction < 1:
            raise click.BadParameter("--beam-fraction must be >= 1")
        log.info(f"Using 1/{beam_fraction} of beams (regularly spaced)")
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

    # Extract data from all beams -> [nchan, ntime]
    log.info("Creating 3D array [nbeam, nfreq, ntime] and computing median...")
    incoh_beam, beam_ids = extract_data_allbeams(
        datfiles_range,
        tstart=unix_start,
        nsamp=nsamp,
        beam_range=beam_range,
        beam_fraction=beam_fraction,
        num_processes=num_processes,
        nchan=nchan
    )

    if incoh_beam is None:
        log.error("Failed to extract data")
        return

    log.info(f"Incoherent beam shape: {incoh_beam.shape} (nfreq, ntime)")
    log.info(f"Number of beams used: {len(beam_ids)}")
    log.info(f"Beam IDs: min={min(beam_ids)}, max={max(beam_ids)}")

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
    # NOTE: 'data' is now 2D array [nfreq, ntime]
    np.savez(
        output,
        data=incoh_beam.astype(np.float32),  # 2D array [nfreq, ntime]
        unix_start=unix_start,
        unix_end=unix_end,
        nsamp=nsamp,
        beam_ids=beam_ids,
        num_beams=len(beam_ids),
        beam_range=beam_range,
        date=date,
        ntime=incoh_beam.shape[1],
        nchan=incoh_beam.shape[0],
    )

    log.info("=" * 60)
    log.info("SUCCESS: Incoherent beam created and saved")
    log.info("=" * 60)
    log.info(f"File: {output}")
    log.info(f"Shape: {incoh_beam.shape} (nfreq, ntime)")
    log.info(f"Number of beams used: {len(beam_ids)}")
    log.info(f"Method: Mean over beams")
    log.info(f"Usage: Channelized subtraction with tiling support for finer channelization")

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
