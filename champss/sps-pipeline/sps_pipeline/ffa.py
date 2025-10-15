"""FFA (Fast Folding Algorithm) search module for the SPS pipeline."""

import logging
import datetime
import os
import numpy as np
from omegaconf import OmegaConf

from FFA.FFA_search import FFA_search
import riptide
from scipy.optimize import curve_fit
from sps_common.constants import TSAMP

log = logging.getLogger(__name__)


class FFAProcessor:
    """
    FFA search processor for the SPS pipeline.

    This class handles FFA search on dedispersed time series, including:
    - Periodogram creation across DM range
    - Peak detection and filtering
    - Candidate clustering
    - Writing detections and candidates to disk

    Parameters
    ----------
    run_ffa_search : bool
        Whether to run FFA search
    create_periodogram_plots : bool
        Whether to create periodogram plots
    create_profile_plots : bool
        Whether to create pulse profile plots
    write_ffa_detections : bool
        Whether to write raw detections to file
    write_ffa_candidates : bool
        Whether to write clustered candidates to file
    write_ffa_stack : bool
        Whether to write stacked periodograms
    dm_downsample : int
        DM downsampling factor (default: 8)
    dm_min : float
        Minimum DM to search (default: 0)
    dm_max : float
        Maximum DM to search (default: None, uses max from dedisp_ts)
    min_rfi_sigma : float
        Minimum sigma for RFI detection (default: 5)
    min_detection_sigma : float
        Minimum sigma for valid detections (default: 7)
    birdie_tolerance : float
        Tolerance for birdie removal in seconds (default: 0.005)
    num_threads : int
        Number of threads for parallel processing (default: 8)
    """

    def __init__(
        self,
        run_ffa_search=True,
        create_periodogram_plots=False,
        create_profile_plots=False,
        write_ffa_detections=True,
        write_ffa_candidates=True,
        write_ffa_stack=True,
        dm_downsample=8,
        dm_min=0,
        dm_max=None,
        min_rfi_sigma=5,
        min_detection_sigma=7,
        birdie_tolerance=0.005,
        num_threads=8,
    ):
        self.run_ffa_search = run_ffa_search
        self.create_periodogram_plots = create_periodogram_plots
        self.create_profile_plots = create_profile_plots
        self.write_ffa_detections = write_ffa_detections
        self.write_ffa_candidates = write_ffa_candidates
        self.write_ffa_stack = write_ffa_stack
        self.dm_downsample = dm_downsample
        self.dm_min = dm_min
        self.dm_max = dm_max
        self.min_rfi_sigma = min_rfi_sigma
        self.min_detection_sigma = min_detection_sigma
        self.birdie_tolerance = birdie_tolerance
        self.num_threads = num_threads

    def run_ffa_search_on_dedisp(
        self,
        dedisp_ts,
        obs_id,
        ra,
        dec,
        date,
        basepath="./",
        birdies=None,
    ):
        """
        Run FFA search on dedispersed time series using the full FFA_search implementation.

        Parameters
        ----------
        dedisp_ts : DedispersedTimeSeries
            Dedispersed time series object
        obs_id : str
            Observation ID
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        date : datetime.datetime
            Observation date
        basepath : str
            Base path for output files
        birdies : list, optional
            List of known birdie periods to filter

        Notes
        -----
        The FFA_search function writes outputs directly to disk
        """
        if not self.run_ffa_search:
            log.info("FFA search is disabled")
            return

        log.info(f"Starting FFA search on RA={ra:.2f}, DEC={dec:.2f}, Date={date}")

        # Call the full FFA_search function from FFA_search.py
        FFA_search(
            dedisp_ts=dedisp_ts,
            obs_id=obs_id,
            date=date,
            ra=ra,
            dec=dec,
            run_ffa_search=self.run_ffa_search,
            create_periodogram_plots=self.create_periodogram_plots,
            create_profile_plots=self.create_profile_plots,
            write_ffa_detections=self.write_ffa_detections,
            write_ffa_candidates=self.write_ffa_candidates,
            write_ffa_stack=self.write_ffa_stack,
            dm_downsample=self.dm_downsample,
            dm_min=self.dm_min,
            dm_max=self.dm_max,
            p_min=2.0,
            min_rfi_sigma=self.min_rfi_sigma,
            min_detection_sigma=self.min_detection_sigma,
            birdie_tolerance=self.birdie_tolerance,
            num_threads=self.num_threads,
            basepath=basepath,
        )



def initialise(config, num_threads=8):
    """
    Initialize FFA processor from configuration.

    Parameters
    ----------
    config : OmegaConf
        Configuration object with 'ffa' section
    num_threads : int
        Number of threads for processing

    Returns
    -------
    FFAProcessor
        Initialized FFA processor
    """
    ffa_config = config.get("ffa", {})

    if isinstance(ffa_config, OmegaConf):
        ffa_config = OmegaConf.to_container(ffa_config)

    # Add num_threads to config
    ffa_config['num_threads'] = num_threads

    log.info("Initializing FFA processor")
    processor = FFAProcessor(**ffa_config)

    return processor


def run(active_pointing, dedisp_ts, ffa_processor, basepath, date, birdies=None):
    """
    Run FFA search on an observation.

    Parameters
    ----------
    active_pointing : Pointing
        Active pointing object with ra, dec, obs_id
    dedisp_ts : DedispersedTimeSeries
        Dedispersed time series
    ffa_processor : FFAProcessor
        Initialized FFA processor
    basepath : str
        Base path for outputs
    date : datetime.datetime
        Observation date
    birdies : list, optional
        Known RFI birdies

    Notes
    -----
    FFA_search writes outputs directly to disk
    """
    log.info(
        f"FFA Search ({active_pointing.ra:.2f} {active_pointing.dec:.2f}) @ {date:%Y-%m-%d}"
    )

    ffa_processor.run_ffa_search_on_dedisp(
        dedisp_ts=dedisp_ts,
        obs_id=active_pointing.obs_id,
        ra=active_pointing.ra,
        dec=active_pointing.dec,
        date=date,
        basepath=basepath,
        birdies=birdies,
    )

    log.info("FFA search complete")
