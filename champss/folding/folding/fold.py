import logging
import os
import subprocess

import numpy as np

from beamformer.strategist.strategist import PointingStrategist
from folding.utilities.utils import candidate_name
from folding.utilities.alias import get_alias_factors, fold_at_alias
from sps_pipeline import beamform

log = logging.getLogger(__name__)


class Fold:
    """
    Class for folding pulsars and candidates.

    This class encapsulates all the functionality needed for beamforming and
    folding, used by both fold_candidate and fold_pulsar scripts.
    """

    def __init__(self, ra, dec, f0, dm, date, ephem_path, foldpath, datpath, config,
                 name=None, filterbank_to_ram=True, exact_coords=False, coord_path=None):
        """
        Initialize Fold object.

        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        f0 : float
            Spin frequency in Hz
        dm : float
            Dispersion measure in pc/cm^3
        date : datetime
            Observation date
        ephem_path : str
            Path to parfile
        foldpath : str
            Base path for fold outputs
        datpath : str
            Path to raw data
        config : dict
            Pipeline configuration, used for beamforming config
        name : str, optional
            Source name (default: creates J-name from coordinates)
        filterbank_to_ram : bool, optional
            Whether to use RAM for filterbank (default: True)
        exact_coords : bool, optional
            Use exact coordinates, otherwise spaps to pointing grid (default: False)
        coord_path : str, optional
            Custom coordinate path for outputs (auto-generated if not provided)
        """
        self.ra = ra
        self.dec = dec
        self.f0 = f0
        self.dm = dm
        self.date = date
        self.ephem_path = ephem_path
        self.foldpath = foldpath
        self.datpath = datpath
        self.config = config
        self.config['beamform']['update_db'] = False
        self.config['beamform']['flatten_bandpass'] = False
        self.name = name if name else candidate_name(ra, dec)
        self.filterbank_to_ram = filterbank_to_ram
        self.exact_coords = exact_coords

        # Will be set during execution
        self.ap = None
        self.fil = None
        self.archive_fname = None
        self.coord_path = coord_path
        self.num_threads = None
        self.nchan = None
        self.turns = None
        self.intflag = None

    def setup_paths(self, dir_suffix="candidates", archive_basename=None):
        """
        Setup output directories and file paths.

        Parameters
        ----------
        dir_suffix : str, optional
            Subdirectory suffix under foldpath (default: "candidates")
        archive_basename : str, optional
            Base name for archive file without date. If not provided, uses
            "cand_{f0}_{dm}" format. Date will be appended as "_{year}-{month:02}-{day:02}"
        """
        year = self.date.year
        month = self.date.month
        day = self.date.day

        directory_path = f"{self.foldpath}/{dir_suffix}"

        if self.coord_path is None:
            self.coord_path = f"{directory_path}/{self.ra:.02f}_{self.dec:.02f}"

        if not os.path.exists(self.coord_path):
            os.makedirs(self.coord_path)
            log.info(f"Created directory: {self.coord_path}")
        else:
            log.info(f"Directory '{self.coord_path}' already exists.")

        # Set archive filename
        if archive_basename is None:
            # Default: candidate format with f0 and dm
            archive_basename = f"cand_{self.f0:.02f}_{self.dm:.02f}"
        self.archive_fname = f"{self.coord_path}/{archive_basename}_{year}-{month:02}-{day:02}"

        # Setup filterbank path
        fname = f"/{self.ra:.02f}_{self.dec:.02f}_{self.f0:.02f}_{self.dm:.02f}_{year}-{month:02}-{day:02}.fil"
        if self.filterbank_to_ram:
            log.info("Using RAM for filterbank file")
            fildir = "/dev/shm"
        else:
            log.info("Using disk for filterbank file")
            fildir = self.coord_path
        self.fil = fildir + fname

    def beamform(self):
        """
        This function sets up the pointing, beamforms, and writes as a filterbank.

        Returns
        -------
        success : bool
            True if beamforming succeeded, False otherwise
        """
        if self.exact_coords:
            log.info(f"Using exact coordinates (RA={self.ra:.6f}, Dec={self.dec:.6f}) without grid snapping")
        else:
            log.info(f"Using coordinates from pointing map (RA={self.ra:.2f}, Dec={self.dec:.2f})")

        pst = PointingStrategist(create_db=False, split_long_pointing=True)
        self.ap = pst.get_single_pointing(self.ra, self.dec, self.date, use_grid=not self.exact_coords)

        # If multiple sub-pointings (at high dec), always write to disk (too large for RAM)
        if len(self.ap) > 1:
            log.info(f"Multiple sub-pointings ({len(self.ap)}), writing filterbank to disk")
            self.filterbank_to_ram = False
            self.config.beamform.beam_to_normalise = None

            # Update filterbank path to disk
            year = self.date.year
            month = self.date.month
            day = self.date.day
            fname = f"/{self.ra:.02f}_{self.dec:.02f}_{self.f0:.02f}_{self.dm:.02f}_{year}-{month:02}-{day:02}.fil"
            self.fil = self.coord_path + fname

        # Calculate channel requirements based on DM
        nchan_tier = int(np.ceil(np.log2(self.dm // 212.5 + 1)))
        self.nchan = 1024 * (2**nchan_tier)
        self.num_threads = 4 * self.nchan // 1024

        if not os.path.isfile(self.fil):
            log.info(f"Beamforming {len(self.ap)} sub-pointing(s)...")
            fdmt = True
            beamformer = beamform.initialise(self.config, rfi_beamform=True,
                                             basepath=self.foldpath, datpath=self.datpath)

            # Loop through all active pointings and append them into one filterbank
            for i, active_pointing in enumerate(self.ap):
                # Adjust nchan if needed
                if self.nchan < active_pointing.nchan:
                    log.info(
                        f"only need nchan = {self.nchan} for dm = {self.dm}, beamforming with"
                        f" {self.nchan} channels"
                    )
                    active_pointing.nchan = self.nchan

                log.info(f"Beamforming sub-pointing {i+1}/{len(self.ap)} with {self.num_threads} threads")

                skybeam, spectra_shared = beamform.run(
                    active_pointing, beamformer, fdmt, self.num_threads, self.foldpath
                )

                if skybeam is None:
                    log.warning(
                        f"Insufficient unmasked data to form skybeam for sub-pointing {i+1}, skipping"
                    )
                    spectra_shared.close()
                    spectra_shared.unlink()
                    continue

                # Write first sub-pointing to create the file, append subsequent ones
                if i == 0:
                    log.info(f"Writing sub-pointing {i+1} to {self.fil}")
                    skybeam.write(self.fil)
                else:
                    log.info(f"Appending sub-pointing {i+1} to {self.fil}")
                    skybeam.append(self.fil)

                spectra_shared.close()
                spectra_shared.unlink()
                del skybeam

        return os.path.isfile(self.fil)

    def fold(self):
        """
        Fold filterbank with dspsr.

        Returns
        -------
        success : bool
            True if folding succeeded and archive was created
        """
        # Set number of turns, roughly equalling 10s
        # Currently hardcoded, may want to allow more dspsr options
        self.turns = int(np.ceil(10 * self.f0))
        if self.turns <= 2:
            self.intflag = "-turns"
        else:
            self.intflag = "-L"
            self.turns = 10

        log.info("Folding with dspsr...")
        result = subprocess.run(
            [
                "dspsr",
                "-t",
                f"{self.num_threads}",
                f"{self.intflag}",
                f"{self.turns}",
                "-A",
                "-k",
                "chime",
                "-E",
                f"{self.ephem_path}",
                "-O",
                f"{self.archive_fname}",
                f"{self.fil}",
            ],
            capture_output=True,
            text=True,
        )

        archive_fname_full = self.archive_fname + ".ar"

        if result.returncode != 0:
            log.error(f"dspsr failed with return code {result.returncode}")
            log.error(f"stderr: {result.stderr}")
            return False

        if not os.path.isfile(archive_fname_full):
            log.error(f"Archive file {archive_fname_full} was not created")
            return False

        # Create frequency and time scrunched version
        log.info("Creating frequency and time scrunched version...")
        create_FT = f"pam -T -F {archive_fname_full} -e FT"
        subprocess.run(create_FT, shell=True, capture_output=True, text=True)

        # Update archive_fname to include .ar extension
        self.archive_fname = archive_fname_full

        return True

    def fold_aliases(self):
        """
        Fold at multiple frequency aliases.

        Returns
        -------
        alias_results : dict
            Dictionary mapping alias labels to archive filenames
        """
        log.info("Folding at frequency aliases...")
        alias_dir = f"{self.coord_path}/aliases"
        if not os.path.exists(alias_dir):
            os.makedirs(alias_dir)

        alias_factors = get_alias_factors()
        alias_results = {}

        for factor, label in alias_factors:
            alias_archive = fold_at_alias(
                self.fil, self.ephem_path, alias_dir, factor, label, self.num_threads
            )
            if alias_archive:
                alias_results[label] = alias_archive

        log.info(f"Completed alias folding: {len(alias_results)} of {len(alias_factors)} successful")

        return alias_results

    def plot(self, sigma=None, foldpath_plots=None):
        """
        Create plots for the folded candidate, using plot_candidate.py

        Parameters
        ----------
        sigma : float, optional
            Pipeline sigma of candidate
        foldpath_plots : str, optional
            Path for saving plots (default: foldpath + "/plots/folded_candidate_plots/")

        Returns
        -------
        SNprof : float
            S/N of profile
        SN_arr : float
            S/N array
        plot_fname : str
            Path to saved plot file
        """
        from folding.plot_candidate import plot_candidate_archive

        if foldpath_plots is None:
            foldpath_plots = self.foldpath + "/plots/folded_candidate_plots/"

        cand_info = {
            'sigma': sigma,
            'ap': self.ap,
        }

        SNprof, SN_arr, plot_fname = plot_candidate_archive(
            self.archive_fname,
            self.coord_path,
            cand_info=cand_info,
            foldpath=foldpath_plots,
        )

        log.info(f"SN of folded profile: {SN_arr}")

        return SNprof, SN_arr, plot_fname

    def cleanup(self):
        """
        Remove temporary filterbank file.
        """
        log.info(f"Cleaning up, deleting {self.fil}")
        if os.path.isfile(self.fil):
            os.remove(self.fil)
