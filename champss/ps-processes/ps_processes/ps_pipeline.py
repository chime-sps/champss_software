import logging
import os

import numpy as np
from attr import ib as attribute
from attr import s as attrs
from attr.validators import instance_of
from ps_processes import (
    FailedChi2TestError,
    IncompleteMonthlyStackError,
    StackNotInDbError,
)
from ps_processes.processes.ps_search import PowerSpectraSearch
from ps_processes.processes.ps_stack import PowerSpectraStack
from sps_common.interfaces import PowerSpectra
from sps_databases import db_api

log = logging.getLogger(__name__)


@attrs(slots=True)
class PowerSpectraPipeline:
    """
    Pipeline to run ps processes.

    The class is initialised with various configurations.
    The required configurations are 'run_ps_creation', 'run_ps_search', 'run_ps_stack'
    as boolean inputs, with the configurations for PowerSpectraCreation,
    PowerSpectraStack and PowerSpectraSearch passed on via ps_creation_config,
    ps_stack_config, and ps_search_config as dictionaries.

    Parameters
    =======
    run_ps_creation: bool
        Whether to create power spectra from dedispersed time series.

    run_ps_search: bool
        Whether to search power spectra for detection.

    run_ps_stack: bool
        Whether to stack individual power spectra to monthly stack.

    ps_creation_config: dict
        Configuration to the PowerSpectraCreation class. Default = {}

    ps_stack_config: dict
        Configuration to the PowerSpectraCreation class. Default = {}

    ps_search_config: dict
        Configuration to the PowerSpectraCreation class. Default = {}

    search_failed_spectra: bool
        Whether to search the power spectra that failed the quality control test. Default = True

    write_ps_detections: bool
        Whether to write the power spectra detection clusters to a hdf5 file with filename
        <ra>_<dec>_power_spectra_detection_clusters.hdf5

    write_ps_raw_detections: bool
        write_ps_detections writes PowerSpectraDetectionClusters, this writes the raw detections
        <ra>_<dec>_power_spectra_detections.npz

    num_threads: int
        Number of threads to run the parallelised power spectra search process. Default = 8

    update_db: bool
        Whether to update the sps-databases on the processes. Default = True
    """

    run_ps_stack = attribute(validator=instance_of(bool))
    run_ps_search = attribute(validator=instance_of(bool))
    ps_stack_config = attribute(validator=instance_of(dict), default={}, type=dict)
    ps_search_config = attribute(validator=instance_of(dict), default={}, type=dict)
    search_failed_spectra = attribute(validator=instance_of(bool), default=False)
    write_ps_detections = attribute(validator=instance_of(bool), default=True)
    write_ps_raw_detections = attribute(validator=instance_of(bool), default=False)
    num_threads = attribute(validator=instance_of(int), default=8)
    update_db = attribute(validator=instance_of(bool), default=True)
    known_source_threshold = attribute(validator=instance_of(float), default=np.inf)
    # private class attributes to store the initialised class objects to run the
    # various power spectra process
    _ps_search = attribute(init=False)
    _ps_stack = attribute(init=False)
    stack_nbit = attribute(init=False)
    spectra_nbit = attribute(init=False)

    def __attrs_post_init__(self):
        self.stack_nbit = 16
        self.spectra_nbit = 32
        if "stack_nbit" in self.ps_stack_config:
            self.stack_nbit = self.ps_stack_config["stack_nbit"]
        if "spectra_nbit" in self.ps_stack_config:
            self.stack_nbit = self.ps_stack_config["spectra_nbit"]
        if self.run_ps_stack:
            self.ps_stack_config.pop("mode", None)
            self.ps_stack_config.pop("update_db", None)
            self.ps_stack_config.pop("spectra_nbit", None)
            self._ps_stack = PowerSpectraStack(
                mode="month",
                update_db=self.update_db,
                spectra_nbit=self.spectra_nbit,
                **self.ps_stack_config,
            )
        if self.run_ps_search:
            self.ps_search_config.pop("num_threads", None)
            self._ps_search = PowerSpectraSearch(
                **self.ps_search_config,
                num_threads=self.num_threads,
                update_db=self.update_db,
                known_source_threshold=self.known_source_threshold,
            )

    def process(self, power_spectra=None, basepath="./", subdir="", prefix=""):
        """
        Load, stack and search an observation.

        Process the observation, given the observation id, base path and sub directory
        to the data products. The data products are saved in
        '<basepath>/<subdir>/<ra>_<dec>_power_spectra.hdf5',
        '<basepath>/<subdir>/<ra>_<dec>_power_spectra_detection_clusters.hdf5' and
        '<basepath>/stack/<ra>_<dec>_power_spectra_stack.hdf5'.

        Parameters
        ----------
        dedisp_ts: DedispersedTimeSeries
            The DedispersedTimeSeries class used to store the dedispersed time series.

        basepath: str
            The base path to the data product. Default : current working directory

        subdir: str
            The subdirectory to the data product. Usually are in YYYY/MM/DD/<ra>_<dec>/ format.
            Default = ''

        prefix: str
            The prefix to save/load the data product. Default = ''
        """
        filepath = f"{basepath}/{subdir}"
        power_spectra_detection_clusters = None
        if not power_spectra:
            power_spectra = self.load_power_spectra(filepath, prefix)
        if self.run_ps_stack:
            if self.run_ps_search and self._ps_stack.readout_stack:
                power_spectra_detection_clusters = self.power_spectra_search(
                    power_spectra, filepath=filepath, prefix=prefix
                )
            power_spectra = self.power_spectra_stack(power_spectra)
        if self.run_ps_search and not power_spectra_detection_clusters:
            power_spectra_detection_clusters = self.power_spectra_search(
                power_spectra, filepath=filepath, prefix=prefix
            )
        return power_spectra_detection_clusters

    def power_spectra_stack(self, power_spectra):
        try:
            power_spectra_to_search = self._ps_stack.stack(power_spectra)
        except FailedChi2TestError as chi2_e:
            log.warning(chi2_e)
            self._ps_stack.unlock_stack()
            if self.search_failed_spectra:
                return power_spectra
            else:
                return None
        except AssertionError as assert_e:
            log.error(assert_e)
            log.error(
                "Consider checking the stack for"
                f" {power_spectra.ra} {power_spectra.dec}"
            )
            self._ps_stack.unlock_stack()
            if self.search_failed_spectra:
                return power_spectra
            else:
                return None
        except StackNotInDbError as stacknotindb_e:
            log.error(stacknotindb_e)
            log.error(
                "Stack on disk not present in current database."
                " It might be used by other databases or the database update failed."
            )
            self._ps_stack.unlock_stack()
            if self.search_failed_spectra:
                return power_spectra
            else:
                return None

        return power_spectra_to_search

    def power_spectra_search(
        self,
        power_spectra_to_search,
        injection_path,
        injection_idx,
        only_injections,
        scale_injections=False,
        filepath="./",
        prefix="",
    ):
        (
            power_spectra_detection_clusters,
            power_spectra_detections,
        ) = self._ps_search.search(
            power_spectra_to_search,
            injection_path,
            injection_idx,
            only_injections,
            scale_injections,
        )
        if self.write_ps_detections and power_spectra_detection_clusters is not None:
            filename = f"{prefix}_power_spectra_detection_clusters.hdf5"
            power_spectra_detection_clusters.write(f"{filepath}/{filename}")
        if self.write_ps_raw_detections and power_spectra_detections is not None:
            filename = f"{prefix}_power_spectra_detections.npz"
            log.info(f"Saving power spectra detections to {filepath}/{filename}")
            np.savez(f"{filepath}/{filename}", detections=power_spectra_detections)
        return power_spectra_detection_clusters

    def load_power_spectra(self, filepath="./", prefix=""):
        filename = f"{prefix}_power_spectra.hdf5"
        power_spectra = PowerSpectra.read(
            f"{filepath}/{filename}", nbit=self.spectra_nbit
        )
        return power_spectra


@attrs(slots=True)
class StackSearchPipeline:
    """
    Pipeline to run stacking and searching of monthly stacks.

    This class performs stacking of monthly stack into cumulative stack and search
    the cumulative stack for pulsars. The monthly stack can also be searched before stacking.

    Parameters
    =======
    run_ps_stack: bool
        Whether to run the power spectra stack, otherwise only load the cumulative
        stack to search. Default = True

    run_ps_search: bool
        Whether to run the power spectra search. If run_ps_stack is False,
        run ps_search will always be True. Default = True

    run_ps_search_monthly: bool
        Whether to search the monthly stack before stacking. Default = True

    ps_stack_config: dict
        Configuration for the power spectra stack process.
        Default: uses the default configuration.

    ps_search_config: dict
        Configuration for the power spectra searching process.
        Default: uses the default configuration.

    min_num_days: int
        The minimum number of days of the span of data in the monthly stack before it is
        stacked to the cumulative stack. Default = 30

    write_ps_detections: bool
        Whether to write the power spectra detection clusters to file. Default = True

    write_ps_raw_detections: bool
        write_ps_detections writes PowerSpectraDetectionClusters, this writes the
        raw detections

    num_threads: int
        Number of threads to run the parallelised power spectra search process. Default = 8

    update_db: bool
        Whether to update the database of the stacking process. Default = True
    """

    run_ps_stack = attribute(validator=instance_of(bool), default=True)
    run_ps_search = attribute(validator=instance_of(bool), default=True)
    run_ps_search_monthly = attribute(validator=instance_of(bool), default=True)
    ps_stack_config = attribute(validator=instance_of(dict), default={}, type=dict)
    ps_search_config = attribute(validator=instance_of(dict), default={}, type=dict)
    min_num_days = attribute(validator=instance_of(int), default=30)
    write_ps_detections = attribute(validator=instance_of(bool), default=True)
    write_ps_raw_detections = attribute(validator=instance_of(bool), default=False)
    num_threads = attribute(validator=instance_of(int), default=8)
    update_db = attribute(validator=instance_of(bool), default=True)
    known_source_threshold = attribute(validator=instance_of(float), default=np.inf)
    # private class attributes to store the initialised class objects to run the
    # various power spectra process
    _ps_search = attribute(init=False)
    _ps_stack = attribute(init=False)
    stack_nbit = attribute(init=False)
    spectra_nbit = attribute(init=False)

    def __attrs_post_init__(self):
        self.stack_nbit = 16
        self.spectra_nbit = 32
        if self.run_ps_stack:
            if "stack_nbit" in self.ps_stack_config:
                self.stack_nbit = self.ps_stack_config["stack_nbit"]
            if "spectra_nbit" in self.ps_stack_config:
                self.spectra_nbit = self.ps_stack_config["spectra_nbit"]
            self.ps_stack_config.pop("mode", None)
            self.ps_stack_config.pop("update_db", None)
            self._ps_stack = PowerSpectraStack(
                mode="cumul", update_db=self.update_db, **self.ps_stack_config
            )
        # Previously when no stacking was performed the cumulative stack searched
        # Now it needs to be chosen via the command line to search
        # the monthly or the cumulative stack
        if self.run_ps_search or self.run_ps_search_monthly:
            self.ps_search_config.pop("num_threads", None)
            self._ps_search = PowerSpectraSearch(
                **self.ps_search_config,
                num_threads=self.num_threads,
                known_source_threshold=self.known_source_threshold,
            )

    def load_and_search_monthly(
        self,
        pointing_id,
        injection_path=None,
        injection_idx=None,
        only_store_injections=False,
        cutoff_frequency=100.0,
        scale_injections=False,
        file=None,
    ):
        """
        Process the monthly stack.

        Load and search the monthly stack.

        Parameters
        =======
        pointing_id: str
            The pointing_id of the pointing to stack the monthly power spectra into the
            cumulative spectra.
        Returns
        =======
        monthly_power_spectra_detection_clusters: PowerSpectraDetectionClusters or None
            The detections from the power spectra search in the PowerSpectraDetectionClusters or
            None if the power spectra search is not performed.

        monthly_power_spectra: sps_common.interfaces.PowerSpectra
            The monthly power spectra
        """
        if pointing_id:
            ps_stack_db = db_api.get_ps_stack(pointing_id)
        if self.run_ps_stack or self.run_ps_search_monthly:
            if pointing_id:
                if (
                    sorted(ps_stack_db.datetimes_month)[-1]
                    - sorted(ps_stack_db.datetimes_month)[0]
                ).days < self.min_num_days:
                    log.error(
                        f"There are less than {self.min_num_days} days of data in the"
                        " current monthly stack. Skipping the process"
                    )
                    return None, None
                if ps_stack_db.datapath_month:
                    try:
                        if not os.path.isfile(ps_stack_db.datapath_month):
                            raise FileNotFoundError(
                                "Cannot locate the monthly power spectra stack"
                                f" '{ps_stack_db.datapath_month}'"
                            )
                        log.info(
                            "loading the monthly power spectra stack"
                            f" {ps_stack_db.datapath_month}"
                        )
                        monthly_power_spectra = PowerSpectra.read(
                            ps_stack_db.datapath_month, nbit=self.spectra_nbit
                        )
                    except (OSError, FileNotFoundError) as e:
                        log.error(
                            f"{e}: Monthly stack file for pointing id"
                            f" {ps_stack_db.pointing_id} does not exist. Exiting"
                        )
                        if self.update_db:
                            payload = {
                                "datapath_month": "",
                                "datetimes_month": [],
                                "num_days_month": 0,
                            }
                            db_api.update_ps_stack(ps_stack_db.pointing_id, payload)
                        return None, None
                else:
                    log.error(
                        "Monthly stack file for pointing id"
                        f" {ps_stack_db.pointing_id} does not exist. Exiting"
                    )
                    return
            elif file:
                monthly_power_spectra = PowerSpectra.read(file, nbit=self.spectra_nbit)
            else:
                log.error(
                    "Need either pointing id or file but got {pointing_id} and {file}"
                )
                return None, None
            if self.run_ps_search_monthly:
                (
                    monthly_power_spectra_detection_clusters,
                    monthly_power_spectra_detections,
                ) = self._ps_search.search(
                    monthly_power_spectra,
                    injection_path,
                    injection_idx,
                    only_store_injections,
                    cutoff_frequency,
                    scale_injections,
                )
            else:
                monthly_power_spectra_detection_clusters = None
        else:
            monthly_power_spectra = None
            monthly_power_spectra_detection_clusters = None

        return monthly_power_spectra_detection_clusters, monthly_power_spectra

    def stack_and_search(
        self,
        pointing_id,
        monthly_power_spectra=None,
        injection_path=None,
        injection_idx=None,
        only_store_injections=False,
        cutoff_frequency=100.0,
        scale_injections=False,
    ):
        """
        Process the cumulative stack.

        Stack the monthly stack into the cumulative stack if satsfies
        quality tests. Search for pulsars in the cumulative stack.

        Parameters
        =======
        pointing_id: str
            The pointing_id of the pointing to stack the monthly power spectra into the
            cumulative spectra.
        monthly_power_spectra: PowerSpectra
            The monthly Spectra that are loaded.
            Default: None
        Returns
        =======
        power_spectra_detection_clusters: PowerSpectraDetectionClusters or None
            The detections from the power spectra search in the PowerSpectraDetectionClusters or
            None if the power spectra search is not performed.

        stacked_power_spectra: sps_common.interfaces.PowerSpectra
            The stacked power spectra
        """
        ps_stack_db = db_api.get_ps_stack(pointing_id)
        stacked_power_spectra = None
        try:
            if self.run_ps_stack:
                if monthly_power_spectra is None:
                    log.info(
                        "Monthly power spectra have not been loaded already. Will load"
                        " now."
                    )
                    (
                        monthly_power_spectra_detection_clusters,
                        monthly_power_spectra,
                    ) = self.load_and_search_monthly(
                        pointing_id,
                        injection_path,
                        injection_idx,
                        only_store_injections,
                        cutoff_frequency,
                        scale_injections,
                    )
                stacked_power_spectra = self._ps_stack.stack(monthly_power_spectra)
                (
                    quality_result_cumul,
                    quality_metrics_cumul,
                ) = self._ps_stack.quality_control(stacked_power_spectra)
                payload_qc = {
                    "qc_test_cumul": quality_metrics_cumul,
                    "cumul_stack_quality_label": quality_result_cumul,
                }
                db_api.append_ps_stack(pointing_id, payload_qc)
                log.info(
                    "Written quality metrics of cumulative stack of pointing id"
                    f" {ps_stack_db.pointing_id} to database"
                )

        except FailedChi2TestError as chi2_e:
            log.warning(chi2_e)
            return None, None
        except AssertionError as assert_e:
            log.error(assert_e)
            log.error(
                "Consider checking the monthly and cumulative stacks for"
                f" {monthly_power_spectra.ra:.2f} "
                f"{monthly_power_spectra.dec:.2f}"
            )
            return None, stacked_power_spectra
        except IncompleteMonthlyStackError as incomplete_e:
            log.error(incomplete_e)
            return None, None
        if stacked_power_spectra is None and self.run_ps_search:
            log.info("Loading the cumulative power spectra stack to search")
            try:
                stacked_power_spectra = PowerSpectra.read(
                    ps_stack_db.datapath_cumul, nbit=self.spectra_nbit
                )
            except (OSError, FileNotFoundError) as e:
                log.error(
                    f"{e}: The cumulative power spectra stack at"
                    f" {ps_stack_db.datapath_cumul} does not exist, will not continue"
                    " with the searching process"
                )
                return None, None
        if self.run_ps_search:
            (
                power_spectra_detection_clusters,
                power_spectra_detections,
            ) = self._ps_search.search(
                stacked_power_spectra,
                injection_path,
                injection_idx,
                only_store_injections,
                cutoff_frequency,
                scale_injections,
            )
            if self.write_ps_detections:
                stack_detection_file = (
                    ps_stack_db.datapath_cumul.split(".hdf5")[0]
                    + "_detection_clusters.hdf5"
                )
                power_spectra_detection_clusters.write(stack_detection_file)
            if self.write_ps_raw_detections:
                stack_raw_detection_file = (
                    ps_stack_db.datapath_cumul.split(".hdf5")[0] + "_detections.npz"
                )
                log.info(
                    f"Saving power spectra detections to {stack_raw_detection_file}"
                )
                np.savez(stack_raw_detection_file, detections=power_spectra_detections)
        else:
            power_spectra_detection_clusters = None
        return power_spectra_detection_clusters, stacked_power_spectra
