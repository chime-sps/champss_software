#!/usr/bin/env python

import datetime
import itertools
import logging
import os
import time
from pathlib import PurePath

import filelock
import h5py
import numpy as np
import pytz
from attr import ib as attribute
from attr import s as attrs
from attr.validators import instance_of
from filelock import FileLock, Timeout
from ps_processes import (
    FailedChi2TestError,
    IncompleteMonthlyStackError,
    NoPsStackDbError,
    StackNotInDbError,
)
from ps_processes.utilities.qc import (
    compare_ps_to_chisqr_kstest,
    validate_ps_chisqr_outlier_bins,
)
from ps_processes.utilities.utilities import check_in_range, grab_metric_history
from scipy.stats import median_abs_deviation
from sps_common.interfaces import PowerSpectra
from sps_databases import db_api

log = logging.getLogger(__name__)


@attrs(slots=True)
class PowerSpectraStack:
    """
    Class to stack power spectra.

    Parameters
    =======
    mode: str
        The mode to stack the power spectra. Either 'month' for stacking daily stack into monthly stack
        or 'cumul' to stack monthly stack to cumulative stack. Default = 'month'

    basepath: str
        The basepath to write the power spectra stack to. Default = current file path

    qc: bool
        Whether to run a quality control to check if the stack is clean for stacking. Default = True

    qc_config: dict
        Dictionary of the configurations to run on the power spectra quality control process. Default = {}

    spectra_nbit: int
        The number of bits to read the power spectra stack with. Default = 32

    stack_nbit: int
        The number of bits to write the power spectra stack into. Default = 16

    delete_monthly_stack: bool
        Whether to delete the existing monthly power spectra stack after stacking into cumulative power spectra stack.
        Default = True

    infile_overwrite: bool
        Whether to read the monthly stack per DM and summing it and overwrite the file instead of loading the whole
        stack at once, summing the stacks together and write the new stack to file. Default = False

    readout_stack: bool
        Whether to read the stack on disk by chunks and adding them to the current stack to form a new stack, instead
        of loading the full stack before stacking process. Default = True

    update_db: bool
        Whether to update the power spectra stack database post stacking. Default = True

    max_lock_age: float
        Maximum age of the stack lock file before it gets ignored. Default = 1800
    """

    mode = attribute(default="month")
    basepath = attribute(default="./", validator=instance_of(str))
    qc = attribute(default=True, validator=instance_of(bool))
    qc_config = attribute(default={}, validator=instance_of(dict))
    spectra_nbit = attribute(default=32, validator=instance_of(int))
    stack_nbit = attribute(default=16, validator=instance_of(int))
    delete_monthly_stack = attribute(default=True, validator=instance_of(bool))
    infile_overwrite = attribute(default=False, validator=instance_of(bool))
    readout_stack = attribute(default=True, validator=instance_of(bool))
    update_db = attribute(default=True, validator=instance_of(bool))
    max_lock_age = attribute(default=1800.0, validator=instance_of(float))
    lock = attribute(init=False, validator=instance_of(filelock._unix.UnixFileLock))
    write_when_file_not_found = attribute(default=True, validator=instance_of(bool))

    @mode.validator
    def _validate_mode(self, attribute, value):
        assert value in [
            "month",
            "cumul",
        ], f"The {attribute.name} must be either 'month' or 'cumul'"

    @spectra_nbit.validator
    @stack_nbit.validator
    def _validate_nbit(self, attribute, value):
        assert value in [
            16,
            32,
            64,
            128,
        ], f"The {attribute.name} value must be 16, 32, 64 or 128"

    def stack(self, pspec):
        """
        Process to stack the power spectra to form a stacked power spectrum across
        multiple days. The power spectra has information for the code to look for the
        relevant stack from the database to load, stack and save the data. The process
        will return the daily power spectra or the cumulative power spectra.

        Parameters
        ==========
        pspec: PowerSpectra
            PowerSpectra object with the properties of the power spectra to stack

        Returns
        =======
        pspec or pspec_stack: PowerSpectra
            PowerSpectra object, which will be the daily power spectra for 'month'
            mode and the cumulative power spectra for 'cumul'
        """

        # qc test to know if we want to load the power spectra stack at all.
        if self.qc:
            log.info("Running Quality Control on the power spectra stack")
            quality_result, quality_metrics = self.quality_control(pspec)
            if self.mode == "month" and self.update_db:
                self.update_obs_database(pspec, quality_result, quality_metrics)
            elif self.mode == "cumul" and self.update_db:
                payload_qc = {
                    "datetimes_per_month": pspec.datetimes,
                    "qc_test_per_month": quality_metrics,
                    "qc_label_per_month": quality_result,
                }
                db_api.append_ps_stack(
                    db_api.get_observation(pspec.obs_id[0]).pointing_id, payload_qc
                )
                log.info(
                    "Written quality metrics of monthly stack of of"
                    f" {pspec.ra:.2f} {pspec.dec:.2f} to database"
                )

            if not quality_result:
                # raise a custom error to catch for ps search purposes.
                raise FailedChi2TestError(
                    f"Power spectra stack from observation {pspec.obs_id} failed"
                    " quality test.Not stacking the current spectra to the power"
                    " spectra stack."
                )
        else:
            if self.mode == "month" and self.update_db:
                self.update_obs_database(pspec, True)

        # get the power spectra stack db
        ps_stack_db = self.get_ps_stack_db(pspec)

        # If no databae entry exists try to lock the file
        stack_file_open = None
        if not ps_stack_db:
            stack_file_path = self.get_stack_file_path(pspec)
            self.lock_stack(stack_file_path)
            # If the stack is already locked by another process,
            # that process will create a new database entry which is then read
            ps_stack_db = self.get_ps_stack_db(pspec)

        # check if there is existing stack entry
        if not ps_stack_db:
            ps_stack_db = self.get_ps_stack_db(pspec)
            if self.mode == "cumul":
                raise NoPsStackDbError(
                    "The power spectra stack database for"
                    f" {pspec.ra:.2f} {pspec.dec:.2f} should exist if we are stacking"
                    " monthly stack to cumulative stack"
                )
            log.info(
                "No database entry for power spectrum stack of"
                f" {pspec.ra:.2f} {pspec.dec:.2f}."
            )
            stack_file_path = self.get_stack_file_path(pspec)
            log.info(f"Writing new power spectra stack to {stack_file_path}.")
            if os.path.isfile(stack_file_path):
                existing_file_ok = PowerSpectra.check_file_keys(stack_file_path)
                if existing_file_ok:
                    raise StackNotInDbError(
                        f"Stack file {stack_file_path} aready exists on disk but not in"
                        " database. File appears to be properly saved."
                    )
                else:
                    log.error(
                        f"Stack file {stack_file_path} aready exists on disk but appears to be broken. Will delete old file."
                    )
                    os.remove(stack_file_path)
            pspec.write(stack_file_path, nbit=self.stack_nbit)
            if self.update_db:
                log.info(
                    f"Creating new database entry for {pspec.ra:.2f} {pspec.dec:.2f}."
                )
                self.create_stack_database(pspec, stack_file_path)
            self.unlock_stack()
            return pspec
        else:
            # Try loading the existing monthly/cumulative stack
            if self.mode == "month":
                stack_file_path = ps_stack_db.datapath_month
            elif self.mode == "cumul":
                # check if the loaded monthly stack is consistent with the database
                if (pspec.num_days != ps_stack_db.num_days_month) or (
                    pspec.datetimes != ps_stack_db.datetimes_month
                ):
                    if self.delete_monthly_stack:
                        # deletes the existing monthly power spectra stack
                        self.delete_stack(
                            ps_stack_db.pointing_id, ps_stack_db.datapath_month
                        )
                    raise IncompleteMonthlyStackError(
                        "The monthly power spectra stack"
                        f" {pspec.ra:.2f} {pspec.dec:.2f} is likely incomplete as the"
                        " information in the stack is inconsistent with the stack"
                        " database."
                    )
                stack_file_path = ps_stack_db.datapath_cumul
            if os.path.isfile(stack_file_path):
                self.lock_stack(stack_file_path)
                if self.infile_overwrite and self.mode == "month":
                    log.info(
                        "Stacking the daily power spectra into power spectra stack via"
                        " infile read and overwrite"
                    )
                    self.stack_power_spectra_infile(pspec, stack_file_path)
                    self.unlock_stack()
                    return pspec
                else:
                    if self.readout_stack:
                        log.info(
                            f"Stacking {self.mode} power spectra stack from"
                            f" {stack_file_path} into current stack via readout method."
                        )
                        pspec_stack = self.stack_power_spectra_readout(
                            pspec, stack_file_path
                        )
                    else:
                        log.info(f"Reading power spectra stack from {stack_file_path}.")
                        pspec_stack = PowerSpectra.read(
                            stack_file_path, nbit=self.spectra_nbit
                        )
                        log.info(
                            "Stacking the current power spectra stack to the"
                            f" {self.mode} power spectra stack."
                        )
                        pspec_stack = self.stack_power_spectra(
                            pspec,
                            pspec_stack,
                        )
                    log.info(f"Writing new power spectra stack to {stack_file_path}.")
                    self.replace_spectra(pspec_stack, stack_file_path)
                    if self.update_db:
                        self.update_stack_database(
                            pspec_stack, stack_file_path, ps_stack_db.pointing_id
                        )
                    if self.mode == "cumul" and self.delete_monthly_stack:
                        # deletes the existing monthly power spectra stack
                        self.delete_stack(
                            ps_stack_db.pointing_id, ps_stack_db.datapath_month
                        )
                    # returns the daily stack to search in mode month and not in readout stack mode, otherwise return
                    # the full stack.
                    self.unlock_stack()
                    if self.mode == "month" and not self.readout_stack:
                        return pspec
                    else:
                        return pspec_stack
            else:
                if self.write_when_file_not_found:
                    log.info(
                        "The power spectra stack for"
                        f" {pspec.ra:.2f} {pspec.dec:.2f} does not exist, creating a"
                        " new power spectra stack"
                    )
                    stack_file_path = self.get_stack_file_path(pspec)
                    self.lock_stack(stack_file_path)
                    pspec.write(stack_file_path, nbit=self.stack_nbit)
                    if self.update_db:
                        self.update_stack_database(
                            pspec, stack_file_path, ps_stack_db.pointing_id
                        )
                    if self.mode == "cumul" and self.delete_monthly_stack:
                        # deletes the existing monthly power spectra stack
                        self.delete_stack(
                            ps_stack_db.pointing_id, ps_stack_db.datapath_month
                        )
                    self.unlock_stack()
                else:
                    raise FileNotFoundError(
                        f"Cannot locate the {self.mode} power spectra stack"
                        f" '{stack_file_path}'. Will not write new file due to"
                        "PowerSpectraStack.write_when_file_not_found==False."
                    )
            return pspec

    def stack_power_spectra_infile(self, pspec, stack_file_path):
        """
        The script to stack the daily power spectra into monthly power spectra using the
        native h5py functions that allows us to read and manipulate the data in chunks.
        This will significantly reduce the memory overhead as only a single power
        spectrum is loaded to be summed at one time. The operation is also much faster
        than loading the full monthly stack to sum and write.

        HOWEVER, this method possess a risk in which, if it is interrupted, the monthly power spectra will be summed
        incompletely, which will render the whole stack useless.

        As a way to double check the updated monthly stack compared to what is recorded in the stack database, the
        process will first update the number of days in the stack, then adding the two stacks together, and lastly
        update the other properties of the power spectra stack, including datetimes. The number of days and list of
        datetimes will then be cross-checked with the stack database to ensure consistency. With this sequence, if the
        process is interrupted during the summing process, the number of days and the length of the datetimes list will
        mismatch and signaling that the process has failed.

        As this is a more risky method of stacking power spectra together, it is only restricted to stacking the daily
        spectra into the monthly spectra, and the traditional method is used to stack the monthly spectra into
        cumulative spectra.

        Parameters
        ----------
        pspec: PowerSpectra
            The PowerSpectra object containing the daily power spectra stack to be stacked into the monthly spectra
            stack.

        stack_file_path: str
            The path to the monthly power spectra stack to load and stack.
        """
        with h5py.File(stack_file_path, "r+") as h5f:
            # Make some check to ensure stacks are compatible with each other
            log.info("Checking for consistency between stacks")
            ps_shape = (len(pspec.power_spectra), len(pspec.power_spectra[0]))
            assert ps_shape == h5f["power spectra"].shape, (
                "The two power spectra stack do not have the same size."
            )
            stack_freq_labels = np.ones(h5f["frequency labels"].shape)
            h5f["frequency labels"].read_direct(stack_freq_labels)
            np.testing.assert_array_equal(
                pspec.freq_labels,
                stack_freq_labels,
                "The two power spectra stack do not have the same frequency labels.",
            )
            datetimes_stack = np.ones(h5f["datetimes"].shape)
            h5f["datetimes"].read_direct(datetimes_stack)
            for dt, dts in itertools.product(pspec.datetimes, datetimes_stack):
                assert np.abs(dt.timestamp() - dts) > 60, (
                    "The two power spectra stack came from the same day."
                )
            log.info("Stacking the power spectra in chunks")
            h5f.attrs["number of days"] += pspec.num_days
            new_num_days = h5f.attrs["number of days"]
            for i in range(len(pspec.dms)):
                h5f["power spectra"][(i,)] += pspec.power_spectra[i]
            log.info("Updating the information of the power spectra stack")
            bad_freq_arrays = [
                key for key in h5f.keys() if "bad frequency indices" in key
            ]
            start_idx = len(bad_freq_arrays)
            for i, indices in enumerate(pspec.bad_freq_indices):
                h5f.create_dataset(
                    f"bad frequency indices {start_idx + i}", data=indices, dtype="i"
                )
            datetimes_stack = np.append(
                datetimes_stack, [date.timestamp() for date in pspec.datetimes]
            )
            del h5f["datetimes"]
            h5f.create_dataset("datetimes", data=datetimes_stack)
            new_obs_ids = np.append(
                h5f.attrs["observation ids"], pspec.obs_id
            ).flatten()
            del h5f.attrs["observation ids"]
            h5f.attrs["observation ids"] = list(new_obs_ids)

            if type(pspec.rn_medians) != np.ndarray:
                log.error("This power spectrum does not have rednoise info saved.")

            elif "rn medians" not in h5f.keys():
                log.error("This h5f file does not have rednoise info saved.")

            else:
                h5f_rn_medians = h5f["rn medians"]
                h5f_rn_scales = h5f["rn scales"]
                # set guard value as -1 to pad
                ndays = len(pspec.rn_medians) + len(h5f_rn_medians)
                stacked_rn_medians = -1 * np.ones(
                    (
                        ndays,
                        pspec.rn_dm_indices.shape[1],
                        max(pspec.rn_medians.shape[2], h5f_rn_medians.shape[2]),
                    )
                )
                stacked_rn_medians[
                    : len(h5f_rn_medians), :, : h5f_rn_medians.shape[2]
                ] = h5f_rn_medians
                stacked_rn_medians[
                    len(h5f_rn_medians) :, :, : pspec.rn_medians.shape[2]
                ] = pspec.rn_medians
                stacked_rn_scales = -1 * np.ones(
                    (ndays, max(pspec.rn_scales.shape[1], h5f_rn_scales.shape[1]))
                )
                # only need one DM value per day of scales
                stacked_rn_scales[: len(h5f_rn_scales)] = h5f_rn_scales
                stacked_rn_scales[len(h5f_rn_scales) :] = pspec.rn_scales

                del h5f["rn medians"]
                del h5f["rn scales"]
                h5f["rn medians"] = stacked_rn_medians
                h5f["rn scales"] = stacked_rn_scales

        log.info("Stacking completed")
        if self.update_db:
            log.info(f"Updating the stack database for {pspec.ra} {pspec.dec}")
            # first get the pointing
            pointing_id = db_api.get_observation(pspec.obs_id[0]).pointing_id
            new_datetimes = [
                datetime.datetime.utcfromtimestamp(date).replace(tzinfo=pytz.UTC)
                for date in datetimes_stack
            ]
            payload = {
                f"datapath_{self.mode}": stack_file_path,
                f"datetimes_{self.mode}": new_datetimes,
                f"num_days_{self.mode}": int(new_num_days),
            }
            db_api.update_ps_stack(pointing_id, payload)

    def stack_power_spectra_readout(self, pspec, stack_file_path):
        """
        The script to stack two power spectra by reading out the stack in disk in
        chunks, adding it to the current stack and then update the properties of the
        current stack. This method will avoid the problem of infile overwrite, as the
        stack will be fully added before any attempts to write to disk is done. This
        process will have similar memory efficiency as the infile overwrite method as
        only one slice of the power spectra is loaded onto memory at once. The primary
        drawback of this method is that the information from the initial stack will be
        lost. If this method is to be applied on stacking of daily stack into monthly
        stack, then the search on daily stack will have to be conducted prior to the
        stacking process. The feasibility of the inverted processing step will depend on
        how much data is generated by the power spectra searching process.

        Parameters
        ----------
        pspec: PowerSpectra
            The PowerSpectra object containing the daily power spectra stack to be stacked into the monthly spectra
            stack.

        stack_file_path: str
            The path to the monthly/cumulative power spectra stack to load and stack.

        Returns
        -------
        pspec_stack: PowerSpectra
            The stacked power spectra between daily/monthly or monthly/cumulative stacks.
        """

        with h5py.File(stack_file_path, "r") as h5f:
            # Make some check to ensure stacks are compatible with each other
            log.info("Checking for consistency between stacks")
            ps_shape = (len(pspec.power_spectra), len(pspec.power_spectra[0]))
            assert ps_shape == h5f["power spectra"].shape, (
                "The two power spectra stack do not have the same size."
            )
            stack_freq_labels = np.ones(h5f["frequency labels"].shape)
            h5f["frequency labels"].read_direct(stack_freq_labels)
            np.testing.assert_array_equal(
                pspec.freq_labels,
                stack_freq_labels,
                "The two power spectra stack do not have the same frequency labels.",
            )
            datetimes_stack = np.ones(h5f["datetimes"].shape)
            h5f["datetimes"].read_direct(datetimes_stack)
            for dt, dts in itertools.product(pspec.datetimes, datetimes_stack):
                assert np.abs(dt.timestamp() - dts) > 60, (
                    "The two power spectra stack came from the same day."
                )
            log.info(f"Stacking the {self.mode} power spectra into the current stack")
            for i in range(len(pspec.dms)):
                pspec.power_spectra[i] += h5f["power spectra"][(i,)]
            log.info(f"Updating the new {self.mode} power spectra information")
            pspec.num_days += h5f.attrs["number of days"]

            if type(pspec.rn_medians) != np.ndarray:
                log.error("This power spectrum does not have rednoise info saved.")

            elif "rn medians" not in h5f.keys():
                log.error("This h5f file does not have rednoise info saved.")

            else:
                h5f_rn_medians = h5f["rn medians"]
                h5f_rn_scales = h5f["rn scales"]

                # set guard value as -1 to pad
                ndays = len(pspec.rn_medians) + len(h5f_rn_medians)
                stacked_rn_medians = -1 * np.ones(
                    (
                        ndays,
                        pspec.rn_dm_indices.shape[1],
                        max(pspec.rn_medians.shape[2], h5f_rn_medians.shape[2]),
                    )
                )
                stacked_rn_medians[
                    : len(h5f_rn_medians), :, : h5f_rn_medians.shape[2]
                ] = h5f_rn_medians
                stacked_rn_medians[
                    len(h5f_rn_medians) :, :, : pspec.rn_medians.shape[2]
                ] = pspec.rn_medians
                stacked_rn_scales = -1 * np.ones(
                    (ndays, max(pspec.rn_scales.shape[1], h5f_rn_scales.shape[1]))
                )
                # only need one DM value per day of scales
                stacked_rn_scales[: len(h5f_rn_scales)] = h5f_rn_scales
                stacked_rn_scales[len(h5f_rn_scales) :] = pspec.rn_scales

                pspec.rn_medians = stacked_rn_medians
                pspec.rn_scales = stacked_rn_scales

            log.info("Stacking completed.")

            bad_freq_arrays = [
                key for key in h5f.keys() if "bad frequency indices" in key
            ]
            bad_freq_indices = []
            for key in bad_freq_arrays:
                bad_freq_indices.append(h5f[key][()])
            bad_freq_indices.extend(pspec.bad_freq_indices)
            pspec.bad_freq_indices = bad_freq_indices
            new_datetimes = [
                datetime.datetime.utcfromtimestamp(date).replace(tzinfo=pytz.UTC)
                for date in datetimes_stack
            ]
            new_datetimes.extend(pspec.datetimes)
            pspec.datetimes = new_datetimes
            pspec.obs_id = (
                np.append(h5f.attrs["observation ids"], pspec.obs_id).flatten().tolist()
            )

        return pspec

    def stack_power_spectra(self, pspec, pspec_stack):
        """
        The actual process of stacking the two power spectra stack together. Starts with
        checks to ensure that the two stacks are of equal size and frequency labels,
        then a third check to make sure we are not stacking data from the same day
        together, before running the stacking process.

        Parameters
        ==========
        pspec: PowerSpectra
            PowerSpectra object of the power spectra to be stacked.

        pspec_stack: PowerSpectra
            PowerSpectra object of the power spectra to be stacked into.

        Returns
        =======
        pspec_stack: PowerSpectra
            PowerSpectra object of the power spectra to be stacked into, post stacking.
        """
        # Make some check to ensure stacks are compatible with each other
        ps_shape = (len(pspec.power_spectra), len(pspec.power_spectra[0]))
        assert ps_shape == pspec_stack.power_spectra.shape, (
            "The two power spectra stack do not have the same size."
        )
        np.testing.assert_array_equal(
            pspec.freq_labels,
            pspec_stack.freq_labels,
            "The two power spectra stack do not have the same frequency labels.",
        )
        for dt, dts in itertools.product(pspec.datetimes, pspec_stack.datetimes):
            assert np.abs(dt.timestamp() - dts.timestamp()) > 60, (
                "The two power spectra stack came from the same day."
            )

        stack_start = time.time()
        log.info("Stacking the two power spectra together")
        pspec_stack.power_spectra += pspec.power_spectra
        pspec_stack.num_days += pspec.num_days
        pspec_stack.bad_freq_indices.extend(pspec.bad_freq_indices)
        pspec_stack.datetimes.extend(pspec.datetimes)
        pspec_stack.obs_id.extend(pspec.obs_id)

        try:
            pspec_stack.rn_medians.extend(pspec.rn_medians)
            pspec_stack.rn_scales.extend(pspec.rn_scales)
            pspec_stack.rn_dm_indices.extend(pspec.rn_scales)

        except:
            if pspec.rn_medians is None:
                log.error("The daily power spectrum does not have rednoise info saved.")
            if pspec_stack.rn_medians is None:
                log.error("The power spectra stack does not have rednoise info saved.")

        stack_end = time.time()
        log.debug(f"Took {stack_end - stack_start} seconds to stack power spectra")

        return pspec_stack

    def replace_spectra(self, pspec, stack_file_path):
        """
        Script to replace the power spectra stack of the stack_file_path with the given
        power spectra stack. The process will write the new stack into a temporary
        location before removing the pre-existing stack, finally rename the new stack to
        stack_file_path.

        Parameters
        ----------
        pspec: PowerSpectra
            The PowerSpectra object of the new stack to replace the power spectra stack on disk at stack_file_path.

        stack_file_path: str
            Path to the power spectra stack to be replaced by the new stack.
        """
        log.info(f"Writing new {self.mode} stack for {pspec.ra:.2f} {pspec.dec:.2f}")
        temp_path = (
            "/".join(stack_file_path.split("/")[:-1])
            + f"/{pspec.ra:.2f}_{pspec.dec:.2f}_{self.mode}_temp.hdf5"
        )
        if os.path.isfile(temp_path):
            log.info(f"Found old temp file {temp_path}. Will remove this file.")
            os.remove(temp_path)
        pspec.write(temp_path, nbit=self.stack_nbit)
        log.info(
            f"Deleting the existing {self.mode} power spectra stack '{stack_file_path}'"
        )
        if os.path.isfile(stack_file_path):
            os.remove(stack_file_path)
        log.info(f"Rename the new stack to the rightful path of '{stack_file_path}'")
        os.rename(temp_path, stack_file_path)

    def quality_control(self, pspec):
        """
        Run quality control for power spectra. Currently there is a chi2 distribution
        test and a test comapring the expected number of sample "outliers" compared to
        pure noise power spectra.

        Parameters
        ==========
        pspec: PowerSpectra
            PowerSpectra object of the power spectra to be tested.

        compared_obs_count: int
            How many observations are used for the dynamic thresholds

        Returns
        =======
        qc_result: bool
            Whether the power spectrum satisfies the quality tests.

        quality_metrics: dict
            Dictionary containing the results from the quality tests.
        """
        log.info("Calculating quality metrics.")

        quality_metrics = {}
        # We use one default red_noise_nbins parameter which can be overwritten in the individual metrics
        red_noise_default = self.qc_config["red_noise_nbins"]
        obs = db_api.get_observation(pspec.obs_id[-1])
        for current_metric_name in self.qc_config["qc_metrics"]:
            current_metric = self.qc_config["qc_metrics"][current_metric_name]
            # By default the DM 0 spectrum will be tested but it is also possible to test other channels
            tested_channels = current_metric.get("channels", [0])
            metric_parameters = current_metric.get("parameters", {})
            minimum_bin_weight = current_metric.get("minimum_bin_weight", 0)
            red_noise_current = current_metric.get("red_noise_nbins", red_noise_default)
            tested_ps = np.asarray(pspec.power_spectra)[tested_channels, 1:]
            # We need to cut both red_noise_bins and the one with a too low bin_weight
            # There are probably quicker ways to peform this slicing
            all_indices = np.arange(tested_ps.shape[1])
            valid_indices = all_indices >= red_noise_current
            if minimum_bin_weight != 0:
                bin_weights = pspec.get_bin_weights_fraction()[1:]
                valid_indices_bins = bin_weights >= minimum_bin_weight
                valid_indices = valid_indices & valid_indices_bins
            tested_ps = tested_ps[:, valid_indices].flatten()

            if current_metric["type"] == "kstest":
                # this method makes a kstest2 comparing against an expected chi2 distribution
                (
                    current_ksstat,
                    current_pval,
                ) = compare_ps_to_chisqr_kstest(
                    tested_ps,
                    ndays=pspec.num_days,
                    red_noise_nbins=0,
                    **metric_parameters,
                )
                quality_metrics[current_metric_name] = {
                    "ksdist": float(current_ksstat),
                    "pval": float(current_pval),
                }
            elif current_metric["type"] == "kstest_chi2_fit":
                # this method allows fitting the number of days
                # this allows testing if the number of degrees of freedom is chi2
                expected_days = pspec.num_days
                day_vals = []
                ksdists = []
                for days in np.linspace(
                    (1 - current_metric["range"]) * expected_days,
                    (1 + current_metric["range"]) * expected_days,
                    current_metric["count"],
                ):
                    (
                        current_ksstat,
                        current_pval,
                    ) = compare_ps_to_chisqr_kstest(
                        tested_ps,
                        ndays=days,
                        red_noise_nbins=0,
                        **metric_parameters,
                    )
                    day_vals.append(days)
                    ksdists.append(float(current_ksstat))
                min_val = min(ksdists)
                min_days = day_vals[np.argmin(ksdists)]
                fraction = min_days / expected_days
                diff = min_days - expected_days
                quality_metrics[current_metric_name] = {
                    "day_vals": day_vals,
                    "ksdists": ksdists,
                    "min_days": min_days,
                    "min_ksdist": min_val,
                    "fraction": fraction,
                    "difference": diff,
                }
            elif current_metric["type"] == "outlier":
                # this method computes the number of outliers compared those of a chi2 distribution
                (
                    expected_outliers,
                    found_outliers,
                    diff_outliers,
                    frac_outliers,
                ) = validate_ps_chisqr_outlier_bins(
                    tested_ps,
                    ndays=pspec.num_days,
                    red_noise_nbins=0,
                    **metric_parameters,
                )
                quality_metrics[current_metric_name] = {
                    "expected": expected_outliers.tolist(),
                    "found": found_outliers.tolist(),
                    "difference": diff_outliers.tolist(),
                    "fraction": frac_outliers.tolist(),
                }
            elif current_metric["type"] == "masked_bins":
                # this method checks how many of of the bins are spectra are completely masked
                birdie_count = np.count_nonzero(tested_ps == 0)
                mask_fraction = birdie_count / len(tested_ps)
                quality_metrics[current_metric_name] = {
                    "count": birdie_count,
                    "mask_fraction": mask_fraction,
                }
            elif current_metric["type"] == "obs_properties":
                quality_metrics[current_metric_name] = {}
                for obs_property in current_metric["properties"]:
                    quality_metrics[current_metric_name][obs_property] = getattr(
                        obs, obs_property, None
                    )
            else:
                # any method included in the imported libraries can be used here, numpy for example
                metric_func = current_metric["type"]
                value = eval(metric_func + "(tested_ps, **metric_parameters)")
                quality_metrics[current_metric_name] = {
                    "value": np.asarray(value).tolist(),
                    "parameters": metric_parameters,
                }
            log.info(
                f"Quality result: {current_metric_name}:"
                f" {quality_metrics[current_metric_name]}"
            )

        log.info("Comparing quality metrics with given thresholds.")
        qc_result = True
        compared_obs = []
        looked_for_compared_obs = False
        for current_metric_name in self.qc_config["qc_metrics"]:
            current_metric = self.qc_config["qc_metrics"][current_metric_name]
            if "qc_tests" in current_metric:
                for current_test in current_metric["qc_tests"]:
                    test_value = quality_metrics[current_metric_name][
                        current_test["metric"]
                    ]

                    llim = current_test.get("lower_limit", -np.inf)
                    ulim = current_test.get("upper_limit", +np.inf)
                    if llim is None or llim == "None":
                        llim = -np.inf
                    if ulim is None or ulim == "None":
                        ulim = +np.inf
                    log.info(
                        f"Testing {current_metric_name} with metric"
                        f" {current_test['metric']}. Static Limit: {llim} - {ulim}"
                    )
                    # Get dynamic limits. Only query the database once for all tests
                    # Dynamic threshold only currently available for self.mode=="month"
                    if current_test.get("upper_limit_dynamic") or current_test.get(
                        "lower_limit_dynamic"
                    ):
                        if self.mode == "month":
                            if not looked_for_compared_obs:
                                oldest_date_to_compare_against = (
                                    datetime.datetime.strptime(
                                        self.qc_config["oldest_date_for_comparison"],
                                        "%Y/%m/%d",
                                    ).replace(tzinfo=datetime.timezone.utc)
                                )
                                compared_obs = db_api.get_observations(obs.pointing_id)
                                # Remove the observation itself if it wa already processed
                                compared_obs = [
                                    obs
                                    for obs in compared_obs
                                    if (obs._id != pspec.obs_id[-1])
                                    and (
                                        obs.last_changed
                                        > oldest_date_to_compare_against
                                    )
                                ][-self.qc_config.get("dynamic_max_obs", 10000) :]
                                looked_for_compared_obs = True
                                log.info(
                                    f"Loaded {len(compared_obs)} observations for dynamic thresholds."
                                )
                            if len(compared_obs) >= self.qc_config.get(
                                "dynamic_min_obs", 2
                            ):
                                all_vals = grab_metric_history(
                                    compared_obs,
                                    current_metric_name,
                                    current_test["metric"],
                                )
                                median = np.nanmedian(all_vals)
                                mad = median_abs_deviation(all_vals, nan_policy="omit")
                                if current_test.get("lower_limit_dynamic"):
                                    lower_limit_dynamic = (
                                        median
                                        + current_test.get("lower_limit_dynamic") * mad
                                    )
                                    llim = np.nanmin((ulim, lower_limit_dynamic))
                                else:
                                    lower_limit_dynamic = -np.inf
                                if current_test.get("upper_limit_dynamic"):
                                    upper_limit_dynamic = (
                                        median
                                        + current_test.get("upper_limit_dynamic") * mad
                                    )
                                    ulim = np.nanmax((ulim, upper_limit_dynamic))
                                else:
                                    upper_limit_dynamic = np.inf
                                log.info(
                                    f"Testing {current_metric_name} with metric"
                                    f" {current_test['metric']}. Dynamic Limit:"
                                    f" {lower_limit_dynamic} - {upper_limit_dynamic}"
                                )
                        else:
                            log.error(
                                "Dynamic Threshold not implemented for cumualtive"
                                " stacking yet."
                            )

                    test_result = check_in_range(test_value, llim, ulim)
                    if not test_result:
                        log.warning(
                            f"Metric {current_test['metric']} in quality test"
                            f" {current_metric_name} is NOT in range {llim} - {ulim}!  "
                            "                   \n                      "
                            f" {current_test['metric']}: {test_value}"
                        )
                        qc_result = False
                    else:
                        log.info(
                            f"Metric {current_test['metric']} in quality test"
                            f" {current_metric_name} is in range {llim} - {ulim}!      "
                            "           \n                      "
                            f" {current_test['metric']}: {test_value}"
                        )

        log.info(
            "Final quality check result. The spectra will be added to the stack:"
            f" {qc_result}"
        )
        return qc_result, quality_metrics

    def get_ps_stack_db(self, pspec):
        """
        Get the database of the pointing of the power spectra to be stacked.

        Parameters
        ==========
        pspec: PowerSpectra
            The PowerSpectra object of the power spectra to be stacked.

        Returns
        =======
        ps_stack_db: PsStack
            The PsStack object of the database entry of the pointing of the power spectra
            to be stacked.
        """
        try:
            pointing_id = db_api.get_observation(pspec.obs_id[0]).pointing_id
            ps_stack_db = db_api.get_ps_stack(pointing_id)
        except Exception as e:
            log.warning(e)
            log.warning("Returning None for the stack database")
            ps_stack_db = None
        return ps_stack_db

    def get_stack_file_path(self, pspec):
        """
        Outputs the location to write the power spectra stack into based on the input
        power spectra, the basepath and the mode of stacking.

        Parameters
        ==========
        psepc: PowerSpectra
            The PowerSpectra object of the power spectra to be stacked.

        Returns
        =======
        stack_file_path: str
            The location to write the stacked power spectra into.
        """
        if self.basepath == "./":
            root_dir = os.getcwd()
        else:
            root_dir = self.basepath
        os.makedirs(f"{root_dir}/stack/", exist_ok=True)
        stack_file_path = (
            f"{root_dir}/stack/{pspec.ra:.2f}_{pspec.dec:.2f}_power_spectra_stack.hdf5"
        )
        if self.mode == "cumul":
            stack_file_path = (
                stack_file_path.split("power_spectra_stack.hdf5")[0]
                + "cumulative_power_spectra_stack.hdf5"
            )
        return stack_file_path

    def create_stack_database(self, pspec, pspec_file_loc, pointing_id=None):
        """
        Create a new database entry for a power spectra stack.

        Parameters
        ==========
        pspec: PowerSpectra
            The PowerSpectra object of the power spectra to be stacked.

        pspec_file_loc: str
            The location to write the power spectra stack to.

        pointing_id: str
            The pointing ID of the power spectra stack, if known. Otherwise,
            it will be retrieved from the observation database.
        """
        if not pointing_id:
            pointing_id = db_api.get_observation(pspec.obs_id[0]).pointing_id
        payload = {
            "pointing_id": pointing_id,
            "datapath_month": pspec_file_loc,
            "datetimes_month": pspec.datetimes,
            "num_days_month": pspec.num_days,
            "datapath_cumul": "",
            "datetimes_cumul": [],
            "num_days_cumul": 0,
        }
        db_api.create_ps_stack(payload)

    def update_stack_database(self, pspec, stack_file_path, pointing_id=None):
        """
        Update the database on the newest stack.

        Parameters
        ==========
        pspec: PowerSpectra
            The PowerSpectra object of the power spectra to be stacked.

        stack_file_path: str
            The location to write the power spectra stack to.

        pointing_id: str
            The pointing ID of the power spectra stack, if known. Otherwise,
            it will be retrieved from the observation database.
        """
        log.info(f"Updating the stack database for {pspec.ra} {pspec.dec}")
        # first get the pointing
        if not pointing_id:
            pointing_id = db_api.get_observation(pspec.obs_id[0]).pointing_id
        payload = {
            f"datapath_{self.mode}": stack_file_path,
            f"datetimes_{self.mode}": pspec.datetimes,
            f"num_days_{self.mode}": int(pspec.num_days),
        }
        db_api.update_ps_stack(pointing_id, payload)

    def update_obs_database(self, pspec, to_stack, qc_test={}):
        """
        Update the database on the observation properties of the quality control
        results.

        Parameters
        ==========
        pspec: PowerSpectra
            PowerSpectra object with the properties of the power spectra to stack.

        to_stack: bool
            Whether the daily power spectra is to be stacked to the monthly power spectra stack.

        qc_test: dict
            A dictionary of the quality control tests done and their results.
        """
        log.info(
            f"Updating the observation database for {pspec.ra:.2f} {pspec.dec:.2f} at"
            f" {pspec.datetimes[-1]}"
        )
        payload = {"add_to_stack": to_stack, "qc_test": qc_test}
        db_api.update_observation(pspec.obs_id[-1], payload)

    def delete_stack(self, pointing_id, stack_file_path):
        """
        Delete the monthly power spectra stack after it is stacked into the cumulative
        power spectra stack.

        Parameters
        ==========
        pointing_id: str
            The pointing ID of the power spectra stack

        stack_file_path: str
            The path to the power spectra stack to be removed.
        """
        log.info(f"Deleting the monthly power spectra stack '{stack_file_path}'")
        if os.path.isfile(stack_file_path):
            os.remove(stack_file_path)
        if self.update_db:
            payload = {
                "datapath_month": "",
                "datetimes_month": [],
                "num_days_month": 0,
            }
            db_api.update_ps_stack(pointing_id, payload)

    def lock_stack(self, stack_file_path=None):
        """
        Checks if a .lock file exists which prevents further processing.

        If the lock file exists a new one will be created.
        The lock will be ignored if the lock stack age surpassed self.max_lock_age seconds.
        Parameters
        ==========
        stack_file_path: str
            The path to the power spectra stack which will be locked.
        """
        if not hasattr(self, "lock"):
            if stack_file_path:
                path_object = PurePath(stack_file_path)
                lock_path = f"{path_object.parent}/.{path_object.stem}.lock"
                self.lock = FileLock(lock_path, timeout=self.max_lock_age)
            else:
                log.error("Trying to set a lock but no path to stack provided.")
                return

        try:
            log.info("Trying to acquire stack lock.")
            self.lock.acquire(poll_interval=1)
            log.info(f"Stack lock acquired at {self.lock.lock_file}")
        except Timeout:
            log.error(
                "Lock not acquired after {self.max_lock_age} seconds. "
                "Will forcefully delete the lock file and try again."
            )
            try:
                os.remove(self.lock.lock_file)
                self.lock_stack()
            except OSError:
                log.info(
                    "Was not able to remove the lock. Will try to process without lock."
                )

    def unlock_stack(self):
        """
        Unlock the file lock. This does not remove the lock file.

        Removing the lock fiel can cause issues with other processes aquiring the lock
        properly.
        """
        if hasattr(self, "lock"):
            self.lock.release()
            log.info("Removed lock on stack.")
