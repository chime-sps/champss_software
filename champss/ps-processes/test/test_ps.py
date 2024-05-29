import datetime
import os
import shutil
import time
from random import random

import numpy as np
import pytest
from astropy.time import Time
from ps_processes import FailedChi2TestError
from ps_processes.processes import ps, ps_search, ps_stack
from sps_common.interfaces.ps_processes import DedispersedTimeSeries, PowerSpectra

TEST_RA = random() * 360
TEST_DEC = random() * 90
TEST_MJD = Time(time.time(), format="unix").mjd


def test_ps_scripts():
    dedisp_array = np.random.random(size=(10, 61440)) * 10
    dms = np.arange(0, 10)
    obs_id = ""
    dedisp_ts = DedispersedTimeSeries(
        dedisp_ts=dedisp_array,
        dms=dms,
        ra=TEST_RA,
        dec=TEST_DEC,
        start_mjd=TEST_MJD,
        obs_id=obs_id,
    )
    ps_creation = ps.PowerSpectraCreation(padded_length=65536, update_db=False)
    power_spectra = ps_creation.transform(dedisp_ts)
    search = ps_search.PowerSpectraSearch(padded_length=65536, remove_duplicates=True)
    ps_detections = search.search(power_spectra)
    stack = ps_stack.PowerSpectraStack(
        update_db=False,
        qc_config={
            "red_noise_nbins": 2048,
            "qc_metrics": {
                "ks_test": {
                    "type": "kstest",
                    "parameters": {
                        "lower_percentile_cut": None,
                        "upper_percentile_cut": None,
                        "plots": True,
                    },
                }
            },
            "qc_tests": {"ks_test": {"metric": "ksdist", "upper_limit": None}},
        },
    )
    try:
        stacked_spectra = stack.stack(power_spectra)
    except FailedChi2TestError:
        print("rerun stacking test without QC")
        stack = ps_stack.PowerSpectraStack(update_db=False, qc=False)
        stacked_spectra = stack.stack(power_spectra)
    # test various individual stacking method
    stack_file_path = stack.get_stack_file_path(power_spectra)
    power_spectra.datetimes[0] += datetime.timedelta(1)
    stack.stack_power_spectra_infile(power_spectra, stack_file_path)
    power_spectra.datetimes[0] += datetime.timedelta(1)
    stack_from_file = PowerSpectra.read(stack_file_path)
    stack.stack_power_spectra(power_spectra, stack_from_file)
    test_stack = stack.stack_power_spectra_readout(power_spectra, stack_file_path)
    stack.replace_spectra(test_stack, stack_file_path)
    det_freq_spacing = power_spectra.freq_labels[1]
    assert power_spectra.bad_freq_indices
    if ps_detections is not None:
        assert ps_detections.freq_spacing == det_freq_spacing
    else:
        print("No detections made.")
    os.remove("raw_and_masked_PS.png")
    os.remove("raw_and_masked_PS_vs_chi2.png")
    os.remove(f"stack/{TEST_RA:.2f}_{TEST_DEC:.2f}_power_spectra_stack.hdf5")
    shutil.rmtree("stack")
