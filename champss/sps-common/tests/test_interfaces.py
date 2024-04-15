import datetime as dt

import numpy as np
import pytz

from sps_common.interfaces import multi_pointing, ps_processes, single_pointing


def test_power_spectra(tmp_path):
    """Reading a saved PowerSpectra instance should be equal to the original"""
    now = dt.datetime.utcnow().replace(tzinfo=pytz.UTC)
    ps = ps_processes.PowerSpectra(
        power_spectra=np.zeros((1, 1)),
        dms=np.zeros(1),
        freq_labels=np.zeros(1),
        ra=210,
        dec=42,
        datetimes=[now],
        num_days=1,
        beta=1,
        bad_freq_indices=[[1]],
        obs_id=["1"],
    )
    file_name = tmp_path / "test_power_spectra.h5"
    ps.write(file_name)
    ps2 = ps_processes.PowerSpectra.read(file_name)
    assert ps == ps2


def test_power_spectra_detections(tmp_path):
    """Reading a saved PowerSpectraDetections instance should be equal to the original"""
    detection_list = np.zeros(
        shape=1,
        dtype=[
            ("freq", float),
            ("dm", float),
            ("nharm", int),
            ("harm_pows", [("power", float, (32,)), ("freq", float, (32,))]),
            ("sigma", float),
        ],
    )
    psd = ps_processes.PowerSpectraDetections(
        detection_list=detection_list,
        freq_spacing=1,
        ra=210,
        dec=42,
        sigma_min=5.0,
        obs_id=["1"],
    )
    file_name = tmp_path / "test_power_spectra_detections.h5"
    psd.write(file_name)
    psd2 = ps_processes.PowerSpectraDetections.read(file_name)
    assert psd == psd2


def test_multi_pointing_candidate(tmp_path):
    """Reading a saved MultiPointingCandidate instance should be equal to the original"""
    spc = single_pointing.SinglePointingCandidate(
        freq=1.0,
        freq_arr=np.array([[1.0, 0.5], [2.0, 0.5]]),
        dm=1.0,
        dm_arr=np.array([[1.0, 0.5], [2.0, 0.5]]),
        dc=0.1,
        sigma=10,
        ra=200,
        dec=42,
        features=np.array([1.0, 0.5, 2]),
        detection_statistic=1,
        obs_id=["1"],
        harmonics_info=np.array([1.0, 0.5, 2]),
    )
    mpc = multi_pointing.MultiPointingCandidate(
        best_freq=1.0,
        # best_freq_arr=np.asarray([[1.0, 5.0]]),
        mean_freq=1.0,
        delta_freq=0.1,
        best_dm=1.0,
        # best_dm_arr=np.asarray([[1.0, 5.0]]),
        mean_dm=1.0,
        delta_dm=1.0,
        best_dc=0.1,
        best_sigma=10.0,
        ra=200.0,
        dec=42.0,
        summary={},
        features=np.asarray((0, 1), dtype=[("a", float), ("b", float)]),
        position_features=np.asarray((0, 1), dtype=[("c", float), ("d", float)]),
        obs_id=["1"],
        all_dms=[1.0, 2.0],
        all_freqs=[1.0, 2.0],
        all_sigmas=[10.0, 5.0],
        summed_raw_harmonic_powers=np.zeros((2, 3)),
        best_candidate=spc,
        position_sigmas=np.zeros((4, 2)),
    )
    # Could using a properly setup SinglePointingCandidate as best_candidate
    file_path = tmp_path / "test_multi_pointing_candidate"
    mpc.write(str(file_path))
    mpc2 = multi_pointing.MultiPointingCandidate.read(
        str(file_path)
        + "_f_{:.3f}_DM_{:.3f}_class_none.npz".format(
            mpc.best_freq,
            mpc.best_dm,
        )
    )
    # best_freq_arr and best_dm_arr are currently removed
    # np.testing.assert_array_equal(mpc.best_freq_arr, mpc2.best_freq_arr)
    # np.testing.assert_array_equal(mpc.best_dm_arr, mpc2.best_dm_arr)
    # mpc.best_freq_arr = None
    # mpc2.best_freq_arr = None
    # mpc.best_dm_arr = None
    # mpc2.best_dm_arr = None
    # assert mpc == mpc2
    for key in mpc.__dict__:
        if type(getattr(mpc, key)) == np.ndarray:
            assert (getattr(mpc, key) == getattr(mpc2, key)).all()
        elif type(getattr(mpc, key)) == single_pointing.SinglePointingCandidate:
            mpc_spc = type(getattr(mpc, key))
            mpc2_spc = type(getattr(mpc2, key))
            for key_spc in mpc_spc.__dict__:
                if type(getattr(mpc_spc, key_spc)) == np.ndarray:
                    assert (
                        getattr(mpc_spc, key_spc) == getattr(mpc2_spc, key_spc)
                    ).all()
                else:
                    assert getattr(mpc_spc, key_spc) == getattr(mpc2_spc, key_spc)
        else:
            assert getattr(mpc, key) == getattr(mpc2, key)
    assert len(mpc.__dict__) == len(mpc.as_dict())
