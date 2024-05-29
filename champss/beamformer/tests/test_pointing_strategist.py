#! /usr/bin/env python

import datetime
import time
from random import random

import numpy as np
import pytest
from astropy.time import Time
from beamformer import AVAILAIBLE_POINTING_MAPS
from beamformer.skybeam import SkyBeamFormer
from beamformer.strategist.strategist import PointingStrategist
from beamformer.utilities.common import get_all_data_list
from sps_common.interfaces.rfi_mitigation import SlowPulsarIntensityChunk

POINTING_MAP = AVAILAIBLE_POINTING_MAPS[0]
TSAMP = 0.00098304 * 64
NCHAN = 1024
TIME_START = time.time()
TIME_END = TIME_START + 900
TEST_BEAM = int(random() * 32)
TEST_DATETIME = datetime.datetime.utcfromtimestamp(TIME_START)
TEST_DATE = float(
    "{:04d}{:02d}{:02d}".format(
        TEST_DATETIME.year, TEST_DATETIME.month, TEST_DATETIME.day
    )
)
TEST_MJD = int(Time(TIME_START, format="unix").mjd)
TEST_FILE_PREFIX = "{}/{}/{}/1{}/".format(
    str(TEST_DATE)[0:4],
    str(TEST_DATE)[4:6],
    str(TEST_DATE)[6:8],
    str(TEST_BEAM).zfill(3),
)


@pytest.fixture(scope="session")
def session_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("sub")
    d.mkdir(parents=True, exist_ok=True)
    p = {}
    for j in range(0, 4):
        ntime = 16384
        if j == 3:
            ntime = 2048
        for i, tick in enumerate(np.arange(TIME_START - 400, TIME_END, ntime * TSAMP)):
            spec = np.random.normal(size=(NCHAN, ntime))
            mask = np.zeros(shape=(NCHAN, ntime))
            mask[0:64] = 1
            spectra = np.ma.array(spec, mask=mask.astype(bool))
            TEST_FILE_PREFIX = "{}/{}/{}/{}{}/".format(
                str(TEST_DATE)[0:4],
                str(TEST_DATE)[4:6],
                str(TEST_DATE)[6:8],
                str(j),
                str(TEST_BEAM).zfill(3),
            )
            p[f"{i}{j}"] = d / TEST_FILE_PREFIX
            p[f"{i}{j}"].mkdir(parents=True, exist_ok=True)
            intensity = SlowPulsarIntensityChunk(
                spectra=spectra,
                nchan=NCHAN,
                ntime=ntime,
                nsamp=NCHAN * ntime,
                start_unix_time=float(tick),
                start_mjd=Time(float(tick), format="unix").mjd,
                beam_number=int(f"{str(j)}{str(TEST_BEAM).zfill(3)}"),
                cleaned=True,
            )
            intensity.write(p[f"{i}{j}"], tsamp=TSAMP)

    yield p


def test_session_file(session_file):
    print(list(session_file["00"].parent.iterdir()))
    assert len(list(session_file["00"].parent.iterdir())) > 0


def test_pointing_strategist(session_file):
    basepath = "/".join(str(session_file["00"]).split("/")[:-4])
    strat = PointingStrategist(split_long_pointing=True, from_db=False)
    # test single pointing at high dec
    sap = strat.get_single_pointing(
        random() * 360.0, random() * 10 + 80.0, datetime.datetime.now()
    )
    assert len(sap) > 1
    # main test
    active = strat.get_pointings(TIME_START, TIME_END, np.asarray([TEST_BEAM]))
    assert len(active) > 0
    random_idx = int(random() * len(active))
    assert type(active) == list
    assert len(active[random_idx].max_beams) > 0
    assert type(active[random_idx].max_beams[0]) == dict
    assert active[0].max_beams[0]["utc_start"] < TIME_END
    assert active[random_idx].max_beams[0]["utc_start"] < TIME_END
    assert active[random_idx].max_beams[-1]["utc_end"] > TIME_START
    all_test_data = get_all_data_list(
        active_pointings=active,
        datapath=basepath,
        extn="hdf5",
    )
    assert len(all_test_data) > 0
    for data in all_test_data:
        assert "hdf5" in data
    active[0].nchan = NCHAN
    sbf = SkyBeamFormer(
        basepath=basepath,
        tsamp=TSAMP,
        masking_timescale=1024,
        mask_channel_frac=1.0,
        update_db=False,
    )
    sb = sbf.form_skybeam(active[0])
    sb_parallel = sbf.form_skybeam(active[0], num_threads=4)
    assert np.array_equal(sb.spectra, sb_parallel.spectra)
    assert sb.spectra.shape == (sb.nchan, sb.ntime)
    assert sb.spectra.sum() != 0.0
    assert sb_parallel.spectra.shape == (sb.nchan, sb.ntime)
    assert sb_parallel.spectra.sum() != 0.0
