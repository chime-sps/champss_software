import pytest
import numpy as np
import time
from beamformer.skybeam.skybeam import SkyBeam
from beamformer.utilities.blurmask import blur
from sps_common.conversion import convert_intensity_to_hdf5, read_intensity_hdf5

RA = np.random.rand() * 360.0
DEC = np.random.rand() * 90.0
NCHAN = 1024
LENGTH = (
    (np.random.randint(40, 400) // 40) * 40
) * 1024  # A number that is a factor of 40 * 1024
FILE_LENGTH = LENGTH + 40 * 1024
MAX_DM = np.random.rand() * 212.5
BEAM_ROW = np.random.randint(0, 224)
TSAMP = 0.00098304
TIME_START = time.time()
TIME_END = TIME_START + FILE_LENGTH * TSAMP
MAX_BEAM = [
    {"beam": BEAM_ROW, "utc_start": int(TIME_START + 1), "utc_end": int(TIME_END - 5)}
]


@pytest.fixture(scope="session")
def session_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("sub")
    p1 = {}
    p2 = {}
    for i, tick in enumerate(np.arange(TIME_START, TIME_END, 16384 * TSAMP)):
        hdf5_file_name = "{}_{}.hdf5".format(int(tick), int(tick+16384 * TSAMP))
        bird_file_name = hdf5_file_name.rstrip(".hdf5") + ".birds"
        p1["{}".format(i)] = d / hdf5_file_name
        p2["{}".format(i)] = d / bird_file_name
        spec = np.random.normal(size=(NCHAN, 16384))
        mask = np.random.randint(0, 2, size=(NCHAN, 16384))
        metadata = dict(
            beam_number=BEAM_ROW,
            nchan=NCHAN,
            ntime=16384,
            start=tick,
            end=tick+16384 * TSAMP,
        )
        convert_intensity_to_hdf5(spec, mask, p1["{}".format(i)], metadata)
        birdies = np.random.uniform(0, 500, size=(5, 2))
        with open(p2["{}".format(i)], "w") as bf:
            bf.write("# frequency(Hz), amplitude\n")
            for (bird, amp) in birdies:
                bf.write("{0:.5f} {1:.2f}\n".format(bird, amp))
    yield p1, p2


def test_session_file(session_file):
    print(dir(session_file[0]["0"]), dir(session_file[1]["0"]))
    print(session_file[0]["0"].parts, session_file[1]["0"].parts)
    print(
        list(session_file[0]["0"].parent.iterdir()), list(session_file[1]["0"].parent.iterdir())
    )
    intensity, mask, metadata = read_intensity_hdf5(session_file[0]["0"])
    assert intensity.shape == (NCHAN, 16384)
    assert mask.shape == (NCHAN, 16384)
    metadata_keys = ["beam_number", "nchan", "ntime", "start", "end"]
    for key in metadata_keys:
        assert key in metadata
    with open(session_file[1]["0"], "r") as bf:
        for line in bf.readlines():
            if not line.startswith("#"):
                for value in line.split():
                    assert float(value) >= 0.0
    assert len(list(session_file[0]["0"].parent.iterdir())) > 2


def test_skybeam(session_file):
    data_list = []
    for i in range(len(session_file[0])):
        data_list.append(str(session_file[0]["{}".format(i)]))
    sb = SkyBeam(
        ra=RA,
        dec=DEC,
        nchans=NCHAN,
        length=LENGTH,
        maxdm=MAX_DM,
        max_beams=MAX_BEAM,
        beam_row=BEAM_ROW,
        tsamp=TSAMP,
        data_list=data_list,
        nbits=32,
    )
    sb.process_storage_hdf5()
    original_masked_frac = sb.get_mask().mean()
    sb.adjust_mask(
        np.random.randint(75, 100) / 100,
        np.random.randint(75, 100) / 100,
        do_clumping=True,
    )
    sb.get_birdies()
    assert sb.get_mask().mean() > original_masked_frac
    assert type(sb.spectra) == np.ma.core.MaskedArray
    assert sb.spectra.data.sum() != 0.
    assert sb.spectra.shape == (NCHAN, LENGTH)
    assert sb.is_complete()
    assert type(sb.get_mask()) == np.ndarray
    assert sb.get_mask().shape == (NCHAN, LENGTH)
    assert 0 <= sb.get_mask().all() <= 1
    for bird in sb.birdies:
        for value in bird.split():
            assert float(value) >= 0.0
    sb.mask_spectra()
    assert type(sb.spectra) == np.ndarray
    assert sb.spectra.shape == (NCHAN, LENGTH)


def test_blurmask():
    test_array = np.random.randint(0, 2, size=(1024, 1024))
    test_output = blur(test_array)
    assert type(test_output) == np.ndarray
    assert test_output.dtype == float
    assert test_output.sum() > 0
    test_output = blur(test_array, mean=False)
    assert type(test_output) == np.ndarray
    assert test_output.dtype == float
    assert test_output.sum() > 0
    test_output = blur(test_array, mean=False, scale_edges=False)
    assert type(test_output) == np.ndarray
    assert test_output.dtype == float
    assert test_output.sum() > 0
