"""Stack rows stored only up to their DM's smearing cutoff."""

import h5py
import numpy as np
import pytest
from sps_common.constants import DM_CONSTANT, FREQ_BOTTOM, FREQ_TOP, TSAMP
from sps_common.freq_cut import (
    freq_cut_settings,
    kept_bins,
    smearing_cutoff,
    stored_freq_cut,
)
from sps_common.stack_quant import read_quantized, round_power, write_quantized

STEP = 0.25
NYQUIST = 0.5 / TSAMP


def test_cutoff_values():
    dms = np.array([0.0, 10.0, 100.0, 1000.0])
    cut = smearing_cutoff(dms, 1024)
    assert cut[0] == NYQUIST
    assert np.all(np.diff(cut) < 0)
    # By hand at DM 100, 1024 channels, 600 MHz
    smear = 2 * DM_CONSTANT * 100 * ((FREQ_TOP - FREQ_BOTTOM) / 1024) / 600.0**3
    assert cut[2] == pytest.approx(0.5 / np.hypot(TSAMP, smear))
    # More channels smear less; a higher reference frequency smears less
    assert smearing_cutoff([100.0], 4096)[0] > cut[2]
    assert smearing_cutoff([100.0], 1024, ref_freq_mhz=800.0)[0] > cut[2]
    # factor scales the cutoff until it reaches Nyquist
    assert smearing_cutoff([1000.0], 1024, factor=1.0)[0] == pytest.approx(2 * cut[3])
    with pytest.raises(ValueError):
        smearing_cutoff([1.0], 0)


def test_kept_bins():
    labels = np.arange(1000) * (NYQUIST / 1000)
    keep = kept_bins(labels, [0.0, 50.0, 500.0], 1024)
    assert keep[0] == 1000
    cut = smearing_cutoff([50.0, 500.0], 1024)
    for k, c in zip(keep[1:], cut):
        assert labels[k - 1] <= c < labels[k]


def noise_stack(days, shape, seed=1):
    rng = np.random.default_rng(seed)
    return sum(round_power(rng.exponential(1, shape), STEP) for _ in range(days)).astype(
        np.float32
    )


@pytest.mark.parametrize("threads", [0, 4])
def test_write_with_keep(tmp_path, threads):
    days = 7
    spectra = noise_stack(days, (40, 5000))
    spectra[:, 4000] += 9000  # bright: overflow where kept, dropped where cut
    keep = np.linspace(5000, 300, 40).astype(np.int64)
    with h5py.File(tmp_path / "s.h5", "w") as f:
        write_quantized(f, spectra, STEP, days, threads=threads, keep=keep)
    with h5py.File(tmp_path / "s.h5") as f:
        np.testing.assert_array_equal(f["kept bins"][()], keep)
        back = read_quantized(f, raw=True)
        overflow_rows = f["overflow row"][()]
    for i, k in enumerate(keep):
        np.testing.assert_array_equal(back[i, :k], spectra[i, :k])
        assert not back[i, k:].any()
    assert set(overflow_rows) == {i for i, k in enumerate(keep) if k > 4000}


def test_cut_rows_compress(tmp_path):
    days = 7
    spectra = noise_stack(days, (40, 20000))
    with h5py.File(tmp_path / "full.h5", "w") as f:
        write_quantized(f, spectra, STEP, days)
    with h5py.File(tmp_path / "cut.h5", "w") as f:
        write_quantized(f, spectra, STEP, days, keep=np.full(40, 5000))
    full = (tmp_path / "full.h5").stat().st_size
    cut = (tmp_path / "cut.h5").stat().st_size
    assert cut < 0.3 * full


def test_keep_checks(tmp_path):
    spectra = noise_stack(1, (4, 100))
    with h5py.File(tmp_path / "s.h5", "w") as f:
        with pytest.raises(ValueError):
            write_quantized(f, spectra, STEP, 1, keep=[10, 10])


def test_settings():
    assert freq_cut_settings({"nchan": 1024}) == {
        "nchan": 1024,
        "ref_freq_mhz": 600.0,
        "factor": 0.5,
    }
    with pytest.raises(ValueError):
        freq_cut_settings({"ref_freq_mhz": 600})
    with pytest.raises(ValueError):
        freq_cut_settings({"nchan": 1024, "edge": "top"})


def test_power_spectra_write(tmp_path):
    from datetime import datetime

    import pytz

    from sps_common.interfaces.ps_processes import PowerSpectra

    ndm, nbin = 6, 4000
    labels = np.arange(nbin) * (NYQUIST / nbin)
    pspec = PowerSpectra(
        power_spectra=noise_stack(1, (ndm, nbin)),
        dms=np.linspace(0, 600, ndm),
        freq_labels=labels,
        ra=10.0,
        dec=20.0,
        datetimes=[datetime(2026, 10, 10, tzinfo=pytz.utc)],
        num_days=1,
        beta=0.0,
        bad_freq_indices=[[1, 2]],
        obs_id=["x"],
    )
    path = tmp_path / "s.h5"
    pspec.write(str(path), quant_step=STEP, freq_cut={"nchan": 1024, "ref_freq_mhz": 600.0})
    with h5py.File(path) as f:
        assert stored_freq_cut(f) == {"nchan": 1024, "ref_freq_mhz": 600.0, "factor": 0.5}
        expected = kept_bins(labels, pspec.dms, 1024)
        np.testing.assert_array_equal(f["kept bins"][()], expected)
        back = read_quantized(f, raw=True)
    for i, k in enumerate(expected):
        np.testing.assert_array_equal(back[i, :k], pspec.power_spectra[i, :k])
        assert not back[i, k:].any()
    pspec.write(str(tmp_path / "plain.h5"), quant_step=STEP)
    with h5py.File(tmp_path / "plain.h5") as f:
        assert stored_freq_cut(f) is None and "kept bins" not in f
    with pytest.raises(ValueError):
        pspec.write(str(tmp_path / "f32.h5"), freq_cut={"nchan": 1024})
