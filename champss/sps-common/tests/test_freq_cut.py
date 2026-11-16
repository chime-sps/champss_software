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


DT_PER_DM = DM_CONSTANT * (FREQ_BOTTOM**-2 - FREQ_TOP**-2) / TSAMP


def test_cutoff_values():
    dms = np.array([0.0, 10.0, 100.0, 1000.0])
    cut = smearing_cutoff(dms, 1024)
    assert np.all(cut <= NYQUIST)
    assert np.all(np.diff(cut) < 0)
    # By hand at DM 100, 1024 channels, 600 MHz: channel smearing, and the RMS DM-step
    # smearing of the S/N-loss grid (step 2 eta w_eff samples, w_eff with the 400 MHz
    # channel's smearing)
    smear = 2 * DM_CONSTANT * 100 * ((FREQ_TOP - FREQ_BOTTOM) / 1024) / 600.0**3
    eta = np.sqrt(1 / 0.9**2 - 1)
    smear400 = 2 * DM_CONSTANT * 100 * ((FREQ_TOP - FREQ_BOTTOM) / 1024) / FREQ_BOTTOM**3
    step = 2 * eta * np.sqrt(2 + (smear400 / TSAMP) ** 2)
    t_dm = step / 2 / np.sqrt(3) * TSAMP
    assert cut[2] == pytest.approx(0.5 / np.sqrt(TSAMP**2 + smear**2 + t_dm**2))
    # More channels smear less; a higher reference frequency smears less
    assert smearing_cutoff([100.0], 4096)[0] > cut[2]
    assert smearing_cutoff([100.0], 1024, ref_freq_mhz=800.0)[0] > cut[2]
    # factor scales the cutoff until it reaches Nyquist
    assert smearing_cutoff([1000.0], 1024, factor=1.0)[0] == pytest.approx(2 * cut[3])
    with pytest.raises(ValueError):
        smearing_cutoff([1.0], 0)


def test_dm_step_smearing():
    from sps_common.dm_grid import snr_loss_dt_grid
    from sps_common.freq_cut import dm_step_smearing

    # A uniform grid of every 2nd delay: half the step, RMS over the offset: 1 / sqrt(3)
    dms = np.arange(0, 400, 2) / DT_PER_DM
    np.testing.assert_allclose(dm_step_smearing(dms, 1024, trials=dms), TSAMP / np.sqrt(3))
    # The S/N-loss grid: from its (whole-sample) trials, close to the grid formula
    grid = snr_loss_dt_grid(1024, 3000, 0.1) / DT_PER_DM
    from_trials = dm_step_smearing(grid, 1024, trials=grid)
    formula = dm_step_smearing(grid, 1024)
    np.testing.assert_allclose(from_trials[10:-10], formula[10:-10], rtol=0.15)
    # kept_bins uses the trials it is given
    labels = np.arange(4000) * (NYQUIST / 4000)
    expected = np.searchsorted(labels, smearing_cutoff(grid, 1024, trials=grid), side="right")
    np.testing.assert_array_equal(kept_bins(labels, grid, 1024), expected)
    # Including it never raises the cutoff above the channel smearing's alone
    smear = 2 * DM_CONSTANT * grid * ((FREQ_TOP - FREQ_BOTTOM) / 1024) / 600.0**3
    assert np.all(smearing_cutoff(grid, 1024, trials=grid) <= np.minimum(0.5 / np.hypot(TSAMP, smear), NYQUIST))


def test_kept_bins():
    labels = np.arange(1000) * (NYQUIST / 1000)
    dms = np.array([0.0, 50.0, 500.0])
    keep = kept_bins(labels, dms, 1024)
    cut = smearing_cutoff(dms, 1024, trials=dms)
    assert keep[0] == np.searchsorted(labels, cut[0], side="right")
    for k, c in zip(keep[1:], cut[1:]):
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
