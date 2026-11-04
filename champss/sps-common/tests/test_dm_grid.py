import numpy as np
import pytest
from sps_common.constants import REFERENCE_DM_SPACING
from sps_common.dm_grid import clustering_dm_spacing, snr_loss_dt_grid


@pytest.mark.parametrize(
    "nchan, max_dt, loss, expected",
    # Trial counts measured with dmtlib.generate_optimal_dm_grid
    [(1024, 4199, 0.1, 1089), (4096, 16798, 0.1, 4352)],
)
def test_snr_loss_grid_matches_library_counts(nchan, max_dt, loss, expected):
    dt = snr_loss_dt_grid(nchan, max_dt, loss)
    assert abs(len(dt) - expected) <= 1


def test_snr_loss_grid_properties():
    dt = snr_loss_dt_grid(1024, 2463, 0.1)
    assert dt[0] == 0
    assert dt[-1] == 2463
    assert np.issubdtype(dt.dtype, np.integer)
    assert np.all(np.diff(dt) >= 1)
    # Steps grow with DM once smearing dominates
    assert np.diff(dt)[-5:].mean() > np.diff(dt)[:5].mean()


def test_snr_loss_grid_prefix_is_stable_with_max_dt():
    a, b = snr_loss_dt_grid(1024, 2000, 0.1), snr_loss_dt_grid(1024, 2463, 0.1)
    np.testing.assert_array_equal(a[:-1], b[: len(a) - 1])


def test_snr_loss_grid_rejects_bad_loss():
    with pytest.raises(ValueError):
        snr_loss_dt_grid(1024, 100, 0)


def test_clustering_spacing_uniform_grid_keeps_its_spacing():
    assert clustering_dm_spacing(np.arange(10) * 0.125) == 0.125


def test_clustering_spacing_non_uniform_grid_uses_reference():
    dms = np.array([0, 0.05, 0.15, 0.2, 0.4])
    assert clustering_dm_spacing(dms) == REFERENCE_DM_SPACING
    assert clustering_dm_spacing(dms, reference=1.0) == 1.0
