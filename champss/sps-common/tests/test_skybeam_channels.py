"""SkyBeam holding only some channels."""

import numpy as np
import pytest
from sps_common.interfaces.beamformer import SkyBeam


def skybeam(spectra, nchan, channels=None):
    return SkyBeam(spectra=spectra, ra=10, dec=20, nchan=nchan, ntime=spectra.shape[1],
                   maxdm=100, beam_row=120, utc_start=1.7e9, channels=channels)


def test_full_spectra_puts_held_rows_back_and_zeros_the_rest():
    held = np.arange(3 * 5, dtype=np.float32).reshape(3, 5) + 1
    sb = skybeam(held, 8, channels=np.array([1, 4, 6]))
    full = sb.full_spectra()
    assert full.shape == (8, 5)
    np.testing.assert_array_equal(full[[1, 4, 6]], held)
    assert not full[[0, 2, 3, 5, 7]].any()


def test_every_channel_held_returns_the_spectra_itself():
    spectra = np.ones((8, 5), np.float32)
    assert skybeam(spectra, 8).full_spectra() is spectra


def test_shape_must_match_the_channels():
    with pytest.raises(ValueError):
        skybeam(np.ones((4, 5), np.float32), 8, channels=np.array([1, 2, 3]))
