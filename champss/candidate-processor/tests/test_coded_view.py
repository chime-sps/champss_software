"""The feature generator's view of 16-bit coded spectra (PowerSpectra.quant_step)."""

import attr
import numpy as np
from candidate_processor import feature_generator as fg


def test_coded_spectra_view_reads_power():
    codes = np.arange(24, dtype=np.uint16).reshape(4, 6)
    view = fg._CodedSpectraView(codes, 0.25)
    power = codes.astype(np.float32) * np.float32(0.25)
    assert view.shape == codes.shape and len(view) == 4
    np.testing.assert_array_equal(view[2], power[2])
    np.testing.assert_array_equal(view[2][[1, 5]], power[2, [1, 5]])
    rows, bins = slice(1, 3), np.array([[0, 2], [4, 5]])
    np.testing.assert_array_equal(view[rows, bins], power[rows, bins])
    assert view[1].dtype == np.float32


def test_data_getters_are_attrs_classes():
    # The view must not take a decorator meant for DataGetter
    assert attr.has(fg.DataGetter) and not attr.has(fg._CodedSpectraView)
