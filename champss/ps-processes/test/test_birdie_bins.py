"""bins_in_ranges against the per-birdie comparisons it replaced."""

import numpy as np
import pytest
from ps_processes.processes.ps import bins_in_ranges


def reference(freq_labels, lefts, rights):
    out = []
    for left, right in zip(lefts, rights):
        out.extend(np.where((freq_labels >= left) & (freq_labels <= right))[0].tolist())
    return out


FREQS = np.fft.rfftfreq(2**20, d=0.00098304)[:-1]


@pytest.mark.parametrize("edge_dtype", [np.float64, np.float32])
def test_random_ranges_match(edge_dtype):
    rng = np.random.default_rng(0)
    centres = rng.uniform(-1, FREQS[-1] + 1, 2000)
    widths = rng.exponential(0.01, 2000)
    lefts = (centres - widths).astype(edge_dtype)
    rights = (centres + widths).astype(edge_dtype)
    assert bins_in_ranges(FREQS, lefts, rights) == reference(FREQS, lefts, rights)


def test_edges_on_bins_overlaps_and_odd_ranges_match():
    lefts = [FREQS[10], FREQS[10], FREQS[20], 5.0, np.nan, 3.0, FREQS[-1], -np.inf, 7.0]
    rights = [FREQS[12], FREQS[11], FREQS[20], 4.0, 6.0, np.nan, np.inf, FREQS[2], 7.0]
    assert bins_in_ranges(FREQS, lefts, rights) == reference(FREQS, lefts, rights)


def test_record_array_fields_and_unequal_lengths_match():
    birdies = np.zeros(3, dtype=[("left", float), ("right", float)])
    birdies["left"], birdies["right"] = [1.0, 2.0, 30.0], [1.001, 2.5, 30.0001]
    for lefts, rights in [(birdies["left"], birdies["right"]),
                          (birdies["left"], birdies["right"][:2]),
                          ([], [])]:
        assert bins_in_ranges(FREQS, lefts, rights) == reference(FREQS, lefts, rights)


def test_unsorted_labels_fall_back_to_comparisons():
    labels = FREQS[:1000][::-1].copy()
    lefts, rights = [0.5, 3.0], [1.0, 3.2]
    assert bins_in_ranges(labels, lefts, rights) == reference(labels, lefts, rights)
