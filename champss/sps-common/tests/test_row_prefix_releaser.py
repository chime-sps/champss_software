"""RowPrefixReleaser frees consumed columns of a shared array and touches nothing else."""

import os

import numpy as np
from sps_common.sm_utils import RowPrefixReleaser, empty_shared_array


def shared(shape, seed=0):
    array, shm = empty_shared_array(shape, np.float32)
    array[:] = np.random.default_rng(seed).normal(5, 1, shape)
    return array, shm


def test_unreleased_columns_keep_their_values_and_released_pages_are_freed():
    # 30001 columns: rows are not a whole number of pages, so pages straddle rows
    shape = (20, 30001)
    array, shm = shared(shape)
    original = array.copy()
    path = f"/dev/shm/{shm.name.lstrip('/')}"
    try:
        before = os.stat(path).st_blocks
        releaser = RowPrefixReleaser(array)
        for col in (5000, 17000, 26000):
            releaser.release_until(col)
        # Measured before anything reads the freed region: reading a freed tmpfs page
        # allocates a new (zero) page
        assert os.stat(path).st_blocks < 0.2 * before
        np.testing.assert_array_equal(array[:, 26000:], original[:, 26000:])
        head = array[:, :26000]
        changed = head != original[:, :26000]
        assert (head[changed] == 0).all()  # freed values read as zero, the rest untouched
        assert changed.mean() > 0.9  # all but the pages shared at row ends were freed
    finally:
        del array
        shm.close()
        shm.unlink()


def test_ordinary_arrays_are_left_alone():
    array = np.random.default_rng(1).normal(5, 1, (10, 5000)).astype(np.float32)
    original = array.copy()
    releaser = RowPrefixReleaser(array)
    releaser.release_until(4000)
    np.testing.assert_array_equal(array, original)
    assert not releaser.enabled


def test_release_rows_frees_only_those_rows():
    from sps_common.sm_utils import release_rows

    shape = (20, 30001)
    array, shm = shared(shape, seed=2)
    original = array.copy()
    path = f"/dev/shm/{shm.name.lstrip('/')}"
    try:
        before = os.stat(path).st_blocks
        assert release_rows(array, 3, 15)
        assert os.stat(path).st_blocks < 0.5 * before  # measured before reading
        np.testing.assert_array_equal(array[:3], original[:3])
        np.testing.assert_array_equal(array[15:], original[15:])
        middle = array[3:15]
        changed = middle != original[3:15]
        assert (middle[changed] == 0).all()
        assert changed.mean() > 0.95
    finally:
        del array
        shm.close()
        shm.unlink()


def test_release_rows_leaves_ordinary_arrays_alone():
    from sps_common.sm_utils import release_rows

    array = np.random.default_rng(3).normal(5, 1, (10, 5000)).astype(np.float32)
    original = array.copy()
    assert not release_rows(array, 2, 8)
    np.testing.assert_array_equal(array, original)
