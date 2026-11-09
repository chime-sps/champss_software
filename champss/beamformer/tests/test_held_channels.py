"""held_channels reads chunk summaries from L1 files without decoding them."""

import numpy as np
import pytest
from beamformer.skybeam import _chunk_summaries, held_channels
from spshuff import l1_io


def write_file(path, nfreq, dead, nchunks=3, seed=0):
    """An L1 intensity file whose channels in dead have zero mean and variance."""
    rng = np.random.default_rng(seed)
    fh = l1_io.FileHeader.from_fields(120, 1, 1.7e9, 1.7e9 + 10)
    chunks = []
    for i in range(nchunks):
        data = rng.normal(0, 1, (nfreq, 1024))
        means = rng.uniform(1, 5, nfreq)
        variance = rng.uniform(0.5, 2, nfreq)
        means[dead[i]] = 0
        variance[dead[i]] = 0
        header = l1_io.ChunkHeader.from_fields(nfreq, 1024, 0, 384 * 1024, i * 384 * 1024)
        mask = rng.random((nfreq, 1024)) > 0.1
        chunks.append(l1_io.Chunk(header, means, variance, mask, data, quantize_now=True))
    with open(path, "wb") as f:
        l1_io.IntensityFile(fh, chunks=chunks).to_file(f)


def test_summaries_match_a_full_read(tmp_path):
    path = tmp_path / "a.dat"
    write_file(path, 64, [[1, 2], [2], [2, 5]])
    with open(path, "rb") as f:
        full = l1_io.IntensityFile.from_file(f, shape=(64, None)).get_chunks()
    fast = _chunk_summaries(path)
    assert len(fast) == len(full) == 3
    for chunk, (nfreq, means, variance) in zip(full, fast):
        assert nfreq == chunk.chunk_header.nfreq
        np.testing.assert_array_equal(means, chunk.means)
        np.testing.assert_array_equal(variance, chunk.variance)


def test_only_channels_dead_in_every_chunk_of_every_file_are_left_out(tmp_path):
    write_file(tmp_path / "a.dat", 64, [[1, 2, 7], [2, 7], [2, 5, 7]])
    write_file(tmp_path / "b.dat", 64, [[2, 7, 9]] * 3, seed=1)
    (tmp_path / "empty.dat").write_bytes(b"")  # skipped, as the loader skips it
    files = [tmp_path / n for n in ("a.dat", "b.dat", "empty.dat")]
    held = held_channels(files, 64)
    np.testing.assert_array_equal(held, np.setdiff1d(np.arange(64), [2, 7]))


@pytest.mark.parametrize("case", ["other_nchan", "no_dead", "all_dead"])
def test_full_layout_when_compaction_does_not_apply(tmp_path, case):
    if case == "other_nchan":
        write_file(tmp_path / "a.dat", 128, [[3]] * 3)
    elif case == "no_dead":
        write_file(tmp_path / "a.dat", 64, [[]] * 3)
    else:
        write_file(tmp_path / "a.dat", 64, [list(range(64))] * 3)
    assert held_channels([tmp_path / "a.dat"], 64) is None


def test_a_chunk_read_from_a_file_holds_the_decoded_array_and_others_copy(tmp_path):
    path = tmp_path / "a.dat"
    write_file(path, 64, [[1]] * 3)
    decoded = []
    original = l1_io.decode

    def recording_decode(*args):
        decoded.append(original(*args))
        return decoded[-1]

    l1_io.decode = recording_decode
    try:
        with open(path, "rb") as f:
            chunk = l1_io.IntensityFile.from_file(f, shape=(64, None)).get_chunks()[0]
    finally:
        l1_io.decode = original
    assert np.shares_memory(chunk.data, decoded[0])  # no copy on the way in

    # The constructor's default still copies what it is given
    data = np.zeros((64, 1024), np.float32)
    header = l1_io.ChunkHeader.from_fields(64, 1024, 0, 0, 0)
    copied = l1_io.Chunk(header, np.zeros(64), np.zeros(64), np.ones((64, 1024), bool), data)
    assert not np.shares_memory(copied.data, data)
