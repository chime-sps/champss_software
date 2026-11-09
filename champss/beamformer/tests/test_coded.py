"""8-bit codes and tables (beamformer.coded) against the float32 loader."""

from multiprocessing import shared_memory

import numpy as np
import pytest
from beamformer import coded
from beamformer.skybeam import SkyBeamFormer
from sps_common.constants import TSAMP
from spshuff import l1_io

NFREQ, CHUNK = 64, 1024
FPGA_PER_CHUNK = 393216  # 1024 samples of 0.98304 ms in 2.56 us FPGA counts
FRAME0_NANO = 1_700_000_000 * 10**9
CDF = 16384 // NFREQ


def make_file(path, fpga0s, seed=0):
    """An L1 file with one chunk per entry of fpga0s."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # Chunk(quantize_now) uses the global state
    chunks = []
    for fpga0 in fpga0s:
        dat = rng.standard_normal((NFREQ, CHUNK))
        means = rng.uniform(0.5, 8, NFREQ)
        means[::9] = 0.3  # low means: some values are not positive, so masked
        variance = rng.uniform(0.5, 3, NFREQ)
        bad = rng.random((NFREQ, CHUNK)) < 0.05
        bad[5] = True  # a fully flagged channel
        header = l1_io.ChunkHeader.from_fields(NFREQ, CHUNK, FRAME0_NANO, 4096 * 128, fpga0)
        chunks.append(l1_io.Chunk(header, means, variance, bad, dat, quantize_now=True))
    fh = l1_io.FileHeader.from_fields(1000, 5, 0, 1)
    with open(path, "wb") as f:
        l1_io.IntensityFile(fh, chunks).to_file(file=f)


def load_float(path, shape, utc_start, windows, channels=None):
    nrow, ntime = shape
    sm_s = shared_memory.SharedMemory(create=True, size=nrow * ntime * 4)
    sm_m = shared_memory.SharedMemory(create=True, size=nrow * ntime)
    try:
        spectra = np.ndarray(shape, np.float32, buffer=sm_s.buf)
        mask = np.ndarray(shape, bool, buffer=sm_m.buf)
        spectra[:] = 0
        mask[:] = True
        slices = []
        for b0, b1 in windows:
            slices += SkyBeamFormer.read_to_shared_spectra(
                None, sm_s.name, sm_m.name, shape, np.float32, CDF, utc_start, str(path),
                b0, b1, channels=channels,
            )  # fmt: skip
        out = spectra.copy(), mask.copy(), slices
        del spectra, mask
        return out
    finally:
        for m in (sm_s, sm_m):
            m.close()
            m.unlink()


def load_coded(path, shape, utc_start, windows, channels=None):
    nrow, ntime = shape
    bstart = coded.block_starts(ntime, [slice(b0, b1) for b0, b1 in windows])
    spec, blocks = coded.allocate_shared(nrow, bstart)
    try:
        slices = []
        for b0, b1 in windows:
            slices += SkyBeamFormer.read_to_coded(
                None, spec, CDF, utc_start, str(path), b0, b1, channels=channels
            )
        cs = coded._arrays(spec, blocks)
        values, mask = cs.expand(raw_masked=True)
        result = coded.CodedSpectra(
            cs.codes.copy(), cs.parts.copy(), cs.params.copy(), cs.rowp.copy(), *cs.layout
        )
        del cs
        return values, mask, result, slices
    finally:
        for b in blocks:
            b.close()
            b.unlink()


def compare(path, shape, utc_start, windows, channels=None):
    spectra, mask, slices = load_float(path, shape, utc_start, windows, channels)
    values, cmask, cs, cslices = load_coded(path, shape, utc_start, windows, channels)
    assert [(s.start, s.stop) for s in slices] == [(s.start, s.stop) for s in cslices]
    assert slices
    np.testing.assert_array_equal(cmask, mask)
    # Raw values everywhere, masked samples included (the SK cleaner uses them)
    np.testing.assert_array_equal(values.view(np.uint32), spectra.view(np.uint32))
    assert mask.any() and (~mask).any()
    return spectra, mask, cs


@pytest.mark.parametrize("offset", [0, 300, 1023])
def test_load_matches_float_loader(tmp_path, offset):
    # Two runs of contiguous chunks with a gap between, starting `offset` samples into
    # the spectra, and a beam window that cuts the first and last chunks
    path = tmp_path / "f.dat"
    make_file(path, [i * FPGA_PER_CHUNK for i in [0, 1, 2, 3, 6, 7, 8]])
    utc_start = FRAME0_NANO * 1e-9 - offset * TSAMP
    ntime = 10 * CHUNK
    compare(path, (NFREQ, ntime), utc_start, [(500, ntime - 700)])


def test_load_held_rows(tmp_path):
    path = tmp_path / "f.dat"
    make_file(path, [i * FPGA_PER_CHUNK for i in range(4)], seed=3)
    channels = np.array([1, 2, 7, 40, 63])
    utc_start = FRAME0_NANO * 1e-9 - 123 * TSAMP
    compare(path, (len(channels), 5 * CHUNK), utc_start, [(0, 5 * CHUNK)], channels)


def test_two_beams_meeting_inside_a_block(tmp_path):
    """Beam windows meeting off the grid: each beam has its own block grid."""
    path = tmp_path / "f.dat"
    make_file(path, [i * FPGA_PER_CHUNK for i in range(3)], seed=5)
    utc_start = FRAME0_NANO * 1e-9 - 200 * TSAMP
    ntime = 4 * CHUNK
    _, _, cs = compare(path, (NFREQ, ntime), utc_start, [(0, 1500), (1500, ntime)])
    assert 1500 in cs.bstart


def test_overlapping_chunks_are_refused(tmp_path):
    path = tmp_path / "f.dat"
    # The second chunk starts 300 samples into the first
    make_file(path, [0, 300 * FPGA_PER_CHUNK // 1024])
    with pytest.raises(coded.OverlappingChunks):
        load_coded(path, (NFREQ, 4 * CHUNK), FRAME0_NANO * 1e-9 - 100 * TSAMP, [(0, 4 * CHUNK)])


def test_block_starts():
    b = coded.block_starts(5000, [slice(0, 2300), slice(2300, 4800)])
    assert b[0] == 0 and b[-1] == 5000
    assert list(b[:4]) == [0, 1024, 2048, 2300]
    assert list(b[4:]) == [3324, 4348, 4800, 5000]


def test_expand_repeats_the_float32_steps():
    rng = np.random.default_rng(1)
    nrow, ntime = 3, 5000
    beams = [slice(0, 2300), slice(2300, 5000)]
    bstart = coded.block_starts(ntime, beams)
    seg_start, seg_npts = coded.segments(bstart, beams, 2048)
    cs = coded.CodedSpectra.new(nrow, bstart, seg_start, seg_npts)
    cs.codes[:] = rng.integers(0, 10, (nrow, ntime)).astype(np.uint8)
    cs.codes[:, ::7] |= coded.MASKED_BIT
    cs.codes[:, ::11] = coded.NO_DATA
    cs.parts[..., 0], cs.parts[..., 1] = 2, 10
    cs.parts[..., 2], cs.parts[..., 3] = 3, 20
    p = cs.params
    p[..., coded.C0] = np.float32(0.37)
    p[..., coded.C1] = np.float32(-1.25)
    p[..., coded.MEDIAN] = np.float32(13.5)
    p[..., coded.FACTOR] = np.float32(0.91)
    p[..., coded.FILL] = np.float32(7)
    cs.rowp[:, coded.MEAN1], cs.rowp[:, coded.STD1] = np.float32(12.1), np.float32(2.3)
    cs.rowp[:, coded.MEAN2], cs.rowp[:, coded.STD2] = np.float32(0.01), np.float32(1.02)
    values, mask = cs.expand()
    # The float path's float32 steps, sample by sample
    f32 = np.float32
    c = cs.codes & 0x7F
    part, lev = c // 5, c % 5
    scale = np.where(part == 0, f32(2), f32(3)).astype(np.float32)
    mean = np.where(part == 0, f32(10), f32(20)).astype(np.float32)
    raw = (coded.DEQUANT[lev] * scale).astype(np.float32) + mean
    t = np.arange(ntime)
    g = np.searchsorted(bstart, t, "right") - 1
    a0 = (t - seg_start[g] + 1).astype(np.float32) / seg_npts[g].astype(np.float32)
    fit = (a0 * f32(0.37)).astype(np.float32) + f32(-1.25)
    v = (raw - fit).astype(np.float32)
    v = (v + f32(13.5)).astype(np.float32)
    v = (v * f32(0.91)).astype(np.float32)
    v = np.where(v != 0, (v - f32(12.1)) / f32(2.3), v).astype(np.float32)
    v = np.where(v != 0, (v - f32(0.01)) / f32(1.02), v).astype(np.float32)
    masked = cs.codes >= coded.MASKED_BIT
    expected = np.where(masked, f32(7), v)
    np.testing.assert_array_equal(values.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(mask, masked)
    # Raw values of masked samples
    raw_values, _ = cs.expand(raw_masked=True)
    raw32 = np.where(cs.codes == coded.NO_DATA, f32(0), raw)
    np.testing.assert_array_equal(raw_values[masked].view(np.uint32), raw32[masked].view(np.uint32))
    # A part of the columns, and the full-width block expansion dedispersion uses
    v2, _ = cs.expand(rows=1, t0=1000, t1=2400)
    np.testing.assert_array_equal(v2[0], values[1, 1000:2400])
    out = np.zeros((6, 1400), np.float32)
    coded.expand_values_into(
        cs.codes, cs.parts, cs.params, cs.rowp, *cs.layout, 1000, 2400, out, np.array([5, 3, 0])
    )
    np.testing.assert_array_equal(out[[5, 3, 0]], values[:, 1000:2400])
    assert not out[[1, 2, 4]].any()


def test_segments():
    beams = [slice(0, 2300), slice(2300, 9000)]
    bstart = coded.block_starts(9500, beams)
    seg_start, seg_npts = coded.segments(bstart, beams, 4096)
    for g in range(len(bstart) - 1):
        t = bstart[g]
        if t < 2300:
            assert (seg_start[g], seg_npts[g]) == (0, 2300)
        elif t < 2300 + 4096:
            assert (seg_start[g], seg_npts[g]) == (2300, 4096)
        elif t < 9000:
            assert (seg_start[g], seg_npts[g]) == (6396, 9000 - 6396)


def test_set_masked_and_count():
    bstart = coded.block_starts(2048, [slice(0, 2048)])
    cs = coded.CodedSpectra.new(2, bstart)
    cs.codes[:, :100] = 3
    mask = np.zeros((2, 50), bool)
    mask[0, 10] = mask[1, 49] = True
    coded.set_masked(cs.codes, mask, 20)
    assert cs.codes[0, 30] == 3 | coded.MASKED_BIT and cs.codes[1, 69] == 3 | coded.MASKED_BIT
    assert coded.count_masked(cs.codes) == 2 * 2048 - 200 + 2
