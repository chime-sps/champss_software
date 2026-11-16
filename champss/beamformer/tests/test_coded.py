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


def _synthetic_shared(seed=3, nrow=4, ntime=9000, nsamp=2048):
    """Random codes over shared memory: two beams, a slope in each row, some masking."""
    rng = np.random.default_rng(seed)
    beams = [slice(0, 4100), slice(4100, 8800)]
    bstart = coded.block_starts(ntime, beams)
    seg_start, seg_npts = coded.segments(bstart, beams, nsamp)
    spec, blocks = coded.allocate_shared(nrow, bstart, seg_start, seg_npts)
    cs = coded._arrays(spec, blocks)
    lev = rng.integers(0, 5, (nrow, ntime))
    part = np.zeros_like(lev)
    cs.codes[:] = (5 * part + lev).astype(np.uint8)
    cs.codes[rng.random((nrow, ntime)) < 0.2] |= coded.MASKED_BIT
    cs.codes[:, 8800:] = coded.NO_DATA
    cs.codes[2, 4100:8800] |= coded.MASKED_BIT  # one beam fully masked in one row
    nblock = len(bstart) - 1
    # Per block a scale and a mean drifting along the row (a trend for the fits)
    cs.parts[:, :, 0] = rng.uniform(1.5, 2.5, (nrow, nblock))
    cs.parts[:, :, 1] = 100 + np.linspace(0, 6, nblock)[None] * rng.uniform(0.5, 1.5, (nrow, 1))
    return spec, blocks, cs, beams, nsamp


def test_block_tables_match_brute_force():
    spec, blocks, cs, beams, nsamp = _synthetic_shared()
    try:
        bstart, seg_start, _ = cs.layout
        nblock = len(bstart) - 1
        for r in range(cs.shape[0]):
            cnt = np.empty((nblock, 15), np.int32)
            si = np.empty((nblock, 15), np.int64)
            si2 = np.empty(nblock, np.int64)
            nmask = np.empty(nblock, np.int32)
            coded.block_tables(cs.codes[r], bstart, seg_start, cnt, si, si2, nmask)
            for g in range(nblock):
                c = cs.codes[r, bstart[g] : bstart[g + 1]]
                x = np.arange(bstart[g], bstart[g + 1]) - seg_start[g] + 1
                good = c < coded.MASKED_BIT
                assert nmask[g] == np.count_nonzero(~good)
                assert si2[g] == (x[good] ** 2).sum()
                for k in range(15):
                    sel = good & (c == k)
                    assert cnt[g, k] == sel.sum() and si[g, k] == x[sel].sum()
    finally:
        del cs
        coded.detach([b.name for b in blocks])
        for b in blocks:
            b.close()
            b.unlink()


def test_weighted_median_and_line_fit():
    from beamformer.skybeam import _line_fit, _weighted_median
    from scipy import linalg
    from sps_common.fast_select import masked_median

    rng = np.random.default_rng(5)
    for n in (1, 2, 7, 10):
        values = rng.choice(np.float32([1.5, 2.25, -3, 7, 7.5]), size=5, replace=False)
        counts = rng.integers(0, 4, 5)
        counts[0] += n % 2 + 1
        full = np.repeat(values, counts).astype(np.float32)
        assert _weighted_median(values, counts) == masked_median(full, np.zeros(full.size, bool))
    # The line fit from block tables against lstsq on the samples
    npts = 3000
    x = np.arange(1, npts + 1)
    good = rng.random(npts) < 0.7
    y = (5 + 2.5 * x / npts + rng.standard_normal(npts)).astype(np.float32)
    a = np.stack([x / npts, np.ones(npts)], 1)
    c0, c1 = linalg.lstsq(a[good], y[good])[0]
    # One "block" per sample: code 0 with that sample's value
    raw = y[:, None].astype(np.float64)
    fit = _line_fit(good[:, None].astype(np.int32), (x * good)[:, None], x**2 * good, raw, float(npts))
    np.testing.assert_allclose(fit, (c0, c1), rtol=1e-5, atol=1e-5)


def test_fill_norm_flatten_tables_close_to_exact():
    """The table statistics against the exact ones on random rows: medians equal,
    everything else at float32 rounding level."""
    from types import SimpleNamespace

    spec, blocks, cs, beams, nsamp = _synthetic_shared()
    bf = SimpleNamespace(detrend_nsamp=nsamp, beam_to_normalise=0)
    try:
        for r in range(cs.shape[0]):
            saved = (cs.codes[r].copy(), cs.params[r].copy(), cs.rowp[r].copy())
            ex = SkyBeamFormer.coded_fill_norm_flatten(bf, spec, beams, 1024, r)
            ex_state = (cs.codes[r].copy(), cs.params[r].copy(), cs.rowp[r].copy())
            cs.codes[r], cs.params[r], cs.rowp[r] = saved
            tb = SkyBeamFormer.coded_fill_norm_flatten_tables(bf, spec, beams, 1024, r)
            assert (ex["masked"], ex["fully"], ex["status2"]) == (tb["masked"], tb["fully"], tb["status2"])
            np.testing.assert_array_equal(cs.codes[r], ex_state[0])
            p, q = ex_state[1], cs.params[r]
            np.testing.assert_array_equal(q[:, coded.MEDIAN], p[:, coded.MEDIAN])
            np.testing.assert_allclose(q[:, coded.C0], p[:, coded.C0], atol=2e-3)
            np.testing.assert_allclose(q[:, coded.C1], p[:, coded.C1], rtol=1e-5)
            np.testing.assert_allclose(q[:, coded.FACTOR], p[:, coded.FACTOR], rtol=2e-3)
            np.testing.assert_allclose(q[:, coded.FILL], p[:, coded.FILL], atol=2e-3)
            np.testing.assert_allclose(cs.rowp[r], ex_state[2], rtol=1e-4)
            np.testing.assert_allclose((tb["mean2"], tb["std2"]), (ex["mean2"], ex["std2"]), rtol=1e-4, atol=1e-5)
            # Bins: NaN where a bin holds no data or a fully masked beam. The exact path
            # also gives NaN wherever masked samples flatten to exactly zero (their fill
            # equal to the float32 row mean), which the tables do not reproduce
            tb_nan, ex_nan = np.isnan(tb["binned"]), np.isnan(ex["binned"])
            assert not (tb_nan & ~ex_nan).any()
            assert r != 2 or tb_nan[4:].all()  # row 2: the second beam fully masked
            # The tables give the cleaner the bin means before flattening: the exact
            # path's flattened ones mapped back with its own mean and std
            ok = ~ex_nan
            m, s = ex_state[2][coded.MEAN1], ex_state[2][coded.STD1]
            unflattened = m + s * ex["binned"][ok].astype(np.float64)
            np.testing.assert_allclose(tb["binned"][ok], unflattened, rtol=1e-4)
    finally:
        del cs
        coded.detach([b.name for b in blocks])
        for b in blocks:
            b.close()
            b.unlink()
