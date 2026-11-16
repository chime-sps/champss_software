"""
Beamformed spectra kept as the L1 data's 8-bit quantization codes plus small tables,
giving exactly the float32 values of the float path.

L1 intensities are 5-level codes with a mean and variance per (channel, 1024-sample
chunk); the float loader turns each into raw = float32(float32(dequant5[level] * scale) +
mean) with scale = sqrt(variance), and keeps a bool mask beside it: 5 bytes a sample.
Here each sample keeps one byte:

    code = 5 * part + level                  an unmasked sample
    code = MASKED_BIT | (5 * part + level)   a masked sample (its level is kept: the
                                             spectral kurtosis cleaner uses the values
                                             of masked samples too)
    code = NO_DATA (255)                     no data: 0 before filling, masked

Blocks are BLOCK samples on each beam's own grid (block_starts, with a final sentinel at
ntime), so detrending segments and beam edges fall on block boundaries. Chunks do not (on
real data almost none start on a block boundary), so a block holds the end of one chunk
(part 0) and the starts of up to two more (parts 1 and 2: a short chunk can end inside
the block it starts in), each with its own scale and mean in `parts`.

The later steps do not rewrite samples: each records its few numbers, and expanding a
sample repeats the float path's float32 operations on it in the same order, so the
values (and so every statistic and decision computed from them) are those of the float
path, bit for bit. For an unmasked sample at index i of its detrending segment of npts
samples (`seg_start`, `seg_npts`, per block):

    v = raw - (float32(float32(i + 1) / npts) * c0 + c1)    detrend_masked_row_fast
    v = v + median                                          + local median
    v = v * factor                                          norm_channels_to_beam
    v = (v - mean1) / std1, unless v == 0                   first flattening
    v = (v - mean2) / std2, unless v == 0                   second flattening

each operation rounded to float32 (`params`: c0, c1, median, factor, fill per (row,
block); `rowp`: mean1, std1, mean2, std2 per row). The defaults (0, 0, 0, 1; 0, 1, 0, 1)
are exact identities, so a step not (yet) applied changes nothing. A masked sample's value
is the block's fill value, which the steps update with the same float32 operations.
"""

import numpy as np
from numba import njit

MASKED_BIT = np.uint8(0x80)
NO_DATA = np.uint8(255)
NLEVEL = 5
NPART = 3
BLOCK = 1024
DEQUANT = np.array([-1.8739534, -0.8309986, 0.0, 0.8309986, 1.8739534], dtype=np.float32)
# Fields of `params`, per (row, block), and of `rowp`, per row
C0, C1, MEDIAN, FACTOR, FILL = range(5)
NPARAM = 5
MEAN1, STD1, MEAN2, STD2 = range(4)


class OverlappingChunks(ValueError):
    """Chunks the layout cannot hold (overlapping, or two files starting chunks inside
    one block): the float loader must be used."""


def block_starts(ntime, beam_slices):
    """
    Block start columns, ending with ntime: each beam's [start, stop) on its own BLOCK
    grid, and any columns outside the beams on a grid from where the gap starts.
    """
    edges = {0, ntime}
    for s in beam_slices:
        edges.add(min(max(s.start, 0), ntime))
        edges.add(min(max(s.stop, 0), ntime))
    edges = sorted(edges)
    starts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        starts.extend(range(lo, hi, BLOCK))
    starts.append(ntime)
    return np.asarray(starts, dtype=np.int64)


def segments(bstart, beam_slices, nsamp):
    """
    Start column and length of the detrending segment of each block: each beam row is
    detrended in segments of nsamp samples from its start (the last one shorter), as
    detrend_masked_row_fast does. Blocks outside the beams get their own block as a
    segment (they hold no data).
    """
    nblock = bstart.shape[0] - 1
    seg_start = bstart[:-1].copy()
    seg_npts = np.diff(bstart)
    for s in beam_slices:
        g0 = int(np.searchsorted(bstart, s.start))
        g1 = int(np.searchsorted(bstart, s.stop))
        for g in range(g0, min(g1, nblock)):
            k = (bstart[g] - s.start) // nsamp
            lo = s.start + k * nsamp
            seg_start[g] = lo
            seg_npts[g] = min(lo + nsamp, s.stop) - lo
    return seg_start.astype(np.int64), np.maximum(seg_npts, 1).astype(np.int64)


def init_params(params, rowp):
    """The identity steps and zero fill."""
    params[...] = 0
    params[..., FACTOR] = 1
    rowp[...] = 0
    rowp[..., STD1] = 1
    rowp[..., STD2] = 1


@njit(cache=True, nogil=True)
def _block_of(bstart, t):
    """Index of the block holding column t."""
    lo, hi = 0, bstart.shape[0] - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if bstart[mid] <= t:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True, nogil=True)
def codes_chunk_into(levels, scale, means, good, codes, parts, bstart, start, rows, first_part):
    """
    Write one chunk piece into codes and parts from column start on; row k takes the
    chunk's channel rows[k]. levels and good are the piece's (channel, time) slices.

    first_part is the piece's part in the block it starts in, if it starts inside one:
    1, or 2 for the second piece of a file to start inside the same block.

    A sample is masked where the float loader masks it: its value is not positive, not
    finite, or not good in the chunk's own mask.

    Returns the number of (row, block) part entries that already held another chunk's
    scale or mean, plus one if first_part is out of range (see OverlappingChunks).
    """
    ntime = levels.shape[1]
    g0 = _block_of(bstart, start)
    starts_inside = start != bstart[g0]
    if starts_inside and first_part >= NPART:
        return 1
    conflicts = 0
    for k in range(rows.shape[0]):
        c = rows[k]
        s = scale[c]
        m = means[c]
        g = g0
        for j in range(ntime):
            t = start + j
            new_block = j == 0
            while t >= bstart[g + 1]:
                g += 1
                new_block = True
            part = first_part if (starts_inside and g == g0) else 0
            if new_block:
                old_s = parts[k, g, 2 * part]
                old_m = parts[k, g, 2 * part + 1]
                if (old_s != 0 or old_m != 0) and (old_s != s or old_m != m):
                    conflicts += 1
                parts[k, g, 2 * part] = s
                parts[k, g, 2 * part + 1] = m
            lev = levels[c, j]
            v = DEQUANT[lev] * s
            v = v + m
            code = np.uint8(part * NLEVEL + lev)
            if v <= 0 or not np.isfinite(v) or good[c, j] == 0:
                code |= MASKED_BIT
            codes[k, t] = code
    return conflicts


@njit(cache=True, nogil=True)
def _raw(code, parts_g):
    c = code & 0x7F
    part = c // NLEVEL
    lev = c - part * NLEVEL
    v = DEQUANT[lev] * parts_g[2 * part]
    return v + parts_g[2 * part + 1]


@njit(cache=True, nogil=True)
def _raw_table(parts_g, lut):
    """lut[c] = the raw value of code c (0..5 * NPART - 1) in a block: the same float32
    operations as the float loader, once per code instead of once per sample."""
    for part in range(NPART):
        s = parts_g[2 * part]
        m = parts_g[2 * part + 1]
        for lev in range(NLEVEL):
            v = DEQUANT[lev] * s
            lut[part * NLEVEL + lev] = v + m


@njit(cache=True, nogil=True)
def _value(code, t, g, parts, params, rowp, seg_start, seg_npts):
    """The float path's value of an unmasked sample (see the module docstring)."""
    v = _raw(code, parts[g])
    i = t - seg_start[g]
    a0 = np.float32(i + 1) / np.float32(seg_npts[g])
    fit = np.float32(a0 * params[g, C0]) + params[g, C1]
    v = v - fit
    v = v + params[g, MEDIAN]
    v = v * params[g, FACTOR]
    if v != 0:
        v = (v - rowp[MEAN1]) / rowp[STD1]
    if v != 0:
        v = (v - rowp[MEAN2]) / rowp[STD2]
    return v


@njit(cache=True, nogil=True)
def _expand_into(codes, parts, params, rowp, bstart, seg_start, seg_npts, t0, t1, out, mask, raw_masked, with_mask):
    """
    expand_row's work, block by block: each block's raw values come from a table of its
    15 codes, and the steps are skipped (exactly: they are identities) while they hold
    their defaults. out[k] is column t0 + k; mask is written if with_mask.
    """
    lut = np.empty(NPART * NLEVEL, dtype=np.float32)
    flat1 = rowp[STD1] != 1 or rowp[MEAN1] != 0
    flat2 = rowp[STD2] != 1 or rowp[MEAN2] != 0
    m1, s1, m2, s2 = rowp[MEAN1], rowp[STD1], rowp[MEAN2], rowp[STD2]
    g = _block_of(bstart, t0)
    t = t0
    while t < t1:
        end = min(bstart[g + 1], t1)
        _raw_table(parts[g], lut)
        fill = params[g, FILL]
        c0, c1 = params[g, C0], params[g, C1]
        median, factor = params[g, MEDIAN], params[g, FACTOR]
        identity = c0 == 0 and c1 == 0 and median == 0 and factor == 1 and not flat1 and not flat2
        npts = np.float32(seg_npts[g])
        base = t - seg_start[g] + 1 - t
        for tt in range(t, end):
            c = codes[tt]
            k = tt - t0
            if c >= MASKED_BIT:
                if with_mask:
                    mask[k] = True
                if raw_masked:
                    out[k] = 0 if c == NO_DATA else lut[c & 0x7F]
                else:
                    out[k] = fill
                continue
            if with_mask:
                mask[k] = False
            v = lut[c]
            if not identity:
                a0 = np.float32(base + tt) / npts
                v = v - (np.float32(a0 * c0) + c1)
                v = v + median
                v = v * factor
                if flat1 and v != 0:
                    v = (v - m1) / s1
                if flat2 and v != 0:
                    v = (v - m2) / s2
            out[k] = v
        t = end
        g += 1


@njit(cache=True, nogil=True)
def expand_row(codes, parts, params, rowp, bstart, seg_start, seg_npts, t0, t1, out, mask, raw_masked):
    """
    Values (float32) and mask (bool) of one row's columns t0:t1 into out and mask.

    raw_masked: give masked samples their loaded (raw) value, and no-data samples 0, as
    the float spectra hold them before filling (for the chunk cleaners and the data
    fraction); otherwise the block's fill value.
    """
    _expand_into(codes, parts, params, rowp, bstart, seg_start, seg_npts, t0, t1, out, mask, raw_masked, True)


@njit(cache=True, nogil=True)
def expand_rows(codes, parts, params, rowp, bstart, seg_start, seg_npts, t0, t1, out, mask, raw_masked):
    """expand_row for every row: out and mask of shape (nrow, t1 - t0)."""
    for r in range(codes.shape[0]):
        expand_row(
            codes[r], parts[r], params[r], rowp[r], bstart, seg_start, seg_npts,
            t0, t1, out[r], mask[r], raw_masked,
        )  # fmt: skip


@njit(cache=True, nogil=True)
def expand_values_into(codes, parts, params, rowp, bstart, seg_start, seg_npts, t0, t1, out, out_rows):
    """
    The values of columns t0:t1 of every row r into out[out_rows[r], :t1 - t0], as
    streaming dedispersion builds its full-width blocks.
    """
    nomask = np.empty(0, dtype=np.bool_)
    for r in range(codes.shape[0]):
        _expand_into(
            codes[r], parts[r], params[r], rowp[r], bstart, seg_start, seg_npts,
            t0, t1, out[out_rows[r]], nomask, False, False,
        )  # fmt: skip


@njit(cache=True, nogil=True)
def expand_u8_into(codes, parts, params, rowp, bstart, seg_start, seg_npts, t0, t1, out, out_rows, step):
    """
    expand_values_into for 8-bit FDMT input: each value v becomes
    clip(128 + rint(v / step), 0, 255) in the uint8 out[out_rows[r], :t1 - t0]. After
    flattening the values are in units of each channel's standard deviation, so step
    1/16 keeps ~0.03% of the variance as quantisation noise, and clips at 8 sigma.
    A row is expanded into a float32 scratch row, small enough to stay in cache.
    """
    nomask = np.empty(0, dtype=np.bool_)
    scratch = np.empty(t1 - t0, dtype=np.float32)
    inv = np.float32(1.0 / step)
    for r in range(codes.shape[0]):
        _expand_into(
            codes[r], parts[r], params[r], rowp[r], bstart, seg_start, seg_npts,
            t0, t1, scratch, nomask, False, False,
        )  # fmt: skip
        dst = out[out_rows[r]]
        for k in range(t1 - t0):
            q = np.rint(scratch[k] * inv) + 128
            if q < 0:
                q = 0
            elif q > 255:
                q = 255
            dst[k] = np.uint8(q)


@njit(cache=True, nogil=True)
def unpack_fdmt_u8(src, dst, offset, step):
    """
    dst = (src - offset) * step, row by row: an FDMT of 8-bit input (offset 128 per
    channel summed) back to the float input's units. Exact: the sums are integers below
    2**24.
    """
    off = np.float32(offset)
    s = np.float32(step)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = (src[i, j] - off) * s


@njit(cache=True, nogil=True)
def unpack_fdmt_i16(src, dst, offset, k):
    """
    dst = rint((src - offset) / k) as int16, clipped to +-32767: an FDMT of 8-bit input
    (exact integer sums) stored in half the bytes of float32. With k = nchan / 256 the
    largest possible sum, 128 * nchan, maps to 32768, so only that extreme clips; the
    values are dst * k / 16 in the float input's units (see unpack_fdmt_u8).
    """
    off = np.float64(offset)
    inv = 1.0 / k
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            v = np.rint((np.float64(src[i, j]) - off) * inv)
            if v > 32767:
                v = 32767
            elif v < -32767:
                v = -32767
            dst[i, j] = np.int16(v)


@njit(cache=True, nogil=True)
def set_masked(codes, mask, t0):
    """Mask the samples of codes[:, t0:t0 + mask.shape[1]] where mask is True."""
    for r in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[r, j]:
                c = codes[r, t0 + j]
                if c != NO_DATA:
                    codes[r, t0 + j] = c | MASKED_BIT


@njit(cache=True, nogil=True)
def count_masked(codes):
    """Masked samples (masked or no data) in a 2D codes array."""
    n = 0
    for r in range(codes.shape[0]):
        for j in range(codes.shape[1]):
            n += codes[r, j] >= MASKED_BIT
    return n


@njit(cache=True, nogil=True)
def column_max(codes, parts, bstart, t0, t1, out):
    """
    out[t - t0] = max(out[t - t0], the largest loaded (raw) value at column t), with
    no-data samples as 0: spectra.max(0) of the float loader's spectra, for the data
    fraction. Initialise out to -inf.
    """
    lut = np.empty(NPART * NLEVEL, dtype=np.float32)
    for r in range(codes.shape[0]):
        g = _block_of(bstart, t0)
        t = t0
        while t < t1:
            end = min(bstart[g + 1], t1)
            _raw_table(parts[r, g], lut)
            for tt in range(t, end):
                c = codes[r, tt]
                v = np.float32(0) if c == NO_DATA else lut[c & 0x7F]
                if v > out[tt - t0]:
                    out[tt - t0] = v
            t = end
            g += 1

@njit(cache=True, nogil=True)
def block_tables(codes, bstart, seg_start, cnt, si, si2, nmask):
    """
    Per-block tables of one row's codes, for statistics without expanding the row:
    cnt[g, c] and si[g, c], the count of unmasked samples with code c in block g and the
    sum of their positions x = t - seg_start[g] + 1 in the detrending segment; si2[g], the
    sum of x * x over the block's unmasked samples; nmask[g], its masked (or no-data)
    samples. Within a block an unmasked sample's value is then affine in x for each code.
    """
    nblock = bstart.shape[0] - 1
    for g in range(nblock):
        for c in range(NPART * NLEVEL):
            cnt[g, c] = 0
            si[g, c] = 0
        x0 = seg_start[g] - 1
        sq = np.int64(0)
        nm = 0
        for t in range(bstart[g], bstart[g + 1]):
            c = codes[t]
            if c >= MASKED_BIT:
                nm += 1
                continue
            x = np.int64(t - x0)
            cnt[g, c] += 1
            si[g, c] += x
            sq += x * x
        si2[g] = sq
        nmask[g] = nm


@njit(cache=True, nogil=True)
def binned_affine_means(codes, bstart, seg_start, a, b, fill, mean, std, flat, time_bin, out):
    """
    Means of consecutive time_bin columns of one row whose unmasked values are
    a[g, code] - b[g] * x (x as in block_tables) and masked values fill[g], then, if flat,
    (v - mean) / std for v != 0: NaN for a bin holding a zero, as
    binned_means_zero_as_nan does on the expanded row. out has one entry per whole bin.
    """
    g = 0
    for k in range(out.shape[0]):
        s = 0.0
        zero = False
        for t in range(k * time_bin, (k + 1) * time_bin):
            while t >= bstart[g + 1]:
                g += 1
            c = codes[t]
            if c >= MASKED_BIT:
                v = fill[g]
            else:
                v = a[g, c] - b[g] * (t - seg_start[g] + 1)
            if v == 0:
                zero = True
                break
            if flat:
                v = (v - mean) / std
            s += v
        out[k] = np.nan if zero else s / time_bin


class CodedSpectra:
    """
    Codes, tables and layout, over shared memory or plain arrays.

    Parameters
    ==========
    codes: np.ndarray
        uint8 (nrow, ntime).
    parts: np.ndarray
        float32 (nrow, nblock, 2 * NPART): each part's scale and mean.
    params: np.ndarray
        float32 (nrow, nblock, NPARAM): c0, c1, median, factor, fill.
    rowp: np.ndarray
        float32 (nrow, 4): mean1, std1, mean2, std2.
    bstart: np.ndarray
        int64 (nblock + 1,): block start columns, from 0, ending with ntime.
    seg_start, seg_npts: np.ndarray
        int64 (nblock,): each block's detrending segment (see segments()).
    """

    def __init__(self, codes, parts, params, rowp, bstart, seg_start, seg_npts):
        nblock = bstart.shape[0] - 1
        nrow = codes.shape[0]
        if codes.dtype != np.uint8 or codes.ndim != 2:
            raise ValueError("codes must be a 2D uint8 array.")
        if bstart[0] != 0 or bstart[-1] != codes.shape[1]:
            raise ValueError("bstart must run from 0 to ntime.")
        if (
            parts.shape != (nrow, nblock, 2 * NPART)
            or params.shape != (nrow, nblock, NPARAM)
            or rowp.shape != (nrow, 4)
            or seg_start.shape != (nblock,)
            or seg_npts.shape != (nblock,)
        ):
            raise ValueError("tables do not match codes.")
        self.codes = codes
        self.parts = parts
        self.params = params
        self.rowp = rowp
        self.bstart = bstart
        self.seg_start = seg_start
        self.seg_npts = seg_npts

    @property
    def shape(self):
        return self.codes.shape

    @property
    def layout(self):
        return self.bstart, self.seg_start, self.seg_npts

    @classmethod
    def new(cls, nrow, bstart, seg_start=None, seg_npts=None):
        """Plain arrays: all no data, identity steps."""
        nblock = bstart.shape[0] - 1
        if seg_start is None:
            seg_start, seg_npts = bstart[:-1].copy(), np.maximum(np.diff(bstart), 1)
        codes = np.full((nrow, bstart[-1]), NO_DATA, dtype=np.uint8)
        parts = np.zeros((nrow, nblock, 2 * NPART), dtype=np.float32)
        params = np.empty((nrow, nblock, NPARAM), dtype=np.float32)
        rowp = np.empty((nrow, 4), dtype=np.float32)
        init_params(params, rowp)
        return cls(codes, parts, params, rowp, bstart, seg_start, seg_npts)

    def expand(self, rows=slice(None), t0=0, t1=None, raw_masked=False):
        """(values, mask) of rows (a slice or index) and columns t0:t1 as new arrays."""
        t1 = self.shape[1] if t1 is None else t1
        sel = (self.codes[rows], self.parts[rows], self.params[rows], self.rowp[rows])
        if sel[0].ndim == 1:
            sel = tuple(a[None] for a in sel)
        out = np.empty((sel[0].shape[0], t1 - t0), dtype=np.float32)
        mask = np.empty((sel[0].shape[0], t1 - t0), dtype=bool)
        expand_rows(*sel, *self.layout, t0, t1, out, mask, raw_masked)
        return out, mask

    def blocks_in(self, t0, t1):
        """Indices g0, g1 of the blocks inside columns [t0, t1), which must be block edges."""
        g0 = int(np.searchsorted(self.bstart, t0))
        g1 = int(np.searchsorted(self.bstart, t1))
        if self.bstart[g0] != t0 or self.bstart[g1] != t1:
            raise ValueError("t0 and t1 must be block edges.")
        return g0, g1


def allocate_shared(nrow, bstart, seg_start=None, seg_npts=None):
    """
    Codes (all no data), parts, params and rowp in new shared memory. Returns (spec,
    blocks): spec is what workers pass to attach(); blocks are the SharedMemory objects,
    for the creator to close and unlink (after detach()).
    """
    from multiprocessing import shared_memory

    nblock = bstart.shape[0] - 1
    ntime = int(bstart[-1])
    if seg_start is None:
        seg_start, seg_npts = bstart[:-1].copy(), np.maximum(np.diff(bstart), 1)
    sizes = (nrow * ntime, nrow * nblock * 2 * NPART * 4, nrow * nblock * NPARAM * 4, nrow * 4 * 4)
    blocks = [shared_memory.SharedMemory(create=True, size=max(s, 1)) for s in sizes]
    spec = dict(
        names=tuple(b.name for b in blocks), nrow=nrow, bstart=bstart,
        seg_start=seg_start, seg_npts=seg_npts,
    )  # fmt: skip
    cs = _arrays(spec, blocks)
    cs.codes[:] = NO_DATA
    cs.parts[:] = 0
    init_params(cs.params, cs.rowp)
    return spec, blocks


def _arrays(spec, blocks):
    bstart = spec["bstart"]
    nrow, nblock, ntime = spec["nrow"], bstart.shape[0] - 1, int(bstart[-1])
    codes = np.ndarray((nrow, ntime), dtype=np.uint8, buffer=blocks[0].buf)
    parts = np.ndarray((nrow, nblock, 2 * NPART), dtype=np.float32, buffer=blocks[1].buf)
    params = np.ndarray((nrow, nblock, NPARAM), dtype=np.float32, buffer=blocks[2].buf)
    rowp = np.ndarray((nrow, 4), dtype=np.float32, buffer=blocks[3].buf)
    cs = CodedSpectra(codes, parts, params, rowp, bstart, spec["seg_start"], spec["seg_npts"])
    # The arrays need their shared memory mapped as long as they live
    cs._blocks = blocks
    return cs


# Shared memory attached by name in this process (pool workers, or the main process
# when the pool runs in it): kept mapped while arrays over it may live
_ATTACHED = {}


def attach(spec):
    """
    The CodedSpectra of a spec from allocate_shared, in this process. Each block is
    attached once per process and kept (a pool worker runs many small tasks on the same
    blocks); the creator calls detach() before unlinking.
    """
    from multiprocessing import shared_memory

    blocks = []
    for name in spec["names"]:
        shm = _ATTACHED.get(name)
        if shm is None:
            shm = _ATTACHED[name] = shared_memory.SharedMemory(name=name)
        blocks.append(shm)
    return _arrays(spec, blocks)


def detach(names):
    """Forget (and close, if no array uses them any more) attachments made by attach()."""
    for name in names:
        shm = _ATTACHED.pop(name, None)
        if shm is not None:
            try:
                shm.close()
            except BufferError:
                pass
