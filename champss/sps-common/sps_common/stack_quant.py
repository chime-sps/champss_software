"""
16-bit integer storage for power spectra stacks that are added to daily.

Each day of power is rounded once to a multiple of `step`, and days are then summed as
integers, so adding a day loses nothing further. Decoded values are divided by the known
mean bias of rounding exponentially distributed power, so a noise bin still has a mean of
1 per day.

On disk, each integer is stored relative to the expected noise sum of the stack (number
of days / step) and zigzag-mapped, so values near the noise level become small unsigned
numbers. Rows are then byte-shuffled and zstd-compressed, which keeps the file close to
the information content of the stack even when sums are well above 255. Values that do
not fit in 16 bits after this mapping are kept exactly in a sparse overflow list.

Reading these files needs the zstd HDF5 filter, which importing this module registers.

HDF5 runs its filters one chunk at a time, so with the zstandard package rows are instead
shuffled and compressed (or decompressed and unshuffled) in threads, and the chunks written
(or read) whole with write_direct_chunk (read_direct_chunk). The chunks are the same bytes
the filters would make, so the data are stored identically (HDF5 places the chunks in the
file differently) and readable with or without zstandard.
"""

import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import hdf5plugin
import numpy as np
from sps_common.cpus import available_cpus
from numba import njit

try:
    import zstandard
except ImportError:  # the HDF5 filters are used instead
    zstandard = None

CODE_MAX = np.iinfo(np.uint16).max
# Level 1 gives the same size as 3 on these stacks (noise does not reward effort), faster
ZSTD_LEVEL = 1
# Integer work is done in int32, which holds any sum below ~5e8 power at step 0.25
CODE_LIMIT = 2**30
READ_BLOCK_ROWS = 32
# Threads that compress or decompress rows; 0 leaves it to the HDF5 filters
THREADS = min(8, available_cpus())
# Rows per threaded task when writing
WRITE_BLOCK_ROWS = 8
# HDF5 filter ids: byte shuffle, and the registered zstd filter
H5Z_SHUFFLE = 2
H5Z_ZSTD = 32015


def quant_bias(step):
    """Mean of rounded unit-mean exponential noise, per day."""
    return step / (2 * np.sinh(step / 2))


def round_power(power, step):
    """Round power to multiples of step, keeping it as a float."""
    return np.rint(power / step) * step


def _zigzag(x):
    """Map signed int32 to unsigned: 0, -1, 1, -2, ... -> 0, 1, 2, 3, ..."""
    return (x << 1) ^ (x >> 31)


def _unzigzag(z):
    return (z >> 1) ^ -(z & 1)


# The compiled loops below do in one pass what the numpy expressions above do in
# several, with the same arithmetic: power is rounded in float32 (numpy 1.x keeps a
# float32 array divided by a Python float in float32), and an int32 code times a float32
# scale is computed in float64 and stored as float32, as np.multiply does.


@njit(cache=True, nogil=True)
def _encode_row(spectrum, inv_step, offset, limit, z, over, keep):
    """
    zigzag(rint(spectrum * inv_step) - offset) into z, as write_quantized stores it,
    with bins from keep on taken as zero power.

    Bins whose code does not fit in 16 bits get 0 and are listed in over. Returns how
    many there are, or -1 if a value is at least limit in size or is not finite.
    """
    n_over = 0
    for j in range(spectrum.shape[0]):
        v = spectrum[j] if j < keep else np.float32(0)
        r = np.rint(v * inv_step)
        if not abs(r) < limit:
            return -1
        x = np.int64(r) - offset
        code = (x << 1) ^ (x >> 63)
        if code > CODE_MAX:
            over[n_over] = j
            n_over += 1
            code = 0
        z[j] = code
    return n_over


@njit(cache=True, nogil=True)
def _decode_block(raw, offset, scale, ov_row, ov_bin, ov_code, out, add, step):
    """
    Decode stored rows into out, or with add, set out to its own power rounded to step
    plus the decoded rows. ov_* are the overflow entries of these rows, rows relative.
    """
    # Values at overflow bins first, while out still holds the power to be rounded
    ov_val = np.empty(ov_row.shape[0], dtype=np.float32)
    for m in range(ov_row.shape[0]):
        v = np.float32(np.float64(np.int32(ov_code[m])) * np.float64(scale))
        if add:
            v = np.rint(out[ov_row[m], ov_bin[m]] / step) * step + v
        ov_val[m] = v
    for r in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            z = np.int64(raw[r, j])
            v = np.float32(np.float64(((z >> 1) ^ -(z & 1)) + offset) * np.float64(scale))
            if add:
                v = np.rint(out[r, j] / step) * step + v
            out[r, j] = v
    for m in range(ov_row.shape[0]):
        out[ov_row[m], ov_bin[m]] = ov_val[m]


@njit(cache=True, nogil=True)
def _shuffle2(z, out):
    """HDF5's shuffle filter for 2-byte values: all low bytes, then all high bytes."""
    n = z.shape[0]
    for j in range(n):
        v = z[j]
        out[j] = v & 0xFF
        out[n + j] = v >> 8


@njit(cache=True, nogil=True)
def _unshuffle2(b, out):
    """The inverse of _shuffle2."""
    n = out.shape[0]
    for j in range(n):
        out[j] = np.uint16(b[j]) | (np.uint16(b[n + j]) << 8)


def _threads(threads):
    """The number of threads to use, or 0 for the HDF5 filters."""
    if zstandard is None:
        return 0
    return THREADS if threads is None else max(int(threads), 0)


def _direct_chunks(dataset):
    """Whether dataset's chunks are single rows, shuffled then zstd-compressed."""
    if dataset.chunks != (1, dataset.shape[1]) or dataset.dtype != np.uint16:
        return False
    plist = dataset.id.get_create_plist()
    filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
    return filters == [H5Z_SHUFFLE, H5Z_ZSTD]


class _Local(threading.local):
    """Per-thread zstd contexts and work buffers."""

    def compressor(self):
        if not hasattr(self, "_c"):
            self._c = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
        return self._c

    def decompressor(self):
        if not hasattr(self, "_d"):
            self._d = zstandard.ZstdDecompressor()
        return self._d

    def buffer(self, name, n, dtype):
        b = getattr(self, name, None)
        if b is None or b.shape[0] < n:
            b = np.empty(n, dtype=dtype)
            setattr(self, name, b)
        return b[:n]


def _ordered(executor, tasks, window):
    """Results of tasks (callables) run on executor, in order, at most window at once."""
    pending = deque()
    for task in tasks:
        pending.append(executor.submit(task))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def write_quantized(h5f, spectra, step, num_days, threads=None, keep=None):
    """
    Write a 2D array of (dm, freq) power, already a sum of rounded days or a single day.

    Parameters
    =========
    h5f: h5py.File
        Open file to write the "power spectra" and "overflow *" datasets into.
    spectra: np.ndarray
        The power spectra, multiples of step.
    step: float
        The rounding step.
    num_days: int
        The number of days in the stack, which sets the expected noise level.
    threads: int or None
        Threads compressing rows (default THREADS); 0 uses the HDF5 filters.
    keep: array_like or None
        Number of leading bins to store in each row; the rest are stored as zero power,
        which compresses to almost nothing (see sps_common.freq_cut). Saved as the
        "kept bins" dataset. None stores every bin.
    """
    offset = int(round(num_days / step))
    dataset = h5f.create_dataset(
        "power spectra",
        spectra.shape,
        dtype=np.uint16,
        chunks=(1, spectra.shape[1]),
        shuffle=True,
        **hdf5plugin.Zstd(clevel=ZSTD_LEVEL),
    )
    rows, bins, codes = [], [], []
    inv_step = np.float32(1 / step)
    nbin = spectra.shape[1] if len(spectra) else 0
    limit = CODE_LIMIT - abs(offset)
    local = _Local()
    if keep is not None:
        keep = np.minimum(np.asarray(keep, dtype=np.int64), nbin)
        if keep.shape != (len(spectra),) or (keep < 0).any():
            raise ValueError("keep needs one non-negative count per row.")

    def encode(i, z, over_buf):
        """Encode row i into z; returns its overflow (bins, codes) or None."""
        spectrum = np.ascontiguousarray(spectra[i], dtype=np.float32)
        k = nbin if keep is None else int(keep[i])
        n_over = _encode_row(spectrum, inv_step, offset, limit, z, over_buf, k)
        if n_over < 0:
            raise ValueError(f"Power in row {i} is too large to store at step {step}.")
        if not n_over:
            return None
        over = over_buf[:n_over].copy()
        values = np.where(over < k, spectrum[over], np.float32(0))
        return over, np.rint(values * inv_step).astype(np.int32)

    def record(i, overflow):
        if overflow is not None:
            rows.append(np.full(len(overflow[0]), i))
            bins.append(overflow[0])
            codes.append(overflow[1])

    threads = _threads(threads)
    if threads and nbin:

        def compress_rows(start, stop):
            z = local.buffer("z", nbin, np.uint16)
            over_buf = local.buffer("over", nbin, np.int64)
            shuffled = local.buffer("shuffled", 2 * nbin, np.uint8)
            out = []
            for i in range(start, stop):
                overflow = encode(i, z, over_buf)
                _shuffle2(z, shuffled)
                out.append((local.compressor().compress(shuffled), overflow))
            return start, out

        tasks = (
            lambda s=s: compress_rows(s, min(s + WRITE_BLOCK_ROWS, len(spectra)))
            for s in range(0, len(spectra), WRITE_BLOCK_ROWS)
        )
        with ThreadPoolExecutor(threads) as executor:
            for start, out in _ordered(executor, tasks, 2 * threads):
                for i, (chunk, overflow) in enumerate(out, start):
                    dataset.id.write_direct_chunk((i, 0), chunk)
                    record(i, overflow)
    else:
        z = np.empty(nbin, dtype=np.uint16)
        over_buf = np.empty(nbin, dtype=np.int64)
        for i in range(len(spectra)):
            record(i, encode(i, z, over_buf))
            dataset[i] = z
    h5f.create_dataset(
        "overflow row", data=np.concatenate(rows) if rows else [], dtype="i4"
    )
    h5f.create_dataset(
        "overflow bin", data=np.concatenate(bins) if bins else [], dtype="i4"
    )
    h5f.create_dataset(
        "overflow code", data=np.concatenate(codes) if codes else [], dtype="i8"
    )
    if keep is not None:
        h5f.create_dataset("kept bins", data=keep, dtype="i4")
    h5f.attrs["quant_step"] = step
    h5f.attrs["quant_offset"] = offset


class QuantizedStack:
    """
    Reader for a stack written by write_quantized.

    The overflow list is read once, so reading a stack row by row stays cheap even when a
    bright source fills the list.
    """

    def __init__(self, h5f, threads=None):
        self.h5f = h5f
        dataset = h5f["power spectra"]
        # Threads decompressing rows (default THREADS); 0 uses the HDF5 filters
        self.threads = _threads(threads) if _direct_chunks(dataset) else 0
        self.step = h5f.attrs["quant_step"]
        self.offset = int(h5f.attrs["quant_offset"])
        self.nrow = h5f["power spectra"].shape[0]
        rows = h5f["overflow row"][()]
        order = np.argsort(rows, kind="stable")
        self.overflow_row = rows[order]
        self.overflow_bin = h5f["overflow bin"][()][order]
        self.overflow_code = h5f["overflow code"][()][order]

    def rows(self, start=0, stop=None, raw=False):
        """
        Read rows as float32.

        Parameters
        =========
        start, stop: int
            The range of DM rows to read.
        raw: bool
            Return the stored sum of rounded days, for adding further days, instead of
            the bias-corrected power.

        Returns
        =======
        spectra: np.ndarray
            The rows, with shape (stop - start, nfreq).
        """
        if stop is None:
            stop = self.nrow
        nbin = self.h5f["power spectra"].shape[1]
        spectra = np.empty((stop - start, nbin), dtype=np.float32)
        scale = np.float32(self.step if raw else self.step / quant_bias(self.step))
        self._decode(spectra, start, stop, scale, add=False, step=np.float32(0))
        return spectra

    def add_rounded(self, spectra, step):
        """
        Add a new day to the stored stack, in place: every row of spectra becomes
        round_power(row, step) plus the stored sum of rounded days (rows(raw=True)).

        Parameters
        =========
        spectra: np.ndarray
            float32 (dm, freq) power of the day, C-contiguous; it may be shared memory.
        step: float
            The rounding step, which must be that of the stack.
        """
        self._decode(
            spectra, 0, self.nrow, np.float32(self.step), add=True, step=np.float32(step)
        )

    def _decode(self, spectra, start, stop, scale, add, step):
        """Decode rows start:stop into spectra, a few rows at a time."""
        # Threads only pay off over several blocks (rows read one at a time would start
        # a pool for each)
        if self.threads and stop - start > READ_BLOCK_ROWS:
            return self._decode_threaded(spectra, start, stop, scale, add, step)
        dataset = self.h5f["power spectra"]
        raw = np.empty((READ_BLOCK_ROWS, dataset.shape[1]), dtype=np.uint16)
        for b0 in range(start, stop, READ_BLOCK_ROWS):
            b1 = min(b0 + READ_BLOCK_ROWS, stop)
            block = raw[: b1 - b0]
            dataset.read_direct(block, np.s_[b0:b1])
            lo, hi = np.searchsorted(self.overflow_row, [b0, b1])
            _decode_block(
                block,
                self.offset,
                scale,
                self.overflow_row[lo:hi] - b0,
                self.overflow_bin[lo:hi],
                self.overflow_code[lo:hi],
                spectra[b0 - start : b1 - start],
                add,
                step,
            )


    def _decode_threaded(self, spectra, start, stop, scale, add, step):
        """
        _decode with the chunks read whole here and decompressed, unshuffled and decoded
        in threads, a block of rows per task.
        """
        dataset = self.h5f["power spectra"]
        nbin = dataset.shape[1]
        local = _Local()

        def decode(b0, b1, chunks):
            raw = local.buffer("raw", READ_BLOCK_ROWS * nbin, np.uint16)
            raw = raw[: (b1 - b0) * nbin].reshape(b1 - b0, nbin)
            for r, chunk in enumerate(chunks):
                data = local.decompressor().decompress(chunk, max_output_size=2 * nbin)
                if len(data) != 2 * nbin:
                    raise ValueError(f"Row {b0 + r} decompressed to {len(data)} bytes.")
                _unshuffle2(np.frombuffer(data, dtype=np.uint8), raw[r])
            lo, hi = np.searchsorted(self.overflow_row, [b0, b1])
            _decode_block(
                raw,
                self.offset,
                scale,
                self.overflow_row[lo:hi] - b0,
                self.overflow_bin[lo:hi],
                self.overflow_code[lo:hi],
                spectra[b0 - start : b1 - start],
                add,
                step,
            )

        def read(b0, b1):
            chunks = []
            for i in range(b0, b1):
                mask, chunk = dataset.id.read_direct_chunk((i, 0))
                if mask:
                    # A chunk HDF5 stored without some filter: read the block through them
                    return None
                chunks.append(chunk)
            return chunks

        def tasks():
            for b0 in range(start, stop, READ_BLOCK_ROWS):
                b1 = min(b0 + READ_BLOCK_ROWS, stop)
                chunks = read(b0, b1)
                if chunks is None:
                    yield lambda b0=b0, b1=b1: self._decode_filtered(
                        spectra, start, b0, b1, scale, add, step
                    )
                else:
                    yield lambda b0=b0, b1=b1, c=chunks: decode(b0, b1, c)

        with ThreadPoolExecutor(self.threads) as executor:
            for _ in _ordered(executor, tasks(), 2 * self.threads):
                pass

    def _decode_filtered(self, spectra, start, b0, b1, scale, add, step):
        """Rows b0:b1 through the HDF5 filters (h5py serialises the call)."""
        dataset = self.h5f["power spectra"]
        block = np.empty((b1 - b0, dataset.shape[1]), dtype=np.uint16)
        dataset.read_direct(block, np.s_[b0:b1])
        lo, hi = np.searchsorted(self.overflow_row, [b0, b1])
        _decode_block(
            block,
            self.offset,
            scale,
            self.overflow_row[lo:hi] - b0,
            self.overflow_bin[lo:hi],
            self.overflow_code[lo:hi],
            spectra[b0 - start : b1 - start],
            add,
            step,
        )


def read_quantized(h5f, start=0, stop=None, raw=False, threads=None):
    """Read rows of a quantized stack as float32. See QuantizedStack.rows."""
    return QuantizedStack(h5f, threads=threads).rows(start, stop, raw)
