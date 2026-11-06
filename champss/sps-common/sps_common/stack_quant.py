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
"""

import hdf5plugin
import numpy as np
from numba import njit

CODE_MAX = np.iinfo(np.uint16).max
# Level 1 gives the same size as 3 on these stacks (noise does not reward effort), faster
ZSTD_LEVEL = 1
# Integer work is done in int32, which holds any sum below ~5e8 power at step 0.25
CODE_LIMIT = 2**30
READ_BLOCK_ROWS = 32


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


@njit(cache=True)
def _encode_row(spectrum, inv_step, offset, limit, z, over):
    """
    zigzag(rint(spectrum * inv_step) - offset) into z, as write_quantized stores it.

    Bins whose code does not fit in 16 bits get 0 and are listed in over. Returns how
    many there are, or -1 if a value is at least limit in size or is not finite.
    """
    n_over = 0
    for j in range(spectrum.shape[0]):
        r = np.rint(spectrum[j] * inv_step)
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


@njit(cache=True)
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


def write_quantized(h5f, spectra, step, num_days):
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
    z = np.empty(nbin, dtype=np.uint16)
    over_buf = np.empty(nbin, dtype=np.int64)
    for i, spectrum in enumerate(spectra):
        spectrum = np.ascontiguousarray(spectrum, dtype=np.float32)
        n_over = _encode_row(
            spectrum, inv_step, offset, CODE_LIMIT - abs(offset), z, over_buf
        )
        if n_over < 0:
            raise ValueError(f"Power in row {i} is too large to store at step {step}.")
        if n_over:
            over = over_buf[:n_over].copy()
            rows.append(np.full(n_over, i))
            bins.append(over)
            codes.append(np.rint(spectrum[over] * inv_step).astype(np.int32))
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
    h5f.attrs["quant_step"] = step
    h5f.attrs["quant_offset"] = offset


class QuantizedStack:
    """
    Reader for a stack written by write_quantized.

    The overflow list is read once, so reading a stack row by row stays cheap even when a
    bright source fills the list.
    """

    def __init__(self, h5f):
        self.h5f = h5f
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


def read_quantized(h5f, start=0, stop=None, raw=False):
    """Read rows of a quantized stack as float32. See QuantizedStack.rows."""
    return QuantizedStack(h5f).rows(start, stop, raw)
