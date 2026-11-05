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
    for i, spectrum in enumerate(spectra):
        k = np.rint(np.asarray(spectrum, dtype=np.float32) * inv_step).astype(np.int32)
        if np.abs(k).max(initial=0) >= CODE_LIMIT - abs(offset):
            raise ValueError(f"Power in row {i} is too large to store at step {step}.")
        z = _zigzag(k - np.int32(offset))
        over = np.flatnonzero(z > CODE_MAX)
        if len(over):
            rows.append(np.full(len(over), i))
            bins.append(over)
            codes.append(k[over])
            z[over] = 0
        dataset[i] = z.astype(np.uint16)
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
        dataset = self.h5f["power spectra"]
        spectra = np.empty((stop - start, dataset.shape[1]), dtype=np.float32)
        scale = np.float32(self.step if raw else self.step / quant_bias(self.step))
        # Decode a few rows at a time, so temporaries stay small next to the output
        for b0 in range(start, stop, READ_BLOCK_ROWS):
            b1 = min(b0 + READ_BLOCK_ROWS, stop)
            codes = _unzigzag(dataset[b0:b1].astype(np.int32))
            codes += np.int32(self.offset)
            lo, hi = np.searchsorted(self.overflow_row, [b0, b1])
            codes[self.overflow_row[lo:hi] - b0, self.overflow_bin[lo:hi]] = (
                self.overflow_code[lo:hi]
            )
            out = spectra[b0 - start : b1 - start]
            np.multiply(codes, scale, out=out, casting="unsafe")
        return spectra


def read_quantized(h5f, start=0, stop=None, raw=False):
    """Read rows of a quantized stack as float32. See QuantizedStack.rows."""
    return QuantizedStack(h5f).rows(start, stop, raw)
