import time
from textwrap import indent

import numpy as np
from spshuff import decode, encode
from spshuff.huff_utils import (
    check_eof,
    data_dtype,
    feq,
    quantize_dequantize,
    summary_dtype,
)

THIS_VERSION = 5


file_dt = np.dtype(
    [
        ("beam_number", np.uint16),
        ("nbins", np.uint16),
        ("start", np.float64),
        ("end", np.float64),
        ("version", np.int32),
    ]
)

chunk_dt = np.dtype(
    [
        ("nfreq", np.uint16),
        ("ntime", np.uint16),
        ("frame0_nano", np.uint64),
        ("fpgaN", np.uint64),
        ("fpga0", np.uint64),
    ]
)


class FileHeader:
    def __init__(self, file_header):
        self.file_header = file_header
        self.beam_number = file_header["beam_number"]
        self.nbins = file_header["nbins"]
        self.start = file_header["start"]
        self.end = file_header["end"]
        self.version = file_header["version"]

    def to_file(self, f):
        self.as_array().tofile(f)

    def as_array(self):
        return np.array(self.file_header, dtype=file_dt)

    def set_start(self, t=time.time()):
        self.file_header["start"] = float(t)
        self.start = self.file_header["start"]

    def set_end(self, t=time.time()):
        self.file_header["end"] = float(t)
        self.end = self.file_header["end"]

    def __str__(self):
        return (
            "FileHeader\n\tbeam {}\n\tnbins {}\n\tstart {}\n\tend {}\n\tversion {}\n"
            .format(self.beam_number, self.nbins, self.start, self.end, self.version)
        )

    def __eq__(self, other):
        return self.as_array() == other.as_array()

    @classmethod
    def from_file(cls, f, check_version=True):
        file_header = np.fromfile(f, dtype=file_dt, count=1)[0]
        fversion = file_header["version"]
        if check_version:
            assert fversion == THIS_VERSION, (
                f"Must use compatible spshuff version. File is version {fversion},"
                f" spshuff is version {THIS_VERSION}"
            )
        return cls(file_header)

    @classmethod
    def from_fields(cls, beam_number, nbins, start, end):
        start = float(start)
        end = float(end)
        assert end >= start
        fh = np.array(
            [
                (beam_number, nbins, start, end, THIS_VERSION),
            ],
            dtype=file_dt,
        )[0]
        return cls(fh)


def test_file_header():
    beam_number = 123
    nbins = 5
    start = time.time()
    end = time.time()

    fh = None
    with open("fh_test.dat", "w") as f:
        fh = FileHeader.from_fields(beam_number, nbins, start, end)
        fh.to_file(f)

    fh2 = None
    with open("fh_test.dat") as f:
        fh2 = FileHeader.from_file(f)

    assert fh.as_array() == fh2.as_array()


class ChunkHeader:
    def __init__(self, chunk_header):
        self.chunk_header = chunk_header
        self.nfreq = chunk_header["nfreq"]
        self.ntime = chunk_header["ntime"]

        self.shape = np.array((self.nfreq, self.ntime), dtype=int)

        self.frame0_nano = chunk_header["frame0_nano"]
        self.fpgaN = chunk_header["fpgaN"]
        self.fpga0 = chunk_header["fpga0"]

    def to_file(self, f):
        ar = np.array(self.chunk_header, dtype=chunk_dt)
        f.write(ar.tobytes())

    def to_file(self, f):
        self.as_array().tofile(f)

    def as_array(self):
        return np.array(self.chunk_header, dtype=chunk_dt)

    def __eq__(self, other):
        return self.as_array() == other.as_array()

    def __str__(self):
        return (
            "ChunkHeader:\n\tnfreq {}\n\tntime {}\n\tframe0_nano {}\n\tfpgaN"
            " {}\n\tfpga0 {}".format(
                self.nfreq, self.ntime, self.frame0_nano, self.fpgaN, self.fpga0
            )
        )

    @classmethod
    def from_file(cls, f):
        chunk_header = np.fromfile(f, dtype=chunk_dt, count=1)[0]
        return cls(chunk_header)

    @classmethod
    def from_fields(cls, nfreq, ntime, frame0_nano, fpgaN, fpga0):
        ch = np.array(
            [
                (nfreq, ntime, frame0_nano, fpgaN, fpga0),
            ],
            dtype=chunk_dt,
        )[0]
        return cls(ch)


def test_chunk_header():
    nfreq = 1024 * 16
    ntime = 1024

    ch = None
    with open("ch_test.dat", "w") as f:
        ch = ChunkHeader.from_fields(nfreq, ntime, 0, 0, 0)
        ch.to_file(f)

    ch2 = None
    with open("ch_test.dat") as f:
        ch2 = ChunkHeader.from_file(f)

    assert ch == ch2


# Takes a uint8_t array and unpacks into an 8x larger bool array
def unpack_mask_reference(intmask):
    ilen = np.prod(intmask.shape)
    imask = intmask.reshape(ilen)
    assert intmask.dtype is np.dtype(np.uint8)

    ret = np.empty(ilen * 8, dtype=np.bool)
    for i in range(ilen):
        val = imask[i]
        imin = 8 * i
        for j in range(7, -1, -1):
            div = 2**j
            d = val // div
            ret[imin + j] = d
            val -= d * div

    return ret


# Takes a uint8_t array and unpacks into an 8x larger bool array
def unpack_mask(intmask):
    return np.unpackbits(intmask, bitorder="little")


def test_unpack():
    ar = np.empty(4, dtype=np.uint8)
    ar[0] = 254
    ar[1] = 64
    ar[2] = 123
    ar[3] = 0

    bools = unpack_mask(ar)
    assert np.all(
        bools[:8] == np.array([False, True, True, True, True, True, True, True])
    )
    assert np.all(
        bools[8:16] == np.array([False, False, False, False, False, False, True, False])
    )
    assert np.all(
        bools[16:24] == np.array([True, True, False, True, True, True, True, False])
    )
    assert np.all(
        bools[24:] == np.array([False, False, False, False, False, False, False, False])
    )


# packs a bool mask into uint8
def pack_mask_reference(bool_mask):
    mlen = np.prod(bool_mask.shape)
    bmask = bool_mask.reshape(mlen)
    assert bmask.dtype is np.dtype(np.bool)
    assert (mlen % 8) == 0
    rlen = mlen // 8
    ret = np.empty(rlen, dtype=np.uint8)

    for i in range(rlen):
        v = 0
        exp = 1
        imin = 8 * i
        for j in range(8):
            v += int(bmask[imin + j]) * exp
            exp *= 2
        ret[i] = v

    return ret
    # TODO: add packing logic


def pack_mask(bool_mask):
    return np.packbits(bool_mask, bitorder="little")


def test_pack():
    bools = np.empty(32, dtype=np.bool)
    bools[:8] = np.array([False, True, True, True, True, True, True, True])
    bools[8:16] = np.array([False, False, False, False, False, False, True, False])
    bools[16:24] = np.array([True, True, False, True, True, True, True, False])
    bools[24:] = np.array([False, False, False, False, False, False, False, False])

    ar = pack_mask(bools)
    assert ar[0] == 254
    assert ar[1] == 64
    assert ar[2] == 123
    assert ar[3] == 0


def downsample(ar, nds):
    nds = np.array(nds)
    tmp = ar.copy()
    for i in range(int(np.log2(nds[0]))):
        tmp = 0.5 * (tmp[0::2, :] + tmp[1::2, :])

    for j in range(int(np.log2(nds[1]))):
        tmp = 0.5 * (tmp[:, 0::2] + tmp[:, 1::2])

    assert np.all(tmp.shape == ar.shape / nds)
    return tmp


# In this paradigm, we leave all chunk object data uncompressed in memory,
# but compress upon file write and decompress upon file read
class Chunk:
    def __init__(
        self,
        chunk_header,
        means,
        variance,
        bad_mask,
        data,
        quantize_now=False,
        shape=None,
    ):
        if data.dtype != data_dtype:
            mydat = data.astype(data_dtype)
        else:
            mydat = data
        self.chunk_header = chunk_header
        nfreq, ntime = self.chunk_header.shape

        self.shape = self.chunk_header.shape

        assert means.ndim == 1
        assert variance.ndim == 1
        assert bad_mask.shape == (nfreq, ntime)
        assert data.shape == (nfreq, ntime)

        self.means = means.astype(summary_dtype)
        self.variance = variance.astype(summary_dtype)
        self.bad_mask = bad_mask

        if quantize_now:
            nfreq, ntime = self.chunk_header.shape
            self.data = quantize_dequantize(mydat.reshape(nfreq * ntime)).reshape(
                (nfreq, ntime)
            )
        else:
            self.data = mydat  # .copy()

    @classmethod
    def min_size(cls):
        return chunk_dt.itemsize + 2 * summary_dtype.itemsize + 2 * data_dtype.itemsize

    @classmethod
    def from_file(cls, f, shape=None):
        chunk_header = ChunkHeader.from_file(f)

        nfreq, ntime = chunk_header.shape
        nsamp = nfreq * ntime
        assert (nsamp % 8) == 0
        nmask = nsamp // 8

        # if only a 1-D downsample is requested (e.g., only time or only frequency)
        # then the other shape member will probably be None, so we should replace
        # that intelligently.
        nds = None
        if shape is not None:
            # This ensures that the shape tuple has valid values for each dimension
            tmp_shape = list(shape)
            if tmp_shape[0] is None:
                tmp_shape[0] = nfreq
            if tmp_shape[1] is None:
                tmp_shape[1] = ntime

            # ensure downsampled shape is never larger than native chunk dims
            shape = np.minimum(tmp_shape, chunk_header.shape)

            assert np.log2(shape[0]) % 1.0 == 0.0
            assert np.log2(shape[1]) % 1.0 == 0.0
            nds = (chunk_header.shape / shape).astype(int)
            assert np.log2(nds[0]) % 1.0 == 0.0
            assert np.log2(nds[1]) % 1.0 == 0.0

            downsample_required = shape[0] != nfreq or shape[1] != ntime

        means = np.fromfile(f, count=nfreq, dtype=summary_dtype)
        variance = np.fromfile(f, count=nfreq, dtype=summary_dtype)
        bad_mask_tight = np.fromfile(f, count=nmask, dtype=np.uint8)

        # make the bool bad mask from the tight packed bad mask
        bad_mask = unpack_mask(bad_mask_tight).reshape((nfreq, ntime))

        if shape is not None and downsample_required:
            bad_mask2 = downsample(bad_mask, nds).astype(int).astype(bool)
            assert np.all(bad_mask2.shape == shape)
        nenc = np.fromfile(f, count=1, dtype=int)[0]  # encoded length (32 bit words)

        ndat = nfreq * ntime
        indat = np.fromfile(f, count=nenc, dtype=np.uint32)
        # indat = np.zeros(nenc//4,dtype=np.uint32)
        data = decode(indat, ndat)
        data = data.reshape((nfreq, ntime))

        if shape is not None and downsample_required:
            data2 = downsample(data * np.sqrt(variance)[:, None] + means[:, None], nds)
            vars2 = np.var(data2, axis=1)
            means2 = np.mean(data2, axis=1)
            data2_scaled = (data2 - means2[:, None]) / np.sqrt(vars2)[:, None]

            assert np.all(data2_scaled.shape == shape)
            chunk_header.shape = shape.copy()
            chunk_header.nfreq = shape[0]
            chunk_header.ntime = shape[1]

            return cls(
                chunk_header, means2, vars2, bad_mask2, data2_scaled, quantize_now=True
            )
        else:
            return cls(chunk_header, means, variance, bad_mask, data)

    def to_file(self, f):
        self.chunk_header.to_file(f)
        self.means.tofile(f)
        self.variance.tofile(f)
        pack_mask(self.bad_mask).tofile(f)  # consider compression

        nfreq, ntime = self.shape
        data_enc = encode(self.data.reshape(nfreq * ntime))
        n_enc = len(data_enc)
        np.array(
            [
                n_enc,
            ],
            dtype=int,
        ).tofile(f)
        data_enc.tofile(f)

    def get_unix_start_time(self):
        return self.chunk_header.frame0_nano * 1e-9 + self.chunk_header.fpga0 * 2.56e-6

    def get_data(self, apply_mask=False):
        """
        Return a copy of the data as recorded in L1, with mean added, i.e. we undo the
        compression normalization :param apply_mask: masking flag :type apply_mask:
        bool.

        :return: the chunk data, as described
        :rtype: ndarray
        """

        nfreq, ntime = self.shape
        ret = self.data * np.tile(
            np.sqrt(self.variance)[:, None], (1, ntime)
        ) + np.tile(self.means[:, None], (1, ntime))
        if apply_mask:
            ret *= self.bad_mask

        return ret

    def __eq__(self, other):
        if self.chunk_header != other.chunk_header:
            return False

        if not np.all(self.means == other.means):
            return False

        if not np.all(self.variance == other.variance):
            return False

        if not np.all(self.bad_mask == other.bad_mask):
            return False

        # In the future, we could broaden quality to mean "encodes to the same value"
        if not np.all(self.data == other.data):
            return False

        return True

    def __str__(self):
        ret = "Chunk:\n"
        ret += indent(self.chunk_header.__str__(), "\t") + "\n\n"
        ret += indent(self.means.__str__(), "\t") + "\n\n"
        ret += indent(self.variance.__str__(), "\t") + "\n\n"
        ret += indent(self.bad_mask.__str__(), "\t") + "\n\n"
        ret += indent(self.data.__str__(), "\t") + "\n"
        return ret


def write_bytes(ar, f):
    f.write(f)


def close_enough(a, b, eps=1.1):
    return np.all(np.abs(a - b) <= eps)


def test_chunk():
    nfreq = 1024 * 16
    ntime = 1024
    fpgaN = 4096 * 128

    dat = np.random.normal(size=(nfreq, ntime))
    ref_dequant = quantize_dequantize(dat.reshape(nfreq * ntime)).reshape(
        (nfreq, ntime)
    )

    means = np.mean(dat, axis=1)
    variance = np.var(dat, axis=1)

    bad_mask = np.random.choice(a=[True, False], size=(nfreq, ntime))

    ch = ChunkHeader.from_fields(nfreq, ntime, 0, fpgaN, 0)
    chunk = Chunk(ch, means, variance, bad_mask, dat, quantize_now=True)
    with open("chunk_test.dat", "w") as f:
        chunk.to_file(f)

    chunk2 = None
    with open("chunk_test.dat") as f:
        chunk2 = Chunk.from_file(f)

    chunk3 = None
    newshape = np.array((4096, 512))
    with open("chunk_test.dat") as f:
        chunk3 = Chunk.from_file(f, shape=newshape)

    nds = ref_dequant.shape // newshape

    assert close_enough(chunk3.get_data(), downsample(ref_dequant, nds))
    assert np.all(chunk2.data == ref_dequant)
    assert chunk2 == chunk


class IntensityFile:
    def __init__(self, fh, chunks=[], file=None, read_now=False, ds_shape=None):
        """
        The high-level intensity file object. This represents either an in-memory
        chunked intensity file without an optional file reference for reads/writes.

        :param fh: file header
        :type fh: :class:`.FileHeader`
        :param chunks: chunks to include initially
        :type chunks: list of :class:`.Chunk`, optional
        :param file: file object for reads/writes
        :type file: file object, optional
        :param read_now: flag for immediate read (appends to chunks list), defaults to False
        :type read_now: bool, optional
        :param ds_shape: shape to downsample if additional downsampling is desired
        :type ds_shape: list (nfreq, ntime), optional
        """
        self.file = file
        self.fh = fh
        self.chunks = chunks.copy()
        self.ds_shape = ds_shape

        if read_now:
            self.read_chunks()

    def get_chunks(self):
        """
        Return a copy of the chunks list.

        :return: a copy of the chunks list
        :rtype: a list of :class:`.Chunk`
        """
        return self.chunks.copy()

    def add_chunk(self, chunk):
        """
        Appends a chunk to the chunk list.

        :param chunk: a :class:`.Chunk` object
        :type chunk: :class:`.Chunk`
        """
        self.chunks.append(chunk)

    def set_chunks(self, chunks):
        """
        Set the data of the chunks list.

        :param chunks: chunks to copy in
        :type chunks: list of :class:`.Chunk`
        """
        self.chunks = chunks.copy()

    def has_chunks(self):
        """
        Determine whether the underlying file has more chunks. Fails assert when the
        file object is None.

        :return: a boolean representing whether the underlying file has at least one
            additional chunk
        :rtype: bool
        """
        assert self.file is not None

        return check_eof(self.file, Chunk.min_size())

    def read_next_chunk(self):
        """
        Return the next chunk, if applicable. Fails assert if file object is None.
        Returns None when the file is exhausted.

        :return: the next chunk from file
        :rtype: :class:`.Chunk`
        """
        assert self.file is not None

        if self.has_chunks():
            this_chunk = Chunk.from_file(self.file, shape=self.ds_shape)
            self.chunks.append(this_chunk)
            return this_chunk
        else:
            return None

    def read_chunks(self):
        """
        Return the next chunk, if applicable.

        Fails assert if file object is None. Returns None when the file is exhausted
        """
        assert self.file is not None

        while self.has_chunks():
            this_chunk = Chunk.from_file(self.file, shape=self.ds_shape)
            self.chunks.append(this_chunk)

    def __getitem__(self, key):
        """
        Index operator overloaded for convenience.

        :param key: chunk index
        :type key: int
        :return: the chunk at index key
        :rtype: :class:`.Chunk`
        """
        return self.chunks[key]

    def to_file(self, file=None, set_end_time=False):
        """
        Write an IntensityFile to disk in a binary format; only writes the chunks in the
        chunks buffer.

        :param file: a file other than the file already associated with the object
        :type file: file object, optional
        :param set_time: a flag to set the :class:`.FileHeader` end time to the current time, defaults to False
        :type set_time: bool, optional
        """
        if file is None:
            file = self.file

        assert file is not None
        assert self.fh is not None

        if set_end_time:
            self.fh.set_end()

        file.seek(0)
        self.fh.to_file(file)
        for chunk in self.chunks:
            chunk.to_file(file)

    def __str__(self):
        ret = "IntensityFile:\n"
        ret += indent(self.fh.__str__(), "\t") + "\n"

        for c in self.chunks:
            ret += indent(c.__str__(), "\t") + "\n"

        return ret

    def __eq__(self, other):
        # here we explicitly ignore the file field

        if self.fh != other.fh:
            return False

        # verify chunk-wise equality
        for a, b in zip(self.chunks, other.chunks):
            if a != b:
                return False

        return True

    @classmethod
    def from_file(cls, f, shape=None):
        """
        Construct an :class:`.IntensityFile` object from a file object.

        :param f: an open file object (note: must open in binary read 'rb' mode!)
        :type f: file object
        :param shape: shape to downsample each chunk if additional downsampling is
            desired
        :type shape: list (nfreq, ntime), optional
        """
        fh = FileHeader.from_file(f)
        int_file = IntensityFile(fh, file=f, ds_shape=shape)
        int_file.read_chunks()
        return int_file


def get_test_chunk(
    nfreq=16 * 1024, ntime=1024, frame0_nano=0, fpgaN=4096 * 128, fpga0=0
):
    dat = np.random.normal(size=(nfreq, ntime))
    ref_dequant = quantize_dequantize(dat.reshape(nfreq * ntime)).reshape(
        (nfreq, ntime)
    )

    means = np.mean(dat, axis=1)
    variance = np.var(dat, axis=1)

    bad_mask = np.random.choice(a=[True, False], size=(nfreq, ntime))

    ch = ChunkHeader.from_fields(nfreq, ntime, frame0_nano, fpgaN, fpga0)
    chunk = Chunk(ch, means, variance, bad_mask, dat, quantize_now=True)
    return chunk, ref_dequant


def test_file():
    beam_number = 1012
    nbins = 5
    start = 0
    end = 1
    fh = FileHeader.from_fields(beam_number, nbins, start, end)
    nfreq = 16 * 1024
    ntime = 1024

    nfreq_ds = 1024
    ntime_ds = 2048  # test "upsampling" ignore

    in_shape = np.array((nfreq, ntime))
    ds_shape = np.array((nfreq_ds, ntime_ds))
    out_shape = np.minimum(in_shape, ds_shape)

    chunk0, ref0 = get_test_chunk(nfreq, ntime)
    chunk1, ref1 = get_test_chunk(nfreq, ntime)
    chunks = [chunk0, chunk1]
    int_file = IntensityFile(fh, chunks)

    with open("intensity.dat", "w") as f:
        int_file.to_file(file=f)

    with open("intensity.dat", "rb") as f:
        int_file_recovered_ds = IntensityFile.from_file(f, shape=ds_shape)
        int_file_recovered_ds.read_chunks()

    with open("intensity.dat", "rb") as f:
        int_file_recovered = IntensityFile.from_file(f)
        int_file_recovered.read_chunks()

    assert np.all(int_file_recovered_ds.chunks[0].shape == np.array((nfreq_ds, ntime)))
    assert int_file == int_file_recovered
    assert int_file != int_file_recovered_ds
    assert int_file_recovered != int_file_recovered_ds
