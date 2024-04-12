import numpy as np
import time
from textwrap import indent

from spshuff import decode, encode
from spshuff.huff_utils import quantize_dequantize, feq, data_dtype, summary_dtype, \
						check_eof


THIS_VERSION = 3


file_dt = np.dtype([('beam_number', np.uint16),
					('nbins', np.uint16),
					('start', np.float64),
					('end', np.float64),
					('version', np.int32)])

chunk_dt = np.dtype([('nfreq', np.uint16),
					 ('ntime', np.uint16),
					 ('frame0_nano', np.uint64),
					 ('fpgaN', np.uint64),
					 ('fpga0', np.uint64),])


class FileHeader:
	def __init__(self, file_header):
		self.file_header = file_header
		self.beam_number = file_header['beam_number']
		self.nbins = file_header['nbins']
		self.start = file_header['start']
		self.end = file_header['end']


	def to_file(self, f):
		self.as_array().tofile(f)


	def as_array(self):
		return np.array(self.file_header, dtype=file_dt)


	def set_start(self, t=time.time()):
		self.file_header['start'] = float(t)
		self.start = self.file_header['start']


	def set_end(self, t=time.time()):
		self.file_header['end'] = float(t)
		self.end = self.file_header['end']


	def __str__(self):
		return 'FileHeader\n\tbeam {}\n\tnbins {}\n\tend {}'.format(self.beam_number,
														self.nbins, self.end)


	def __eq__(self, other):
		return self.as_array() == other.as_array()


	@classmethod
	def from_file(cls, f):
		file_header = np.fromfile(f, dtype=file_dt, count=1)[0]
		fversion = file_header['version']
		assert fversion == THIS_VERSION, \
		f'Must use compatible spshuff version. File is version {fversion}, spshuff is version {THIS_VERSION}'
		return cls(file_header)


	@classmethod
	def from_fields(cls, beam_number, nbins, start, end):
		start = float(start)
		end = float(end)
		assert end >= start
		fh = np.array([(beam_number, nbins, start, end, THIS_VERSION),], 
															dtype=file_dt)[0]
		return cls(fh)


def test_file_header():
	beam_number = 123
	nbins = 5
	start = time.time()
	end = time.time()

	fh = None
	with open('fh_test.dat', 'w') as f:
		fh = FileHeader.from_fields(beam_number, nbins, start, end)
		fh.to_file(f)

	fh2 = None
	with open('fh_test.dat', 'r') as f:
		fh2 = FileHeader.from_file(f)

	assert fh.as_array() == fh2.as_array()


class ChunkHeader:
	def __init__(self, chunk_header):
		self.chunk_header = chunk_header
		self.nfreq = chunk_header['nfreq']
		self.ntime = chunk_header['ntime']
		self.shape = np.array((self.nfreq, self.ntime), dtype=int)

		self.frame0_nano = chunk_header['frame0_nano']
		self.fpgaN = chunk_header['fpgaN']
		self.fpga0 = chunk_header['fpga0']


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
		return 'ChunkHeader:\n\tnfreq {}\n\tntime {}\n\tframe0_nano{}\n\tfpgaN{}\n\tfpga0{}'.format(self.nfreq, 
															 self.ntime, self.frame0_nano, self.fpgaN, self.fpga0)


	@classmethod
	def from_file(cls, f):
		chunk_header = np.fromfile(f, dtype=chunk_dt, count=1)[0]
		return cls(chunk_header)


	@classmethod
	def from_fields(cls, nfreq, ntime, frame0_nano, fpgaN, fpga0):
		ch = np.array([(nfreq, ntime, frame0_nano, fpgaN, fpga0),], dtype=chunk_dt)[0]
		return cls(ch)


def test_chunk_header():
	nfreq = 1024 * 16
	ntime = 1024

	ch = None
	with open('ch_test.dat', 'w') as f:
		ch = ChunkHeader.from_fields(nfreq, ntime, 0, 0, 0)
		ch.to_file(f)

	ch2 = None
	with open('ch_test.dat', 'r') as f:
		ch2 = ChunkHeader.from_file(f)

	assert ch == ch2


# In this paradigm, we leave all chunk object data uncompressed in memory,
# but compress upon file write and decompress upon file read
class Chunk:
	def __init__(self, chunk_header, means, variance, 
						bad_mask, data, quantize_now=False):
		mydat = data.astype(data_dtype)
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
			self.data = quantize_dequantize(mydat.reshape(nfreq * ntime)).reshape((nfreq, ntime))
		else:
			self.data = mydat


	@classmethod
	def min_size(cls):
		return chunk_dt.itemsize + 2 * summary_dtype.itemsize \
					+ 2 * data_dtype.itemsize


	@classmethod
	def from_file(cls, f):
		chunk_header = ChunkHeader.from_file(f) 
		nfreq, ntime = chunk_header.shape
		means = np.fromfile(f, count=nfreq, dtype=summary_dtype)
		variance = np.fromfile(f, count=nfreq, dtype=summary_dtype)
		bad_mask = np.fromfile(f, count=nfreq * ntime, dtype=bool).reshape((nfreq, ntime))

		n_enc = np.fromfile(f, count=1, dtype=int)[0]

		ndat = nfreq * ntime
		data = decode(np.fromfile(f, count=n_enc, dtype=np.uint32), ndat)
		data = data.reshape((nfreq, ntime))
		return cls(chunk_header, means, variance, bad_mask, data)


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
		ret = 'Chunk:\n'
		ret += indent(self.chunk_header.__str__(), '\t') + '\n\n'
		ret += indent(self.means.__str__(), '\t') + '\n\n'
		ret += indent(self.variance.__str__(), '\t') + '\n\n'
		ret += indent(self.bad_mask.__str__(), '\t') + '\n\n'
		ret += indent(self.data.__str__(), '\t') + '\n'
		return ret


	def to_file(self, f):
		self.chunk_header.to_file(f)
		self.means.tofile(f)
		self.variance.tofile(f)
		self.bad_mask.tofile(f) # consider compression

		nfreq, ntime = self.shape
		data_enc = encode(self.data.reshape(nfreq * ntime))
		n_enc = len(data_enc)
		np.array([n_enc,], dtype=int).tofile(f)
		data_enc.tofile(f)


def write_bytes(ar, f):
	f.write(f)


def test_chunk():
	nfreq = 1024 * 16
	ntime = 1024
	fpgaN = 4096 * 128

	dat = np.random.normal(size=(nfreq, ntime))
	ref_dequant = quantize_dequantize(dat.reshape(nfreq * ntime)).reshape((nfreq, ntime))

	means = np.mean(dat, axis=1)
	variance = np.var(dat, axis=1)

	bad_mask = np.random.choice(a=[True, False], size=(nfreq, ntime))

	ch = ChunkHeader.from_fields(nfreq, ntime, 0, fpgaN, 0)
	chunk = Chunk(ch, means, variance, bad_mask, dat, quantize_now=True)
	with open('chunk_test.dat', 'w') as f:
		chunk.to_file(f)

	chunk2 = None
	with open('chunk_test.dat', 'r') as f:
		chunk2 = Chunk.from_file(f)

	assert(np.all(chunk2.data == ref_dequant))
	assert(chunk2 == chunk)


class IntensityFile:
	def __init__(self, fh, chunks = [], file=None, read_now=False):
		"""The high-level intensity file object. This represents either an in-memory chunked intensity file without an optional file reference for reads/writes
		
		:param fh: file header
		:type fh: :class:`.FileHeader`
		:param chunks: chunks to include initially
		:type chunks: list of :class:`.Chunk`, optional
		:param file: file object for reads/writes
		:type file: file object, optional
		:param read_now: flag for immediate read (appends to chunks list), defaults to False
		:type read_now: bool, optional
		"""
		self.file = file
		self.fh = fh
		self.chunks = chunks.copy()

		if read_now:
			self.read_chunks()


	def get_chunks(self):
		"""Return a copy of the chunks list
		
		:return: a copy of the chunks list
		:rtype: a list of :class:`.Chunk`
		"""
		return self.chunks.copy()


	def add_chunk(self, chunk):
		"""Appends a chunk to the chunk list

		:param chunk: a :class:`.Chunk` object
		:type chunk: :class:`.Chunk`
		"""
		self.chunks.append(chunk)


	def set_chunks(self, chunks):
		"""Set the data of the chunks list
			
		:param chunks: chunks to copy in
		:type chunks: list of :class:`.Chunk`
		"""
		self.chunks = chunks.copy()


	def has_chunks(self):
		"""Determine whether the underlying file has more chunks. Fails assert when the file object is None

		:return: a boolean representing whether the underlying file has at least one additional chunk
		:rtype: bool
		"""
		assert self.file is not None

		return check_eof(self.file, Chunk.min_size())


	def read_next_chunk(self):
		"""Return the next chunk, if applicable. Fails assert if file object is None. Returns None when the file is exhausted

		:return: the next chunk from file
		:rtype: :class:`.Chunk`
		"""
		assert self.file is not None

		if self.has_chunks():
			this_chunk = Chunk.from_file(self.file)
			self.chunks.append(this_chunk)
			return this_chunk
		else:
			return None


	def read_chunks(self):
		"""Return the next chunk, if applicable. Fails assert if file object is None. Returns None when the file is exhausted"""
		assert self.file is not None

		while self.has_chunks():
			this_chunk = Chunk.from_file(self.file)
			self.chunks.append(this_chunk)


	def __getitem__(self, key):
		"""Index operator overloaded for convenience

		:param key: chunk index
		:type key: int
		:return: the chunk at index key
		:rtype: :class:`.Chunk`
		"""
		return self.chunks[key]


	def to_file(self, file=None, set_end_time=False):
		"""Write an IntensityFile to disk in a binary format; only writes the chunks in the chunks buffer

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
		ret = 'IntensityFile:\n'
		ret += indent(self.fh.__str__(), '\t') + '\n'

		for c in self.chunks:
			ret += indent(c, '\t') + '\n'

		return ret


	def __eq__(self, other):
		# here we explicitly ignore the file field

		if self.fh != other.fh:
			return False

		# verify chunk-wise equality
		for a,b in zip(self, other):
			if a != b:
				return False

		return True


	@classmethod
	def from_file(cls, f):
		"""Construct an :class:`.IntensityFile` object from a file object

		:param f: an open file object (note: must open in binary read 'rb' mode!)
		:type f: file object
		"""
		fh = FileHeader.from_file(f)
		int_file = IntensityFile(fh, file=f)
		int_file.read_chunks()
		return int_file


def get_test_chunk(nfreq=16*1024, ntime=1024, frame0_nano=0,
				   fpgaN=4096*128, fpga0=0):
	dat = np.random.normal(size=(nfreq, ntime))
	ref_dequant = quantize_dequantize(dat.reshape(nfreq * ntime)).reshape((nfreq, ntime))

	means = np.mean(dat, axis=1)
	variance = np.var(dat, axis=1)

	bad_mask = np.random.choice(a=[True, False], size=(nfreq, ntime))

	ch = ChunkHeader.from_fields(nfreq, ntime, frame0_nano, fpgaN, fpga0)
	chunk = Chunk(ch, means, variance, bad_mask, dat, quantize_now=True)
	return chunk, ref_dequant


def test_file():

	with open('intensity.dat', 'w') as f:
		beam_number = 1012
		nbins = 5
		start = 0
		end = 1
		fh = FileHeader.from_fields(beam_number, nbins, start, end)
		chunk0, ref0 = get_test_chunk()
		chunk1, ref1 = get_test_chunk()
		chunks = [chunk0, chunk1]

		int_file = IntensityFile(fh, chunks)
		int_file.to_file(file=f)

	with open('intensity.dat', 'rb') as f:
		int_file_recovered = IntensityFile.from_file(f)

	assert int_file == int_file_recovered
