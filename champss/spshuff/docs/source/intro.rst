Installation
============

Install ``spshuff`` via

.. code-block:: python

    python setup.py install


Example
=======

All of the classes relevant to external ``sps`` interaction are contained in
``spshuff.l1_io``. For example, suppose we wanted to create an ``IntensityFile``
object and populate it in memory with normally distributed random samples:

.. code-block:: python

    import numpy as np
    from spshuff.l1_io import Chunk, ChunkHeader, FileHeader, IntensityFile

    def get_chunk()
	    nfreq = 16 * 1024
	    ntime = 1024

	    dat = np.random.normal(size=(nfreq, ntime)) # array will be flattened

	    means = np.mean(dat, axis=1)
		variance = np.var(dat, axis=1)

		bad_mask = np.random.choice(a=[True, False], size=(nfreq, ntime))

		frame0_nano = fpga0 = 0 # we're not an FPGA, so we don't even try
		fpgaN = 128 * 4096

		ch = ChunkHeader.from_fields(nfreq, ntime, frame0_nano, fpgaN, fpga0)
		return Chunk(ch, means, variance, bad_mask, dat, quantize_now=True)

	# generate random chunks
	chunk0 = get_chunk()
	chunk1 = get_chunk()
	chunks = [chunk0, chunk1]

	beamid = 1234
	nbins = 5 # this is the only hard-coded choice so far
	start = 0
	end = 0
	fh = FileHeader.from_fields(beamid, nbins, start, end)

	int_file = IntensityFile(fh, chunks)

Now, suppose we wanted to write this object in binary format to ``myfile.dat``:

.. code-block:: python

    with open('myfile.dat', 'w') as f:
    	int_file.to_file(f)

now, if we wished to read back the same file:

.. code-block:: python

    with open('myfile.dat', 'rb') as f:
    	int_file_read = IntensityFile.from_file(f)

and we could even verify equality via the overloaded ``__eq__`` method:

.. code-block:: python

    assert int_file_read == int_file

which would pass, of course.