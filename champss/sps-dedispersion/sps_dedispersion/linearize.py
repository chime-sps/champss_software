#!/usr/bin/env python3
import sys

import numpy as np

# The following code reads a block of 1024 channels of binary float32
# data from STDIN, "linearizes" the band by smearing and/or zero-padding
# in the freq direction, and outputs that to STDOUT.  Run it like:
#
# FILTERBANKDATA | python linearize.py | TREECODE

# Following makes the low edge of bottom channel at 400.1953125 MHz and
# top edge of top channel at 800.1953125 MHz
nchan = 1024
freqs = np.arange(nchan) * 0.390625 + 400.390625
lnchan = 2048  # linearized band number of channels
# The following is the linearized "freq" lookup table
lsqrd = 1.0 / freqs**2 - 1.0 / freqs[-1] ** 2
# if the band needs flipped, remove the first "(lnchan - 1) -"
lsqrd = (lnchan - 1) - ((lsqrd / lsqrd[0]) * (lnchan - 1) + 0.5).astype(np.int32)
# lsqrd = ((lsqrd / lsqrd[0]) * (lnchan - 1) + 0.5).astype(np.int32)

nbytes = nchan * 4  # 1024 channels times 4 bytes each
indata = sys.stdin.buffer.read(nbytes)
while indata != b"":
    outarr = np.zeros(lnchan, dtype=np.float32)
    # The following properly deals with duplicated indices.
    # See https://stackoverflow.com/questions/24099404/numpy-array-iadd-and-repeated-indices
    np.add.at(outarr, lsqrd, np.frombuffer(indata, dtype=np.float32))
    sys.stdout.buffer.write(outarr.tobytes())
    # if we want to double the data rate (and pretend we
    # are sampling twice as fast as we are) uncomment the next line
    # print(outarr.tobytes())
    indata = sys.stdin.buffer.read(nbytes)
