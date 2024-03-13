#!/usr/bin/env python3

from time import time

import numpy as np
from attr import attrib, attrs, cmp_using

from fdmt.fdmt_njit import fdmt_iter_par


@attrs
class FDMT:
    """
    Collection of attributes and helper arrays necessary for dispersion using FDMT.

    Parameters
    ---------
    fmin: float
        Frequency of the lowest edge of the band, in MHz
    fmax: float
        Frequency of the highest edge of the band, in MHz
    nchan: int
        Number of frequency channels
    maxDT: int
        Number of time samples corresponding to the maximum delay between the
        top and bottom of the band (defines the maximum DM of the search)
    num_threads: int
        Number of threads to use for parallelization. If 1, no parallelization
    """

    fmin: float = attrib(default=400.1953125)
    fmax: float = attrib(default=800.1953125)
    nchan: int = attrib(default=1024)
    maxDT: int = attrib(default=2048)
    num_threads: int = attrib(default=1)
    A: np.ndarray = attrib(default=None, eq=cmp_using(eq=np.array_equal))
    B: np.ndarray = attrib(default=None, eq=cmp_using(eq=np.array_equal))
    df: float = attrib(init=False)
    fs: np.ndarray = attrib(init=False, eq=cmp_using(eq=np.array_equal))
    Q: list = attrib(init=False, eq=cmp_using(eq=np.array_equal))

    def __attrs_post_init__(self):
        self.fs, self.df = fs, df = np.linspace(
            self.fmin, self.fmax, self.nchan, endpoint=False, retstep=True
        )

    def subDT(self, f, dF=None):
        "Get needed DT of subband to yield maxDT over entire band"
        if dF is None:
            dF = self.df
        loc = f**-2 - (f + dF) ** -2
        glo = self.fmin**-2 - self.fmax**-2
        return np.ceil((self.maxDT - 1) * loc / glo).astype(int) + 1

    def buildAB(self, numCols, dtype=np.uint32):
        numRowsA = (self.subDT(self.fs)).sum()
        numRowsB = (self.subDT(self.fs[::2], self.fs[2] - self.fs[0])).sum()
        self.A = np.zeros([numRowsA, numCols], dtype)
        self.B = np.zeros([numRowsB, numCols], dtype)

    def buildQ(self):
        for i in range(int(np.log2(self.nchan)) + 1):
            needed = self.subDT(self.fs[:: 2**i], self.df * 2**i)
            if i == 0:
                self.Q = np.zeros(
                    (int(np.log2(self.nchan)) + 1, len(needed)), dtype="int32"
                )
            self.Q[i, : len(needed)] = np.cumsum(needed) - needed

    def prep(self, cols, dtype=np.uint32):
        "Prepares necessary matrices for FDMT"
        self.buildAB(cols, dtype=dtype)
        self.buildQ()

    def fdmt(self, I, retDMT=False, verbose=False, padding=False, frontpadding=True):
        """
        Computes DM Transform. If retDMT returns transform, else returns max sigma.

        Parameters
        ==========
        I: np.ndarray
            The array of spectra to be dedispersed. Must have shape (nchan, nsamp)
            where nchan is the number of frequency channels and nsamp is the number
            of time samples. The 0th channel must be the lowest frequency channel.
        verbose: bool
            Whether to print info to stdout.
        padding: bool
            Whether to return additional, incompletely integrated time samples
            at the end of the dedispersed time series.
        frontpadding: bool
            Whether to return additional, incompletely integrated time samples
            at the beginning of the dedispersed time series.
        retDMT: bool
            Whether to return the DM transform instead of the SNRs.
        num_threads: int
            Number of threads to use for parallelization. If 1, no parallelization

        Returns
        =======
        dedisp: np.ndarray
            An array of dedispersed time series. Has shape
            (n_dms, nsamp_dedisp), where nsamp_dedisp is given as follows:
                - if padding and frontpadding:
                    nsamp_dedisp = nsamp + maxDT
                - if padding and not frontpadding:
                    nsamp_dedisp = nsamp
                - if not padding and frontpadding:
                    nsamp_dedisp = nsamp
                - if not padding and not frontpadding:
                    nsamp_dedisp = nsamp - maxDT
            Only returned if retDMT is True.
        sigmi: float
            The SNR of the most prominent feature in the DMT.
            Only returned if retDMT is False.
        """

        if I.dtype.itemsize < 4:
            I = I.astype(np.uint32)

        # Concatenate some zero padding arrays (but there may be zero such
        # arrays)
        concat_tuple = [I]
        if padding:
            concat_tuple.append(np.zeros((self.nchan, self.maxDT), dtype=I.dtype))
        if frontpadding:
            # We need to add maxDT time samples to the front of I when
            # frontpadding is desired, due to edge-related effects
            concat_tuple.insert(0, np.zeros((self.nchan, self.maxDT), dtype=I.dtype))

        I = np.concatenate(concat_tuple, axis=1)

        if (
            self.A is None
            or self.A.shape[1] != I.shape[1]
            or self.A.dtype != I.dtype
            or True
        ):
            self.prep(I.shape[1], dtype=I.dtype)

        t1 = time()
        self.fdmt_initialize(I)

        t2 = time()
        for i in range(1, int(np.log2(self.nchan)) + 1):
            src, dest = (self.A, self.B) if (i % 2 == 1) else (self.B, self.A)
            self.fdmt_iteration(src, dest, i)

        if verbose:
            t3 = time()
            print("Initializing time:  %.2f s" % (t2 - t1))
            print("Iterating time:  %.2f s" % (t3 - t2))
            print("Total time: %.2f s" % (t3 - t1))

        DMT = dest[: self.maxDT]

        if retDMT:
            # We need to cut off the first maxDT samples either way
            # because now frontpadding works by inserting maxDT samples' worth
            # of zeros at the front of I
            return DMT[:, self.maxDT :]
        noiseRMS = np.array([DMT[i, i:].std() for i in range(self.maxDT)])
        noiseMean = np.array([DMT[i, i:].mean() for i in range(self.maxDT)])
        sigmi = (DMT.T - noiseMean) / noiseRMS
        if verbose:
            print("Maximum sigma value: %.3f" % sigmi.max())
        return sigmi.max()

    def fdmt_initialize(self, I):
        self.A[self.Q[0], :] = I
        chDTs = self.subDT(self.fs)
        T = I.shape[1]
        commonDTs = [T for _ in range(1, chDTs.min())]
        DTsteps = list(np.where(chDTs[:-1] - chDTs[1:] != 0)[0])
        DTplan = commonDTs + DTsteps[::-1]
        for i, t in enumerate(DTplan, 1):
            self.A[self.Q[0][:t] + i, i:] = (
                self.A[self.Q[0][:t] + i - 1, i:] + I[:t, :-i]
            )
        for i, t in enumerate(DTplan, 1):
            # A[Q[0][:t]+i,i:] /= int(i+1)
            self.A[self.Q[0][:t] + i, i:] /= int(i + 1)

    def fdmt_iteration(self, src, dest, i):
        df = self.df
        fs = self.fs
        nchan = self.nchan
        Q = self.Q
        subDT = self.subDT
        fmin = self.fmin
        fmax = self.fmax
        maxDT = self.maxDT
        num_threads = self.num_threads

        fdmt_iter_par(
            fs, nchan, df, Q, src, dest, i, fmin, fmax, np.float32(maxDT), num_threads
        )

    def reset_ABQ(self):
        self.A = None
        self.B = None
        self.Q = []

    def recursive_fdmt(self, I, depth=0, curMax=0):
        """Performs FDMT, downsamples and repeats recursively, returning max sigma I
        should have shape (nchan, nsamp) where nsamp is the number of time samples.
        """
        curMax = max(curMax, fdmt(I))
        if depth <= 0:
            return curMax
        else:
            I2 = (
                I[:, ::2] + I[:, 1::2]
                if (I.shape[1] % 2 == 0)
                else I[:, :-1:2] + I[:, 1::2]
            )
            return self.recursive_fdmt(I2, depth - 1, curMax)
