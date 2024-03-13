"""FDMT."""
from time import time
from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt


# Subband delta time or delay time
@numba.njit(parallel=True, boundscheck=False)  # type: ignore
def subDT(
    freqs: npt.NDArray[np.float64],
    freqs_stepsize: np.float64,
    min_freq_mhz: float,
    max_freq_mhz: float,
    max_time_samples: int,
) -> npt.NDArray[np.int32]:
    """Get needed DT of subband to yield maxDT over entire band

    Args:
        freqs (npt.NDArray[np.float32]): Frequency channels.
        freqs_stepsize (np.float32): Frequency step size.
        min_freq_mhz (float): Minimum frequency.
        max_freq_mhz (float): Maximum frequency.
        max_time_samples (int): Maximum time samples.

    Returns:
        npt.NDArray[np.int32]: Frequency channels.

    Yields:
        Iterator[npt.NDArray[np.int32]]: Frequency channels.
    """
    loc = np.power(freqs, -2.0) - np.power((freqs + freqs_stepsize), -2.0)
    glo = np.power(min_freq_mhz, -2.0) - np.power(max_freq_mhz, -2.0)
    dt: npt.NDArray[np.int32] = np.ceil((max_time_samples - 1) * loc / glo).astype(
        np.int32
    ) + np.int32(1)
    return dt


@numba.njit(parallel=True, boundscheck=False)  # type: ignore
def buildQ(
    freqs: npt.NDArray[np.float64],
    freqs_stepsize: np.float64,
    freq_channels: int,
    min_freq_mhz: float,
    max_freq_mhz: float,
    max_time_samples: int,
) -> npt.NDArray[np.int32]:
    """Build Q required for FDMT."""
    # Build matrices required for FDMT
    Q = np.zeros((int(np.log2(freq_channels)) + 1, freq_channels), dtype=np.int32)
    for idx in numba.prange(int(np.log2(freq_channels)) + 1):
        needed = subDT(
            freqs[:: 2**idx],
            freqs_stepsize * np.power(2, idx),
            min_freq_mhz,
            max_freq_mhz,
            max_time_samples,
        )
        Q[idx, : len(needed)] = np.cumsum(needed) - needed
    return Q


@numba.njit(parallel=True, boundscheck=False)  # type: ignore
def fdmt_iter_par(
    fs: npt.NDArray[np.float64],
    nchan: int,
    df: float,
    Q: npt.NDArray[np.int32],
    src: npt.NDArray[np.float32],
    dest: npt.NDArray[np.float32],
    i: int,
    fmin: float,
    fmax: float,
    maxDT: int,
    threads: int,
) -> npt.NDArray[np.float32]:
    """
    Perform a single iteration of the Fast Dispersion Measure Transform (FDMT)

    Args:
        fs (npt.NDArray[np.float64]): Array of center frequencies for each channel.
        nchan (int): Number of frequency channels.
        df (float): Frequency resolution.
        Q (npt.NDArray[np.int32]): List of indices for each frequency channel.
        src (npt.NDArray[np.float32]): Input data array.
        dest (npt.NDArray[np.float32]): Output data array.
        i (int): Iteration index.
        fmin (float): Minimum frequency.
        fmax (float): Maximum frequency.
        maxDT (int): Maximum time delay.
        threads (int): Number of threads to use for parallelization.
    """

    numba.set_num_threads(threads)  # type: ignore
    T = src.shape[1]
    dF: float = df * 2**i
    f_starts: npt.NDArray[np.float64] = fs[:: 2**i]
    f_ends = f_starts + dF
    f_mids = fs[2 ** (i - 1) :: 2**i]
    for i_F in range(nchan // 2**i):
        f0 = f_starts[i_F]
        f1 = f_mids[i_F]
        f2 = f_ends[i_F]
        # Using cor = df seems to give the best behaviour at high DMs, judging from
        # presto output. Nevertheless it may be worth adjusting this option
        # for a specific use case
        cor = df if i > 1 else 0

        C = (f1**-2 - f0**-2) / (f2**-2 - f0**-2)
        C01 = ((f1 - cor) ** -2 - f0**-2) / (f2**-2 - f0**-2)
        C12 = ((f1 + cor) ** -2 - f0**-2) / (f2**-2 - f0**-2)

        # SDT
        loc = f0**-2 - (f0 + dF) ** -2
        glo = fmin**-2 - fmax**-2
        R = int((maxDT - 1) * loc / glo) + 2

        for i_dT in numba.prange(0, R):
            dT_mid01 = round(i_dT * C01)
            dT_mid12 = round(i_dT * C12)
            dT_rest = i_dT - dT_mid12
            dest[Q[i][i_F] + i_dT, :] = src[Q[i - 1][2 * i_F] + dT_mid01, :]
            dest[Q[i][i_F] + i_dT, dT_mid12:] += src[
                Q[i - 1][2 * i_F + 1] + dT_rest, : T - dT_mid12
            ]

    return dest

#@numba.njit(boundscheck=False)  # type: ignore
def fdmt(
    spectra: npt.NDArray[np.float32],
    min_freq_mhz: float = 400.1953125,
    max_freq_mhz: float = 800.1953125,
    freq_channels: int = 4096,
    max_time_samples: int = 2048,
    frontpadding: bool = False,
    backpadding: bool = False,
    threads: int = 4,
) -> npt.NDArray[np.float32]:
    """Perform the Fast Dispersion Measure Transform (FDMT).

    Args:
        spectra (npt.NDArray[np.float32]): Intensity spectra.
        min_freq_mhz (float, optional): Minimum Frequency.
            Defaults to 400.1953125.
        max_freq_mhz (float, optional): Maximum Frequency.
            Defaults to 800.1953125.
        freq_channels (int, optional): Frequency Channels.
            Defaults to 1024.
        max_time_samples (int, optional): Frequency Channels.
            Defaults to 2048.
        frontpadding (bool, optional): Whether to pad the front.
            Defaults to True.
        backpadding (bool, optional): Whether to pad the back.
            Defaults to False.
        threads (int, optional): Number of Numba threads to use.
            Defaults to 1.

    Returns:
        npt.NDArray[np.float32]: Dedispersed time series.
    """
    ### removed like
    freqs: npt.NDArray[np.float32] = np.zeros(
        freq_channels, dtype=np.float32, #like=np.empty_like(np.float32)
    )
    freqs_stepsize: np.float32 = np.float32(np.NAN)
    # Compute Frequencies and Frequency Step Size

    ### removed retstep, endpoint, dtype
    freqs = np.linspace(
        min_freq_mhz,
        max_freq_mhz,
        freq_channels,
    )
    freqs_stepsize = freqs[1] - freqs[0]

    chDTs = subDT(freqs, freqs_stepsize, min_freq_mhz, max_freq_mhz, max_time_samples)

    # Build matrices required for FDMT
    columns = spectra.shape[1]
    rows_A = chDTs.sum(axis=0, dtype=np.int32)  # type: ignore
    ### numba.empty requires tuple for 2D array axes instead of a list
    A: npt.NDArray[np.float32] = np.zeros((rows_A, columns), dtype=np.float32)
    rows_B = subDT(freqs[::2], freqs_stepsize * 2, min_freq_mhz, max_freq_mhz, max_time_samples).sum(axis=0, dtype=np.int32)  # type: ignore
    B: npt.NDArray[np.float32] = np.zeros((rows_B, columns), dtype=np.float32)
    # A and B are the matrices used in the FDMT algorithm
    Q = buildQ(
        freqs=freqs,
        freqs_stepsize=freqs_stepsize,
        freq_channels=freq_channels,
        min_freq_mhz=min_freq_mhz,
        max_freq_mhz=max_freq_mhz,
        max_time_samples=max_time_samples,
    )

    A[Q[0], :] = spectra
    commonDTs: npt.NDArray[np.int32] = np.ones(chDTs.min() - 1, dtype=np.int32) * spectra.shape[1]  # type: ignore
    DTsteps: npt.NDArray[np.int32] = np.where(chDTs[:-1] - chDTs[1:] != 0)[0]
    DTplan: npt.NDArray[np.int32] = np.concatenate( (commonDTs, DTsteps[::-1]) )

    for i, t in enumerate(DTplan, 1):
        A[Q[0][:t] + i, i:] = A[Q[0][:t] + i - 1, i:] + spectra[:t, :-i]
    for i, t in enumerate(DTplan, 1):
        # A[Q[0][:t]+i,i:] /= int(i+1)
        A[Q[0][:t] + i, i:] /= int(i + 1)

    # for idx, timestep in np.ndenumerate(DTplan):
    #     idx = idx[0] + 1
    #     A[Q[0][:timestep] + idx, idx:] = (
    #         A[Q[0][:timestep] + idx - 1, idx:] + spectra[:timestep, : - idx]
    #     )
    # for idx, timestep in np.ndenumerate(DTplan):
    #      idx = idx[0] + 1
    #      A[Q[0][:timestep] + idx, idx:] /= int(idx + 1)

    for i in range(1, int(np.log2(freq_channels)) + 1):
        src, dest = (A, B) if (i % 2 == 1) else (B, A)
        dest = fdmt_iter_par(
            fs=freqs,
            nchan=freq_channels,
            df=freqs_stepsize,
            Q=Q,
            src=src,
            dest=dest,
            i=i,
            fmin=min_freq_mhz,
            fmax=max_freq_mhz,
            maxDT=max_time_samples,
            threads=threads,
        )

    return dest[:max_time_samples]#[:, max_time_samples:]


if __name__ == "__main__":
    data = np.random.normal(size=(4096, 40960))
    data = data.astype(np.float32)

    max_time_samples = 10484
    padding = ((0, 0), (0, max_time_samples))
    data = np.pad(data, padding, mode="constant", constant_values=np.float32(0.0))

    start = time()
    fdmt(data, max_time_samples=max_time_samples)
    end = time()
    print(f"Compile Iteration: {end - start}s")
    fdmt(data, max_time_samples=max_time_samples)
    end2 = time()
    print(f"Execution Iteration: {end2 - end}s")
