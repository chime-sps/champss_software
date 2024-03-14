from numba import njit, prange, set_num_threads


@njit(parallel=True)
def fdmt_iter_par(fs, nchan, df, Q, src, dest, i, fmin, fmax, maxDT, num_threads):
    """
    Perform a single iteration of the Fast Dispersion Measure Transform (FDMT)
    algorithm, parallelized using numba.

    Parameters:
    fs (ndarray): Array of center frequencies for each channel.
    nchan (int): Number of frequency channels.
    df (float): Frequency resolution.
    Q (list): List of indices for each frequency channel.
    src (ndarray): Input data array.
    dest (ndarray): Output data array.
    i (int): Iteration index.
    fmin (float): Minimum frequency.
    fmax (float): Maximum frequency.
    maxDT (int): Maximum time delay.
    num_threads (int): Number of threads to use for parallelization.

    Returns: None
    """

    set_num_threads(num_threads)
    T = src.shape[1]
    dF = df * 2**i
    f_starts = fs[:: 2**i]
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

        for i_dT in prange(0, R):
            dT_mid01 = round(i_dT * C01)
            dT_mid12 = round(i_dT * C12)
            dT_rest = i_dT - dT_mid12
            dest[Q[i][i_F] + i_dT, :] = src[Q[i - 1][2 * i_F] + dT_mid01, :]
            dest[Q[i][i_F] + i_dT, dT_mid12:] += src[
                Q[i - 1][2 * i_F + 1] + dT_rest, : T - dT_mid12
            ]
