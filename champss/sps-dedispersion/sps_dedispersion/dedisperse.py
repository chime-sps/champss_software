import numpy as np
from sps_common.constants import DM_CONSTANT, FREQ_BOTTOM, FREQ_TOP, TSAMP
from sps_common.conversion import unix_to_mjd
from sps_common.interfaces import DedispersedTimeSeries
from sps_common.sm_utils import RowPrefixReleaser, empty_shared_array


def dedisperse(
    fdmt,
    skybeam,
    chunk_size,
    maxDT,
    dm_step=1,
    cpp=True,
    freq_bottom=FREQ_BOTTOM,
    freq_top=FREQ_TOP,
    tsamp=TSAMP,
    dt_trials=None,
):
    """
    Dedisperse the skybeam into a collection of dedispersed time series with chunking.

    Parameters
    =========
    fdmt: FDMT
        An initialized FDMT instance.
    skybeam: SkyBeam
        The skybeam to be dedispersed.
    chunk_size: int
        The number of time samples per chunk with overlap of maxDT.
    maxDT: int
        The delay at max dm in units of time samples.
    dm_step: int
        Integer spacing of DMsteps to compute, in units of time samples across the band.
    dt_trials: np.ndarray
        Sorted integer delays, in time samples across the band, to keep instead of the
        uniform grid set by dm_step. All must be less than maxDT.
    """

    if getattr(skybeam, "channels", None) is not None:
        spectra = skybeam.full_spectra()[::-1]  # expanded to every channel
    else:
        spectra = skybeam.spectra[::-1]  # this returns a view

    # Chunking setup
    ntbin = spectra.shape[1]
    Nchunk = (ntbin - maxDT) // chunk_size + 1
    if dt_trials is None:
        rows = np.arange(0, maxDT, dm_step)
    else:
        rows = np.asarray(dt_trials)
    dedisp = np.zeros((len(rows), Nchunk * chunk_size), dtype=spectra.dtype)

    for i in range(Nchunk):
        # these are all views
        spectra_chunk = spectra[:, i * chunk_size : (i + 1) * chunk_size + maxDT]
        if i == Nchunk - 1:
            npad = (chunk_size + maxDT) - spectra_chunk.shape[1]
            spectra_chunk = np.pad(spectra_chunk, ((0, 0), (0, npad)), mode="constant")

        # Keeping every 2 steps instead of compute every 2 steps, otherwise there is induced time smearing
        if cpp:
            dedisp[:, i * chunk_size : (i + 1) * chunk_size] = fdmt.execute(
                spectra_chunk
            )[rows, maxDT:]
        else:
            dedisp[:, i * chunk_size : (i + 1) * chunk_size] = fdmt.fdmt(
                spectra_chunk, padding=True, frontpadding=False
            )[rows, maxDT:]

    dedisp = dedisp[:, :-npad]

    return make_time_series(dedisp, rows, skybeam, freq_bottom, freq_top, tsamp)


def dedisperse_streaming(
    fdmt,
    skybeam,
    chunk_size,
    maxDT,
    dt_trials,
    freq_bottom=FREQ_BOTTOM,
    freq_top=FREQ_TOP,
    tsamp=TSAMP,
    shared=False,
    release_input=False,
    expand_threads=1,
):
    """
    Dedisperse the skybeam with a dmtlib FDMT in its streaming ("valid") mode.

    Blocks join through the FDMT's internal history, so no overlap is recomputed. The
    first maxDT output samples are dropped, which gives the same length and start time
    as `dedisperse`.

    Parameters
    =========
    fdmt: dmtlib.FDMT
        A new instance planned for chunk_size samples and the delays in dt_trials.
    skybeam: SkyBeam
        The skybeam to be dedispersed.
    chunk_size: int
        The number of time samples per block.
    maxDT: int
        The number of samples to drop from the start. No trial may exceed it.
    dt_trials: np.ndarray
        Sorted integer delays computed by the FDMT, in time samples across the band.
    shared: bool
        Allocate the output in shared memory, so power spectrum workers can read it
        without a copy. It is unlinked when the returned object is garbage collected.
    release_input: bool
        Free the memory of each block of the skybeam's spectra once the FDMT has read it,
        so input and output are not both held in full (see RowPrefixReleaser). Only for
        spectra in shared memory that nothing reads afterwards: freed samples read as
        zeros. Ignored for other arrays.
    expand_threads: int
        For a skybeam of 8-bit codes (skybeam.coded): threads that expand each block's
        rows to float32, the next block being expanded while the FDMT runs on this one.
    """
    if fdmt.startup_samples > maxDT:
        raise ValueError(
            f"FDMT needs {fdmt.startup_samples} startup samples, only dropping {maxDT}"
        )

    channels = getattr(skybeam, "channels", None)
    coded = getattr(skybeam, "coded", None)
    releaser = RowPrefixReleaser(skybeam.spectra) if release_input else None
    if coded is not None:
        # 8-bit codes and tables (spectra holds the codes): each block is expanded to
        # float32 at full width, highest frequency first; releasing frees the codes
        from beamformer.coded import expand_values_into

        from concurrent.futures import ThreadPoolExecutor

        spectra = skybeam.spectra
        rows = np.arange(spectra.shape[0]) if channels is None else np.asarray(channels)
        flipped_rows = skybeam.nchan - 1 - rows
        # Two full-width buffers: the FDMT reads one while the next block fills the other
        buffers = [np.zeros((skybeam.nchan, chunk_size), dtype=np.float32) for _ in range(2)]
        nthreads = max(1, int(expand_threads))
        bounds = np.linspace(0, len(rows), nthreads + 1).astype(int)
        groups = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        row_pool = ThreadPoolExecutor(len(groups))
        prefetcher = ThreadPoolExecutor(1)

        def expand_block(buf, start, stop):
            tasks = [
                row_pool.submit(
                    expand_values_into, coded.codes[a:b], coded.parts[a:b],
                    coded.params[a:b], coded.rowp[a:b], *coded.layout, start, stop,
                    buf, flipped_rows[a:b],
                )  # fmt: skip
                for a, b in groups
            ]
            for task in tasks:
                task.result()
            buf[flipped_rows, stop - start :] = 0  # padding of the last block
            return buf
    elif channels is None:
        spectra = skybeam.spectra[::-1]  # this returns a view
    else:
        # The skybeam holds only some channels (the rest are zero): each block is built
        # at full width, highest frequency first, so the FDMT sees exactly what it would
        spectra = skybeam.spectra
        flipped_rows = skybeam.nchan - 1 - np.asarray(channels)
        full_block = np.zeros((skybeam.nchan, chunk_size), dtype=spectra.dtype)

    ntbin = spectra.shape[1]
    out_dtype = np.float32 if coded is not None else spectra.dtype
    shape = (len(dt_trials), ntbin - maxDT)
    shm = None
    if shared:
        dedisp, shm = empty_shared_array(shape, out_dtype)
    else:
        dedisp = np.zeros(shape, dtype=out_dtype)
    block_out = np.empty((len(dt_trials), chunk_size), dtype=np.float32)

    if coded is not None:
        pending = prefetcher.submit(expand_block, buffers[0], 0, min(chunk_size, ntbin))
    for k, start in enumerate(range(0, ntbin, chunk_size)):
        if coded is not None:
            stop = min(start + chunk_size, ntbin)
            block = pending.result()
            if stop < ntbin:
                pending = prefetcher.submit(
                    expand_block, buffers[(k + 1) % 2], stop, min(stop + chunk_size, ntbin)
                )
        elif channels is None:
            block = spectra[:, start : start + chunk_size]
            stop = start + block.shape[1]
            if block.shape[1] < chunk_size:
                # Output samples only depend on earlier input, so padding the end is safe
                block = np.pad(block, ((0, 0), (0, chunk_size - block.shape[1])))
        else:
            stop = min(start + chunk_size, ntbin)
            full_block[flipped_rows, : stop - start] = spectra[:, start:stop]
            full_block[flipped_rows, stop - start :] = 0  # padding of the last block
            block = full_block
        fdmt.execute(block, out=block_out)
        if releaser is not None:
            # The FDMT keeps what it needs of earlier blocks in its own history
            releaser.release_until(stop)
        first = max(start, maxDT)
        if first < stop:
            dedisp[:, first - maxDT : stop - maxDT] = block_out[
                :, first - start : stop - start
            ]
    if coded is not None:
        prefetcher.shutdown()
        row_pool.shutdown()

    return make_time_series(
        dedisp, dt_trials, skybeam, freq_bottom, freq_top, tsamp, shared_memory=shm
    )


def make_time_series(
    dedisp, dt_trials, skybeam, freq_bottom, freq_top, tsamp, shared_memory=None
):
    """Wrap dedispersed data, labelled by delay across the band, with its DMs."""
    dms = (
        np.asarray(dt_trials)
        / DM_CONSTANT
        / (1 / freq_bottom**2 - 1 / freq_top**2)
    ) * tsamp

    dts = DedispersedTimeSeries(
        dedisp_ts=dedisp,
        dms=dms,
        ra=skybeam.ra,
        dec=skybeam.dec,
        start_mjd=unix_to_mjd(skybeam.utc_start),
        obs_id=skybeam.obs_id,
        shared_memory=shared_memory,
    )

    return dts
