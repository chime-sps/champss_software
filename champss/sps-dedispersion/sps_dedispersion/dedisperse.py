import numpy as np
from sps_common.constants import DM_CONSTANT, FREQ_BOTTOM, FREQ_TOP, TSAMP
from sps_common.conversion import unix_to_mjd
from sps_common.interfaces import DedispersedTimeSeries


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
    """
    if fdmt.startup_samples > maxDT:
        raise ValueError(
            f"FDMT needs {fdmt.startup_samples} startup samples, only dropping {maxDT}"
        )

    spectra = skybeam.spectra[::-1]  # this returns a view

    ntbin = spectra.shape[1]
    dedisp = np.zeros((len(dt_trials), ntbin - maxDT), dtype=spectra.dtype)
    block_out = np.empty((len(dt_trials), chunk_size), dtype=np.float32)

    for start in range(0, ntbin, chunk_size):
        block = spectra[:, start : start + chunk_size]
        stop = start + block.shape[1]
        if block.shape[1] < chunk_size:
            # Output samples only depend on earlier input, so padding the end is safe
            block = np.pad(block, ((0, 0), (0, chunk_size - block.shape[1])))
        fdmt.execute(block, out=block_out)
        first = max(start, maxDT)
        if first < stop:
            dedisp[:, first - maxDT : stop - maxDT] = block_out[
                :, first - start : stop - start
            ]

    return make_time_series(dedisp, dt_trials, skybeam, freq_bottom, freq_top, tsamp)


def make_time_series(dedisp, dt_trials, skybeam, freq_bottom, freq_top, tsamp):
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
    )

    return dts
