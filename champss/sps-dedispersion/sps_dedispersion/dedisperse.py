import numpy as np
from sps_common.constants import DM_CONSTANT, FREQ_BOTTOM, FREQ_TOP, TSAMP
from sps_common.conversion import unix_to_mjd
from sps_common.interfaces import DedispersedTimeSeries


def dedisperse(fdmt, skybeam, chunk_size, maxDT, dm_step=1, cpp=True):
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
    """

    spectra = skybeam.spectra[::-1]  # this returns a view

    # Chunking setup
    ntbin = spectra.shape[1]
    Nchunk = (ntbin - maxDT) // chunk_size + 1
    dedisp = np.zeros(
        ((maxDT - 1) // dm_step + 1, Nchunk * chunk_size), dtype=spectra.dtype
    )

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
            )[::dm_step, maxDT:][:dedisp.shape[0]]
        else:
            dedisp[:, i * chunk_size : (i + 1) * chunk_size] = fdmt.fdmt(
                spectra_chunk, padding=True, frontpadding=False
            )[::dm_step, maxDT:]

    dedisp = dedisp[:, :-npad]
    dms = (
        np.arange(0, maxDT, dm_step)
        / DM_CONSTANT
        / (1 / FREQ_BOTTOM**2 - 1 / FREQ_TOP**2)
    ) * TSAMP

    dts = DedispersedTimeSeries(
        dedisp_ts=dedisp,
        dms=dms,
        ra=skybeam.ra,
        dec=skybeam.dec,
        start_mjd=unix_to_mjd(skybeam.utc_start),
        obs_id=skybeam.obs_id,
    )

    return dts
