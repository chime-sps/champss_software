import numpy as np
from sps_common.interfaces import SkyBeam, DedispersedTimeSeries
from sps_common.conversion import unix_to_mjd
from sps_common.constants import DM_CONSTANT, TSAMP
from sps_dedispersion.fdmt.cpu_fdmt import FDMT

def simple_dedisperse(fdmt, skybeam, dm_step=1):
    """Dedisperse the skybeam into a collection of dedispersed
    time series without chunking.

    Parameters
    =========
    fdmt: FDMT
        An initialized FDMT instance.
    skybeam: SkyBeam
        The skybeam to be dedispersed.
    dm_step: int
        Keep 1 DM for every dm_step DMs from the tree dedispersion.
    """
    # Delete the first maxdm samples, because 
    # data from some frequencies are missing for these samples
    dedisp = fdmt.fdmt(skybeam.spectra[::-1], frontpadding=False)[::dm_step]

    # At this point the entire returned array should still be in memory...
    # since the above line should only return a view, not a copy
    # of the memory.
    # Uncommenting the following line will fix this, but there will
    # temporarily be two copies of the dedisp array in memory.
    dedisp = dedisp.copy()

    dms = (np.arange(0, fdmt.maxDT, dm_step) /
        DM_CONSTANT / (1/fdmt.fmin**2 - 1/fdmt.fmax**2)) * TSAMP

    dts = DedispersedTimeSeries(dedisp_ts=dedisp, 
                                dms=dms, 
                                ra=skybeam.ra, 
                                dec=skybeam.dec, 
                                start_mjd=unix_to_mjd(skybeam.utc_start),
                                obs_id=skybeam.obs_id)

    return dts



def dedisperse(fdmt, skybeam, chunk_size, dm_step=1):
    """Dedisperse the skybeam into a collection of dedispersed
    time series with chunking.

    Parameters
    =========
    fdmt: FDMT
        An initialized FDMT instance.
    skybeam: SkyBeam
        The skybeam to be dedispersed.
    chunk_size: int
        The number of time samples per chunk beyond the minimum
        (this minimum is equal to the number of time samples'
        delay at max dm).
    dm_step: int
        Keep 1 DM for every dm_step DMs from the tree dedispersion.
    """
    if chunk_size <= fdmt.maxDT:
        raise ValueError('chunk_size is too small: is {}, '
                         'must be more than max_dms, {} (and should be'
                         'much larger)'.format(chunk_size, fdmt.maxDT))
    
    if chunk_size + fdmt.maxDT > skybeam.spectra.shape[1]:
        # requested chunk size large enough that 
        # chunking is not necessary
        return simple_dedisperse(fdmt, skybeam, dm_step=dm_step)

    spectra = skybeam.spectra[::-1]   # this returns a view

    # Chunking setup.
    dedisp = np.zeros(((fdmt.maxDT - 1) // dm_step + 1,
                      spectra.shape[1] - fdmt.maxDT),
                      dtype=spectra.dtype)
    # (the shape of dedisp is given by the final result of the slices below)

    idx = 0
    while (idx+1) * chunk_size + fdmt.maxDT < spectra.shape[1]:
        # while the end of the current chunk is still inside
        # the spectra array...

        # these are all views

        if idx == 0:
            dedisp_chunk = dedisp[:, :chunk_size]

            spectra_chunk = spectra[:, :chunk_size]
            # Don't add first fdmt.maxDT time samples because not
            # all frequencies are available for those time samples
            np.add(dedisp_chunk,
                   fdmt.fdmt(spectra_chunk, padding=True,
                             frontpadding=False)[::dm_step],
                   out=dedisp_chunk)
        else:
            dedisp_chunk = dedisp[:, idx*chunk_size - fdmt.maxDT:
                                     (idx+1)*chunk_size]

            spectra_chunk = spectra[:, idx*chunk_size : (idx+1)*chunk_size]
            np.add(dedisp_chunk,
                   fdmt.fdmt(spectra_chunk, padding=True)[::dm_step],
                    out=dedisp_chunk)

        idx += 1

    # now check for one last 'partial' chunk
    dedisp_chunk = dedisp[:, idx*chunk_size - fdmt.maxDT:]

    spectra_chunk = spectra[:, idx*chunk_size:]

    np.add(dedisp_chunk,
           fdmt.fdmt(spectra_chunk, padding=False)[::dm_step],
           out=dedisp_chunk)

    dms = (np.arange(0, fdmt.maxDT, dm_step) /
        DM_CONSTANT / (1/fdmt.fmin**2 - 1/fdmt.fmax**2)) * TSAMP

    dts = DedispersedTimeSeries(dedisp_ts=dedisp, 
                                dms=dms, 
                                ra=skybeam.ra, 
                                dec=skybeam.dec, 
                                start_mjd=unix_to_mjd(skybeam.utc_start),
                                obs_id=skybeam.obs_id)

    return dts



