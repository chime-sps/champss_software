import your
import numpy as np
from omegaconf import OmegaConf
from astropy.time import Time
import multiprocessing
import click
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from astropy.coordinates import EarthLocation
import astropy.units as u

log_stream = logging.StreamHandler()
logging.root.addHandler(log_stream)
log = logging.getLogger(__name__)

from sps_pipeline.config_utils import load_config
from sps_common.constants import DM_CONSTANT
from sps_dedispersion.dedisperse import dedisperse
from sps_common.interfaces.beamformer import SkyBeam
from dmt.libdmt import FDMT
from ps_processes.processes.ps import PowerSpectraCreation
from ps_processes.processes.ps_search import PowerSpectraSearch
from candidate_processor.feature_generator import Features
from ps_processes.processes.ps_stack import PowerSpectraStack


def psrfits_start_mjd(observation):
    """
    Hard coded utility to find the start date of a filterbank file.
    This is necessary because both `your`
    and PRESTO assume NSUBOFFS is NOT cumulative, but Murriyang uses a cumulative
    count.
    """
    fits = observation.fits
    primary = fits[0].header
    subint = fits["SUBINT"]
    tsub = subint.header["TBIN"] * subint.header["NSBLK"]
    start_subint = int(subint.header.get("NSUBOFFS", 0))
    names = subint.columns.names
    if "OFFS_SUB" in names:
        offs_sub = float(subint.data["OFFS_SUB"][0])
        if offs_sub != 0.0:
            start_subint = int((offs_sub - 0.5 * tsub) / tsub + 1e-7)
        elif "INDEXVAL" in names:
            start_subint = int(float(subint.data["INDEXVAL"][0]) + 1e-7 - 1.0)
        else:
            start_subint = 0
    stt = primary["STT_IMJD"] + (primary["STT_SMJD"] + primary["STT_OFFS"]) / 86400.0
    return stt + start_subint * tsub / 86400.0


def read_rfifind_mask(mask_file, observation, chans=slice(None), tstart=None):
    """
    Reads in a PRESTO rfifind .mask file.
    NOTE: only reads in the section that overlaps with the .sf file.
    This is largely yoinked from PRESTO's python parser.
    """
    with open(mask_file, "rb") as x:
        time_sig, freq_sig, MJD, dtint, lofreq, df = np.fromfile(
            x, dtype=np.float64, count=6
        )
        nchan, nint, ptsperint = np.fromfile(x, dtype=np.int32, count=3)
        nzap = np.fromfile(x, dtype=np.int32, count=1)[0]
        if nzap:
            mask_zap_chans = np.fromfile(x, dtype=np.int32, count=nzap)
        else:
            mask_zap_chans = np.asarray([], dtype=np.int32)
        if len(mask_zap_chans) == nchan:
            log.warning("All channels recommended for masking!")
        nzap = np.fromfile(x, dtype=np.int32, count=1)[0]
        if nzap:
            mask_zap_ints = np.fromfile(x, dtype=np.int32, count=nzap)
        else:
            mask_zap_ints = np.asarray([], dtype=np.int32)
        if len(mask_zap_ints) == nint:
            log.warning("All intervals recommended for masking!")
        nzap_per_int = np.fromfile(x, dtype=np.int32, count=nint)
        mask_zap_chans_per_int = []
        for nzap in nzap_per_int:
            if nzap:
                if nzap == nchan:
                    tozap = np.arange(0, nchan, dtype=np.int32)
                else:
                    tozap = np.fromfile(x, dtype=np.int32, count=nzap)
            else:
                tozap = np.asarray([], dtype=np.int32)
            mask_zap_chans_per_int.append(tozap)

    nchan, nint, ptsperint = int(nchan), int(nint), int(ptsperint)

    zap = np.zeros((nint, nchan), dtype=bool)
    for i, mchans in enumerate(mask_zap_chans_per_int):
        zap[i, mchans] = True
    zap[:, mask_zap_chans] = True
    zap[mask_zap_ints, :] = True

    hdr = observation.your_header
    if tstart is None:
        tstart = hdr.tstart

    # find overlap in time
    if not np.isclose(dtint, ptsperint * hdr.tsamp, rtol=1e-3):
        raise ValueError("Mask - .sf file time resolution mismatch.")
    offset = int(round((tstart - MJD) * 86400.0 / hdr.tsamp))
    mask_nsamp = nint * ptsperint
    if offset < 0:
        raise ValueError(
            f"File starts {-offset * hdr.tsamp:.2f} s before the rfifind mask "
        )
    if offset >= mask_nsamp:
        raise ValueError(
            f"File starts {(offset - mask_nsamp) * hdr.tsamp:.2f} s after the end of "
            f"the rfifind mask."
        )
    i0 = offset // ptsperint
    i1 = min(-(-(offset + hdr.nspectra) // ptsperint), nint)
    sample_offset = offset - i0 * ptsperint
    uncovered = offset + hdr.nspectra - mask_nsamp

    # now find frequency overlap
    if not np.isclose(abs(df), abs(hdr.foff), rtol=1e-6):
        raise ValueError(
            f"rfifind channel width {df} MHz != data channel width {abs(hdr.foff)} MHz."
        )
    data_freqs = hdr.fch1 + hdr.foff * np.arange(hdr.nchans)[chans]
    pos = (data_freqs - lofreq) / abs(df)
    mask_idx = np.rint(pos).astype(int)
    if not np.allclose(pos, mask_idx, atol=0.01):
        raise ValueError(
            "rfifind channel centres do not line up with the data channel centres "
        )
    if mask_idx.min() < 0 or mask_idx.max() >= nchan:
        raise ValueError(
            f"Selected band {data_freqs.min():.3f}-{data_freqs.max():.3f} MHz "
            f"(channel centres) extends beyond the rfifind mask band "
        )

    log.info(
        f"Using rfifind intervals {i0}-{i1 - 1} of {nint} and mask channels "
        f"{mask_idx.min()}-{mask_idx.max()} of {nchan}."
    )
    return zap[i0:i1][:, mask_idx], ptsperint, sample_offset


def select_channels(hdr, fmin=None, fmax=None):
    """
    Select channels where fmin < f < fmax.
    """
    chan_bw = abs(hdr.foff)
    centers = hdr.fch1 + hdr.foff * np.arange(hdr.nchans)
    lo = -np.inf if fmin is None else fmin
    hi = np.inf if fmax is None else fmax
    tol = 1e-6 * chan_bw
    keep = np.flatnonzero(
        (centers - chan_bw / 2 >= lo - tol) & (centers + chan_bw / 2 <= hi + tol)
    )
    if keep.size == 0:
        raise click.BadParameter(f"No channels lie within {fmin}-{fmax}. ")
    c0, c1 = int(keep[0]), int(keep[-1]) + 1
    f_lo = float(centers[c0:c1].min() - chan_bw / 2)
    f_hi = float(centers[c0:c1].max() + chan_bw / 2)
    return c0, c1, f_lo, f_hi


def read_raw_block(observation, start, nsamp, chans=slice(None)):
    """
    Read nsamp samples starting at start in the file's native dtype.

    Returns an array of shape (nsamp, nchan).
    """
    if observation.format == "fits":
        return observation.formatclass.get_data(observation, start, nsamp)[:, 0, chans]
    return observation.get_data(start, nsamp)[:, chans]


def downsample_block(block, tdec):
    """
    Average groups of tdec time samples. Integer data is summed in an integer
    accumulator first, which is much faster than converting to float32 first.
    """
    n, nchan = block.shape
    if tdec == 1:
        return block.astype(np.float32, copy=block.dtype != np.float32)
    grouped = block.reshape(n // tdec, tdec, nchan)
    if (
        np.issubdtype(block.dtype, np.unsignedinteger)
        and block.dtype.itemsize <= 2
        and tdec < 2**16
    ):
        out = grouped.sum(axis=1, dtype=np.uint32).astype(np.float32)
    else:
        out = grouped.sum(axis=1, dtype=np.float64).astype(np.float32)
    out *= np.float32(1.0 / tdec)
    return out


def fill_non_finite(block):
    """
    Replace non-finite samples with the mean of the finite samples in their channel.
    Only needed for float data; integer data cannot be non-finite.
    """
    if not np.issubdtype(block.dtype, np.floating):
        return block, 0
    bad = ~np.isfinite(block)
    n_bad = int(bad.sum())
    if n_bad:
        block = block.astype(np.float32, copy=True)
        good = ~bad
        fill = np.where(good, block, 0).sum(axis=0, dtype=np.float64) / np.maximum(
            good.sum(axis=0), 1
        )
        np.copyto(
            block, np.broadcast_to(fill.astype(np.float32), block.shape), where=bad
        )
    return block, n_bad


def apply_rfi_mask(block, start, tdec, rfi_mask):
    """
    Apply the rfifind mask to a downsampled block in place.

    block has shape (nout, nchan); output sample j covers input samples
    [start + j*tdec, start + (j+1)*tdec). An output sample is masked in a channel
    if any input sample it covers is masked. Masked values are replaced by the
    channel's mean over the unmasked output samples of this block.

    Works on runs of output samples that share the same rfifind intervals, so the
    cost scales with the number of intervals instead of the number of samples.

    Returns the number of masked input samples (masked output samples * tdec).
    """
    zap, ptsperint, sample_offset = rfi_mask
    nout, nchan = block.shape
    last_int = zap.shape[0] - 1
    first = sample_offset + start + np.arange(nout) * tdec
    int_first = np.minimum(first // ptsperint, last_int)
    int_last = np.minimum((first + tdec - 1) // ptsperint, last_int)

    # boundaries of runs with identical (int_first, int_last)
    change = np.flatnonzero((np.diff(int_first) != 0) | (np.diff(int_last) != 0)) + 1
    edges = np.concatenate(([0], change, [nout]))

    runs = []
    masked_sum = np.zeros(nchan, dtype=np.float64)
    masked_count = np.zeros(nchan, dtype=np.int64)
    for r0, r1 in zip(edges[:-1], edges[1:]):
        chan_mask = zap[int_first[r0]] | zap[int_last[r0]]
        if not chan_mask.any():
            continue
        runs.append((r0, r1, chan_mask))
        masked_sum += np.where(
            chan_mask, block[r0:r1].sum(axis=0, dtype=np.float64), 0.0
        )
        masked_count += chan_mask * (r1 - r0)

    if not runs:
        return 0
    total = block.sum(axis=0, dtype=np.float64)
    fill = ((total - masked_sum) / np.maximum(nout - masked_count, 1)).astype(
        np.float32
    )
    for r0, r1, chan_mask in runs:
        block[r0:r1, chan_mask] = fill[chan_mask]
    return int(masked_count.sum()) * tdec


def process_block(raw, start, tdec=1, rfi_mask=None):
    """
    Turn a raw block (nsamp, nchan) into a downsampled, masked float32 block of
    shape (nsamp // tdec, nchan).
    """
    raw, n_bad = fill_non_finite(raw)
    block = downsample_block(raw, tdec)
    n_masked = 0
    if rfi_mask is not None:
        n_masked = apply_rfi_mask(block, start, tdec, rfi_mask)
    return block, n_bad, n_masked


def load_block(observation, start, nsamp, rfi_mask=None, chans=slice(None), tdec=1):
    """
    Load in observation, downsample by tdec and apply rfifind mask if provided.

    Returns a float32 array of shape (nsamp // tdec, nchan).
    """
    raw = read_raw_block(observation, start, nsamp, chans)
    return process_block(raw, start, tdec, rfi_mask)


def apply_logging_config(config, log_file="./logs/default.log"):
    """
    Applies logging settings from the given configuration.
    """
    log_stream.setFormatter(
        logging.Formatter(fmt=config.logging.format, datefmt="%b %d %H:%M:%S")
    )

    if config.logging.get("file_logging", False):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(fmt=config.logging.format, datefmt="%b %d %H:%M:%S")
        )
        logging.root.addHandler(file_handler)
    logging.root.setLevel(config.logging.level.upper())
    log.debug("Set default level to: %s", config.logging.level)

    if "modules" in config.logging:
        for module_name, level in config.logging["modules"].items():
            logging.getLogger(module_name).setLevel(level.upper())
            log.debug("Set %s level to: %s", module_name, level)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--obs-file",
    type=str,
    help="File containing the observation. Needs to be readable using your.",
    required=True,
)
@click.option(
    "--stack-file", type=str, help="File where power spectra are saved.", required=True
)
@click.option(
    "--config-file",
    default="sps_config.yml",
    type=str,
    help="Name of used config. Default: sps_config.yml",
)
@click.option(
    "--config-options",
    default="{}",
    type=str,
    help=(
        "Additional options that overwrite the config options. Provide a string that would define a python dictionary"
        """For example use --config-options '{"beamform": {"max_mask_frac": 0.1}}' """
    ),
)
@click.option(
    "--num-threads",
    default=4,
    type=int,
    help=("Number of multiprocessing threads to use."),
)
@click.option(
    "--maxdm",
    default=200.0,
    type=float,
    help="Maximum DM. Config value will be ignored.",
)
@click.option(
    "-t",
    "--time-downsample",
    default=1,
    type=int,
    help="Average this many adjacent time samples on read.",
)
@click.option(
    "--rfifind-mask",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="PRESTO rfifind .mask file.",
)
@click.option(
    "--fmin",
    default=None,
    type=float,
    help="Lowest frequency to use (MHz).",
)
@click.option(
    "--fmax",
    default=None,
    type=float,
    help="Highest frequency to use (MHz).",
)
@click.option(
    "--zero-dm-birdies",
    is_flag=True,
    default=False,
    help="\n".join(
        [
            "\b",
            "Zap zero-DM birdies! Pew pew!!!!",
            "       /",
            "      /_   *ZAP*",
            "       /",
            "      /",
            " ",
            "    <(x )___",
            "     ( ._> /",
            "      `---'",
            "       _||",
        ]
    ),
)
@click.option(
    "--telescope",
    default=None,
    type=str,
    help="Name of used telescope, could probably also be extracted, but your does not interpret telescope ids.",
)
@click.option(
    "--norm-input",
    is_flag=True,
    default=False,
)
def run_file_pipeline(
    obs_file,
    stack_file,
    config_file,
    config_options,
    num_threads,
    maxdm,
    time_downsample,
    rfifind_mask,
    fmin,
    fmax,
    zero_dm_birdies,
    telescope,
    norm_input,
):
    config = load_config(config_file, config_options)
    apply_logging_config(config)
    multiprocessing.set_start_method("forkserver", force=True)

    observation = your.Your(obs_file)
    obs_header = observation.your_header
    log.info(obs_header)
    tstart = obs_header.tstart
    if observation.format == "fits":
        tstart = psrfits_start_mjd(observation)
        dt = (obs_header.tstart - tstart) * 86400.0
    if telescope:
        telescope_location = EarthLocation.of_site(telescope)
    else:
        telescope_location = EarthLocation(
            x=float(observation.header["ANT_X"]) * u.m,
            y=float(observation.header["ANT_Y"]) * u.m,
            z=float(observation.header["ANT_Z"]) * u.m,
        )

    # pick channels between fmin and fmax
    c0, c1, FREQ_BOTTOM, FREQ_TOP = select_channels(obs_header, fmin, fmax)
    chans = slice(c0, c1)
    nchan = c1 - c0
    log.info(
        f"Using channels {c0}-{c1 - 1} ({nchan} chans), "
        f"{FREQ_BOTTOM:.1f}-{FREQ_TOP:.1f} MHz."
    )

    # rfifind mask
    rfi_mask = None
    if rfifind_mask is not None:
        rfi_mask = read_rfifind_mask(rfifind_mask, observation, chans, tstart)
        log.info("Yeehaw! Loaded rfifind mask!")
    # read-in and downsampling
    tdec = time_downsample
    log.info(f"Downsampling in time by a factor of {tdec}.")
    nout = obs_header.nspectra // tdec
    target_bytes = 1 * 1024**3
    gulp = max(1, target_bytes // (nchan * 4 * tdec)) * tdec
    obs_spectrum = np.empty((nchan, nout), dtype=np.float32)

    n_bad = n_masked = 0
    starts = list(range(0, nout * tdec, gulp))

    def read(start):
        return read_raw_block(observation, start, min(gulp, nout * tdec - start), chans)

    # read the next block in a background thread while processing the current one
    with ThreadPoolExecutor(max_workers=1) as reader:
        next_raw = reader.submit(read, starts[0])
        for i, start in enumerate(starts):
            raw = next_raw.result()
            if i + 1 < len(starts):
                next_raw = reader.submit(read, starts[i + 1])
            n = raw.shape[0]
            block, nb, nm = process_block(raw, start, tdec, rfi_mask)
            del raw
            n_bad += nb
            n_masked += nm
            input_block = block.T
            if norm_input:
                # FOr now ignore do not properly take zero input into account
                input_block = (
                    input_block - np.nanmean(input_block, 1)[:, None]
                ) / np.nanstd(input_block, 1)[:, None]
            obs_spectrum[:, start // tdec : (start + n) // tdec] = input_block
            if i % 10 == 0:
                log.info(f"Read {start + n:,} / {nout * tdec:,} samples")

    ntot = nout * tdec * nchan
    if n_bad:
        log.warning(
            f"Replaced {n_bad} non-finite samples ({n_bad / ntot:.2e} of data)."
        )
    if n_masked:
        log.info(
            f"rfifind mask replaced {n_masked} samples ({n_masked / ntot:.1%} of data)."
        )

    if obs_header.foff > 0:
        # make sure channels go high -> low (CHIME order); dedisperse() flips them
        # back to low -> high, which is what the FDMT expects
        obs_spectrum = obs_spectrum[::-1, :]
    log.info(
        f"Loaded data with shape {obs_spectrum.shape} (channels ordered high -> low)."
    )

    TSAMP = obs_header.tsamp * tdec
    fdmt_config = config.dedisp.fdmt
    num_dms_fac = fdmt_config.num_dms_fac

    num_dms = (
        int(
            DM_CONSTANT
            * maxdm
            * (1 / FREQ_BOTTOM**2 - 1 / FREQ_TOP**2)
            / TSAMP
            // num_dms_fac
        )
        + 1
    ) * num_dms_fac

    # restrict num_dms to number of channels for now
    # num_dms = min(num_dms, obs_spectrum.shape[0])
    nsamps = fdmt_config.chunk_size + num_dms
    dm_step = fdmt_config.dm_step
    fdmt = FDMT(
        FREQ_BOTTOM,
        FREQ_TOP,
        obs_spectrum.shape[0],
        nsamps,
        TSAMP,
        dt_max=num_dms,
        dt_min=0,
        dt_step=1,
    )
    fdmt.set_num_threads(num_threads)

    skybeam = SkyBeam(
        spectra=obs_spectrum,
        ra=obs_header.ra_deg,
        dec=obs_header.dec_deg,
        nchan=obs_spectrum.shape[0],
        ntime=obs_spectrum.shape[1],
        maxdm=maxdm,
        beam_row=0,
        utc_start=Time(tstart, format="mjd").unix,
        nbits=obs_spectrum.dtype.itemsize * 8,
    )
    dts = dedisperse(
        fdmt,
        skybeam,
        fdmt_config.chunk_size,
        num_dms,
        dm_step,
        True,
        freq_bottom=FREQ_BOTTOM,
        freq_top=FREQ_TOP,
        tsamp=TSAMP,
    )

    log.info(f"Created {len(dts.dms)} DM trials up to a DM of {max(dts.dms)}")

    padded_length = int(
        np.power(2, np.ceil(np.log(dts.dedisp_ts.shape[1]) / np.log(2)))
    )
    spectra_nbit = config.ps_creation.nbit

    psc_pipeline = PowerSpectraCreation(
        num_threads=num_threads,
        use_db=False,
        update_db=False,
        tsamp=TSAMP,
        find_common_birdies=False,
        padded_length=padded_length,
        clean_rfi=zero_dm_birdies,
        run_static_filter=False,
        run_dynamic_filter=zero_dm_birdies,
        dynamic_filter_config=OmegaConf.to_container(
            config.ps_creation.dynamic_filter_config
        ),
        nbit=spectra_nbit,
        telescope_location=telescope_location,
    )
    power_spectra = psc_pipeline.transform(dts)

    pss_pipeline = PowerSpectraSearch(
        **OmegaConf.to_container(config.ps.ps_search_config),
        num_threads=num_threads,
        update_db=False,
    )

    (
        power_spectra_detection_clusters,
        power_spectra_detections,
    ) = pss_pipeline.search(
        power_spectra,
    )

    fg = Features.from_config(
        OmegaConf.to_container(config.cands.features),
        OmegaConf.to_container(config.cands.arrays),
        num_threads=num_threads,
    )

    spcc = fg.make_single_pointing_candidate_collection(
        power_spectra_detection_clusters, power_spectra
    )

    plotted_cands = spcc.plot_candidates(
        folder="./single_observation_plots/", sigma_threshold=0, dm_threshold=-1
    )
    log.info(f"Plotted {len(plotted_cands)} candidates.")
    cand_folder = "./single_observation_candidates/"
    os.makedirs(cand_folder, exist_ok=True)
    cand_path = (
        cand_folder
        + os.path.splitext(os.path.basename(obs_file))[0]
        + "_candidates.npz"
    )
    spcc.write(cand_path)
    log.info(f"Wrote candidates to {cand_path}.")

    stack_nbit = config.ps.ps_stack_config.stack_nbit

    stack_exists = os.path.isfile(stack_file)
    if not stack_exists:
        log.info(f"Stack does not exist. Writing new stack to {stack_file}.")
        power_spectra.write(stack_file, nbit=stack_nbit)
    else:
        stack_config = OmegaConf.to_container(config.ps.ps_stack_config)
        stack_config["qc"] = False
        stack_config["qc_config"] = {}
        ps_stack_pipeline = PowerSpectraStack(
            mode="month",
            update_db=False,
            spectra_nbit=stack_nbit,
            **stack_config,
        )
        power_spectra_stacked = ps_stack_pipeline.stack_power_spectra_readout(
            power_spectra, stack_file
        )
        stack_path = os.path.abspath(stack_file)
        ps_stack_pipeline.replace_spectra(power_spectra_stacked, stack_path, stack_path)

        pss_pipeline.full_harm_bins = None  # clear cached array
        (
            power_spectra_detection_clusters_stacked,
            power_spectra_detections_stacked,
        ) = pss_pipeline.search(power_spectra_stacked)

        spcc_stacked = fg.make_single_pointing_candidate_collection(
            power_spectra_detection_clusters_stacked, power_spectra_stacked
        )

        plotted_cands = spcc_stacked.plot_candidates(
            folder="./stack_plots/", sigma_threshold=0, dm_threshold=-1
        )
        log.info(f"Plotted {len(plotted_cands)} stack candidates.")
        cand_folder = "./stack_candidates/"
        os.makedirs(cand_folder, exist_ok=True)
        cand_path = (
            cand_folder
            + os.path.splitext(os.path.basename(stack_file))[0]
            + "_candidates.npz"
        )
        spcc_stacked.write(cand_path)
        log.info(f"Wrote stack candidates to {cand_path}.")

    power_spectra.unlink_shared_memory()
