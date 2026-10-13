import your
import numpy as np
from omegaconf import OmegaConf
from astropy.time import Time
import multiprocessing
import click
import logging
import os
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


def read_rfifind_mask(mask_file, observation):
    """
    Reads a PRESTO rfifind .mask file into a boolean zap array.
    Yoinked directly from PRESTO.
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
    for i, chans in enumerate(mask_zap_chans_per_int):
        zap[i, chans] = True
    zap[:, mask_zap_chans] = True
    zap[mask_zap_ints, :] = True

    hdr = observation.your_header
    if nchan != hdr.nchans:
        raise ValueError(
            f"rfifind mask has {nchan} channels but the data has {hdr.nchans}."
        )
    if not np.isclose(dtint, ptsperint * hdr.tsamp, rtol=1e-3):
        raise ValueError("Make sure you run the mask on the same resolution of data...")
    if abs(MJD - hdr.tstart) * 86400 > 1.0:
        log.warning("Is this the right mask for the file...?")

    # make sure to flip mask if data are numbers high-low
    if hdr.foff < 0:
        zap = zap[:, ::-1]
    return zap, ptsperint


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


def load_block(observation, start, nsamp, rfi_mask=None, chans=slice(None)):
    """
    Load in observation and apply rfifind mask if provided.
    """
    if observation.format == "fits":
        block = observation.formatclass.get_data(observation, start, nsamp)[:, 0, chans]
    else:
        block = observation.get_data(start, nsamp)[:, chans].astype(np.float32)

    bad = ~np.isfinite(block)
    n_bad = int(bad.sum())

    n_masked = 0
    if rfi_mask is not None:
        zap, ptsperint = rfi_mask
        ints = np.minimum((start + np.arange(nsamp)) // ptsperint, zap.shape[0] - 1)
        masked = zap[ints]
        n_masked = int((masked & ~bad).sum())
        bad |= masked

    if bad.any():
        good = ~bad
        fill = np.where(good, block, 0).sum(axis=0, dtype=np.float64) / np.maximum(
            good.sum(axis=0), 1
        )
        np.copyto(
            block, np.broadcast_to(fill.astype(np.float32), block.shape), where=bad
        )
    return block, n_bad, n_masked


def apply_logging_config(config, log_file="./logs/default.log"):
    """
    Applies logging settings from the given configuration.

    Logging settings are under the 'logging' key, and include:
    - format: string for the `logging.formatter`
    - level: logging level for the root logger
    - modules: a dictionary of submodule names and logging level to be
               applied to that submodule's logger
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
):
    config = load_config(config_file, config_options)
    apply_logging_config(config)
    multiprocessing.set_start_method("forkserver", force=True)

    observation = your.Your(obs_file)
    obs_header = observation.your_header
    log.info(obs_header)

    telescope_location = EarthLocation(
        x=float(observation.header["ANT_X"]) * u.m,
        y=float(observation.header["ANT_Y"]) * u.m,
        z=float(observation.header["ANT_Z"]) * u.m,
    )

    # Channel selection; FREQ_BOTTOM/FREQ_TOP are band edges, always bottom < top
    c0, c1, FREQ_BOTTOM, FREQ_TOP = select_channels(obs_header, fmin, fmax)
    chans = slice(c0, c1)
    nchan = c1 - c0
    log.info(
        f"Using channels {c0}-{c1 - 1} ({nchan} chans), "
        f"{FREQ_BOTTOM:.1f}-{FREQ_TOP:.1f} MHz."
    )

    # rfifind mask: read and validated for the full band, then cut to `chans`
    rfi_mask = None
    if rfifind_mask is not None:
        zap, ptsperint = read_rfifind_mask(rfifind_mask, observation)
        rfi_mask = (zap[:, chans], ptsperint)
        log.info(
            f"Loaded rfifind mask {rfifind_mask}: {rfi_mask[0].mean():.1%} of "
            "(interval, channel) cells masked in the selected band."
        )

    # Read in chunks, averaging each run of `tdec` consecutive samples
    tdec = time_downsample
    log.info(f"Downsampling in time by a factor of {tdec}.")
    nout = obs_header.nspectra // tdec  # drops any trailing remainder
    target_bytes = 1 * 1024**3  # ~1 GB of float32 per chunk
    gulp = max(1, target_bytes // (nchan * 4 * tdec)) * tdec
    obs_spectrum = np.empty((nchan, nout), dtype=np.float32)

    n_bad = n_masked = 0
    for i, start in enumerate(range(0, nout * tdec, gulp)):
        n = min(gulp, nout * tdec - start)
        block, nb, nm = load_block(observation, start, n, rfi_mask, chans)
        n_bad += nb
        n_masked += nm
        # average each run of `tdec` consecutive samples (not `your`'s strided mean)
        obs_spectrum[:, start // tdec : (start + n) // tdec] = (
            block.reshape(n // tdec, tdec, nchan).mean(axis=1).T
        )
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

    if obs_header.foff < 0:
        obs_spectrum = obs_spectrum[::-1, :]  # make sure channels go low -> high
    log.info(f"Loaded data with shape {obs_spectrum.shape}.")

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
        obs_header.fch1,
        obs_header.fch1 + obs_header.bw,
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
        utc_start=Time(obs_header.tstart, format="mjd").unix,
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
        clean_rfi=False,
        run_static_filter=False,
        run_dynamic_filter=False,
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
