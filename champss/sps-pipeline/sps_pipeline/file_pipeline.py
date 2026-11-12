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


def read_par_orbit(par_file):
    """
    Read the position and binary orbit from a TEMPO/PINT par file.

    Supports orbits given with T0/ECC/OM (BT, DD, DDK, ...) or with
    TASC/EPS1/EPS2 (ELL1). Returns a dict with the orbital parameters in
    days, light-seconds and radians, and the pulsar position as a SkyCoord.
    """
    from astropy.coordinates import SkyCoord

    values = {}
    with open(par_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and not line.startswith("#"):
                values[parts[0].upper()] = parts[1]

    def get(key, default=None):
        val = values.get(key, default)
        if val is None:
            raise click.BadParameter(f"{key} missing in par file {par_file}.")
        return float(str(val).replace("D", "E"))

    pb = get("PB")
    a1 = get("A1")
    if "TASC" in values:
        eps1, eps2 = get("EPS1", 0.0), get("EPS2", 0.0)
        ecc = np.hypot(eps1, eps2)
        om = np.arctan2(eps1, eps2)
        t0 = get("TASC") + om / (2 * np.pi) * pb
    else:
        ecc = get("ECC", get("E", 0.0) if "E" in values else 0.0)
        om = np.radians(get("OM", 0.0))
        t0 = get("T0")
    ra = values.get("RAJ")
    dec = values.get("DECJ")
    if ra is None or dec is None:
        raise click.BadParameter(f"RAJ/DECJ missing in par file {par_file}.")
    coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    return dict(
        pb=pb,
        a1=a1,
        ecc=ecc,
        om=om,
        t0=t0,
        omdot=np.radians(get("OMDOT", 0.0)) / 365.25,  # rad / day
        pbdot=get("PBDOT", 0.0),
        coord=coord,
    )


def orbital_radial_velocity(orbit, mjd_utc, location):
    """
    Line-of-sight velocity of the pulsar in its orbit (in units of c, positive
    when moving away) at the given topocentric UTC MJD.

    The observed (barycentric) spin frequency is f_intrinsic / (1 + v).
    """
    t = Time(mjd_utc, format="mjd", scale="utc", location=location)
    t_bary = (t.tdb + t.light_travel_time(orbit["coord"], "barycentric")).mjd
    dt = t_bary - orbit["t0"]
    pb = orbit["pb"] + orbit["pbdot"] * dt
    mean_anomaly = 2 * np.pi * (dt / pb % 1)
    ecc = orbit["ecc"]
    ecc_anomaly = mean_anomaly
    for _ in range(50):
        ecc_anomaly = mean_anomaly + ecc * np.sin(ecc_anomaly)
    true_anomaly = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(ecc_anomaly / 2),
        np.sqrt(1 - ecc) * np.cos(ecc_anomaly / 2),
    )
    om = orbit["om"] + orbit["omdot"] * dt
    return (
        2
        * np.pi
        * orbit["a1"]
        / (pb * 86400.0 * np.sqrt(1 - ecc**2))
        * (np.cos(om + true_anomaly) + ecc * np.cos(om))
    )


def doppler_correct_power_spectra(power_spectra, velocity, chunk=64):
    """
    Resample a single-observation PowerSpectra in place into the pulsar frame.

    The pulsar's power at intrinsic frequency f appears at f / (1 + velocity),
    so output bin k takes the power of input bin round(k / (1 + velocity)).
    The day's masked bins are mapped the same way so that bin weights stay
    correct in the stack. Bins that map outside the spectrum are zeroed and
    masked.
    """
    if len(power_spectra.bad_freq_indices) != 1:
        raise ValueError("Doppler correction expects a single-observation spectrum.")
    nbin = len(power_spectra.freq_labels)
    src = np.rint(np.arange(nbin) / (1.0 + velocity)).astype(np.int64)
    valid = src < nbin
    src = np.minimum(src, nbin - 1)

    bad = np.zeros(nbin, dtype=bool)
    bad[np.asarray(power_spectra.bad_freq_indices[0], dtype=np.int64)] = True
    bad_new = bad[src] | ~valid

    ps = power_spectra.power_spectra
    for i in range(0, ps.shape[0], chunk):
        rows = ps[i : i + chunk][:, src]
        rows[:, ~valid] = 0
        ps[i : i + chunk] = rows
    power_spectra.bad_freq_indices = [np.flatnonzero(bad_new).tolist()]
    return src


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
@click.option(
    "--static-birdies",
    is_flag=True,
    default=False,
    help=(
        "Zap the static list of known periodic RFI frequencies used by the SPS/CHAMPSS "
        "pipeline (rfi_mitigation/data/birdies.yaml, e.g. LTE at 1.5625 Hz and 60 Hz "
        "mains with their harmonics). These values were derived from CHIME/SPS data, "
        "so they may not apply to data from other telescopes or backends."
    ),
)
@click.option(
    "--manual-candidates",
    "--mc",
    default=[],
    type=str,
    multiple=True,
    help=(
        "Create manual candidates in the single-observation and stack search. "
        "Modes:"
        "'--mc skip_search': Skip normal search; "
        "'--mc \"para freq dm\"': Create candidate at freq and dm; "
        "'--mc \"para freq dm search\"': Same, but search +-0.01 Hz and +-5 DM trials around it; "
        "The following modes need the SPS database (see --db-host): "
        "'--mc \"PSR psr_name\"': Create candidate at known source psr_name; "
        "'--mc \"FS fs_id\"': Create candidate for followup source at given fs_id; "
        "'--mc \"nearby_ks 1\"': Create candidate for all known sources within given radius "
        "(append ' search' to search around them); "
        "'--mc \"nearby_fs 1\"': Create candidate for followup sources within given radius; "
    ),
)
@click.option(
    "--db-port",
    default=27017,
    type=int,
    help="Port used for the mongodb database. Only used for database manual candidates.",
)
@click.option(
    "--db-host",
    default="sps-archiver1",
    type=str,
    help="Host used for the mongodb database. Only used for database manual candidates.",
)
@click.option(
    "--db-name",
    default="sps",
    type=str,
    help="Name used for the mongodb database. Only used for database manual candidates.",
)
@click.option(
    "--doppler-par",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Par file of a binary pulsar. Before stacking, each observation's power "
        "spectrum is shifted into the pulsar frame using the orbital Doppler factor "
        "at the middle of the observation, so the pulsar adds up at its intrinsic "
        "frequency. Single-observation candidates are not affected. Only use a "
        "stack file that was built with the same par file."
    ),
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
    static_birdies,
    manual_candidates,
    db_port,
    db_host,
    db_name,
    doppler_par,
):
    config = load_config(config_file, config_options)
    apply_logging_config(config)

    manual_candidates = list(manual_candidates)
    db_modes = {"PSR", "FS", "nearby_ks", "nearby_fs"}
    for mc in manual_candidates:
        mode = mc.split(" ")[0]
        if mode not in db_modes | {"para", "skip_search"}:
            raise click.BadParameter(
                f"Unknown manual candidate mode in '{mc}'.",
                param_hint="--manual-candidates",
            )
        if mode == "para" and len(mc.split(" ")) < 3:
            raise click.BadParameter(
                f"'{mc}' needs the format 'para freq dm'.",
                param_hint="--manual-candidates",
            )
    if any(mc.split(" ")[0] in db_modes for mc in manual_candidates):
        from sps_databases import db_utils

        log.info(f"Connecting to database {db_name} at {db_host}:{db_port}.")
        db_utils.connect(host=db_host, port=db_port, name=db_name)
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
                # Normalise each channel to zero mean and unit std within this block.
                # The data is finite at this point, so plain mean/std are enough.
                # Channels without variance (e.g. fully masked by rfifind and filled
                # with a constant) are set to 0 instead of dividing by 0, which would
                # put NaNs into every DM trial.
                mean = block.mean(axis=0, dtype=np.float64)
                std = block.std(axis=0, dtype=np.float64)
                std[std <= 1e-6 * np.abs(mean)] = np.inf  # constant up to rounding
                block -= mean.astype(np.float32)
                block /= std.astype(np.float32)
                input_block = block.T
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
        clean_rfi=zero_dm_birdies or static_birdies,
        run_static_filter=static_birdies,
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
        manual_candidates=manual_candidates,
    )

    fg = Features.from_config(
        OmegaConf.to_container(config.cands.features),
        OmegaConf.to_container(config.cands.arrays),
        num_threads=num_threads,
    )

    spcc = fg.make_single_pointing_candidate_collection(
        power_spectra_detection_clusters, power_spectra
    )

    obs_name = os.path.splitext(os.path.basename(obs_file))[0]
    plotted_cands = spcc.plot_candidates(
        folder=f"./single_observation_plots/{obs_name}/",
        sigma_threshold=0,
        dm_threshold=-1,
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

    if doppler_par is not None:
        orbit = read_par_orbit(doppler_par)
        mid_mjd = tstart + obs_header.nspectra * obs_header.tsamp / 2 / 86400.0
        velocity = orbital_radial_velocity(orbit, mid_mjd, telescope_location)
        log.info(
            f"Doppler correcting spectrum with {doppler_par}: v/c = {velocity:+.3e} "
            f"at MJD {mid_mjd:.5f} (frequency shift factor {1 / (1 + velocity):.8f})."
        )
        doppler_correct_power_spectra(power_spectra, velocity)

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

        # clear cached arrays; the manual candidate arrays depend on the stack's
        # bin weights and are only recomputed when full_harm_bins_raw is None
        pss_pipeline.full_harm_bins = None
        pss_pipeline.full_harm_bins_raw = None
        (
            power_spectra_detection_clusters_stacked,
            power_spectra_detections_stacked,
        ) = pss_pipeline.search(
            power_spectra_stacked, manual_candidates=manual_candidates
        )

        spcc_stacked = fg.make_single_pointing_candidate_collection(
            power_spectra_detection_clusters_stacked, power_spectra_stacked
        )

        stack_name = (
            os.path.splitext(os.path.basename(stack_file))[0]
            + f"_{power_spectra_stacked.num_days:03d}days"
        )
        plotted_cands = spcc_stacked.plot_candidates(
            folder=f"./stack_plots/{stack_name}/", sigma_threshold=0, dm_threshold=-1
        )
        log.info(f"Plotted {len(plotted_cands)} stack candidates.")
        cand_folder = "./stack_candidates/"
        os.makedirs(cand_folder, exist_ok=True)
        # include the number of stacked days so earlier results are not overwritten
        cand_path = cand_folder + stack_name + "_candidates.npz"
        spcc_stacked.write(cand_path)
        log.info(f"Wrote stack candidates to {cand_path}.")

    power_spectra.unlink_shared_memory()
