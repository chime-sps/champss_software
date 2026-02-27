from decimal import Decimal, InvalidOperation

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time


def readpsrarch(fname, dedisperse=True, verbose=False):
    """
    Read pulsar archive directly using psrchive Requires the python psrchive bindings.

    Parameters
    ----------
    fname: string, file_directory
    dedisperse: Bool
    apply psrchive's by-channel incoherent de-dispersion

    Returns
    -------
    data : ndarray
        Archive data cube
    params : dict
        Dictionary containing: F (frequency array), T (time/mjd array),
        psr (source name), tel (telescope), dm, ra, dec, f0
    """
    import psrchive

    arch = psrchive.Archive.load(fname)
    source = arch.get_source()
    tel = arch.get_telescope()
    if verbose:
        print(f"Read archive of {source} from {fname}")

    if dedisperse:
        if verbose:
            print("Dedispersing...")
        arch.dedisperse()
    data = arch.get_data()
    midf = arch.get_centre_frequency()
    bw = arch.get_bandwidth()
    F = np.linspace(midf - bw / 2.0, midf + bw / 2.0, data.shape[2], endpoint=False)
    # F = arch.get_frequencies()

    a = arch.start_time()
    t0 = a.strtempo()
    t0 = Time(float(t0), format="mjd", precision=9)

    # Get frequency and time info for plot axes
    nt = data.shape[0]
    Tobs = arch.integration_length()
    dt = (Tobs / nt) * u.s
    T = t0 + np.arange(nt) * dt
    T = T.mjd

    # Extract additional parameters from archive
    dm = arch.get_dispersion_measure()
    coords = arch.get_coordinates()
    ra = coords.ra().getDegrees()
    dec = coords.dec().getDegrees()
    eph = arch.get_ephemeris()
    f0 = float(eph.get_value("F0"))

    params = {
        "F": F,
        "T": T,
        "psr": source,
        "tel": tel,
        "dm": dm,
        "ra": ra,
        "dec": dec,
        "f0": f0,
    }

    return data, params


def get_archive_parameter(fname, param):
    """
    Get a parameter from a pulsar archive file.

    Parameters
    ----------
    fname: string
    param: string, name of parameter to extract

    Returns
    -------
    value of parameter
    """
    import psrchive

    arch = psrchive.Archive.load(fname)
    eph = arch.get_ephemeris()
    return float(eph.get_value(param))


def clean_foldspec(
    I,
    plots=False,
    apply_mask=True,
    rfimethod="var",
    flagval=7,
    offpulse=False,
    tolerance=0.7,
    off_gates=0,
):
    """
    Clean and rescale folded spectrum.

    Parameters
    ----------
    I: ndarray [time, pol, freq, phase]
    or [time, freq, phase] for single pol
    plots: Bool
    Create diagnostic plots
    apply_mask: Bool
    Multiply dynamic spectrum by mask
    rfimethod: String
    RFI flagging method, currently only supports var
    tolerance: float
    % of subints per channel to zap whole channel
    off_gates: slice
    manually chosen off_gate region.  If unset, the bottom 50%
    of the profile is chosen

    Returns
    -------
    foldspec: folded spectrum, after bandpass division,
    off-gate subtraction and RFI masking
    flag: std. devs of each subint used for RFI flagging
    mask: boolean RFI mask
    bg: Ibg(t, f) subtracted from foldspec
    bpass: Ibg(f), an estimate of the bandpass
    """

    # Sum to form total intensity, mostly a memory/speed concern
    if len(I.shape) == 4:
        I = I[:, (0, 1)].mean(1)

    # Use median over time to not be dominated by outliers
    bpass = np.median(I.mean(-1, keepdims=True), axis=0, keepdims=True)
    foldspec = I / bpass

    mask = np.ones_like(I.mean(-1))

    prof_dirty = (I - I.mean(-1, keepdims=True)).mean(0).mean(0)
    if not off_gates:
        off_gates = np.argwhere(prof_dirty < np.median(prof_dirty)).squeeze()
        recompute_offgates = 1
    else:
        recompute_offgates = 0

    if rfimethod == "var":
        if offpulse:
            flag = np.std(foldspec[..., off_gates], axis=-1)
        else:
            flag = np.std(foldspec, axis=-1)
        # find std. dev of flag values without outliers
        flag_series = np.sort(flag.ravel())
        flagsize = len(flag_series)
        flagmid = slice(int(flagsize // 4), int(3 * flagsize // 4))
        flag_std = np.std(flag_series[flagmid])
        flag_mean = np.mean(flag_series[flagmid])

        # Mask all values over 10 sigma of the mean
        mask[flag > flag_mean + flagval * flag_std] = 0

        # If more than 50% of subints are bad in a channel, zap whole channel
        mask[:, mask.mean(0) < tolerance] = 0
        mask[mask.mean(1) < tolerance] = 0
        if apply_mask:
            I[mask < 0.5] = np.mean(I[mask > 0.5])

    profile = I.mean(0).mean(0)

    # redetermine off_gates, if off_gates not specified
    if recompute_offgates:
        off_gates = np.argwhere(profile < np.median(profile)).squeeze()

    # renormalize, now that RFI are zapped
    bpass = I[..., off_gates].mean(-1, keepdims=True).mean(0, keepdims=True)
    foldspec = I / bpass
    foldspec[np.isnan(foldspec)] = np.nanmean(foldspec)
    bg = np.mean(foldspec[..., off_gates], axis=-1, keepdims=True)
    foldspec = foldspec - bg

    if plots:
        plot_diagnostics(foldspec, flag, mask)

    return foldspec, flag, mask, bg.squeeze(), bpass.squeeze()


def create_dynspec(foldspec, template=[1], profsig=5.0, bint=1, binf=1):
    """
    Create dynamic spectrum from folded data cube.

    Uses average profile as a weight, sums over phase

    Returns: dynspec, np array [t, f]

    Parameters
    ----------
    foldspec: [time, frequency, phase] array
    template: pulse profile I(phase), phase weights to create dynamic spectrum
    profsig: S/N value, mask all profile below this (only if no template given)
    bint: integer, bin dynspec by this value in time
    binf: integer, bin dynspec by this value in frequency
    """

    # If no template provided, create profile by summing over time, frequency
    if len(template) <= 1:
        template = foldspec.mean(0).mean(0)
        template = template / np.max(template)

        # Noise estimated from bottom 50% of profile
        tnoise = np.std(template[template < np.median(template)])
        # Template zeroed below threshold
        template[template < tnoise * profsig] = 0

    # Multiply the profile by the template, sum over phase
    dynspec = (foldspec * template[np.newaxis, np.newaxis, :]).mean(-1)

    if bint > 1:
        tbins = int(dynspec.shape[0] // bint)
        dynspec = dynspec[-bint * tbins :].reshape(tbins, bint, -1).mean(1)
    if binf > 1:
        dynspec = dynspec.reshape(dynspec.shape[0], -1, binf).mean(-1)

    return dynspec


def plot_foldspec(fn):
    """Convenience function to read, clean, plot I(t,p), I(f,p), I(p) of a folded
    archive.
    """

    data, params = readpsrarch(fn)
    F, T = params["F"], params["T"]
    fs, flag, mask, bg, bpass = clean_foldspec(data.squeeze())

    bpass = np.std(fs.mean(0), axis=-1)
    bpass /= np.median(bpass)
    # fs_clean = np.copy(fs)
    fs[:, bpass > 1.2] = 0

    taxis = ((T - T[0]) * u.day).to(u.min)
    T0 = Time(T[0], format="mjd")

    fs_bg = fs - np.median(fs, axis=-1, keepdims=True)
    fs_bg = fs[1:-1]

    ngate = fs.shape[-1]
    nc = 256
    ng = 512
    binf = fs.shape[1] // nc
    bing = fs.shape[-1] // ng
    fs_bin = np.copy(fs_bg)

    maskchan = np.argwhere(fs_bin.mean(0).mean(-1) == 0).squeeze()
    fs_bin[:, maskchan] = np.nan
    fs_bin = np.nanmean(
        fs_bin.reshape(fs_bin.shape[0], fs_bin.shape[1] // binf, binf, -1), 2
    )
    if fs.shape[-1] > ng:
        fs_bin = np.nanmean(
            fs_bin.reshape(fs_bin.shape[0], fs_bin.shape[1], -1, bing), -1
        )
        ngate = np.min([ngate, ng])

    profile = np.nanmean(np.nanmean(fs_bin, 0), 0)
    SNsort = np.sort(profile)
    SN = (profile - np.nanmean(SNsort[: 3 * ngate // 4])) / np.nanstd(
        SNsort[: 3 * ngate // 4]
    )

    vfmin = np.nanmean(fs_bin) - 2 * np.nanstd(np.nanmean(fs_bin, 0))
    vfmax = np.nanmean(fs_bin) + 5 * np.nanstd(np.nanmean(fs_bin, 0))
    vtmin = np.nanmean(fs_bin) - 2 * np.nanstd(np.nanmean(fs_bin, 1))
    vtmax = np.nanmean(fs_bin) + 5 * np.nanstd(np.nanmean(fs_bin, 1))

    fig = plt.figure(figsize=(12, 8))
    ax0 = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((3, 4), (1, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2, rowspan=2)

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    ax0.set_title(f"{psr} {T0.isot[:10]}")

    ax1.imshow(
        np.nanmean(fs_bin, 0),
        aspect="auto",
        interpolation="nearest",
        vmin=vfmin,
        vmax=vfmax,
        origin="lower",
        extent=[0, 1, F[0], F[-1]],
    )
    ax3.imshow(
        np.nanmean(fs_bin, 1),
        aspect="auto",
        interpolation="nearest",
        vmin=vtmin,
        vmax=vtmax,
        origin="lower",
        extent=[0, 1, 0, max(taxis.to(u.min).value)],
    )

    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_xlabel("phase")

    ax3.set_ylabel("time (min)")
    ax3.set_xlabel("phase")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    phaseaxis = np.linspace(0, 1, ngate, endpoint=False)
    ax0.plot(phaseaxis, SN)
    ax0.set_xlim(0, 1)
    ax0.set_xticks([])


def get_SN(profile, return_profile=False, offpulse=True):
    """
    Get S/N of 1D pulse profile by smoothing over different pulse widths,
    in powers of 2 up to 1/4 the pulse width

    Input: 1D array of pulse profile

    Returns: max S/N, index of max S/N, binning factor
    """

    from scipy.ndimage import uniform_filter

    ngate = len(profile)
    # More conservative max width - only up to 1/4 of profile
    maxbin = int(np.log2(ngate // 4))

    if offpulse:
        # Precompute mean and std dev from bottom 3/4 of sorted profile
        profsort = np.sort(profile)
        prof_N = profsort[: int(3 * ngate / 4)]
        sigma_off = np.std(prof_N)
        prof_mean = np.mean(prof_N)
    else:
        sigma_off = np.std(profile)
        prof_mean = np.mean(profile)

    binning = 2 ** np.arange(maxbin + 1)
    SNprofs = np.zeros((len(binning), len(profile)))

    SNmax = 0
    for i, b in enumerate(binning):
        # Scale noise by sqrt(width) to account for boxcar smoothing
        scaled_sigma = sigma_off / np.sqrt(b)

        prof_filtered = uniform_filter(profile, b)
        SNprof = (prof_filtered - prof_mean) / scaled_sigma
        SNprofs[i] = SNprof

        if np.max(SNprof) > SNmax:
            SNmax = np.max(SNprof)
            SNindex = np.argmax(SNprof)
            ibin = b
    if return_profile:
        return SNmax, SNprofs[0]
    return SNmax


def compute_profile_SNs(profiles):
    """
    Compute S/N for multiple profiles using statistics computed across the 2D array.
    This matches the stable S/N computation from phase_aligned_search.py, reducing
    peaks/divergences from repeated noise estimation of every profile

    Parameters
    ----------
    profiles : ndarray
        2D array of profiles with shape (n_trials, npbin)

    Returns
    -------
    SNs : ndarray
        S/N values for each profile
    """
    npbin = profiles.shape[1]

    # Pre-compute noise from average of all profiles
    sigma_off = np.std(np.nanmean(profiles, axis=0))

    # Pre-compute boxcar widths and scaled noise
    max_exp = int(np.log2(npbin // 4))
    n_widths = max_exp + 1
    widths = 2 ** np.arange(n_widths)
    scaled_sigmas = sigma_off / np.sqrt(widths)

    # Compute S/N for each profile
    SNs = np.zeros(len(profiles))
    for i_prof, prof in enumerate(profiles):
        prof_mean = np.nanmean(prof)
        snmax = 0.0

        # Cumulative sum for boxcar computation
        cumsum = np.zeros(npbin + 1)
        for idx in range(npbin):
            cumsum[idx + 1] = cumsum[idx] + prof[idx]

        for iw in range(n_widths):
            width = widths[iw]
            # Find max of boxcar-filtered profile
            boxcar_max = -1e30
            for idx in range(npbin):
                end_idx = idx + width
                if end_idx <= npbin:
                    val = (cumsum[end_idx] - cumsum[idx]) / width
                else:
                    val = (cumsum[npbin] - cumsum[idx] + cumsum[end_idx - npbin]) / width
                if val > boxcar_max:
                    boxcar_max = val

            sn = (boxcar_max - prof_mean) / scaled_sigmas[iw]
            if sn > snmax:
                snmax = sn

        SNs[i_prof] = snmax

    return SNs


def read_par(parfile):
    """Reads a par file and return a dictionary of parameter names and values."""
    par = {}
    ignore = [
        "DMMODEL",
        "DMOFF",
        "DM_",
        "CM_",
        "CONSTRAIN",
        "JUMP",
        "NITS",
        "NTOA",
        "CORRECT_TROPOSPHERE",
        "PLANET_SHAPIRO",
        "DILATEFREQ",
        "TIMEEPH",
        "MODE",
        "TZRMJD",
        "TZRSITE",
        "TZRFRQ",
        "EPHVER",
        "T2CMETHOD",
    ]
    file = open(parfile)
    for line in file.readlines():
        err = None
        p_type = None
        sline = line.split()
        if len(sline) == 0 or line[0] == "#" or line[0:2] == "C " or sline[0] in ignore:
            continue
        param = sline[0]
        if param == "E":
            param = "ECC"
        val = sline[1]
        if len(sline) == 3 and sline[2] not in ["0", "1"]:
            err = sline[2].replace("D", "E")
        elif len(sline) == 4:
            err = sline[3].replace("D", "E")
        try:
            val = int(val)
            p_type = "d"
        except ValueError:
            try:
                val = float(Decimal(val.replace("D", "E")))
                if "e" in sline[1] or "E" in sline[1].replace("D", "E"):
                    p_type = "e"
                else:
                    p_type = "f"
            except InvalidOperation:
                p_type = "s"
        par[param] = val
        if err:
            par[param + "_ERR"] = float(err)
        if p_type:
            par[param + "_TYPE"] = p_type
    file.close()
    return par
